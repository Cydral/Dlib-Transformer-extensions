// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Implementation of the software-extended device memory described in extended_memory.h.

#ifndef DLIB_EXTENDED_MEMORY_CPP_
#define DLIB_EXTENDED_MEMORY_CPP_

#include "extended_memory.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    std::string default_extended_memory_store_path ()
    {
        /* Deployment decides where the store lives, not the build. The environment variable
           comes first so that the same binary can be pointed at a fast local disk on one
           machine and at whatever is available on another, and only then do we fall back on
           the platform's temporary directory. */
        if (const char* v = std::getenv("DLIB_XMEM_STORE"))
            if (*v) return std::string(v);
#ifdef _WIN32
        if (const char* v = std::getenv("TEMP"))
            if (*v) return std::string(v);
        if (const char* v = std::getenv("TMP"))
            if (*v) return std::string(v);
        return std::string("C:\\Windows\\Temp");
#else
        if (const char* v = std::getenv("TMPDIR"))
            if (*v) return std::string(v);
        return std::string("/tmp");
#endif
    }

// ----------------------------------------------------------------------------------------

#ifndef DLIB_USE_CUDA

    void print_extended_memory_stats (std::ostream& out)
    {
        out << "extended memory: unavailable, this build of dlib does not use CUDA\n";
    }

    void print_extended_memory_blocks (std::ostream& out, std::size_t)
    {
        out << "extended memory: unavailable, this build of dlib does not use CUDA\n";
    }

    device_scope::device_scope (std::initializer_list<const tensor*>) {}
    device_scope::device_scope (std::initializer_list<const gpu_data*>) {}
    device_scope::~device_scope () {}
    void device_scope::take (const gpu_data*) {}

#endif
}

#ifdef DLIB_USE_CUDA

/* tensor.h first: it pulls in ../matrix.h, and the matrix headers call the C library's
   memcpy() unqualified from inside namespace dlib. Declaring dlib::memcpy through
   gpu_data.h ahead of them would hide the global one. */
#include "tensor.h"
#include "gpu_data.h"
#include "cuda_utils.h"
#include "../error.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <process.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/statfs.h>
#  include <sys/statvfs.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace dlib
{
    // Defined in gpu_data.cpp. Works around a stream synchronization bug on some toolkits.
    void synchronize_stream(cudaStream_t stream);

    namespace cuda
    {
        // Defined in extended_memory_kernels.cu.
        void extended_memory_fingerprint(const void* device_ptr, std::size_t bytes,
                                         unsigned long long& out_xor, unsigned long long& out_add,
                                         cudaStream_t stream);
    }

namespace xmem
{

// ----------------------------------------------------------------------------------------
// Shared state read by the accessor fast path.
//
// The rings are allocated once and never released. Freeing them would mean proving that no
// thread is between the load of g::active and the write into the ring, and a megabyte is a
// cheaper answer to that question than a quiescence protocol.
// ----------------------------------------------------------------------------------------

    static const std::size_t   trace_capacity = 1u << 16;
    static const std::size_t   hot_capacity   = 1u << 10;
    static const std::uint32_t no_block       = 0xffffffffu;

// ----------------------------------------------------------------------------------------
// The store.
//
// One sparse file, mapped once, with the blocks at fixed offsets inside it. The mapping is
// what gives this tier its two useful properties: its pages are page cache, so the kernel
// decides how much of the store stays in memory and writes the rest back on its own, and a
// block sitting in it can be read on the host without any copy at all.
//
// The slot allocator rounds to the page size and bins by exact rounded size. A network
// reallocates the same shapes on every step, so the bins hit and the file stops growing
// after the first pass.
// ----------------------------------------------------------------------------------------

    /*
        Free space on the volume holding the store. Zero when it cannot be determined, which
        is treated as "do not second guess the caller" rather than as an error.
    */
    static std::size_t volume_free_bytes (const std::string& dir)
    {
#ifdef _WIN32
        ULARGE_INTEGER avail;
        avail.QuadPart = 0;
        if (!GetDiskFreeSpaceExA(dir.c_str(), &avail, nullptr, nullptr))
            return 0;
        return (std::size_t)avail.QuadPart;
#else
        struct statvfs st;
        if (::statvfs(dir.c_str(), &st) != 0)
            return 0;
        return (std::size_t)st.f_bavail * (std::size_t)st.f_frsize;
#endif
    }

    /*
        The store path is the one option a program is likely to get wrong, so it is worth
        saying which part of it is wrong rather than letting the failure surface later as a
        file that cannot be created inside a directory that was never there.
    */
    static void check_directory (const std::string& dir)
    {
#ifdef _WIN32
        const DWORD attr = GetFileAttributesA(dir.c_str());
        const bool exists    = (attr != INVALID_FILE_ATTRIBUTES);
        const bool directory = exists && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
        struct stat st;
        const bool exists    = (::stat(dir.c_str(), &st) == 0);
        const bool directory = exists && S_ISDIR(st.st_mode);
#endif
        if (!exists)
            throw dlib::error("extended memory: store_path names something that does not exist: \"" +
                              dir + "\". Create the directory, or set DLIB_XMEM_STORE.");
        if (!directory)
            throw dlib::error("extended memory: store_path is not a directory: \"" + dir + "\".");
    }

    class memory_arena
    {
    public:

        memory_arena (const std::string& dir, std::size_t bytes, bool verbose) : cap(bytes)
        {
            check_directory(dir);

            /* The file is sparse, so sizing it beyond the volume succeeds and then fails
               later, on a write, as a bus error rather than an exception. Capping the
               mapping here turns that into a line the reader can act on. */
            const std::size_t avail = volume_free_bytes(dir);
            if (avail > 0 && cap > avail)
            {
                const std::size_t capped = (std::size_t)((double)avail * 0.9);
                std::cerr << "extended memory: the store was asked for " << (cap >> 20)
                          << " MiB but only " << (avail >> 20) << " MiB is free in " << dir
                          << ", capping it at " << (capped >> 20) << " MiB\n";
                cap = capped;
            }
            if (cap < (256ul << 20))
                throw dlib::error("extended memory: no usable room for the store in " + dir +
                                  ". Point store_path at a volume with space, or leave it empty "
                                  "to keep evicted blocks in pinned host memory.");

            std::ostringstream name;
            name << dir;
            if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
                name << '/';

#ifdef _WIN32
            name << "dlib_xmem_" << (unsigned long)GetCurrentProcessId() << ".store";
            const std::string path = name.str();

            file = CreateFileA(path.c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               nullptr,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                               nullptr);
            if (file == INVALID_HANDLE_VALUE)
            {
                file = nullptr;
                throw dlib::error("extended memory: cannot create the store file " + path);
            }

            // Best effort. Without it the mapping still works, it just reserves the whole
            // capacity on disk instead of only the pages that are written.
            DWORD returned = 0;
            DeviceIoControl((HANDLE)file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);

            section = CreateFileMappingA((HANDLE)file, nullptr, PAGE_READWRITE,
                                         (DWORD)((std::uint64_t)cap >> 32),
                                         (DWORD)((std::uint64_t)cap & 0xffffffffu),
                                         nullptr);
            if (!section)
            {
                CloseHandle((HANDLE)file);
                file = nullptr;
                throw dlib::error("extended memory: cannot create the store mapping");
            }

            mapping = (char*)MapViewOfFile((HANDLE)section, FILE_MAP_ALL_ACCESS, 0, 0, cap);
            if (!mapping)
            {
                CloseHandle((HANDLE)section);
                CloseHandle((HANDLE)file);
                section = nullptr;
                file = nullptr;
                throw dlib::error("extended memory: cannot map the store");
            }

            HMODULE k32 = GetModuleHandleA("kernel32.dll");
            if (k32)
                prefetch_fn = (prefetch_t)GetProcAddress(k32, "PrefetchVirtualMemory");
#else
            name << "dlib_xmem_" << (long)getpid() << ".store";
            const std::string path = name.str();

            fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0)
                throw dlib::error("extended memory: cannot create the store file " + path);

            // The file has no name from here on, so it cannot outlive the process even if
            // the process dies without unwinding.
            ::unlink(path.c_str());

            if (::ftruncate(fd, (off_t)cap) != 0)
            {
                ::close(fd);
                fd = -1;
                throw dlib::error("extended memory: cannot size the store file");
            }

            void* p = ::mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (p == MAP_FAILED)
            {
                ::close(fd);
                fd = -1;
                throw dlib::error("extended memory: cannot map the store");
            }
            mapping = (char*)p;
#endif
            report_medium(dir, verbose);
        }

        ~memory_arena ()
        {
#ifdef _WIN32
            if (mapping) UnmapViewOfFile(mapping);
            if (section) CloseHandle((HANDLE)section);
            if (file)    CloseHandle((HANDLE)file);
#else
            if (mapping) ::munmap(mapping, cap);
            if (fd >= 0) ::close(fd);
#endif
        }

        memory_arena(const memory_arena&) = delete;
        memory_arena& operator=(const memory_arena&) = delete;

        char*       base () const { return mapping; }
        std::size_t capacity () const { return cap; }
        std::size_t bytes_in_use () const { return used; }

        std::int64_t allocate (std::size_t bytes)
        {
            const std::size_t slot = round_up(bytes);

            auto it = free_slots.find(slot);
            if (it != free_slots.end() && !it->second.empty())
            {
                const std::int64_t off = it->second.back();
                it->second.pop_back();
                used += slot;
                return off;
            }

            if ((std::size_t)next + slot > cap)
                return -1;

            const std::int64_t off = next;
            next += (std::int64_t)slot;
            used += slot;
            return off;
        }

        void release (std::int64_t off, std::size_t bytes)
        {
            if (off < 0) return;
            const std::size_t slot = round_up(bytes);
            free_slots[slot].push_back(off);
            used -= std::min(used, slot);
        }

        /*
            Asks the kernel to bring a slot's pages in before anyone touches them. This is
            the second horizon of the prefetcher: the schedule says which block is coming,
            and this puts its pages in memory ahead of the transfer that will read them, so
            the transfer does not take a major fault half way through.
        */
        void advise_willneed (std::int64_t off, std::size_t bytes) const
        {
            if (off < 0 || !mapping) return;
#ifdef _WIN32
            if (!prefetch_fn) return;
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = mapping + off;
            range.NumberOfBytes  = bytes;
            prefetch_fn(GetCurrentProcess(), 1, &range, 0);
#else
            ::madvise(mapping + off, bytes, MADV_WILLNEED);
#endif
        }

    private:

        /*
            Where the store ended up is the single decision that most affects how the
            subsystem behaves, and it is the one a program cannot check for itself. Two
            things are worth saying out loud: the kind of filesystem, because a network or
            projected one cannot carry a shared mapping usefully and a memory one defeats
            the purpose entirely, and the measured cost of getting a few megabytes onto the
            device behind it, because no property of a filesystem reveals that the volume is
            a memory card.
        */
        void report_medium (const std::string& dir, bool verbose) const
        {
#ifndef _WIN32
            struct statfs sf;
            if (::statfs(dir.c_str(), &sf) == 0)
            {
                const long t = (long)sf.f_type;
                if (t == 0x01021997L)
                    std::cerr << "extended memory: the store is on a projected filesystem (9p). "
                                 "Shared mappings there are slow at best; put it on a native "
                                 "volume.\n";
                else if (t == 0x6969L || t == (long)0xFF534D42L)
                    std::cerr << "extended memory: the store is on a network filesystem. "
                                 "Put it on a local volume.\n";
                else if (t == 0x01021994L)
                    std::cerr << "extended memory: the store is on a memory filesystem, so it "
                                 "occupies RAM rather than relieving it.\n";
            }
#endif
            if (!verbose)
                return;

            const std::size_t n = 8ul << 20;
            if (cap < 2*n || !mapping)
                return;

            char* probe = mapping + (cap - n);
            const auto t0 = std::chrono::steady_clock::now();
            std::memset(probe, 0xa5, n);
#ifdef _WIN32
            FlushViewOfFile(probe, n);
#else
            ::msync(probe, n, MS_SYNC);
#endif
            const double dt = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0).count();
            if (dt > 0)
                std::cerr << "extended memory: the store writes at about "
                          << (long)((double)(n >> 20) / dt) << " MiB/s\n";
        }

        static std::size_t round_up (std::size_t bytes)
        {
            const std::size_t g = 1u << 20;
            return ((bytes + g - 1) / g) * g;
        }

        char*        mapping = nullptr;
        std::size_t  cap     = 0;
        std::int64_t next    = 0;
        std::size_t  used    = 0;
        std::map<std::size_t, std::vector<std::int64_t> > free_slots;

#ifdef _WIN32
        typedef BOOL (WINAPI *prefetch_t)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        prefetch_t prefetch_fn = nullptr;
        void* file    = nullptr;
        void* section = nullptr;
#else
        int fd = -1;
#endif
    };

// ----------------------------------------------------------------------------------------
// The device pool.
//
// cudaFree synchronizes the whole device, so returning a block to the driver on every
// eviction would cost more than the eviction saves. Blocks are binned by exact size, which
// wastes nothing here because a network reallocates the same shapes on every step.
//
// The pool is held through a shared_ptr by the deleter of every buffer it hands out, so it
// outlives the manager and a block freed during static destruction still finds it.
// ----------------------------------------------------------------------------------------

    class device_pool
    {
    public:

        device_pool () {}

        ~device_pool () { trim(); }

        float* allocate (std::size_t bytes)
        {
            std::lock_guard<std::mutex> lk(mu);

            auto it = bins.find(bytes);
            if (it != bins.end() && !it->second.empty())
            {
                float* p = it->second.back();
                it->second.pop_back();
                cached -= bytes;
                return p;
            }

            void* p = nullptr;
            cudaError_t err = cudaMalloc(&p, bytes);
            if (err == cudaErrorMemoryAllocation)
            {
                // Give the driver back everything we are hoarding and ask once more.
                trim_locked();
                err = cudaMalloc(&p, bytes);
            }
            if (err != cudaSuccess)
                throw cuda_error("extended memory: cudaMalloc failed for " +
                                 std::to_string(bytes) + " bytes: " + cudaGetErrorString(err));
            return (float*)p;
        }

        void release (float* p, std::size_t bytes)
        {
            if (!p) return;
            std::lock_guard<std::mutex> lk(mu);
            bins[bytes].push_back(p);
            cached += bytes;
        }

        void trim ()
        {
            std::lock_guard<std::mutex> lk(mu);
            trim_locked();
        }

    private:

        void trim_locked ()
        {
            for (auto& kv : bins)
            {
                for (float* p : kv.second)
                {
                    const cudaError_t err = cudaFree(p);
                    if (err != cudaSuccess)
                        std::cerr << "extended memory: cudaFree() failed. Reason: "
                                  << cudaGetErrorString(err) << std::endl;
                }
                kv.second.clear();
            }
            bins.clear();
            cached = 0;
        }

        mutable std::mutex mu;
        std::size_t cached = 0;
        std::unordered_map<std::size_t, std::vector<float*> > bins;
    };

// ----------------------------------------------------------------------------------------
// The transfer engine.
//
// Pageable memory cannot feed a DMA engine, so every transfer between the store and the
// device is chunked through a pinned buffer split in two. The host copy of one chunk then
// runs while the DMA of the chunk before it is in flight, which is the difference between
// paying for the memcpy and the transfer one after the other and paying for the slower of
// the two.
// ----------------------------------------------------------------------------------------

    class transfer_engine
    {
    public:

        transfer_engine (std::size_t bytes, cudaStream_t s) : stream(s)
        {
            half = std::max<std::size_t>(bytes / 2, 1u << 20);
            for (int i = 0; i < 2; ++i)
            {
                void* p = nullptr;
                const cudaError_t err = cudaMallocHost(&p, half);
                if (err != cudaSuccess)
                    throw cuda_error("extended memory: cannot pin the staging buffer: " +
                                     std::string(cudaGetErrorString(err)));
                stage[i] = (char*)p;
                CHECK_CUDA(cudaEventCreateWithFlags(&evt[i], cudaEventDisableTiming));
            }
        }

        ~transfer_engine ()
        {
            for (int i = 0; i < 2; ++i)
            {
                if (evt[i])   cudaEventDestroy(evt[i]);
                if (stage[i]) cudaFreeHost(stage[i]);
            }
        }

        transfer_engine(const transfer_engine&) = delete;
        transfer_engine& operator=(const transfer_engine&) = delete;

        /*
            Chunk size for one transfer. Sizing it off the staging buffer rather than off
            the block was a mistake worth naming: a block smaller than a staging half then
            moves as a single chunk, the memcpy and the DMA run one after the other, and the
            pipeline this class exists for never engages. Most blocks in a decoder are a few
            megabytes, so that was most of them. Four chunks is enough to hide the copy
            behind the transfer, and a floor of two megabytes keeps the per-chunk event from
            costing more than the chunk carries.
        */
        std::size_t chunk_for (std::size_t bytes) const
        {
            return std::min(half, std::max<std::size_t>(bytes / 4, 2ul << 20));
        }

        // Host to device. The copy into the staging half of chunk k overlaps the transfer
        // of chunk k-1.
        void to_device (void* dst_dev, const void* src_host, std::size_t bytes)
        {
            bool        pending[2] = {false, false};
            const std::size_t step = chunk_for(bytes);
            std::size_t done = 0;
            int         k    = 0;
            while (done < bytes)
            {
                const std::size_t n = std::min(step, bytes - done);
                const int         h = k & 1;

                if (pending[h])
                {
                    CHECK_CUDA(cudaEventSynchronize(evt[h]));
                    pending[h] = false;
                }
                std::memcpy(stage[h], (const char*)src_host + done, n);
                CHECK_CUDA(cudaMemcpyAsync((char*)dst_dev + done, stage[h], n,
                                           cudaMemcpyHostToDevice, stream));
                CHECK_CUDA(cudaEventRecord(evt[h], stream));
                pending[h] = true;

                done += n;
                ++k;
            }
            synchronize_stream(stream);
        }

        // Device to host. Lagged by one chunk so the copy out of the staging half runs
        // while the next chunk is still being fetched.
        /*
            Device to host, optionally writing only the chunks that differ from what the
            destination already holds. Returns true when anything was written.
        */
        bool from_device (void* dst_host, const void* src_dev, std::size_t bytes,
                          bool only_if_different = false)
        {
            bool wrote = false;
            bool        pending[2] = {false, false};
            std::size_t size_of[2] = {0, 0};
            const std::size_t step = chunk_for(bytes);
            std::size_t done = 0, drained = 0;
            int         k    = 0;

            while (done < bytes)
            {
                const std::size_t n = std::min(step, bytes - done);
                const int         h = k & 1;

                if (pending[h])
                {
                    CHECK_CUDA(cudaEventSynchronize(evt[h]));
                    char* out = (char*)dst_host + drained;
                    if (!only_if_different || std::memcmp(out, stage[h], size_of[h]) != 0)
                    {
                        std::memcpy(out, stage[h], size_of[h]);
                        wrote = true;
                    }
                    drained += size_of[h];
                    pending[h] = false;
                }
                CHECK_CUDA(cudaMemcpyAsync(stage[h], (const char*)src_dev + done, n,
                                           cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA(cudaEventRecord(evt[h], stream));
                pending[h]  = true;
                size_of[h]  = n;

                done += n;
                ++k;
            }

            for (int i = 0; i < 2 && drained < bytes; ++i)
            {
                const int h = (k - 2 + i) & 1;
                if (!pending[h]) continue;
                CHECK_CUDA(cudaEventSynchronize(evt[h]));
                char* out = (char*)dst_host + drained;
                if (!only_if_different || std::memcmp(out, stage[h], size_of[h]) != 0)
                {
                    std::memcpy(out, stage[h], size_of[h]);
                    wrote = true;
                }
                drained += size_of[h];
                pending[h] = false;
            }
            synchronize_stream(stream);
            return wrote;
        }

        std::mutex mu;

    private:

        char*        stage[2] = {nullptr, nullptr};
        cudaEvent_t  evt[2]   = {nullptr, nullptr};
        std::size_t  half     = 0;
        cudaStream_t stream   = nullptr;
    };

// ----------------------------------------------------------------------------------------
// The access schedule.
//
// A published schedule is immutable, so the compute thread can read it under the manager
// lock while the observation thread builds the next one without any coordination beyond
// the pointer swap.
// ----------------------------------------------------------------------------------------

    struct schedule
    {
        std::vector<std::uint32_t> cycle;
        std::unordered_map<std::uint32_t, std::vector<std::size_t> > positions;

        // Number of accesses between the cursor and the next use of this block. A block
        // that is not part of the cycle never comes back, which makes it the best victim
        // there is.
        std::size_t next_use (std::uint32_t id, std::size_t cursor) const
        {
            auto it = positions.find(id);
            if (it == positions.end())
                return (std::size_t)-1;
            const std::vector<std::size_t>& v = it->second;
            auto p = std::lower_bound(v.begin(), v.end(), cursor);
            if (p != v.end())
                return *p - cursor;
            return v.front() + cycle.size() - cursor;
        }
    };

    static bool verify_period (
        const std::vector<std::uint32_t>& s,
        std::size_t d
    )
    {
        const std::size_t L = s.size();
        if (d < 4 || d > L/3)
            return false;

        // Confirmed over two full repetitions, which is what keeps an accidental
        // coincidence from being mistaken for the step of the network.
        const std::size_t span = std::min(L - d, 3*d);
        if (span < 2*d)
            return false;

        for (std::size_t i = 0; i < span; ++i)
        {
            if (s[L-1-i] != s[L-1-i-d])
                return false;
        }
        return true;
    }

    /*
        Looks for the period of the tail of the trace.

        Candidates come from the gaps between repetitions of one chosen block, and which
        block is chosen decides whether the search finds anything. Anchoring on the newest
        access, which is the obvious choice, fails on a real network: the last thing read
        before a layer boundary is often a normalization scale that every layer reads, so
        its gaps are layer spacings and never the step itself, and every attempt is spent on
        candidates that cannot be right. Anchoring on the rarest block in the window instead
        gives gaps that are the step, or a small multiple of it.
    */
    static bool find_period (
        const std::vector<std::uint32_t>& s,
        std::size_t& out_period
    )
    {
        const std::size_t L = s.size();
        if (L < 64)
            return false;

        const std::size_t window = std::min<std::size_t>(L, 8192);

        std::unordered_map<std::uint32_t, unsigned> freq;
        for (std::size_t i = L - window; i < L; ++i)
            ++freq[s[i]];

        std::uint32_t anchor = s[L-1];
        unsigned      rarest = (unsigned)-1;
        for (auto& kv : freq)
        {
            if (kv.second >= 2 && kv.second < rarest)
            {
                rarest = kv.second;
                anchor = kv.first;
            }
        }

        /* Distances from the anchor's last occurrence back to each earlier one. Taking all
           of them rather than only consecutive gaps matters when the anchor is read more
           than once per step: the step is then the distance to the k-th occurrence back,
           not to the previous one. */
        std::vector<std::size_t> positions;
        for (std::size_t i = L; i-- > L - window; )
        {
            if (s[i] == anchor)
                positions.push_back(i);
            if (positions.size() >= 17)
                break;
        }

        for (std::size_t i = 1; i < positions.size(); ++i)
        {
            const std::size_t d = positions[0] - positions[i];
            if (verify_period(s, d))
            {
                out_period = d;
                return true;
            }
        }

        // Fall back on the newest block when the anchor led nowhere.
        unsigned tried = 0;
        for (std::size_t d = 4; d <= L/3 && tried < 16; ++d)
        {
            if (s[L-1-d] != s[L-1])
                continue;
            ++tried;
            if (verify_period(s, d))
            {
                out_period = d;
                return true;
            }
        }
        return false;
    }

// ----------------------------------------------------------------------------------------
// The manager.
// ----------------------------------------------------------------------------------------

    class manager
    {
    public:

        static manager*& singleton ()
        {
            // Deliberately leaked. A gpu_data destroyed during static destruction still
            // unregisters, and it must not find a manager that has already gone.
            static manager* m = nullptr;
            return m;
        }

        static block_record* record_of (const gpu_data& g) { return g.xrec; }

        manager (const extended_memory_options& o, int device_id, std::size_t budget)
            : opt(o), dev(device_id), vram_budget(budget)
        {
            std::size_t free_b = 0;
            CHECK_CUDA(cudaMemGetInfo(&free_b, &total_device_bytes));
            pool = std::make_shared<device_pool>();
            CHECK_CUDA(cudaStreamCreateWithFlags(&xstream, cudaStreamNonBlocking));
            engine.reset(new transfer_engine(opt.staging_bytes, xstream));
            if (!opt.store_path.empty())
                arena.reset(new memory_arena(opt.store_path, opt.store_bytes, opt.verbose));
        }

        ~manager ()
        {
            stop_worker();
            if (xstream) cudaStreamDestroy(xstream);
        }

        void start_worker ()
        {
            if (!opt.predictive) return;
            stopping.store(false);
            worker = std::thread(&manager::worker_loop, this);
        }

        /*
            Idempotent, and the one piece of teardown that has to happen on purpose.
            Everything else the subsystem holds is reclaimed by the driver and the kernel
            when the process ends, but a thread is not: left running into static
            destruction it would keep calling CUDA after the runtime has torn itself down,
            and the failure that follows arrives in a thread with nobody to catch it.
        */
        void stop_worker ()
        {
            stopping.store(true, std::memory_order_release);
            if (worker.joinable())
                worker.join();
        }

        bool has_arena () const { return arena != nullptr; }

        // ------------------------------------------------------------------ registration

        block_record* register_block (gpu_data* owner, std::size_t bytes, int device_id)
        {
            if (device_id != dev)
                return nullptr;

            /* Residency management spreads the total across tiers; it cannot split one
               tensor, because a kernel reads a block as a single contiguous range. A block
               larger than the card can therefore never be placed, and saying so here keeps
               that failure where stock dlib puts it, in set_size(), instead of deferring it
               to whichever accessor happens to touch the block first. */
            if (bytes > total_device_bytes)
            {
                const double mib = 1024.0*1024.0;
                std::ostringstream m;
                m << "extended memory: a single block of " << (bytes/mib)
                  << " MiB cannot be placed on a device of " << (total_device_bytes/mib)
                  << " MiB. Extended memory raises the total a network may occupy, not the "
                     "size of any one tensor in it.";
                throw cuda_error(m.str());
            }

            std::lock_guard<std::mutex> lk(mu);

            block_record* r = new block_record();
            r->owner     = owner;
            r->bytes     = bytes;
            r->device_id = device_id;
            r->evictable = bytes >= opt.min_block_bytes;
            largest_block = std::max(largest_block, bytes);
            if (!r->evictable)
                immovable_bytes += bytes;

            if (free_ids.empty())
            {
                r->id = (std::uint32_t)by_id.size();
                by_id.push_back(r);
            }
            else
            {
                r->id = free_ids.back();
                free_ids.pop_back();
                by_id[r->id] = r;
            }

            if (arena && r->evictable)
            {
                /* The block's host copy lives in the mapping from the start, so writing it
                   for the first time, which is what loading a model does, goes straight to
                   the store instead of through a pinned buffer that would then have to be
                   spilled. The slot holds nothing yet, hence store_valid false. */
                r->slot = arena->allocate(bytes);
                if (r->slot >= 0)
                {
                    r->slot_written = false;
                    r->hash_valid   = false;
                    r->state.store(tier_store, std::memory_order_release);
                    r->store_valid.store(false, std::memory_order_relaxed);
                    ++managed_blocks;
                    return r;
                }
                // The arena is full. Fall through and give this block a pinned mirror.
                if (opt.verbose && !arena_full_reported)
                {
                    arena_full_reported = true;
                    std::cerr << "extended memory: the store is full, later blocks keep "
                                 "pinned mirrors\n";
                }
            }

            r->state.store(owner->data_host ? tier_host : tier_store, std::memory_order_release);
            if (owner->data_host)
                pinned_bytes += bytes;
            ++managed_blocks;
            return r;
        }

        void unregister_block (block_record* r)
        {
            std::unique_lock<std::mutex> lk(mu);
            wait_out_transit(lk, r);

            /* Stock dlib got away without this because cudaFree synchronizes the device, so
               a buffer could not be recycled under a kernel that was still reading it. The
               pool does not synchronize, so a tensor destroyed while its last kernel is in
               flight would hand a live buffer straight to the next allocation. One
               synchronization clears the flag on every block at once, so tearing down a
               network costs one and not one per tensor.

               Failures are swallowed: this also runs during static destruction, where the
               CUDA runtime may already be gone, and throwing out of a destructor would end
               the process over something that no longer protects anything. */
            if (r->owner && r->owner->device_in_use &&
                r->state.load(std::memory_order_relaxed) == tier_device)
            {
                try
                {
                    quiesced = false;
                    quiesce_locked();
                }
                catch (...)
                {
                }
            }

            if (r->state.load(std::memory_order_relaxed) == tier_device)
                device_bytes -= std::min(device_bytes, r->bytes);
            if (r->owner && r->owner->data_host && r->slot < 0)
                pinned_bytes -= std::min(pinned_bytes, r->bytes);
            if (arena && r->slot >= 0)
                arena->release(r->slot, r->bytes);
            if (!r->evictable)
                immovable_bytes -= std::min(immovable_bytes, r->bytes);

            if (r->id < by_id.size() && by_id[r->id] == r)
            {
                by_id[r->id] = nullptr;
                free_ids.push_back(r->id);
            }
            --managed_blocks;
            delete r;
        }

        void lock_registry ()   { mu.lock(); }
        void unlock_registry () { mu.unlock(); }

        bool store_backed (std::size_t bytes) const
        {
            return arena != nullptr && bytes >= opt.min_block_bytes;
        }

        /*
            Workspaces are the one part of dlib's device memory that never passes through
            gpu_data. Left outside, they do not merely go unaccounted: a cuDNN workspace can
            fail to allocate while the subsystem is sitting on gigabytes of weights it would
            gladly have released. Routing them here fixes both, at the cost of nothing, since
            a scratch buffer has no contents worth tiering.
        */
        std::shared_ptr<void> acquire_scratch (std::size_t bytes)
        {
            std::shared_ptr<float> block;
            {
                std::lock_guard<std::mutex> lk(mu);
                block = acquire_locked(bytes, true);
            }
            scratch_bytes.fetch_add(bytes, std::memory_order_relaxed);

            /* The deleter carries the block itself, so releasing the void pointer returns
               the memory to the pool, and it touches only an atomic, so it is safe to run
               from a destructor while any lock is held. */
            return std::shared_ptr<void>(
                (void*)block.get(),
                [block, bytes](void*) mutable
                {
                    manager* m = manager::singleton();
                    if (m)
                        m->scratch_bytes.fetch_sub(bytes, std::memory_order_relaxed);
                    block.reset();
                });
        }

        void pin_block   (block_record* r) { r->pins.fetch_add(1, std::memory_order_acq_rel); }
        void unpin_block (block_record* r) { r->pins.fetch_sub(1, std::memory_order_acq_rel); }

        // ------------------------------------------------------------------- residency

        void restore_device (block_record* r, bool need_content)
        {
            std::unique_lock<std::mutex> lk(mu);
            wait_out_transit(lk, r);

            gpu_data& g = *r->owner;

            if (r->state.load(std::memory_order_relaxed) == tier_device && g.data_device)
                return;

            const unsigned char from = r->state.load(std::memory_order_relaxed);

            pin_block(r);
            std::shared_ptr<float> buf;
            try
            {
                buf = acquire_locked(r->bytes, true);
            }
            catch (...)
            {
                unpin_block(r);
                throw;
            }

            g.data_device = buf;
            device_bytes += r->bytes;
            device_peak = std::max(device_peak, device_bytes);

            /* From here the block is half promoted: it owns a device buffer and the budget
               counts it, but its tier still says otherwise. A transfer that throws in
               between has to put all three back, or the block is left pinned forever, out
               of the reach of eviction, with the budget charging for memory the tier says
               is not there. */
            try
            {
                if (from == tier_host)
                {
                    /* The pinned mirror is current, so the transfer is left to gpu_data
                       itself: the accessor that called us runs copy_to_device() next, on
                       its own stream and with its own completion bookkeeping. */
                    if (!need_content)
                    {
                        g.device_current = true;
                        g.host_current   = false;
                    }
                }
                else
                {
                    const bool have = r->slot >= 0 &&
                                  (r->store_valid.load(std::memory_order_relaxed) ||
                                   (g.data_host && g.host_current));
                    if (need_content && have)
                        store_to_device_locked(r, g);
                    g.device_current = true;
                    /* When the window is installed and the device now holds what the mapping
                       holds, the host copy is current as well, and a later read of it costs
                       nothing rather than a transfer back. */
                    g.host_current   = (g.data_host != nullptr) && have && need_content;
                }
            }
            catch (...)
            {
                g.data_device.reset();
                g.device_current = false;
                device_bytes -= std::min(device_bytes, r->bytes);
                r->state.store(from, std::memory_order_release);
                unpin_block(r);
                throw;
            }

            r->state.store(tier_device, std::memory_order_release);
            unpin_block(r);
            ++restores;
        }

        void before_host (block_record* r, bool need_content, bool writes)
        {
            r->stamp.store(g::clock.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);

            std::unique_lock<std::mutex> lk(mu);
            wait_out_transit(lk, r);

            gpu_data& g = *r->owner;

            /* A host access that finds the device ahead forces the block back over the bus,
               and the non-const overload then marks the device copy stale so the next kernel
               pushes it up again. That round trip is invisible from outside and no amount of
               residency management can remove it, so the only useful thing to do is count it:
               a program moving tensors through host memory between kernels shows up here as
               a large number, and nowhere else. */
            if (!g.host_current && g.data_device)
            {
                ++host_pulls;
                host_pull_bytes += r->bytes;
            }

            if (r->slot >= 0)
            {
                /* The host copy of this block is the mapping, so there is nothing to
                   allocate and nothing to transfer: install the window once and every later
                   host access is free. */
                if (!g.data_host)
                    install_store_window_locked(g, r);

                if (r->state.load(std::memory_order_relaxed) == tier_store)
                {
                    /* Nothing on the device, so the mapping is by definition current. A
                       block that has never been written reads as whatever the sparse file
                       gives back, which is the same guarantee a fresh set_size() offers. */
                    g.host_current = true;
                    r->store_valid.store(true, std::memory_order_relaxed);
                }
                /* A write through the window changes the store behind the device's back, so
                   the fingerprint recorded for it stops describing anything. */
                if (writes)
                    r->hash_valid = false;
                /* When the device is ahead it is copy_to_host() that brings the window up to
                   date, on the way out of this call. Claiming here that the mapping is
                   current would be a promise about a copy that has not happened and might
                   still fail, and a block whose write back was then skipped on that promise
                   would be silently wrong. host_current says the same thing afterwards, and
                   says it only once it is true. */
                (void)need_content;
                (void)writes;
                return;
            }

            if (g.data_host)
                return;

            allocate_pinned_mirror_locked(g, r);
            if (r->state.load(std::memory_order_relaxed) == tier_device)
                g.host_current = false;         // copy_to_host() picks it up on return
            else
            {
                g.host_current   = true;
                g.device_current = false;
                r->state.store(tier_host, std::memory_order_release);
            }
        }

        // ----------------------------------------------------------------------- stats

        extended_memory_stats stats () const
        {
            std::lock_guard<std::mutex> lk(mu);
            extended_memory_stats s;
            s.enabled           = true;
            s.vram_budget       = vram_budget;
            s.device_bytes      = device_bytes;
            s.device_peak_bytes = device_peak;
            s.scratch_bytes     = scratch_bytes.load(std::memory_order_relaxed);
            s.pinned_bytes      = pinned_bytes;
            s.store_bytes       = arena ? arena->bytes_in_use() : 0;
            s.store_capacity    = arena ? arena->capacity() : 0;
            s.managed_blocks    = managed_blocks;
            s.evictions         = evictions;
            s.restores          = restores;
            s.prefetch_hits     = prefetch_hits;
            s.prefetch_issued   = prefetch_issued;
            s.store_writes      = store_writes;
            s.store_skipped     = store_skipped;
            s.store_unchanged   = store_unchanged;
            s.host_pulls        = host_pulls;
            s.host_pull_bytes   = host_pull_bytes;
            s.largest_block     = largest_block;
            s.immovable_bytes   = immovable_bytes;
            s.hash_threshold    = fingerprint_threshold();
            s.hash_count        = hash_count;

            /* How the trace is fed. A period search reads one sequence, so a trace shared
               by several threads is the interleaving of theirs and may have no period even
               when each of them repeats exactly. */
            std::uint64_t total = 0, busiest = 0;
            unsigned threads = 0;
            for (unsigned i = 0; i < xmem::max_traced_threads; ++i)
            {
                const std::uint64_t h = xmem::g::thread_hits[i].load(std::memory_order_relaxed);
                if (h == 0) continue;
                ++threads;
                total += h;
                busiest = std::max(busiest, h);
            }
            s.trace_threads = threads;
            s.trace_busiest = total > 0 ? 100.0 * (double)busiest / (double)total : 0.0;
            s.sync_seconds      = sync_seconds;
            s.sync_count        = sync_count;
            s.wait_seconds      = wait_seconds;
            s.wait_bytes        = wait_bytes;
            s.ahead_seconds     = ahead_seconds;
            s.ahead_bytes       = ahead_bytes;
            s.hash_seconds      = hash_seconds;
            s.store_reads       = store_reads;
            s.pages_advised     = pages_advised;
            s.h2d_bytes         = h2d_bytes;
            s.d2h_bytes         = d2h_bytes;
            s.cycle_locked      = (plan != nullptr);
            s.cycle_period      = plan ? plan->cycle.size() : 0;
            s.prefetch_depth    = prefetch_depth;
            s.access_rate       = access_rate.load(std::memory_order_relaxed);
            s.idle_releases     = idle_releases;
            s.idle_purges       = idle_purges;
            s.idle_released_bytes = idle_released_bytes;
            return s;
        }

        /*
            Sizes and tiers of the managed blocks, largest first, with how many times each
            appears in the detected cycle. The subsystem does not know what a block is called,
            but a size and a number of uses per step are usually enough to recognise a tensor
            in a topology, and they are the only place a graph's real footprint can be seen
            once the layers have stopped agreeing with the arithmetic on paper.
        */
        void report_blocks (std::ostream& out, std::size_t max_rows) const
        {
            std::lock_guard<std::mutex> lk(mu);

            struct row { std::size_t bytes; unsigned char tier; std::size_t uses; };
            std::vector<row> rows;
            const std::shared_ptr<const schedule> p = plan;

            std::size_t total = 0;
            for (block_record* r : by_id)
            {
                if (!r) continue;
                std::size_t uses = 0;
                if (p)
                {
                    auto it = p->positions.find(r->id);
                    if (it != p->positions.end())
                        uses = it->second.size();
                }
                rows.push_back(row{r->bytes, r->state.load(std::memory_order_relaxed), uses});
                total += r->bytes;
            }

            std::sort(rows.begin(), rows.end(),
                      [](const row& a, const row& b) { return a.bytes > b.bytes; });

            const double mib = 1024.0*1024.0;
            out << "extended memory blocks, largest first, " << rows.size() << " managed, "
                << (total/mib) << " MiB in all\n";

            std::size_t shown = 0, shown_bytes = 0;
            for (const row& r : rows)
            {
                if (shown >= max_rows)
                    break;
                const char* tier_name = r.tier == tier_device ? "device"
                                      : r.tier == tier_host   ? "host"
                                      : r.tier == tier_store  ? "store" : "moving";
                out << "  " << (r.bytes/mib) << " MiB\t" << tier_name
                    << "\t" << (100.0*r.bytes/(double)std::max<std::size_t>(total,1)) << " % of the whole"
                    << "\t" << r.uses << " uses per step\n";
                shown_bytes += r.bytes;
                ++shown;
            }
            if (rows.size() > shown)
                out << "  the remaining " << (rows.size() - shown) << " blocks hold "
                    << ((total - shown_bytes)/mib) << " MiB\n";
        }

    private:

        // ------------------------------------------------------------------- allocation

        std::shared_ptr<float> acquire_locked (std::size_t bytes, bool allow_evict)
        {
            quiesced = false;

            if (allow_evict)
            {
                while (resident_locked() + bytes > vram_budget)
                {
                    if (!evict_one_locked())
                    {
                        /* Nothing left that may be moved. A budget below what the blocks
                           under min_block_bytes add up to cannot be honoured, and the run
                           then quietly exceeds it while paying for the eviction machinery
                           that is trying to make it fit. Better to say which of the two
                           numbers is wrong than to let the budget become decorative. */
                        if (!floor_reported && resident_locked() + bytes > vram_budget)
                        {
                            floor_reported = true;
                            report_budget_shortfall(bytes);
                        }
                        break;
                    }
                }
            }

            std::shared_ptr<device_pool> keep = pool;
            for (int attempt = 0; ; ++attempt)
            {
                try
                {
                    float* p = keep->allocate(bytes);
                    return std::shared_ptr<float>(p, [keep, bytes](float* q) { keep->release(q, bytes); });
                }
                catch (cuda_error&)
                {
                    // The budget is advisory; the driver has the last word. Free whatever
                    // is still evictable and try once more before giving up.
                    if (attempt >= 1 || !allow_evict)
                        report_placement_failure(bytes);
                    bool freed = false;
                    while (evict_one_locked())
                        freed = true;
                    keep->trim();
                    if (!freed)
                        report_placement_failure(bytes);
                }
            }
        }

        /*
            Reached only after the subsystem has evicted everything it was allowed to, so
            the numbers in the message are the ones that matter: what is left resident is
            what could not be moved, and the gap between that and the block being asked for
            is what the caller has to close.
        */
        void report_placement_failure (std::size_t bytes) const
        {
            std::size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);

            const double mib = 1024.0*1024.0;
            std::ostringstream m;
            m << "extended memory: cannot place a block of " << (bytes/mib) << " MiB. After "
                 "evicting everything it could, the subsystem still holds " << (device_bytes/mib)
              << " MiB in blocks that are pinned, in use, or below min_block_bytes, and the "
                 "device reports " << (free_b/mib) << " MiB free of " << (total_b/mib) << " MiB. "
                 "Lower vram_budget or min_block_bytes if the resident set is the problem; if "
                 "the block itself is close to the size of the card, no budget can help.";
            throw cuda_error(m.str());
        }

        /*
            Says why the budget could not be met, which is not the same question every time.

            A block that never moves because it is smaller than the threshold is one thing,
            and raising the budget or lowering the threshold answers it. A block that is
            merely pinned at this instant, because a thread is looking at it or a transfer
            is in flight, is another: the budget is reachable and the run has simply asked
            for more at once than the window allows. Naming the wrong one sends the reader
            to the wrong knob, so both are counted before anything is said.
        */
        void report_budget_shortfall (std::size_t wanted)
        {
            std::unordered_set<std::uint32_t> hot;
            collect_hot_locked(hot);

            std::size_t immovable = 0, pinned = 0, in_flight = 0, recent = 0;
            for (block_record* r : by_id)
            {
                if (!r) continue;
                const unsigned char st = r->state.load(std::memory_order_relaxed);
                if (st == tier_transit)   { in_flight += r->bytes; continue; }
                if (st != tier_device)    continue;
                if (!r->evictable)                                    immovable += r->bytes;
                else if (r->pins.load(std::memory_order_relaxed) > 0) pinned    += r->bytes;
                else if (hot.count(r->id))                            recent    += r->bytes;
            }

            const double mib = 1024.0*1024.0;
            std::cerr << "extended memory: the budget of " << (vram_budget/mib)
                      << " MiB cannot be met for a request of " << (wanted/mib) << " MiB. ";

            /* Whichever holds the most is what the reader should act on, and the three call
               for opposite remedies: a block below the threshold never moves, a pinned one
               is being used right now, and one inside the hot window is merely too recent to
               be taken. Naming the wrong one sends the reader to the wrong knob. */
            if (recent >= pinned && recent >= immovable && recent > 0)
            {
                std::cerr << (recent/mib) << " MiB was accessed too recently to be evicted: "
                             "the hot window of " << opt.hot_window << " blocks covers most of "
                             "what is resident. Lower hot_window, or raise the budget.\n";
            }
            else if (pinned >= immovable && pinned > 0)
            {
                std::cerr << (pinned/mib) << " MiB is pinned by threads currently using it";
                if (in_flight > 0)
                    std::cerr << " and " << (in_flight/mib) << " MiB is in transit";
                std::cerr << ", so the budget is reachable but not at this instant. Raise the "
                             "budget if two threads work at once.\n";
            }
            else
            {
                std::cerr << (immovable/mib) << " MiB sits in blocks below min_block_bytes "
                             "and never moves. Raise the budget or lower min_block_bytes.\n";
            }
        }

        /*
            Smallest block worth hashing, from what the run has measured about itself.

            A fingerprint costs a kernel launch, the same handful of microseconds whatever
            the block holds; moving the block costs its size divided by the link. The two
            cross at one launch worth of bandwidth, and nothing about that figure can be
            guessed in advance: it depends on the card, on the driver and on where the store
            sits. Both terms are already counted here, so the threshold is read off them
            rather than chosen, and until there is enough to read it defaults to the size
            below which a block is not evicted at all.

            Guessing it once cost a run twice the traffic it needed, which is the argument
            for not guessing it again.
        */
        std::size_t fingerprint_threshold () const
        {
            if (opt.fingerprint_min_bytes != 0)
                return opt.fingerprint_min_bytes;
            if (hash_count < 32 || hash_seconds <= 0 || wait_seconds <= 0 || wait_bytes == 0)
                return opt.min_block_bytes;

            const double per_hash = hash_seconds / (double)hash_count;
            const double bytes_per_second = (double)wait_bytes / wait_seconds;
            const double breakeven = per_hash * bytes_per_second;

            const std::size_t lo = 256ul * 1024ul;
            const std::size_t hi = 64ul * 1024ul * 1024ul;
            if (breakeven < (double)lo) return lo;
            if (breakeven > (double)hi) return hi;
            return (std::size_t)breakeven;
        }

        // Everything the budget has to cover: managed blocks plus workspaces.
        std::size_t resident_locked () const
        {
            return device_bytes + scratch_bytes.load(std::memory_order_relaxed);
        }

        void quiesce_locked ()
        {
            if (quiesced)
                return;
            /* One synchronization of the compute stream proves that no kernel is running,
               which settles the question for every block at once rather than one at a
               time. It is also the one place in the subsystem that stops the pipeline dead,
               so it is timed: an eviction path that spends its life here is a different
               problem from one that spends it moving bytes, and the two call for opposite
               remedies. */
            const auto t0 = std::chrono::steady_clock::now();
            synchronize_stream(0);
            sync_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
            ++sync_count;
            for (block_record* r : by_id)
            {
                if (r && r->owner)
                    r->owner->device_in_use = false;
            }
            quiesced = true;
        }

        void collect_hot_locked (std::unordered_set<std::uint32_t>& hot) const
        {
            hot.clear();
            const std::uint64_t head = g::hot_head.load(std::memory_order_relaxed);
            const std::uint64_t n    = std::min<std::uint64_t>(
                                           head, std::min<std::uint64_t>(opt.hot_window, g::hot_mask + 1));
            for (std::uint64_t i = 0; i < n; ++i)
                hot.insert(g::hot_ring[(head - 1 - i) & g::hot_mask]);
        }

        block_record* choose_victim_locked ()
        {
            std::unordered_set<std::uint32_t> hot;
            collect_hot_locked(hot);

            const std::uint64_t now = g::clock.load(std::memory_order_relaxed);
            const std::size_t   cur = cursor.load(std::memory_order_relaxed);
            const std::shared_ptr<const schedule> p = plan;

            block_record* best = nullptr;
            std::size_t   best_key = 0;

            for (block_record* r : by_id)
            {
                if (!r || !r->owner) continue;
                if (r->state.load(std::memory_order_relaxed) != tier_device) continue;
                if (!r->evictable) continue;
                if (r->pins.load(std::memory_order_relaxed) != 0) continue;
                if (hot.count(r->id)) continue;

                /* With a schedule, the victim is the block whose next use is furthest away,
                   which is the optimal choice and is only available because the sequence of
                   accesses is known. Without one, fall back on age. */
                const std::size_t key = p
                    ? p->next_use(r->id, cur)
                    : (std::size_t)(now - r->stamp.load(std::memory_order_relaxed));

                if (!best || key > best_key)
                {
                    best = r;
                    best_key = key;
                }
            }
            return best;
        }

        bool evict_one_locked ()
        {
            block_record* r = choose_victim_locked();
            if (!r)
                return false;

            gpu_data& g = *r->owner;
            quiesce_locked();
            g.wait_for_transfer_to_finish();

            if (r->slot >= 0)
            {
                /* The window is the mapping, so a current host copy is a current store
                   copy. Both flags have to be consulted: store_valid covers blocks the
                   program has never read on the host, host_current covers those it has, and
                   the second is the one that is only set once the copy behind it really
                   happened. */
                /* The window is the mapping, so a current host copy is a current store
                   copy. Both flags have to be consulted: store_valid covers blocks the
                   program has never read on the host, host_current covers those it has, and
                   the second is the one that is only set once the copy behind it really
                   happened. */
                bool store_current = r->store_valid.load(std::memory_order_relaxed) ||
                                     (g.data_host && g.host_current);

                /* Neither flag can be trusted to say a block is clean, only to say it is
                   current for a reason the subsystem watched happen. Everything else is
                   marked dirty because a layer takes its parameters through the non-const
                   device(), and a raw pointer handed out for reading is indistinguishable
                   from one handed out for writing. Asking the card what the block now holds
                   settles it: hashing where the data already sits costs a read of device
                   memory, some fifty times less than moving the same bytes over the bus to
                   find out they did not change. */
                /* Only where the answer is worth the question. Hashing a block costs a
                   kernel launch, which is the same handful of microseconds whatever the
                   block holds, while transferring it costs its size divided by the link.
                   Below a few megabytes the transfer is cheaper than the enquiry, and
                   asking anyway turns a saving into a tax: a run whose evictions are mostly
                   small activations spends a tenth of itself hashing blocks it would have
                   been quicker to move. */
                unsigned long long hx = 0, ha = 0;
                bool hashed = false;
                if (!store_current && opt.fingerprint && r->slot_written &&
                    r->bytes >= fingerprint_threshold())
                {
                    try
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        cuda::extended_memory_fingerprint(g.data_device.get(), r->bytes,
                                                          hx, ha, xstream);
                        hash_seconds += std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - t0).count();
                        ++hash_count;
                        hashed = true;
                    }
                    catch (...)
                    {
                        // A fingerprint that cannot be taken is simply one fewer shortcut.
                        hashed = false;
                    }
                }

                if (hashed && r->hash_valid && hx == r->hash_xor && ha == r->hash_add)
                {
                    /* The store already holds exactly this. The eviction becomes a matter of
                       letting go of the device buffer. */
                    store_current = true;
                    ++store_unchanged;
                    r->store_valid.store(true, std::memory_order_relaxed);
                    if (opt.paranoid)
                        verify_unchanged_locked(r, g);
                }

                if (!store_current)
                {
                    /* The destination is the mapping itself, whether or not a window has
                       been handed to the program, so writing the block back and refreshing
                       the host mirror are the same operation rather than two. */
                    device_to_store_locked(r, g);
                    r->store_valid.store(true, std::memory_order_relaxed);
                    if (g.data_host)
                        g.host_current = true;

                    r->hash_valid = hashed;
                    r->hash_xor   = hx;
                    r->hash_add   = ha;
                }
                r->state.store(tier_store, std::memory_order_release);
            }
            else
            {
                if (!g.data_host)
                    allocate_pinned_mirror_locked(g, r);
                if (!g.host_current)
                {
                    CHECK_CUDA(cudaMemcpy(g.data_host.get(), g.data_device.get(),
                                          r->bytes, cudaMemcpyDeviceToHost));
                    d2h_bytes += r->bytes;
                    g.host_current = true;
                }
                r->state.store(tier_host, std::memory_order_release);
            }

            if (opt.paranoid)
                poison_locked(g.data_device.get(), r->bytes);

            g.data_device.reset();
            g.device_current = false;
            g.device_in_use  = false;
            device_bytes -= std::min(device_bytes, r->bytes);
            ++evictions;
            return true;
        }

        void poison_locked (float* p, std::size_t bytes)
        {
            if (!p) return;
            /* Every byte set to 0xff reads back as a NaN, so code that kept this pointer
               across the eviction produces visible NaNs on the very next step instead of
               plausible numbers that drift. */
            const cudaError_t err = cudaMemset(p, 0xff, bytes);
            if (err != cudaSuccess)
                std::cerr << "extended memory: cudaMemset() failed. Reason: "
                          << cudaGetErrorString(err) << std::endl;
        }

        void allocate_pinned_mirror_locked (gpu_data& g, block_record* r)
        {
            void* p = nullptr;
            CHECK_CUDA(cudaMallocHost(&p, r->bytes));
            g.data_host.reset((float*)p, [](float* q) {
                const cudaError_t err = cudaFreeHost(q);
                if (err != cudaSuccess)
                    std::cerr << "cudaFreeHost() failed. Reason: " << cudaGetErrorString(err) << std::endl;
            });
            pinned_bytes += r->bytes;
        }

        /*
            Hands the program a pointer into the mapping. The deleter does nothing because
            the slot belongs to the record and outlives any particular window onto it.
        */
        void install_store_window_locked (gpu_data& g, block_record* r)
        {
            float* window = (float*)(arena->base() + r->slot);
            g.data_host.reset(window, [](float*) {});
        }

        /*
            Reads the block back and compares it against the store, to check a decision the
            fingerprint has already taken. A difference here means two blocks with different
            contents produced the same pair of accumulators, which at 128 bits should not
            happen in the life of any program; seeing it reported means something else is
            wrong, most likely that the block was written while the eviction was in flight.
        */
        void verify_unchanged_locked (block_record* r, gpu_data& g)
        {
            std::lock_guard<std::mutex> sg(engine->mu);
            std::vector<char> shadow(r->bytes);
            std::memcpy(shadow.data(), arena->base() + r->slot, r->bytes);
            engine->from_device(shadow.data(), g.data_device.get(), r->bytes, false);
            if (std::memcmp(shadow.data(), arena->base() + r->slot, r->bytes) != 0)
            {
                std::cerr << "extended memory: a block the fingerprint reported as unchanged "
                             "does not match the store. The eviction has been carried out on a "
                             "wrong answer and data has been lost.\n";
            }
        }

        void device_to_store_locked (block_record* r, gpu_data& g)
        {
            /* Comparing the block against the slot before writing costs a host memcmp and
               saves a disk write. It matters because dlib hands a layer's parameters out
               through the non-const device(), which the subsystem has no choice but to read
               as "this may be written": in a forward pass nothing is, but every weight is
               marked dirty all the same and would be written back on every eviction. A
               generation loop then writes the whole model to disk once per pass, for nothing.

               The comparison is only worth doing while it keeps succeeding. A block whose
               device copy has differed twice running is one that really is being modified,
               most likely a gradient or an optimizer state, and it stops being compared. */
            arena->advise_willneed(r->slot, r->bytes);
            std::lock_guard<std::mutex> sg(engine->mu);

            const bool compare = r->slot_written && r->store_diffs < 2;
            const auto t0 = std::chrono::steady_clock::now();
            const bool wrote = engine->from_device(arena->base() + r->slot, g.data_device.get(),
                                                   r->bytes, compare);
            wait_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
            wait_bytes += r->bytes;
            d2h_bytes += r->bytes;
            r->slot_written = true;

            if (compare && !wrote)
            {
                r->store_diffs = 0;
                ++store_skipped;
            }
            else
            {
                if (compare)
                    ++r->store_diffs;
                ++store_writes;
            }
        }

        void store_to_device_locked (block_record* r, gpu_data& g)
        {
            const char* src = arena->base() + r->slot;
            std::lock_guard<std::mutex> sg(engine->mu);
            const auto t0 = std::chrono::steady_clock::now();
            engine->to_device(g.data_device.get(), src, r->bytes);
            wait_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
            wait_bytes += r->bytes;
            h2d_bytes += r->bytes;
            ++store_reads;
        }

        void wait_out_transit (std::unique_lock<std::mutex>& lk, block_record* r)
        {
            if (r->state.load(std::memory_order_acquire) != tier_transit)
                return;
            // The block being asked for is already on its way in, which is the prefetcher
            // doing exactly what it is there for.
            ++prefetch_hits;
            while (r->state.load(std::memory_order_acquire) == tier_transit)
                transit_done.wait(lk);
        }

        // -------------------------------------------------------------------- observing

        void worker_loop ()
        {
            /* Nothing this thread does is required for correctness: it makes residency
               better informed, and a failure in it should cost the schedule rather than the
               program. Anything thrown here therefore ends the thread quietly and leaves the
               compute side to carry on with least recently used eviction. */
            try
            {
                cudaSetDevice(dev);

                std::vector<std::uint32_t> snapshot;
                snapshot.reserve(trace_capacity);

                std::uint64_t last_head   = g::trace_head.load(std::memory_order_relaxed);
                auto          last_time   = std::chrono::steady_clock::now();
                auto          last_access = last_time;
                auto          nap         = std::chrono::microseconds(2000);
                unsigned      faults      = 0;

                while (!stopping.load(std::memory_order_acquire))
                {
                    std::this_thread::sleep_for(nap);
                    if (!g::active.load(std::memory_order_relaxed))
                        continue;

                    /* Prefetching against a cursor that is not moving is work done for a
                       position the program is not approaching, and while an idle release is
                       under way it is worse than useless: the two would fight over the same
                       blocks, one evicting what the other has just fetched. */
                    const bool moving = g::trace_head.load(std::memory_order_relaxed) != last_head;

                    /* A pass can fail on its own: a transfer that hits a transient driver
                       error, a store read that fails. None of that is a reason to give up
                       the schedule for the rest of the run, so a pass that throws is
                       counted and the next one is tried. Only a fault that keeps repeating
                       is treated as the thread having nothing useful left to do. */
                    try
                    {
                        update_plan(snapshot);
                        if (moving)
                        {
                            advise_pass();
                            prefetch_pass();
                        }
                        faults = 0;
                    }
                    catch (std::exception& e)
                    {
                        if (++faults >= 16)
                            throw;
                        if (opt.verbose)
                            std::cerr << "extended memory: a scheduling pass failed: "
                                      << e.what() << "\n";
                    }

                    if (opt.idle_release_ms > 0 && !moving)
                    {
                        const auto quiet = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - last_access).count();
                        const long long purge_after = opt.idle_purge_ms > 0
                            ? (long long)opt.idle_purge_ms
                            : 8ll * (long long)opt.idle_release_ms;

                        if (!idle_purged && quiet >= purge_after)
                        {
                            std::lock_guard<std::mutex> lk(mu);
                            if (release_idle_locked(true))
                            {
                                idle_purged  = true;
                                idle_relaxed = true;
                            }
                        }
                        else if (!idle_relaxed && quiet >= (long long)opt.idle_release_ms)
                        {
                            std::lock_guard<std::mutex> lk(mu);
                            idle_relaxed = release_idle_locked(false);
                        }
                    }

                    /* Pace the loop off the program rather than off a constant. A fixed
                       interval is either too slow, in which case the cursor moves further
                       between two passes than the horizon reaches and the prefetcher is
                       permanently behind, or too fast, in which case the thread spends its
                       time taking a lock the compute side needs. Waking roughly every
                       quarter of a horizon keeps it ahead without either. */
                    const auto        now   = std::chrono::steady_clock::now();
                    const std::uint64_t head = g::trace_head.load(std::memory_order_relaxed);
                    const double dt = std::chrono::duration<double>(now - last_time).count();
                    const std::uint64_t moved = head - std::min(head, last_head);
                    last_head = head;
                    last_time = now;

                    /* Only while there is something to measure. Left unguarded, the first
                       pass after the program stops working overwrites the figure with zero,
                       which is precisely the moment anyone reads it. */
                    if (dt > 0 && moved > 0)
                        access_rate.store((double)moved / dt, std::memory_order_relaxed);

                    if (moved > 0)
                    {
                        last_access  = now;
                        idle_relaxed = false;
                        idle_purged  = false;
                    }

                    const double rate = access_rate.load(std::memory_order_relaxed);
                    if (moved == 0 || rate <= 0)
                    {
                        nap = std::chrono::microseconds(4000);
                    }
                    else
                    {
                        const double target = (opt.lookahead / 4.0) / rate;
                        const long   us     = (long)(target * 1e6);
                        nap = std::chrono::microseconds(std::min<long>(std::max<long>(us, 200), 5000));
                    }
                }
            }
            catch (std::exception& e)
            {
                if (opt.verbose)
                    std::cerr << "extended memory: the observation thread stopped: "
                              << e.what() << "\n";
            }
            catch (...)
            {
            }
        }

        void update_plan (std::vector<std::uint32_t>& snapshot)
        {
            const std::uint64_t head = g::trace_head.load(std::memory_order_acquire);
            if (head < 64)
                return;

            {
                std::lock_guard<std::mutex> lk(mu);
                if (plan)
                {
                    /* Follow the cursor arithmetically and check it now and then. The trace
                       is the only witness of where the program is in its cycle, so a wrong
                       answer here costs prefetch accuracy and never correctness. */
                    const std::size_t P = plan->cycle.size();
                    const std::size_t c = (anchor_pos + (std::size_t)((head - 1 - anchor_head) % P)) % P;
                    cursor.store(c, std::memory_order_relaxed);

                    if (g::trace_ring[(head - 1) & g::trace_mask] != plan->cycle[c])
                    {
                        if (++drift >= 8)
                        {
                            plan.reset();
                            drift = 0;
                            if (opt.verbose)
                                std::cerr << "extended memory: access cycle lost, "
                                             "falling back on least recently used\n";
                        }
                    }
                    else
                    {
                        drift = 0;
                    }
                    if (plan)
                        return;
                }
            }

            const std::size_t L = (std::size_t)std::min<std::uint64_t>(head, trace_capacity);
            snapshot.resize(L);
            for (std::size_t i = 0; i < L; ++i)
                snapshot[i] = g::trace_ring[(head - L + i) & g::trace_mask];

            std::size_t period = 0;
            if (!find_period(snapshot, period))
                return;

            std::shared_ptr<schedule> s = std::make_shared<schedule>();
            s->cycle.assign(snapshot.end() - (std::ptrdiff_t)period, snapshot.end());
            for (std::size_t i = 0; i < s->cycle.size(); ++i)
                s->positions[s->cycle[i]].push_back(i);

            /* A handful of blocks going round in a tight loop is not the step of a network,
               it is a tensor being filled. Deserializing a model produces exactly that, and
               adopting it means announcing a cycle and losing it again a moment later. */
            if (s->positions.size() < 8)
                return;

            /* Nor is a period claiming an implausible number of accesses per block. The
               bound has to admit a training step, which touches each of its tensors far
               more often than a forward pass does: the forward reads a weight, the backward
               reads it again and writes its gradient, and the solver then reads that
               gradient and writes the weight and its optimizer moments. A dozen or so per
               block is ordinary there against about five in inference, so two dozen is
               generous for anything real while still rejecting a coincidence the trace
               happened to confirm over the window, which runs to several dozen. */
            const std::size_t per_block = period / std::max<std::size_t>(s->positions.size(), 1);
            if (per_block > 24)
            {
                if (opt.verbose)
                    std::cerr << "extended memory: a period of " << period << " over "
                              << s->positions.size() << " blocks is " << per_block
                              << " accesses each, too many to be one step; not adopted\n";
                return;
            }

            std::lock_guard<std::mutex> lk(mu);
            plan        = s;
            anchor_head = head - 1;
            anchor_pos  = period - 1;
            cursor.store(anchor_pos, std::memory_order_relaxed);
            drift = 0;
            if (opt.verbose)
                std::cerr << "extended memory: access cycle identified over " << period
                          << " accesses across " << s->positions.size() << " blocks\n";
        }

        /*
            The deeper of the two horizons. Bringing a slot's pages into memory costs
            nothing but a hint, so this runs much further ahead than the transfers do and
            leaves the kernel time to fetch from disk before the block's turn arrives.
        */
        void advise_pass ()
        {
            if (!arena || opt.advise_horizon <= opt.lookahead)
                return;

            std::lock_guard<std::mutex> lk(mu);
            const std::shared_ptr<const schedule> p = plan;
            if (!p || p->cycle.empty())
                return;

            const std::size_t P = p->cycle.size();
            const std::size_t c = cursor.load(std::memory_order_relaxed);

            /* Only the far edge of the horizon is new since the last pass. Advising the
               whole window every time would issue the same hint dozens of times and turn a
               cheap idea into a syscall storm. */
            const std::uint64_t head  = g::trace_head.load(std::memory_order_relaxed);
            const std::uint64_t moved = head - std::min(head, advise_head);
            advise_head = head;
            if (moved == 0)
                return;

            const unsigned span = (unsigned)std::min<std::uint64_t>(
                                      moved, opt.advise_horizon - opt.lookahead);

            for (unsigned k = opt.advise_horizon - span + 1; k <= opt.advise_horizon; ++k)
            {
                const std::uint32_t id = p->cycle[(c + k) % P];
                if (id >= by_id.size())
                    continue;
                block_record* r = by_id[id];
                if (!r || r->slot < 0)
                    continue;
                if (r->state.load(std::memory_order_relaxed) != tier_store)
                    continue;
                if (!r->store_valid.load(std::memory_order_relaxed))
                    continue;
                arena->advise_willneed(r->slot, r->bytes);
                ++pages_advised;
            }
        }

        /*
            Hands the card back when the program stops using it.

            This is the one place where a thread other than a compute thread lowers the tier
            of a block, and the condition that makes it safe is the same condition that makes
            it useful: nothing has been accessed for idle_release_ms. A device nobody is
            touching has no kernel in flight, so the synchronization that has to precede a
            recycle is free rather than a stall injected into a running workload, and the
            cudaFree that actually returns the memory to the driver costs nothing either.

            The hot window is respected here as everywhere else, and during idleness that
            protection is at its strongest: no access has occurred, so the ring has not
            moved, and every block a stalled thread might still hold a pointer to is exactly
            the set the window covers. The last blocks touched are therefore kept, which is
            a bounded amount of memory and the price of not having to reason about what a
            blocked thread is holding.

            Evicting is not enough on its own. A released buffer goes back to the pool, where
            nvidia-smi still counts it against the process, so the pool is trimmed once the
            sweep runs out of victims. That call is what another process on the same card
            actually sees.
        */
        /*
            Hands memory back when the program stops using it, in two steps.

            This is the one place where a thread other than a compute thread lowers the tier
            of a block, and the condition that makes it safe is the same condition that makes
            it useful: nothing has been accessed for the configured interval. A device nobody
            is touching has no kernel in flight, so the synchronization that has to precede a
            recycle is free rather than a stall injected into a running workload, and the
            cudaFree that actually returns memory to the driver costs nothing either.

            The hot window is respected here as everywhere else, and during idleness that
            protection is at its strongest: no access has occurred, so the ring has not moved,
            and every block a stalled thread might still hold a pointer to is exactly the set
            the window covers. Those blocks are kept, which bounds what a sweep can reclaim
            and is the price of not having to reason about what a blocked thread is holding.

            The first step only falls back to a fraction of the budget, because on a host
            serving several models a brief pause is not a reason to drop the one that is
            about to be asked for again. Which blocks go first needs no special handling: a
            model absent from the observed cycle has no next use, so furthest next use ranks
            its blocks ahead of everything belonging to the model still in play. The second
            step, after a longer silence, empties what is left and trims the pool, since a
            buffer sitting in the pool is still charged to the process as far as anything
            outside it can tell.
        */
        bool release_idle_locked (bool purge)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            const std::size_t before = device_bytes;
            const std::size_t target = purge
                ? 0
                : (std::size_t)((double)vram_budget * opt.idle_release_keep);

            /* A sweep holds the lock, so a request that arrives in the middle of one waits
               for it. Watching the trace and stopping the moment it moves keeps that wait
               to a single eviction instead of the whole deadline, which on a server is the
               difference between a pause nobody notices and one that shows up in the
               latency of the first request after every lull. */
            const std::uint64_t quiet_at = g::trace_head.load(std::memory_order_relaxed);

            quiesced = false;
            bool more = true;
            while (more && device_bytes > target &&
                   std::chrono::steady_clock::now() < deadline)
            {
                if (g::trace_head.load(std::memory_order_relaxed) != quiet_at)
                    return false;             // the program woke up, do not make it wait
                more = evict_one_locked();
            }

            if (device_bytes > target && more)
                return false;                 // out of time, the next pass carries on

            if (purge)
            {
                pool->trim();
                ++idle_purges;
            }

            const std::size_t freed = before - std::min(before, device_bytes);
            idle_released_bytes += freed;
            ++idle_releases;

            if (opt.verbose && freed > 0)
                std::cerr << "extended memory: idle, released " << (freed >> 20) << " MiB"
                          << (purge ? " back to the driver" : "")
                          << ", " << (device_bytes >> 20) << " MiB still held\n";
            return true;
        }

        void prefetch_pass ()
        {
            std::unique_lock<std::mutex> lk(mu);
            const std::shared_ptr<const schedule> p = plan;
            if (!p || p->cycle.empty())
                return;

            const std::size_t P = p->cycle.size();
            const std::size_t c = cursor.load(std::memory_order_relaxed);

            /* The horizon is counted in blocks actually brought in, not in cycle positions.
               Most of a decoder's accesses land on small tensors that never leave the
               device, so a horizon of twenty-four positions can cover only a handful of the
               blocks that are worth moving, and the prefetcher falls behind while appearing
               to be configured generously. Scanning until the work is queued, rather than
               until a position count is reached, is what makes lookahead mean what it says.

               The scan is bounded in both directions: by how far ahead it is willing to
               look, and by how long the pass may take, because a pass that runs for longer
               than the cursor takes to move is planning against a position the program has
               already left. */
            const unsigned scan_cap = std::max<unsigned>(opt.advise_horizon, opt.lookahead * 8);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(8);

            unsigned queued  = 0;
            unsigned skipped = 0;

            for (unsigned k = 1; k <= scan_cap && queued < opt.lookahead; ++k)
            {
                const std::uint32_t id = p->cycle[(c + k) % P];
                if (id >= by_id.size())
                    continue;
                block_record* r = by_id[id];
                if (!r || !r->owner)
                    continue;
                const unsigned char st = r->state.load(std::memory_order_acquire);
                if (st != tier_host && st != tier_store)
                    continue;

                /* The observation thread never evicts. It fills the room the budget already
                   leaves: making residency smaller is the compute thread's business alone,
                   and that division is what lets the accessor fast path run without a lock.
                   A block that does not fit is passed over rather than ending the scan,
                   since a smaller one further out still fits and would otherwise be missed
                   for no reason. */
                if (resident_locked() + r->bytes > vram_budget)
                {
                    if (++skipped >= 8)
                        break;
                    continue;
                }

                bring_in_locked(lk, r);
                ++queued;
                ++prefetch_issued;

                if (std::chrono::steady_clock::now() >= deadline)
                    break;
            }
            prefetch_depth = queued;
        }

        void bring_in_locked (std::unique_lock<std::mutex>& lk, block_record* r)
        {
            gpu_data& g = *r->owner;
            const unsigned char from = r->state.load(std::memory_order_relaxed);

            std::shared_ptr<float> buf;
            try
            {
                buf = acquire_locked(r->bytes, false);
            }
            catch (...)
            {
                return;                       // no room: the compute thread will do it later
            }

            g.data_device = buf;
            device_bytes += r->bytes;
            device_peak = std::max(device_peak, device_bytes);
            r->pins.fetch_add(1, std::memory_order_acq_rel);
            r->state.store(tier_transit, std::memory_order_release);

            const bool have_store = r->slot >= 0 &&
                                    (r->store_valid.load(std::memory_order_relaxed) ||
                                     (g.data_host && g.host_current));

            /* Everything the transfer needs is copied out here, while the lock is held. A
               gpu_data can be swapped with another one at any moment, and a reference taken
               now would then describe the wrong object; the raw pointers do not, because a
               swap moves the buffer and the record together, so the memory written is the
               memory the record still owns. Which object to mark afterwards is read again
               below, once the lock is back. */
            float*      dst = g.data_device.get();
            const void* src = (from == tier_host)
                ? (const void*)g.data_host.get()
                : (have_store ? (const void*)(arena->base() + r->slot) : nullptr);
            const std::size_t nbytes = r->bytes;
            bool ok = true;
            const auto t0 = std::chrono::steady_clock::now();

            lk.unlock();
            try
            {
                if (from == tier_host && src)
                {
                    // A pinned mirror can feed the DMA engine directly, so staging it a
                    // second time would only add a memcpy.
                    std::lock_guard<std::mutex> sg(engine->mu);
                    CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, xstream));
                    synchronize_stream(xstream);
                }
                else if (src)
                {
                    std::lock_guard<std::mutex> sg(engine->mu);
                    engine->to_device(dst, src, nbytes);
                }
            }
            catch (...)
            {
                ok = false;
            }
            lk.lock();
            /* This one runs on the observation thread, alongside the computation rather than
               in front of it, so it belongs in a different column: a second spent here is a
               second the program did not wait for. Adding it to the same total was what made
               the figure exceed the wall clock. */
            ahead_seconds += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0).count();
            ahead_bytes += nbytes;

            r->pins.fetch_sub(1, std::memory_order_acq_rel);

            gpu_data& owner = *r->owner;
            if (ok)
            {
                h2d_bytes += nbytes;
                if (from == tier_store && have_store)
                    ++store_reads;
                owner.device_current = true;
                if (from != tier_host)
                    owner.host_current = (owner.data_host != nullptr) && owner.host_current;
                r->state.store(tier_device, std::memory_order_release);
            }
            else
            {
                owner.data_device.reset();
                owner.device_current = false;
                device_bytes -= std::min(device_bytes, nbytes);
                r->state.store(from, std::memory_order_release);
            }
            transit_done.notify_all();
        }

        // ----------------------------------------------------------------------- state

        mutable std::mutex      mu;
        std::condition_variable transit_done;

        extended_memory_options opt;
        int                     dev                = 0;
        std::size_t             vram_budget        = 0;
        std::size_t             total_device_bytes = 0;

        std::shared_ptr<device_pool>     pool;
        std::unique_ptr<memory_arena>    arena;
        std::unique_ptr<transfer_engine> engine;
        cudaStream_t                     xstream = nullptr;

        std::vector<block_record*>  by_id;
        std::vector<std::uint32_t>  free_ids;

        std::size_t device_bytes   = 0;
        std::size_t device_peak    = 0;
        std::atomic<std::size_t> scratch_bytes {0};
        std::size_t pinned_bytes   = 0;
        std::size_t managed_blocks = 0;
        bool        quiesced       = false;
        bool        arena_full_reported = false;

        std::size_t evictions       = 0;
        std::size_t restores        = 0;
        std::size_t prefetch_hits   = 0;
        std::size_t prefetch_issued = 0;
        std::size_t store_writes    = 0;
        std::size_t store_skipped   = 0;
        std::size_t store_unchanged = 0;
        std::size_t host_pulls      = 0;
        std::size_t largest_block   = 0;
        std::size_t immovable_bytes = 0;
        bool        floor_reported  = false;
        unsigned long long host_pull_bytes = 0;
        double      sync_seconds    = 0;
        std::size_t sync_count      = 0;
        double      wait_seconds    = 0;
        double      ahead_seconds   = 0;
        unsigned long long wait_bytes  = 0;
        unsigned long long ahead_bytes = 0;
        double      hash_seconds    = 0;
        std::size_t hash_count      = 0;
        std::size_t store_reads     = 0;
        std::size_t pages_advised   = 0;
        std::size_t idle_releases   = 0;
        std::size_t idle_purges     = 0;
        std::size_t idle_released_bytes = 0;
        bool        idle_relaxed    = false;
        bool        idle_purged     = false;
        unsigned    prefetch_depth  = 0;
        std::atomic<double> access_rate {0.0};
        unsigned long long h2d_bytes = 0;
        unsigned long long d2h_bytes = 0;

        std::shared_ptr<const schedule> plan;
        std::atomic<std::size_t>        cursor      {0};
        std::uint64_t                   anchor_head = 0;
        std::size_t                     anchor_pos  = 0;
        std::uint64_t                   advise_head = 0;
        unsigned                        drift       = 0;

        std::thread       worker;
        std::atomic<bool> stopping {false};
    };

// ----------------------------------------------------------------------------------------
// Bindings called from gpu_data.
// ----------------------------------------------------------------------------------------

    block_record* register_block (gpu_data* owner, std::size_t bytes, int device_id)
    {
        manager* m = manager::singleton();
        if (!m || !g::active.load(std::memory_order_relaxed))
            return nullptr;
        return m->register_block(owner, bytes, device_id);
    }

    void unregister_block (block_record* r)
    {
        manager* m = manager::singleton();
        if (m) m->unregister_block(r);
    }

    registry_guard::registry_guard (bool engage) : held(false)
    {
        manager* m = manager::singleton();
        if (engage && m)
        {
            m->lock_registry();
            held = true;
        }
    }

    registry_guard::~registry_guard ()
    {
        manager* m = manager::singleton();
        if (held && m)
            m->unlock_registry();
    }

    void restore_device (block_record* r, bool need_content)
    {
        manager* m = manager::singleton();
        if (m) m->restore_device(r, need_content);
    }

    void before_host (block_record* r, bool need_content, bool writes)
    {
        manager* m = manager::singleton();
        if (m) m->before_host(r, need_content, writes);
    }

    void pin_block (block_record* r)
    {
        manager* m = manager::singleton();
        if (m) m->pin_block(r);
    }

    void unpin_block (block_record* r)
    {
        manager* m = manager::singleton();
        if (m) m->unpin_block(r);
    }

    bool store_backed (std::size_t bytes)
    {
        manager* m = manager::singleton();
        return m && g::active.load(std::memory_order_relaxed) && m->store_backed(bytes);
    }

    std::shared_ptr<void> acquire_scratch (std::size_t bytes)
    {
        manager* m = manager::singleton();
        if (!m || bytes == 0 || !g::active.load(std::memory_order_relaxed))
            return std::shared_ptr<void>();
        return m->acquire_scratch(bytes);
    }

    /*
        Registered with atexit() when the subsystem starts, which is during main() and so
        after every static object has been constructed. Exit handlers run in reverse order
        of registration, so this one runs before the destructor of any global network and
        before the CUDA runtime's own teardown, which is exactly the window in which
        stopping the thread is still safe.
    */
    static void stop_worker_at_exit ()
    {
        manager* m = manager::singleton();
        if (m) m->stop_worker();
    }

} // namespace xmem

// ----------------------------------------------------------------------------------------
// Public interface.
// ----------------------------------------------------------------------------------------

    bool enable_extended_memory (
        const extended_memory_options& opts
    )
    {
        DLIB_CASSERT(opts.vram_budget_fraction > 0 && opts.vram_budget_fraction <= 0.95,
                     "extended memory: vram_budget_fraction must be in (0, 0.95]");
        DLIB_CASSERT(opts.hot_window > 0, "extended memory: hot_window must be positive");
        DLIB_CASSERT(opts.advise_horizon >= opts.lookahead,
                     "extended memory: advise_horizon must not be shorter than lookahead");

        if (xmem::g::active.load(std::memory_order_relaxed))
            throw dlib::error("extended memory: already enabled. The subsystem is switched on "
                              "once, and stays on for the life of the process.");

        /* Where a block's host copy lives is settled when the block is sized, so a network
           built before this call would keep pinned mirrors for the rest of the run. Saying
           so is better than letting the memory quietly fail to drop. */
        if (xmem::g::blocks_created.load(std::memory_order_relaxed) != 0)
            throw dlib::error("extended memory: tensors already exist. enable_extended_memory() "
                              "belongs in the first statements of main(), before any network or "
                              "tensor is constructed.");

        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));

        std::size_t free_bytes = 0, total_bytes = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));

        extended_memory_options o = opts;
        if (o.vram_budget == 0)
            o.vram_budget = (std::size_t)((double)free_bytes * o.vram_budget_fraction);
        if (o.store_bytes == 0)
        {
            /* The store is a sparse file, so its declared size costs nothing until pages
               are written to it, and the only thing a small default buys is the risk of
               filling up mid run and falling back to pinned mirrors. Training holds
               gradients and optimizer states on top of the weights, several times what a
               budget sized for inference would suggest, so the default is generous and the
               cap below is what actually bounds it. */
            o.store_bytes = o.vram_budget * 64;
        }

        if (!xmem::g::trace_ring)
        {
            xmem::g::trace_ring = new std::uint32_t[xmem::trace_capacity];
            xmem::g::trace_mask = xmem::trace_capacity - 1;
            std::fill(xmem::g::trace_ring, xmem::g::trace_ring + xmem::trace_capacity, xmem::no_block);
        }
        if (!xmem::g::hot_ring)
        {
            xmem::g::hot_ring = new std::uint32_t[xmem::hot_capacity];
            xmem::g::hot_mask = xmem::hot_capacity - 1;
            std::fill(xmem::g::hot_ring, xmem::g::hot_ring + xmem::hot_capacity, xmem::no_block);
        }

        xmem::manager*& m = xmem::manager::singleton();
        m = new xmem::manager(o, dev, o.vram_budget);
        xmem::g::active.store(true, std::memory_order_release);
        m->start_worker();
        std::atexit(&xmem::stop_worker_at_exit);

        if (o.verbose)
        {
            std::cerr << "extended memory: enabled on device " << dev
                      << " with a budget of " << (o.vram_budget >> 20) << " MiB";
            if (m->has_arena())
                std::cerr << ", store of " << (o.store_bytes >> 20) << " MiB mapped in "
                          << o.store_path;
            else
                std::cerr << ", no store, evicted blocks stay in pinned host memory";
            std::cerr << "\n";
        }
        return true;
    }

    bool extended_memory_enabled ()
    {
        return xmem::g::active.load(std::memory_order_relaxed);
    }

    extended_memory_stats get_extended_memory_stats ()
    {
        xmem::manager* m = xmem::manager::singleton();
        if (!m || !xmem::g::active.load(std::memory_order_relaxed))
            return extended_memory_stats();
        return m->stats();
    }

    void print_extended_memory_blocks (
        std::ostream& out,
        std::size_t max_rows
    )
    {
        xmem::manager* m = xmem::manager::singleton();
        if (!m || !xmem::g::active.load(std::memory_order_relaxed))
        {
            out << "extended memory: disabled\n";
            return;
        }
        m->report_blocks(out, max_rows);
    }

// ----------------------------------------------------------------------------------------

    void print_extended_memory_stats (
        std::ostream& out
    )
    {
        const extended_memory_stats s = get_extended_memory_stats();
        if (!s.enabled)
        {
            out << "extended memory: disabled\n";
            return;
        }

        const double mib = 1024.0*1024.0;
        out << "extended memory\n"
            << "  budget          " << (s.vram_budget/mib)       << " MiB\n"
            << "  on device       " << (s.device_bytes/mib)      << " MiB (peak "
                                    << (s.device_peak_bytes/mib) << " MiB)\n"
            << "  workspaces      " << (s.scratch_bytes/mib)     << " MiB\n"
            << "  pinned mirrors  " << (s.pinned_bytes/mib)      << " MiB\n"
            << "  in the store    " << (s.store_bytes/mib)       << " MiB of "
                                    << (s.store_capacity/mib)    << " MiB mapped\n"
            << "  blocks          " << s.managed_blocks          << ", largest "
                                    << (s.largest_block/mib)     << " MiB, immovable "
                                    << (s.immovable_bytes/mib)   << " MiB\n"
            << "  evictions       " << s.evictions               << "\n"
            << "  restores        " << s.restores                << " of which anticipated "
                                    << s.prefetch_hits           << "\n"
            << "  prefetched      " << s.prefetch_issued         << ", pages advised "
                                    << s.pages_advised           << "\n"
            << "  store traffic   " << s.store_writes << " writes, " << s.store_skipped
                                    << " skipped, " << s.store_reads << " reads\n"
            << "  not read back   " << s.store_unchanged
                                    << " evictions settled by fingerprint\n"
            << "  transfers       " << (s.h2d_bytes/mib) << " MiB up, "
                                    << (s.d2h_bytes/mib) << " MiB down\n"
            << "  host pull backs " << s.host_pulls << ", " << (s.host_pull_bytes/mib)
                                    << " MiB brought back by host()\n"
            << "  moving on path  " << s.wait_seconds << " s for " << (s.wait_bytes/mib)
                                    << " MiB, "
                                    << (s.wait_seconds > 0 ? (s.wait_bytes/mib/1024.0)/s.wait_seconds : 0.0)
                                    << " GiB/s\n"
            << "  moving ahead    " << s.ahead_seconds << " s for " << (s.ahead_bytes/mib)
                                    << " MiB, "
                                    << (s.ahead_seconds > 0 ? (s.ahead_bytes/mib/1024.0)/s.ahead_seconds : 0.0)
                                    << " GiB/s\n"
            << "  time hashing    " << s.hash_seconds << " s over " << s.hash_count
                                    << " fingerprints, above " << (s.hash_threshold/1024)
                                    << " KiB\n"
            << "  time stalled    " << s.sync_seconds     << " s over " << s.sync_count
                                    << " device synchronisations\n"
            << "  access trace    " << s.trace_threads << " thread"
                                    << (s.trace_threads == 1 ? "" : "s") << " feeding it, busiest "
                                    << s.trace_busiest << " %\n"
            << "  access cycle    ";
        if (s.cycle_locked)
            out << "identified, period " << s.cycle_period << "\n";
        else
            out << "not identified, evicting by least recent use\n";

        if (s.restores > 0)
        {
            /* The ratio that says whether the schedule is being followed or merely found.
               A low figure with a locked cycle means the horizon is not reaching far enough
               or the bus cannot keep up, and those are different problems: raise lookahead
               for the first, raise the budget for the second. */
            const double coverage = 100.0 * (double)s.prefetch_hits / (double)s.restores;
            out << "  anticipated     " << coverage << " % of restores, last pass queued "
                << s.prefetch_depth << " blocks\n";
        }
        if (s.access_rate > 0)
            out << "  access rate     " << (s.access_rate/1000.0) << " k/s\n";
        if (s.idle_releases > 0)
            out << "  idle releases   " << s.idle_releases << " (" << s.idle_purges
                << " full), " << (s.idle_released_bytes/mib) << " MiB freed\n";
    }

// ----------------------------------------------------------------------------------------

    void device_scope::take (const gpu_data* g)
    {
        if (!g || count >= (unsigned)(sizeof(pinned)/sizeof(pinned[0])))
            return;
        xmem::block_record* r = xmem::manager::record_of(*g);
        if (!r)
            return;
        xmem::pin_block(r);
        pinned[count++] = r;
    }

    device_scope::device_scope (
        std::initializer_list<const tensor*> tensors
    )
    {
        if (!xmem::active())
            return;
        for (const tensor* t : tensors)
        {
            if (t)
                take(&t->data());
        }
    }

    device_scope::device_scope (
        std::initializer_list<const gpu_data*> blocks
    )
    {
        if (!xmem::active())
            return;
        for (const gpu_data* g : blocks)
            take(g);
    }

    device_scope::~device_scope ()
    {
        for (unsigned i = 0; i < count; ++i)
            xmem::unpin_block(pinned[i]);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_USE_CUDA

#endif // DLIB_EXTENDED_MEMORY_CPP_
