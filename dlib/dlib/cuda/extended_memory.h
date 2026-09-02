// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Software-extended device memory.
//
// A dlib network holds its weights, gradients, optimizer states and activations in
// gpu_data blocks, and every one of those blocks keeps a device copy alive from the moment
// it is sized until the network is destroyed. The footprint of a run is therefore the sum
// of everything the graph contains, which is why a model that would fit comfortably in a
// streamed execution refuses to start on a smaller card.
//
// This subsystem removes that constraint by making device residency a property the runtime
// chooses rather than a consequence of the block existing. A managed block lives in one of
// three tiers,
//
//     device   the block occupies VRAM and can be read by a kernel,
//     host     the device buffer has been released, the contents are in a pinned mirror,
//     store    the device buffer has been released, the contents are in the mapped arena,
//
// and moves between them under a budget. Nothing above gpu_data is aware of this: the
// pointers returned by device() and host() are the same pointers as before, obtained
// through the same calls, and a program built against stock dlib keeps working unchanged.
//
// WHY THIS IS NOT THE OPERATING SYSTEM'S JOB
//
// The obvious alternative is to let a pager do the work and fault pages in as the GPU
// touches them. That gives transparency and not performance, for one reason: a pager learns
// that a page is needed at the instant it is too late to fetch it, so the device stalls for
// the full latency of the transfer on every miss. Prefetching is what makes streaming
// viable, and prefetching requires knowing the future.
//
// A pager cannot know it. This subsystem can, because the access pattern of a network is
// not arbitrary. Training and inference walk the same blocks in the same order on every
// step, the set of blocks is fixed after the first step, and the order is a property of
// the graph rather than of the data. So the access sequence is periodic, and a periodic
// sequence is exactly the case where the future is a lookup rather than a guess.
//
// The subsystem exploits that in three places. It prefetches the blocks the schedule says
// are coming, far enough ahead to cover the transfer. It evicts by furthest next use rather
// than by least recent use, which is the optimal replacement policy and is unavailable to
// anything that cannot see forward. And it warms the page cache further ahead still, so
// that when a block's turn comes its pages are already in memory.
//
// Two details decide whether that horizon is real. It is counted in blocks brought in
// rather than in positions of the cycle, because most accesses in a network land on small
// tensors that never leave the device and a horizon counted in positions would be mostly
// spent on them. And the observation thread paces itself off the measured access rate
// instead of a fixed interval, because a cursor that moves further between two passes than
// the horizon reaches leaves the prefetcher permanently behind while appearing configured
// generously.
//
// The schedule is not declared, it is observed. An observation thread reads the access
// trace, finds its period, verifies the candidate over several repetitions before adopting
// it, and follows the cursor from then on. If the pattern changes, for instance because
// the program switches from training to generation, the verification fails, the schedule
// is dropped and the search starts again. Until a schedule exists the policy is least
// recently used, which is correct but not optimal, so the cost of a wrong guess is a
// slower step and never a wrong result.
//
// THE STORE IS A MAPPING, AND THAT IS THE POINT
//
// The store tier is one sparse file, mapped once at startup, with the blocks living at
// fixed offsets inside it. Two properties follow, and both matter more than they look.
//
// The first is that the store has no size to budget. Its pages are page cache: while there
// is host memory they stay resident and the tier runs at memory speed, and when there is
// not the kernel writes them back and reclaims them. So the same tier is a RAM cache on a
// machine with room and a disk overflow on a machine without, with no threshold to pick and
// no cliff to fall off. Explicit reads and writes cannot do this, because a buffer the
// program holds is memory the kernel may not reclaim.
//
// The second is that a block sitting in the store needs no copy to be read on the host.
// host() returns a pointer straight into the mapping, so loading a model writes through to
// the store rather than into a pinned buffer that must then be spilled, and reading one
// back costs nothing at all. This is what makes it possible to deserialize a model larger
// than host memory: the writes land in the mapping, the kernel decides how much of it to
// keep, and at no point does the whole thing have to exist in RAM.
//
// What a mapping cannot do is feed a DMA engine well. Its pages are pageable, so a transfer
// straight out of it degrades to a staged copy inside the driver, and a page fault in the
// middle of one stalls it. Transfers between the store and the device therefore go through
// a small pinned buffer, in chunks, with the host copy of one chunk overlapping the DMA of
// the one before it. That buffer is the only pinned allocation whose size does not follow
// the model.
//
// POINTER LIVENESS
//
// The one thing transparency cannot hide is that device() hands out a raw pointer, and a
// caller holds it across the kernel launch that uses it. A block whose pointer is live must
// not be evicted. Two mechanisms cover this.
//
// The first is automatic. The subsystem keeps the identifiers of the last hot_window
// blocks whose device pointer was handed out and refuses to evict any of them. A single
// tensor operation touches a bounded number of tensors, well under the default window, so
// in practice this is airtight for dlib's own layers and for anything written in their
// style. The second is explicit: device_scope pins a named set of tensors for as long as it
// is alive, for code that gathers many pointers before launching.
//
// Setting options.paranoid fills every released device block with a signalling NaN, which
// turns a violation of this rule into visible NaNs on the first step rather than into a
// silent numerical drift. It is the right setting for a first run on new code.
//
// ACTIVATION IS ONCE, AND FOR THE WHOLE PROGRAM
//
// enable_extended_memory() has to run before any tensor exists, and there is no way to turn
// the subsystem off again. Both restrictions are deliberate.
//
// A block decides where its host copy lives when it is sized, and that decision cannot be
// revisited cheaply: a network built before the subsystem started would keep pinned mirrors
// for the rest of the run and quietly lose most of the benefit. Refusing to start late is
// how that stays visible instead of becoming a mystery about memory that never drops.
//
// Switching off is worse. It means bringing every block back to VRAM at once, which is
// precisely the thing the subsystem exists because you cannot do, and converting every
// mapped mirror into a pinned one, which doubles the host footprint at the worst possible
// moment. A switch that only works when it is not needed is not a feature.
//
// So the call belongs in the first few lines of main, guarded by whatever condition the
// program uses to decide, and after that the answer to "is this on" is fixed for the run.
//
// WHAT HAPPENS WHEN THE PROGRAM ENDS
//
// The manager is deliberately leaked. A gpu_data belonging to a global object is destroyed
// during static destruction, and it must still find a manager to unregister with; deleting
// the manager first would trade a bounded leak at exit for a use-after-free.
//
// Everything that leak holds is reclaimed anyway, by the two parties that outlive the
// process. The driver tears down the CUDA context, which returns the device buffers and
// the pinned staging halves. The kernel unmaps the store and closes its descriptor, and
// since the file was unlinked the moment it was created on POSIX and opened with
// FILE_FLAG_DELETE_ON_CLOSE on Windows, its blocks go back to the volume at that point.
// None of this depends on the program unwinding: a store does not survive a crash any more
// than it survives a clean return.
//
// One thing is not reclaimed on its own, and that is the observation thread. Left running
// into static destruction it would keep issuing transfers after the CUDA runtime has torn
// itself down, and the failure that follows arrives in a thread with nobody to catch it.
// enable_extended_memory() therefore registers an atexit handler that joins it. Exit
// handlers run in reverse order of registration and this one is registered from main, so
// it runs before every static destructor and before the CUDA runtime's own teardown, which
// is the window in which stopping the thread is still safe.
//
// CONCURRENCY
//
// One invariant carries the thread safety of the whole design:
//
//     only a compute thread lowers the tier of a block, and it does so under the lock;
//     the observation thread only raises it, and only into headroom the budget already
//     allows.
//
// A block that is on the device therefore stays on the device for as long as the thread
// that put it there is looking at it, which is what lets the common case run without
// taking a lock at all: the accessor traces the access, stamps the block and returns. Only
// a block that is not on the device costs a lock.
//
// idle_release_ms is the single exception, and it is allowed because the condition that
// triggers it is the condition that makes it safe. Nothing has been accessed for the
// configured interval, so no kernel is in flight and the hot window has not moved; the
// blocks a stalled thread could still hold a pointer to are exactly the ones the window
// covers, and those are kept. What the sweep can reclaim is bounded by the window, and it
// finishes by trimming the pool, since a buffer sitting in the pool is still charged to the
// process as far as anything outside it can tell.
//
// The subsystem manages one device, the one current when it was enabled. Blocks allocated
// on any other device are left alone and behave as they do in stock dlib.
//
// WHAT IS AND IS NOT UNDER THE BUDGET
//
// Two kinds of device memory exist in dlib. Tensors go through gpu_data and are managed
// here. Workspaces go through cuda_data_void_ptr: cuDNN convolution scratch, cuSOLVER
// buffers, the temporaries the loss kernels use. Those have no contents worth tiering, but
// they are routed through the same pool anyway, for a reason that has nothing to do with
// tiering: left outside, a workspace allocation could fail with the subsystem sitting on
// gigabytes of weights it would gladly have evicted for it. They are counted against the
// budget and they can force an eviction, and that is all.
//
// What remains outside is what no library user can reach: the CUDA context, and whatever
// cuDNN, cuBLAS and cuSOLVER allocate inside their own handles. That is typically a few
// hundred megabytes, and it is why the budget defaults to a fraction of the free memory
// rather than to all of it. Comparing device_peak_bytes plus scratch_bytes against what
// nvidia-smi reports gives the size of that remainder on a given machine.

#ifndef DLIB_EXTENDED_MEMORY_H_
#define DLIB_EXTENDED_MEMORY_H_

#include "extended_memory_abstract.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class gpu_data;
    class tensor;

// ----------------------------------------------------------------------------------------

    struct extended_memory_options
    {
        std::size_t vram_budget          = 0;
        double      vram_budget_fraction = 0.90;
        std::string store_path;
        std::size_t store_bytes          = 0;
        std::size_t staging_bytes        = 64ul * 1024ul * 1024ul;
        std::size_t min_block_bytes      = 1ul * 1024ul * 1024ul;
        unsigned    hot_window           = 64;
        unsigned    lookahead            = 24;
        bool        fingerprint          = true;
        std::size_t fingerprint_min_bytes = 0;
        unsigned    advise_horizon       = 96;
        unsigned    idle_release_ms      = 0;
        double      idle_release_keep    = 0.5;
        unsigned    idle_purge_ms        = 0;
        bool        predictive           = true;
        bool        verbose              = false;
        bool        paranoid             = false;
    };

    struct extended_memory_stats
    {
        bool               enabled           = false;
        std::size_t        vram_budget       = 0;
        std::size_t        device_bytes      = 0;
        std::size_t        device_peak_bytes = 0;
        std::size_t        scratch_bytes     = 0;
        std::size_t        pinned_bytes      = 0;
        std::size_t        store_bytes       = 0;
        std::size_t        store_capacity    = 0;
        std::size_t        managed_blocks    = 0;
        std::size_t        evictions         = 0;
        std::size_t        restores          = 0;
        std::size_t        prefetch_hits     = 0;
        std::size_t        prefetch_issued   = 0;
        std::size_t        store_writes      = 0;
        std::size_t        store_skipped     = 0;
        std::size_t        store_unchanged   = 0;
        std::size_t        host_pulls        = 0;
        unsigned long long host_pull_bytes   = 0;
        std::size_t        largest_block     = 0;
        std::size_t        immovable_bytes   = 0;
        std::size_t        hash_threshold    = 0;
        std::size_t        hash_count        = 0;
        // Where the wall clock goes inside the subsystem, in seconds.
        double             sync_seconds      = 0;
        std::size_t        sync_count        = 0;
        // Split by thread: only what the compute thread waits for is on the critical path.
        double             wait_seconds      = 0;
        unsigned long long wait_bytes        = 0;
        double             ahead_seconds     = 0;
        unsigned long long ahead_bytes       = 0;
        double             hash_seconds      = 0;
        std::size_t        store_reads       = 0;
        std::size_t        pages_advised     = 0;
        unsigned long long h2d_bytes         = 0;
        unsigned long long d2h_bytes         = 0;
        bool               cycle_locked      = false;
        std::size_t        cycle_period      = 0;
        unsigned           prefetch_depth    = 0;
        double             access_rate       = 0;
        std::size_t        idle_releases     = 0;
        std::size_t        idle_purges       = 0;
        std::size_t        idle_released_bytes = 0;
    };

// ----------------------------------------------------------------------------------------

    /*
        A location for the store that does not have to be written into the program: the
        DLIB_XMEM_STORE environment variable when it is set, and the platform's temporary
        directory otherwise. Available whether or not this build uses CUDA.
    */
    std::string default_extended_memory_store_path ();

#ifdef DLIB_USE_CUDA

    bool enable_extended_memory (const extended_memory_options& opts = extended_memory_options());
    bool extended_memory_enabled ();

    extended_memory_stats get_extended_memory_stats ();
    void print_extended_memory_stats (std::ostream& out);
    void print_extended_memory_blocks (std::ostream& out, std::size_t max_rows = 12);

#else

    inline bool enable_extended_memory (const extended_memory_options& = extended_memory_options()) { return false; }
    inline bool extended_memory_enabled () { return false; }

    inline extended_memory_stats get_extended_memory_stats () { return extended_memory_stats(); }
    void print_extended_memory_stats (std::ostream& out);
    void print_extended_memory_blocks (std::ostream& out, std::size_t max_rows = 12);

#endif // DLIB_USE_CUDA

// ----------------------------------------------------------------------------------------

    namespace xmem
    {
        /*
            Internals shared between the manager and gpu_data. Nothing here is part of the
            public interface, but the record and the trace live in the header rather than in
            the implementation file so that the accessor fast path stays inline: a device()
            call on a resident block must not cost a cross-unit call.
        */

        enum tier : unsigned char
        {
            tier_device  = 0,   // data_device holds the block
            tier_host    = 1,   // data_device released, a pinned mirror holds the block
            tier_store   = 2,   // data_device released, the mapped arena holds the block
            tier_transit = 3    // a restore issued by the observation thread is in flight
        };

        class manager;

        struct block_record
        {
            gpu_data*                  owner       = nullptr;
            std::uint32_t              id          = 0;
            std::atomic<unsigned char> state       {tier_device};
            std::atomic<unsigned>      pins        {0};
            std::size_t                bytes       = 0;
            std::atomic<std::uint64_t> stamp       {0};
            // True when the arena slot holds the block's current content. Cleared on the
            // fast path by any accessor that lets the caller write to the device.
            std::atomic<bool>          store_valid {false};
            std::int64_t               slot        = -1;
            int                        device_id   = 0;
            bool                       evictable   = false;
            // Whether the slot has ever held this block, and how many evictions in a row
            // have found the device copy different from it. See device_to_store_locked().
            bool                       slot_written = false;
            unsigned char              store_diffs  = 0;
            /* Fingerprint of what the store holds for this block, taken on the device the
               last time it was written there. An eviction that recomputes it and finds it
               unchanged needs no transfer at all. */
            bool                       hash_valid   = false;
            unsigned long long         hash_xor     = 0;
            unsigned long long         hash_add     = 0;
        };

#ifdef DLIB_USE_CUDA

        /*
            State the accessor fast path reads on every call.

            These live as static members of a class template rather than as extern variables
            for two reasons. A template's statics are defined by the header itself, in every
            translation unit, and merged by the linker, so the inline accessors below do not
            depend on any particular source file having been compiled: a build that forgot to
            add extended_memory.cpp then fails on a dozen named functions instead of on
            several hundred references to data. And because each unit resolves them locally,
            the linker no longer has to write relocations into read-only text, which is what
            produced the DT_TEXTREL warning when they were plain externs.

            The activity flag is read on every accessor call and written once in the life of
            a process, so a relaxed load is both correct and free.
        */
        template <typename tag>
        struct globals
        {
            static std::atomic<bool>          active;
            static std::atomic<std::uint64_t> clock;
            static std::atomic<std::uint64_t> blocks_created;
            static std::atomic<std::uint64_t> trace_head;
            static std::uint32_t*             trace_ring;
            static std::uint64_t              trace_mask;
            static std::atomic<std::uint64_t> hot_head;
            static std::uint32_t*             hot_ring;
            static std::uint64_t              hot_mask;
        };

        template <typename tag> std::atomic<bool>          globals<tag>::active         {false};
        template <typename tag> std::atomic<std::uint64_t> globals<tag>::clock          {1};
        template <typename tag> std::atomic<std::uint64_t> globals<tag>::blocks_created {0};
        template <typename tag> std::atomic<std::uint64_t> globals<tag>::trace_head     {0};
        template <typename tag> std::uint32_t*             globals<tag>::trace_ring     = nullptr;
        template <typename tag> std::uint64_t              globals<tag>::trace_mask     = 0;
        template <typename tag> std::atomic<std::uint64_t> globals<tag>::hot_head       {0};
        template <typename tag> std::uint32_t*             globals<tag>::hot_ring       = nullptr;
        template <typename tag> std::uint64_t              globals<tag>::hot_mask       = 0;

        typedef globals<void> g;

        inline bool active () { return g::active.load(std::memory_order_relaxed); }

        /*
            Counts the blocks a process has sized. enable_extended_memory() refuses to start
            once this is non-zero, because a block already sized has already settled where
            its host copy lives.
        */
        inline void note_block_created () { g::blocks_created.fetch_add(1, std::memory_order_relaxed); }

        inline void trace_and_stamp (block_record* r)
        {
            r->stamp.store(g::clock.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);

            const std::uint64_t t = g::trace_head.fetch_add(1, std::memory_order_relaxed);
            g::trace_ring[t & g::trace_mask] = r->id;

            const std::uint64_t h = g::hot_head.fetch_add(1, std::memory_order_relaxed);
            g::hot_ring[h & g::hot_mask] = r->id;
        }

        // Out of line: everything below is either rare or already dominated by a transfer.
        block_record* register_block   (gpu_data* owner, std::size_t bytes, int device_id);
        void          unregister_block (block_record* r);
        void          retarget_block   (block_record* r, gpu_data* owner);
        void          restore_device   (block_record* r, bool need_content);
        void          before_host      (block_record* r, bool need_content, bool writes);
        void          pin_block        (block_record* r);
        void          unpin_block      (block_record* r);

        /*
            Held across gpu_data::swap. A swap moves the buffers and the records together,
            and between the two assignments a record's owner briefly names an object that no
            longer holds its memory. The observation thread reads that owner when a transfer
            it started comes back, so the two have to move under the same lock as everything
            else the manager owns.
        */
        class registry_guard
        {
        public:
            explicit registry_guard (bool engage);
            ~registry_guard ();
            registry_guard(const registry_guard&) = delete;
            registry_guard& operator=(const registry_guard&) = delete;
        private:
            bool held;
        };

        /*
            True when set_size() should leave the pinned host mirror unallocated because the
            block is large enough to be streamed and an arena is available to hold it. The
            block's host copy then lives in the mapping, which is what keeps pinned memory
            bounded by the staging buffer rather than by the size of the model.
        */
        bool          store_backed (std::size_t bytes);

        /*
            Device memory for something that is not a gpu_data block: cuDNN convolution
            workspaces, cuSOLVER scratch, the loss kernels' temporaries. The contents never
            have to survive a call, so these never move between tiers, but they do have to be
            counted and they do have to be able to make room. Returns null when the subsystem
            is not running, in which case the caller allocates as it always did.
        */
        std::shared_ptr<void> acquire_scratch (std::size_t bytes);

        /*
            The accessor fast path. A resident block costs a stamp, two ring writes and a
            relaxed load of the tier; anything else falls through to restore_device().
        */
        inline void before_device (block_record* r, bool need_content, bool writes)
        {
            trace_and_stamp(r);
            if (writes)
                r->store_valid.store(false, std::memory_order_relaxed);
            if (r->state.load(std::memory_order_acquire) != tier_device)
                restore_device(r, need_content);
        }

#else // DLIB_USE_CUDA

        inline bool active () { return false; }
        inline void note_block_created () {}
        inline void before_device (block_record*, bool, bool) {}
        inline void before_host (block_record*, bool, bool) {}
        inline void unregister_block (block_record*) {}
        inline void retarget_block (block_record*, gpu_data*) {}

        class registry_guard
        {
        public:
            explicit registry_guard (bool) {}
            ~registry_guard () {}
            registry_guard(const registry_guard&) = delete;
            registry_guard& operator=(const registry_guard&) = delete;
        };
        inline bool store_backed (std::size_t) { return false; }
        inline std::shared_ptr<void> acquire_scratch (std::size_t) { return std::shared_ptr<void>(); }

#endif // DLIB_USE_CUDA
    }

// ----------------------------------------------------------------------------------------

    class device_scope
    {
        /*!
            Pins a set of tensors on the device for the lifetime of the object. See the
            abstract header. Costs nothing when the subsystem is not running.
        !*/

    public:

        explicit device_scope (std::initializer_list<const tensor*> tensors);
        explicit device_scope (std::initializer_list<const gpu_data*> blocks);
        ~device_scope ();

        device_scope(const device_scope&) = delete;
        device_scope& operator=(const device_scope&) = delete;

    private:

        void take (const gpu_data* g);

        xmem::block_record* pinned[16];
        unsigned            count = 0;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EXTENDED_MEMORY_H_
