// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tensor_backend.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <dlib/cuda/extended_memory.h>
#include <dlib/cuda/gguf_blocks.h>
#include <dlib/cuda/kv_pages.h>
#include <dlib/cuda/tensor_tools.h>
#ifdef DLIB_USE_CUDA
#include <cuda_runtime_api.h>
#endif

/*
    The oldest cards the CUDA engine runs on, as ten times the major version of the compute
    capability plus its minor version, set by the build from the architectures of the kernels
    and the version of cuDNN; 0 accepts every card.
*/
#ifndef ALBIREO_CUDA_MIN_CC
#define ALBIREO_CUDA_MIN_CC 0
#endif

#include "../common/engine_api.h"
#include "../common/system_info.h"
#include "kv_cache.h"
#include "thread_pool.h"

namespace albireo
{
    namespace
    {
        typedef std::chrono::steady_clock clock_type;

        double seconds_since(clock_type::time_point t)
        {
            return std::chrono::duration<double>(clock_type::now() - t).count();
        }

        std::string megabytes(size_t bytes)
        {
            return std::to_string(bytes >> 20) + " MB";
        }

#if defined(DLIB_USE_CUDA) || defined(DLIB_USE_NPU)
#if defined(_WIN32)
        std::wstring widen(const std::string& s)
        {
            if (s.empty()) return std::wstring();
            const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<size_t>(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
            return w;
        }
#endif

        // Sets a variable of the environment, unless it is already set: the variables
        // of the user take precedence over the configuration.
        void default_environment(const char* name, const std::string& value)
        {
#if defined(_WIN32)
            const std::wstring wname = widen(name);
            if (_wgetenv(wname.c_str())) return;
            _wputenv_s(wname.c_str(), widen(value).c_str());
#else
            setenv(name, value.c_str(), 0);
#endif
        }

        void set_environment(const char* name, const std::string& value)
        {
#if defined(_WIN32)
            _wputenv_s(widen(name).c_str(), widen(value).c_str());
#else
            setenv(name, value.c_str(), 1);
#endif
        }

        std::string join(const std::string& dir, const std::string& name)
        {
            if (dir.empty()) return name;
            const char last = dir.back();
            if (last == '/' || last == '\\') return dir + name;
#if defined(_WIN32)
            return dir + "\\" + name;
#else
            return dir + "/" + name;
#endif
        }

        // Creates a folder and its parents; true when it exists afterwards.
        bool make_folder(const std::string& path)
        {
            if (path.empty()) return false;
            for (size_t i = 1; i <= path.size(); ++i)
            {
                if (i < path.size() && path[i] != '/' && path[i] != '\\') continue;
                const std::string part = path.substr(0, i);
                if (part.size() == 2 && part[1] == ':') continue;
#if defined(_WIN32)
                if (_wmkdir(widen(part).c_str()) != 0 && errno != EEXIST) return false;
#else
                if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) return false;
#endif
            }
            return true;
        }

#ifdef DLIB_USE_CUDA
        // Throws the error of a call to the runtime of CUDA, after what the call did.
        void cuda_check(cudaError_t e, const char* what)
        {
            if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
        }

        int capability_of(const cudaDeviceProp& p)
        {
            return 10 * p.major + p.minor;
        }

        std::string card_text(const cudaDeviceProp& p)
        {
            return std::string(p.name) + " (compute capability " + std::to_string(p.major) + "."
                + std::to_string(p.minor) + ")";
        }

        std::string too_old(const std::string& card)
        {
            return card + " predates the cards this engine was built for, of compute capability "
                + std::to_string(ALBIREO_CUDA_MIN_CC / 10) + "." + std::to_string(ALBIREO_CUDA_MIN_CC % 10)
                + " and later";
        }

        // The cards this engine runs on; the first older one is described in older.
        std::vector<int> usable_cards(std::string& older)
        {
            std::vector<int> cards;
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
            for (int i = 0; i < count; ++i)
            {
                cudaDeviceProp p;
                if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
                if (capability_of(p) >= ALBIREO_CUDA_MIN_CC) cards.push_back(i);
                else if (older.empty()) older = card_text(p);
            }
            return cards;
        }

        /*
            Host memory that the driver of CUDA keeps in place: the copies between the device and
            this memory run asynchronously, in the order of their stream, at the full rate of the
            bus.
        */
        class pinned_floats
        {
        public:
            pinned_floats() {}
            ~pinned_floats()
            {
                if (!p_) return;
                // A copy of the default stream may still read or write the buffer.
                cudaStreamSynchronize(0);
                cudaFreeHost(p_);
            }
            pinned_floats(const pinned_floats&) = delete;
            pinned_floats& operator=(const pinned_floats&) = delete;

            // Room for n floats, by half of the size held more at least.
            float* reserve(size_t n)
            {
                if (n <= size_) return p_;
                // The copies of the default stream that use the buffer end before it is freed.
                cuda_check(cudaStreamSynchronize(0), "synchronization with the device");
                if (p_) cudaFreeHost(p_);
                p_ = nullptr;
                const size_t grown = std::max(n, size_ + size_ / 2);
                size_ = 0;
                void* p = nullptr;
                cuda_check(cudaMallocHost(&p, sizeof(float) * grown), "allocation of pinned host memory");
                p_ = static_cast<float*>(p);
                size_ = grown;
                return p_;
            }

        private:
            float* p_ = nullptr;
            size_t size_ = 0;
        };
#endif
#endif

        // Number of tokens of a product, by classes whose costs differ.
        int token_class(long n)
        {
            if (n <= 1) return 0;
            if (n <= 4) return 1;
            if (n <= 16) return 2;
            if (n <= 64) return 3;
            if (n <= 256) return 4;
            return 5;
        }

        const char* token_class_name(int c)
        {
            static const char* names[] = { "1", "2-4", "5-16", "17-64", "65-256", "257+" };
            return names[std::max(0, std::min(5, c))];
        }

        // Averages the first samples, then follows the recent ones.
        void add_sample(double& average, int& count, double value)
        {
            count = std::min(count + 1, 8);
            average += (value - average) / count;
        }

        // A matrix larger than this never crosses the bus at each product: its copy is kept, or
        // the processor computes it.
        const size_t max_staged_bytes = size_t(512) << 20;

        /*
            The tokens of a product from which a matrix without a copy on the device crosses the
            bus for that product alone: below, the processor computes it from the quantized weights
            faster than the matrix crosses the bus.
        */
        const long staged_min_tokens = 128;

        // The bytes of a matrix in single precision that cross the bus at a time when its copy is made.
        const size_t upload_block_bytes = size_t(64) << 20;

        /*
            The tokens of a product from which the device expands a matrix kept in its GGUF format,
            a tile of rows at a time, and multiplies the tiles with cuBLAS: below, it reads the
            blocks in place, the weights of a row decoded once for up to eight tokens.
        */
        const long expanded_product_tokens = 64;

        // The bytes of a tile of a matrix expanded to single precision for a product.
        const size_t expanded_tile_bytes = size_t(64) << 20;

        // The pages of keys and values a first growth of the copies of a layer pass makes at least.
        const long min_kv_pages = 8;

        /*
            The rows of a new copy in a GGUF format that the device multiplies with probe
            activations, against the same products of the processor, before the copy serves: a
            format the device would decode wrongly then keeps its matrices in single precision.
        */
        const long verified_rows = 64;

        // The memory of the card the copies leave to the products, the staged matrices and cuBLAS.
        const size_t card_reserve_bytes = size_t(1) << 30;

        // Samples of each place before the first choice of a shape.
        const int samples_per_place = 2;

        // Thrown for a matrix whose copy does not fit and which is too large to expand at
        // each product: the processor computes its products.
        struct beyond_budget {};

        /*
            Failures of a shape for lack of device memory after which it stays on the processor,
            and the products of the shape it waits before the first new attempt, a number that
            doubles at each failure in a row.
        */
        const int max_memory_failures = 4;
        const unsigned long long memory_retry_calls = 256;

        /*
            Whether a failure of the device comes from a lack of memory, which the end of
            another program or the copies another model gives back may end: an allocation
            refused by the runtime of CUDA ("out of memory"), by cuBLAS ("could not be
            allocated") or by the extended memory ("cannot place a block").
        */
        bool lack_of_memory(const std::exception& e)
        {
            if (dynamic_cast<const std::bad_alloc*>(&e)) return true;
            const std::string what = e.what();
            for (const char* sign : { "out of memory", "could not be allocated", "cannot place a block" })
                if (what.find(sign) != std::string::npos) return true;
            return false;
        }

#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
        /*
            Whether the reason the OpenVINO runtime gives for using no device says only that
            none of the devices it looked for is present, as in "no device found among NPU,GPU
            (no NPU device; no GPU device)", rather than that a runtime or a device failed.
        */
        bool no_device_present(const std::string& why)
        {
            static const std::string head = "no device found among ";
            if (why.compare(0, head.size(), head) != 0) return false;
            const size_t open = why.find(" (");
            if (open == std::string::npos) return true;
            if (why.back() != ')') return false;
            const std::string list = why.substr(open + 2, why.size() - open - 3);
            for (size_t at = 0; at < list.size();)
            {
                size_t end = list.find("; ", at);
                if (end == std::string::npos) end = list.size();
                if (list.compare(at, 3, "no ") != 0) return false;
                at = end + 2;
            }
            return true;
        }
#endif

        // The name of a weight format of gguf_blocks.h, for the messages.
        std::string format_name(int type)
        {
            namespace gb = dlib::gguf_blocks;
            switch (type)
            {
            case gb::f32: return "F32";
            case gb::f16: return "F16";
            case gb::bf16: return "BF16";
            case gb::q4_0: return "Q4_0";
            case gb::q4_1: return "Q4_1";
            case gb::q5_0: return "Q5_0";
            case gb::q5_1: return "Q5_1";
            case gb::q8_0: return "Q8_0";
            case gb::iq4_nl: return "IQ4_NL";
            case gb::q4_K: return "Q4_K";
            case gb::q5_K: return "Q5_K";
            case gb::q6_K: return "Q6_K";
            default: return "the format " + std::to_string(type);
            }
        }

        long long now_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock_type::now().time_since_epoch()).count();
        }
    }

    class tensor_offload;
    class tensor_layer_pass;

    struct tensor_device::impl
    {
        impl(const json& c, thread_pool& p, log_function l) : config(c), pool(p), log(std::move(l)) {}

        json config;
        thread_pool& pool;
        log_function log;

        std::string backend;
        std::string name;
        bool v2ram = false;
        long min_tokens = 1;
        size_t budget = 0;
        // 1: every product goes to the device, -1: none does, 0: measured.
        int forced = 0;

        std::atomic<size_t> resident{0};
        /*
            The offloads of the loaded models.  A model that needs room for a copy takes
            it from the models without requests, the least recently used first.  The
            generation counts the events that may give room, a release of copies or a
            model that has no more requests; a matrix refused for lack of room is offered
            again after one of them.
        */
        std::mutex registry_m;
        std::vector<tensor_offload*> offloads;
        std::atomic<unsigned long long> generation{0};
        std::atomic<unsigned long long> evictions{0};
        std::atomic<unsigned long long> evicted_bytes{0};
        std::atomic<unsigned long long> device_products{0};
        std::atomic<unsigned long long> host_products{0};
        std::atomic<unsigned long long> device_us{0};
        std::atomic<unsigned long long> host_us{0};
        std::atomic<unsigned long long> failures{0};
        std::atomic<unsigned long long> layer_passes{0};
        std::atomic<unsigned long long> layer_pass_tokens{0};
        // The passes the device attended itself, and the bytes of the budget its copies of keys and values hold.
        std::atomic<unsigned long long> attention_passes{0};
        std::atomic<size_t> kv_page_bytes{0};
        std::atomic<bool> disabled{false};
        // The GGUF formats, as bits 1 << type, whose products on the device differed from those of
        // the processor, whose matrices are copied in single precision, and those that agreed.
        std::atomic<unsigned> faulty_formats{0};
        std::atomic<unsigned> verified_formats{0};
        // The card of the engine (CUDA), -1 for the other backends.
        int cuda_device = -1;
        // The bytes of copies the memory of the card holds; beyond, the extended memory moves them over the bus.
        size_t card_bytes = std::numeric_limits<size_t>::max();

        mutable std::mutex m;
        std::map<std::string, json> routes;   // the choices made, for the statistics

        void message(int level, const std::string& text) const
        {
            if (log) log(level, text);
        }

        /*
            Makes the card of the engine the current device of the calling thread, CUDA
            keeping one current device per thread; false when the runtime refuses it.
        */
        bool bind_thread() const
        {
#if defined(DLIB_USE_CUDA)
            static thread_local int bound = -1;
            if (cuda_device < 0 || bound == cuda_device) return true;
            try
            {
                dlib::cuda::set_device(cuda_device);
            }
            catch (const std::exception&)
            {
                return false;
            }
            bound = cuda_device;
#endif
            return true;
        }

        // Takes bytes of the budget of the copies; false when they do not fit.
        bool reserve(size_t bytes)
        {
            size_t current = resident.load();
            do
            {
                if (current + bytes > budget) return false;
            } while (!resident.compare_exchange_weak(current, current + bytes));
            return true;
        }

        void release(size_t bytes)
        {
            resident -= bytes;
            ++generation;
        }

        /*
            Frees copies of the models without requests, the least recently used model
            first, until bytes fit in the budget, and a sixteenth of the budget more so
            that the next matrices of the requester find room too; true when bytes fit.
            A model keeps the copies it does not have to give.
        */
        bool make_room(size_t bytes, const tensor_offload* requester);

        void note_route(const std::string& key, const json& value)
        {
            std::lock_guard<std::mutex> lock(m);
            routes[key] = value;
        }
    };

    class tensor_offload : public matmul_offload
    {
    public:
        tensor_offload(tensor_device::impl& d, const std::string& model) : d_(d), model_(model)
        {
            std::lock_guard<std::mutex> lock(d_.registry_m);
            d_.offloads.push_back(this);
        }

        ~tensor_offload() override
        {
            std::lock_guard<std::mutex> lock(d_.registry_m);
            d_.offloads.erase(std::remove(d_.offloads.begin(), d_.offloads.end(), this), d_.offloads.end());
            evict_copies(std::numeric_limits<size_t>::max());
        }

        void set_busy(bool busy) override
        {
            // A model whose requests are all served may give its room.
            if (busy_.exchange(busy) && !busy) ++d_.generation;
        }

        bool busy() const { return busy_.load(); }
        long long last_use() const { return last_use_.load(); }
        const std::string& model() const { return model_; }

        // Whether the copies this model gave since its last product were already
        // reported, so that a model giving its room in several steps is reported once.
        bool reported_eviction()
        {
            const long long use = last_use_.load();
            if (reported_use_ == use) return true;
            reported_use_ = use;
            return false;
        }

        size_t copy_bytes() const
        {
            size_t total = pass_room();
            for (const auto& kv : weights_) total += kv.second.bytes;
            return total;
        }

        /*
            Frees copies of this model until wanted bytes are freed or none is left,
            under the lock of the registry, and returns the bytes freed.  The forward
            passes of every model run on the thread of the scheduler of the engine, the
            one that asks for room: no product of this model runs meanwhile.  A freed
            copy is made again when the model needs the matrix.
        */
        size_t evict_copies(size_t wanted)
        {
            size_t freed = 0;
            for (auto it = weights_.begin(); it != weights_.end() && freed < wanted;)
            {
                freed += it->second.bytes;
                it = weights_.erase(it);
            }
            if (freed > 0) d_.release(freed);
            // The copies of the keys and values go last: the next pass of the model copies its blocks again.
            if (freed < wanted) freed += release_pass_pages();
            return freed;
        }

        void matmul(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            const std::function<void()>& host) override
        {
            last_use_ = now_ns();
            if (n < d_.min_tokens || d_.disabled.load(std::memory_order_relaxed) || !d_.bind_thread())
            {
                host_product(host);
                return;
            }
            /*
                Once the copies exceed the memory of the card, the extended memory brings part of
                them over the bus for each pass.  A product of a few tokens, as a generated token
                is, then waits for its matrix longer than the processor takes to compute it from
                the quantized weights, whatever the measures of the matrices the card holds say.
            */
            if (n < qk_float_product_tokens && d_.resident.load(std::memory_order_relaxed) > d_.card_bytes)
            {
                host_product(host);
                return;
            }
#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
            routed_by_dlib(W, X, n, x_stride, Y, y_stride, host);
#else
            routed_here(W, X, n, x_stride, Y, y_stride, host);
#endif
        }

        std::string description() const override
        {
            return d_.name;
        }

        std::unique_ptr<layer_pass> make_layer_pass(const dense_model& model) override;

    private:
        friend class tensor_layer_pass;

        // The bytes of the budget the pages of the layer pass hold, and their release (tensor_layer_pass).
        size_t pass_room() const;
        size_t release_pass_pages();

        struct copy
        {
            std::unique_ptr<dlib::resizable_tensor> t;
            size_t bytes = 0;
            // The blocks of the GGUF format of the matrix, which the kernels of dlib read in
            // place, or false for a copy in single precision.
            bool quantized = false;
        };

        struct route
        {
            double device = 0;
            double host = 0;
            int device_n = 0;
            int host_n = 0;
            bool decided = false;
            bool on_device = false;
            bool failed = false;
            /*
                No room for the copies of the shape, until the generation of the device moves or,
                after a lack of device memory, until the products of the shape reach retry_at.
            */
            bool no_room = false;
            unsigned long long room_generation = 0;
            unsigned long long retry_at = std::numeric_limits<unsigned long long>::max();
            // Failures for lack of device memory in a row.
            int memory_failures = 0;
            unsigned long long calls = 0;
            unsigned long long next_check = 0;
            unsigned long long interval = 64;

            // The matrices of the shape on which the device is measured, so that the
            // measures copy two matrices instead of all of them.
            const uint8_t* probes[2] = { nullptr, nullptr };

            bool known_probe(const uint8_t* p) const { return p == probes[0] || p == probes[1]; }

            bool probe(const uint8_t* p)
            {
                if (known_probe(p)) return true;
                for (auto& slot : probes)
                    if (!slot)
                    {
                        slot = p;
                        return true;
                    }
                return false;
            }
        };

        void host_product(const std::function<void()>& host)
        {
            const clock_type::time_point t0 = clock_type::now();
            host();
            d_.host_products++;
            d_.host_us += static_cast<unsigned long long>(seconds_since(t0) * 1e6);
        }

        void count_device(double seconds)
        {
            d_.device_products++;
            d_.device_us += static_cast<unsigned long long>(seconds * 1e6);
        }

        // Expands the rows [first, first + count) of W to single precision floats, with the threads of the engine.
        void expand_rows(const weight_matrix& W, long first, long count, float* dst)
        {
            const long cols = W.cols;
            const long rows_per_task = std::max<long>(1, 262144 / std::max<long>(1, cols));
            const long tasks = (count + rows_per_task - 1) / rows_per_task;
            d_.pool.parallel_for(tasks, [&](long task, int) {
                const long a = task * rows_per_task;
                const long b = std::min(count, a + rows_per_task);
                for (long r = a; r < b; ++r)
                    qk_dequantize_row(W.type, W.data + static_cast<size_t>(first + r) * W.row_bytes, dst + r * cols,
                        cols);
            });
        }

        // Expands the rows of W to single precision floats, with the threads of the engine.
        void expand(const weight_matrix& W, float* dst)
        {
            expand_rows(W, 0, W.rows, dst);
        }

#if defined(DLIB_USE_CUDA)
        /*
            Expands W into the memory of the card at dst, by blocks of rows of upload_block_bytes at
            most that a buffer of the host holds in turn: the copy has no image in the memory of the
            host, and the extended memory gives the block of its tensor a place on the card alone.
        */
        void upload_expanded(const weight_matrix& W, float* dst)
        {
            const long cols = W.cols;
            const size_t row_bytes = sizeof(float) * static_cast<size_t>(std::max<long>(1, cols));
            const long block = std::max<long>(1,
                std::min<long>(W.rows, static_cast<long>(upload_block_bytes / row_bytes)));
            const size_t values = static_cast<size_t>(block) * static_cast<size_t>(cols);
            if (expanded_.size() < values) expanded_.resize(values);
            for (long first = 0; first < W.rows; first += block)
            {
                const long count = std::min(block, W.rows - first);
                expand_rows(W, first, count, expanded_.data());
                cuda_check(cudaMemcpy(dst + static_cast<size_t>(first) * cols, expanded_.data(),
                    sizeof(float) * static_cast<size_t>(count) * cols, cudaMemcpyHostToDevice),
                    "copy of a matrix to the device");
            }
            // The buffer of a large matrix goes back to the system.
            if (expanded_.size() * sizeof(float) > upload_block_bytes) std::vector<float>().swap(expanded_);
        }
#endif

        /*
            Whether the device keeps W in its GGUF format: a format of gguf_blocks.h, in rows of
            whole blocks, that the device was not found to decode wrongly.  The routing of dlib
            (OpenVINO) multiplies matrices of floats only.
        */
        bool quantizable(const weight_matrix& W) const
        {
#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
            (void)W;
            return false;
#else
            const int type = static_cast<int>(W.type);
            return dlib::gguf_blocks::known_format(type) && type < 32
                && dlib::gguf_blocks::row_bytes(type, W.cols) == W.row_bytes
                && (d_.faulty_formats.load() & (1u << type)) == 0;
#endif
        }

        // Keeps the matrices of a GGUF format that the device decodes wrongly in single precision.
        void faulty_format(int type)
        {
            if ((d_.faulty_formats.fetch_or(1u << type) & (1u << type)) == 0)
                d_.message(ALBIREO_LOG_WARNING, "device: the products of the weights in " + format_name(type)
                    + " on the device differ from those of the processor; the device keeps the matrices of this "
                    "format in single precision");
        }

        // The bytes the device holds W in: its blocks, a whole number of floats, or single precision.
        static size_t copy_size(const weight_matrix& W, bool quantized)
        {
            if (quantized) return (static_cast<size_t>(W.rows) * W.row_bytes + sizeof(float) - 1) / sizeof(float)
                * sizeof(float);
            return static_cast<size_t>(W.rows) * static_cast<size_t>(W.cols) * sizeof(float);
        }

        // Writes the blocks of W, as the file holds them, at the start of t.
        static void upload_blocks(const weight_matrix& W, dlib::tensor& t)
        {
            const size_t total = static_cast<size_t>(W.rows) * W.row_bytes;
#if defined(DLIB_USE_CUDA)
            cuda_check(cudaMemcpy(t.device_write_only(), W.data, total, cudaMemcpyHostToDevice),
                "copy of a matrix to the device");
#else
            std::memcpy(t.host_write_only(), W.data, total);
#endif
        }

        /*
            Multiplies the first rows of a copy of W in its GGUF format with probe activations on
            the device and on the processor, and tells whether they agree: a format the device
            decodes wrongly shows in every row.
        */
        static bool verify_copy(const weight_matrix& W, const dlib::tensor& t)
        {
            const long rows = std::min(W.rows, verified_rows);
            const long n = 3;
            dlib::resizable_tensor x(n, W.cols), y(n, rows);
            float* xp = x.host_write_only();
            uint32_t s = 2463534242u;
            for (size_t i = 0; i < x.size(); ++i)
            {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                xp[i] = static_cast<float>(s >> 8) * (2.0f / 16777216.0f) - 1.0f;
            }
            dlib::tt::gguf_matvec(y, x, t, static_cast<int>(W.type), rows, W.cols, W.row_bytes);
            // The expansion of the same rows, which the products of 64 tokens and more read.
            dlib::resizable_tensor expanded(rows, W.cols);
            dlib::tt::gguf_dequantize(expanded, t, static_cast<int>(W.type), 0, rows, W.cols, W.row_bytes);
            const float* got = y.host();
            const float* values = expanded.host();
            const float* probe = x.host();
            std::vector<float> row(static_cast<size_t>(W.cols));
            for (long r = 0; r < rows; ++r)
            {
                qk_dequantize_row(W.type, W.data + static_cast<size_t>(r) * W.row_bytes, row.data(), W.cols);
                for (long i = 0; i < W.cols; ++i)
                {
                    const float a = row[static_cast<size_t>(i)];
                    const float b = values[r * W.cols + i];
                    if (!(std::fabs(b - a) <= 1e-5f * std::max(1.0f, std::fabs(a)))) return false;
                }
                for (long j = 0; j < n; ++j)
                {
                    double sum = 0, magnitude = 0;
                    for (long i = 0; i < W.cols; ++i)
                    {
                        const double term = static_cast<double>(row[static_cast<size_t>(i)]) * probe[j * W.cols + i];
                        sum += term;
                        magnitude += std::fabs(term);
                    }
                    // Written so that a value that is not a number fails.
                    if (!(std::fabs(got[j * rows + r] - sum) <= 1e-3 * magnitude + 1e-6)) return false;
                }
            }
            return true;
        }

        // The copy of W, made when it fits in the budget; null otherwise.  created tells
        // whether this call made it.
        const copy* resident_copy(const weight_matrix& W, bool& created)
        {
            created = false;
            auto it = weights_.find(W.data);
            if (it != weights_.end()) return &it->second;
            const unsigned long long generation = d_.generation.load();
            auto refused = refused_.find(W.data);
            if (refused != refused_.end())
            {
                if (refused->second == generation) return nullptr;
                refused_.erase(refused);
            }
            const bool quantized = quantizable(W);
            const size_t bytes = copy_size(W, quantized);
            if (!d_.reserve(bytes) && !(d_.make_room(bytes, this) && d_.reserve(bytes)))
            {
                refused_[W.data] = d_.generation.load();
                return nullptr;
            }
            copy c;
            c.bytes = bytes;
            c.quantized = quantized;
            bool same = true;
            try
            {
                c.t.reset(new dlib::resizable_tensor());
                if (quantized)
                {
                    c.t->set_size(static_cast<long long>(bytes / sizeof(float)));
                    upload_blocks(W, *c.t);
                    same = verify_copy(W, *c.t);
                }
                else
                {
                    c.t->set_size(W.rows, W.cols);
#if defined(DLIB_USE_CUDA)
                    upload_expanded(W, c.t->device_write_only());
#else
                    expand(W, c.t->host_write_only());
#endif
                }
            }
            catch (...)
            {
                d_.release(bytes);
                refused_[W.data] = d_.generation.load();
                throw;
            }
            if (!same)
            {
                // The matrices of this format, this one included, are copied in single precision.
                faulty_format(static_cast<int>(W.type));
                c.t.reset();
                d_.release(bytes);
                return resident_copy(W, created);
            }
            if (quantized) d_.verified_formats |= 1u << static_cast<int>(W.type);
            copy& stored = weights_[W.data];
            stored = std::move(c);
            created = true;
            return &stored;
        }

        /*
            Gives b room for values floats, by half of its size more at least, so that products
            of growing sizes reallocate it a few times only, but beyond the size asked by no more
            than the largest matrix expanded at each product.
        */
        static void reserve_buffer(dlib::resizable_tensor& b, size_t values)
        {
            if (b.size() >= values) return;
            const size_t limit = std::max(values, max_staged_bytes / sizeof(float));
            const size_t grown = std::min(limit, std::max(values, b.size() + b.size() / 2));
            try
            {
                b.set_size(static_cast<long long>(grown));
            }
            catch (...)
            {
                // A tensor whose memory was refused keeps the size asked: it is emptied, so that
                // the next product asks again instead of writing through no memory.
                b.clear();
                throw;
            }
        }

        /*
            y[t][first + i] = part[t][i] on the device, for t < n and i < count, the rows of y
            holding width floats.
        */
        static void place_columns(const dlib::tensor& part, dlib::tensor& y, long n, long width, long first,
            long count)
        {
#if defined(DLIB_USE_CUDA)
            cuda_check(cudaMemcpy2DAsync(y.device() + first, sizeof(float) * width, part.device(),
                sizeof(float) * count, sizeof(float) * count, static_cast<size_t>(n), cudaMemcpyDeviceToDevice, 0),
                "copy of products on the device");
#else
            const float* src = part.host();
            float* dst = y.host();
            for (long t = 0; t < n; ++t) std::memcpy(dst + t * width + first, src + t * count, sizeof(float) * count);
#endif
        }

        /*
            y = x W^T on the device, x holding n rows of W.cols floats and y n rows of W.rows
            floats, w being the weights of W in their GGUF format (quantized) or in single
            precision.  A product of fewer than expanded_product_tokens tokens reads the blocks
            in place; a larger one expands them by tiles of rows that cuBLAS multiplies, as it
            does the weights in single precision.
        */
        void product_on_device(const weight_matrix& W, const dlib::tensor& w, bool quantized, const dlib::tensor& x,
            dlib::tensor& y)
        {
            if (!quantized)
            {
                dlib::tt::gemm(0, y, 1, x, false, w, true);
                return;
            }
            const int type = static_cast<int>(W.type);
            const long n = x.num_samples();
            if (n < expanded_product_tokens)
            {
                dlib::tt::gguf_matvec(y, x, w, type, W.rows, W.cols, W.row_bytes);
                return;
            }
            const long tile = tile_rows(W);
            reserve_buffer(tile_, static_cast<size_t>(tile) * static_cast<size_t>(W.cols));
            if (tile < W.rows) reserve_buffer(tile_out_, static_cast<size_t>(n) * static_cast<size_t>(tile));
            for (long first = 0; first < W.rows; first += tile)
            {
                const long count = std::min(tile, W.rows - first);
                dlib::alias_tensor wa(count, W.cols);
                dlib::alias_tensor_instance expanded = wa(tile_);
                dlib::tt::gguf_dequantize(expanded, w, type, first, count, W.cols, W.row_bytes);
                if (count == W.rows)
                {
                    dlib::tt::gemm(0, y, 1, x, false, expanded, true);
                    return;
                }
                dlib::alias_tensor pa(n, count);
                dlib::alias_tensor_instance part = pa(tile_out_);
                dlib::tt::gemm(0, part, 1, x, false, expanded, true);
                place_columns(part, y, n, W.rows, first, count);
            }
        }

        // The rows of W that a tile of its expansion to single precision holds.
        static long tile_rows(const weight_matrix& W)
        {
            const size_t row = sizeof(float) * static_cast<size_t>(std::max<long>(1, W.cols));
            return std::max<long>(1, std::min<long>(W.rows, static_cast<long>(expanded_tile_bytes / row)));
        }

        // Copies n rows of cols floats of the host, x_stride floats apart, to the view x of the device.
        static void rows_to_device(const float* X, long n, long cols, long x_stride, dlib::tensor& x)
        {
#if defined(DLIB_USE_CUDA)
            cuda_check(cudaMemcpy2D(x.device(), sizeof(float) * cols, X, sizeof(float) * x_stride, sizeof(float) * cols,
                static_cast<size_t>(n), cudaMemcpyHostToDevice), "copy of the activations to the device");
#else
            float* xp = x.host();
            if (x_stride == cols)
                std::memcpy(xp, X, sizeof(float) * static_cast<size_t>(n) * cols);
            else
                for (long t = 0; t < n; ++t) std::memcpy(xp + t * cols, X + t * x_stride, sizeof(float) * cols);
#endif
        }

        // Copies the n rows of rows floats of the view y of the device to the host, y_stride floats apart.
        static void rows_to_host(const dlib::tensor& y, long n, long rows, float* Y, long y_stride)
        {
#if defined(DLIB_USE_CUDA)
            cuda_check(cudaMemcpy2D(Y, sizeof(float) * y_stride, y.device(), sizeof(float) * rows, sizeof(float) * rows,
                static_cast<size_t>(n), cudaMemcpyDeviceToHost), "copy of the products from the device");
#else
            const float* yp = y.host();
            if (y_stride == rows)
                std::memcpy(Y, yp, sizeof(float) * static_cast<size_t>(n) * rows);
            else
                for (long t = 0; t < n; ++t) std::memcpy(Y + t * y_stride, yp + t * rows, sizeof(float) * rows);
#endif
        }

        /*
            Y = X W^T with the device, w being the weights of W in their GGUF format (quantized) or
            in single precision.

            The staging tensors keep the size of the largest product, and dlib moves whole blocks:
            through the views of a product, each one would cross the bus twice, in both directions,
            as the host or the device writes it.  The rows of the product are written and read here
            through the device pointers of the views instead, and nothing else crosses the bus.  The
            copies and the product share the default stream of CUDA, which orders them.
        */
        void device_gemm(const weight_matrix& W, const dlib::tensor& w, bool quantized, const float* X, long n,
            long x_stride, float* Y, long y_stride)
        {
            reserve_buffer(x_, static_cast<size_t>(n) * W.cols);
            reserve_buffer(y_, static_cast<size_t>(n) * W.rows);
            dlib::alias_tensor xa(n, W.cols), ya(n, W.rows);
            dlib::alias_tensor_instance x = xa(x_);
            dlib::alias_tensor_instance y = ya(y_);
            rows_to_device(X, n, W.cols, x_stride, x);
            product_on_device(W, w, quantized, x, y);
            rows_to_host(y, n, W.rows, Y, y_stride);
        }

        // The outcome of device_product().
        enum class placed
        {
            steady,     // computed on the device, in a time that reflects the products to come
            created,    // computed on the device after the copy of the matrix was made
            declined    // left to the processor: a matrix without a copy, for a product of few tokens
        };

        /*
            The product on the device.  A matrix without a copy, whose copy does not fit, crosses
            the bus for this product alone when it has staged_min_tokens tokens at least, and is
            declined otherwise.  Throws when the matrix cannot go to the device at all.
        */
        placed device_product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y,
            long y_stride)
        {
            bool created = false;
            const copy* c = resident_copy(W, created);
            if (c)
            {
                device_gemm(W, *c->t, c->quantized, X, n, x_stride, Y, y_stride);
                return created ? placed::created : placed::steady;
            }
            if (n < staged_min_tokens) return placed::declined;
            // The matrix alone crosses the bus, in its GGUF format when the device reads it, as the
            // rows of the products do (device_gemm()).
            bool quantized = quantizable(W);
            if (copy_size(W, quantized) > max_staged_bytes) throw beyond_budget();
            if (quantized)
            {
                reserve_buffer(w_, copy_size(W, true) / sizeof(float));
                upload_blocks(W, w_);
                const int type = static_cast<int>(W.type);
                if ((d_.verified_formats.load() & (1u << type)) == 0)
                {
                    if (verify_copy(W, w_))
                        d_.verified_formats |= 1u << type;
                    else
                    {
                        faulty_format(type);
                        quantized = false;
                    }
                }
                if (quantized)
                {
                    device_gemm(W, w_, true, X, n, x_stride, Y, y_stride);
                    return placed::steady;
                }
            }
            const size_t values = static_cast<size_t>(W.rows) * static_cast<size_t>(W.cols);
            if (values * sizeof(float) > max_staged_bytes) throw beyond_budget();
            reserve_buffer(w_, values);
            dlib::alias_tensor wa(W.rows, W.cols);
            dlib::alias_tensor_instance staged = wa(w_);
#if defined(DLIB_USE_CUDA)
            if (expanded_.size() < values) expanded_.resize(values);
            expand(W, expanded_.data());
            cuda_check(cudaMemcpy(w_.device_write_only(), expanded_.data(), sizeof(float) * values,
                cudaMemcpyHostToDevice), "copy of a matrix to the device");
            // The buffer of a large matrix goes back to the system, as after a copy.
            if (expanded_.size() * sizeof(float) > upload_block_bytes) std::vector<float>().swap(expanded_);
#else
            expand(W, staged.host());
#endif
            device_gemm(W, staged, false, X, n, x_stride, Y, y_stride);
            return placed::steady;
        }

        void decide(route& r, const std::tuple<long, long, int>& key)
        {
            r.decided = true;
            r.on_device = r.device < r.host;
            r.next_check = r.calls + r.interval;
            json j = json::object();
            j["rows"] = std::get<0>(key);
            j["cols"] = std::get<1>(key);
            j["tokens"] = token_class_name(std::get<2>(key));
            j["device_ms"] = r.device * 1e3;
            j["host_ms"] = r.host * 1e3;
            j["on_device"] = r.on_device;
            d_.note_route(std::to_string(std::get<0>(key)) + "x" + std::to_string(std::get<1>(key)) + "/"
                + token_class_name(std::get<2>(key)), j);
        }

        /*
            Products routed by this object (CUDA, and the host code of dlib).  Each shape is
            measured on both places: the device on two of its matrices, the processor on the
            others, or on the same ones once the device has its samples.  The shape then runs
            where it was faster, and the choice is checked again after 64 products of the
            shape, then after twice as many products at each check, up to every 65536.
        */
        void routed_here(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            const std::function<void()>& host)
        {
            const std::tuple<long, long, int> key(W.rows, W.cols, token_class(n));
            route& r = routes_[key];
            ++r.calls;
            if (r.no_room && (r.room_generation != d_.generation.load() || r.calls >= r.retry_at)) r.no_room = false;
            bool on_device = false;
            bool check = false;
            if (r.failed || r.no_room || d_.forced < 0)
                on_device = false;
            else if (d_.forced > 0)
                on_device = true;
            else if (!r.decided)
                on_device = r.probe(W.data) && !(r.device_n >= samples_per_place && r.host_n < samples_per_place);
            else if (r.calls >= r.next_check && (r.on_device || r.known_probe(W.data)))
            {
                // A check of the device waits for a matrix it already holds.
                on_device = !r.on_device;
                check = true;
            }
            else
                on_device = r.on_device;

            double seconds = 0;
            bool steady = true;
            if (on_device)
            {
                const clock_type::time_point t0 = clock_type::now();
                try
                {
                    const placed p = device_product(W, X, n, x_stride, Y, y_stride);
                    if (p == placed::declined)
                    {
                        // A matrix the card does not hold, for a few tokens: the processor computes
                        // it, and the measures of the shape, which the matrices it holds give, stay.
                        host_product(host);
                        return;
                    }
                    steady = p == placed::steady;
                    r.memory_failures = 0;
                }
                catch (const beyond_budget&)
                {
                    // No room for now: the processor computes the shape until copies leave
                    // the device or a model has no more requests.
                    r.no_room = true;
                    r.room_generation = d_.generation.load();
                    r.retry_at = std::numeric_limits<unsigned long long>::max();
                    host_product(host);
                    return;
                }
                catch (const std::exception& e)
                {
                    if (lack_of_memory(e) && r.memory_failures < max_memory_failures)
                        memory_failure(r, W, e.what());
                    else
                        device_failure(r, W, e.what());
                    host_product(host);
                    return;
                }
                seconds = seconds_since(t0);
                count_device(seconds);
            }
            else
            {
                const clock_type::time_point t0 = clock_type::now();
                host();
                seconds = seconds_since(t0);
                d_.host_products++;
                d_.host_us += static_cast<unsigned long long>(seconds * 1e6);
            }

            if (r.failed || r.no_room || d_.forced != 0) return;
            if (!r.decided)
            {
                if (!on_device) add_sample(r.host, r.host_n, seconds);
                else if (steady) add_sample(r.device, r.device_n, seconds);
                if (r.device_n >= samples_per_place && r.host_n >= samples_per_place) decide(r, key);
            }
            else if (check)
            {
                if (on_device && !steady)
                {
                    r.next_check = r.calls + 1;
                    return;
                }
                const double current = r.on_device ? r.device : r.host;
                if (seconds < 0.9 * current)
                {
                    // The other place looks faster now: both are measured again.
                    r.decided = false;
                    r.device_n = r.host_n = 0;
                    r.device = r.host = 0;
                }
                else
                {
                    if (on_device) add_sample(r.device, r.device_n, seconds);
                    else add_sample(r.host, r.host_n, seconds);
                    r.interval = std::min<unsigned long long>(r.interval * 2, 65536);
                    r.next_check = r.calls + r.interval;
                }
            }
            else if (!on_device || steady)
            {
                if (on_device) add_sample(r.device, r.device_n, seconds);
                else add_sample(r.host, r.host_n, seconds);
            }
        }

        /*
            A product the device could not place for lack of memory: the processor computes the
            shape until copies leave the device, a model has no more requests, or the products
            of the shape reach a number that doubles at each such failure in a row.
        */
        void memory_failure(route& r, const weight_matrix& W, const std::string& what)
        {
            ++r.memory_failures;
            r.no_room = true;
            r.room_generation = d_.generation.load();
            r.retry_at = r.calls + (memory_retry_calls << r.memory_failures);
            d_.message(ALBIREO_LOG_INFO, "device: no memory for a product of " + std::to_string(W.rows) + "x"
                + std::to_string(W.cols) + ", the processor computes it for now (" + what + ")");
        }

        void device_failure(route& r, const weight_matrix& W, const std::string& what)
        {
            r.failed = true;
            const unsigned long long n = ++d_.failures;
            d_.message(ALBIREO_LOG_WARNING, "device: a product of " + std::to_string(W.rows) + "x"
                + std::to_string(W.cols) + " stays on the processor (" + what + ")");
            if (n >= 8 && !d_.disabled.exchange(true))
                d_.message(ALBIREO_LOG_ERROR, "device: too many failures, every product stays on the processor");
        }

#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
        /*
            Products routed by dlib: the routing of the NPU module receives each product,
            computes it on a device when it measured one faster for its shape, and otherwise
            times the processor kernels of the engine, which then run under a cpu_measure.
            Only the matrices whose copy fits in the budget are offered.
        */
        void routed_by_dlib(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            const std::function<void()>& host)
        {
            const copy* c = nullptr;
            try
            {
                bool created = false;
                c = resident_copy(W, created);
            }
            catch (const std::exception& e)
            {
                d_.message(ALBIREO_LOG_WARNING, std::string("device: no copy of a matrix (") + e.what() + ")");
            }
            if (!c)
            {
                host_product(host);
                return;
            }
            const clock_type::time_point t0 = clock_type::now();
            reserve_buffer(x_, static_cast<size_t>(n) * W.cols);
            reserve_buffer(y_, static_cast<size_t>(n) * W.rows);
            dlib::alias_tensor xa(n, W.cols), ya(n, W.rows);
            dlib::alias_tensor_instance x = xa(x_);
            dlib::alias_tensor_instance y = ya(y_);
            float* xp = x.host_write_only();
            if (x_stride == W.cols)
                std::memcpy(xp, X, sizeof(float) * static_cast<size_t>(n) * W.cols);
            else
                for (long t = 0; t < n; ++t) std::memcpy(xp + t * W.cols, X + t * x_stride, sizeof(float) * W.cols);
            if (dlib::npu::gemm(0, y, 1, x, false, *c->t, true, dlib::operation_mode::CHANNEL_WISE))
            {
                rows_to_host(y, n, W.rows, Y, y_stride);
                count_device(seconds_since(t0));
                return;
            }
            dlib::npu::cpu_measure measure;
            host_product(host);
        }
#endif

        tensor_device::impl& d_;
        std::string model_;
        std::atomic<bool> busy_{false};
        std::atomic<long long> last_use_{0};
        long long reported_use_ = -1;
        std::unordered_map<const uint8_t*, copy> weights_;
        // The matrices refused for lack of room, with the generation of the refusal.
        std::unordered_map<const uint8_t*, unsigned long long> refused_;
        std::map<std::tuple<long, long, int>, route> routes_;
        dlib::resizable_tensor x_, y_, w_;
        // A tile of a matrix expanded to single precision on the device, and its products.
        dlib::resizable_tensor tile_, tile_out_;
        // The matrix expanded on the host before it goes to w_ (CUDA).
        std::vector<float> expanded_;
        // The layer pass of the model, which keeps copies of its keys and values.
        tensor_layer_pass* pass_ = nullptr;
    };

    /*
        The layers of a dense decoder on the device of a tensor_offload (layer_pass).  The hidden
        states of the tokens of a pass stay in a tensor of the device from the first layer to the
        last, and each layer exchanges with the processor the queries, keys and values of the
        tokens and the outputs of their attention, through pinned host memory that the copies
        read and write asynchronously, in the order of the default stream of CUDA: a layer waits
        for the device once, when its queries, keys and values arrive.  The matrices are the
        copies of the offload, which the first pass makes and which stay while the model has
        requests; the norms are copied to the device at the first pass.

        The device also attends the passes the runtime gives it (begin_attention()).  It keeps
        copies of the blocks of keys and values of the pool of the runtime in pages, a page per
        block in the layout of the pool (dlib::kv_page_layout), and knows of each copy the epoch
        of its block and the rows it holds.  Before a pass, a block the pass reads whose copy
        lacks rows crosses whole from the pool; the layers of the pass then run without an
        exchange with the processor, their keys and values going into the pages; at the end of
        the pass, the rows of its tokens cross to the host in one copy and go into the blocks of
        the pool.  The pages take room in the budget of the copies and grow as the passes need
        them, the page of the block least recently read going to another block; an idle model
        gives them back with its matrices.
    */
    class tensor_layer_pass : public layer_pass
    {
    public:
        tensor_layer_pass(tensor_offload& o, const dense_model& m) : o_(o), m_(m)
        {
            const long hd = m.head_dim;
            attends_ = m.attention && m.n_heads > 0 && m.n_kv_heads > 0 && m.n_heads % m.n_kv_heads == 0
                && (hd == 32 || hd == 64 || hd == 96 || hd == 128 || hd == 256)
                && m.q_dim == m.n_heads * hd && m.kv_dim == m.n_kv_heads * hd
                && m.rope_dims > 0 && m.rope_dims <= hd && m.rope_dims % 2 == 0;
            // make_room() reaches the pages of an idle model through its offload, under this lock.
            std::lock_guard<std::mutex> lock(o_.d_.registry_m);
            o_.pass_ = this;
        }

        ~tensor_layer_pass() override
        {
            std::lock_guard<std::mutex> lock(o_.d_.registry_m);
            o_.pass_ = nullptr;
            // The pages go with their tensor; their room returns to the budget.
            if (reserved_ > 0) release_room();
        }

        bool begin(const float* x, long n) override
        {
            auto& d = o_.d_;
            o_.last_use_ = now_ns();
            attending_ = false;
            if (n <= 0 || n < d.min_tokens || d.disabled.load() || !d.bind_thread()) return false;
            // Once the copies exceed the card, a generated token waits for the extended memory.
            if (n < qk_float_product_tokens && d.resident.load(std::memory_order_relaxed) > d.card_bytes) return false;
            try
            {
                if (!gather_copies()) return false;
                if (norms_.empty()) copy_norms();
                const long d_model = m_.d_model;
                const size_t rows = static_cast<size_t>(n);
                tensor_offload::reserve_buffer(x_, rows * d_model);
                tensor_offload::reserve_buffer(xn_, rows * d_model);
                tensor_offload::reserve_buffer(tmp_, rows * d_model);
                tensor_offload::reserve_buffer(qkv_, rows * (m_.q_dim + 2 * m_.kv_dim));
                tensor_offload::reserve_buffer(ctx_, rows * m_.q_dim);
                tensor_offload::reserve_buffer(gate_, rows * m_.d_ffn);
                tensor_offload::reserve_buffer(up_, rows * m_.d_ffn);
                // The tiles of the expansions of the larger passes, so that no layer allocates.
                if (n >= expanded_product_tokens)
                {
                    long tile = 1;
                    size_t tile_values = 0;
                    for (const dense_layer& L : m_.layers)
                        for (const weight_matrix* W : { &L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down })
                        {
                            const long t = tensor_offload::tile_rows(*W);
                            if (t < W->rows) tile = std::max(tile, t);
                            tile_values = std::max(tile_values, static_cast<size_t>(t) * W->cols);
                        }
                    tensor_offload::reserve_buffer(o_.tile_, tile_values);
                    if (tile > 1) tensor_offload::reserve_buffer(o_.tile_out_, rows * tile);
                }
                n_ = n;
                dlib::device_scope pin({ &x_ });
                dlib::alias_tensor_instance states = view(x_, n_, d_model);
                upload(states, x, rows * d_model);
            }
            catch (const std::exception& e)
            {
                settle();
                // No room on the device for now: the processor computes the pass.
                if (lack_of_memory(e)) return false;
                throw device_failure(e.what(), false);
            }
            ++d.layer_passes;
            d.layer_pass_tokens += static_cast<unsigned long long>(n);
            return true;
        }

        void attention_inputs(long layer, float* q, float* k, float* v) override
        {
            guarded([&]() {
                const long qd = m_.q_dim, kvd = m_.kv_dim;
                dlib::device_scope pin({ &x_, &xn_, &qkv_, &o_.tile_, &o_.tile_out_ });
                project(layer);
                const size_t qn = static_cast<size_t>(n_ * qd), kn = static_cast<size_t>(n_ * kvd);
                const float* all = download(view(qkv_, n_, qd + 2 * kvd), qn + 2 * kn);
                std::memcpy(q, all, sizeof(float) * qn);
                std::memcpy(k, all + qn, sizeof(float) * kn);
                std::memcpy(v, all + qn + kn, sizeof(float) * kn);
            });
        }

        void attention_output(long layer, const float* ctx) override
        {
            guarded([&]() {
                dlib::device_scope pin({ &x_, &xn_, &ctx_, &tmp_, &gate_, &up_, &o_.tile_, &o_.tile_out_ });
                dlib::alias_tensor_instance attended = view(ctx_, n_, m_.q_dim);
                upload(attended, ctx, static_cast<size_t>(n_ * m_.q_dim));
                finish(layer);
            });
        }

        bool attends() const override
        {
            return attends_ && allowed_;
        }

        void allow_attention(bool on) override
        {
            allowed_ = on;
            // The copies of the keys and values serve the attention of the device alone.
            if (!on) release_pages();
        }

        bool begin_attention(const attention_plan& plan) override
        {
            attending_ = false;
            if (!attends() || n_ <= 0 || !plan.pool) return false;
            const kv_pool& pool = *plan.pool;
            if (pool.layers() != static_cast<long>(m_.layers.size()) || pool.kv_heads() != m_.n_kv_heads
                || pool.head_dim() != m_.head_dim || static_cast<long>(plan.positions.size()) != n_)
                return false;
            dlib::kv_page_layout layout;
            layout.layers = pool.layers();
            layout.heads = pool.kv_heads();
            layout.head_dim = pool.head_dim();
            layout.block_positions = pool.block_positions();
            layout.q8 = pool.format() == kv_format::q8;
            if (!layout.valid() || layout.page_bytes() != pool.block_bytes()) return false;
            bool taken = false;
            try
            {
                taken = prepare(plan, layout);
            }
            catch (const std::exception& e)
            {
                settle();
                if (lack_of_memory(e)) return false;
                throw device_failure(e.what(), false);
            }
            if (!taken) return false;
            attending_ = true;
            ++o_.d_.attention_passes;
            return true;
        }

        void attend(long layer) override
        {
            guarded([&]() {
                const long qd = m_.q_dim, kvd = m_.kv_dim, H = m_.n_heads, KVH = m_.n_kv_heads, hd = m_.head_dim;
                const dense_layer& L = m_.layers[static_cast<size_t>(layer)];
                dlib::device_scope pin({ &x_, &xn_, &qkv_, &ctx_, &tmp_, &gate_, &up_, &o_.tile_, &o_.tile_out_ });
                dlib::device_scope pin_pages({ &pages_, &plan_ });
                project(layer);
                dlib::alias_tensor_instance q = view(qkv_, n_, qd);
                dlib::alias_tensor_instance k = view(qkv_, n_, kvd, n_ * qd);
                dlib::alias_tensor_instance v = view(qkv_, n_, kvd, n_ * (qd + kvd));
                const size_t at = 5 * static_cast<size_t>(layer);
                if (L.bq)
                {
                    dlib::tt::add(1, q, 1, extras_[at]);
                    dlib::tt::add(1, k, 1, extras_[at + 1]);
                    dlib::tt::add(1, v, 1, extras_[at + 2]);
                }
                if (L.q_norm)
                {
                    dlib::tt::rms_norm_rows(q, q, extras_[at + 3], m_.rms_eps);
                    dlib::tt::rms_norm_rows(k, k, extras_[at + 4], m_.rms_eps);
                }
                const long T = n_;
                dlib::alias_tensor_instance slots = view(plan_, T, 1);
                dlib::alias_tensor_instance tokens = view(plan_, 2 * T, 1, T);
                dlib::alias_tensor_instance tables = view(plan_, table_entries_, 1, 3 * T);
                dlib::alias_tensor_instance angles = view(plan_, T, m_.rope_dims, 3 * T + table_entries_);
                dlib::tt::rope_rows(q, angles, H, hd, m_.rope_dims, m_.neox);
                dlib::tt::rope_rows(k, angles, KVH, hd, m_.rope_dims, m_.neox);
                dlib::tt::paged_kv_store(pages_, layout_, layer, k, v, slots);
                dlib::alias_tensor_instance attended = view(ctx_, n_, qd);
                dlib::tt::paged_attention(attended, q, H, m_.attention_scale, pages_, layout_, layer, tokens, tables,
                    span_);
                finish(layer);
            });
        }

        void truncated(const kv_pool& pool, int block, long row) override
        {
            if (pool.id() != mirrored_pool_) return;
            auto it = entries_.find(block);
            if (it != entries_.end()) it->second.rows = std::min(it->second.rows, std::max(0L, row));
        }

        void end(const std::vector<long>& rows, float* x) override
        {
            guarded([&]() {
                const long d = m_.d_model;
                dlib::device_scope pin({ &x_, &pages_, &plan_, &gathered_ });
                const uint8_t* written = attending_ ? gather_written() : nullptr;
                if (rows.size() == 1)
                {
                    // The row of the next token of a single sequence, the usual case.
                    const long r = rows[0];
                    const float* row = download(view(x_, 1, d, r * d), static_cast<size_t>(d));
                    std::memcpy(x + r * d, row, sizeof(float) * static_cast<size_t>(d));
                }
                else if (!rows.empty())
                {
                    const float* all = download(view(x_, n_, d), static_cast<size_t>(n_ * d));
                    for (long r : rows) std::memcpy(x + r * d, all + r * d, sizeof(float) * static_cast<size_t>(d));
                }
#if defined(DLIB_USE_CUDA)
                else
                {
                    // A pass without logits, a part of a prompt, ends once the device has computed it,
                    // so that a failure of the device belongs to the pass that met it.
                    cuda_check(cudaStreamSynchronize(0), "computation of a layer on the device");
                }
#endif
                // The copies of the rows crossed with the hidden states, which the device waited for.
                if (written) scatter_written(written);
            });
            attending_ = false;
            n_ = 0;
        }

        long min_tokens() const override
        {
            return std::max<long>(1, o_.d_.min_tokens);
        }

        long tiled_tokens() const override
        {
            return expanded_product_tokens;
        }

        void report(bool ok, const std::string& detail) override
        {
            if (ok && attends())
                o_.d_.message(ALBIREO_LOG_INFO, "device: the layers of '" + o_.model() + "' run on the device, which "
                    "attends the tokens of the generation itself, from its copies of the keys and values, and "
                    "exchanges the inputs and outputs of the attention of the longer passes with the processor once "
                    "per layer (" + detail + ")");
            else if (ok)
                o_.d_.message(ALBIREO_LOG_INFO, "device: the layers of '" + o_.model() + "' run on the device, which "
                    "exchanges the inputs and outputs of the attention with the processor once per layer (" + detail
                    + ")");
            else
                o_.d_.message(ALBIREO_LOG_WARNING, "device: the layers of '" + o_.model() + "' return to the "
                    "processor, which hands their products to the device one by one (" + detail + ")");
        }

        // The bytes of the budget the pages hold.
        size_t page_room() const
        {
            return reserved_;
        }

        // Frees the pages and returns the bytes of the budget they held; the copies of the blocks are forgotten.
        size_t release_pages()
        {
            const size_t freed = reserved_;
            {
                // The tensor goes without waiting for the device, which no pass uses between two passes.
                dlib::resizable_tensor gone;
                pages_.swap(gone);
            }
            capacity_ = 0;
            entries_.clear();
            free_pages_.clear();
            mirrored_pool_ = 0;
            if (freed > 0) release_room();
            return freed;
        }

    private:
        /*
            The copies of the seven matrices of every layer, made when they are missing; false when
            one of them lacks room.  The copies of a model leave the device only for another model,
            between two passes, and never for the copies of their own model: the pointers stay
            valid for the pass.
        */
        bool gather_copies()
        {
            copies_.clear();
            for (const dense_layer& L : m_.layers)
                for (const weight_matrix* W : { &L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down })
                {
                    bool created = false;
                    const tensor_offload::copy* c = o_.resident_copy(*W, created);
                    if (!c) return false;
                    copies_.push_back(c);
                }
            return true;
        }

        // Copies a vector of the model to a tensor of one row, or leaves the tensor empty without one.
        static void copy_vector(dlib::resizable_tensor& t, const float* values, long count)
        {
            if (!values) return;
            t.set_size(1, count);
            std::memcpy(t.host_write_only(), values, sizeof(float) * static_cast<size_t>(count));
        }

        /*
            The norms of the blocks, and for the attention of the device the biases of the
            queries, keys and values and the norms of the queries and keys, five tensors per
            layer, empty when the model lacks them.
        */
        void copy_norms()
        {
            const long d = m_.d_model;
            std::vector<dlib::resizable_tensor> norms(2 * m_.layers.size());
            std::vector<dlib::resizable_tensor> extras(5 * m_.layers.size());
            for (size_t i = 0; i < m_.layers.size(); ++i)
            {
                const dense_layer& L = m_.layers[i];
                copy_vector(norms[2 * i], L.attn_norm, d);
                copy_vector(norms[2 * i + 1], L.ffn_norm, d);
                if (!attends_) continue;
                copy_vector(extras[5 * i], L.bq, m_.q_dim);
                copy_vector(extras[5 * i + 1], L.bk, m_.kv_dim);
                copy_vector(extras[5 * i + 2], L.bv, m_.kv_dim);
                copy_vector(extras[5 * i + 3], L.q_norm, m_.head_dim);
                copy_vector(extras[5 * i + 4], L.k_norm, m_.head_dim);
            }
            norms_.swap(norms);
            extras_.swap(extras);
        }

        // The queries, keys and values of the tokens of the pass for layer, into qkv_.
        void project(long layer)
        {
            const long d = m_.d_model, qd = m_.q_dim, kvd = m_.kv_dim;
            const dense_layer& L = m_.layers[static_cast<size_t>(layer)];
            const tensor_offload::copy* const* c = &copies_[static_cast<size_t>(layer) * 7];
            dlib::alias_tensor_instance x = view(x_, n_, d);
            dlib::alias_tensor_instance xn = view(xn_, n_, d);
            dlib::tt::rms_norm_rows(xn, x, norms_[2 * static_cast<size_t>(layer)], m_.rms_eps);
            dlib::alias_tensor_instance qv = view(qkv_, n_, qd);
            dlib::alias_tensor_instance kv = view(qkv_, n_, kvd, n_ * qd);
            dlib::alias_tensor_instance vv = view(qkv_, n_, kvd, n_ * (qd + kvd));
            o_.product_on_device(L.wq, *c[0]->t, c[0]->quantized, xn, qv);
            o_.product_on_device(L.wk, *c[1]->t, c[1]->quantized, xn, kv);
            o_.product_on_device(L.wv, *c[2]->t, c[2]->quantized, xn, vv);
        }

        // The projection of the outputs of the attention in ctx_, then the feed-forward block, with both residual sums.
        void finish(long layer)
        {
            const long d = m_.d_model, qd = m_.q_dim, df = m_.d_ffn;
            const dense_layer& L = m_.layers[static_cast<size_t>(layer)];
            const tensor_offload::copy* const* c = &copies_[static_cast<size_t>(layer) * 7];
            dlib::alias_tensor_instance x = view(x_, n_, d);
            dlib::alias_tensor_instance xn = view(xn_, n_, d);
            dlib::alias_tensor_instance tmp = view(tmp_, n_, d);
            dlib::alias_tensor_instance attended = view(ctx_, n_, qd);
            o_.product_on_device(L.wo, *c[3]->t, c[3]->quantized, attended, tmp);
            dlib::tt::add(1, x, m_.residual_scale, tmp);
            dlib::tt::rms_norm_rows(xn, x, norms_[2 * static_cast<size_t>(layer) + 1], m_.rms_eps);
            dlib::alias_tensor_instance gate = view(gate_, n_, df);
            dlib::alias_tensor_instance up = view(up_, n_, df);
            o_.product_on_device(L.w_gate, *c[4]->t, c[4]->quantized, xn, gate);
            o_.product_on_device(L.w_up, *c[5]->t, c[5]->quantized, xn, up);
            dlib::tt::silu_multiply(gate, up);
            o_.product_on_device(L.w_down, *c[6]->t, c[6]->quantized, gate, tmp);
            dlib::tt::add(1, x, m_.residual_scale, tmp);
        }

        /*
            Gives the blocks of the pass their pages, copies the blocks whose pages lack rows the
            pass reads, and the tokens of the pass to the device; false when the pages cannot hold
            the blocks of the pass.
        */
        bool prepare(const attention_plan& plan, const dlib::kv_page_layout& layout)
        {
            const kv_pool& pool = *plan.pool;
            const long B = layout.block_positions;
            const long T = n_;
            const long R = m_.rope_dims;
            if (static_cast<long>(plan.firsts.size()) != T || static_cast<long>(plan.angles.size()) != T * R
                || plan.blocks.empty())
                return false;
            if (pool.id() != mirrored_pool_ || layout.q8 != layout_.q8 || layout.page_bytes() != layout_.page_bytes()
                || B != layout_.block_positions)
            {
                // Another pool: its blocks have other numbers, and pages of another size are made again.
                if (layout.page_bytes() != layout_.page_bytes()) release_pages();
                entries_.clear();
                free_pages_.clear();
                for (long p = capacity_; p-- > 0;) free_pages_.push_back(p);
                mirrored_pool_ = pool.id();
                layout_ = layout;
            }
            ++clock_;

            // The rows each block of the pass must hold before the pass, a block shared by
            // segments needing the most of them.
            needed_.clear();
            for (const attention_plan::segment& s : plan.segments)
            {
                const long blocks = (s.length + s.count + B - 1) / B;
                if (s.first < 0 || s.first + blocks > static_cast<long>(plan.blocks.size())) return false;
                for (long i = 0; i < blocks; ++i)
                    needed_.push_back(std::make_pair(plan.blocks[static_cast<size_t>(s.first + i)],
                        std::max(0L, std::min(B, s.length - i * B))));
            }
            std::sort(needed_.begin(), needed_.end());
            size_t kept = 0;
            for (size_t i = 0; i < needed_.size(); ++i)
            {
                if (kept > 0 && needed_[kept - 1].first == needed_[i].first)
                    needed_[kept - 1].second = std::max(needed_[kept - 1].second, needed_[i].second);
                else
                    needed_[kept++] = needed_[i];
            }
            needed_.resize(kept);

            // The copies of the blocks of the pass whose blocks changed since leave their pages.
            size_t missing = 0;
            for (const auto& b : needed_)
            {
                auto it = entries_.find(b.first);
                if (it != entries_.end() && it->second.epoch == pool.epoch(b.first))
                {
                    it->second.used = clock_;
                    continue;
                }
                if (it != entries_.end())
                {
                    free_pages_.push_back(it->second.page);
                    entries_.erase(it);
                }
                ++missing;
            }
            if (missing > free_pages_.size())
            {
                // The pages of the blocks the pass does not read can go to its blocks.
                size_t idle = 0;
                for (const auto& e : entries_)
                    if (e.second.used != clock_) ++idle;
                if (missing > free_pages_.size() + idle && !grow(capacity_ + static_cast<long>(missing
                    - free_pages_.size() - idle), pool))
                {
                    const long wanted = capacity_ + static_cast<long>(missing - free_pages_.size() - idle);
                    if (!told_room_)
                        o_.d_.message(ALBIREO_LOG_INFO, "device: the copies of the keys and values of '" + o_.model()
                            + "' found no room for " + std::to_string(wanted) + (wanted == 1 ? " block" : " blocks")
                            + " of " + megabytes(layout.page_bytes()) + ": the processor attends the passes that read "
                            "more blocks than the device holds");
                    told_room_ = true;
                    return false;
                }
            }
            while (free_pages_.size() < missing)
            {
                auto oldest = entries_.end();
                for (auto it = entries_.begin(); it != entries_.end(); ++it)
                    if (it->second.used != clock_
                        && (oldest == entries_.end() || it->second.used < oldest->second.used))
                        oldest = it;
                if (oldest == entries_.end()) return false;
                free_pages_.push_back(oldest->second.page);
                entries_.erase(oldest);
            }

            /*
                The buffers of the tokens and of the rows they write take their size before the
                pins: a tensor of the extended memory that grows leaves its record, which a pin
                would still hold.
            */
            table_entries_ = static_cast<long>(plan.blocks.size());
            const size_t words = static_cast<size_t>(3 * T + table_entries_ + T * R);
            tensor_offload::reserve_buffer(plan_, words);
            tensor_offload::reserve_buffer(gathered_, (static_cast<size_t>(T) * token_kv_bytes() + sizeof(float) - 1)
                / sizeof(float));

            dlib::device_scope pin({ &pages_, &plan_ });
            for (const auto& b : needed_)
            {
                auto it = entries_.find(b.first);
                if (it == entries_.end())
                {
                    page_entry e;
                    e.page = free_pages_.back();
                    e.epoch = pool.epoch(b.first);
                    e.used = clock_;
                    free_pages_.pop_back();
                    it = entries_.emplace(b.first, e).first;
                }
                if (it->second.rows < b.second)
                {
                    upload_block(pool, b.first, it->second.page);
                    it->second.rows = b.second;
                }
            }

            // The tokens: their slots, their positions and first entries, the pages of the blocks, their angles.
            plan_host_.resize(words);
            written_.resize(static_cast<size_t>(T));
            long span = 0;
            for (long t = 0; t < T; ++t)
            {
                const long p = plan.positions[static_cast<size_t>(t)];
                const long f = plan.firsts[static_cast<size_t>(t)];
                if (p < 0 || f < 0 || f + p / B >= table_entries_ || p >= std::numeric_limits<int32_t>::max())
                    return false;
                const int block = plan.blocks[static_cast<size_t>(f + p / B)];
                const page_entry& e = entries_.at(block);
                put_integer(static_cast<size_t>(t), static_cast<int32_t>(e.page * B + p % B));
                put_integer(static_cast<size_t>(T + 2 * t), static_cast<int32_t>(p));
                put_integer(static_cast<size_t>(T + 2 * t + 1), static_cast<int32_t>(f));
                written_[static_cast<size_t>(t)] = std::make_pair(block, p % B);
                span = std::max(span, p + 1);
            }
            for (long i = 0; i < table_entries_; ++i)
                put_integer(static_cast<size_t>(3 * T + i),
                    static_cast<int32_t>(entries_.at(plan.blocks[static_cast<size_t>(i)]).page));
            std::memcpy(plan_host_.data() + 3 * T + table_entries_, plan.angles.data(),
                sizeof(float) * static_cast<size_t>(T * R));
            span_ = span;
            pool_ = plan.pool;
            upload_plan(words);
            return true;
        }

        void put_integer(size_t at, int32_t value)
        {
            std::memcpy(&plan_host_[at], &value, sizeof(value));
        }

        /*
            Gives the pages room for wanted pages at least, twice as many as they had if the budget
            allows, within the blocks the pool can ever hold; false when the budget lacks room.
            The pages keep their contents.
        */
        bool grow(long wanted, const kv_pool& pool)
        {
            const long limit = static_cast<long>(pool.max_blocks());
            if (wanted > limit) return false;
            const size_t page = layout_.page_bytes();
            auto& d = o_.d_;
            for (long target : { std::min(limit, std::max(wanted, std::max(2 * capacity_, min_kv_pages))), wanted })
            {
                if (target < wanted) continue;
                const size_t extra = static_cast<size_t>(target - capacity_) * page;
                /*
                    Without the extended memory, every tensor of dlib has a copy in the memory of the
                    host: the pages then keep to an eighth of the budget.
                */
                if (!d.v2ram && reserved_ + extra > d.budget / 8) continue;
                /*
                    Beyond the memory of the card, the passes of a few tokens stay on the processor
                    (begin()): the pages keep within it.
                */
                if (d.card_bytes != std::numeric_limits<size_t>::max()
                    && d.resident.load(std::memory_order_relaxed) + extra > d.card_bytes)
                    continue;
                if (!d.reserve(extra) && !(d.make_room(extra, &o_) && d.reserve(extra))) continue;
                try
                {
                    dlib::resizable_tensor bigger;
                    bigger.set_size(static_cast<long long>((static_cast<size_t>(target) * page + sizeof(float) - 1)
                        / sizeof(float)));
                    if (capacity_ > 0)
                    {
                        dlib::device_scope pin({ &pages_, &bigger });
                        const size_t bytes = static_cast<size_t>(capacity_) * page;
#if defined(DLIB_USE_CUDA)
                        cuda_check(cudaMemcpy(bigger.device(), pages_.device(), bytes, cudaMemcpyDeviceToDevice),
                            "copy of keys and values on the device");
#else
                        std::memcpy(bigger.host(), pages_.host(), bytes);
#endif
                    }
                    pages_.swap(bigger);
                }
                catch (const std::exception& e)
                {
                    d.release(extra);
                    if (lack_of_memory(e)) continue;
                    throw;
                }
                reserved_ += extra;
                d.kv_page_bytes += extra;
                for (long p = target; p-- > capacity_;) free_pages_.push_back(p);
                capacity_ = target;
                return true;
            }
            return false;
        }

        void release_room()
        {
            o_.d_.release(reserved_);
            o_.d_.kv_page_bytes -= reserved_;
            reserved_ = 0;
        }

        // Copies block b of the pool, whole, into a page.
        void upload_block(const kv_pool& pool, int b, long page)
        {
            const size_t bytes = layout_.page_bytes();
#if defined(DLIB_USE_CUDA)
            uint8_t* dst = reinterpret_cast<uint8_t*>(pages_.device()) + static_cast<size_t>(page) * bytes;
            cuda_check(cudaMemcpy(dst, pool.block_memory(b), bytes, cudaMemcpyHostToDevice),
                "copy of keys and values to the device");
#else
            std::memcpy(reinterpret_cast<uint8_t*>(pages_.host()) + static_cast<size_t>(page) * bytes,
                pool.block_memory(b), bytes);
#endif
        }

        // Copies the first words of plan_host_ to plan_.
        void upload_plan(size_t words)
        {
#if defined(DLIB_USE_CUDA)
            // The pinned buffer is free: the previous pass waited for the device at its end.
            float* staging = plan_staging_.reserve(words);
            std::memcpy(staging, plan_host_.data(), sizeof(float) * words);
            cuda_check(cudaMemcpyAsync(plan_.device(), staging, sizeof(float) * words, cudaMemcpyHostToDevice, 0),
                "copy of the tokens of a pass to the device");
#else
            std::memcpy(plan_.host(), plan_host_.data(), sizeof(float) * words);
#endif
        }

        // The bytes of the keys and values of a token in the order of dlib::tt::paged_kv_gather().
        size_t token_kv_bytes() const
        {
            return 2 * static_cast<size_t>(layout_.layers) * static_cast<size_t>(layout_.heads) * layout_.row_bytes();
        }

        /*
            Gathers the rows the tokens of the pass wrote in the pages and starts their copy to the
            host: the returned bytes are complete once the device has been waited for.  prepare()
            gave gathered_ its size.
        */
        const uint8_t* gather_written()
        {
            const size_t bytes = static_cast<size_t>(n_) * token_kv_bytes();
            const size_t values = (bytes + sizeof(float) - 1) / sizeof(float);
            dlib::alias_tensor_instance gathered = view(gathered_, static_cast<long>(values), 1);
            dlib::alias_tensor_instance slots = view(plan_, n_, 1);
            dlib::tt::paged_kv_gather(gathered, pages_, layout_, slots);
#if defined(DLIB_USE_CUDA)
            float* staging = kv_from_device_.reserve(values);
            cuda_check(cudaMemcpyAsync(staging, gathered_.device(), bytes, cudaMemcpyDeviceToHost, 0),
                "copy of keys and values from the device");
            return reinterpret_cast<const uint8_t*>(staging);
#else
            return reinterpret_cast<const uint8_t*>(gathered_.host());
#endif
        }

        // Writes the rows of the tokens of the pass into their blocks of the pool, which the pages now hold too.
        void scatter_written(const uint8_t* src)
        {
            kv_pool& pool = *pool_;
            const size_t data = layout_.data_bytes();
            const size_t scales = layout_.scale_bytes();
            const size_t at_scales = static_cast<size_t>(layout_.block_positions) * data;
            for (const auto& w : written_)
            {
                uint8_t* block = pool.block_memory(w.first);
                const size_t r = static_cast<size_t>(w.second);
                for (long kind = 0; kind < 2; ++kind)
                    for (long l = 0; l < layout_.layers; ++l)
                        for (long h = 0; h < layout_.heads; ++h)
                        {
                            uint8_t* slab = block + layout_.slab_offset(kind, l, h);
                            std::memcpy(slab + r * data, src, data);
                            src += data;
                            if (scales == 0) continue;
                            std::memcpy(slab + at_scales + r * scales, src, scales);
                            src += scales;
                        }
                auto it = entries_.find(w.first);
                if (it != entries_.end()) it->second.rows = std::max(it->second.rows, w.second + 1);
            }
        }

        // The n rows of cols floats of a buffer that start first floats in, as a tensor.
        static dlib::alias_tensor_instance view(dlib::resizable_tensor& buffer, long n, long cols, long first = 0)
        {
            dlib::alias_tensor a(n, cols);
            return a(buffer, static_cast<size_t>(first));
        }

        // Copies count floats of the host to the start of the view dst of the device.
        void upload(dlib::tensor& dst, const float* src, size_t count)
        {
#if defined(DLIB_USE_CUDA)
            // A copy of the buffer is still running only before the wait of the layer that follows.
            float* staging = to_device_.reserve(count);
            std::memcpy(staging, src, sizeof(float) * count);
            cuda_check(cudaMemcpyAsync(dst.device(), staging, sizeof(float) * count, cudaMemcpyHostToDevice, 0),
                "copy of the activations to the device");
#else
            std::memcpy(dst.host(), src, sizeof(float) * count);
#endif
        }

        // The first count floats of the view src of the device, once the device has computed them.
        const float* download(const dlib::tensor& src, size_t count)
        {
#if defined(DLIB_USE_CUDA)
            float* staging = from_device_.reserve(count);
            cuda_check(cudaMemcpyAsync(staging, src.device(), sizeof(float) * count, cudaMemcpyDeviceToHost, 0),
                "copy of the activations from the device");
            cuda_check(cudaStreamSynchronize(0), "computation of a layer on the device");
            return staging;
#else
            (void)count;
            return src.host();
#endif
        }

        // Runs f, a failure of the device becoming a device_failure.
        template <typename F>
        void guarded(F&& f)
        {
            try
            {
                f();
            }
            catch (const device_failure&)
            {
                settle();
                throw;
            }
            catch (const std::exception& e)
            {
                settle();
                throw device_failure(e.what(), lack_of_memory(e));
            }
        }

        /*
            Waits for the copies and the kernels a failed pass left queued, so that the pinned
            buffers are free for the next pass; the error of a failed device stays with the call
            that met it.
        */
        static void settle()
        {
#if defined(DLIB_USE_CUDA)
            cudaStreamSynchronize(0);
#endif
        }

        struct page_entry
        {
            long page = 0;
            uint32_t epoch = 0;
            // The rows of the block the page holds as the pool does, from the first.
            long rows = 0;
            // The last pass that read the block.
            unsigned long long used = 0;
        };

        tensor_offload& o_;
        const dense_model m_;
        long n_ = 0;
        std::vector<const tensor_offload::copy*> copies_;
        // The norms of the attention and of the feed-forward block of each layer, in turn.
        std::vector<dlib::resizable_tensor> norms_;
        // The biases and norms of the queries and keys of each layer (copy_norms()).
        std::vector<dlib::resizable_tensor> extras_;
        dlib::resizable_tensor x_, xn_, tmp_, qkv_, ctx_, gate_, up_;

        // The attention of the device: whether the model and the runtime allow it, and the pass takes it.
        bool attends_ = false;
        bool allowed_ = true;
        bool attending_ = false;
        // Whether the log told that the pages found no room.
        bool told_room_ = false;
        // The pages, capacity_ of them, the bytes of the budget they hold, and the pages without a block.
        dlib::kv_page_layout layout_;
        dlib::resizable_tensor pages_;
        long capacity_ = 0;
        size_t reserved_ = 0;
        std::vector<long> free_pages_;
        // The copies of the blocks of the pool whose id is mirrored_pool_, by block.
        std::unordered_map<int, page_entry> entries_;
        unsigned long long mirrored_pool_ = 0;
        unsigned long long clock_ = 0;
        std::vector<std::pair<int, long>> needed_;
        /*
            The tokens of the pass on the device, as dlib::tt::paged_attention() reads them: their
            slots, their positions with their first entries, the pages of the blocks of the plan
            (table_entries_ of them), as integers, then their angles.
        */
        dlib::resizable_tensor plan_;
        std::vector<float> plan_host_;
        long table_entries_ = 0;
        long span_ = 0;
        // The block and the row of each token of the pass, in the pool the pass writes.
        std::vector<std::pair<int, long>> written_;
        kv_pool* pool_ = nullptr;
        dlib::resizable_tensor gathered_;
#if defined(DLIB_USE_CUDA)
        // The rows that cross the bus, each way, the tokens of a pass, and the keys and values they wrote.
        pinned_floats to_device_, from_device_, plan_staging_, kv_from_device_;
#endif
    };

    size_t tensor_offload::pass_room() const
    {
        return pass_ ? pass_->page_room() : 0;
    }

    size_t tensor_offload::release_pass_pages()
    {
        return pass_ ? pass_->release_pages() : 0;
    }

    std::unique_ptr<layer_pass> tensor_offload::make_layer_pass(const dense_model& model)
    {
#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
        // The routing of dlib takes products only.
        (void)model;
        return std::unique_ptr<layer_pass>();
#else
        if (d_.forced < 0 || model.layers.empty()) return std::unique_ptr<layer_pass>();
        return std::unique_ptr<layer_pass>(new tensor_layer_pass(*this, model));
#endif
    }

    bool tensor_device::impl::make_room(size_t bytes, const tensor_offload* requester)
    {
        // A matrix larger than the whole budget never fits: nothing leaves for it.
        if (bytes > budget) return false;
        std::lock_guard<std::mutex> lock(registry_m);
        std::vector<tensor_offload*> idle;
        for (tensor_offload* o : offloads)
            if (o != requester && !o->busy() && o->copy_bytes() > 0) idle.push_back(o);
        std::sort(idle.begin(), idle.end(),
            [](const tensor_offload* a, const tensor_offload* b) { return a->last_use() < b->last_use(); });
        const size_t room = budget - bytes;
        const size_t target = room - std::min(room, budget / 16);
        for (tensor_offload* o : idle)
        {
            const size_t current = resident.load();
            if (current <= target) break;
            const size_t freed = o->evict_copies(current - target);
            ++evictions;
            evicted_bytes += freed;
            message(o->reported_eviction() ? ALBIREO_LOG_DEBUG : ALBIREO_LOG_INFO, "device: copies of '"
                + o->model() + "', without requests, leave the device for '" + requester->model() + "' ("
                + megabytes(freed) + ", " + megabytes(o->copy_bytes()) + " stay)");
        }
        return resident.load() + bytes <= budget;
    }

    tensor_device::tensor_device(const json& config, thread_pool& pool, log_function log)
        : impl_(new impl(config, pool, std::move(log)))
    {
        impl& d = *impl_;
        const memory_info mem = read_memory_info();
        const size_t available = static_cast<size_t>(mem.available);
        const size_t requested = static_cast<size_t>(std::max<long long>(0, config["offload_mb"].as_int(0))) << 20;
        const std::string cache_dir = config["cache_dir"].str();
        const std::string device = config["device"].str();
        d.min_tokens = std::max<long>(1, static_cast<long>(config["offload_min_tokens"].as_int(1)));
        const std::string routing = config["routing"].str("auto");
        d.forced = routing == "device" ? 1 : routing == "host" ? -1 : 0;

#if defined(DLIB_USE_CUDA)
        d.backend = "cuda";
        std::string older;
        const std::vector<int> cards = usable_cards(older);
        int dev = -1;
        if (!device.empty())
        {
            dev = std::atoi(device.c_str());
            if (std::find(cards.begin(), cards.end(), dev) == cards.end())
            {
                cudaDeviceProp p;
                if (dev >= 0 && cudaGetDeviceProperties(&p, dev) == cudaSuccess)
                    throw std::runtime_error(too_old(card_text(p)));
                throw std::runtime_error("there is no graphics processor " + device);
            }
        }
        else
        {
            // The card with the most memory.
            size_t most = 0;
            for (int i : cards)
            {
                cudaDeviceProp p;
                if (cudaGetDeviceProperties(&p, i) == cudaSuccess && (dev < 0 || p.totalGlobalMem > most))
                {
                    most = p.totalGlobalMem;
                    dev = i;
                }
            }
            if (dev < 0)
                throw std::runtime_error(older.empty() ? "no NVIDIA graphics processor was found" : too_old(older));
        }
        dlib::cuda::set_device(dev);
        d.cuda_device = dev;
        d.name = dlib::cuda::get_device_name(dev);

        if (config["v2ram"].as_bool(true) && !dlib::extended_memory_enabled())
        {
            dlib::extended_memory_options o;
            if (!cache_dir.empty())
            {
                const std::string store = join(cache_dir, "v2ram");
                if (make_folder(store)) o.store_path = store;
            }
            if (o.store_path.empty()) o.store_path = dlib::default_extended_memory_store_path();
            o.store_bytes = static_cast<size_t>(std::max<long long>(0, config["v2ram_max_mb"].as_int(0))) << 20;
            /*
                The memory of the card is not given back while the engine is idle.  The idle
                release of the extended memory evicts blocks on its observation thread, while
                the scheduler takes the device pointers of the same blocks without its lock: a
                request that arrives during a release could compute with a buffer already given
                back.  idle_release_ms stays at 0 until the release runs on the thread that
                computes.
            */
            o.idle_release_ms = 0;
            try
            {
                d.v2ram = dlib::enable_extended_memory(o);
                if (!d.v2ram)
                    d.message(ALBIREO_LOG_WARNING, "device: the extended memory did not start (the folder of its file "
                        "is too slow); the models use the memory of the card only");
            }
            catch (const std::exception& e)
            {
                d.message(ALBIREO_LOG_WARNING, std::string("device: the extended memory did not start: ") + e.what());
            }
        }
        else
        {
            d.v2ram = dlib::extended_memory_enabled();
        }

        if (requested > 0)
            d.budget = requested;
        else if (d.v2ram)
        {
            const dlib::extended_memory_stats s = dlib::get_extended_memory_stats();
            /*
                The copies stay within the memory of the card, less what the products, the matrices
                that cross the bus for a long prompt and cuBLAS take: beyond it, the extended memory
                would bring them over the bus and through its file at each pass, which a generated
                token then waits for, and the matrices left out run on the processor from their
                quantized weights.  A size given to the file of the extended memory (v2ram_max_mb)
                lets the copies go beyond the card into it.
            */
            const size_t reserve = std::min(card_reserve_bytes, s.vram_budget / 4);
            d.budget = s.vram_budget - reserve;
            if (config["v2ram_max_mb"].as_int(0) > 0) d.budget += s.store_capacity;
        }
        else
        {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) free_bytes = 0;
            // Every tensor also has a copy in the memory of the host.
            d.budget = std::min(static_cast<size_t>(free_bytes * 0.8), available / 2);
        }
        if (d.v2ram) d.card_bytes = dlib::get_extended_memory_stats().vram_budget;
        d.message(ALBIREO_LOG_INFO, "device: " + d.name + " through CUDA, " + (d.v2ram ? "extended memory on, " : "")
            + "weights in their GGUF formats up to " + megabytes(d.budget));
        d.name += d.v2ram ? " (CUDA, V\xC2\xB2" "RAM)" : " (CUDA)";
#elif defined(DLIB_USE_NPU)
        d.backend = "openvino";
        if (!device.empty()) set_environment("DLIB_NPU_DEVICE", device);
        if (!cache_dir.empty())
        {
            const std::string graphs = join(cache_dir, "openvino");
            if (make_folder(graphs)) default_environment("DLIB_NPU_CACHE_DIR", graphs);
        }
        // The copies of the weights in single precision on the host, and their copies in
        // half precision for the devices, which share the memory of the host.
        d.budget = requested > 0 ? requested : available / 4;
        default_environment("DLIB_NPU_RESIDENT_MB", std::to_string((d.budget >> 21) + 256));
        // Measures and tests need the graphs before the products they time.
        dlib::npu::set_synchronous_compilation(config["benchmark"].as_bool(false) || d.forced > 0);
        dlib::npu::set_routing(d.forced > 0 ? dlib::npu::routing::always
            : d.forced < 0 ? dlib::npu::routing::never : dlib::npu::routing::automatic);
        if (!dlib::npu::is_available())
            throw std::runtime_error("no NPU or Intel graphics processor was found"
                + (device.empty() ? std::string() : " for '" + device + "'") + ": " + dlib::npu::unavailable_reason());
        d.name = dlib::npu::device_name() + " (OpenVINO " + dlib::npu::runtime_version() + ")";
        d.message(ALBIREO_LOG_INFO, "device: " + d.name + ", weights in single precision up to " + megabytes(d.budget));
#else
        d.backend = "dlib";
        d.budget = requested > 0 ? requested : available / 4;
        d.name = "the host code of dlib";
        d.message(ALBIREO_LOG_INFO, "device: host code of dlib, weights in their GGUF formats up to "
            + megabytes(d.budget));
#endif
    }

    tensor_device::~tensor_device()
    {
    }

    std::unique_ptr<matmul_offload> tensor_device::make_offload(const std::string& model)
    {
        return std::unique_ptr<matmul_offload>(new tensor_offload(*impl_, model));
    }

    bool tensor_device::bind_thread()
    {
        return impl_->bind_thread();
    }

    json tensor_device::stats() const
    {
        const impl& d = *impl_;
        json s = json::object();
        s["backend"] = d.backend;
        s["name"] = d.name;
        s["weights_bytes"] = static_cast<unsigned long long>(d.resident.load());
        s["weights_budget"] = static_cast<unsigned long long>(d.budget);
        // Beyond these bytes of copies, the products of fewer than 8 tokens stay on the processor.
        if (d.card_bytes != std::numeric_limits<size_t>::max())
            s["card_bytes"] = static_cast<unsigned long long>(d.card_bytes);
        s["device_products"] = d.device_products.load();
        s["host_products"] = d.host_products.load();
        s["device_seconds"] = d.device_us.load() / 1e6;
        s["host_seconds"] = d.host_us.load() / 1e6;
        s["failures"] = d.failures.load();
        // The passes whose layers ran on the device (layer_pass), and their tokens.
        s["layer_passes"] = d.layer_passes.load();
        s["layer_pass_tokens"] = d.layer_pass_tokens.load();
        // The passes the device attended itself, and the bytes of its copies of keys and values.
        s["attention_passes"] = d.attention_passes.load();
        s["kv_bytes"] = static_cast<unsigned long long>(d.kv_page_bytes.load());
        s["disabled"] = d.disabled.load();
        s["evictions"] = d.evictions.load();
        s["evicted_bytes"] = d.evicted_bytes.load();
#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
        for (const auto& op : dlib::npu::get_statistics())
        {
            if (op.name != "gemm") continue;
            json g = json::object();
            g["device_calls"] = op.device_calls;
            g["host_calls"] = op.host_calls;
            g["graphs"] = op.graphs;
            g["compile_failures"] = op.compile_failures;
            g["rejected"] = op.rejected;
            g["device_seconds"] = op.device_seconds;
            g["compile_seconds"] = op.compile_seconds;
            const std::vector<std::string> names = dlib::npu::device_names();
            json by = json::object();
            const size_t n = std::min(names.size(), op.calls_by_device.size());
            for (size_t i = 0; i < n; ++i) by[names[i]] = op.calls_by_device[i];
            g["calls_by_device"] = by;
            s["routing"] = g;
        }
#else
        json routes = json::array();
        {
            std::lock_guard<std::mutex> lock(d.m);
            for (const auto& kv : d.routes) routes.push_back(kv.second);
        }
        s["routes"] = routes;
#endif
        return s;
    }

    json tensor_device::memory() const
    {
#if defined(DLIB_USE_CUDA)
        if (!dlib::extended_memory_enabled()) return json();
        const dlib::extended_memory_stats s = dlib::get_extended_memory_stats();
        json m = json::object();
        m["vram_budget"] = static_cast<unsigned long long>(s.vram_budget);
        m["device_bytes"] = static_cast<unsigned long long>(s.device_bytes);
        m["device_peak_bytes"] = static_cast<unsigned long long>(s.device_peak_bytes);
        m["scratch_bytes"] = static_cast<unsigned long long>(s.scratch_bytes);
        m["pinned_bytes"] = static_cast<unsigned long long>(s.pinned_bytes);
        m["store_bytes"] = static_cast<unsigned long long>(s.store_bytes);
        m["store_capacity"] = static_cast<unsigned long long>(s.store_capacity);
        m["pooled_bytes"] = static_cast<unsigned long long>(s.pooled_bytes);
        m["device_free_bytes"] = static_cast<unsigned long long>(s.device_free_bytes);
        m["managed_blocks"] = static_cast<unsigned long long>(s.managed_blocks);
        m["evictions"] = static_cast<unsigned long long>(s.evictions);
        m["restores"] = static_cast<unsigned long long>(s.restores);
        m["prefetch_hits"] = static_cast<unsigned long long>(s.prefetch_hits);
        m["h2d_bytes"] = s.h2d_bytes;
        m["d2h_bytes"] = s.d2h_bytes;
        m["cycle_locked"] = s.cycle_locked;
        return m;
#else
        return json();
#endif
    }

    const char* tensor_device::backend_name()
    {
#if defined(DLIB_USE_CUDA)
        return "cuda";
#elif defined(DLIB_USE_NPU)
        return "openvino";
#else
        return "dlib";
#endif
    }

    json tensor_device::describe(std::string& backend, std::string& runtime, std::string& reason, std::string& code,
        json& args)
    {
        json devices = json::array();
#if defined(DLIB_USE_CUDA)
        backend = "cuda";
        int version = 0;
        if (cudaRuntimeGetVersion(&version) == cudaSuccess)
            runtime = "CUDA " + std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
        int driver = 0;
        if (cudaDriverGetVersion(&driver) != cudaSuccess || driver == 0)
        {
            reason = "no NVIDIA driver is installed";
            code = "no_nvidia_driver";
            return devices;
        }
        if (driver < version)
        {
            const std::string supported = std::to_string(driver / 1000) + "." + std::to_string((driver % 1000) / 10);
            reason = "the NVIDIA driver supports CUDA " + supported + ", this engine needs " + runtime
                + "; update the driver";
            code = "old_nvidia_driver";
            args["driver"] = "CUDA " + supported;
            args["runtime"] = runtime;
            return devices;
        }
        std::string older;
        for (int i : usable_cards(older))
        {
            cudaDeviceProp p;
            if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
            json d = json::object();
            d["type"] = "gpu";
            d["id"] = std::to_string(i);
            d["name"] = std::string(p.name);
            d["memory"] = static_cast<unsigned long long>(p.totalGlobalMem);
            d["compute_capability"] = std::to_string(p.major) + "." + std::to_string(p.minor);
            devices.push_back(d);
        }
        if (devices.empty() && older.empty())
        {
            reason = "no NVIDIA graphics processor was found";
            code = "no_nvidia_card";
        }
        else if (devices.empty())
        {
            reason = too_old(older);
            code = "old_nvidia_card";
            args["card"] = older;
            args["cc"] = std::to_string(ALBIREO_CUDA_MIN_CC / 10) + "." + std::to_string(ALBIREO_CUDA_MIN_CC % 10);
        }
#elif defined(DLIB_USE_NPU)
        backend = "openvino";
        runtime = "OpenVINO " + dlib::npu::runtime_version();
        // Why the plugins of OpenVINO left a device out, as they report it: the reason of an
        // engine that cannot run, else the devices missing beside those found.
        const std::string why = dlib::npu::unavailable_reason();
        if (!dlib::npu::is_available())
        {
            // A machine without the devices gets no_intel_device; a runtime or a plugin that
            // failed, or the routing disabled by DLIB_NPU, gets the detail of the failure.
            if (why.empty() || no_device_present(why))
            {
                reason = "no NPU or Intel graphics processor was found" + (why.empty() ? std::string() : ": " + why);
                code = "no_intel_device";
            }
            else
            {
                reason = "no NPU or Intel graphics processor could be used: " + why;
                code = "intel_device_unusable";
                args["detail"] = why;
            }
            return devices;
        }
        reason = why;
        // "NPU, Intel(R) AI Boost + GPU, Intel(R) Graphics"
        const std::string all = dlib::npu::device_name();
        size_t start = 0;
        while (start < all.size())
        {
            size_t end = all.find(" + ", start);
            if (end == std::string::npos) end = all.size();
            const std::string one = all.substr(start, end - start);
            start = end + 3;
            const size_t comma = one.find(", ");
            const std::string id = one.substr(0, comma);
            const std::string full = comma == std::string::npos ? id : one.substr(comma + 2);
            json d = json::object();
            const std::string kind = id.substr(0, 3);
            d["type"] = kind == "NPU" ? "npu" : kind == "GPU" ? "gpu" : "cpu";
            d["id"] = id;
            d["name"] = full;
            devices.push_back(d);
        }
        if (devices.empty())
        {
            reason = "no NPU or Intel graphics processor was found";
            code = "no_intel_device";
        }
#else
        backend = "dlib";
        runtime = "dlib host code";
        (void)reason;
        (void)code;
        (void)args;
#endif
        return devices;
    }

}
