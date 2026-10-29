// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tensor_backend.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

#include <dlib/cuda/tensor_tools.h>
#include <dlib/cuda/extended_memory.h>
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

        // A matrix larger than this is never expanded at each product: its copy is kept, or
        // the processor computes it.
        const size_t max_staged_bytes = size_t(512) << 20;

        // Samples of each place before the first choice of a shape.
        const int samples_per_place = 2;

        // Thrown for a matrix whose copy does not fit and which is too large to expand at
        // each product: the processor computes its products.
        struct beyond_budget {};
    }

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
        std::atomic<unsigned long long> device_products{0};
        std::atomic<unsigned long long> host_products{0};
        std::atomic<unsigned long long> device_us{0};
        std::atomic<unsigned long long> host_us{0};
        std::atomic<unsigned long long> failures{0};
        std::atomic<bool> disabled{false};

        mutable std::mutex m;
        std::map<std::string, json> routes;   // the choices made, for the statistics

        void message(int level, const std::string& text) const
        {
            if (log) log(level, text);
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

        void release(size_t bytes) { resident -= bytes; }

        void note_route(const std::string& key, const json& value)
        {
            std::lock_guard<std::mutex> lock(m);
            routes[key] = value;
        }
    };

    class tensor_offload : public matmul_offload
    {
    public:
        explicit tensor_offload(tensor_device::impl& d) : d_(d) {}

        ~tensor_offload() override
        {
            size_t total = 0;
            for (const auto& kv : weights_) total += kv.second.bytes;
            d_.release(total);
        }

        void matmul(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            const std::function<void()>& host) override
        {
            if (n < d_.min_tokens || d_.disabled.load(std::memory_order_relaxed))
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

    private:
        struct copy
        {
            std::unique_ptr<dlib::resizable_tensor> t;
            size_t bytes = 0;
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

        // Expands the rows of W to single precision floats, with the threads of the engine.
        void expand(const weight_matrix& W, float* dst)
        {
            const long cols = W.cols;
            const long rows_per_task = std::max<long>(1, 262144 / std::max<long>(1, cols));
            const long tasks = (W.rows + rows_per_task - 1) / rows_per_task;
            d_.pool.parallel_for(tasks, [&](long task, int) {
                const long a = task * rows_per_task;
                const long b = std::min(W.rows, a + rows_per_task);
                for (long r = a; r < b; ++r)
                    qk_dequantize_row(W.type, W.data + static_cast<size_t>(r) * W.row_bytes, dst + r * cols, cols);
            });
        }

        // The copy of W, made when it fits in the budget; null otherwise.  created tells
        // whether this call made it.
        const dlib::tensor* resident_copy(const weight_matrix& W, bool& created)
        {
            created = false;
            auto it = weights_.find(W.data);
            if (it != weights_.end()) return it->second.t.get();
            if (refused_.count(W.data)) return nullptr;
            const size_t bytes = static_cast<size_t>(W.rows) * static_cast<size_t>(W.cols) * sizeof(float);
            if (!d_.reserve(bytes))
            {
                refused_.insert(W.data);
                return nullptr;
            }
            copy c;
            try
            {
                c.t.reset(new dlib::resizable_tensor());
                c.t->set_size(W.rows, W.cols);
                expand(W, c.t->host_write_only());
            }
            catch (...)
            {
                d_.release(bytes);
                refused_.insert(W.data);
                throw;
            }
            c.bytes = bytes;
            const dlib::tensor* t = c.t.get();
            weights_[W.data] = std::move(c);
            created = true;
            return t;
        }

        static void reserve_buffer(dlib::resizable_tensor& b, size_t values)
        {
            if (b.size() >= values) return;
            const size_t grown = std::max(values, b.size() + b.size() / 2);
            b.set_size(static_cast<long long>(grown));
        }

        // Y = X W^T with the device, the weights given as a tensor.
        void device_gemm(const dlib::tensor& w, long rows, long cols, const float* X, long n, long x_stride,
            float* Y, long y_stride)
        {
            reserve_buffer(x_, static_cast<size_t>(n) * cols);
            reserve_buffer(y_, static_cast<size_t>(n) * rows);
            dlib::alias_tensor xa(n, cols), ya(n, rows);
            dlib::alias_tensor_instance x = xa(x_);
            dlib::alias_tensor_instance y = ya(y_);
            float* xp = x.host_write_only();
            if (x_stride == cols)
                std::memcpy(xp, X, sizeof(float) * static_cast<size_t>(n) * cols);
            else
                for (long t = 0; t < n; ++t) std::memcpy(xp + t * cols, X + t * x_stride, sizeof(float) * cols);
            dlib::tt::gemm(0, y, 1, x, false, w, true);
            copy_out(y, rows, n, Y, y_stride);
        }

        static void copy_out(const dlib::tensor& y, long rows, long n, float* Y, long y_stride)
        {
            const float* yp = y.host();
            if (y_stride == rows)
                std::memcpy(Y, yp, sizeof(float) * static_cast<size_t>(n) * rows);
            else
                for (long t = 0; t < n; ++t) std::memcpy(Y + t * y_stride, yp + t * rows, sizeof(float) * rows);
        }

        /*
            The product on the device.  Returns false when the time it took does not reflect
            the products to come, because it made the copy of the weights.  Throws when the
            matrix cannot go to the device at all.
        */
        bool device_product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride)
        {
            bool created = false;
            const dlib::tensor* w = resident_copy(W, created);
            if (w)
            {
                device_gemm(*w, W.rows, W.cols, X, n, x_stride, Y, y_stride);
                return !created;
            }
            const size_t values = static_cast<size_t>(W.rows) * static_cast<size_t>(W.cols);
            if (values * sizeof(float) > max_staged_bytes) throw beyond_budget();
            reserve_buffer(w_, values);
            dlib::alias_tensor wa(W.rows, W.cols);
            dlib::alias_tensor_instance staged = wa(w_);
            expand(W, staged.host_write_only());
            device_gemm(staged, W.rows, W.cols, X, n, x_stride, Y, y_stride);
            return true;
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
            bool on_device = false;
            bool check = false;
            if (r.failed || d_.forced < 0)
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
                    steady = device_product(W, X, n, x_stride, Y, y_stride);
                }
                catch (const beyond_budget&)
                {
                    r.failed = true;
                    host_product(host);
                    return;
                }
                catch (const std::exception& e)
                {
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

            if (r.failed || d_.forced != 0) return;
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
            const dlib::tensor* w = nullptr;
            try
            {
                bool created = false;
                w = resident_copy(W, created);
            }
            catch (const std::exception& e)
            {
                d_.message(ALBIREO_LOG_WARNING, std::string("device: no copy of a matrix (") + e.what() + ")");
            }
            if (!w)
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
            if (dlib::npu::gemm(0, y, 1, x, false, *w, true, dlib::operation_mode::CHANNEL_WISE))
            {
                copy_out(y, W.rows, n, Y, y_stride);
                count_device(seconds_since(t0));
                return;
            }
            dlib::npu::cpu_measure measure;
            host_product(host);
        }
#endif

        tensor_device::impl& d_;
        std::unordered_map<const uint8_t*, copy> weights_;
        std::unordered_set<const uint8_t*> refused_;
        std::map<std::tuple<long, long, int>, route> routes_;
        dlib::resizable_tensor x_, y_, w_;
    };

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
            // While the engine is idle, the card goes back to the other programs.
            o.idle_release_ms = 120000;
            o.idle_release_keep = 0.5;
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
            // Beyond the card, the file of the extended memory stays mostly in the memory of
            // the host when it is not given a size.
            const size_t beyond = config["v2ram_max_mb"].as_int(0) > 0 ? s.store_capacity
                : std::min<size_t>(s.store_capacity > 0 ? s.store_capacity : s.vram_budget, available / 2);
            d.budget = s.vram_budget + beyond;
        }
        else
        {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) free_bytes = 0;
            // Every tensor also has a copy in the memory of the host.
            d.budget = std::min(static_cast<size_t>(free_bytes * 0.8), available / 2);
        }
        d.message(ALBIREO_LOG_INFO, "device: " + d.name + " through CUDA, " + (d.v2ram ? "extended memory on, " : "")
            + "weights in single precision up to " + megabytes(d.budget));
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
                + (device.empty() ? std::string() : " for '" + device + "'"));
        d.name = dlib::npu::device_name() + " (OpenVINO " + dlib::npu::runtime_version() + ")";
        d.message(ALBIREO_LOG_INFO, "device: " + d.name + ", weights in single precision up to " + megabytes(d.budget));
#else
        d.backend = "dlib";
        d.budget = requested > 0 ? requested : available / 4;
        d.name = "the host code of dlib";
        d.message(ALBIREO_LOG_INFO, "device: host code of dlib, weights in single precision up to "
            + megabytes(d.budget));
#endif
    }

    tensor_device::~tensor_device()
    {
    }

    std::unique_ptr<matmul_offload> tensor_device::make_offload()
    {
        return std::unique_ptr<matmul_offload>(new tensor_offload(*impl_));
    }

    json tensor_device::stats() const
    {
        const impl& d = *impl_;
        json s = json::object();
        s["backend"] = d.backend;
        s["name"] = d.name;
        s["weights_bytes"] = static_cast<unsigned long long>(d.resident.load());
        s["weights_budget"] = static_cast<unsigned long long>(d.budget);
        s["device_products"] = d.device_products.load();
        s["host_products"] = d.host_products.load();
        s["device_seconds"] = d.device_us.load() / 1e6;
        s["host_seconds"] = d.host_us.load() / 1e6;
        s["failures"] = d.failures.load();
        s["disabled"] = d.disabled.load();
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

    json tensor_device::describe(std::string& backend, std::string& runtime, std::string& reason)
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
            return devices;
        }
        if (driver < version)
        {
            reason = "the NVIDIA driver supports CUDA " + std::to_string(driver / 1000) + "."
                + std::to_string((driver % 1000) / 10) + ", this engine needs " + runtime + "; update the driver";
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
        if (devices.empty()) reason = older.empty() ? "no NVIDIA graphics processor was found" : too_old(older);
#elif defined(DLIB_USE_NPU)
        backend = "openvino";
        runtime = "OpenVINO " + dlib::npu::runtime_version();
        if (!dlib::npu::is_available())
        {
            reason = "no NPU or Intel graphics processor was found";
            return devices;
        }
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
        if (devices.empty()) reason = "no NPU or Intel graphics processor was found";
#else
        backend = "dlib";
        runtime = "dlib host code";
        (void)reason;
#endif
        return devices;
    }

}
