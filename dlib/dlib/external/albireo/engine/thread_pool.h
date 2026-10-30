// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_THREAD_POOL_H_
#define ALBIREO_ENGINE_THREAD_POOL_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define ALBIREO_CPU_RELAX() _mm_pause()
#else
#define ALBIREO_CPU_RELAX() std::this_thread::yield()
#endif

namespace albireo
{

    class thread_pool
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The threads that compute the products, the attention and the
                normalizations of a forward pass.  A generation step runs a few hundred
                parallel sections, one per product, so what matters is the cost of
                starting one: the workers spin on an atomic counter for a short while
                after each section and only then go to sleep, which keeps the hand-over
                in the microsecond range while tokens are produced and lets the
                processor rest when the engine is idle.

                parallel_for() splits a range of tasks among the workers and the calling
                thread, hands them out one at a time through an atomic counter so that
                uneven tasks balance themselves, and returns once every task is done.
                Calls are serialized: a single thread drives the pool.  An exception thrown
                by a task skips the tasks not yet started and reaches the caller once every
                worker has left the section.
        !*/

    public:

        explicit thread_pool(int threads)
        {
            const int n = threads > 0 ? threads : 1;
            for (int i = 1; i < n; ++i)
                workers_.emplace_back([this, i]() { worker_loop(i); });
        }

        ~thread_pool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_.store(true, std::memory_order_release);
                generation_.fetch_add(1, std::memory_order_release);
            }
            cv_.notify_all();
            for (auto& t : workers_) t.join();
        }

        thread_pool(const thread_pool&) = delete;
        thread_pool& operator=(const thread_pool&) = delete;

        int size() const { return static_cast<int>(workers_.size()) + 1; }

        // How long the workers spin after a section before sleeping.
        void set_spin_time(std::chrono::microseconds t) { spin_us_.store(t.count(), std::memory_order_relaxed); }

        /*!
            ensures
                - Calls fn(task, thread_index) once for every task in [0, n_tasks),
                  thread_index being in [0, size()), and returns when all calls returned.
                - Runs everything on the calling thread when n_tasks == 1 or the pool
                  has a single thread.
                - Rethrows the first exception a task threw, after every worker is done.
        !*/
        template <typename F>
        void parallel_for(long n_tasks, F&& fn)
        {
            if (n_tasks <= 0) return;
            if (n_tasks == 1 || workers_.empty())
            {
                for (long t = 0; t < n_tasks; ++t) fn(t, 0);
                return;
            }
            job_ctx_ = static_cast<void*>(&fn);
            job_fn_ = &invoke<typename std::remove_reference<F>::type>;
            n_tasks_ = n_tasks;
            next_.value.store(0, std::memory_order_relaxed);
            pending_.value.store(static_cast<long>(workers_.size()), std::memory_order_relaxed);
            // Sequentially consistent, as the sleeping count below: a worker that is
            // about to sleep either sees the new generation or is seen sleeping.
            generation_.fetch_add(1);
            if (sleeping_.load() > 0)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cv_.notify_all();
            }
            run_tasks(0);
            // Spin while the workers finish, then give the processor away: a worker that
            // the system preempted must be able to run on it.
            unsigned spins = 0;
            while (pending_.value.load(std::memory_order_acquire) != 0)
            {
                if (++spins < 4096) ALBIREO_CPU_RELAX();
                else std::this_thread::yield();
            }
            if (failed_.load(std::memory_order_acquire))
            {
                std::exception_ptr e;
                {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    e = error_;
                    error_ = nullptr;
                    failed_.store(false, std::memory_order_relaxed);
                }
                std::rethrow_exception(e);
            }
        }

    private:

        template <typename F>
        static void invoke(void* ctx, long task, int thread)
        {
            (*static_cast<F*>(ctx))(task, thread);
        }

        void run_tasks(int thread)
        {
            for (;;)
            {
                const long t = next_.value.fetch_add(1, std::memory_order_relaxed);
                if (t >= n_tasks_) break;
                try
                {
                    job_fn_(job_ctx_, t, thread);
                }
                catch (...)
                {
                    // The tasks not yet handed out are skipped; the first exception is kept for the caller.
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    if (!error_) error_ = std::current_exception();
                    failed_.store(true, std::memory_order_release);
                    next_.value.store(n_tasks_, std::memory_order_relaxed);
                }
            }
        }

        void worker_loop(int thread)
        {
            uint64_t seen = generation_.load(std::memory_order_acquire);
            for (;;)
            {
                // Spin for a while, then sleep until the next section.
                const auto spin_until = std::chrono::steady_clock::now()
                    + std::chrono::microseconds(spin_us_.load(std::memory_order_relaxed));
                uint64_t g = generation_.load(std::memory_order_acquire);
                unsigned polls = 0;
                while (g == seen)
                {
                    // After a short spin, yield now and then, so that a busy machine
                    // still runs the thread that will start the next section.
                    if (polls > 2048 && (polls & 63) == 0) std::this_thread::yield();
                    else ALBIREO_CPU_RELAX();
                    g = generation_.load(std::memory_order_acquire);
                    if (g != seen) break;
                    if ((++polls & 255) == 0 && std::chrono::steady_clock::now() > spin_until)
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        sleeping_.fetch_add(1);
                        cv_.wait(lock, [&]() {
                            return generation_.load() != seen
                                || stop_.load(std::memory_order_acquire);
                        });
                        sleeping_.fetch_sub(1, std::memory_order_acq_rel);
                        g = generation_.load(std::memory_order_acquire);
                        break;
                    }
                }
                seen = g;
                if (stop_.load(std::memory_order_acquire)) return;
                run_tasks(thread);
                pending_.value.fetch_sub(1, std::memory_order_acq_rel);
            }
        }

        // A counter alone on its cache line: the counters below change with every task
        // and every worker, while the spinning workers keep reading the fields of the
        // section.
        struct counter_line
        {
            char before[64];
            std::atomic<long> value{0};
            char after[64];
        };

        std::vector<std::thread> workers_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::atomic<uint64_t> generation_{0};
        std::atomic<int> sleeping_{0};
        std::atomic<long long> spin_us_{2000};
        long n_tasks_ = 0;
        void* job_ctx_ = nullptr;
        void (*job_fn_)(void*, long, int) = nullptr;
        std::atomic<bool> stop_{false};
        std::atomic<bool> failed_{false};
        std::mutex error_mutex_;
        std::exception_ptr error_;
        counter_line next_;
        counter_line pending_;
    };

}

#endif // ALBIREO_ENGINE_THREAD_POOL_H_
