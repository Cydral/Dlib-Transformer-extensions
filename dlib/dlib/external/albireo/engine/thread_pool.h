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
#include <stdexcept>
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
                parallel sections, so what matters is the cost of one: the workers spin on
                an atomic counter for a short while after each section and only then go to
                sleep, which keeps the hand-over well under a microsecond while tokens are
                produced and lets the processor rest when the engine is idle.

                parallel_for() splits a range of tasks among the calling thread and the
                workers, which take them one at a time from a shared ticket so that uneven
                tasks balance themselves, and returns once every task is done.  A section
                ends with its tasks: the calling thread waits for the tasks the workers
                took, and for no worker that took none, so that a section of a few tasks
                costs what these tasks cost whatever the number of threads, and a worker
                the system keeps away from its processor delays the tasks it holds alone.
                The callers size their tasks by their work (qk_matmul()), a product too
                small to share running whole on the calling thread.

                The pool runs one section at a time: a thread that calls parallel_for()
                while another thread's section runs waits until that section ends.  A task
                must not open a section of its own pool.  An exception thrown by a task
                skips the tasks not yet started and reaches the caller once the tasks
                under way are done.

            HOW A SECTION RUNS
                The ticket holds the number of tasks of the section and the next task to
                hand out in one atomic word.  A thread takes a task by moving the ticket
                from (next, tasks) to (next + 1, tasks) when next < tasks: the exchange
                succeeds on the ticket of the section under way alone, whose function the
                thread then reads, since that section cannot end before the task is done.
                A worker that comes late finds a ticket without a task left, or the ticket
                of a later section, whose tasks it may take as well, and writes nothing in
                the first case.  Each thread adds the tasks it has done to a counter once
                it finds no task left, and the calling thread leaves when the counter
                reaches the tasks it did not do itself.
        !*/

    public:

        explicit thread_pool(int threads)
        {
            const int n = threads > 0 ? threads : 1;
            /*
                Every worker starts from generation 0, the generation of a pool without any
                section, so that a worker takes part in a section opened before the system
                first runs it.  A thread that cannot be started stops the others.
            */
            try
            {
                workers_.reserve(static_cast<size_t>(n - 1));
                for (int i = 1; i < n; ++i)
                    workers_.emplace_back([this, i]() { worker_loop(i, 0); });
            }
            catch (...)
            {
                stop_workers();
                throw;
            }
        }

        ~thread_pool()
        {
            stop_workers();
        }

        thread_pool(const thread_pool&) = delete;
        thread_pool& operator=(const thread_pool&) = delete;

        int size() const { return static_cast<int>(workers_.size()) + 1; }

        // How long the workers spin after a section before sleeping.
        void set_spin_time(std::chrono::microseconds t) { spin_us_.store(t.count(), std::memory_order_relaxed); }

        /*!
            requires
                - n_tasks <= max_tasks
            ensures
                - Calls fn(task, thread_index) once for every task in [0, n_tasks),
                  thread_index being in [0, size()), and returns when all calls returned.
                - Runs everything on the calling thread when n_tasks == 1 or the pool
                  has a single thread.
                - Rethrows the first exception a task threw, once the tasks under way are done.
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
            if (n_tasks > max_tasks) throw std::length_error("thread_pool: too many tasks in a section");
            // The fields of the section below belong to one caller at a time.
            std::lock_guard<std::mutex> serial(section_mutex_);
            job_.ctx = static_cast<void*>(&fn);
            job_.fn = &invoke<typename std::remove_reference<F>::type>;
            done_.value.store(0, std::memory_order_relaxed);
            // The ticket publishes the function above to the threads that take a task from it.
            ticket_.value.store(static_cast<uint64_t>(n_tasks) << 32, std::memory_order_release);
            // Sequentially consistent, as the sleeping count below: a worker that is
            // about to sleep either sees the new generation or is seen sleeping.
            wake_.generation.fetch_add(1);
            if (wake_.sleeping.load() > 0)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cv_.notify_all();
            }
            const long own = run_tasks(0);
            /*
                Spin while the workers finish their tasks, then give the processor away: a
                worker that the system preempted must be able to run on it.
            */
            unsigned spins = 0;
            while (done_.value.load(std::memory_order_acquire) + own != n_tasks)
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

        // The tasks of a section, at most.
        static const long max_tasks = 0x7fffffffL;

    private:

        template <typename F>
        static void invoke(void* ctx, long task, int thread)
        {
            (*static_cast<F*>(ctx))(task, thread);
        }

        /*
            Takes the tasks left of the section under way, one at a time, and returns the
            tasks this thread has done or skipped.
        */
        long run_tasks(int thread)
        {
            long count = 0;
            uint64_t t = ticket_.value.load(std::memory_order_acquire);
            for (;;)
            {
                const uint32_t next = static_cast<uint32_t>(t);
                const uint32_t tasks = static_cast<uint32_t>(t >> 32);
                if (next >= tasks) break;
                if (!ticket_.value.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel,
                    std::memory_order_acquire))
                    continue;
                // The section of this task stays open until the task is counted.
                try
                {
                    job_.fn(job_.ctx, static_cast<long>(next), thread);
                }
                catch (...)
                {
                    // The first exception is kept for the caller, and the tasks not yet handed out are skipped.
                    {
                        std::lock_guard<std::mutex> lock(error_mutex_);
                        if (!error_) error_ = std::current_exception();
                        failed_.store(true, std::memory_order_release);
                    }
                    uint64_t open = ticket_.value.load(std::memory_order_acquire);
                    for (;;)
                    {
                        const uint32_t at = static_cast<uint32_t>(open);
                        const uint32_t all = static_cast<uint32_t>(open >> 32);
                        if (at >= all) break;
                        if (ticket_.value.compare_exchange_weak(open, (static_cast<uint64_t>(all) << 32) | all,
                            std::memory_order_acq_rel, std::memory_order_acquire))
                        {
                            count += static_cast<long>(all - at);
                            break;
                        }
                    }
                }
                ++count;
                t = ticket_.value.load(std::memory_order_acquire);
            }
            return count;
        }

        void stop_workers()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_.store(true, std::memory_order_release);
                wake_.generation.fetch_add(1, std::memory_order_release);
            }
            cv_.notify_all();
            for (auto& t : workers_) t.join();
        }

        void worker_loop(int thread, uint64_t seen)
        {
            for (;;)
            {
                // Spin for a while, then sleep until the next section.
                const auto spin_until = std::chrono::steady_clock::now()
                    + std::chrono::microseconds(spin_us_.load(std::memory_order_relaxed));
                uint64_t g = wake_.generation.load(std::memory_order_acquire);
                unsigned polls = 0;
                while (g == seen)
                {
                    // After a short spin, yield now and then, so that a busy machine
                    // still runs the thread that will start the next section.
                    if (polls > 2048 && (polls & 63) == 0) std::this_thread::yield();
                    else ALBIREO_CPU_RELAX();
                    g = wake_.generation.load(std::memory_order_acquire);
                    if (g != seen) break;
                    if ((++polls & 255) == 0 && std::chrono::steady_clock::now() > spin_until)
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        wake_.sleeping.fetch_add(1);
                        cv_.wait(lock, [&]() {
                            return wake_.generation.load() != seen
                                || stop_.load(std::memory_order_acquire);
                        });
                        wake_.sleeping.fetch_sub(1, std::memory_order_acq_rel);
                        g = wake_.generation.load(std::memory_order_acquire);
                        break;
                    }
                }
                seen = g;
                if (stop_.load(std::memory_order_acquire)) return;
                const long count = run_tasks(thread);
                if (count > 0) done_.value.fetch_add(count, std::memory_order_acq_rel);
            }
        }

        /*
            The fields the threads share, each group alone on its cache lines: the spinning
            workers read the generation, which changes once per section; the threads that
            take tasks write the ticket; those that did some write the counter of the tasks
            done; and the function of the section is written by its caller and read by the
            threads that took a task.
        */
        struct wake_line
        {
            char before[64];
            std::atomic<uint64_t> generation{0};
            std::atomic<int> sleeping{0};
            char after[64];
        };

        struct ticket_line
        {
            char before[64];
            // The tasks of the section in the high half, the next task to hand out in the low half.
            std::atomic<uint64_t> value{0};
            char after[64];
        };

        struct counter_line
        {
            char before[64];
            std::atomic<long> value{0};
            char after[64];
        };

        struct job_line
        {
            char before[64];
            void* ctx = nullptr;
            void (*fn)(void*, long, int) = nullptr;
            char after[64];
        };

        std::vector<std::thread> workers_;
        // Held by the thread whose section runs, for the whole section.
        std::mutex section_mutex_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::atomic<long long> spin_us_{2000};
        std::atomic<bool> stop_{false};
        std::atomic<bool> failed_{false};
        std::mutex error_mutex_;
        std::exception_ptr error_;
        wake_line wake_;
        ticket_line ticket_;
        counter_line done_;
        job_line job_;
    };

}

#endif // ALBIREO_ENGINE_THREAD_POOL_H_
