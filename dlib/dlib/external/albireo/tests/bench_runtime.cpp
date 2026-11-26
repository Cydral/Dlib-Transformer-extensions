// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Measures the runtime of a model as llama-bench of llama.cpp measures its own, so that the
    two programs give figures of the same kind:

        bench_runtime model.gguf [options]

    The prompt test (pp) reads tokens drawn at random in passes of the size of the engine,
    with the logits of the last token of each pass; the generation test (tg) writes tokens one
    pass each, with their logits, the token of a pass being drawn at random too.  Each test
    starts from a sequence that holds the positions of --depth alone, runs once unmeasured
    (the whole prompt, one generated token), then --repetitions times; its line gives the mean
    of the tokens per second of the repetitions and their standard deviation.  The times
    hold neither a tokenization nor a sampling.  The test of --parallel advances several
    sequences in each pass, each after the first tokens of the prompt, 64 at most.

    The program computes with the processor runtime and the kernels it was built with
    (bench_runtime, bench_runtime_avxvnni).  Built with a device of the tensors of dlib
    (bench_offload), it hands the products and the layers to the device as the engine of that
    device does, and its unmeasured runs hold the checks the runtime makes before it uses a
    device.

    Options:
        --threads N[,N...]   threads of the processor, a test series for each number; the
                             physical cores by default, as in the engine
        --prompt N           tokens of the prompt test, 512 by default, 0 for no such test
        --generate N         tokens of the generation test, 128 by default, 0 for none
        --repetitions N      measured runs of each test, 5 by default
        --batch N            tokens a pass of the prompt reads at most, 512 by default
        --depth N            positions the sequence holds before each test, 0 by default
        --parallel N         sequences of the test of the passes of several sequences, none
                             by default
        --kv f16|q8          format of the keys and values; by default q8 for a dense decoder
                             whose heads hold groups of 32 values, as in the engine, else f16
        --floats             products of the prompts on floats (qk_set_integer_prompt())
        --spin MICROSECONDS  how long the threads wait for the next parallel section before
                             they sleep, 2000 by default
        --no-warmup          measures from the first run
        --routing MODE       auto, device or host (device builds)
        --device NAME        device of the OpenVINO build or number of the card (device builds)
        --offload-mb N       megabytes the copies of the weights take at most (device builds)
        --precision MODE     arithmetic of the products of the longer passes of a card: auto, the
                             fastest one the device times, single, tf32 or half (device builds)
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/system_info.h"
#include "../engine/cpu_runtime.h"
#ifdef ALBIREO_TENSOR_BACKEND
#include "../common/engine_api.h"
#include "../engine/tensor_backend.h"
#endif

using namespace albireo;
using clock_type = std::chrono::steady_clock;

namespace
{
    struct options
    {
        std::string model;
        std::vector<int> threads;
        long prompt = 512;
        long generate = 128;
        long repetitions = 5;
        long batch = 512;
        long depth = 0;
        long parallel = 0;
        std::string kv = "auto";
        bool floats = false;
        long spin_us = 2000;
        bool warmup = true;
        std::string routing = "auto";
        std::string device;
        long offload_mb = 0;
        std::string precision = "auto";
    };

    double seconds_since(clock_type::time_point t)
    {
        return std::chrono::duration<double>(clock_type::now() - t).count();
    }

    // The tokens of the tests: the same ones on every system, which std::rand() would not give.
    struct token_source
    {
        explicit token_source(long vocab_size, uint64_t seed) : vocab(vocab_size), state(seed) {}

        int next()
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<int>((state >> 33) % static_cast<uint64_t>(vocab));
        }

        long vocab;
        uint64_t state;
    };

    bool read_long(const char* text, long low, long high, long& value)
    {
        char* end = nullptr;
        const long v = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || v < low || v > high) return false;
        value = v;
        return true;
    }

    int usage()
    {
        std::printf("usage: bench_runtime model.gguf [--threads N[,N...]] [--prompt N] [--generate N] "
            "[--repetitions N]\n"
            "       [--batch N] [--depth N] [--parallel N] [--kv f16|q8] [--floats] [--spin MICROSECONDS] "
            "[--no-warmup]\n"
            "       [--routing auto|device|host] [--device NAME] [--offload-mb N] "
            "[--precision auto|single|tf32|half]\n");
        return 2;
    }

    bool parse(int argc, char** argv, options& o)
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string a(argv[i]);
            const bool has_value = i + 1 < argc;
            if (a == "--floats") o.floats = true;
            else if (a == "--no-warmup") o.warmup = false;
            else if (a == "--threads" && has_value)
            {
                const std::string list(argv[++i]);
                size_t at = 0;
                while (at <= list.size())
                {
                    const size_t comma = std::min(list.find(',', at), list.size());
                    long n = 0;
                    if (!read_long(list.substr(at, comma - at).c_str(), 1, 4096, n)) return false;
                    o.threads.push_back(static_cast<int>(n));
                    at = comma + 1;
                }
            }
            else if (a == "--prompt" && has_value) { if (!read_long(argv[++i], 0, 1L << 20, o.prompt)) return false; }
            else if (a == "--generate" && has_value)
            {
                if (!read_long(argv[++i], 0, 1L << 20, o.generate)) return false;
            }
            else if (a == "--repetitions" && has_value)
            {
                if (!read_long(argv[++i], 1, 1000, o.repetitions)) return false;
            }
            else if (a == "--batch" && has_value) { if (!read_long(argv[++i], 1, 65536, o.batch)) return false; }
            else if (a == "--depth" && has_value) { if (!read_long(argv[++i], 0, 1L << 20, o.depth)) return false; }
            else if (a == "--parallel" && has_value) { if (!read_long(argv[++i], 0, 1024, o.parallel)) return false; }
            else if (a == "--spin" && has_value) { if (!read_long(argv[++i], 0, 100000, o.spin_us)) return false; }
            else if (a == "--offload-mb" && has_value)
            {
                if (!read_long(argv[++i], 0, 1L << 20, o.offload_mb)) return false;
            }
            else if (a == "--kv" && has_value)
            {
                o.kv = argv[++i];
                if (o.kv != "f16" && o.kv != "q8") return false;
            }
            else if (a == "--routing" && has_value)
            {
                o.routing = argv[++i];
                if (o.routing != "auto" && o.routing != "device" && o.routing != "host") return false;
            }
            else if (a == "--device" && has_value) o.device = argv[++i];
            else if (a == "--precision" && has_value)
            {
                o.precision = argv[++i];
                if (o.precision != "auto" && o.precision != "single" && o.precision != "tf32" && o.precision != "half")
                    return false;
            }
            else if (!a.empty() && a[0] != '-' && o.model.empty()) o.model = a;
            else return false;
        }
        return !o.model.empty();
    }

    struct sample_stats
    {
        double mean = 0;
        double deviation = 0;
    };

    // The mean of the samples and their standard deviation, as llama-bench computes them.
    sample_stats summarize(const std::vector<double>& samples)
    {
        sample_stats s;
        if (samples.empty()) return s;
        for (double v : samples) s.mean += v;
        s.mean /= static_cast<double>(samples.size());
        if (samples.size() > 1)
        {
            double squares = 0;
            for (double v : samples) squares += (v - s.mean) * (v - s.mean);
            s.deviation = std::sqrt(squares / static_cast<double>(samples.size() - 1));
        }
        return s;
    }

    void print_row(int threads, const std::string& test, const std::vector<double>& samples)
    {
        const sample_stats s = summarize(samples);
        char rate[64];
        std::snprintf(rate, sizeof(rate), "%.2f +/- %.2f", s.mean, s.deviation);
        std::printf("| %7d | %10s | %20s |\n", threads, test.c_str(), rate);
        std::fflush(stdout);
    }

    class bench
    {
    public:
        bench(cpu_runtime& rt, const options& o) : rt_(rt), o_(o), tokens_(rt.config().vocab, 0x9E3779B97F4A7C15ULL)
        {
        }

        /*
            A sequence that holds the positions of the depth: computed once in a sequence of its
            own, then copied when the runtime can give a sequence the positions of another, a
            runtime whose layers keep a state reading them again otherwise.
        */
        bool start(int& seq, long more)
        {
            seq = rt_.create_sequence();
            if (!rt_.reserve_sequence(seq, o_.depth + more)) return refuse(o_.depth + more);
            if (o_.depth == 0) return true;
            if (base_ < 0)
            {
                base_ = rt_.create_sequence();
                if (!rt_.reserve_sequence(base_, o_.depth)) return refuse(o_.depth);
                read(base_, o_.depth);
                copies_ = rt_.restorable_length(base_, o_.depth) == o_.depth;
            }
            if (copies_ && rt_.copy_sequence(base_, seq, o_.depth)) return true;
            read(seq, o_.depth);
            return true;
        }

        // Reads count tokens into seq, a pass of the batch size at a time, with the logits of its last token.
        void read(int seq, long count)
        {
            for (long done = 0; done < count;)
            {
                const long n = std::min(o_.batch, count - done);
                batch_.clear();
                for (long i = 0; i < n; ++i) batch_.tokens.push_back(tokens_.next());
                forward_batch::segment s;
                s.sequence = seq;
                s.start = 0;
                s.count = n;
                s.prompt = true;
                batch_.segments.push_back(s);
                batch_.logit_rows.push_back(n - 1);
                rt_.forward(batch_, logits_);
                done += n;
            }
        }

        // Writes count tokens after seq, one pass each, with their logits.
        void write(int seq, long count)
        {
            for (long i = 0; i < count; ++i)
            {
                batch_.clear();
                batch_.tokens.push_back(tokens_.next());
                batch_.segments.push_back({ seq, 0, 1 });
                batch_.logit_rows.push_back(0);
                rt_.forward(batch_, logits_);
            }
        }

        bool prompt_test(int threads)
        {
            if (o_.prompt <= 0) return true;
            std::vector<double> rates;
            for (long r = o_.warmup ? -1 : 0; r < o_.repetitions; ++r)
            {
                int seq = -1;
                if (!start(seq, o_.prompt)) return false;
                const auto t = clock_type::now();
                read(seq, o_.prompt);
                const double s = seconds_since(t);
                rt_.release_sequence(seq);
                if (r < 0)
                    std::printf("unmeasured prompt run: %ld tokens in %.1f ms\n", o_.prompt, 1e3 * s);
                else
                    rates.push_back(static_cast<double>(o_.prompt) / s);
            }
            print_row(threads, "pp" + std::to_string(o_.prompt) + depth_name(), rates);
            return true;
        }

        bool generation_test(int threads)
        {
            if (o_.generate <= 0) return true;
            std::vector<double> rates;
            for (long r = o_.warmup ? -1 : 0; r < o_.repetitions; ++r)
            {
                const long count = r < 0 ? 1 : o_.generate;
                int seq = -1;
                if (!start(seq, count)) return false;
                const auto t = clock_type::now();
                write(seq, count);
                const double s = seconds_since(t);
                rt_.release_sequence(seq);
                if (r < 0)
                    std::printf("unmeasured generation run: 1 token in %.1f ms\n", 1e3 * s);
                else
                    rates.push_back(static_cast<double>(count) / s);
            }
            print_row(threads, "tg" + std::to_string(o_.generate) + depth_name(), rates);
            return true;
        }

        // Several sequences advancing together, each after the first tokens of a prompt.
        bool parallel_test(int threads)
        {
            if (o_.parallel <= 0 || o_.generate <= 0) return true;
            const long head = std::max<long>(1, std::min<long>(64, o_.prompt));
            std::vector<double> rates;
            for (long r = o_.warmup ? -1 : 0; r < o_.repetitions; ++r)
            {
                const long steps = r < 0 ? 1 : o_.generate;
                std::vector<int> seqs;
                for (long p = 0; p < o_.parallel; ++p)
                {
                    int seq = -1;
                    if (!start(seq, head + steps)) return false;
                    read(seq, head);
                    seqs.push_back(seq);
                }
                const auto t = clock_type::now();
                for (long s = 0; s < steps; ++s)
                {
                    batch_.clear();
                    for (long p = 0; p < o_.parallel; ++p)
                    {
                        batch_.tokens.push_back(tokens_.next());
                        batch_.segments.push_back({ seqs[static_cast<size_t>(p)], p, 1 });
                        batch_.logit_rows.push_back(p);
                    }
                    rt_.forward(batch_, logits_);
                }
                const double s = seconds_since(t);
                for (int seq : seqs) rt_.release_sequence(seq);
                if (r >= 0) rates.push_back(static_cast<double>(steps * o_.parallel) / s);
            }
            print_row(threads, "tg" + std::to_string(o_.generate) + "x" + std::to_string(o_.parallel) + depth_name(),
                rates);
            return true;
        }

    private:
        std::string depth_name() const
        {
            return o_.depth > 0 ? "@d" + std::to_string(o_.depth) : std::string();
        }

        bool refuse(long positions) const
        {
            std::printf("bench_runtime: the runtime cannot hold the keys and values of a sequence of %ld positions\n",
                positions);
            return false;
        }

        cpu_runtime& rt_;
        const options& o_;
        token_source tokens_;
        forward_batch batch_;
        std::vector<float> logits_;
        int base_ = -1;
        bool copies_ = false;
    };

#ifdef ALBIREO_TENSOR_BACKEND
    void print_log(int level, const std::string& message)
    {
        static const char* names[] = { "debug", "info", "warning", "error" };
        std::printf("[%s] %s\n", names[std::max(0, std::min(3, level))], message.c_str());
        std::fflush(stdout);
    }
#endif

    bool run(const options& o, const std::shared_ptr<model_file>& file, const model_config& cfg, int threads)
    {
        thread_pool pool(threads);
        pool.set_spin_time(std::chrono::microseconds(o.spin_us));
        // The format of the keys and values of the engine: 8 bits for the dense decoders whose heads hold whole groups.
        const bool whole_groups = cfg.head_dim % kv_q8_group == 0;
        const bool dense = !cfg.hybrid() && !cfg.is_embedding();
        const kv_format format = o.kv == "q8" || (o.kv == "auto" && whole_groups && dense) ? kv_format::q8
            : kv_format::f16;
        if (format == kv_format::q8 && !whole_groups)
        {
            std::printf("bench_runtime: heads of %ld values hold no whole groups of the 8-bit format\n", cfg.head_dim);
            return false;
        }
        /*
            The keys and values, which the runtime holds by pages of 256 positions: those of the
            depth, of the sequence of a test, and of the sequences that advance together.
        */
        const long page = 256;
        const long head = std::max<long>(1, std::min<long>(64, o.prompt));
        const long longest = o.depth + std::max(o.prompt, std::max<long>(o.generate, 1));
        const long layers = cfg.hybrid() ? cfg.attention_layers() : cfg.n_layers;
        const size_t per_token = 2 * static_cast<size_t>(layers) * static_cast<size_t>(cfg.n_kv_heads)
            * kv_row_bytes(format, cfg.head_dim);
        const size_t pages = static_cast<size_t>((o.depth + page - 1) / page)
            + static_cast<size_t>((longest + page - 1) / page)
            + static_cast<size_t>(o.parallel) * static_cast<size_t>((o.depth + head + o.generate + page - 1) / page)
            + 4;
        const size_t budget = std::max<size_t>(size_t(1) << 30, per_token * static_cast<size_t>(page) * pages);
#ifdef ALBIREO_TENSOR_BACKEND
        json config = json::object();
        if (!o.device.empty()) config["device"] = o.device;
        config["routing"] = o.routing;
        if (o.offload_mb > 0) config["offload_mb"] = o.offload_mb;
        config["gemm_precision"] = o.precision;
        // The graphs of a device compile on their first use, inside the unmeasured runs.
        config["benchmark"] = true;
        // The device outlives the runtime, whose offload it gave.
        tensor_device device(config, pool, print_log);
        if (!device.bind_thread())
        {
            std::printf("bench_runtime: the device of the build cannot be selected\n");
            return false;
        }
#endif
        cpu_runtime rt(file, cfg, pool, budget, page, format);
#ifdef ALBIREO_TENSOR_BACKEND
        if (o.routing != "host") rt.set_offload(device.make_offload(o.model));
#endif
        std::printf("%s\n", rt.description().c_str());
        std::printf("| threads |       test |                  t/s |\n");
        std::printf("| ------: | ---------: | -------------------: |\n");
        bench b(rt, o);
        const bool ok = b.prompt_test(threads) && b.generation_test(threads) && b.parallel_test(threads);
#ifdef ALBIREO_TENSOR_BACKEND
        std::printf("device: %s\n", device.stats().dump().c_str());
#endif
        return ok;
    }
}

int main(int argc, char** argv)
{
    options o;
    if (!parse(argc, argv, o)) return usage();
    if (o.threads.empty()) o.threads.push_back(std::max(1, read_cpu_info().physical_cores));
    if (o.floats) qk_set_integer_prompt(false);
    /*
        The thread that starts the sections computes with the workers, at the same quality of
        service: asked for the process, and for this thread should the system refuse the first.
    */
    const bool process_unthrottled = leave_process_throttling();
    const bool unthrottled = leave_thread_throttling() || process_unthrottled;
    try
    {
        auto file = std::make_shared<model_file>(o.model);
        const model_config cfg = read_model_config(*file);
#ifdef ALBIREO_TENSOR_BACKEND
        const std::string engine = std::string("device ") + tensor_device::backend_name() + ", routing " + o.routing;
#else
        const std::string engine = "processor";
#endif
        std::printf("bench_runtime: %s, %.2f GiB, %s, kernels %s, passes of %ld tokens, %ld repetitions%s\n",
            o.model.c_str(), static_cast<double>(file->size()) / static_cast<double>(size_t(1) << 30), engine.c_str(),
            qk_isa(), o.batch, o.repetitions, unthrottled ? ", high quality of service asked of the system" : "");
        file->warm_up(false);
        for (int threads : o.threads)
            if (!run(o, file, cfg, threads)) return 2;
    }
    catch (const std::exception& e)
    {
        std::printf("bench_runtime: %s\n", e.what());
        return 1;
    }
    return 0;
}
