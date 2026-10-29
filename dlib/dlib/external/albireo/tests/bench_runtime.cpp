// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Measures the processor runtime on a model:

        bench_runtime model.gguf [threads] [prompt_tokens] [steps] [parallel]

    Reports the prompt throughput, the time of one generation step for a single
    sequence, and the throughput of steps that advance several sequences together.
*/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "../engine/cpu_runtime.h"

using namespace albireo;
using clock_type = std::chrono::steady_clock;

static double since(clock_type::time_point t)
{
    return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: bench_runtime model.gguf [threads] [prompt_tokens] [steps] [parallel]\n");
        return 2;
    }
    const int threads = argc > 2 ? std::atoi(argv[2]) : static_cast<int>(std::thread::hardware_concurrency());
    const long prompt = argc > 3 ? std::atol(argv[3]) : 128;
    const long steps = argc > 4 ? std::atol(argv[4]) : 32;
    const long parallel = argc > 5 ? std::atol(argv[5]) : 4;

    auto file = std::make_shared<model_file>(argv[1]);
    const model_config cfg = read_model_config(*file);
    thread_pool pool(threads);
    cpu_runtime rt(file, cfg, pool, size_t(1) << 30);
    std::printf("%s\n", rt.description().c_str());
    file->warm_up(false);

    // Prompt.
    std::vector<int> tokens(static_cast<size_t>(prompt));
    for (long i = 0; i < prompt; ++i) tokens[static_cast<size_t>(i)] = static_cast<int>((i * 7919 + 13) % cfg.vocab);
    const int seq = rt.create_sequence();
    rt.reserve_sequence(seq, prompt + steps);
    forward_batch b;
    b.tokens = tokens;
    b.segments.push_back({ seq, 0, prompt });
    b.logit_rows.push_back(prompt - 1);
    std::vector<float> logits;
    auto t = clock_type::now();
    rt.forward(b, logits);
    const double pms = since(t);
    std::printf("prompt: %ld tokens in %.1f ms, %.1f tokens/s\n", prompt, pms, 1000.0 * prompt / pms);

    // Single sequence steps.
    t = clock_type::now();
    for (long s = 0; s < steps; ++s)
    {
        b.clear();
        b.tokens.push_back(static_cast<int>((s * 31 + 7) % cfg.vocab));
        b.segments.push_back({ seq, 0, 1 });
        b.logit_rows.push_back(0);
        rt.forward(b, logits);
    }
    const double sms = since(t) / steps;
    std::printf("generation: %.2f ms per token, %.1f tokens/s\n", sms, 1000.0 / sms);
    rt.release_sequence(seq);

    // Several sequences advancing together.
    std::vector<int> seqs;
    for (long p = 0; p < parallel; ++p)
    {
        const int s = rt.create_sequence();
        rt.reserve_sequence(s, 64 + steps);
        b.clear();
        b.tokens.assign(tokens.begin(), tokens.begin() + 64);
        b.segments.push_back({ s, 0, 64 });
        b.logit_rows.push_back(63);
        rt.forward(b, logits);
        seqs.push_back(s);
    }
    t = clock_type::now();
    for (long s = 0; s < steps; ++s)
    {
        b.clear();
        for (long p = 0; p < parallel; ++p)
        {
            b.tokens.push_back(static_cast<int>((s * 31 + p) % cfg.vocab));
            b.segments.push_back({ seqs[static_cast<size_t>(p)], p, 1 });
            b.logit_rows.push_back(p);
        }
        rt.forward(b, logits);
    }
    const double bms = since(t) / steps;
    std::printf("batched generation, %ld sequences: %.2f ms per step, %.1f tokens/s in total\n",
        parallel, bms, 1000.0 * parallel / bms);
    return 0;
}
