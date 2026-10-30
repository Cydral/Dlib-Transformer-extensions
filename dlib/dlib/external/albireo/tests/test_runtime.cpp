// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the processor runtime on a real model.

        test_runtime model.gguf reference.bin [exact] id id id ...

    reference.bin holds the logits of every position of the token sequence computed by
    tests/reference_forward.py.  The program compares them with the logits of the
    runtime, then checks that the runtime gives the same results however the work is
    cut: the whole sequence in one pass, a prompt followed by one token at a time, two
    sequences sharing a batch, and a sequence truncated and extended again.

    With "exact", the products run on floats (qk_set_exact_products) and the comparison
    with the reference is tight; otherwise the activations are quantized as in normal
    operation and the comparison bears on the distributions.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../engine/cpu_runtime.h"

using namespace albireo;

namespace
{
    int failures = 0;

    struct stats
    {
        double max_abs = 0;     // largest logit difference
        double scale = 0;       // largest reference logit magnitude
        double mean_kl = 0;     // mean KL(reference || runtime) over positions
        double top1 = 0;        // fraction of positions with the same argmax
    };

    stats compare(const float* ref, const float* got, long rows, long vocab)
    {
        stats s;
        long same = 0;
        for (long r = 0; r < rows; ++r)
        {
            const float* a = ref + r * vocab;
            const float* b = got + r * vocab;
            double ma = -1e30, mb = -1e30;
            long ia = 0, ib = 0;
            for (long v = 0; v < vocab; ++v)
            {
                s.max_abs = std::max(s.max_abs, static_cast<double>(std::fabs(a[v] - b[v])));
                s.scale = std::max(s.scale, static_cast<double>(std::fabs(a[v])));
                if (a[v] > ma) { ma = a[v]; ia = v; }
                if (b[v] > mb) { mb = b[v]; ib = v; }
            }
            if (ia == ib) ++same;
            double za = 0, zb = 0;
            for (long v = 0; v < vocab; ++v)
            {
                za += std::exp(a[v] - ma);
                zb += std::exp(b[v] - mb);
            }
            double kl = 0;
            for (long v = 0; v < vocab; ++v)
            {
                const double pa = std::exp(a[v] - ma) / za;
                const double lpa = a[v] - ma - std::log(za);
                const double lpb = b[v] - mb - std::log(zb);
                kl += pa * (lpa - lpb);
            }
            s.mean_kl += kl;
        }
        s.mean_kl /= rows;
        s.top1 = static_cast<double>(same) / rows;
        return s;
    }

    void report(const char* what, const stats& s)
    {
        std::printf("  %-34s max |diff| %.3g (logits up to %.3g), mean KL %.3g, top-1 %.1f%%\n",
            what, s.max_abs, s.scale, s.mean_kl, 100.0 * s.top1);
    }

    void expect(bool ok, const char* what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what);
        }
    }

    std::vector<float> run_whole(cpu_runtime& rt, const std::vector<int>& tokens)
    {
        const int seq = rt.create_sequence();
        expect(rt.reserve_sequence(seq, static_cast<long>(tokens.size())), "reserve");
        forward_batch b;
        b.tokens = tokens;
        b.segments.push_back({ seq, 0, static_cast<long>(tokens.size()) });
        for (long i = 0; i < static_cast<long>(tokens.size()); ++i) b.logit_rows.push_back(i);
        std::vector<float> logits;
        rt.forward(b, logits);
        expect(rt.sequence_length(seq) == static_cast<long>(tokens.size()), "sequence length after a pass");
        rt.release_sequence(seq);
        return logits;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: test_runtime model.gguf reference.bin [exact] id...\n");
        return 2;
    }
    int arg = 3;
    bool exact = false;
    if (std::string(argv[arg]) == "exact")
    {
        exact = true;
        ++arg;
    }
    std::vector<int> tokens;
    for (; arg < argc; ++arg) tokens.push_back(std::atoi(argv[arg]));
    const long T = static_cast<long>(tokens.size());

    auto file = std::make_shared<model_file>(argv[1]);
    const model_config cfg = read_model_config(*file);
    std::printf("%s\n", describe(cfg).c_str());
    const long V = cfg.vocab;

    std::vector<float> ref(static_cast<size_t>(T * V));
    {
        std::ifstream in(argv[2], std::ios::binary);
        in.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * sizeof(float)));
        if (!in)
        {
            std::printf("cannot read %ld x %ld logits from %s\n", T, V, argv[2]);
            return 2;
        }
    }

    thread_pool pool(2);
    // Small blocks, so that every sequence spans several of them.
    cpu_runtime rt(file, cfg, pool, size_t(512) << 20, 16);
    std::printf("%s\n", rt.description().c_str());
    qk_set_exact_products(exact);

    // 1. The whole sequence in one pass, against the reference.
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> whole = run_whole(rt, tokens);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const stats s = compare(ref.data(), whole.data(), T, V);
    report(exact ? "reference (exact products)" : "reference (quantized activations)", s);
    std::printf("  one pass of %ld tokens: %.1f ms\n", T, ms);
    if (exact)
    {
        // Keys and values are stored in half precision, the only rounding left.
        expect(s.mean_kl < 1e-4, "exact distribution");
        expect(s.max_abs <= 0.1 * std::max(1.0, s.scale / 10), "exact logits");
        expect(s.top1 >= 0.99, "exact argmax");
    }
    else
    {
        // 8-bit activations, as every integer kernel of the GGUF ecosystem uses them.
        expect(s.mean_kl < 0.05, "distribution against the reference");
        expect(s.top1 >= 0.85, "argmax against the reference");
    }

    // 2. A prompt, then one token at a time.
    {
        const long k = std::max<long>(1, T / 3);
        const int seq = rt.create_sequence();
        std::vector<float> got(static_cast<size_t>(T * V));
        std::vector<float> logits;
        forward_batch b;
        expect(rt.reserve_sequence(seq, k), "reserve prompt");
        b.tokens.assign(tokens.begin(), tokens.begin() + k);
        b.segments.push_back({ seq, 0, k });
        for (long i = 0; i < k; ++i) b.logit_rows.push_back(i);
        rt.forward(b, logits);
        std::copy(logits.begin(), logits.end(), got.begin());
        for (long i = k; i < T; ++i)
        {
            expect(rt.reserve_sequence(seq, i + 1), "reserve step");
            b.clear();
            b.tokens.push_back(tokens[static_cast<size_t>(i)]);
            b.segments.push_back({ seq, 0, 1 });
            b.logit_rows.push_back(0);
            rt.forward(b, logits);
            std::copy(logits.begin(), logits.end(), got.begin() + i * V);
        }
        // The prompt goes through the float products and the single steps through the
        // integer kernels, so the two passes agree as distributions, not bit for bit.
        const stats d = compare(whole.data(), got.data(), T, V);
        report("prompt then single steps", d);
        if (exact)
            expect(d.max_abs <= 1e-3 * std::max(1.0, d.scale), "single steps against one pass");
        else
            expect(d.mean_kl < 0.05 && d.top1 >= 0.85, "single steps against one pass");

        // 4. Truncation: drop the last third and compute it again.
        const long m = std::max<long>(1, 2 * T / 3);
        rt.truncate_sequence(seq, m);
        expect(rt.sequence_length(seq) == m, "length after truncation");
        // The blocks beyond the kept positions went back to the pool.
        expect(rt.reserve_sequence(seq, T), "reserve after truncation");
        b.clear();
        b.tokens.assign(tokens.begin() + m, tokens.end());
        b.segments.push_back({ seq, 0, T - m });
        for (long i = 0; i < T - m; ++i) b.logit_rows.push_back(i);
        rt.forward(b, logits);
        const stats r = compare(whole.data() + m * V, logits.data(), T - m, V);
        report("truncated and extended", r);
        if (exact)
            expect(r.max_abs <= 1e-3 * std::max(1.0, r.scale), "truncated sequence");
        else
            expect(r.mean_kl < 0.05 && r.top1 >= 0.85, "truncated sequence");
        rt.release_sequence(seq);
    }

    // 3. Two sequences in one batch: the whole sequence, and its second half as a
    //    sequence of its own that starts at position 0.
    {
        const long h = T / 2;
        std::vector<int> second(tokens.begin() + h, tokens.end());
        const std::vector<float> alone = run_whole(rt, second);

        const int a = rt.create_sequence();
        const int c = rt.create_sequence();
        expect(rt.reserve_sequence(a, T) && rt.reserve_sequence(c, T - h), "reserve batch");
        forward_batch b;
        b.tokens = second;
        b.tokens.insert(b.tokens.end(), tokens.begin(), tokens.end());
        b.segments.push_back({ c, 0, T - h });
        b.segments.push_back({ a, T - h, T });
        for (long i = 0; i < static_cast<long>(b.tokens.size()); ++i) b.logit_rows.push_back(i);
        std::vector<float> logits;
        rt.forward(b, logits);
        const stats d1 = compare(alone.data(), logits.data(), T - h, V);
        const stats d2 = compare(whole.data(), logits.data() + (T - h) * V, T, V);
        report("batched, short sequence", d1);
        report("batched, long sequence", d2);
        expect(d1.max_abs <= 1e-3 * std::max(1.0, d1.scale), "batched short sequence");
        expect(d2.max_abs <= 1e-3 * std::max(1.0, d2.scale), "batched long sequence");
        rt.release_sequence(a);
        rt.release_sequence(c);
        expect(rt.kv_bytes_used() == 0, "every block returned");
    }

    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
