// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the draws of the sampler with drafts, as the verification of the drafts of the
    answers uses them (speculative decoding).

        test_sampler

    The tokens drawn with a draft must follow the distribution of the draws without one,
    whatever the draft: a likely token, an unlikely one, or a token the filters leave out,
    which a draw then never returns.  At a temperature of 0 the draft changes nothing.  The
    tokens given as pending count for the penalties as the tokens accept() records.  The
    program returns 0 when every check passes.
*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../engine/sampler.h"

using namespace albireo;

namespace
{
    int failures = 0;

    void check(bool ok, const std::string& what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    // The frequencies of the tokens of n draws from logits, with draft (-1 for none).
    std::vector<double> frequencies(const sampling_params& p, const std::vector<float>& logits, int draft, long n)
    {
        sampler s(p);
        const long vocab = static_cast<long>(logits.size());
        std::vector<double> f(logits.size(), 0.0);
        std::vector<float> row;
        for (long i = 0; i < n; ++i)
        {
            row = logits;
            const int t = s.sample(row.data(), vocab, draft);
            if (t >= 0 && t < vocab) f[static_cast<size_t>(t)] += 1.0 / static_cast<double>(n);
        }
        return f;
    }

    void test_distribution()
    {
        std::mt19937 rng(17);
        std::normal_distribution<float> value(0.0f, 1.5f);
        const long vocab = 40;
        std::vector<float> logits(static_cast<size_t>(vocab));
        for (auto& l : logits) l = value(rng);
        sampling_params p;
        p.temperature = 0.8f;
        p.top_k = 12;
        p.top_p = 0.9f;
        p.min_p = 0.0f;
        p.seed = 1234;
        const long n = 200000;
        const std::vector<double> reference = frequencies(p, logits, -1, n);

        // The candidates in the order of their logits: the most likely, one at the edge of the
        // filters, and one they leave out.
        std::vector<int> order(static_cast<size_t>(vocab));
        for (long i = 0; i < vocab; ++i) order[static_cast<size_t>(i)] = static_cast<int>(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
        });
        int kept_last = order[0];
        for (int t : order)
            if (reference[static_cast<size_t>(t)] > 0) kept_last = t;
        const int left_out = order[static_cast<size_t>(vocab - 1)];
        check(reference[static_cast<size_t>(left_out)] == 0.0, "the filters leave the least likely token out");

        double worst = 0;
        for (int draft : { order[0], order[1], kept_last, left_out })
        {
            p.seed = 99 + static_cast<uint64_t>(draft);
            const std::vector<double> f = frequencies(p, logits, draft, n);
            for (long t = 0; t < vocab; ++t)
            {
                const double r = reference[static_cast<size_t>(t)];
                // Two estimates of n draws each: a standard deviation of sqrt(2 r (1 - r) / n).
                const double sigma = std::sqrt(2.0 * std::max(r, 1.0 / n) * (1.0 - r) / static_cast<double>(n));
                const double gap = std::fabs(f[static_cast<size_t>(t)] - r);
                worst = std::max(worst, gap / sigma);
                check(gap <= 5.0 * sigma, "draft " + std::to_string(draft) + ", token " + std::to_string(t)
                    + ": frequency " + std::to_string(f[static_cast<size_t>(t)]) + " against " + std::to_string(r));
            }
            check(f[static_cast<size_t>(left_out)] == 0.0, "a draft the filters leave out is never returned");
        }
        std::printf("  draws with a draft: the distribution without one, within %.1f standard deviations\n", worst);
    }

    void test_greedy()
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> value(0.0f, 2.0f);
        sampling_params p;
        p.temperature = 0.0f;
        sampler s(p);
        long same = 0;
        for (int trial = 0; trial < 1000; ++trial)
        {
            std::vector<float> logits(64);
            for (auto& l : logits) l = value(rng);
            const int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::vector<float> row = logits;
            const int t = s.sample(row.data(), 64, static_cast<int>(rng() % 64));
            same += t == best ? 1 : 0;
        }
        check(same == 1000, "at a temperature of 0, the most likely token whatever the draft");
        std::printf("  temperature 0: the most likely token whatever the draft\n");
    }

    void test_pending()
    {
        std::mt19937 rng(11);
        std::normal_distribution<float> value(0.0f, 1.0f);
        const long vocab = 30;
        for (int last_n : { 64, 3, -1 })
        {
            sampling_params p;
            p.temperature = 0.9f;
            p.top_k = 0;
            p.top_p = 1.0f;
            p.min_p = 0.0f;
            p.repeat_penalty = 1.3f;
            p.presence_penalty = 0.4f;
            p.frequency_penalty = 0.2f;
            p.repeat_last_n = last_n;
            p.seed = 7;
            sampler accepted(p), pending(p);
            long same = 0;
            const int trials = 2000;
            std::vector<int> history;
            for (int trial = 0; trial < trials; ++trial)
            {
                std::vector<float> logits(static_cast<size_t>(vocab));
                for (auto& l : logits) l = value(rng);
                // The tokens drawn before this one in a verification.
                const int a = static_cast<int>(rng() % vocab), b = static_cast<int>(rng() % vocab);
                accepted.accept(a);
                accepted.accept(b);
                const int ahead[2] = { a, b };
                std::vector<float> r1 = logits, r2 = logits;
                const int t1 = accepted.sample(r1.data(), vocab);
                const int t2 = pending.sample(r2.data(), vocab, -1, ahead, 2);
                same += t1 == t2 ? 1 : 0;
                pending.accept(a);
                pending.accept(b);
            }
            check(same == trials, "pending tokens count as accepted ones (repeat_last_n " + std::to_string(last_n)
                + "): " + std::to_string(same) + " of " + std::to_string(trials));
        }
        std::printf("  pending tokens: the penalties of the tokens accepted\n");
    }
}

int main()
{
    std::printf("albireo sampler\n");
    test_distribution();
    test_greedy();
    test_pending();
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
