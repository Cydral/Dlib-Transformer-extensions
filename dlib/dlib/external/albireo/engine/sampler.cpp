// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace albireo
{

    namespace
    {
        const long scan_block = 16;

        // Whether one of the scan_block values of x exceeds t; NaN exceeds nothing.
        inline bool block_above(const float* x, float t)
        {
#if defined(__AVX2__)
            const __m256 v = _mm256_set1_ps(t);
            const __m256 a = _mm256_cmp_ps(_mm256_loadu_ps(x), v, _CMP_GT_OQ);
            const __m256 b = _mm256_cmp_ps(_mm256_loadu_ps(x + 8), v, _CMP_GT_OQ);
            return _mm256_movemask_ps(_mm256_or_ps(a, b)) != 0;
#else
            bool above = false;
            for (long i = 0; i < scan_block; ++i) above |= x[i] > t;
            return above;
#endif
        }
    }

    sampler::sampler(const sampling_params& p) : p_(p)
    {
        uint64_t seed = p_.seed;
        if (seed == 0)
        {
            std::random_device rd;
            seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
        }
        rng_.seed(seed);
    }

    void sampler::accept(int token)
    {
        history_.push_back(token);
    }

    int sampler::sample(float* logits, long vocab)
    {
        for (const auto& b : p_.logit_bias)
            if (b.first >= 0 && b.first < vocab) logits[b.first] += b.second;

        // Penalties over the recent tokens.
        if (p_.repeat_penalty != 1.0f || p_.presence_penalty != 0.0f || p_.frequency_penalty != 0.0f)
        {
            std::unordered_map<int, int> counts;
            const size_t n = p_.repeat_last_n < 0 ? history_.size()
                : std::min(history_.size(), static_cast<size_t>(p_.repeat_last_n));
            for (size_t i = history_.size() - n; i < history_.size(); ++i) ++counts[history_[i]];
            for (const auto& c : counts)
            {
                if (c.first < 0 || c.first >= vocab) continue;
                float& l = logits[c.first];
                if (p_.repeat_penalty != 1.0f) l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty;
                l -= p_.frequency_penalty * static_cast<float>(c.second) + p_.presence_penalty;
            }
        }

        if (p_.temperature <= 0.0f)
        {
            // The first of the largest logits; a NaN is never selected.
            long best = 0;
            float best_value = -std::numeric_limits<float>::infinity();
            for (long v = 0; v < vocab; ++v)
            {
                if (logits[v] > best_value)
                {
                    best_value = logits[v];
                    best = v;
                }
            }
            last_p_ = 1.0f;
            return static_cast<int>(best);
        }

        // The k best candidates; without top-k, enough of them for the nucleus.
        const long k = p_.top_k > 0 ? std::min<long>(p_.top_k, vocab) : std::min<long>(1024, vocab);
        candidates_.clear();
        candidates_.reserve(static_cast<size_t>(k) + 1);
        {
            /*
                A bounded selection keeps the k largest in a min-heap, whose smallest value is
                the threshold a logit must exceed to enter.  Blocks of logits that hold none above
                it are passed over at once, which leaves the heap as the full scan does.  NaN
                values never enter.
            */
            auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
            float threshold = -std::numeric_limits<float>::infinity();
            long v = 0;
            for (; v < vocab && static_cast<long>(candidates_.size()) < k; ++v)
            {
                const float l = logits[v];
                if (l != l) continue;
                candidates_.emplace_back(l, static_cast<int>(v));
                std::push_heap(candidates_.begin(), candidates_.end(), cmp);
            }
            if (static_cast<long>(candidates_.size()) == k) threshold = candidates_.front().first;
            auto offer = [&](long u) {
                const float l = logits[u];
                if (!(l > threshold)) return;
                std::pop_heap(candidates_.begin(), candidates_.end(), cmp);
                candidates_.back() = std::make_pair(l, static_cast<int>(u));
                std::push_heap(candidates_.begin(), candidates_.end(), cmp);
                threshold = candidates_.front().first;
            };
            for (; v < vocab && v % scan_block != 0; ++v) offer(v);
            for (; v + scan_block <= vocab; v += scan_block)
                if (block_above(logits + v, threshold))
                    for (long u = v; u < v + scan_block; ++u) offer(u);
            for (; v < vocab; ++v) offer(v);
            if (candidates_.empty())
            {
                // Every logit is NaN.
                last_p_ = 1.0f;
                return 0;
            }
            std::sort(candidates_.begin(), candidates_.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
        }

        // Temperature and softmax over the candidates.
        const float inv_t = 1.0f / p_.temperature;
        const float top = candidates_.front().first;
        double sum = 0;
        for (auto& c : candidates_)
        {
            // An infinite best logit takes the whole mass, shared with its equals.
            c.first = std::isinf(top) ? (c.first == top ? 1.0f : 0.0f) : std::exp((c.first - top) * inv_t);
            sum += c.first;
        }
        for (auto& c : candidates_) c.first = static_cast<float>(c.first / sum);

        // min-p: candidates far less likely than the best one.
        size_t keep = candidates_.size();
        if (p_.min_p > 0.0f)
        {
            const float floor = candidates_.front().first * p_.min_p;
            size_t m = 1;
            while (m < keep && candidates_[m].first >= floor) ++m;
            keep = m;
        }
        // top-p: the smallest prefix reaching the probability mass.
        if (p_.top_p > 0.0f && p_.top_p < 1.0f)
        {
            double mass = 0;
            size_t m = 0;
            while (m < keep)
            {
                mass += candidates_[m].first;
                ++m;
                if (mass >= p_.top_p) break;
            }
            keep = m;
        }
        double total = 0;
        for (size_t i = 0; i < keep; ++i) total += candidates_[i].first;
        std::uniform_real_distribution<double> uni(0.0, total);
        const double r = uni(rng_);
        double acc = 0;
        for (size_t i = 0; i < keep; ++i)
        {
            acc += candidates_[i].first;
            if (r < acc)
            {
                last_p_ = static_cast<float>(candidates_[i].first / total);
                return candidates_[i].second;
            }
        }
        last_p_ = static_cast<float>(candidates_[keep - 1].first / total);
        return candidates_[keep - 1].second;
    }

}
