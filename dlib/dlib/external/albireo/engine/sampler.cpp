// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace albireo
{

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
            long best = 0;
            for (long v = 1; v < vocab; ++v)
                if (logits[v] > logits[best]) best = v;
            last_p_ = 1.0f;
            return static_cast<int>(best);
        }

        // The k best candidates; without top-k, enough of them for the nucleus.
        const long k = p_.top_k > 0 ? std::min<long>(p_.top_k, vocab) : std::min<long>(1024, vocab);
        candidates_.clear();
        candidates_.reserve(static_cast<size_t>(k) + 1);
        {
            // A bounded selection: keep the k largest in a min-heap.
            auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
            for (long v = 0; v < vocab; ++v)
            {
                const float l = logits[v];
                if (static_cast<long>(candidates_.size()) < k)
                {
                    candidates_.emplace_back(l, static_cast<int>(v));
                    std::push_heap(candidates_.begin(), candidates_.end(), cmp);
                }
                else if (l > candidates_.front().first)
                {
                    std::pop_heap(candidates_.begin(), candidates_.end(), cmp);
                    candidates_.back() = std::make_pair(l, static_cast<int>(v));
                    std::push_heap(candidates_.begin(), candidates_.end(), cmp);
                }
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
            c.first = std::exp((c.first - top) * inv_t);
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
