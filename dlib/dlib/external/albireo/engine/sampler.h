// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_SAMPLER_H_
#define ALBIREO_ENGINE_SAMPLER_H_

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace albireo
{

    struct sampling_params
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                How the next token is drawn from the logits of the model, with the
                parameters of the chat completion API and the usual extensions of local
                servers.  A temperature of 0 selects the most likely token.
        !*/
        float temperature = 0.7f;
        float top_p = 0.95f;
        int top_k = 40;               // 0 disables
        float min_p = 0.05f;          // relative to the most likely token; 0 disables
        float repeat_penalty = 1.0f;  // 1 disables
        int repeat_last_n = 64;
        float presence_penalty = 0.0f;
        float frequency_penalty = 0.0f;
        uint64_t seed = 0;            // 0: drawn from the system
        std::unordered_map<int, float> logit_bias;
    };

    class sampler
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The sampling state of one request: its parameters, its random generator
                and the history of the tokens it generated, which the penalties read.

                The filters run in the order of the reference servers: biases and
                penalties on the logits, then top-k, temperature, min-p and top-p on the
                candidates, then a draw.  Only the k best candidates are ever sorted, so a
                step costs one pass over the vocabulary.
        !*/
    public:
        explicit sampler(const sampling_params& p);

        // Draws the next token from a row of logits (modified in place).
        int sample(float* logits, long vocab);

        // Records a token of the sequence (generated or prompt) for the penalties.
        void accept(int token);

        // Probability of the token last returned by sample(), for log probabilities.
        float last_probability() const { return last_p_; }

        const sampling_params& params() const { return p_; }

    private:
        sampling_params p_;
        std::mt19937_64 rng_;
        std::vector<int> history_;
        std::vector<std::pair<float, int>> candidates_;
        float last_p_ = 1.0f;
    };

}

#endif // ALBIREO_ENGINE_SAMPLER_H_
