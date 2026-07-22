// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Shared text-generation path over the runtime inference engine.
//
// Chat front ends (interactive console loop, OpenAI-compatible service, batch probes)
// differ only in how they collect the conversation and how they display the answer.
// Everything numeric is the same work: resolve the chat template into a token stream,
// run the prefill, sample the next token from the last logits row, extend through the
// incremental step, and decide when to stop. Duplicating that work once per front end
// is how a prompt encoded with the same ids and sampled with the same parameters ends
// up producing two different answers, so it lives here once and both front ends drive
// it through callbacks.
//
// The two traps this header closes by construction:
//   - forward_prefill() returns the logits of every prompt position in a
//     [1, 1, N, vocab] tensor. Sampling must read row N-1. Reading row 0 samples the
//     continuation of the first prompt token, which stays syntactically plausible and
//     silently derails the whole answer instead of failing.
//   - the repetition penalty applies to the generated text only. Seeding its window
//     with the prompt penalizes the words of the question itself and moves the argmax.

#ifndef DLIB_DNN_TEXT_GENERATION_H_
#define DLIB_DNN_TEXT_GENERATION_H_

#include "text_generation_abstract.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "runtime_transformer.h"
#include "../tokenizer/hf_tokenizer.h"
#include "../tokenizer/chat_template.h"

namespace dlib
{
    /* Row r of a [1, 1, N, vocab] logits tensor. This accessor is the only supported
       way to reach a logits row: the raw host() pointer addresses position 0, which is
       the right answer for a step() result and the wrong one for a prefill. */
    inline const float* logits_row(const tensor& logits, long r)
    {
        DLIB_CASSERT(logits.nr() > 0 && logits.nc() > 0);
        DLIB_CASSERT(r >= 0 && r < logits.nr());
        return logits.host() + r * logits.nc();
    }

    inline const float* last_logits_row(const tensor& logits)
    {
        return logits_row(logits, logits.nr() - 1);
    }

    // ---------------------------------------------------------------------------------

    struct sampling_params
    {
        double temperature = 0.7;
        size_t top_k = 40;
        float  top_p = 0.95f;
        float  min_p = 0.05f;
        float  repeat_penalty = 1.1f;
        long   repeat_window = 64;    // number of trailing generated tokens penalized
        bool   greedy = false;        // argmax, ignores temperature and the filters
    };

    /* Next-token sampler. The scratch buffers are members rather than locals because a
       generation step touches the whole vocabulary: on a 150k-token vocabulary two
       fresh allocations per token dominate the host-side cost of the loop. */
    class token_sampler
    {
    public:
        token_sampler() : rng_(std::random_device{}()) {}
        explicit token_sampler(unsigned int seed) : rng_(seed) {}

        void seed(unsigned int s) { rng_.seed(s); }

        int pick(const float* row, long vocab, const std::vector<int>& recent, const sampling_params& sp)
        {
            DLIB_CASSERT(row != nullptr && vocab > 0);
            const size_t V = static_cast<size_t>(vocab);
            scratch_.assign(row, row + V);

            for (int t : recent)
            {
                if (t < 0 || static_cast<size_t>(t) >= V) continue;
                double& v = scratch_[static_cast<size_t>(t)];
                v = v > 0.0 ? v / sp.repeat_penalty : v * sp.repeat_penalty;
            }

            if (sp.greedy || sp.temperature <= 0.0)
                return static_cast<int>(std::max_element(scratch_.begin(), scratch_.end()) - scratch_.begin());

            order_.resize(V);
            for (size_t v = 0; v < V; ++v) order_[v] = v;
            const size_t k = std::min(sp.top_k ? sp.top_k : V, V);
            std::partial_sort(order_.begin(), order_.begin() + k, order_.end(),
                [this](size_t a, size_t b) { return scratch_[a] > scratch_[b]; });

            probs_.resize(k);
            const double mx = scratch_[order_[0]];
            double sum = 0.0;
            for (size_t i = 0; i < k; ++i)
            {
                probs_[i] = std::exp((scratch_[order_[i]] - mx) / sp.temperature);
                sum += probs_[i];
            }
            for (size_t i = 0; i < k; ++i) probs_[i] /= sum;

            /* Nucleus and min-p truncation on the sorted head: min-p drops everything
               far below the mode, top_p closes the mass once it is reached. */
            size_t last = k;
            double kept = 0.0;
            for (size_t i = 0; i < k; ++i)
            {
                if (probs_[i] < sp.min_p * probs_[0]) { last = i; break; }
                kept += probs_[i];
                if (kept >= sp.top_p) { last = i + 1; break; }
            }
            if (last == 0) last = 1;

            std::discrete_distribution<size_t> dist(probs_.begin(), probs_.begin() + last);
            return static_cast<int>(order_[dist(rng_)]);
        }

    private:
        std::mt19937 rng_;
        std::vector<double> scratch_;
        std::vector<size_t> order_;
        std::vector<double> probs_;
    };

    // ---------------------------------------------------------------------------------

    /* One exchange of a conversation. assistant is empty for the turn being answered. */
    struct chat_turn
    {
        std::string user;
        std::string assistant;
    };

    /* Token ids of a single user turn. allow_space_prefix is off on every call: the
       chat templates carry their own separators, and letting the tokenizer insert its
       dummy prefix shifts the ids of the whole turn. */
    inline std::vector<int> encode_turn(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt,
        const std::string& user_text,
        bool first_turn)
    {
        return first_turn
            ? tok.encode(fmt.first_turn(system_prompt, user_text), fmt.add_bos_on_first_turn(), false, true, false)
            : tok.encode(fmt.next_turn(user_text), false, false, true, false);
    }

    /* Token stream of a whole conversation, in the exact order an interactive loop
       produces it turn by turn: system block and first user turn, then for every past
       exchange the assistant text, the eos that closed it, and the next user turn.
       A stateless service replaying a conversation with this function reaches the same
       ids as the interactive loop that lived it. */
    inline std::vector<int> encode_conversation(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt,
        const std::vector<chat_turn>& turns)
    {
        std::vector<int> ids;
        for (size_t i = 0; i < turns.size(); ++i)
        {
            const std::vector<int> t = encode_turn(tok, fmt, system_prompt, turns[i].user, i == 0);
            ids.insert(ids.end(), t.begin(), t.end());
            if (!turns[i].assistant.empty())
            {
                const std::vector<int> a = tok.encode(turns[i].assistant, false, false, false, false);
                ids.insert(ids.end(), a.begin(), a.end());
                if (tok.eos_id() >= 0) ids.push_back(tok.eos_id());
            }
        }
        return ids;
    }

    /* Leading positions pinned across KV cache evictions: the BOS and the immutable
       system block. Both front ends must pass the same value, otherwise the sliding
       window evicts different rows for the same conversation. */
    inline long system_keep_length(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt)
    {
        long keep = (fmt.add_bos_on_first_turn() && tok.bos_id() >= 0) ? 1 : 0;
        const std::string prefix = fmt.system_prefix(system_prompt);
        if (!prefix.empty())
            keep += static_cast<long>(tok.encode(prefix, false, false, true, false).size());
        return keep;
    }

    // ---------------------------------------------------------------------------------

    /* Streaming event of the generation loop, one per accepted token. clean_delta is
       the stable suffix of the visible answer: it never contains a partial stop marker
       nor a trailing blank, so a front end can print it without ever having to unprint.
       It is empty while the model is still inside a reasoning span, which the raw
       answer exposes for front ends that display the trace. */
    struct generation_event
    {
        const std::string& answer;
        const std::string& clean;
        std::string clean_delta;
        bool reasoning_open;
    };

    struct generation_options
    {
        long max_new_tokens = 512;
        std::function<bool()> is_cancelled;                  // may be empty
        std::function<void(const generation_event&)> on_token;  // may be empty
    };

    struct generation_result
    {
        std::string text;            // final cleaned answer
        std::vector<int> tokens;     // generated ids, stop marker included when hit
        bool hit_eos = false;
        bool hit_stop = false;
        bool cancelled = false;
        bool truncated = false;      // max_new_tokens reached before any stop condition
    };

    /* Autoregressive generation from the logits of an already computed prompt. The
       caller runs the prefill (or the incremental steps of a follow-up turn) and hands
       over the resulting tensor; this function reads its last row, so the same code
       serves a [1, 1, N, vocab] prefill and a [1, 1, 1, vocab] step.

       recent is the repetition window; the caller owns it and it must contain generated
       tokens only. The KV cache is extended in place, so the engine holds the prompt and
       the accepted tokens when the call returns; the caller closes the turn (step(eos))
       when it keeps the cache alive across turns. */
    inline generation_result generate_reply(
        runtime_transformer& rt,
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const tensor& prompt_logits,
        token_sampler& sampler,
        const sampling_params& sp,
        std::vector<int>& recent,
        const generation_options& opt)
    {
        generation_result res;
        const long V = prompt_logits.nc();
        const int eos = tok.eos_id();
        const std::string stop = fmt.stop_string();
        const float* row = last_logits_row(prompt_logits);

        std::string answer, clean, shown;
        for (long n = 0; n < opt.max_new_tokens; ++n)
        {
            if (opt.is_cancelled && opt.is_cancelled()) { res.cancelled = true; break; }

            const int next = sampler.pick(row, V, recent, sp);
            if (next == eos) { res.hit_eos = true; break; }

            res.tokens.push_back(next);
            recent.push_back(next);
            while (static_cast<long>(recent.size()) > sp.repeat_window) recent.erase(recent.begin());

            /* The whole sequence is re-decoded at every step: SentencePiece carries the
               inter-word spaces in the piece markers, so decoding a token in isolation
               loses or invents the leading space, and byte-fallback pieces only form a
               character once their partners have arrived. */
            answer = tok.decode(res.tokens, true);

            bool stopped = false;
            if (!stop.empty())
            {
                const size_t b = answer.find(stop);
                if (b != std::string::npos) { answer.erase(b); stopped = true; res.hit_stop = true; }
            }
            clean = fmt.clean_answer(answer);
            if (!stopped)
            {
                /* Hold back the longest tail that is a prefix of the stop marker: it may
                   complete into the marker at the next token. Trailing blanks follow the
                   same rule, the final clean_answer() trimming them when terminal. */
                if (!stop.empty())
                {
                    const size_t maxh = std::min(clean.size(), stop.size() - 1);
                    for (size_t h = maxh; h > 0; --h)
                        if (clean.compare(clean.size() - h, h, stop, 0, h) == 0)
                        {
                            clean.erase(clean.size() - h);
                            break;
                        }
                }
                while (!clean.empty() && (clean.back() == '\n' || clean.back() == '\r'
                    || clean.back() == ' ' || clean.back() == '\t'))
                    clean.pop_back();
            }

            std::string delta;
            if (clean.size() > shown.size() && clean.compare(0, shown.size(), shown) == 0)
            {
                delta = clean.substr(shown.size());
                shown = clean;
            }
            if (opt.on_token)
            {
                const generation_event ev{ answer, clean, delta, shown.empty() };
                opt.on_token(ev);
            }

            if (stopped) break;
            row = last_logits_row(rt.step(next));
        }

        if (!res.hit_eos && !res.hit_stop && !res.cancelled
            && static_cast<long>(res.tokens.size()) >= opt.max_new_tokens)
            res.truncated = true;

        res.text = fmt.clean_answer(answer);
        return res;
    }
}

#endif // DLIB_DNN_TEXT_GENERATION_H_
