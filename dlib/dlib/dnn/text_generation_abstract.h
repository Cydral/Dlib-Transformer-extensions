// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_TEXT_GENERATION_ABSTRACT_H_
#ifdef DLIB_DNN_TEXT_GENERATION_ABSTRACT_H_

#include "runtime_transformer_abstract.h"
#include "../tokenizer/hf_tokenizer_abstract.h"
#include "../tokenizer/chat_template_abstract.h"

namespace dlib
{
    /*!
        WHAT THIS FILE REPRESENTS
            The single numeric path shared by every chat front end built on the runtime
            inference engine: token-stream assembly from a chat template, next-token
            sampling, and the autoregressive loop with its stop conditions.

            Front ends differ in how they collect the conversation and how they render
            the answer, not in what they compute. Keeping one implementation of the
            computation is what guarantees that an interactive console session and a
            stateless HTTP service answer the same prompt identically; two copies of the
            loop diverge on details that no unit test covers, such as which logits row
            is sampled or which tokens the repetition penalty sees.

        TYPICAL USAGE
            // --- Interactive, one live KV cache across turns ---
            rt.set_context(ctx, system_keep_length(tok, fmt, sys));
            std::vector<int> ids = encode_turn(tok, fmt, sys, user, first_turn);
            const tensor* lg = first_turn ? &rt.forward_prefill(ids) : nullptr;
            if (!first_turn) for (int t : ids) lg = &rt.step(t);
            std::vector<int> recent;
            generation_result r = generate_reply(rt, tok, fmt, *lg, sampler, sp, recent, opt);
            rt.step(tok.eos_id());   // close the assistant turn

            // --- Stateless service, the whole conversation replayed per request ---
            rt.set_context(ctx, system_keep_length(tok, fmt, sys));
            std::vector<int> ids = encode_conversation(tok, fmt, sys, turns);
            std::vector<int> recent;
            generation_result r = generate_reply(rt, tok, fmt, rt.forward_prefill(ids),
                                                 sampler, sp, recent, opt);
    !*/

// ----------------------------------------------------------------------------------------

    const float* logits_row(
        const tensor& logits,
        long r
    );
    /*!
        requires
            - logits is a [1, 1, N, vocab] tensor produced by
              runtime_transformer::forward_prefill() or ::step()
            - 0 <= r < logits.nr()
        ensures
            - Returns a pointer to the first of the logits.nc() values of row r,
              valid until the next forward call on the engine that produced logits.
    !*/

    const float* last_logits_row(
        const tensor& logits
    );
    /*!
        ensures
            - Returns logits_row(logits, logits.nr() - 1), the only row a next-token
              sampler may read: forward_prefill() returns every prompt position and the
              prediction of the continuation lives on the last one.
    !*/

// ----------------------------------------------------------------------------------------

    struct sampling_params
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The next-token sampling policy of one generation.

            FIELDS
                temperature    - Logit temperature; <= 0 forces the greedy path
                top_k          - Candidates kept before the probability filters; 0 keeps
                                 the whole vocabulary
                top_p          - Nucleus threshold on the cumulated probability mass
                min_p          - Floor relative to the mode's probability
                repeat_penalty - Divides positive logits and multiplies negative ones of
                                 the tokens present in the repetition window
                repeat_window  - Number of trailing generated tokens penalized
                greedy         - Argmax, ignoring temperature and all filters
        !*/
    };

    class token_sampler
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A next-token sampler holding its own random engine and the scratch
                buffers of the vocabulary-wide passes. Instances are not thread safe;
                one per generation thread.
        !*/

    public:

        token_sampler();
        explicit token_sampler(unsigned int seed);
        /*!
            ensures
                - The default constructor seeds the engine from std::random_device;
                  the explicit form seeds it from the given value, which makes a
                  non-greedy generation reproducible.
        !*/

        void seed(unsigned int s);
        /*!
            ensures
                - Reseeds the random engine.
        !*/

        int pick(
            const float* row,
            long vocab,
            const std::vector<int>& recent,
            const sampling_params& sp
        );
        /*!
            requires
                - row points to vocab logits (see last_logits_row())
                - vocab > 0
            ensures
                - Applies the repetition penalty to the ids present in recent, then
                  returns the argmax when sp.greedy or sp.temperature <= 0, otherwise a
                  draw from the top_k / min_p / top_p truncated distribution.
                - recent must hold generated tokens only: including prompt tokens
                  penalizes the words of the question and moves the argmax.
                - The returned id is in [0, vocab).
        !*/
    };

// ----------------------------------------------------------------------------------------

    struct chat_turn
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One exchange of a conversation: the user message and the assistant reply
                it received. assistant is empty for the turn being answered.
        !*/
    };

    std::vector<int> encode_turn(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt,
        const std::string& user_text,
        bool first_turn
    );
    /*!
        ensures
            - Returns the token ids of one user turn: the first turn carries the BOS
              (when the template asks for it) and the system block, later turns carry
              the plain turn markers.
            - Special tokens are parsed, the tokenizer's dummy space prefix is disabled:
              the templates carry their own separators and the prefix would shift every
              id of the turn.
    !*/

    std::vector<int> encode_conversation(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt,
        const std::vector<chat_turn>& turns
    );
    /*!
        requires
            - turns is not empty
        ensures
            - Returns the token ids of the whole conversation in the order an
              interactive loop produces them turn by turn: encode_turn() for each user
              message, and after every answered turn the assistant text followed by the
              tokenizer's eos.
            - A stateless service replaying a conversation reaches the same ids as the
              interactive loop that lived it.
    !*/

    long system_keep_length(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::string& system_prompt
    );
    /*!
        ensures
            - Returns the number of leading token positions that a KV cache eviction
              must preserve: the BOS when the template emits one, plus the length of the
              template's system prefix.
            - Pass it to runtime_transformer::set_context() as the keep length.
    !*/

// ----------------------------------------------------------------------------------------

    struct generation_event
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The streaming state after one accepted token.

            FIELDS
                answer         - Raw decode of every token generated so far, truncated at
                                 the stop marker once it appears
                clean          - answer after the template cleaning, with any partial stop
                                 marker and any trailing blank held back
                clean_delta    - Stable suffix appended to clean at this step; safe to
                                 print or stream as is, empty when nothing became visible
                reasoning_open - True while the visible answer is still empty, that is
                                 while the model is inside a reasoning span
        !*/
    };

    struct generation_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What drives one generation beyond the sampling policy.

            FIELDS
                max_new_tokens - Hard bound on the number of generated tokens
                is_cancelled   - Polled before every token; generation stops when it
                                 returns true. May be empty.
                on_token       - Called after every accepted token. May be empty.
        !*/
    };

    struct generation_result
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The outcome of one generation.

            FIELDS
                text      - Final cleaned answer
                tokens    - Generated ids, stop marker included when it was reached
                hit_eos   - The tokenizer's eos was sampled
                hit_stop  - The template's stop string appeared in the decoded text
                cancelled - is_cancelled() returned true
                truncated - max_new_tokens was reached before any stop condition
        !*/
    };

    generation_result generate_reply(
        runtime_transformer& rt,
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const tensor& prompt_logits,
        token_sampler& sampler,
        const sampling_params& sp,
        std::vector<int>& recent,
        const generation_options& opt
    );
    /*!
        requires
            - prompt_logits was returned by the last forward call on rt, and rt was
              configured with a context capacity (see set_context()).
            - recent holds generated tokens only, and is empty at the start of a turn.
        ensures
            - Samples from last_logits_row(prompt_logits), then extends the sequence
              through rt.step() until the eos is sampled, the template's stop string
              appears, is_cancelled() returns true, or max_new_tokens is reached.
            - Appends every accepted token to recent, trimmed to sp.repeat_window.
            - Calls opt.on_token once per accepted token, with the stable suffix of the
              visible answer; a front end printing clean_delta never has to unprint.
            - The KV cache holds the prompt and the accepted tokens on return; a caller
              keeping the cache alive across turns closes the turn with rt.step(eos).
            - #generation_result::text is the cleaned answer, and exactly one of the
              four outcome flags is set unless generation ended on an empty answer.
    !*/
}

#endif // DLIB_DNN_TEXT_GENERATION_ABSTRACT_H_
