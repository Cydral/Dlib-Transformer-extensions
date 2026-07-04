// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CHAT_TEMPLATE_ABSTRACT_H_
#ifdef DLIB_CHAT_TEMPLATE_ABSTRACT_H_

#include <string>
#include "hf_tokenizer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class chat_template_kind
    {
        raw,     // no markup, plain text completion
        zephyr,  // "<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n"
        chatml   // "<|im_start|>role\n...<|im_end|>\n"
    };
    /*!
        WHAT THIS OBJECT REPRESENTS
            Identifies the conversation markup a chat model was trained with. Decoder
            chat models only generate reliably when their turns are delimited by the
            exact markers seen during fine-tuning; zephyr covers TinyLlama-Chat and
            other Zephyr-formatted Llama-family models, chatml covers the Qwen family
            and derivatives such as SmolLM2-Instruct, and raw disables any markup for
            plain completion.
    !*/

// ----------------------------------------------------------------------------------------

    class chat_template_formatter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Produces the exact text fed to the tokenizer for each turn of a chat
                session, according to the template the target model was trained with,
                so generation programs stay model-agnostic instead of hardcoding
                markers.

                The turn strings are designed so that the running token stream of a
                conversation matches a single continuous tokenization of the whole
                dialogue: the first turn carries the system block, every later turn
                begins with the newline that follows the assistant's closing token,
                and each assistant turn is closed by the tokenizer's eos token (</s>
                for zephyr, <|im_end|> for chatml). A generation loop therefore only
                has to feed first_turn() once, then next_turn() per exchange, and to
                append the eos token after each generated answer.

                For chatml, the assistant header depends on the model: thinking-capable
                models (e.g. Qwen3, recognized because "<think>" encodes to a single
                special token) receive an empty "<think>\n\n</think>\n\n" block, the
                documented soft switch that disables the reasoning trace, while plain
                ChatML instruct models (e.g. SmolLM2, Qwen2) receive the bare header.

            THREAD SAFETY
                Instances are immutable after construction, so concurrent calls to the
                const member functions are safe.
        !*/

    public:

        chat_template_formatter(
        );
        /*!
            ensures
                - #kind() == chat_template_kind::raw
        !*/

        explicit chat_template_formatter(
            chat_template_kind k
        );
        /*!
            ensures
                - #kind() == k
                - The thinking soft switch is disabled; use for_tokenizer() to enable
                  it from the tokenizer's capabilities.
        !*/

        static chat_template_kind detect(
            const hf_tokenizer& tok
        );
        /*!
            ensures
                - Identifies the template family from the text of the tokenizer's eos
                  special token: "<|im_end|>" yields chatml, "</s>" yields zephyr, any
                  other piece yields raw.
                - Detecting from the tokenizer, rather than from container metadata,
                  makes the same logic work for models imported live from a GGUF and
                  for models loaded back from a serialized archive, where the
                  container is no longer available.
        !*/

        static chat_template_formatter for_tokenizer(
            const hf_tokenizer& tok
        );
        /*!
            ensures
                - Returns a formatter F such that F.kind() == detect(tok).
                - If F.kind() == chat_template_kind::chatml, probes whether "<think>"
                  encodes to a single special token; when it does, the model is
                  thinking-capable and the assistant header emitted by first_turn()
                  and next_turn() carries the empty think block that disables the
                  reasoning trace.
        !*/

        static const char* name(
            chat_template_kind k
        );
        /*!
            ensures
                - Returns a human-readable identifier of k: "raw", "zephyr" or
                  "chatml".
        !*/

        chat_template_kind kind(
        ) const;
        /*!
            ensures
                - Returns the template family this formatter produces.
        !*/

        bool add_bos_on_first_turn(
        ) const;
        /*!
            ensures
                - Returns true if the first prefill of a conversation must be
                  prepended with the tokenizer's BOS token (zephyr), false otherwise
                  (chatml models use no leading BOS, raw follows the caller's
                  convention).
        !*/

        std::string system_prefix(
            const std::string& system_prompt
        ) const;
        /*!
            ensures
                - Returns the immutable conversation prefix holding the system block,
                  or an empty string for the raw kind.
                - The returned text is a strict prefix of first_turn(system_prompt, u)
                  for any user text u, and tokenizes to a strict prefix of the first
                  turn's token sequence. Generation loops measure this prefix (encoded
                  with add_bos_on_first_turn()) to pin the attention-sink positions
                  that must survive KV-cache evictions.
        !*/

        std::string first_turn(
            const std::string& system_prompt,
            const std::string& user_text
        ) const;
        /*!
            ensures
                - Returns the text of the first conversation turn: system block, first
                  user message and assistant header, in the markup of kind(). For the
                  raw kind, returns user_text unchanged.
                - The text is meant to be encoded with add_bos == add_bos_on_first_turn()
                  and parse_special == true, then fed as the initial prefill.
        !*/

        std::string next_turn(
            const std::string& user_text
        ) const;
        /*!
            ensures
                - Returns the text of every later turn: user message and assistant
                  header, in the markup of kind(). For the raw kind, returns user_text
                  unchanged.
                - The text begins with the newline that follows the eos token the
                  generation loop fed to close the previous assistant turn, so the
                  running token stream stays identical to a one-shot tokenization of
                  the whole conversation.
        !*/

        std::string clean_answer(
            std::string text
        ) const;
        /*!
            ensures
                - For the chatml kind, returns text with every "<think>...</think>"
                  reasoning span removed (an unterminated span is removed up to the end
                  of the text) and with the leading whitespace left behind trimmed.
                - For the other kinds, returns text unchanged.
        !*/

        double default_temperature(
        ) const;
        /*!
            ensures
                - Returns the sampling temperature recommended for the template
                  family: 0.6 for chatml, 0.8 otherwise.
        !*/

        size_t default_top_k(
        ) const;
        /*!
            ensures
                - Returns the top-k filter recommended for the template family: 20 for
                  chatml, 40 otherwise.
        !*/

        float default_top_p(
        ) const;
        /*!
            ensures
                - Returns the nucleus threshold recommended for the template family:
                  0.95 for chatml, 0.9 otherwise.
        !*/

        float default_min_p(
        ) const;
        /*!
            ensures
                - Returns the relative min-p threshold preset (0.05).
        !*/

        float default_repeat_penalty(
        ) const;
        /*!
            ensures
                - Returns the repetition penalty preset (1.1).
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CHAT_TEMPLATE_ABSTRACT_H_
