// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Chat-template formatting for imported open-weight models.
//
// Decoder-only chat models are trained on a fixed conversation markup (the "chat
// template"); feeding a different markup degrades or breaks generation. This header
// centralizes the supported templates so generation programs stay model-agnostic:
// they ask the formatter for the text of each turn instead of hardcoding markers.
//
// Supported templates:
//   - zephyr : "<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n"
//              (TinyLlama-Chat and other Zephyr-formatted Llama-family models);
//   - chatml : "<|im_start|>role\n...<|im_end|>\n" (Qwen family and derivatives);
//   - raw    : no markup, plain text completion.
//
// The template kind is detected from the tokenizer itself: the text of the eos
// special token identifies the family ("</s>" -> zephyr, "<|im_end|>" -> chatml).
// Detecting from the tokenizer, rather than from GGUF metadata, makes the same
// logic work for models imported live from a GGUF and for models loaded back from
// a serialized .dat archive, where the container metadata is no longer available.

#ifndef DLIB_CHAT_TEMPLATE_H_
#define DLIB_CHAT_TEMPLATE_H_

#include <string>
#include <vector>

#include "hf_tokenizer.h"

namespace dlib
{
    enum class chat_template_kind { raw, zephyr, chatml, guanaco };

    class chat_template_formatter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Produces the exact text fed to the tokenizer for each conversation turn
                of a chat session, according to the template the target model was
                trained with. The turn strings are designed so that the running token
                stream matches a single continuous tokenization of the whole
                conversation: the first turn carries the system block, later turns
                begin with the newline that follows the assistant's closing token, and
                every assistant turn is closed by the tokenizer's eos token (</s> for
                zephyr, <|im_end|> for chatml).

                For chatml, the assistant header depends on the model: thinking-capable
                models (Qwen3, detected by "<think>" encoding to a single special token)
                receive an empty "<think>\n\n</think>\n\n" block, the documented soft
                switch for non-thinking mode, while plain ChatML instruct models
                (SmolLM2, Qwen2) receive the bare header. clean_answer() additionally
                strips any reasoning span from a decoded answer before display.
        !*/

    public:

        chat_template_formatter() = default;

        explicit chat_template_formatter(chat_template_kind k) : kind_(k) {}

        // Identify the template family. The primary criterion is the chat template the
        // model itself declares (persisted with the tokenizer): fingerprinting its
        // marker strings follows the reference implementation's approach and tracks
        // whatever format the model was actually fine-tuned with. When no template is
        // declared, the text of the eos special token is used as a fallback heuristic.
        static chat_template_kind detect(const hf_tokenizer& tok)
        {
            const std::string& tpl = tok.chat_template();
            if (!tpl.empty())
            {
                if (tpl.find("<|im_start|>") != std::string::npos) return chat_template_kind::chatml;
                if (tpl.find("<|user|>") != std::string::npos)     return chat_template_kind::zephyr;
            }
            const std::vector<int> one{ tok.eos_id() };
            const std::string piece = tok.decode(one, /*skip_special=*/false);
            if (piece == "<|im_end|>") return chat_template_kind::chatml;
            if (piece == "</s>")       return chat_template_kind::zephyr;
            return chat_template_kind::raw;
        }

        static chat_template_formatter for_tokenizer(const hf_tokenizer& tok)
        {
            return for_tokenizer(tok, detect(tok));
        }

        // Same as above with the template kind forced by the caller, for models whose
        // container declares no template and whose fallback detection is wrong (e.g.
        // Guanaco fine-tunes carried by old GGUFs).
        static chat_template_formatter for_tokenizer(const hf_tokenizer& tok, chat_template_kind forced)
        {
            chat_template_formatter fmt(forced);
            if (fmt.kind_ == chat_template_kind::chatml)
            {
                /* ChatML covers both reasoning models (Qwen3) and plain instruct models
                   (SmolLM2, Qwen2). The empty think block that disables the reasoning
                   trace must be emitted only for the former: probing whether "<think>"
                   encodes to a single special token tells the two apart. */
                const std::vector<int> t = tok.encode("<think>", /*add_bos=*/false,
                    /*add_eos=*/false, /*parse_special=*/true, /*allow_space_prefix=*/false);
                fmt.thinking_ = (t.size() == 1);
            }
            return fmt;
        }

        // Parse a template name ("raw", "zephyr", "chatml", "guanaco"); anything else
        // yields raw.
        static chat_template_kind from_name(const std::string& n)
        {
            if (n == "zephyr")  return chat_template_kind::zephyr;
            if (n == "chatml")  return chat_template_kind::chatml;
            if (n == "guanaco") return chat_template_kind::guanaco;
            return chat_template_kind::raw;
        }

        static const char* name(chat_template_kind k)
        {
            switch (k)
            {
            case chat_template_kind::zephyr:  return "zephyr";
            case chat_template_kind::chatml:  return "chatml";
            case chat_template_kind::guanaco: return "guanaco";
            default:                         return "raw";
            }
        }

        chat_template_kind kind() const { return kind_; }

        // Whether the detected model exposes a reasoning mode (ChatML with a native
        // "<think>" special token, e.g. Qwen3).
        bool supports_reasoning() const { return thinking_; }

        // Enables or disables the reasoning trace for thinking-capable models. Off by
        // default: the assistant header then carries the empty think block, the
        // documented soft switch for non-thinking mode. When enabled, the bare header
        // lets the model open its own "<think>" span; clean_answer() strips it from
        // the displayed text. No effect on models without a reasoning mode.
        void set_reasoning(bool enabled) { reasoning_ = enabled; }
        bool reasoning() const { return reasoning_ && thinking_; }

        // Whether the first prefill must be prepended with the BOS token. Zephyr
        // conversations start with BOS; ChatML models (Qwen) use no leading BOS.
        bool add_bos_on_first_turn() const
        {
            return kind_ == chat_template_kind::zephyr
                || kind_ == chat_template_kind::guanaco;
        }

        // The immutable conversation prefix: BOS-side system block. Generation loops
        // measure this prefix (with add_bos_on_first_turn()) to pin the attention-sink
        // positions that must survive KV-cache evictions.
        std::string system_prefix(const std::string& system_prompt) const
        {
            switch (kind_)
            {
            case chat_template_kind::zephyr:
                return "<|system|>\n" + system_prompt + "</s>\n";
            case chat_template_kind::chatml:
                return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
            case chat_template_kind::guanaco:
                /* Guanaco has no formal system block; a leading instruction followed by
                   a blank line is the common usage. */
                return system_prompt.empty() ? std::string() : system_prompt + "\n\n";
            default:
                return std::string();
            }
        }

        // Text of the first turn: system block, first user message, assistant header.
        std::string first_turn(const std::string& system_prompt, const std::string& user_text) const
        {
            switch (kind_)
            {
            case chat_template_kind::zephyr:
                return system_prefix(system_prompt)
                    + "<|user|>\n" + user_text + "</s>\n<|assistant|>\n";
            case chat_template_kind::chatml:
                return system_prefix(system_prompt)
                    + "<|im_start|>user\n" + user_text + "<|im_end|>\n"
                    + assistant_header();
            case chat_template_kind::guanaco:
                return system_prefix(system_prompt)
                    + "### Human: " + user_text + "\n### Assistant:";
            default:
                return user_text;
            }
        }

        // Text of every later turn. Starts with the newline that follows the eos the
        // generation loop fed to close the previous assistant turn, so the token
        // stream stays identical to a one-shot tokenization of the conversation.
        std::string next_turn(const std::string& user_text) const
        {
            switch (kind_)
            {
            case chat_template_kind::zephyr:
                return "\n<|user|>\n" + user_text + "</s>\n<|assistant|>\n";
            case chat_template_kind::chatml:
                return "\n<|im_start|>user\n" + user_text + "<|im_end|>\n"
                    + assistant_header();
            case chat_template_kind::guanaco:
                return "\n### Human: " + user_text + "\n### Assistant:";
            default:
                return user_text;
            }
        }

        // Remove template artifacts from a decoded answer: reasoning spans for chatml
        // ("<think> ... </think>"), and anything from a leaked next-turn marker onward
        // for guanaco (these models often start the next "### Human:" turn instead of
        // emitting eos). Leading and trailing whitespace left behind is trimmed.
        std::string clean_answer(std::string text) const
        {
            if (kind_ == chat_template_kind::chatml)
            {
                for (;;)
                {
                    const size_t b = text.find("<think>");
                    if (b == std::string::npos) break;
                    const size_t e = text.find("</think>", b);
                    if (e == std::string::npos) { text.erase(b); break; }
                    text.erase(b, e + 8 - b);
                }
            }
            else if (kind_ == chat_template_kind::guanaco)
            {
                const size_t b = text.find("###");
                if (b != std::string::npos) text.erase(b);
            }
            else
            {
                return text;
            }
            const size_t p = text.find_first_not_of(" \t\r\n");
            if (p == std::string::npos) return std::string();
            const size_t q = text.find_last_not_of(" \t\r\n");
            return text.substr(p, q - p + 1);
        }

        // Text whose appearance in the generated answer must stop the generation, or an
        // empty string when eos is the only stop condition. Guanaco models frequently
        // end a turn by opening a new "### ..." section instead of emitting eos, and
        // the section name varies ("### Human:", "### Interaction:", ...), so any
        // section opening on a fresh line stops the turn.
        std::string stop_string() const
        {
            return kind_ == chat_template_kind::guanaco ? "\n###" : std::string();
        }

        // Sampling presets matching the model family's published recommendations.
        // Generation programs apply them when the user did not override the values.
        double default_temperature() const
        {
            return kind_ == chat_template_kind::chatml ? 0.6 : 0.8;
        }
        size_t default_top_k() const
        {
            return kind_ == chat_template_kind::chatml ? 20 : 40;
        }
        float default_top_p() const
        {
            return kind_ == chat_template_kind::chatml ? 0.95f : 0.9f;
        }
        float default_min_p() const { return 0.05f; }
        float default_repeat_penalty() const { return 1.1f; }

    private:

        // ChatML assistant header. For thinking-capable models (Qwen3), the empty think
        // block is the documented soft switch that prevents the model from opening a
        // reasoning trace; plain ChatML instruct models (SmolLM2, Qwen2) get the bare
        // header, as their template defines no think markup.
        std::string assistant_header() const
        {
            return (thinking_ && !reasoning_)
                ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                : "<|im_start|>assistant\n";
        }

        chat_template_kind kind_ = chat_template_kind::raw;
        bool thinking_ = false;
        bool reasoning_ = false;
    };
}

#endif // DLIB_CHAT_TEMPLATE_H_
