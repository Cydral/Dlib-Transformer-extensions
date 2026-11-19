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
//   - zephyr  : "<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n"
//               (TinyLlama-Chat and other Zephyr-formatted Llama-family models);
//   - chatml  : "<|im_start|>role\n...<|im_end|>\n" (Qwen family and derivatives);
//   - guanaco : "### Human: ...\n### Assistant:" (OpenAssistant-Guanaco fine-tunes);
//   - granite : "<|start_of_role|>role<|end_of_role|>...<|end_of_text|>\n"
//               (IBM Granite instruct models, dense and MoE);
//   - channels: "<|im_start|>role<|channel_start|>name<|content|>...<|channel_end|><|im_end|>"
//               (Olala hybrid decoders: ChatML turns whose content lies in named channels,
//               after a system turn whose instructions the template of the model holds);
//   - raw     : no markup, plain text completion.
//
// The template kind is detected from the declared template when the model carries
// one, else from the text of the eos special token ("</s>" -> zephyr,
// "<|im_end|>" -> chatml, "<|end_of_text|>" -> granite).
// Detecting from the tokenizer, rather than from GGUF metadata, makes the same
// logic work for models imported live from a GGUF and for models loaded back from
// a serialized .dat archive, where the container metadata is no longer available.

#ifndef DLIB_CHAT_TEMPLATE_H_
#define DLIB_CHAT_TEMPLATE_H_

#include "chat_template_abstract.h"

#include <string>
#include <utility>
#include <vector>

#include "hf_tokenizer.h"

namespace dlib
{
    enum class chat_template_kind { raw, zephyr, chatml, guanaco, granite, idefics3, channels };

    /* The instructions of the system turn of a template with channels (the olala family),
       for a reasoning effort ("disabled", "low", "medium" or "high") and with or without
       tools: the text the template writes between
       "<|im_start|>system<|channel_start|>instructions<|content|>" and
       "<|channel_end|><|im_end|>".  The template writes it with a few statements, read
       here as Jinja reads them with the options of the chat templates (the line break after
       a statement and the spaces before it on its line are not text, and every line ends
       with a line feed): comments, the value of reasoning_effort, and the tests of
       reasoning_effort and of tools with their branches.  Returns false, out being then
       unspecified, when the template holds no such turn, when its instructions hold
       anything else, or when they exceed 256 KB or 64 nested tests: the template comes
       from a model file, and its reading takes a time proportional to its length. */
    inline bool channel_instructions(const std::string& chat_template, const std::string& effort, bool tools,
        std::string& out)
    {
        const std::string open = "<|im_start|>system<|channel_start|>instructions<|content|>";
        const size_t first = chat_template.find(open);
        const size_t last = first == std::string::npos ? first
            : chat_template.find("<|channel_end|><|im_end|>", first);
        if (last == std::string::npos) return false;
        const size_t max_length = size_t(256) << 10, max_depth = 64;
        if (last - first - open.size() > max_length) return false;
        // The instructions with the line ends Jinja gives its templates.
        std::string s;
        s.reserve(last - first - open.size());
        for (size_t i = first + open.size(); i < last; ++i)
        {
            const char c = chat_template[i];
            if (c != '\r') s += c;
            else if (i + 1 >= last || chat_template[i + 1] != '\n') s += '\n';
        }
        auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
        out.clear();
        // For each test in progress: whether its text is written, and whether its test held.
        std::vector<std::pair<bool, bool>> tests;
        // The tests in progress whose text is not written: the text is written when there is none.
        size_t hidden = 0;
        size_t at = 0;
        while (at < s.size())
        {
            const size_t next = s.find('{', at);
            const bool tag = next != std::string::npos && next + 1 < s.size()
                && (s[next + 1] == '%' || s[next + 1] == '#' || s[next + 1] == '{');
            if (!tag)
            {
                const size_t end = next == std::string::npos ? s.size() : next + 1;
                if (hidden == 0) out.append(s, at, end - at);
                at = end;
                continue;
            }
            const char kind = s[next + 1];
            const size_t close = s.find(kind == '%' ? "%}" : kind == '#' ? "#}" : "}}", next + 2);
            if (close == std::string::npos) return false;
            // The statement without its trimming marks and the blanks around it.
            size_t from = next + 2, to = close;
            const bool trim_before = from < to && s[from] == '-';
            if (trim_before) ++from;
            const bool trim_after = from < to && s[to - 1] == '-';
            if (trim_after) --to;
            while (from < to && blank(s[from])) ++from;
            while (to > from && blank(s[to - 1])) --to;
            const std::string body = s.substr(from, to - from);
            // The text before the statement, without the spaces that precede a statement on its line.
            size_t end = next;
            if (kind != '{')
            {
                size_t b = end;
                while (b > at && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
                if (b > 0 && s[b - 1] == '\n') end = b;
            }
            if (trim_before)
                while (end > at && blank(s[end - 1])) --end;
            if (hidden == 0) out.append(s, at, end - at);
            at = close + 2;
            if (trim_after)
                while (at < s.size() && blank(s[at])) ++at;
            else if (kind != '{' && at < s.size() && s[at] == '\n') ++at;

            if (kind == '#') continue;
            if (kind == '{')
            {
                if (body != "reasoning_effort") return false;
                if (hidden == 0) out += effort;
            }
            else if (body == "if tools" || body == "if reasoning_effort is not defined")
            {
                if (tests.size() >= max_depth) return false;
                const bool holds = body == "if tools" && tools;
                tests.push_back(std::make_pair(holds, holds));
                if (!holds) ++hidden;
            }
            else if (body == "else")
            {
                if (tests.empty()) return false;
                if (!tests.back().first) --hidden;
                tests.back().first = !tests.back().second;
                if (!tests.back().first) ++hidden;
            }
            else if (body == "endif")
            {
                if (tests.empty()) return false;
                if (!tests.back().first) --hidden;
                tests.pop_back();
            }
            else if (body.compare(0, 4, "set ") != 0 || hidden == 0) return false;
        }
        return tests.empty();
    }

    /* Markers of the idefics3 family, exposed because a caller assembling a multimodal
       prompt has to reserve the positions its vision tower will fill and therefore needs
       to know what stands in for an image. */
    struct idefics3_markers
    {
        static const char* conversation_start() { return "<|im_start|>"; }
        static const char* end_of_utterance()   { return "<end_of_utterance>"; }
        static const char* image_boundary()     { return "<fake_token_around_image>"; }
        static const char* global_image()       { return "<global-img>"; }
        static const char* image_placeholder()  { return "<image>"; }

        /* The run of markers standing for one whole-image view: a boundary, the view
           marker, one placeholder per position the tower will produce, and a closing
           boundary.

           The count comes from the model's preprocessing rather than from the container,
           and the wrapping is what the processor emits: verified against it rather than
           guessed. Splitting an image into crops, which that preprocessing does by
           default, repeats the same block per crop with a row and column marker in place
           of the whole-image one and closes the whole run once; a single view is the
           simple case and the only one this library encodes so far. */
        static std::string image_block(long positions)
        {
            std::string s = image_boundary();
            s += global_image();
            for (long i = 0; i < positions; ++i) s += image_placeholder();
            s += image_boundary();
            return s;
        }
    };

    class chat_template_formatter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Produces the exact text fed to the tokenizer for each conversation turn
                of a chat session, according to the template the target model was
                trained with. The turn strings are designed so that the running token
                stream matches a single continuous tokenization of the whole
                conversation: the first turn carries the system block, later turns
                begin with the newline that follows the assistant's closing token (the
                channels kind writes none between its turns), and every assistant turn
                is closed by the tokenizer's eos token (</s> for zephyr, <|im_end|> for
                chatml and channels, <|end_of_text|> for granite).

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
        static chat_template_kind detect(const hf_tokenizer& tok, const std::string& name_hint = "")
        {
            /* The guanaco format is plain-text markup: no dedicated special token
               (its eos is the generic </s>) and, when the container declares a
               template at all, it is typically inherited from the base model by the
               conversion rather than reflecting the fine-tune. The model identity
               is therefore the strongest signal and is consulted first. */
            if (!name_hint.empty())
            {
                std::string h = name_hint;
                for (char& c : h) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
                if (h.find("guanaco") != std::string::npos) return chat_template_kind::guanaco;
            }
            const std::string& tpl = tok.chat_template();
            if (!tpl.empty())
            {
                /* The utterance marker is the signature of the idefics3 family, and it is
                   tested before the others because their template also mentions the ChatML
                   conversation marker, which would otherwise capture it. */
                if (tpl.find("<end_of_utterance>") != std::string::npos)
                    return chat_template_kind::idefics3;
                if (tpl.find("<|start_of_role|>") != std::string::npos) return chat_template_kind::granite;
                // The channels come in the turns of ChatML: their marker is tested first.
                if (tpl.find("<|channel_start|>") != std::string::npos) return chat_template_kind::channels;
                if (tpl.find("<|im_start|>") != std::string::npos) return chat_template_kind::chatml;
                if (tpl.find("<|user|>") != std::string::npos)     return chat_template_kind::zephyr;
                if (tpl.find("### Human") != std::string::npos)    return chat_template_kind::guanaco;
            }
            const std::vector<int> one{ tok.eos_id() };
            const std::string piece = tok.decode(one, /*skip_special=*/false);
            if (piece == "<|im_end|>")      return chat_template_kind::chatml;
            if (piece == "</s>")            return chat_template_kind::zephyr;
            if (piece == "<|end_of_text|>") return chat_template_kind::granite;
            return chat_template_kind::raw;
        }

        static chat_template_formatter for_tokenizer(const hf_tokenizer& tok)
        {
            return for_tokenizer(tok, detect(tok));
        }

        // Same as above with a model-name hint refining the detection (used for
        // formats such as guanaco that leave no signature in the tokenizer).
        static chat_template_formatter for_tokenizer(const hf_tokenizer& tok,
            const std::string& name_hint)
        {
            return for_tokenizer(tok, detect(tok, name_hint));
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
            if (fmt.kind_ == chat_template_kind::channels)
            {
                /* The instructions of the system turn come from the template of the
                   model, for an answer without reasoning (the prompt closes the analysis
                   channel at once) and without tools. */
                if (!channel_instructions(tok.chat_template(), "disabled", false, fmt.instructions_))
                    fmt.instructions_ = "Reasoning effort: disabled";
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
            if (n == "granite") return chat_template_kind::granite;
            if (n == "idefics3" || n == "smolvlm") return chat_template_kind::idefics3;
            if (n == "channels" || n == "olala") return chat_template_kind::channels;
            return chat_template_kind::raw;
        }

        static const char* name(chat_template_kind k)
        {
            switch (k)
            {
            case chat_template_kind::zephyr:  return "zephyr";
            case chat_template_kind::chatml:  return "chatml";
            case chat_template_kind::guanaco: return "guanaco";
            case chat_template_kind::granite: return "granite";
            case chat_template_kind::idefics3: return "idefics3";
            case chat_template_kind::channels: return "channels";
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
            /* idefics3 opens its conversation with a marker of its own rather than with
               the tokenizer's BOS, even though the two happen to be the same id in the
               SmolLM2 vocabulary; emitting it as text keeps the two roles apart. */
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
                /* The ChatML reference templates (Qwen family) never emit an empty
                   system block: when the conversation carries no system message,
                   they inject the canonical default. An empty block is outside the
                   training distribution and destabilizes small quantized models. */
                return "<|im_start|>system\n"
                    + (system_prompt.empty()
                       ? std::string("You are a helpful assistant.") : system_prompt)
                    + "<|im_end|>\n";
            case chat_template_kind::guanaco:
                /* Guanaco has no formal system block; a leading instruction followed by
                   a blank line is the common usage. */
                return system_prompt.empty() ? std::string() : system_prompt + "\n\n";
            case chat_template_kind::granite:
                return system_prompt.empty() ? std::string()
                    : "<|start_of_role|>system<|end_of_role|>" + system_prompt
                      + "<|end_of_text|>\n";
            case chat_template_kind::idefics3:
                /* The conversation marker opens the stream once, whether or not a system
                   block follows. The reference template capitalizes the role name and
                   closes every message with the utterance marker and a newline. */
                return std::string(idefics3_markers::conversation_start())
                    + (system_prompt.empty() ? std::string()
                       : "System: " + system_prompt
                         + idefics3_markers::end_of_utterance() + "\n");
            case chat_template_kind::channels:
                /* The system turn belongs to the model; the system prompt of the
                   conversation is the instructions channel of a developer turn. */
                {
                    // The template writes the system prompt without the white space around it.
                    const size_t a = system_prompt.find_first_not_of(" \t\r\n");
                    const size_t b = system_prompt.find_last_not_of(" \t\r\n");
                    return "<|endoftext|><|im_start|>system<|channel_start|>instructions<|content|>" + instructions_
                        + "<|channel_end|><|im_end|>"
                        + (a == std::string::npos ? std::string()
                           : "<|im_start|>developer<|channel_start|>instructions<|content|>"
                             + system_prompt.substr(a, b - a + 1) + "<|channel_end|><|im_end|>\n");
                }
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
            case chat_template_kind::granite:
                return system_prefix(system_prompt)
                    + "<|start_of_role|>user<|end_of_role|>" + user_text
                    + "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
            case chat_template_kind::idefics3:
                /* No space after the role when the message opens on an image, which is
                   what the reference template does and what the model was trained on. */
                return system_prefix(system_prompt)
                    + "User:" + (opens_on_image(user_text) ? "" : " ") + user_text
                    + idefics3_markers::end_of_utterance() + "\nAssistant:";
            case chat_template_kind::channels:
                return system_prefix(system_prompt) + channel_turn(user_text);
            default:
                return user_text;
            }
        }

        // Text of every later turn. Starts with the newline that follows the eos the
        // generation loop fed to close the previous assistant turn, so the token
        // stream stays identical to a one-shot tokenization of the conversation; the
        // channels kind writes no newline between its turns.
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
            case chat_template_kind::idefics3:
                return "\nUser:" + std::string(opens_on_image(user_text) ? "" : " ")
                    + user_text + idefics3_markers::end_of_utterance() + "\nAssistant:";
            case chat_template_kind::granite:
                return "\n<|start_of_role|>user<|end_of_role|>" + user_text
                    + "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
            case chat_template_kind::channels:
                // No line break between the turns of this template.
                return channel_turn(user_text);
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
            else if (kind_ == chat_template_kind::channels)
            {
                /* The markers of the channels are control tokens, which the decoding
                   leaves out: what remains of them is the name of each channel, before
                   its text.  The answer is the text of the final channel, which the
                   model opens first, the prompt having closed its analysis; a tools
                   channel the model would open after it holds no answer. */
                if (text.compare(0, 5, "final") == 0) text.erase(0, 5);
                const size_t b = text.find("tools<toolcalls>");
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
            if (kind_ == chat_template_kind::guanaco) return "\n###";
            if (kind_ == chat_template_kind::idefics3)
                return idefics3_markers::end_of_utterance();
            return std::string();
        }

        // Sampling presets matching the model family's published recommendations.
        // Generation programs apply them when the user did not override the values.
        // The channels family takes the values of the card of its model: 0.8, 40 and 0.9.
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

        /* A message opening on an image is spelled without the space that follows the
           role name otherwise. Detected on the rendered text rather than passed in, so
           that a caller assembling a turn does not have to declare its shape twice. */
        static bool opens_on_image(const std::string& text)
        {
            const std::string mark = idefics3_markers::image_boundary();
            return text.compare(0, mark.size(), mark) == 0;
        }

        // A question in the text channel of a user turn, then the turn of the answer, its
        // analysis channel closed at once: the model answers in its final channel.
        static std::string channel_turn(const std::string& user_text)
        {
            return "<|im_start|>user<|channel_start|>text<|content|>" + user_text + "<|channel_end|><|im_end|>"
                "<|im_start|>assistant<|channel_start|>analysis<|content|><|channel_end|>";
        }

        chat_template_kind kind_ = chat_template_kind::raw;
        bool thinking_ = false;
        bool reasoning_ = false;
        // The instructions of the system turn of a template with channels.
        std::string instructions_;
    };
}

#endif // DLIB_CHAT_TEMPLATE_H_
