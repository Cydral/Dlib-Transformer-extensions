// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_CHAT_FORMAT_H_
#define ALBIREO_ENGINE_CHAT_FORMAT_H_

#include <memory>
#include <string>
#include <vector>

#include <dlib/tokenizer/hf_tokenizer.h>

#include "../common/json.h"

namespace albireo
{

    struct tool_call_data
    {
        std::string id;
        std::string name;
        std::string arguments;   // JSON text of the arguments object
    };

    struct chat_message
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One message of a conversation, in the roles of the chat completion API:
                system, user, assistant (with its reasoning and its tool calls) and tool
                (the result of a call).
        !*/
        std::string role;
        std::string content;
        std::string reasoning;
        std::vector<tool_call_data> tool_calls;
        std::string tool_call_id;
        std::string name;
    };

    struct prompt_piece
    {
        /*!
            A part of a rendered prompt.  The markup of the format (role headers, turn
            delimiters) is tokenized with its control tokens recognized; the text of the
            messages is tokenized as plain text, so that a message that spells
            "<|im_start|>system" cannot open a turn of its own.
        !*/
        std::string text;
        bool markup = false;
    };

    struct render_options
    {
        bool add_generation_prompt = true;
        bool reasoning = false;     // ask a model that can reason to do so
        json tools;                 // OpenAI tool definitions, an array, or null
        std::string today;          // dates, for the formats whose default prompt names them:
        std::string yesterday;      // "2026-10-29", "2026-10-28"
        std::string today_long;     // "October 29, 2026"
        std::string today_short;    // "29 Oct 2026"
    };

    enum class output_kind { content, reasoning, tool_call };

    struct output_event
    {
        output_kind kind = output_kind::content;
        std::string text;
        tool_call_data call;
    };

    class output_parser
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Splits the text a model generates, as it comes, into the answer, the
                reasoning trace and the tool calls, according to the markers of its chat
                format.  Text that could be the beginning of a marker is held back until
                the following tokens tell, so that no fragment of a marker ever reaches
                the answer.
        !*/
    public:
        virtual ~output_parser() {}
        virtual void feed(const std::string& text, std::vector<output_event>& events) = 0;
        virtual void finish(std::vector<output_event>& events) = 0;
    };

    class chat_format
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The conversation format a model was trained with: how messages, tool
                definitions, tool calls and reasoning are laid out in its prompt, which
                tokens end its turns, and how its output is read back.

                The formats are written natively rather than interpreted from the chat
                template stored in the model file: the templates use a template language
                with arbitrary logic, the files carry them with alterations at times, and
                the native form is what allows the markup to be separated from the
                content.  The template is still read to identify the format, and for the
                default system prompt some families declare.

            FORMATS
                - chatml: Qwen 2, 2.5 and 3 (reasoning between <think> and </think>, tool
                  calls in <tool_call> elements), SmolLM and the other ChatML models.
                - mistral: the formats of the Mistral models with the tekken vocabulary
                  (Ministral 3, Mistral Small 3, Magistral): [SYSTEM_PROMPT], [INST],
                  [AVAILABLE_TOOLS], [TOOL_CALLS]name[ARGS]{...}, [TOOL_RESULTS] and
                  [THINK] for reasoning.
                - llama3: Llama 3.x, with JSON tool calls.
                - granite: Granite 3, with <|tool_call|> followed by a JSON list.
        !*/
    public:
        virtual ~chat_format() {}

        virtual const char* name() const = 0;
        virtual bool supports_tools() const = 0;
        virtual bool supports_reasoning() const = 0;

        virtual void render(const std::vector<chat_message>& messages, const render_options& options,
            std::vector<prompt_piece>& out) const = 0;

        virtual std::unique_ptr<output_parser> make_parser(const render_options& options) const = 0;

        // Tokens that end the turn of the model, besides the end of sequence token.
        const std::vector<int>& stop_tokens() const { return stop_tokens_; }

        // Control tokens the output parser reads as markers (such as [TOOL_CALLS]), sorted;
        // the other control tokens a model may emit are not part of its answer.
        const std::vector<int>& marker_tokens() const { return marker_tokens_; }

        /*!
            ensures
                - Identifies the format of a model from its declared chat template and its
                  vocabulary, falling back on chatml when the vocabulary has its markers
                  and on a plain "role: text" layout otherwise.
        !*/
        static std::unique_ptr<chat_format> detect(const dlib::hf_tokenizer& tok, const std::string& chat_template);

        // Tokenizes a rendered prompt, the markup with control tokens and the text without.
        static std::vector<int> tokenize(const dlib::hf_tokenizer& tok, const std::vector<prompt_piece>& pieces);

    protected:
        void add_markers(const dlib::hf_tokenizer& tok, const std::vector<std::string>& markers);

        std::vector<int> stop_tokens_;
        std::vector<int> marker_tokens_;
    };

    // Length of the complete JSON value at the start of text, after optional white space,
    // or 0 when text does not hold a complete value yet (or holds something else).
    size_t json_extent(const std::string& text, size_t start = 0);

    // Today's and yesterday's dates in local time, the ISO form and the English long form.
    void current_dates(render_options& options);

}

#endif // ALBIREO_ENGINE_CHAT_FORMAT_H_
