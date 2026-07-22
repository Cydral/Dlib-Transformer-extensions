// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SERVER_CHAT_ABSTRACT_H_
#ifdef DLIB_SERVER_CHAT_ABSTRACT_H_

#include <string>
#include <vector>
#include "server_http_abstract.h"

namespace dlib
{
    // ------------------------------------------------------------------------------------

    class chat_json
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A minimal read-only JSON document, sufficient for the chat completion
                API: objects, arrays, strings (with the standard escapes, including
                \uXXXX and surrogate pairs decoded to UTF-8), numbers, booleans and
                null. Accessors are tolerant: reading a member that is absent or of
                another type yields the caller's default instead of throwing, which
                matches the optional nature of most API fields. Documents are built
                by parse() only; responses are emitted by hand with escape().

            THREAD SAFETY
                Instances are immutable after parse(), so concurrent reads are safe.
        !*/

    public:

        enum class kind { null_v, boolean_v, number_v, string_v, array_v, object_v };

        chat_json();
        /*!
            ensures
                - #type() == kind::null_v
        !*/

        kind type() const;
        bool is_object() const;
        bool is_array() const;
        bool is_string() const;
        bool is_number() const;
        /*!
            ensures
                - Report the type of this value.
        !*/

        bool has(const std::string& key) const;
        /*!
            ensures
                - Returns true if this value is an object holding the given key.
        !*/

        const chat_json& operator[](const std::string& key) const;
        /*!
            ensures
                - Returns the member stored under key when this value is an object
                  and the key is present; otherwise returns a shared null value.
        !*/

        size_t size() const;
        const chat_json& operator[](size_t i) const;
        /*!
            ensures
                - size() returns the element count when this value is an array,
                  0 otherwise; operator[] returns element i, or a shared null value
                  when out of range or not an array.
        !*/

        std::string as_string(const std::string& def = "") const;
        double as_number(double def = 0.0) const;
        long as_long(long def = 0) const;
        bool as_bool(bool def = false) const;
        /*!
            ensures
                - Return this value converted to the requested type, or def when the
                  value is of another type.
        !*/

        static chat_json parse(const std::string& text);
        /*!
            ensures
                - Parses text as a JSON document and returns it.
            throws
                - std::runtime_error on malformed input (including trailing content
                  after the document).
        !*/

        static std::string escape(const std::string& s);
        /*!
            ensures
                - Returns s escaped for inclusion between JSON double quotes: the
                  standard two-character escapes, and \u00XX for the remaining
                  control characters. UTF-8 bytes above 0x1F pass through.
        !*/
    };

    // ------------------------------------------------------------------------------------

    struct chat_model_info
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One model served by the chat endpoint: its identifier (listed by
                /v1/models and matched against the "model" field of requests) and
                whether it exposes a reasoning (deep thinking) mode, reported by
                /v1/models as an additive per-entry "reasoning" boolean so clients
                surface the toggle only where it applies.
        !*/

        std::string id;
        bool reasoning;
    };

    struct chat_message
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One turn of a conversation in the OpenAI wire format: a role
                ("system", "user" or "assistant") and its content. The wire content
                may be a plain string or an array of typed parts (the multimodal
                form): all text parts are concatenated into content, and every
                image_url part (data or http URL) is collected into image_urls for
                image-capable backends; text-only backends can ignore them.
        !*/

        std::string role;
        std::string content;
        std::vector<std::string> image_urls;
    };

    struct chat_request
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A parsed chat completion request. The common fields are lifted into
                typed members; raw keeps the whole request body so backends can read
                additional parameters without an API change.

            FIELDS
                model       - The model id resolved by the server: the requested
                              id when it matches a declared model, the default
                              model otherwise.
                messages    - The full conversation, oldest message first. The
                              server is stateless: every request carries the whole
                              history the client wants the model to see, and the
                              client owns the context window.
                temperature - Sampling temperature, or a negative value when the
                              request did not set it (backend default applies).
                top_p       - Nucleus threshold, or a negative value when unset.
                top_k       - Top-k cutoff, or a negative value when unset.
                min_p       - Minimum-probability cutoff, or negative when unset.
                repeat_penalty - Repetition penalty, or negative when unset. These
                              three mirror the homonymous extension fields common
                              across OpenAI-compatible servers.
                max_tokens  - Response length limit, or 0 when unset.
                stream      - True when the request asked for server-sent-events
                              streaming ("stream": true).
                reasoning   - Tri-state mirror of the additive "reasoning" boolean
                              request field: 1 when true, 0 when false, -1 when
                              absent. Backends of reasoning-capable models enable
                              or disable their deep-thinking mode accordingly.
                request_id  - Client-chosen identifier of the request (the
                              non-standard "request_id" field), empty when absent.
                is_cancelled- Cooperative cancellation probe, set by the server
                              when a request_id was given: backends should test it
                              between generation steps and return the partial
                              answer when it turns true (a cancel posted to
                              /v1/internal/cancel with that id triggers it).
                raw         - The parsed request body.
        !*/

        std::string model;
        std::vector<chat_message> messages;
        double temperature;
        double top_p;
        long max_tokens;
        chat_json raw;
    };

    struct chat_result
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The backend's answer to a chat completion request.

            FIELDS
                content           - The assistant's answer text.
                finish_reason     - "stop" (natural end) or "length" (the
                                    max_tokens limit was reached).
                prompt_tokens     - Token count of the prompt, for the usage block.
                completion_tokens - Token count of the answer.
        !*/

        std::string content;
        std::string finish_reason;
        long prompt_tokens;
        long completion_tokens;
    };

    // ------------------------------------------------------------------------------------

    class server_chat : public server_http
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An HTTP server exposing a text generation backend as a chat service,
                in two complementary forms:
                  - GET  /                     : a self-contained browser interface
                    (conversation management persisted in the browser's IndexedDB,
                    search, rename, delete, markdown export, settings and about
                    dialogs, client-side context window management);
                  - POST /v1/chat/completions : the OpenAI chat completion API;
                  - GET  /v1/models           : the OpenAI model listing;
                  - POST /v1/internal/cancel  : cooperative cancellation of a
                    running completion, by the request_id the client attached to
                    it (the backend then returns the partial answer).
                Cross-origin requests are allowed (permissive CORS, OPTIONS
                preflight handled), so external tools of the OpenAI ecosystem can
                call the API directly.

                The server holds no conversation state: every request carries the
                complete message history, and the embedded interface stores
                conversations locally in the browser. A completion is answered in
                one JSON piece by default; when the request sets "stream": true it
                is answered as OpenAI-format server-sent events instead, one chunk
                per fragment the backend emits, closed by a final chunk carrying
                finish_reason and, additively, the definitive answer in a
                "final_content" field, then the [DONE] sentinel.

                To build a service, inherit from this class and implement
                on_chat_completion(). Backend calls are serialized by an internal
                mutex, so single-threaded inference engines (such as the runtime
                transformer engine) can be driven directly. Everything else
                (listening port, start(), start_async(), clear()) is inherited from
                server_http / server.

            THREAD SAFETY
                on_request() runs in per-connection threads (server_http behavior);
                this class serializes on_chat_completion() calls and protects its
                own settings, so implementing backends only need to be safe for one
                generation at a time.
        !*/

    public:

        server_chat();
        /*!
            ensures
                - #model_names() == { "dlib-model" }
                - The embedded interface (default_chat_web_ui()) is served on "/".
        !*/

        void set_model_name(const std::string& name);
        void set_model_names(const std::vector<std::string>& names);
        void set_models(const std::vector<chat_model_info>& models);
        std::string model_name() const;
        std::vector<std::string> model_names() const;
        std::vector<chat_model_info> models() const;
        /*!
            ensures
                - Declare / return the models served. /v1/models lists them all;
                  a chat completion request selects one by its "model" field, the
                  first declared model being the default; the response echoes the
                  resolved id, also passed to the backend in chat_request::model.
                - The string forms declare models without a reasoning mode;
                  set_models carries the per-model reasoning capability. Empty
                  vectors are ignored.
        !*/

        void set_ui_page(const std::string& html);
        /*!
            ensures
                - Replaces the page served on "/" and "/index.html" with html.
        !*/

    protected:

        virtual chat_result on_chat_completion(const chat_request& req,
            const std::function<void(const std::string&)>& emit) = 0;
        /*!
            requires
                - req.messages is non-empty.
                - emit is callable.
            ensures
                - Produces the assistant's answer to the given conversation. Called
                  with an internal mutex held: at most one invocation runs at a
                  time, whatever the number of concurrent connections.
                - Backends should pass every produced text fragment to emit as soon
                  as it exists: for a streaming request it is forwarded to the
                  client immediately, for a plain request it is discarded, so the
                  call is unconditional. The returned chat_result stays the
                  authoritative final answer in both cases.
                - Exceptions thrown here are converted into an OpenAI error object
                  with HTTP status 500; malformed requests never reach this method
                  (they are answered with status 400 upstream).
        !*/
    };

    // ------------------------------------------------------------------------------------

    const std::string& default_chat_web_ui();
    /*!
        ensures
            - Returns the HTML of the embedded chat interface served by server_chat
              by default (defined in chat_web_ui.h). The page is self-contained
              except for the font and icon sets, referenced from public CDNs, and
              talks only to the relative endpoints of the hosting server.
    !*/
}

#endif // DLIB_SERVER_CHAT_ABSTRACT_H_
