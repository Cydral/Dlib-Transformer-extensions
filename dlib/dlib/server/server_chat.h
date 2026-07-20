// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// OpenAI-compatible chat completion server.
//
// server_chat turns any text generation backend into a web service: it serves an
// embedded browser interface on "/" and exposes the OpenAI chat completion API
// ("/v1/chat/completions", "/v1/models"), so any client of that ecosystem can talk
// to the backend. The server is stateless: every request carries the full message
// history, and the reference interface keeps conversations in the browser's own
// storage. A minimal JSON reader (chat_json) is included so the component brings
// no external dependency.

#ifndef DLIB_SERVER_CHAT_H_
#define DLIB_SERVER_CHAT_H_

#include "server_chat_abstract.h"

#include <cstdint>
#include <ctime>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "server_http.h"
#include "chat_web_ui.h"

namespace dlib
{
    // ------------------------------------------------------------------------------------

    class chat_json
    {
        /* Minimal read-only JSON document: parses the subset needed by the chat API
           (objects, arrays, strings with escapes, numbers, booleans, null) and offers
           tolerant typed accessors. Responses are emitted by hand with escape(). */
    public:
        enum class kind { null_v, boolean_v, number_v, string_v, array_v, object_v };

        chat_json() = default;

        kind type() const { return type_; }
        bool is_object() const { return type_ == kind::object_v; }
        bool is_array()  const { return type_ == kind::array_v; }
        bool is_string() const { return type_ == kind::string_v; }
        bool is_number() const { return type_ == kind::number_v; }

        bool has(const std::string& key) const
        {
            return type_ == kind::object_v && obj_.find(key) != obj_.end();
        }

        // Object member access; a shared null value is returned for absent keys.
        const chat_json& operator[](const std::string& key) const
        {
            if (type_ == kind::object_v)
            {
                auto it = obj_.find(key);
                if (it != obj_.end()) return it->second;
            }
            return null_value();
        }

        size_t size() const { return type_ == kind::array_v ? arr_.size() : 0; }
        const chat_json& operator[](size_t i) const
        {
            if (type_ == kind::array_v && i < arr_.size()) return arr_[i];
            return null_value();
        }

        std::string as_string(const std::string& def = "") const
        {
            return type_ == kind::string_v ? str_ : def;
        }
        double as_number(double def = 0.0) const
        {
            return type_ == kind::number_v ? num_ : def;
        }
        long as_long(long def = 0) const
        {
            return type_ == kind::number_v ? static_cast<long>(num_) : def;
        }
        bool as_bool(bool def = false) const
        {
            return type_ == kind::boolean_v ? (num_ != 0.0) : def;
        }

        static chat_json parse(const std::string& text)
        {
            size_t pos = 0;
            chat_json v = parse_value(text, pos);
            skip_ws(text, pos);
            if (pos != text.size())
                throw std::runtime_error("chat_json: trailing characters after document");
            return v;
        }

        // Escape a UTF-8 string for inclusion between JSON double quotes.
        static std::string escape(const std::string& s)
        {
            std::string out;
            out.reserve(s.size() + 8);
            for (unsigned char c : s)
            {
                switch (c)
                {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20)
                    {
                        static const char* hex = "0123456789abcdef";
                        out += "\\u00";
                        out += hex[(c >> 4) & 0xF];
                        out += hex[c & 0xF];
                    }
                    else out += static_cast<char>(c);
                }
            }
            return out;
        }

    private:
        static const chat_json& null_value()
        {
            static const chat_json v;
            return v;
        }

        static void skip_ws(const std::string& s, size_t& p)
        {
            while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
        }

        static char need(const std::string& s, size_t& p)
        {
            if (p >= s.size()) throw std::runtime_error("chat_json: unexpected end of document");
            return s[p];
        }

        static bool literal(const std::string& s, size_t& p, const char* word)
        {
            size_t n = 0;
            while (word[n]) ++n;
            if (s.compare(p, n, word) != 0) return false;
            p += n;
            return true;
        }

        static void append_utf8(std::string& out, unsigned long cp)
        {
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800)
            {
                out += static_cast<char>(0xC0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            else if (cp < 0x10000)
            {
                out += static_cast<char>(0xE0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            else
            {
                out += static_cast<char>(0xF0 | (cp >> 18));
                out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }

        static unsigned long hex4(const std::string& s, size_t& p)
        {
            unsigned long v = 0;
            for (int i = 0; i < 4; ++i)
            {
                const char c = need(s, p);
                ++p;
                v <<= 4;
                if (c >= '0' && c <= '9') v |= static_cast<unsigned long>(c - '0');
                else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned long>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned long>(c - 'A' + 10);
                else throw std::runtime_error("chat_json: invalid \\u escape");
            }
            return v;
        }

        static std::string parse_string(const std::string& s, size_t& p)
        {
            if (need(s, p) != '"') throw std::runtime_error("chat_json: string expected");
            ++p;
            std::string out;
            for (;;)
            {
                const char c = need(s, p);
                ++p;
                if (c == '"') return out;
                if (c != '\\') { out += c; continue; }
                const char e = need(s, p);
                ++p;
                switch (e)
                {
                case '"': case '\\': case '/': out += e; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                {
                    unsigned long cp = hex4(s, p);
                    if (cp >= 0xD800 && cp <= 0xDBFF && p + 1 < s.size()
                        && s[p] == '\\' && s[p + 1] == 'u')
                    {
                        p += 2;
                        const unsigned long lo = hex4(s, p);
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    throw std::runtime_error("chat_json: invalid escape sequence");
                }
            }
        }

        static chat_json parse_value(const std::string& s, size_t& p)
        {
            skip_ws(s, p);
            chat_json v;
            const char c = need(s, p);
            if (c == '{')
            {
                ++p;
                v.type_ = kind::object_v;
                skip_ws(s, p);
                if (need(s, p) == '}') { ++p; return v; }
                for (;;)
                {
                    skip_ws(s, p);
                    std::string key = parse_string(s, p);
                    skip_ws(s, p);
                    if (need(s, p) != ':') throw std::runtime_error("chat_json: ':' expected");
                    ++p;
                    v.obj_[key] = parse_value(s, p);
                    skip_ws(s, p);
                    const char n = need(s, p);
                    ++p;
                    if (n == '}') return v;
                    if (n != ',') throw std::runtime_error("chat_json: ',' or '}' expected");
                }
            }
            if (c == '[')
            {
                ++p;
                v.type_ = kind::array_v;
                skip_ws(s, p);
                if (need(s, p) == ']') { ++p; return v; }
                for (;;)
                {
                    v.arr_.push_back(parse_value(s, p));
                    skip_ws(s, p);
                    const char n = need(s, p);
                    ++p;
                    if (n == ']') return v;
                    if (n != ',') throw std::runtime_error("chat_json: ',' or ']' expected");
                }
            }
            if (c == '"')
            {
                v.type_ = kind::string_v;
                v.str_ = parse_string(s, p);
                return v;
            }
            if (literal(s, p, "true"))  { v.type_ = kind::boolean_v; v.num_ = 1.0; return v; }
            if (literal(s, p, "false")) { v.type_ = kind::boolean_v; v.num_ = 0.0; return v; }
            if (literal(s, p, "null"))  { v.type_ = kind::null_v; return v; }

            /* Number. */
            size_t start = p;
            if (need(s, p) == '-') ++p;
            while (p < s.size() && ((s[p] >= '0' && s[p] <= '9') || s[p] == '.'
                || s[p] == 'e' || s[p] == 'E' || s[p] == '+' || s[p] == '-')) ++p;
            if (p == start) throw std::runtime_error("chat_json: value expected");
            v.type_ = kind::number_v;
            v.num_ = std::stod(s.substr(start, p - start));
            return v;
        }

        kind type_ = kind::null_v;
        double num_ = 0.0;
        std::string str_;
        std::vector<chat_json> arr_;
        std::map<std::string, chat_json> obj_;
    };

    // ------------------------------------------------------------------------------------

    struct chat_model_info
    {
        std::string id;          // model identifier
        bool reasoning = false;  // exposes a deep-thinking mode
    };

    struct chat_message
    {
        std::string role;                    // "system", "user" or "assistant"
        std::string content;                 // concatenated text parts
        std::vector<std::string> image_urls; // image_url parts (data or http URLs)
    };

    struct chat_request
    {
        std::string model;                   // requested model id (informational)
        std::vector<chat_message> messages;  // full conversation, oldest first
        double temperature = -1.0;           // < 0 => backend default
        double top_p = -1.0;                 // < 0 => backend default
        long max_tokens = 0;                 // 0 => backend default
        bool stream = false;                 // server-sent-events streaming requested
        int reasoning = -1;                  // -1 unset, 0 disable, 1 enable deep thinking
        std::string request_id;              // client-chosen id, empty when absent
        std::function<bool()> is_cancelled;  // cooperative cancellation probe
        chat_json raw;                       // the whole parsed request body
    };

    struct chat_result
    {
        std::string content;                 // the assistant's answer
        std::string finish_reason = "stop";  // "stop" or "length"
        long prompt_tokens = 0;
        long completion_tokens = 0;
    };

    // ------------------------------------------------------------------------------------

    class server_chat : public server_http
    {
    public:

        server_chat() : ui_page_(default_chat_web_ui()) {}

        void set_model_name(const std::string& name)
        {
            set_model_names(std::vector<std::string>(1, name));
        }
        void set_model_names(const std::vector<std::string>& names)
        {
            std::vector<chat_model_info> infos;
            for (const std::string& n : names) infos.push_back(chat_model_info{ n, false });
            set_models(infos);
        }
        void set_models(const std::vector<chat_model_info>& models)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!models.empty()) models_ = models;
        }
        std::string model_name() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return models_.front().id;
        }
        std::vector<std::string> model_names() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            std::vector<std::string> names;
            for (const chat_model_info& m : models_) names.push_back(m.id);
            return names;
        }
        std::vector<chat_model_info> models() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return models_;
        }

        void set_ui_page(const std::string& html)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            ui_page_ = html;
        }

    protected:

        /* The backend: turn a parsed chat request into an answer. Calls are serialized
           by the server (one generation at a time), so a single-threaded inference
           engine can be used directly. emit hands out incremental text as it is
           produced; it forwards to the client when the request asked for streaming
           and discards otherwise, so backends can call it unconditionally. */
        virtual chat_result on_chat_completion(const chat_request& req,
            const std::function<void(const std::string&)>& emit) = 0;

    private:

        /* Connection entry point. Browsers routinely open speculative or keep-alive
           connections and close them without sending a request; the base class
           parses them, fails on the immediate end of stream and logs an error.
           Handling the connection here lets those empty connections be ignored
           silently, while real parse failures stay visible at a lower severity. */
        void on_connect(std::istream& in, std::ostream& out,
            const std::string& foreign_ip, const std::string& local_ip,
            unsigned short foreign_port, unsigned short local_port, uint64) override
        {
            if (in.peek() == EOF)
                return;   // connection closed without a request: not an error
            try
            {
                incoming_things incoming(foreign_ip, local_ip, foreign_port, local_port);
                outgoing_things outgoing;
                parse_http_request(in, incoming, get_max_content_length());
                read_body(in, incoming);
                /* Streaming completions bypass the single-response model of
                   on_request and write server-sent events to the connection as the
                   backend produces text. */
                if (incoming.request_type == "POST"
                    && strip_query(incoming.path) == "/v1/chat/completions"
                    && incoming.body.find("\"stream\"") != std::string::npos
                    && stream_completion(incoming, out))
                    return;
                const std::string result = on_request(incoming, outgoing);
                write_http_response(out, outgoing, result);
            }
            catch (http_parse_error& e)
            {
                logger_() << LINFO << "malformed request from " << foreign_ip << ": " << e.what();
                write_http_response(out, e);
            }
            catch (std::exception& e)
            {
                logger_() << LERROR << "error processing request from " << foreign_ip << ": " << e.what();
                write_http_response(out, e);
            }
        }

        static logger& logger_()
        {
            static logger l("dlib.server_chat");
            return l;
        }

        const std::string on_request(const incoming_things& incoming,
            outgoing_things& outgoing) override
        {
            /* Permissive CORS: the API is meant to be callable from pages and tools
               beyond the embedded interface. */
            outgoing.headers["Access-Control-Allow-Origin"] = "*";
            outgoing.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
            outgoing.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";

            if (incoming.request_type == "OPTIONS")
                return std::string();

            const std::string path = strip_query(incoming.path);

            if (incoming.request_type == "GET" && (path == "/" || path == "/index.html"))
            {
                outgoing.headers["Content-Type"] = "text/html; charset=utf-8";
                std::lock_guard<std::mutex> lock(state_mutex_);
                return ui_page_;
            }

            if (incoming.request_type == "GET" && path == "/favicon.ico")
            {
                outgoing.headers["Content-Type"] = "image/svg+xml";
                return "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
                       "<rect width='24' height='24' rx='6' fill='#4f8cff'/>"
                       "<text x='12' y='17' font-size='13' font-family='sans-serif'"
                       " fill='white' text-anchor='middle'>D</text></svg>";
            }

            if (incoming.request_type == "GET" && path == "/v1/models")
            {
                outgoing.headers["Content-Type"] = "application/json";
                std::ostringstream o;
                o << "{\"object\":\"list\",\"data\":[";
                const std::vector<chat_model_info> ms = models();
                for (size_t i = 0; i < ms.size(); ++i)
                    o << (i ? "," : "") << "{\"id\":\"" << chat_json::escape(ms[i].id)
                      << "\",\"object\":\"model\",\"owned_by\":\"dlib\",\"reasoning\":"
                      << (ms[i].reasoning ? "true" : "false") << "}";
                o << "]}";
                return o.str();
            }

            if (incoming.request_type == "POST" && path == "/v1/chat/completions")
                return handle_completion(incoming, outgoing);

            if (incoming.request_type == "POST" && path == "/v1/internal/cancel")
            {
                /* Marks a running completion for cooperative cancellation: the
                   backend observes chat_request::is_cancelled between generation
                   steps and returns the partial answer. */
                try
                {
                    const chat_json body = chat_json::parse(incoming.body);
                    const std::string id = body["id"].as_string();
                    if (id.empty()) throw std::runtime_error("'id' is required");
                    {
                        std::lock_guard<std::mutex> lock(cancel_mutex_);
                        cancelled_.insert(id);
                    }
                    outgoing.headers["Content-Type"] = "application/json";
                    return "{\"cancelled\":true}";
                }
                catch (const std::exception& e)
                {
                    return error_response(outgoing, 400, "invalid_request_error", e.what());
                }
            }

            return error_response(outgoing, 404, "invalid_request_error",
                "Unknown endpoint: " + path);
        }

        void parse_completion_request(const std::string& body, chat_request& req)
        {
            {
                req.raw = chat_json::parse(body);
                if (!req.raw.is_object() || !req.raw["messages"].is_array()
                    || req.raw["messages"].size() == 0)
                    throw std::runtime_error("'messages' must be a non-empty array");

                req.model = req.raw["model"].as_string();
                const chat_json& msgs = req.raw["messages"];
                for (size_t i = 0; i < msgs.size(); ++i)
                {
                    chat_message m;
                    m.role = msgs[i]["role"].as_string();
                    if (m.role.empty())
                        throw std::runtime_error("every message needs a 'role'");
                    /* content is either a plain string or an array of typed parts
                       (OpenAI multimodal form): text parts are concatenated, and
                       image_url parts are collected for image-capable backends. */
                    const chat_json& c = msgs[i]["content"];
                    if (c.is_array())
                    {
                        for (size_t p = 0; p < c.size(); ++p)
                        {
                            const std::string t = c[p]["type"].as_string();
                            if (t == "text")
                            {
                                if (!m.content.empty()) m.content += "\n";
                                m.content += c[p]["text"].as_string();
                            }
                            else if (t == "image_url")
                                m.image_urls.push_back(c[p]["image_url"]["url"].as_string());
                        }
                    }
                    else m.content = c.as_string();
                    req.messages.push_back(std::move(m));
                }
                req.request_id = req.raw["request_id"].as_string();
                if (req.raw["temperature"].is_number()) req.temperature = req.raw["temperature"].as_number();
                if (req.raw["top_p"].is_number())       req.top_p = req.raw["top_p"].as_number();
                if (req.raw["max_tokens"].is_number())  req.max_tokens = req.raw["max_tokens"].as_long();
                req.stream = req.raw["stream"].as_bool(false);
                if (req.raw["reasoning"].type() == chat_json::kind::boolean_v)
                    req.reasoning = req.raw["reasoning"].as_bool(false) ? 1 : 0;
            }

            const std::vector<std::string> names = model_names();
            std::string used = names.front();
            for (const std::string& n : names)
                if (n == req.model) { used = n; break; }
            req.model = used;

            if (!req.request_id.empty())
            {
                const std::string rid = req.request_id;
                req.is_cancelled = [this, rid]() {
                    std::lock_guard<std::mutex> lock(cancel_mutex_);
                    return cancelled_.count(rid) != 0;
                };
            }
        }

        std::string handle_completion(const incoming_things& incoming, outgoing_things& outgoing)
        {
            chat_request req;
            try
            {
                parse_completion_request(incoming.body, req);
            }
            catch (const std::exception& e)
            {
                return error_response(outgoing, 400, "invalid_request_error", e.what());
            }

            chat_result res;
            try
            {
                /* One generation at a time: the inference engines behind this server
                   assume a single generation thread. */
                std::lock_guard<std::mutex> lock(completion_mutex_);
                res = on_chat_completion(req, [](const std::string&) {});
            }
            catch (const std::exception& e)
            {
                forget_cancel(req.request_id);
                return error_response(outgoing, 500, "server_error", e.what());
            }
            forget_cancel(req.request_id);

            outgoing.headers["Content-Type"] = "application/json";
            std::ostringstream o;
            o << "{\"id\":\"chatcmpl-" << random_id()
              << "\",\"object\":\"chat.completion\",\"created\":" << std::time(nullptr)
              << ",\"model\":\"" << chat_json::escape(req.model)
              << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
              << chat_json::escape(res.content)
              << "\"},\"finish_reason\":\"" << chat_json::escape(res.finish_reason)
              << "\"}],\"usage\":{\"prompt_tokens\":" << res.prompt_tokens
              << ",\"completion_tokens\":" << res.completion_tokens
              << ",\"total_tokens\":" << (res.prompt_tokens + res.completion_tokens) << "}}";
            return o.str();
        }

        std::string error_response(outgoing_things& outgoing, unsigned short code,
            const std::string& type, const std::string& message)
        {
            outgoing.http_return = code;
            outgoing.http_return_status = (code == 400) ? "Bad Request"
                : (code == 404) ? "Not Found" : "Internal Server Error";
            outgoing.headers["Content-Type"] = "application/json";
            return "{\"error\":{\"message\":\"" + chat_json::escape(message)
                 + "\",\"type\":\"" + chat_json::escape(type) + "\"}}";
        }

        /* Server-sent-events streaming of one completion, in the OpenAI chunk
           format: role chunk, one delta chunk per emitted fragment (flushed so the
           client renders as generation goes), a final chunk carrying finish_reason
           plus the definitive cleaned answer in the additive "final_content" field,
           then the [DONE] sentinel. The connection close terminates the HTTP/1.0
           response. Returns false when the parsed request did not ask for
           streaming, letting the regular path answer it. */
        bool stream_completion(const incoming_things& incoming, std::ostream& out)
        {
            chat_request req;
            try
            {
                parse_completion_request(incoming.body, req);
            }
            catch (const std::exception& e)
            {
                outgoing_things outgoing;
                write_http_response(out, outgoing,
                    "{\"error\":{\"message\":\"" + chat_json::escape(e.what())
                    + "\",\"type\":\"invalid_request_error\"}}");
                return true;
            }
            if (!req.stream)
                return false;

            const std::string id = "chatcmpl-" + random_id();
            const std::time_t created = std::time(nullptr);
            out << "HTTP/1.0 200 OK\r\n"
                   "Content-Type: text/event-stream\r\n"
                   "Cache-Control: no-cache\r\n"
                   "Access-Control-Allow-Origin: *\r\n"
                   "Connection: close\r\n\r\n";
            out << sse_chunk(id, created, req.model, "\"role\":\"assistant\"", "null", "");
            out.flush();

            /* Byte-level tokenizers may split a multi-byte UTF-8 character across
               two emitted fragments; a chunk carrying half a character would be
               rendered as a replacement mark by the client. Bytes past the last
               complete character boundary are therefore held back and prepended to
               the next fragment. */
            std::string held;
            auto emit = [&](const std::string& delta) {
                if (delta.empty()) return;
                held += delta;
                const size_t cut = utf8_complete_prefix(held);
                if (cut == 0) return;
                out << sse_chunk(id, created, req.model,
                    "\"content\":\"" + chat_json::escape(held.substr(0, cut)) + "\"", "null", "");
                out.flush();
                held.erase(0, cut);
            };

            chat_result res;
            bool failed = false;
            try
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                res = on_chat_completion(req, emit);
            }
            catch (const std::exception& e)
            {
                res.content = std::string("Error: ") + e.what();
                res.finish_reason = "stop";
                failed = true;
            }
            forget_cancel(req.request_id);
            (void)failed;

            out << sse_chunk(id, created, req.model, "",
                "\"" + chat_json::escape(res.finish_reason) + "\"",
                ",\"final_content\":\"" + chat_json::escape(res.content) + "\"");
            out << "data: [DONE]\n\n";
            out.flush();
            return true;
        }

        /* Length of the longest prefix of s that ends on a complete UTF-8
           character. At most the last three bytes can belong to an unfinished
           sequence. */
        static size_t utf8_complete_prefix(const std::string& s)
        {
            const size_t n = s.size();
            size_t p = n;
            while (p > 0 && n - p < 4)
            {
                const unsigned char c = static_cast<unsigned char>(s[p - 1]);
                if ((c & 0x80) == 0) return p;                    // ASCII tail
                if ((c & 0xC0) == 0xC0)                            // lead byte at p-1
                {
                    const size_t len = (c & 0xF8) == 0xF0 ? 4 : (c & 0xF0) == 0xE0 ? 3 : 2;
                    return (p - 1 + len <= n) ? p - 1 + len : p - 1;
                }
                --p;                                               // continuation byte
            }
            return p;
        }

        static std::string sse_chunk(const std::string& id, std::time_t created,
            const std::string& model, const std::string& delta_body,
            const std::string& finish, const std::string& extra)
        {
            std::ostringstream o;
            o << "data: {\"id\":\"" << id
              << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
              << ",\"model\":\"" << chat_json::escape(model)
              << "\",\"choices\":[{\"index\":0,\"delta\":{" << delta_body
              << "},\"finish_reason\":" << finish << "}]" << extra << "}\n\n";
            return o.str();
        }

        void forget_cancel(const std::string& id)
        {
            if (id.empty()) return;
            std::lock_guard<std::mutex> lock(cancel_mutex_);
            cancelled_.erase(id);
        }

        static std::string strip_query(const std::string& path)
        {
            const std::string::size_type q = path.find('?');
            return q == std::string::npos ? path : path.substr(0, q);
        }

        static std::string random_id()
        {
            static const char* alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
            std::mt19937_64 rng(std::random_device{}());
            std::string id(24, '0');
            for (char& c : id) c = alphabet[rng() % 36];
            return id;
        }

        mutable std::mutex state_mutex_;
        std::mutex completion_mutex_;
        std::mutex cancel_mutex_;
        std::set<std::string> cancelled_;
        std::vector<chat_model_info> models_{ chat_model_info{ "dlib-model", false } };
        std::string ui_page_;
    };
}

#endif // DLIB_SERVER_CHAT_H_
