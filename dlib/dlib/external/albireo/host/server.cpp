// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The routes of the local HTTPS server:

        /                         the web application (files of the web folder)
        /docs/                    the interactive documentation of the interface
        /openapi.json             its OpenAPI description
        /v1/models                OpenAI compatible interface
        /v1/chat/completions
        /v1/completions
        /v1/embeddings
        /albireo/v1/...           the interface of the web application and of the tray

    The operations that change the installation (downloads, activation, settings,
    restart) require the administration token of the data folder in the
    X-Albireo-Admin header; the tray reads it there.  Every request must name the local
    server in its Host header, and a request coming from a page must come from the local
    server too, so that no web site can drive the service through the browser.
*/

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <dlib/external/httplib/httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>

#include "../common/engine_api.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "crypto.h"
#include "http_fetch.h"
#include "service.h"

namespace albireo
{

    namespace
    {
        // Bodies of the JSON routes; a model is sent to its own route without this limit.
        const uint64_t max_json_body = 64u << 20;
        // The values of a body, each taking about a hundred bytes once read: more than the
        // 300,000 tokens a request of embeddings may list.
        const size_t max_body_values = size_t(1) << 19;

        const char* const upload_route = "/albireo/v1/models/upload";

        void send_json(httplib::Response& res, int status, const json& j)
        {
            res.status = status;
            res.set_content(j.dump(), "application/json; charset=utf-8");
        }

        // The error body of the OpenAI interface.
        void send_error(httplib::Response& res, int status, const std::string& type, const std::string& message,
            const std::string& code = std::string())
        {
            json e = json::object();
            json inner = json::object();
            inner["message"] = message;
            inner["type"] = type;
            inner["param"] = nullptr;
            inner["code"] = code.empty() ? json() : json(code);
            e["error"] = inner;
            send_json(res, status, e);
        }

        /*
            The answer to a request whose model is not in the engine yet, because the
            service starts or the model loads: 503 with a code and the delay after which to
            try again.  Returns false when the model is missing for good.
        */
        bool send_not_ready(service& svc, httplib::Response& res, const std::string& requested)
        {
            const std::string phase = svc.starting_phase();
            if (!phase.empty())
            {
                send_error(res, 503, "server_error", "the service is starting (" + phase + ")", "starting");
            }
            else if (svc.model_loading(requested))
            {
                send_error(res, 503, "server_error", "the model is loading", "model_loading");
            }
            else
            {
                return false;
            }
            res.set_header("Retry-After", "5");
            return true;
        }

        json ok_result(bool ok, const std::string& error = std::string())
        {
            json o = json::object();
            o["ok"] = ok;
            if (!error.empty()) o["error"] = error;
            return o;
        }

        // The result of an operation of the windows, with the code of its error.
        json ok_result(bool ok, const user_error& error)
        {
            json o = json::object();
            o["ok"] = ok;
            if (!ok) error.add_to(o);
            return o;
        }

        bool parse_body(const httplib::Request& req, httplib::Response& res, json& body)
        {
            std::string error;
            if (req.body.empty())
            {
                body = json::object();
                return true;
            }
            if (!json::parse(req.body, body, &error, 64, max_body_values) || !body.is_object())
            {
                send_error(res, 400, "invalid_request_error", "the body is not a JSON object: " + error);
                return false;
            }
            return true;
        }

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        // Compares a secret in a time that does not depend on where the strings differ.
        bool same_secret(const std::string& a, const std::string& b)
        {
            if (a.size() != b.size()) return false;
            unsigned char d = 0;
            for (size_t i = 0; i < a.size(); ++i) d |= static_cast<unsigned char>(a[i] ^ b[i]);
            return d == 0;
        }

        bool is_local_host(const std::string& host, int port)
        {
            const std::string h = lower(host);
            const std::string p = ":" + std::to_string(port);
            return h == "localhost" || h == "127.0.0.1" || h == "[::1]" || h == "localhost" + p || h == "127.0.0.1" + p
                || h == "[::1]" + p;
        }

        bool is_local_origin(const std::string& origin, int port)
        {
            const std::string o = lower(origin);
            const std::string p = ":" + std::to_string(port);
            return o == "https://localhost" + p || o == "https://127.0.0.1" + p || o == "https://[::1]" + p;
        }

        /*
            The paths the service imports from: local drives and mounted media, never a
            network share (the service would authenticate to it) nor a device of the system.
            On Windows the service reads them with the rights of the system: the path must
            reach its drive through no link, and a folder must hold no link where the import
            reads, since a junction or a link may lead to a share, to a device or to the
            files of another user.
        */
        bool acceptable_import_path(const std::string& p)
        {
#if defined(_WIN32)
            if (p.size() < 3 || !std::isalpha(static_cast<unsigned char>(p[0])) || p[1] != ':'
                || (p[2] != '\\' && p[2] != '/'))
                return false;
            const UINT drive = GetDriveTypeW(widen(p.substr(0, 2) + "\\").c_str());
            if (drive != DRIVE_FIXED && drive != DRIVE_REMOVABLE && drive != DRIVE_CDROM && drive != DRIVE_RAMDISK)
                return false;
            if (!direct_path(p)) return false;
            if (dir_exists(p))
                for (const std::string& dir : { p, join_path(p, "models") })
                    for (const auto& n : list_dir(dir))
                        if (is_link(join_path(dir, n))) return false;
            return true;
#else
            return !p.empty() && p[0] == '/' && p.compare(0, 5, "/dev/") != 0 && p.compare(0, 6, "/proc/") != 0
                && p.compare(0, 5, "/sys/") != 0;
#endif
        }

        struct event_channel
        {
            std::mutex m;
            std::condition_variable cv;
            std::deque<std::pair<int, std::string>> q;
            bool finished = false;

            void push(int kind, const std::string& data)
            {
                std::lock_guard<std::mutex> lock(m);
                q.emplace_back(kind, data);
                if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR) finished = true;
                cv.notify_all();
            }

            bool pop(std::pair<int, std::string>& e, int ms)
            {
                std::unique_lock<std::mutex> lock(m);
                if (!cv.wait_for(lock, std::chrono::milliseconds(ms), [&]() { return !q.empty(); })) return false;
                e = std::move(q.front());
                q.pop_front();
                return true;
            }
        };

        /*
            Whether the client of a request left: its connection closed, or ended by a TLS
            close_notify, whose bytes wait on the socket where the check of httplib sees a live
            client.  The TLS probe is the one httplib makes for the answers it streams, and
            gives back to the next read the byte it takes from a request that follows.
        */
        bool client_left(const httplib::Request& req)
        {
            if (req.is_connection_closed()) return true;
#if defined(CPPHTTPLIB_MBEDTLS_SUPPORT)
            auto session = static_cast<httplib::tls::impl::MbedTlsSession*>(const_cast<void*>(req.ssl));
            return session && session->sock != INVALID_SOCKET
                && httplib::tls::is_peer_closed(session, session->sock);
#else
            return false;
#endif
        }

        /*
            Ends the engine request id of an answer sent whole when the service stops, with
            503, or when its client left.  Looks at them once a second at most, whether the
            events of the engine flow or not; returns true when it ended the request.
        */
        bool end_abandoned(service& svc, const httplib::Request& req, httplib::Response& res, uint64_t id,
            std::chrono::steady_clock::time_point& next_look)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_look) return false;
            next_look = now + std::chrono::seconds(1);
            if (svc.stopping())
            {
                svc.engine().cancel(id);
                send_error(res, 503, "server_error", "the service is stopping");
                return true;
            }
            // A client that left ends its request, whose place in the engine the others take.
            if (client_left(req))
            {
                svc.engine().cancel(id);
                return true;
            }
            return false;
        }

        // The engine request of a completion of the OpenAI interface.
        bool engine_request(service& svc, const json& body, json& out, std::string& error, std::string& model_name)
        {
            const std::string requested = body["model"].as_string();
            const std::string model = svc.resolve_model(requested);
            if (model.empty())
            {
                error = requested.empty() || requested == "auto" ? std::string("no model is active")
                    : "the model '" + requested + "' is not active";
                return false;
            }
            model_name = requested.empty() ? model : requested;
            if (body["n"].as_int(1) != 1)
            {
                error = "only one choice per request (n = 1) is supported";
                return false;
            }
            out = json::object();
            out["model"] = model;
            static const char* const copied[] = { "messages", "prompt", "tools", "tool_choice", "max_tokens",
                "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "stop", "seed", "presence_penalty",
                "frequency_penalty", "repeat_penalty", "repetition_penalty", "repeat_last_n", "logit_bias",
                "reasoning_effort", "reasoning", "chat_template_kwargs" };
            for (const char* k : copied)
                if (body.contains(k)) out[k] = body[k];
            if (body["think"].is_bool()) out["reasoning"] = body["think"];
            return true;
        }

        json usage_of(const json& done)
        {
            const json& u = done["usage"];
            json usage = json::object();
            usage["prompt_tokens"] = u["prompt_tokens"];
            usage["completion_tokens"] = u["completion_tokens"];
            usage["total_tokens"] = u["total_tokens"];
            json details = json::object();
            details["cached_tokens"] = u["cached_tokens"];
            usage["prompt_tokens_details"] = details;
            return usage;
        }

        json tool_call_of(const json& c, int index)
        {
            json call = json::object();
            if (index >= 0) call["index"] = index;
            call["id"] = c["id"];
            call["type"] = "function";
            json fn = json::object();
            fn["name"] = c["name"];
            fn["arguments"] = c["arguments"];
            call["function"] = fn;
            return call;
        }

        void handle_completion(service& svc, const httplib::Request& req, httplib::Response& res, bool chat)
        {
            json body;
            if (!parse_body(req, res, body)) return;
            json request;
            std::string error, model_name;
            if (!engine_request(svc, body, request, error, model_name))
            {
                // A model resolved and an invalid option, or no model in the engine.
                if (!model_name.empty()) send_error(res, 400, "invalid_request_error", error);
                else if (!send_not_ready(svc, res, body["model"].as_string()))
                    send_error(res, 404, "invalid_request_error", error, "model_not_found");
                return;
            }
            if (chat && !request["messages"].is_array())
            {
                send_error(res, 400, "invalid_request_error", "messages must be an array");
                return;
            }
            if (!chat && !request["prompt"].is_string())
            {
                send_error(res, 400, "invalid_request_error", "prompt must be a string");
                return;
            }
            const bool stream = body["stream"].as_bool(false);
            const bool include_usage = body["stream_options"]["include_usage"].as_bool(false);
            auto ch = std::make_shared<event_channel>();
            uint64_t id = 0;
            auto sink = [ch](int kind, const std::string& data) { ch->push(kind, data); };
            if (!svc.engine().submit(request, sink, id, error))
            {
                const bool overflow = error.find("context") != std::string::npos;
                send_error(res, 400, "invalid_request_error", error, overflow ? "context_length_exceeded" : "");
                return;
            }
            const std::string cid = std::string(chat ? "chatcmpl-" : "cmpl-") + random_hex(24);
            const long long created = static_cast<long long>(now_unix());

            if (!stream)
            {
                std::string content, reasoning;
                json calls = json::array();
                json done;
                bool failed = false;
                auto next_look = std::chrono::steady_clock::now();
                for (;;)
                {
                    std::pair<int, std::string> e;
                    if (ch->pop(e, 1000))
                    {
                        if (e.first == ALBIREO_EVENT_CONTENT) content += e.second;
                        else if (e.first == ALBIREO_EVENT_REASONING) reasoning += e.second;
                        else if (e.first == ALBIREO_EVENT_TOOL_CALL)
                            calls.push_back(tool_call_of(json::parse_or_null(e.second), -1));
                        else if (e.first == ALBIREO_EVENT_DONE || e.first == ALBIREO_EVENT_ERROR)
                        {
                            done = json::parse_or_null(e.second);
                            failed = e.first == ALBIREO_EVENT_ERROR;
                            break;
                        }
                    }
                    if (end_abandoned(svc, req, res, id, next_look)) return;
                }
                if (failed)
                {
                    send_error(res, 500, "server_error", done["message"].as_string(), done["code"].as_string());
                    return;
                }
                svc.record_usage(static_cast<long>(done["usage"]["prompt_tokens"].as_int(0)),
                    static_cast<long>(done["usage"]["completion_tokens"].as_int(0)));
                json out = json::object();
                out["id"] = cid;
                out["object"] = chat ? "chat.completion" : "text_completion";
                out["created"] = created;
                out["model"] = model_name;
                json choice = json::object();
                choice["index"] = 0;
                if (chat)
                {
                    json msg = json::object();
                    msg["role"] = "assistant";
                    msg["content"] = content.empty() && !calls.empty() ? json() : json(content);
                    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
                    if (!calls.empty()) msg["tool_calls"] = calls;
                    choice["message"] = msg;
                }
                else
                {
                    choice["text"] = content;
                }
                choice["logprobs"] = nullptr;
                choice["finish_reason"] = done["finish_reason"];
                json choices = json::array();
                choices.push_back(choice);
                out["choices"] = choices;
                out["usage"] = usage_of(done);
                out["timings"] = done["timings"];
                out["system_fingerprint"] = std::string("albireo-") + ALBIREO_VERSION;
                send_json(res, 200, out);
                return;
            }

            // The post-routing handler marks the answers of the interface "no-store".
            res.set_header("X-Accel-Buffering", "no");
            auto first = std::make_shared<bool>(true);
            auto tool_index = std::make_shared<int>(0);
            service* s = &svc;
            res.set_chunked_content_provider("text/event-stream",
                [ch, cid, created, model_name, chat, include_usage, first, tool_index, s, id](
                    size_t, httplib::DataSink& sink) {
                    auto chunk = [&](const json& delta, const json& finish) {
                        json c = json::object();
                        c["id"] = cid;
                        c["object"] = chat ? "chat.completion.chunk" : "text_completion";
                        c["created"] = created;
                        c["model"] = model_name;
                        json choice = json::object();
                        choice["index"] = 0;
                        if (chat) choice["delta"] = delta;
                        else choice["text"] = delta["content"].is_string() ? delta["content"] : json("");
                        choice["logprobs"] = nullptr;
                        choice["finish_reason"] = finish;
                        json choices = json::array();
                        choices.push_back(choice);
                        c["choices"] = choices;
                        const std::string line = "data: " + c.dump() + "\n\n";
                        return sink.write(line.data(), line.size());
                    };
                    // A service that stops ends the answers in progress, whose tokens may flow
                    // without pause.
                    if (s->stopping())
                    {
                        s->engine().cancel(id);
                        return false;
                    }
                    std::pair<int, std::string> e;
                    if (!ch->pop(e, 2000))
                    {
                        if (!sink.is_writable() || s->stopping())
                        {
                            s->engine().cancel(id);
                            return false;
                        }
                        static const char ping[] = ": keep-alive\n\n";
                        return sink.write(ping, sizeof(ping) - 1);
                    }
                    if (*first && chat)
                    {
                        *first = false;
                        json delta = json::object();
                        delta["role"] = "assistant";
                        delta["content"] = "";
                        if (!chunk(delta, json())) return false;
                    }
                    json delta = json::object();
                    switch (e.first)
                    {
                    case ALBIREO_EVENT_CONTENT:
                        delta["content"] = e.second;
                        return chunk(delta, json());
                    case ALBIREO_EVENT_REASONING:
                        delta["reasoning_content"] = e.second;
                        return chunk(delta, json());
                    case ALBIREO_EVENT_TOOL_CALL:
                    {
                        json list = json::array();
                        list.push_back(tool_call_of(json::parse_or_null(e.second), (*tool_index)++));
                        delta["tool_calls"] = list;
                        return chunk(delta, json());
                    }
                    case ALBIREO_EVENT_PROGRESS:
                    {
                        // The progress of a long prompt, as a comment line the clients of
                        // the standard ignore and the web application shows.
                        const std::string line = ": progress " + e.second + "\n\n";
                        return sink.write(line.data(), line.size());
                    }
                    case ALBIREO_EVENT_DONE:
                    {
                        const json done = json::parse_or_null(e.second);
                        s->record_usage(static_cast<long>(done["usage"]["prompt_tokens"].as_int(0)),
                            static_cast<long>(done["usage"]["completion_tokens"].as_int(0)));
                        if (!chunk(json::object(), done["finish_reason"])) return false;
                        if (include_usage)
                        {
                            json c = json::object();
                            c["id"] = cid;
                            c["object"] = chat ? "chat.completion.chunk" : "text_completion";
                            c["created"] = created;
                            c["model"] = model_name;
                            c["choices"] = json::array();
                            c["usage"] = usage_of(done);
                            c["timings"] = done["timings"];
                            const std::string line = "data: " + c.dump() + "\n\n";
                            if (!sink.write(line.data(), line.size())) return false;
                        }
                        static const char end[] = "data: [DONE]\n\n";
                        sink.write(end, sizeof(end) - 1);
                        sink.done();
                        return true;
                    }
                    default:
                    {
                        const json err = json::parse_or_null(e.second);
                        json body = json::object();
                        json inner = json::object();
                        inner["message"] = err["message"];
                        inner["type"] = "server_error";
                        inner["code"] = err["code"];
                        body["error"] = inner;
                        const std::string line = "data: " + body.dump() + "\n\n";
                        sink.write(line.data(), line.size());
                        sink.done();
                        return true;
                    }
                    }
                },
                [ch, s, id](bool) {
                    std::lock_guard<std::mutex> lock(ch->m);
                    if (!ch->finished) s->engine().cancel(id);
                });
        }

        // Checks a change of the settings sent by the tray; the keys outside this list are
        // set by the installer only.
        bool validate_settings(const json& body, std::string& error)
        {
            static const char* const keys[] = { "port", "language", "engine", "accelerator", "v2ram", "v2ram_max_mb",
                "device_mb", "threads", "context", "max_sequences", "kv_mb", "check_updates", "offline", "proxy" };
            for (const auto& kv : body.items())
            {
                if (std::find_if(std::begin(keys), std::end(keys), [&](const char* k) { return kv.first == k; })
                    == std::end(keys))
                {
                    error = "the setting '" + kv.first + "' cannot be changed through this interface";
                    return false;
                }
            }
            auto one_of = [&](const char* key, std::initializer_list<const char*> values) {
                if (!body.contains(key)) return true;
                for (const char* v : values)
                    if (body[key].as_string() == v) return true;
                error = std::string("invalid value for ") + key;
                return false;
            };
            auto in_range = [&](const char* key, long long lo, long long hi) {
                if (!body.contains(key)) return true;
                if (body[key].is_integer() && body[key].as_int() >= lo && body[key].as_int() <= hi) return true;
                error = std::string("invalid value for ") + key;
                return false;
            };
            auto boolean = [&](const char* key) {
                if (!body.contains(key) || body[key].is_bool()) return true;
                error = std::string("invalid value for ") + key;
                return false;
            };
            const json& proxy = body["proxy"];
            if (body.contains("proxy") && !(proxy.is_string() && valid_proxy_setting(proxy.as_string())))
            {
                error = "invalid value for proxy: http://host:port, \"none\" or empty";
                return false;
            }
            return one_of("language", { "auto", "en", "fr", "de", "it", "es" })
                && one_of("engine", { "auto", "best", "cpu", "cpu-avx2", "cpu-avxvnni", "cpu-avx512", "openvino",
                    "cuda" })
                && one_of("accelerator", { "auto", "none", "npu", "gpu", "both" })
                && in_range("port", 1024, 65535) && in_range("threads", 0, 1024) && in_range("context", 0, 1 << 20)
                && in_range("max_sequences", 1, 64) && in_range("kv_mb", 0, 1 << 24)
                && in_range("v2ram_max_mb", 0, 1 << 24) && in_range("device_mb", 0, 1 << 24) && boolean("v2ram")
                && boolean("check_updates")
                && boolean("offline");
        }

        // The floats of the base64 form of a vector.
        std::vector<float> floats_of(const std::string& text)
        {
            const std::string bytes = base64_decode(text);
            std::vector<float> v(bytes.size() / sizeof(float));
            if (!v.empty()) std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
            return v;
        }

        /*
            The vectors of texts, as the OpenAI interface computes them: "input" a text or a
            list of texts (or of token lists), "encoding_format" float or base64, and
            "dimensions" for the models that keep their meaning when shortened.  The
            requests of every client share the passes of the embedding model.
        */
        void handle_embeddings(service& svc, const httplib::Request& req, httplib::Response& res)
        {
            json body;
            if (!parse_body(req, res, body)) return;
            const std::string requested = body["model"].as_string();
            const std::string model = svc.resolve_embedding_model(requested);
            if (model.empty())
            {
                const bool alias = requested.empty() || requested == "auto" || requested == "embedding"
                    || requested == "albireo-embedding";
                if (send_not_ready(svc, res, alias ? std::string("embedding") : requested)) return;
                send_error(res, 404, "invalid_request_error", requested.empty() || requested == "auto"
                    ? std::string("no embedding model is active")
                    : "the embedding model '" + requested + "' is not active", "model_not_found");
                return;
            }
            const std::string format = body["encoding_format"].str("float");
            if (format != "float" && format != "base64")
            {
                send_error(res, 400, "invalid_request_error", "encoding_format must be float or base64");
                return;
            }
            json request = json::object();
            request["model"] = model;
            request["input"] = body["input"];
            if (body.contains("dimensions")) request["dimensions"] = body["dimensions"];
            auto ch = std::make_shared<event_channel>();
            uint64_t id = 0;
            std::string error;
            if (!svc.engine().submit(request, [ch](int kind, const std::string& data) { ch->push(kind, data); }, id,
                    error))
            {
                send_error(res, 400, "invalid_request_error", error);
                return;
            }
            std::pair<int, std::string> e;
            auto next_look = std::chrono::steady_clock::now();
            for (;;)
            {
                if (ch->pop(e, 1000) && (e.first == ALBIREO_EVENT_DONE || e.first == ALBIREO_EVENT_ERROR)) break;
                if (end_abandoned(svc, req, res, id, next_look)) return;
            }
            const json done = json::parse_or_null(e.second);
            if (e.first == ALBIREO_EVENT_ERROR)
            {
                send_error(res, 500, "server_error", done["message"].as_string(), done["code"].as_string());
                return;
            }
            json data = json::array();
            for (const auto& d : done["data"].elements())
            {
                json item = json::object();
                item["object"] = "embedding";
                item["index"] = d["index"];
                if (format == "base64")
                {
                    item["embedding"] = d["embedding"];
                }
                else
                {
                    json values = json::array();
                    for (float x : floats_of(d["embedding"].as_string())) values.push_back(static_cast<double>(x));
                    item["embedding"] = values;
                }
                data.push_back(item);
            }
            svc.record_usage(static_cast<long>(done["usage"]["prompt_tokens"].as_int(0)), 0);
            json out = json::object();
            out["object"] = "list";
            out["data"] = data;
            out["model"] = requested.empty() ? model : requested;
            json usage = json::object();
            usage["prompt_tokens"] = done["usage"]["prompt_tokens"];
            usage["total_tokens"] = done["usage"]["total_tokens"];
            out["usage"] = usage;
            send_json(res, 200, out);
        }

        uint64_t content_length(const httplib::Request& req)
        {
            const std::string v = req.get_header_value("Content-Length");
            return v.empty() ? 0 : std::strtoull(v.c_str(), nullptr, 10);
        }

        // Refuses an operation that needs the network on a computer set offline.
        bool refuse_offline(service& svc, httplib::Response& res)
        {
            if (!svc.settings().get().offline) return false;
            send_json(res, 409, ok_result(false, user_error().set("offline mode: the catalog, the updates and the "
                "models come from files (import)", "offline")));
            return true;
        }
    }

    void install_routes(httplib::Server& srv, service& svc)
    {
        const int port = svc.port();
        service* s = &svc;

        // Every request names the local server, and the pages that call the interfaces
        // are those of the local server.
        srv.set_pre_routing_handler([port](const httplib::Request& req, httplib::Response& res) {
            auto refuse = [&](int status, const char* message) {
                res.status = status;
                res.set_header("Connection", "close");
                res.set_content(message, "text/plain");
                return httplib::Server::HandlerResponse::Handled;
            };
            if (!is_local_host(req.get_header_value("Host"), port)) return refuse(421, "misdirected request");
            const std::string origin = req.get_header_value("Origin");
            if (!origin.empty() && !is_local_origin(origin, port))
                return refuse(403, "cross-origin requests are refused");
            // The methods that send no data take no body, which httplib would read whole before
            // their handlers run.  Any other body is JSON of a known length, of 64 MB at most,
            // apart from the models sent to their own route.
            const bool no_body = req.method == "GET" || req.method == "HEAD" || req.method == "DELETE"
                || req.method == "OPTIONS";
            if (no_body && (req.has_header("Transfer-Encoding") || content_length(req) > 0))
                return refuse(400, "this method takes no body");
            if (!(req.method == "POST" && req.path == upload_route))
            {
                if (req.has_header("Transfer-Encoding")) return refuse(411, "the body must have a length");
                if (content_length(req) > max_json_body) return refuse(413, "the body is too large");
                if (content_length(req) > 0 && req.get_header_value("Content-Type").find("application/json") != 0)
                    return refuse(415, "the body must be application/json");
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        srv.set_post_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("X-Content-Type-Options", "nosniff");
            res.set_header("Referrer-Policy", "no-referrer");
            res.set_header("Cross-Origin-Opener-Policy", "same-origin");
            if (req.path == "/sandbox.html")
            {
                /*
                    The page in which the web application runs the code of the agent: framed
                    by the application only, in an iframe whose sandbox gives it an origin of
                    its own, so that the code reaches neither the data of the application nor
                    the network.  Its script and the workers it starts may evaluate code.
                */
                res.set_header("X-Frame-Options", "SAMEORIGIN");
                res.set_header("Cross-Origin-Resource-Policy", "same-origin");
                res.set_header("Content-Security-Policy", "default-src 'none'; script-src 'self' blob: 'unsafe-eval'; "
                    "worker-src blob:; style-src 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'; "
                    "form-action 'none'");
                return;
            }
            res.set_header("X-Frame-Options", "DENY");
            // The script of the sandbox is loaded by a page of another (opaque) origin.
            const bool sandbox = req.path == "/js/sandbox.js";
            res.set_header("Cross-Origin-Resource-Policy", sandbox ? "cross-origin" : "same-origin");
            if (req.path.compare(0, 5, "/docs") == 0)
                res.set_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'");
            else
                res.set_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; "
                    "style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self' data:; connect-src 'self'; "
                    "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'");
            if (req.path.compare(0, 4, "/v1/") == 0 || req.path.compare(0, 11, "/albireo/v1") == 0)
                res.set_header("Cache-Control", "no-store");
        });
        srv.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "internal error";
            try
            {
                std::rethrow_exception(ep);
            }
            catch (const std::exception& e)
            {
                what = e.what();
            }
            catch (...)
            {
            }
            log_error("http: " + req.method + " " + req.path + ": " + what);
            send_error(res, 500, "server_error", what);
        });

        auto admin = [s](const httplib::Request& req, httplib::Response& res) {
            const std::string token = s->admin_token();
            if (!token.empty() && same_secret(req.get_header_value("X-Albireo-Admin"), token)) return true;
            res.set_header("Connection", "close");
            send_error(res, 401, "authentication_error", "this operation requires the administration token");
            return false;
        };

        // OpenAI compatible interface.
        srv.Get("/v1/models", [s](const httplib::Request&, httplib::Response& res) {
            json list = json::array();
            const host_settings st = s->settings().get();
            const long long now = static_cast<long long>(now_unix());
            for (const auto& id : s->engine().loaded_models())
            {
                json m = json::object();
                m["id"] = id;
                m["object"] = "model";
                m["created"] = now;
                m["owned_by"] = "albireo";
                json roles = json::array();
                for (const auto& kv : st.active)
                    if (kv.second == id) roles.push_back(kv.first);
                m["roles"] = roles;
                list.push_back(m);
            }
            for (const char* role : model_roles)
            {
                const std::string target = s->resolve_model(role);
                if (target.empty()) continue;
                json m = json::object();
                m["id"] = std::string("albireo-") + role;
                m["object"] = "model";
                m["created"] = now;
                m["owned_by"] = "albireo";
                m["alias_of"] = target;
                list.push_back(m);
            }
            const std::string background = s->resolve_model("background");
            if (!background.empty())
            {
                json m = json::object();
                m["id"] = "albireo-background";
                m["object"] = "model";
                m["created"] = now;
                m["owned_by"] = "albireo";
                m["alias_of"] = background;
                list.push_back(m);
            }
            const std::string embedder = s->resolve_embedding_model(std::string());
            if (!embedder.empty())
            {
                json m = json::object();
                m["id"] = std::string("albireo-") + embedding_role;
                m["object"] = "model";
                m["created"] = now;
                m["owned_by"] = "albireo";
                m["alias_of"] = embedder;
                list.push_back(m);
            }
            json out = json::object();
            out["object"] = "list";
            out["data"] = list;
            send_json(res, 200, out);
        });
        srv.Post("/v1/chat/completions", [s](const httplib::Request& req, httplib::Response& res) {
            handle_completion(*s, req, res, true);
        });
        srv.Post("/v1/completions", [s](const httplib::Request& req, httplib::Response& res) {
            handle_completion(*s, req, res, false);
        });
        srv.Post("/v1/embeddings", [s](const httplib::Request& req, httplib::Response& res) {
            handle_embeddings(*s, req, res);
        });

        // Interface of the web application and of the tray.
        srv.Get("/albireo/v1/status", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->status());
        });
        srv.Get("/albireo/v1/models", [s](const httplib::Request& req, httplib::Response& res) {
            std::string lang = req.get_param_value("lang");
            if (lang.empty() || lang == "auto") lang = "en";
            json o = s->models().overview(lang);
            const host_settings st = s->settings().get();
            o["free"] = static_cast<unsigned long long>(free_disk_space(s->models().models_dir()));
            o["offline"] = st.offline;
            json active = json::object(), selected = json::object();
            for (const auto& kv : s->effective_models(st)) active[kv.first] = kv.second;
            for (const auto& kv : st.active) selected[kv.first] = kv.second;
            o["active"] = active;
            o["selected"] = selected;
            json loaded = json::array();
            for (const auto& id : s->engine().loaded_models()) loaded.push_back(id);
            o["loaded"] = loaded;
            o["leaving"] = s->leaving_models();
            o["background"] = s->background_model();
            send_json(res, 200, o);
        });
        srv.Get("/albireo/v1/usage", [s](const httplib::Request&, httplib::Response& res) {
            json o = json::object();
            o["days"] = s->db().usage(30);
            send_json(res, 200, o);
        });
        srv.Get("/albireo/v1/engines", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->engines());
        });
        srv.Get("/albireo/v1/updates", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->updates().status());
        });
        srv.Post("/albireo/v1/count", [s](const httplib::Request& req, httplib::Response& res) {
            json body;
            if (!parse_body(req, res, body)) return;
            json q = json::object();
            q["op"] = body["messages"].is_array() ? "count" : "tokenize";
            q["model"] = s->resolve_model(body["model"].as_string());
            if (body["messages"].is_array()) q["messages"] = body["messages"];
            if (body["tools"].is_array()) q["tools"] = body["tools"];
            if (body["text"].is_string()) q["text"] = body["text"];
            json r = s->engine().query(q);
            if (q["op"].as_string() == "tokenize" && r["tokens"].is_array())
                r["tokens"] = static_cast<long long>(r["tokens"].size());
            send_json(res, r.contains("error") ? 400 : 200, r);
        });

        // The catalog is fetched in the background; its state is in /albireo/v1/models.
        srv.Post("/albireo/v1/models/refresh", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res) || refuse_offline(*s, res)) return;
            s->models().refresh_catalog_async();
            json o = ok_result(true);
            o["catalog"] = s->models().catalog_status();
            send_json(res, 202, o);
        });
        srv.Post(R"(/albireo/v1/models/([A-Za-z0-9._-]+)/download)", [s, admin](const httplib::Request& req,
                                                                               httplib::Response& res) {
            if (!admin(req, res) || refuse_offline(*s, res)) return;
            user_error error;
            const bool ok = s->models().start_download(req.matches[1], error);
            send_json(res, ok ? 202 : 400, ok_result(ok, error));
        });
        srv.Delete(R"(/albireo/v1/models/([A-Za-z0-9._-]+)/download)", [s, admin](const httplib::Request& req,
                                                                                 httplib::Response& res) {
            if (!admin(req, res)) return;
            s->models().cancel_download(req.matches[1]);
            send_json(res, 200, ok_result(true));
        });
        srv.Post(R"(/albireo/v1/models/([A-Za-z0-9._-]+)/activate)", [s, admin](const httplib::Request& req,
                                                                               httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            user_error error;
            const bool ok = s->activate(req.matches[1], body["role"].str("general"), error);
            send_json(res, ok ? 200 : 400, ok_result(ok, error));
        });
        srv.Post(R"(/albireo/v1/models/([A-Za-z0-9._-]+)/deactivate)", [s, admin](const httplib::Request& req,
                                                                                 httplib::Response& res) {
            if (!admin(req, res)) return;
            user_error error;
            const bool ok = s->deactivate(req.matches[1], error);
            send_json(res, ok ? 200 : 400, ok_result(ok, error));
        });
        srv.Post(R"(/albireo/v1/models/([A-Za-z0-9._-]+)/verify)", [s, admin](const httplib::Request& req,
                                                                             httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 200, s->models().verify(req.matches[1]));
        });
        srv.Delete(R"(/albireo/v1/models/([A-Za-z0-9._-]+))", [s, admin](const httplib::Request& req,
                                                                        httplib::Response& res) {
            if (!admin(req, res)) return;
            const std::string id = req.matches[1];
            user_error error;
            bool pending = false;
            const bool ok = s->remove_model(id, error, pending);
            json o = ok_result(ok, error);
            if (ok) o["pending"] = pending;
            send_json(res, ok ? 200 : 400, o);
        });
        // A folder or a file on a local drive or a medium: models, and the signed catalog
        // and update of an offline bundle.
        auto import = [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            const std::string path = body["path"].as_string();
            if (!acceptable_import_path(path))
            {
                send_error(res, 400, "invalid_request_error", "the path must be a local folder or file");
                return;
            }
            send_json(res, 200, s->import_folder(path));
        };
        srv.Post("/albireo/v1/import", import);
        srv.Post("/albireo/v1/models/import", import);
        // A model file sent by the tray, for the files the service cannot read itself.
        srv.Post(upload_route, [s, admin](const httplib::Request& req, httplib::Response& res,
                                    const httplib::ContentReader& reader) {
            if (!admin(req, res)) return;
            const std::string name = file_name(req.get_param_value("name"));
            const uint64_t length = content_length(req);
            if (name.empty() || length == 0 || req.has_header("Transfer-Encoding"))
            {
                res.set_header("Connection", "close");
                send_error(res, 400, "invalid_request_error", "a file name and a length are required");
                return;
            }
            const std::string dir = s->models().models_dir();
            if (free_disk_space(dir) < length + (256u << 20))
            {
                res.set_header("Connection", "close");
                send_error(res, 507, "server_error", "not enough free space in " + dir);
                return;
            }
            const std::string tmp = join_path(dir, ".transfer-" + random_hex(16) + ".partial");
            std::ofstream out;
#if defined(_WIN32)
            out.open(widen(tmp).c_str(), std::ios::binary | std::ios::trunc);
#else
            out.open(tmp, std::ios::binary | std::ios::trunc);
#endif
            sha256_stream hash;
            uint64_t received = 0;
            const bool ok = out && reader([&](const char* data, size_t n) {
                out.write(data, static_cast<std::streamsize>(n));
                hash.update(data, n);
                received += n;
                return static_cast<bool>(out) && received <= length && !s->stopping();
            });
            out.close();
            if (!ok || received != length)
            {
                remove_file(tmp);
                res.set_header("Connection", "close");
                send_error(res, 400, "invalid_request_error", "the transfer did not complete");
                return;
            }
            const json r = s->models().import_received(tmp, name, hash.hex());
            send_json(res, r["imported"].empty() ? 400 : 200, r);
        });

        // The password of the proxy is given as proxy_password_mask (host_settings::to_json()).
        srv.Get("/albireo/v1/settings", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 200, s->settings().get().to_json());
        });
        srv.Put("/albireo/v1/settings", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            std::string error;
            if (!validate_settings(body, error))
            {
                send_error(res, 400, "invalid_request_error", error);
                return;
            }
            // Read through a const reference, which adds no member to the body.
            const json& changes = body;
            // A new port is opened once before it is saved, so that a port taken by another
            // program or reserved by the system never leaves the service without an address.
            const long long port = changes["port"].as_int(s->port());
            if (port != s->port() && !port_available(static_cast<int>(port)))
            {
                send_error(res, 409, "invalid_request_error", "the port " + std::to_string(port)
                    + " is in use or cannot be opened", "port_unavailable");
                return;
            }
            host_settings before;
            bool masked_elsewhere = false;
            const host_settings after = s->settings().update([&](host_settings& current) {
                before = current;
                // The mask of the password stands for the stored password, with the address and
                // the user it was given for only: it never sends the password to another server.
                const std::string proxy = changes["proxy"].as_string();
                if (proxy_password_masked(proxy) && proxy != masked_proxy(current.proxy))
                {
                    masked_elsewhere = true;
                    return;
                }
                json merged = current.to_json();
                for (const auto& kv : body.items()) merged[kv.first] = kv.second;
                current.from_json(merged);
                if (current.proxy == masked_proxy(before.proxy)) current.proxy = before.proxy;
            });
            if (masked_elsewhere)
            {
                send_error(res, 400, "invalid_request_error", "the masked password of the proxy applies to its "
                    "stored address and user only: type the password again", "proxy_password");
                return;
            }
            s->apply_network_settings(after);
            // What the running engine or server cannot take without a restart.
            const bool restart = after.port != before.port || after.engine != before.engine
                || after.accelerator != before.accelerator || after.v2ram != before.v2ram
                || after.v2ram_max_mb != before.v2ram_max_mb || after.device_mb != before.device_mb
                || after.threads != before.threads
                || after.context != before.context || after.max_sequences != before.max_sequences
                || after.kv_mb != before.kv_mb;
            json o = ok_result(true);
            o["restart"] = restart;
            send_json(res, 200, o);
            if (restart) s->request_restart();
        });
        srv.Post("/albireo/v1/updates/check", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res) || refuse_offline(*s, res)) return;
            s->updates().check_async();
            send_json(res, 202, s->updates().status());
        });
        srv.Post("/albireo/v1/updates/apply", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            // The staged installer runs at the start of the service: a restart applies it.
            if (s->updates().staged_installer().empty())
            {
                send_json(res, 400, ok_result(false, "no update is ready"));
                return;
            }
            send_json(res, 202, ok_result(true));
            s->request_restart();
        });
        srv.Post("/albireo/v1/restart", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 202, ok_result(true));
            s->request_restart();
        });
        srv.Post("/albireo/v1/engines/benchmark", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 200, s->benchmark_engines());
        });
        srv.Get("/albireo/v1/logs", [admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json lines = json::array();
            for (const auto& l : log_file::instance().recent()) lines.push_back(l);
            json o = json::object();
            o["lines"] = lines;
            send_json(res, 200, o);
        });

        // The web application, the documentation viewer and their assets.
        const std::string web = join_path(svc.program_dir(), "web");
        if (!srv.set_mount_point("/", web)) log_warning("http: no web folder at " + web);
        srv.set_file_extension_and_mimetype_mapping("mjs", "text/javascript");
        srv.set_file_extension_and_mimetype_mapping("woff2", "font/woff2");
        srv.set_file_extension_and_mimetype_mapping("webmanifest", "application/manifest+json");
        srv.set_file_request_handler([](const httplib::Request& req, httplib::Response& res) {
            // The pages are checked at each load, so that an update shows at once.
            if (req.path == "/" || (req.path.size() > 5 && req.path.compare(req.path.size() - 5, 5, ".html") == 0))
                res.set_header("Cache-Control", "no-cache");
        });
        srv.Get("/docs", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/docs/"); });
    }

}
