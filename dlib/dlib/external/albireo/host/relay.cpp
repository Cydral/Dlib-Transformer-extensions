// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "relay.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <utility>

#include "../common/engine_api.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "certificates.h"
#include "crypto.h"
#include "http_fetch.h"

namespace albireo
{

    const char* const relay_model_id = "albireo-relay";

    namespace
    {
        struct known_model
        {
            const char* id;
            const char* name;
            long context;
            long max_output;
            bool images;
            const char* thinking;
            const char* efforts;            // separated by spaces
            const char* default_effort;
        };

        /*
            The seven latest models of the interface of Anthropic that every key reaches, the newest
            first, with their published context windows and outputs (platform.claude.com, November
            2026): each reads images, thinks as it needs (adaptive thinking), and takes five levels of
            effort, its default being the level it takes without one.
        */
        const known_model claude_models[] = {
            { "claude-fable-5-2", "Claude Fable 5.2", 1000000, 128000, true, "always", "low medium high xhigh max",
                "high" },
            { "claude-haiku-5-5", "Claude Haiku 5.5", 1000000, 128000, true, "always", "low medium high xhigh max",
                "medium" },
            { "claude-sonnet-5-5", "Claude Sonnet 5.5", 1000000, 128000, true, "always", "low medium high xhigh max",
                "high" },
            { "claude-opus-5-5", "Claude Opus 5.5", 1000000, 128000, true, "always", "low medium high xhigh max",
                "medium" },
            { "claude-fable-5-1", "Claude Fable 5.1", 1000000, 128000, true, "always", "low medium high xhigh max",
                "high" },
            { "claude-opus-5", "Claude Opus 5", 1000000, 128000, true, "always", "low medium high xhigh max", "high" },
            { "claude-sonnet-5", "Claude Sonnet 5", 1000000, 128000, true, "always", "low medium high xhigh max",
                "high" },
        };

        /*
            The models of the cloud of Ollama of the lowest prices per token (ollama.com/pricing, November
            2026), which an account without subscription pays with its credits, under the names of its
            interface (ollama.com/api/tags), with the context windows of their pages.  The tokens of their
            answers bound the max_tokens of the requests that give one.
        */
        const known_model ollama_models[] = {
            { "gpt-oss:20b", "gpt-oss 20B", 131072, 32768, false, "always", "low medium high", "medium" },
            { "gpt-oss:120b", "gpt-oss 120B", 131072, 32768, false, "always", "low medium high", "medium" },
            { "gemma4:31b", "Gemma 4 31B", 262144, 32768, true, "toggle", "", "" },
            { "nemotron-3-super", "Nemotron 3 Super", 262144, 32768, false, "toggle", "", "" },
            { "glm-5.3-flash", "GLM-5.3 Flash", 1000000, 32768, true, "always", "low high max", "" },
        };

        // The levels of effort of an engine of the interface of OpenAI whose model reasons.
        const char* const openai_efforts = "low medium high";

        /*
            The tokens of an answer of Claude, its reasoning included, when the request gives none: the
            documentation of Anthropic advises 64000 to start with at the high levels of effort.
        */
        const long long claude_default_tokens = 64000;

        std::mutex origins_m;
        std::map<std::string, std::string> test_origins;

        // Thrown from the socket of a client about to connect once its answer or the requests of the service end.
        struct answer_cancelled
        {
        };

        std::vector<std::string> split_words(const char* text)
        {
            std::vector<std::string> words;
            std::string w;
            for (const char* p = text; ; ++p)
            {
                if (*p == ' ' || *p == '\0')
                {
                    if (!w.empty()) words.push_back(w);
                    w.clear();
                    if (*p == '\0') break;
                }
                else
                {
                    w += *p;
                }
            }
            return words;
        }

        relay_model model_of(const known_model& k)
        {
            relay_model m;
            m.id = k.id;
            m.name = k.name;
            m.context = k.context;
            m.max_output = k.max_output;
            m.images = k.images;
            m.thinking = k.thinking;
            m.efforts = split_words(k.efforts);
            m.default_effort = k.default_effort;
            return m;
        }

        json model_json(const relay_model& m)
        {
            json j = json::object();
            j["id"] = m.id;
            j["name"] = m.name;
            j["context"] = static_cast<long long>(m.context);
            j["max_output"] = static_cast<long long>(m.max_output);
            j["images"] = m.images;
            j["thinking"] = m.thinking;
            json efforts = json::array();
            for (const std::string& e : m.efforts) efforts.push_back(e);
            j["efforts"] = efforts;
            j["default_effort"] = m.default_effort;
            return j;
        }

        const char* provider_name(const std::string& provider)
        {
            if (provider == "claude") return "Claude";
            if (provider == "ollama") return "Ollama";
            return "OpenAI";
        }

        // The authentication a provider takes: the key alone for those whose address relay.cpp knows.
        std::string auth_of(const relay_settings& r)
        {
            return r.provider == "openai" ? r.auth : std::string("key");
        }

        bool has_effort(const relay_model& m, const std::string& level)
        {
            return std::find(m.efforts.begin(), m.efforts.end(), level) != m.efforts.end();
        }

        // The headers of a request to the provider of r, the secret included.
        httplib::Headers headers_of(const relay_settings& r)
        {
            httplib::Headers h;
            const std::string auth = auth_of(r);
            if (r.provider == "claude")
            {
                h.emplace("x-api-key", r.secret);
                h.emplace("anthropic-version", "2023-06-01");
            }
            else if (auth == "key")
            {
                h.emplace("Authorization", "Bearer " + r.secret);
            }
            else if (auth == "basic")
            {
                h.emplace("Authorization", "Basic " + base64_encode(r.user + ":" + r.secret));
            }
            h.emplace("Accept", "application/json, text/event-stream");
            return h;
        }

        /*
            A text of the external engine without the secret, the user and the host of the settings,
            which an error may quote: a gateway that echoes the header of the authentication.
        */
        std::string redacted(std::string text, const relay_settings& r)
        {
            std::vector<std::string> hidden;
            if (r.secret.size() >= 4) hidden.push_back(r.secret);
            if (!r.secret.empty() && !r.user.empty()) hidden.push_back(base64_encode(r.user + ":" + r.secret));
            if (r.user.size() >= 3) hidden.push_back(r.user);
            url_parts u;
            if (r.provider == "openai" && parse_url(r.url, u))
            {
                const size_t host = u.origin.find("://");
                if (host != std::string::npos && u.origin.size() > host + 3)
                {
                    // The host with its port, then alone.
                    const std::string named = u.origin.substr(host + 3);
                    hidden.push_back(named);
                    const size_t colon = named.rfind(':');
                    if (colon != std::string::npos && named.find(']', colon) == std::string::npos && colon >= 3)
                        hidden.push_back(named.substr(0, colon));
                }
            }
            for (const std::string& h : hidden)
                for (size_t at = text.find(h); at != std::string::npos; at = text.find(h, at + 8))
                    text.replace(at, h.size(), "********");
            return text;
        }

        // The message of an error that an external engine answered, without the request it may quote.
        std::string provider_error(int status, const std::string& body, const relay_settings& r)
        {
            const json j = json::parse_or_null(body.substr(0, 64 * 1024));
            std::string message;
            if (j["error"].is_string()) message = j["error"].as_string();
            else if (j["error"]["message"].is_string()) message = j["error"]["message"].as_string();
            else if (j["message"].is_string()) message = j["message"].as_string();
            message = redacted(message, r);
            if (message.size() > 300) message = message.substr(0, 300) + "...";
            std::string what;
            if (status == 401 || status == 403) what = "the external engine refuses the key or the user";
            else if (status == 404) what = "the external engine does not know the model or the address";
            else if (status == 413) what = "the external engine refuses a request this large";
            else if (status == 429) what = "the external engine limits the requests or the credits of the account";
            else if (status >= 500) what = "the external engine is overloaded or failing";
            else what = "the external engine refuses the request";
            return what + " (HTTP " + std::to_string(status) + ")" + (message.empty() ? "" : ": " + message);
        }

        std::string transfer_error(const httplib::Error e)
        {
            switch (e)
            {
            case httplib::Error::Connection: return "the external engine cannot be reached";
            case httplib::Error::ConnectionTimeout: return "the external engine does not accept the connection";
            case httplib::Error::Read: return "the answer of the external engine was cut";
            case httplib::Error::Write: return "the request could not be sent to the external engine";
            case httplib::Error::SSLConnection: return "the secure connection to the external engine failed";
            case httplib::Error::SSLServerVerification:
                return "the certificate of the external engine is not trusted, as when a filter of the network "
                    "inspects the connections";
            case httplib::Error::SSLServerHostnameVerification:
                return "the certificate of the external engine names another server";
            case httplib::Error::Timeout: return "the external engine stopped answering";
            default: return std::string("the request to the external engine failed: ") + httplib::to_string(e);
            }
        }

        /*
            The events of a stream of server-sent events: each event, once its blank line arrives,
            goes to the receiver with its name and its data, the lines "data:" joined by "\n".
        */
        class sse_reader
        {
        public:
            // Returns false when an event is larger than max_event bytes.
            bool feed(const char* data, size_t n, const std::function<void(const std::string&, const std::string&)>& f)
            {
                buf_.append(data, n);
                size_t start = 0;
                for (;;)
                {
                    const size_t eol = buf_.find('\n', start);
                    if (eol == std::string::npos) break;
                    std::string line = buf_.substr(start, eol - start);
                    start = eol + 1;
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty())
                    {
                        if (has_data_) f(event_, data_);
                        event_.clear();
                        data_.clear();
                        has_data_ = false;
                        continue;
                    }
                    if (line[0] == ':') continue;
                    const size_t colon = line.find(':');
                    const std::string field = line.substr(0, colon);
                    std::string value = colon == std::string::npos ? std::string() : line.substr(colon + 1);
                    if (!value.empty() && value[0] == ' ') value.erase(0, 1);
                    if (field == "event")
                    {
                        event_ = value;
                    }
                    else if (field == "data")
                    {
                        if (has_data_) data_ += '\n';
                        data_ += value;
                        has_data_ = true;
                    }
                }
                buf_.erase(0, start);
                return buf_.size() + data_.size() <= max_event;
            }

            // The event left without its blank line when the stream ends.
            void finish(const std::function<void(const std::string&, const std::string&)>& f)
            {
                static const char blank[] = "\n\n";
                feed(blank, 2, f);
            }

        private:
            static const size_t max_event = size_t(16) << 20;
            std::string buf_, event_, data_;
            bool has_data_ = false;
        };

        // A part of a message as the relay sends it: a text, or an image as a data URL.
        struct part
        {
            bool image = false;
            std::string text;
        };

        struct message
        {
            std::string role;       // system, user or assistant
            std::vector<part> parts;
        };

        /*
            The messages of a request in the three roles the providers share: the developer
            messages join the system ones, the calls of tools of the earlier answers go, and the
            results of tools become texts of the user, since the relay offers no tool.
        */
        std::vector<message> messages_of(const json& request)
        {
            std::vector<message> out;
            for (const json& m : request["messages"].elements())
            {
                message msg;
                const std::string role = m["role"].as_string();
                msg.role = role == "developer" ? "system" : role == "tool" ? "user" : role;
                if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") continue;
                const json& content = m["content"];
                if (content.is_string())
                {
                    part p;
                    p.text = content.as_string();
                    msg.parts.push_back(p);
                }
                for (const json& c : content.elements())
                {
                    const std::string kind = c["type"].as_string();
                    part p;
                    if (kind == "text")
                    {
                        p.text = c["text"].as_string();
                    }
                    else if (kind == "image_url")
                    {
                        const json& image = c["image_url"];
                        p.image = true;
                        p.text = image.is_string() ? image.as_string() : image["url"].as_string();
                    }
                    else
                    {
                        continue;
                    }
                    msg.parts.push_back(p);
                }
                if (role == "tool" && !msg.parts.empty() && !msg.parts[0].image)
                    msg.parts[0].text = "[The result of a tool]\n" + msg.parts[0].text;
                bool empty = true;
                for (const part& p : msg.parts) empty = empty && !p.image && p.text.empty();
                if (!empty) out.push_back(msg);
            }
            return out;
        }

        // The media type and the base64 data of a data URL of an image.
        bool split_data_url(const std::string& url, std::string& media_type, std::string& data)
        {
            const size_t comma = url.find(',');
            if (url.compare(0, 5, "data:") != 0 || comma == std::string::npos) return false;
            const std::string header = url.substr(5, comma - 5);
            const size_t semicolon = header.find(';');
            media_type = header.substr(0, semicolon);
            data = url.substr(comma + 1);
            return !media_type.empty();
        }

        // The body of a request of the interface of OpenAI, for an engine of that interface or Ollama.
        json openai_body(const relay_settings& r, const relay_model& m, const std::vector<message>& msgs,
            const json& request, const std::string& effort, bool reasoning_off)
        {
            json body = json::object();
            body["model"] = m.id;
            json list = json::array();
            for (const message& msg : msgs)
            {
                json one = json::object();
                one["role"] = msg.role;
                if (msg.parts.size() == 1 && !msg.parts[0].image)
                {
                    one["content"] = msg.parts[0].text;
                }
                else
                {
                    json parts = json::array();
                    for (const part& p : msg.parts)
                    {
                        json c = json::object();
                        if (p.image)
                        {
                            c["type"] = "image_url";
                            json url = json::object();
                            url["url"] = p.text;
                            c["image_url"] = url;
                        }
                        else
                        {
                            c["type"] = "text";
                            c["text"] = p.text;
                        }
                        parts.push_back(c);
                    }
                    one["content"] = parts;
                }
                list.push_back(one);
            }
            body["messages"] = list;
            body["stream"] = true;
            json options = json::object();
            options["include_usage"] = true;
            body["stream_options"] = options;
            // The sampling passes as numbers alone, which hold no text of the discussion.
            for (const char* k : { "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty" })
                if (request[k].is_number()) body[k] = request[k];
            /*
                The stop sequences, texts, leave only without pseudonymization: they could name a
                person, and could not match an answer written with the substitutes anyway.
            */
            const json& stop = request["stop"];
            bool texts = stop.is_string() || (stop.is_array() && stop.size() <= 4);
            for (const json& one : stop.elements()) texts = texts && one.is_string();
            if (!r.pseudonymize && texts) body["stop"] = stop;
            // The bound of the answer when the request gives one, within the tokens of the model.
            for (const char* k : { "max_tokens", "max_completion_tokens" })
            {
                const long long n = request[k].as_int(0);
                if (n > 0) body[k] = std::min<long long>(n, m.max_output > 0 ? m.max_output : n);
            }
            if (!effort.empty()) body["reasoning_effort"] = effort;
            else if (reasoning_off && r.provider == "ollama") body["reasoning_effort"] = "none";
            return body;
        }

        // The body of a request of the interface of Anthropic.
        json claude_body(const relay_model& m, const std::vector<message>& msgs, const json& request,
            const std::string& effort)
        {
            json body = json::object();
            body["model"] = m.id;
            // The tokens of an answer, its reasoning included: those of the request, else 64000.
            long long max_tokens = std::min<long long>(claude_default_tokens, m.max_output);
            for (const char* k : { "max_tokens", "max_completion_tokens" })
            {
                const long long n = request[k].as_int(0);
                if (n > 0) max_tokens = std::min<long long>(n, m.max_output);
            }
            body["max_tokens"] = max_tokens;
            std::string system;
            json list = json::array();
            for (const message& msg : msgs)
            {
                if (msg.role == "system")
                {
                    for (const part& p : msg.parts)
                        if (!p.image && !p.text.empty()) system += (system.empty() ? "" : "\n\n") + p.text;
                    continue;
                }
                json blocks = json::array();
                for (const part& p : msg.parts)
                {
                    json b = json::object();
                    std::string media_type, data;
                    if (p.image && split_data_url(p.text, media_type, data))
                    {
                        b["type"] = "image";
                        json source = json::object();
                        source["type"] = "base64";
                        source["media_type"] = media_type;
                        source["data"] = data;
                        b["source"] = source;
                    }
                    else if (!p.image && !p.text.empty())
                    {
                        b["type"] = "text";
                        b["text"] = p.text;
                    }
                    else
                    {
                        continue;
                    }
                    blocks.push_back(b);
                }
                if (blocks.size() == 0) continue;
                // The discussion opens with the user, and the messages of one role in a row join.
                if (list.size() == 0 && msg.role != "user") continue;
                if (list.size() > 0 && list.elements().back()["role"].as_string() == msg.role)
                {
                    for (const json& b : blocks.elements()) list.elements().back()["content"].push_back(b);
                    continue;
                }
                json one = json::object();
                one["role"] = msg.role;
                one["content"] = blocks;
                list.push_back(one);
            }
            if (!system.empty()) body["system"] = system;
            body["messages"] = list;
            body["stream"] = true;
            // The reasoning in its summary, which the models of this generation omit unless asked.
            json thinking = json::object();
            thinking["type"] = "adaptive";
            thinking["display"] = "summarized";
            body["thinking"] = thinking;
            if (!effort.empty())
            {
                json output = json::object();
                output["effort"] = effort;
                body["output_config"] = output;
            }
            return body;
        }

        std::string finish_of_claude(const std::string& stop)
        {
            if (stop == "max_tokens" || stop == "model_context_window_exceeded") return "length";
            if (stop == "refusal") return "content_filter";
            return "stop";
        }

        json error_event(const std::string& message, const std::string& code)
        {
            json e = json::object();
            e["message"] = message;
            e["code"] = code;
            return e;
        }
    }

    // ----------------------------------------------------------------------------------------

    std::vector<relay_model> known_relay_models(const std::string& provider)
    {
        std::vector<relay_model> list;
        if (provider == "claude")
            for (const known_model& k : claude_models) list.push_back(model_of(k));
        if (provider == "ollama")
            for (const known_model& k : ollama_models) list.push_back(model_of(k));
        return list;
    }

    bool valid_relay_url(const std::string& url)
    {
        url_parts u;
        if (url.empty() || url.size() > 2048 || !parse_url(url, u)) return false;
        for (const char c : url)
            if (static_cast<unsigned char>(c) <= ' ' || c == '"' || c == '<' || c == '>' || c == '\\') return false;
        return u.target.find('?') == std::string::npos && u.target.find('#') == std::string::npos;
    }

    std::string relay_base_url(const relay_settings& r)
    {
        {
            std::lock_guard<std::mutex> lock(origins_m);
            auto it = test_origins.find(r.provider);
            if (it != test_origins.end()) return it->second + "/v1";
        }
        if (r.provider == "claude") return "https://api.anthropic.com/v1";
        if (r.provider == "ollama") return "https://ollama.com/v1";
        std::string url = r.url;
        while (!url.empty() && url.back() == '/') url.pop_back();
        return url;
    }

    void relay_origin_for_tests(const std::string& provider, const std::string& origin)
    {
        std::lock_guard<std::mutex> lock(origins_m);
        if (origin.empty()) test_origins.erase(provider);
        else test_origins[provider] = origin;
    }

    bool relay_model_of(const relay_settings& r, relay_model& m)
    {
        if (r.model.empty()) return false;
        if (r.provider == "openai")
        {
            m = relay_model();
            m.id = r.model;
            m.name = r.model;
            m.context = r.context;
            m.max_output = r.max_output;
            m.images = r.images;
            m.thinking = r.reasoning ? "always" : "none";
            if (r.reasoning) m.efforts = split_words(openai_efforts);
            return true;
        }
        for (const relay_model& k : known_relay_models(r.provider))
            if (k.id == r.model)
            {
                m = k;
                return true;
            }
        return false;
    }

    bool relay_ready(const relay_settings& r, std::string& why)
    {
        relay_model m;
        const std::string auth = auth_of(r);
        if (!r.enabled) why = "the relay is turned off";
        else if (r.provider == "openai" && !valid_relay_url(r.url)) why = "the relay has no valid address";
        else if (r.model.empty()) why = "the relay has no model";
        else if (!relay_model_of(r, m)) why = "the model of the relay is not one of its provider here";
        else if (auth == "basic" && r.user.empty()) why = "the relay has no user";
        else if (auth != "none" && r.secret.empty()) why = auth == "basic" ? "the relay has no password"
            : "the relay has no key";
        else
        {
            why.clear();
            return true;
        }
        return false;
    }

    json relay_catalog()
    {
        json providers = json::array();
        for (const char* id : { "openai", "ollama", "claude" })
        {
            relay_settings r;
            r.provider = id;
            json p = json::object();
            p["id"] = id;
            p["name"] = provider_name(id);
            p["url"] = std::string(id) == "openai" ? std::string() : relay_base_url(r);
            json auths = json::array();
            if (std::string(id) == "openai")
                for (const char* a : { "key", "basic", "none" }) auths.push_back(a);
            else
                auths.push_back("key");
            p["auths"] = auths;
            json models = json::array();
            for (const relay_model& m : known_relay_models(id)) models.push_back(model_json(m));
            p["models"] = models;
            providers.push_back(p);
        }
        json j = json::object();
        j["providers"] = providers;
        return j;
    }

    json relay_description(const relay_settings& r)
    {
        json j = json::object();
        std::string why;
        const bool ready = relay_ready(r, why);
        j["enabled"] = r.enabled;
        j["ready"] = ready;
        j["reason"] = why;
        j["provider"] = r.provider;
        j["provider_name"] = provider_name(r.provider);
        relay_model m;
        if (relay_model_of(r, m))
        {
            const json traits = model_json(m);
            for (const auto& kv : traits.items()) j[kv.first == "id" ? "model" : kv.first] = kv.second;
        }
        else
        {
            j["model"] = r.model;
        }
        j["pseudonymize"] = r.pseudonymize;
        j["tags"] = r.tags;
        // What the relay was set for, its address included, which this fingerprint stands for without naming it.
        j["scope"] = sha256_hex(r.secret_scope()).substr(0, 16);
        return j;
    }

    // ----------------------------------------------------------------------------------------

    struct relay_service::job
    {
        uint64_t id = 0;
        json request;
        sink_type sink;
        // The cancellation of the answer and the connection it holds, to Pseudalis or to the external engine.
        cancellable control;
        std::atomic<bool> done{false};
        std::thread thread;
    };

    relay_service::relay_service(settings_source settings, image_describer describe)
        : settings_(std::move(settings)), describe_(std::move(describe))
    {
    }

    relay_service::~relay_service()
    {
        stop();
    }

    void relay_service::reap()
    {
        for (auto it = jobs_.begin(); it != jobs_.end();)
        {
            if ((*it)->done)
            {
                if ((*it)->thread.joinable()) (*it)->thread.join();
                it = jobs_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    bool relay_service::submit(const json& request, sink_type sink, uint64_t& id, std::string& error, int& status,
        std::string& code)
    {
        const host_settings st = settings_();
        auto refuse = [&](int s, const char* c, const std::string& e) {
            status = s;
            code = c;
            error = e;
            return false;
        };
        std::string why;
        if (!relay_ready(st.relay, why)) return refuse(404, "relay_unavailable", why);
        if (st.offline) return refuse(503, "offline", "offline mode: the relay sends nothing out of this computer");
        if (request["tools"].size() > 0)
            return refuse(400, "tools_not_supported", "the relay takes no tools: the agent mode and the web access "
                "stay with the models of the computer");
        if (request["n"].as_int(1) != 1)
            return refuse(400, "invalid_value", "only one choice per request (n = 1) is supported");
        if (!request["messages"].is_array() || request["messages"].size() == 0)
            return refuse(400, "invalid_value", "messages must be a non-empty array");
        for (const json& m : request["messages"].elements())
            for (const json& c : m["content"].elements())
                if (c["type"].as_string() == "input_audio")
                    return refuse(400, "audio_input_not_supported", "the relay sends no sound");
        std::lock_guard<std::mutex> lock(jobs_m_);
        if (stopped_) return refuse(503, "server_error", "the service is stopping");
        reap();
        if (jobs_.size() >= max_jobs)
            return refuse(503, "engine_busy", "the relay answers " + std::to_string(max_jobs)
                + " requests at once at most: try again in a moment");
        auto j = std::make_shared<job>();
        j->id = next_id_++;
        j->request = request;
        j->sink = std::move(sink);
        id = j->id;
        // An exception of the answer ends it with its error, never the service.
        auto answer = [this, j]() {
            try
            {
                run(j);
            }
            catch (const std::exception& x)
            {
                if (!j->done) j->sink(ALBIREO_EVENT_ERROR, error_event(std::string("the relay failed: ") + x.what(),
                    "relay_failed").dump());
            }
            catch (...)
            {
                if (!j->done) j->sink(ALBIREO_EVENT_ERROR, error_event("the relay failed", "relay_failed").dump());
            }
            j->request = json();
            j->done = true;
        };
        // A thread the system refuses leaves no job behind.
        try
        {
            j->thread = std::thread(answer);
        }
        catch (const std::exception&)
        {
            return refuse(503, "engine_busy", "the relay cannot start an answer now: try again in a moment");
        }
        jobs_.push_back(j);
        return true;
    }

    void relay_service::cancel(uint64_t id)
    {
        std::shared_ptr<job> j;
        {
            std::lock_guard<std::mutex> lock(jobs_m_);
            for (const auto& one : jobs_)
                if (one->id == id) j = one;
        }
        if (j) j->control.cancel();
    }

    void relay_service::stop()
    {
        std::list<std::shared_ptr<job>> all;
        {
            std::lock_guard<std::mutex> lock(jobs_m_);
            stopped_ = true;
            all = jobs_;
        }
        for (const auto& j : all) j->control.cancel();
        for (const auto& j : all)
            if (j->thread.joinable()) j->thread.join();
        {
            std::lock_guard<std::mutex> lock(jobs_m_);
            jobs_.clear();
        }
        forget_pseudonyms();
    }

    json relay_service::pseudonymization_status()
    {
        json j = pseudalis_.status();
        j["memory"] = memory_.stats();
        return j;
    }

    void relay_service::forget_pseudonyms()
    {
        memory_.clear();
        // The descriptions of the images, which may hold the values they show.
        std::lock_guard<std::mutex> lock(descriptions_m_);
        for (auto& d : descriptions_) wipe_secret(d.second);
        descriptions_.clear();
    }

    void relay_service::run(const std::shared_ptr<job>& j)
    {
        const host_settings st = settings_();
        const relay_settings& r = st.relay;
        const json& request = j->request;
        const auto started = std::chrono::steady_clock::now();
        // The log names the provider and the code of a failure, never its message, which may quote the request.
        auto fail = [&](const std::string& message, const std::string& code) {
            if (code != "cancelled") log_warning("relay: an answer of " + r.provider + " failed (" + code + ")");
            j->sink(ALBIREO_EVENT_ERROR, error_event(message, code).dump());
            j->done = true;
        };
        relay_model m;
        std::string why;
        if (!relay_ready(r, why) || !relay_model_of(r, m))
        {
            fail(why.empty() ? std::string("the relay changed its model") : why, "relay_unavailable");
            return;
        }
        std::vector<message> msgs = messages_of(request);

        /*
            The images go to the external model when it reads them and no pseudonymization holds
            them back; otherwise a model of the computer describes each of them, and its
            description, a text, takes its place.
        */
        const bool send_images = m.images && !r.pseudonymize;
        size_t pictures = 0;
        for (message& msg : msgs)
            for (part& p : msg.parts)
            {
                if (!p.image || send_images) continue;
                if (j->control.cancelled())
                {
                    fail("the answer was cancelled", "cancelled");
                    return;
                }
                const std::string key = sha256_hex(p.text);
                std::string description, error;
                {
                    std::lock_guard<std::mutex> lock(descriptions_m_);
                    for (auto it = descriptions_.begin(); it != descriptions_.end(); ++it)
                        if (it->first == key)
                        {
                            description = it->second;
                            descriptions_.splice(descriptions_.begin(), descriptions_, it);
                            break;
                        }
                }
                if (description.empty())
                {
                    if (!describe_ || !describe_(p.text, j->control.flag(), description, error))
                    {
                        const bool cancelled = j->control.cancelled();
                        fail(cancelled ? std::string("the answer was cancelled")
                            : "the images of the discussion cannot be described on this computer: " + error,
                            cancelled ? "cancelled" : "image_input_not_supported");
                        return;
                    }
                    std::lock_guard<std::mutex> lock(descriptions_m_);
                    descriptions_.emplace_front(key, description);
                    if (descriptions_.size() > max_descriptions)
                    {
                        wipe_secret(descriptions_.back().second);
                        descriptions_.pop_back();
                    }
                }
                p.image = false;
                p.text = "[Picture " + std::to_string(++pictures) + ", described on the computer of the user: "
                    + description + "]";
            }

        // The texts leave pseudonymized, the values of their substitutes kept here for the answer.
        std::map<std::string, std::string> values;
        if (r.pseudonymize)
        {
            for (message& msg : msgs)
                for (part& p : msg.parts)
                {
                    if (p.image || p.text.empty()) continue;
                    std::string out, error;
                    if (j->control.cancelled() || !memory_.protect(p.text, r.tags, pseudalis_, &j->control, out,
                            values, error))
                    {
                        const bool cancelled = j->control.cancelled();
                        fail(cancelled ? std::string("the answer was cancelled") : error,
                            cancelled ? "cancelled" : "pseudonymization_failed");
                        return;
                    }
                    p.text = std::move(out);
                }
        }
        stream_restorer content_restorer(values), reasoning_restorer(values);
        for (auto& v : values) wipe_secret(v.second);
        values.clear();

        // The reasoning: the level of effort of the request, or the reasoning turned on or off.
        std::string effort = request["reasoning_effort"].as_string();
        const json& on = request["reasoning"].is_bool() ? request["reasoning"] : request["think"];
        bool reasoning_off = false;
        if (!has_effort(m, effort)) effort.clear();
        if (effort.empty() && m.thinking == "toggle" && on.is_bool())
        {
            if (on.as_bool()) effort = "high";
            else reasoning_off = true;
        }
        const bool claude = r.provider == "claude";
        const json body = claude ? claude_body(m, msgs, request, effort)
                                 : openai_body(r, m, msgs, request, effort, reasoning_off);
        msgs.clear();

        url_parts u;
        if (!parse_url(relay_base_url(r) + (claude ? "/messages" : "/chat/completions"), u))
        {
            fail("the relay has no valid address", "relay_unavailable");
            return;
        }
        std::string error;
        std::shared_ptr<httplib::Client> client = http_client(u.origin, 300, error);
        if (!client)
        {
            fail(error.empty() ? std::string("the external engine cannot be reached") : error, "relay_unreachable");
            return;
        }
        // The connection opens for an answer that goes on alone, whatever the moment of its cancellation.
        const cancellable* control = &j->control;
        client->set_socket_options([control](socket_t sock) {
            if (!http_cancelled() && !control->cancelled()) return;
            httplib::detail::close_socket(sock);
            throw answer_cancelled();
        });
        http_registration registered(client);
        // An answer cancelled while it was prepared sends nothing; one cancelled since stops its client.
        j->control.hold(client);
        if (j->control.cancelled())
        {
            j->control.release();
            fail("the answer was cancelled", "cancelled");
            return;
        }

        // The answer as it streams, and its end.
        std::string finish = "stop", provider_failure;
        long long prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
        bool first = true, ended = false;
        // The events of the stream read, and an answer sent whole, which some engines give despite "stream".
        size_t events = 0;
        bool streamed = true;
        std::string whole;
        auto first_token = started;
        auto emit = [&](int kind, const std::string& text) {
            if (text.empty()) return;
            if (first)
            {
                first = false;
                first_token = std::chrono::steady_clock::now();
            }
            const std::string out = kind == ALBIREO_EVENT_REASONING ? reasoning_restorer.feed(text)
                                                                     : content_restorer.feed(text);
            if (!out.empty()) j->sink(kind, out);
        };
        auto on_event = [&](const std::string& name, const std::string& data) {
            if (!claude && data == "[DONE]")
            {
                ended = true;
                return;
            }
            const json e = json::parse_or_null(data);
            if (!e.is_object()) return;
            ++events;
            if (claude)
            {
                const std::string type = e["type"].is_string() ? e["type"].as_string() : name;
                if (type == "message_start")
                {
                    const json& usage = e["message"]["usage"];
                    cached_tokens = usage["cache_read_input_tokens"].as_int(0);
                    prompt_tokens = usage["input_tokens"].as_int(0) + cached_tokens
                        + usage["cache_creation_input_tokens"].as_int(0);
                }
                else if (type == "content_block_delta")
                {
                    const json& d = e["delta"];
                    const std::string kind = d["type"].as_string();
                    if (kind == "text_delta") emit(ALBIREO_EVENT_CONTENT, d["text"].as_string());
                    else if (kind == "thinking_delta") emit(ALBIREO_EVENT_REASONING, d["thinking"].as_string());
                }
                else if (type == "message_delta")
                {
                    if (e["delta"]["stop_reason"].is_string())
                        finish = finish_of_claude(e["delta"]["stop_reason"].as_string());
                    completion_tokens = e["usage"]["output_tokens"].as_int(completion_tokens);
                }
                else if (type == "message_stop")
                {
                    ended = true;
                }
                else if (type == "error")
                {
                    provider_failure = "the external engine ended its answer with an error: "
                        + redacted(e["error"]["message"].as_string(), r).substr(0, 300);
                }
                return;
            }
            if (e["error"].is_object() || e["error"].is_string())
            {
                provider_failure = "the external engine ended its answer with an error: "
                    + redacted(e["error"].is_string() ? e["error"].as_string() : e["error"]["message"].as_string(), r)
                    .substr(0, 300);
                return;
            }
            const json& choice = e["choices"][0];
            const json& d = choice["delta"];
            if (d["reasoning_content"].is_string()) emit(ALBIREO_EVENT_REASONING, d["reasoning_content"].as_string());
            else if (d["reasoning"].is_string()) emit(ALBIREO_EVENT_REASONING, d["reasoning"].as_string());
            if (d["content"].is_string()) emit(ALBIREO_EVENT_CONTENT, d["content"].as_string());
            if (choice["finish_reason"].is_string())
            {
                const std::string f = choice["finish_reason"].as_string();
                finish = f == "length" || f == "content_filter" ? f : "stop";
            }
            const json& usage = e["usage"];
            if (usage.is_object())
            {
                prompt_tokens = usage["prompt_tokens"].as_int(prompt_tokens);
                completion_tokens = usage["completion_tokens"].as_int(completion_tokens);
                cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_int(cached_tokens);
            }
        };

        httplib::Request req;
        req.method = "POST";
        req.path = u.target;
        req.headers = headers_of(r);
        req.body = body.dump();
        req.set_header("Content-Type", "application/json");
        int status = 0;
        std::string error_body;
        sse_reader reader;
        bool too_large = false;
        /*
            A stream of events, by its type, or by its first bytes when its type names no stream
            and no JSON: some engines stream without the type of the events.
        */
        std::string type;
        bool sniffed = false;
        req.response_handler = [&](const httplib::Response& res) {
            status = res.status;
            type = res.get_header_value("Content-Type");
            std::transform(type.begin(), type.end(), type.begin(),
                [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
            streamed = type.find("text/event-stream") != std::string::npos;
            sniffed = streamed || type.find("json") != std::string::npos;
            return !j->control.cancelled();
        };
        req.content_receiver = [&](const char* data, size_t n, uint64_t, uint64_t) {
            if (j->control.cancelled()) return false;
            if (status != 200)
            {
                if (error_body.size() < 64 * 1024) error_body.append(data, std::min<size_t>(n, 64 * 1024));
                return true;
            }
            if (!sniffed)
            {
                whole.append(data, n);
                const size_t first = whole.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) return true;
                sniffed = true;
                for (const char* field : { "data:", "event:", "id:", "retry:", ":" })
                    streamed = streamed || whole.compare(first, std::strlen(field), field) == 0;
                if (!streamed) return true;
                const std::string start = whole;
                whole.clear();
                if (!reader.feed(start.data(), start.size(), on_event))
                {
                    too_large = true;
                    return false;
                }
                return provider_failure.empty();
            }
            if (!streamed)
            {
                whole.append(data, n);
                too_large = whole.size() > (size_t(16) << 20);
                return !too_large;
            }
            if (!reader.feed(data, n, on_event))
            {
                too_large = true;
                return false;
            }
            return provider_failure.empty();
        };
        httplib::Response res;
        httplib::Error transfer = httplib::Error::Success;
        bool sent = false;
        try
        {
            sent = client->send(req, res, transfer);
        }
        catch (...)
        {
            // The cancellation of the requests of the service, before the connection.
            transfer = httplib::Error::Canceled;
        }
        req.body.clear();
        req.headers.clear();
        j->control.release();
        if (status == 200 && sent && streamed) reader.finish(on_event);
        if (status == 200 && sent && !streamed && !too_large)
        {
            /*
                An answer sent whole: the completion of the interface of OpenAI, or the message of
                Anthropic, read as the last events of a stream would be.
            */
            const json a = json::parse_or_null(whole);
            wipe_secret(whole);
            if (claude && a["content"].is_array())
            {
                ++events;
                for (const json& b : a["content"].elements())
                {
                    if (b["type"].as_string() == "thinking") emit(ALBIREO_EVENT_REASONING, b["thinking"].as_string());
                    if (b["type"].as_string() == "text") emit(ALBIREO_EVENT_CONTENT, b["text"].as_string());
                }
                if (a["stop_reason"].is_string()) finish = finish_of_claude(a["stop_reason"].as_string());
                cached_tokens = a["usage"]["cache_read_input_tokens"].as_int(0);
                prompt_tokens = a["usage"]["input_tokens"].as_int(0) + cached_tokens
                    + a["usage"]["cache_creation_input_tokens"].as_int(0);
                completion_tokens = a["usage"]["output_tokens"].as_int(0);
            }
            else if (!claude && a["choices"].is_array())
            {
                ++events;
                const json& choice = a["choices"][0];
                const json& msg = choice["message"];
                if (msg["reasoning_content"].is_string())
                    emit(ALBIREO_EVENT_REASONING, msg["reasoning_content"].as_string());
                else if (msg["reasoning"].is_string())
                    emit(ALBIREO_EVENT_REASONING, msg["reasoning"].as_string());
                if (msg["content"].is_string()) emit(ALBIREO_EVENT_CONTENT, msg["content"].as_string());
                const std::string f = choice["finish_reason"].as_string();
                finish = f == "length" || f == "content_filter" ? f : "stop";
                prompt_tokens = a["usage"]["prompt_tokens"].as_int(0);
                completion_tokens = a["usage"]["completion_tokens"].as_int(0);
                cached_tokens = a["usage"]["prompt_tokens_details"]["cached_tokens"].as_int(0);
            }
            else if (a["error"].is_object() || a["error"].is_string())
            {
                provider_failure = "the external engine answered with an error: " + redacted(a["error"].is_string()
                    ? a["error"].as_string() : a["error"]["message"].as_string(), r).substr(0, 300);
            }
        }
        if (j->control.cancelled() || http_cancelled())
        {
            fail(http_cancelled() ? std::string("the request to the external engine was cancelled")
                : std::string("the answer was cancelled"), "cancelled");
            return;
        }
        if (status != 0 && status != 200)
        {
            fail(provider_error(status, error_body, r), "relay_refused");
            return;
        }
        if (too_large)
        {
            fail("the external engine sent an answer or an event larger than 16 MB", "relay_failed");
            return;
        }
        if (!provider_failure.empty())
        {
            fail(provider_failure, "relay_failed");
            return;
        }
        if (!sent && !ended)
        {
            fail(transfer_error(transfer), "relay_failed");
            return;
        }
        // An answer of 200 whose body the relay could not read as a stream or as a completion: a web page.
        if (events == 0)
        {
            fail("the external engine gave no answer that the relay can read", "relay_failed");
            return;
        }
        // The substitutes that the end of the answer completes, then the end itself.
        const std::string reasoning_rest = reasoning_restorer.finish();
        if (!reasoning_rest.empty()) j->sink(ALBIREO_EVENT_REASONING, reasoning_rest);
        const std::string content_rest = content_restorer.finish();
        if (!content_rest.empty()) j->sink(ALBIREO_EVENT_CONTENT, content_rest);
        const auto now = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(first_token - started).count();
        const double generation_ms = std::chrono::duration<double, std::milli>(now - first_token).count();
        json usage = json::object();
        usage["prompt_tokens"] = prompt_tokens;
        usage["completion_tokens"] = completion_tokens;
        usage["total_tokens"] = prompt_tokens + completion_tokens;
        usage["cached_tokens"] = cached_tokens;
        json timings = json::object();
        timings["queued_ms"] = 0.0;
        timings["prompt_ms"] = prompt_ms;
        timings["generation_ms"] = generation_ms;
        timings["prompt_per_second"] = prompt_ms > 0 ? 1000.0 * static_cast<double>(prompt_tokens) / prompt_ms : 0.0;
        timings["generation_per_second"] = generation_ms > 0 && completion_tokens > 1
            ? 1000.0 * static_cast<double>(completion_tokens - 1) / generation_ms : 0.0;
        json done = json::object();
        done["finish_reason"] = finish;
        done["usage"] = usage;
        done["timings"] = timings;
        done["relay"] = true;
        // One line per answer, its provider, its model and its tokens, never its texts.
        log_info("relay: " + r.provider + " " + m.id + ", " + std::to_string(prompt_tokens) + " tokens read, "
            + std::to_string(completion_tokens) + " written");
        j->sink(ALBIREO_EVENT_DONE, done.dump());
        j->done = true;
    }

    json relay_service::test(const relay_settings& r)
    {
        json o = json::object();
        o["ok"] = false;
        const relay_settings& t = r;
        if (t.provider == "openai" && !valid_relay_url(t.url))
        {
            o["error"] = "the address is not valid: http:// or https:// followed by the base of the interface";
            return o;
        }
        if (auth_of(t) != "none" && t.secret.empty())
        {
            o["error"] = auth_of(t) == "basic" ? "the password is missing" : "the key is missing";
            return o;
        }
        if (settings_().offline)
        {
            o["error"] = "offline mode: the relay sends nothing out of this computer";
            return o;
        }
        url_parts u;
        parse_url(relay_base_url(t) + "/models", u);
        std::string error;
        std::shared_ptr<httplib::Client> client = http_client(u.origin, 30, error);
        if (!client)
        {
            o["error"] = error.empty() ? std::string("the external engine cannot be reached") : error;
            return o;
        }
        http_registration registered(client);
        httplib::Headers h = headers_of(t);
        httplib::Result res;
        try
        {
            res = client->Get(u.target, h);
        }
        catch (...)
        {
            o["error"] = "the request to the external engine was cancelled";
            return o;
        }
        if (!res)
        {
            o["error"] = transfer_error(res.error());
            return o;
        }
        if (res->status != 200)
        {
            o["error"] = provider_error(res->status, res->body, t);
            return o;
        }
        const json answer = json::parse_or_null(res->body.substr(0, size_t(8) << 20));
        json models = json::array();
        for (const json& m : answer["data"].elements())
        {
            if (models.size() >= 500) break;
            if (m["id"].is_string()) models.push_back(m["id"]);
        }
        o["ok"] = true;
        o["models"] = models;
        return o;
    }

}
