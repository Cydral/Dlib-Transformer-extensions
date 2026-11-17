// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "relay.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <utility>

#include "../common/engine_api.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
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

        /*
            The searches of the web that an answer of Claude makes at most, when its request asks for
            them (web_search): each one is billed by Anthropic, and a turn this short never pauses.
        */
        const long long claude_web_searches = 5;

        std::mutex origins_m;
        std::map<std::string, std::string> test_origins;
        std::map<std::string, std::vector<std::string>> test_names;

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

        // The message that an external engine gives with an error, without the secret, the user and the host.
        std::string provider_message(const std::string& body, const relay_settings& r)
        {
            const json j = json::parse_or_null(body.substr(0, 64 * 1024));
            std::string message;
            if (j["error"].is_string()) message = j["error"].as_string();
            else if (j["error"]["message"].is_string()) message = j["error"]["message"].as_string();
            else if (j["message"].is_string()) message = j["message"].as_string();
            else if (j["detail"].is_string()) message = j["detail"].as_string();
            message = redacted(message, r);
            if (message.size() > 300) message = message.substr(0, 300) + "...";
            return message;
        }

        // The code of an HTTP status of an external engine, by which the tray words it in its language.
        const char* status_code(int status)
        {
            if (status == 401 || status == 403) return "relay_http_key";
            if (status == 402) return "relay_http_credit";
            if (status == 404) return "relay_http_address";
            if (status == 413) return "relay_http_size";
            if (status == 429) return "relay_http_limit";
            if (status >= 500) return "relay_http_failing";
            return "relay_http_refused";
        }

        std::string status_text(int status)
        {
            std::string what;
            if (status == 401 || status == 403) what = "the external engine refuses the key or the user";
            else if (status == 402) what = "the account of the external engine has no credit left";
            else if (status == 404) what = "the external engine knows neither this model nor this address";
            else if (status == 413) what = "the external engine refuses a request this large";
            else if (status == 429) what = "the external engine limits the requests or the credits of the account";
            else if (status >= 500) what = "the external engine is overloaded or failing";
            else what = "the external engine refuses the request";
            return what + " (HTTP " + std::to_string(status) + ")";
        }

        // The message of an error that an external engine answered, without the request it may quote.
        std::string provider_error(int status, const std::string& body, const relay_settings& r)
        {
            const std::string message = provider_message(body, r);
            return status_text(status) + (message.empty() ? "" : ": " + message);
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

        // The code of a failed transfer, by which the tray words it in its language.
        const char* transfer_code(const httplib::Error e)
        {
            switch (e)
            {
            case httplib::Error::SSLServerVerification:
            case httplib::Error::SSLServerHostnameVerification: return "relay_certificate";
            case httplib::Error::Read:
            case httplib::Error::Timeout: return "relay_no_answer";
            default: return "relay_unreachable";
            }
        }

        /*
            Whether the origin "scheme://host:port" holds a host and a port alone: a name of
            letters, digits, dots, hyphens and underscores, or an IPv6 address in brackets, then
            ":" and a port from 1 to 65535.  A user, a query or a fragment there would make the
            client reach another host than the one the offline mode checks.
        */
        bool plain_authority(const std::string& origin)
        {
            const size_t scheme = origin.find("://");
            if (scheme == std::string::npos) return false;
            const std::string a = origin.substr(scheme + 3);
            size_t i = 0;
            if (!a.empty() && a[0] == '[')
            {
                const size_t close = a.find(']');
                if (close == std::string::npos || close < 3) return false;
                for (i = 1; i < close; ++i)
                    if (!std::isxdigit(static_cast<unsigned char>(a[i])) && a[i] != ':' && a[i] != '.') return false;
                i = close + 1;
            }
            else
            {
                while (i < a.size() && (std::isalnum(static_cast<unsigned char>(a[i])) || a[i] == '.' || a[i] == '-'
                    || a[i] == '_'))
                    ++i;
                if (i == 0) return false;
            }
            if (i == a.size()) return true;
            const std::string port = a.substr(i + 1);
            if (a[i] != ':' || port.empty() || port.size() > 5
                || port.find_first_not_of("0123456789") != std::string::npos)
                return false;
            const long value = std::atol(port.c_str());
            return value >= 1 && value <= 65535;
        }

        // The host of an origin "scheme://host:port" as it is written, an IPv6 address without its brackets.
        std::string written_host(const std::string& origin)
        {
            const size_t scheme = origin.find("://");
            std::string host = origin.substr(scheme == std::string::npos ? 0 : scheme + 3);
            if (!host.empty() && host[0] == '[')
            {
                const size_t end = host.find(']');
                return host.substr(1, end == std::string::npos ? std::string::npos : end - 1);
            }
            return host.substr(0, host.find_first_of(":/"));
        }

        // The host of an origin in lowercase.
        std::string origin_host(const std::string& origin)
        {
            std::string host = written_host(origin);
            for (char& c : host) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return host;
        }

        /*
            The addresses of the name host, which the resolver of the system gives within the bound
            of http_resolve_local(), whatever the cancellation of the downloads; empty with error
            when the name has none.
        */
        std::vector<std::string> resolve_name(const std::string& host, std::string& error)
        {
            {
                std::lock_guard<std::mutex> lock(origins_m);
                auto it = test_names.find(host);
                if (it != test_names.end()) return it->second;
            }
            std::vector<std::string> addresses;
            if (!http_resolve_local(host, addresses, error)) addresses.clear();
            return addresses;
        }

        // Whether the IPv4 address of 4 bytes lies on this computer or its local network.
        bool local_v4(const unsigned char* a)
        {
            const bool unspecified = a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0;
            return unspecified || a[0] == 127 || a[0] == 10 || (a[0] == 172 && (a[1] & 0xf0) == 16)
                || (a[0] == 192 && a[1] == 168) || (a[0] == 169 && a[1] == 254);
        }

        /*
            Whether the address numeric, its zone left aside, lies on this computer or its local
            network: unspecified or loopback (0.0.0.0, 127.0.0.0/8, ::, ::1), private
            (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7) or link-local (169.254.0.0/16,
            fe80::/10), an IPv4 address mapped in IPv6 as that IPv4 address.  Every other
            address, the shared space of the operators and the tunnels or translations of IPv6
            included, leads off the local network.
        */
        bool local_network_address(const std::string& numeric)
        {
            const std::string text = numeric.substr(0, numeric.find('%'));
            unsigned char b[16] = {};
            if (inet_pton(AF_INET, text.c_str(), b) == 1) return local_v4(b);
            if (inet_pton(AF_INET6, text.c_str(), b) != 1) return false;
            static const unsigned char mapped[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
            if (std::memcmp(b, mapped, sizeof(mapped)) == 0) return local_v4(b + 12);
            bool zeros = true;
            for (int i = 0; i < 15; ++i) zeros = zeros && b[i] == 0;
            return (zeros && b[15] <= 1) || (b[0] & 0xfe) == 0xfc || (b[0] == 0xfe && (b[1] & 0xc0) == 0x80);
        }

        // Whether host is an IPv4 or IPv6 address rather than a name.
        bool address_literal(const std::string& host)
        {
            unsigned char bytes[16];
            return inet_pton(AF_INET, host.c_str(), bytes) == 1 || inet_pton(AF_INET6, host.c_str(), bytes) == 1;
        }

        // "localhost" and the names under .localhost, which name this computer whatever the resolver gives.
        bool loopback_name(const std::string& host)
        {
            const std::string tail = ".localhost";
            return host == "localhost"
                || (host.size() > tail.size() && host.compare(host.size() - tail.size(), tail.size(), tail) == 0);
        }

        const char* const offline_remote = "offline mode: the address of the engine is neither on this computer nor "
            "on its local network";

        /*
            Whether host lies on this computer or its local network, which the offline mode keeps
            within reach: a loopback name, an address of the local network
            (local_network_address()), or a name whose every address is so.  why and code say
            otherwise.  address receives the address checked that the request connects to, so
            that no later resolution of the name leads elsewhere: 127.0.0.1 for a loopback name,
            which the resolver of Windows may answer otherwise, and for another name an IPv4
            address first, where the engines listen most often, else an IPv6 address with its
            zone.  It stays empty for an address, which the client connects to as it is.
        */
        bool local_host(const std::string& host, std::string& why, std::string& code, std::string& address)
        {
            address.clear();
            if (loopback_name(host))
            {
                address = "127.0.0.1";
                return true;
            }
            const bool literal = address_literal(host);
            std::string error;
            const std::vector<std::string> addresses = literal ? std::vector<std::string>{ host }
                : resolve_name(host, error);
            if (addresses.empty())
            {
                why = "offline mode: the name of the engine is not resolved on this network"
                    + (error.empty() ? std::string() : ": " + error);
                code = "relay_offline_unresolved";
                return false;
            }
            for (const std::string& a : addresses)
                if (!local_network_address(a))
                {
                    why = offline_remote;
                    code = "relay_offline_remote";
                    return false;
                }
            if (literal) return true;
            address = addresses.front();
            for (const std::string& a : addresses)
                if (a.find(':') == std::string::npos)
                {
                    address = a;
                    break;
                }
            return true;
        }

        /*
            The client of the relay for origin.  Online, a client of the downloads (http_client()),
            with their proxy and their cancellation.  In the offline mode, which cancels the
            downloads, an engine of the interface of OpenAI on this computer or its local network
            alone, reached directly: its requests end with the cancellation of their answer, and
            with the stop of the service through their registration (http_registration).  code
            names a refusal of the offline mode for the tray.
        */
        std::shared_ptr<httplib::Client> relay_client(const relay_settings& r, bool offline, const std::string& origin,
            int timeout_seconds, std::string& error, std::string& code)
        {
            if (!offline)
            {
                std::shared_ptr<httplib::Client> c = http_client(origin, timeout_seconds, error);
                if (!c && !http_cancelled())
                {
                    // The server or the proxy out of reach, which the tray words in its language.
                    std::string host, user, password;
                    int port = 0;
                    code = http_proxy_for(origin, host, port, user, password) ? "relay_proxy" : "relay_unreachable";
                    wipe_secret(password);
                }
                return c;
            }
            if (r.provider != "openai")
            {
                error = std::string("offline mode: ") + provider_name(r.provider) + " needs the Internet";
                code = "relay_offline";
                return nullptr;
            }
            std::string address;
            if (!local_host(origin_host(origin), error, code, address)) return nullptr;
            auto c = std::make_shared<httplib::Client>(origin);
            if (!address.empty()) c->set_hostname_addr_map({ { written_host(origin), address } });
            c->set_follow_location(false);
            c->set_connection_timeout(connect_seconds, 0);
            c->set_read_timeout(timeout_seconds, 0);
            c->set_write_timeout(timeout_seconds, 0);
            c->set_keep_alive(false);
            c->set_default_headers({ { "User-Agent", std::string(ALBIREO_PRODUCT_NAME) + "/" + ALBIREO_VERSION } });
            return c;
        }

        // The answer of GET <base>/models: its status, or the failure of its transfer, and the models it lists.
        struct model_listing
        {
            int status = 0;
            httplib::Error transfer = httplib::Error::Success;
            std::string error;              // the client could not be made: offline, a proxy, the service stopping
            std::string code;               // the refusal of the offline mode, for the tray
            bool listed = false;            // a list of models, {"data": [{"id"}, ...]}
            json models = json::array();
            std::string body;               // the start of an answer of another status, for its message
        };

        /*
            Asks base for its models, the connection held by control when given, which a
            cancellation stops; offline, the client of the offline mode (relay_client()).
        */
        model_listing list_models(const relay_settings& r, bool offline, const std::string& base, int timeout_seconds,
            cancellable* control)
        {
            model_listing out;
            url_parts u;
            if (!parse_url(base + "/models", u))
            {
                out.error = "the relay has no valid address";
                return out;
            }
            std::shared_ptr<httplib::Client> client = relay_client(r, offline, u.origin, timeout_seconds, out.error,
                out.code);
            if (!client) return out;
            // The cancellation of the downloads, which the offline mode holds, ends the requests online only.
            if (control)
            {
                client->set_socket_options([control, offline](socket_t sock) {
                    if (!(!offline && http_cancelled()) && !control->cancelled()) return;
                    httplib::detail::close_socket(sock);
                    throw answer_cancelled();
                });
                control->hold(client);
            }
            http_registration registered(client);
            httplib::Result res;
            try
            {
                if (!control || !control->cancelled()) res = client->Get(u.target, headers_of(r));
            }
            catch (...)
            {
            }
            if (control) control->release();
            if (!res)
            {
                // A request not sent, or stopped by its socket, is a cancellation.
                out.transfer = (control && control->cancelled()) || (!offline && http_cancelled())
                    ? httplib::Error::Canceled : res.error();
                return out;
            }
            out.status = res->status;
            if (res->status != 200)
            {
                out.body = res->body.substr(0, 64 * 1024);
                return out;
            }
            const json answer = json::parse_or_null(res->body.substr(0, size_t(8) << 20));
            out.listed = answer["data"].is_array();
            for (const json& m : answer["data"].elements())
            {
                if (out.models.size() >= 500) break;
                if (m["id"].is_string()) out.models.push_back(m["id"]);
            }
            return out;
        }

        /*
            The bases where the interface of an "openai" engine may answer for its address: the
            address typed, without the path of an operation pasted with it, then the same with /v1,
            where most engines serve it (Ollama, vLLM, LM Studio).
        */
        std::vector<std::string> openai_bases(const relay_settings& r)
        {
            std::string base = relay_base_url(r);
            for (const char* tail : { "/chat/completions", "/completions", "/models" })
            {
                const size_t n = std::strlen(tail);
                if (base.size() > n && base.compare(base.size() - n, n, tail) == 0)
                {
                    base.erase(base.size() - n);
                    break;
                }
            }
            while (!base.empty() && base.back() == '/') base.pop_back();
            std::vector<std::string> bases{ base };
            if (base.size() < 3 || base.compare(base.size() - 3, 3, "/v1") != 0) bases.push_back(base + "/v1");
            return bases;
        }

        /*
            The answer of the base where the interface of the provider of r lies, which base gives:
            the first base whose answer is no "nothing here" (404, 405, or a page that lists no
            models), the address typed when every base answers so.
        */
        model_listing reach_interface(const relay_settings& r, bool offline, int timeout_seconds, cancellable* control,
            std::string& base)
        {
            const std::vector<std::string> bases = r.provider == "openai" ? openai_bases(r)
                : std::vector<std::string>{ relay_base_url(r) };
            model_listing first;
            base = bases.front();
            for (size_t i = 0; i < bases.size(); ++i)
            {
                model_listing l = list_models(r, offline, bases[i], timeout_seconds, control);
                const bool nothing = l.status == 404 || l.status == 405 || (l.status == 200 && !l.listed);
                if (i == 0) first = l;
                // A client refused settles every base, which lie on the same server.
                if (!l.error.empty()) return l;
                // A base that cannot be reached settles nothing for the next one, which lies on the same server.
                if (i == 0 ? !nothing : l.status != 0 && !nothing)
                {
                    base = bases[i];
                    return l;
                }
                if (control && control->cancelled()) break;
            }
            return first;
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

        // The body of a request of the interface of Anthropic, with its search of the web when web is true.
        json claude_body(const relay_model& m, const std::vector<message>& msgs, const json& request,
            const std::string& effort, bool web)
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
            if (web)
            {
                json tool = json::object();
                tool["type"] = "web_search_20250305";
                tool["name"] = "web_search";
                tool["max_uses"] = claude_web_searches;
                json tools = json::array();
                tools.push_back(tool);
                body["tools"] = tools;
            }
            return body;
        }

        /*
            The sources of a text of Claude, the links that follow it: the domain of each address of
            the web cited, once each, in markdown, which the web application shows as links.  The
            addresses that are not http:// or https:// are left out.
        */
        std::string source_links(const std::vector<std::string>& urls)
        {
            std::string out;
            std::vector<std::string> seen;
            for (const std::string& url : urls)
            {
                const bool web = url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0;
                if (!web || url.size() > 2048 || std::find(seen.begin(), seen.end(), url) != seen.end()) continue;
                seen.push_back(url);
                std::string host = url.substr(url.find("://") + 3);
                host = host.substr(0, host.find_first_of("/?#"));
                if (host.compare(0, 4, "www.") == 0) host = host.substr(4);
                std::string target;
                for (const char c : url)
                {
                    if (c == '(') target += "%28";
                    else if (c == ')') target += "%29";
                    else if (c == ' ') target += "%20";
                    else if (static_cast<unsigned char>(c) < 0x20) continue;
                    else target += c;
                }
                std::string text;
                for (const char c : host)
                    if (c != '[' && c != ']' && c != '*' && c != '_' && c != '`') text += c;
                out += (out.empty() ? "" : ", ") + std::string("[") + text + "](" + target + ")";
            }
            return out.empty() ? out : " (" + out + ")";
        }

        // The end of an answer of Claude; a turn paused among its searches (pause_turn) ends with what it wrote.
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
        return plain_authority(u.origin) && u.target.find('?') == std::string::npos
            && u.target.find('#') == std::string::npos;
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

    void relay_name_for_tests(const std::string& name, const std::vector<std::string>& addresses)
    {
        std::lock_guard<std::mutex> lock(origins_m);
        if (addresses.empty()) test_names.erase(name);
        else test_names[name] = addresses;
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

    bool relay_offline_ready(const relay_settings& r, std::string& why)
    {
        if (r.provider != "openai")
        {
            why = std::string("offline mode: ") + provider_name(r.provider) + " needs the Internet";
            return false;
        }
        url_parts u;
        if (!parse_url(relay_base_url(r), u)) return true;
        // A name of the network is resolved by the request, which checks each of its addresses.
        const std::string host = origin_host(u.origin);
        if (loopback_name(host) || !address_literal(host) || local_network_address(host)) return true;
        why = offline_remote;
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
        j["keep_substitutes"] = r.pseudonymize && r.keep_substitutes;
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
        if (st.offline && !relay_offline_ready(st.relay, why)) return refuse(503, "offline", why);
        if (request["tools"].size() > 0)
            return refuse(400, "tools_not_supported", "the relay takes no tools: the agent mode and the web access "
                "stay with the models of the computer");
        // The search of the web that Claude makes itself, which the web access of the settings allows.
        if (request["web_search"].as_bool(false))
        {
            if (st.relay.provider != "claude")
                return refuse(400, "web_search_not_supported", "the search of the web of the relay takes Claude, which "
                    "searches the web itself");
            if (!st.web_access) return refuse(403, "web_access_off", "the administrator turned the web access off");
            // Pseudalis replaces the names before they leave: the searches would look for its substitutes.
            if (st.relay.pseudonymize)
                return refuse(400, "web_search_pseudonymized", "the relay pseudonymizes the texts with Pseudalis: a "
                    "search of the web would look for its substitutes rather than for the names of the question");
        }
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
        // The check of what left keeps the substitutes in the answer: no value comes back.
        if (r.keep_substitutes)
        {
            for (auto& v : values) wipe_secret(v.second);
            values.clear();
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
        // The settings of the answer itself: the web access and the pseudonymization may have changed since submit().
        const bool web = claude && request["web_search"].as_bool(false) && st.web_access && !r.pseudonymize;
        const json body = claude ? claude_body(m, msgs, request, effort, web)
                                 : openai_body(r, m, msgs, request, effort, reasoning_off);
        msgs.clear();

        // The base of an engine of the interface of OpenAI, as typed or with /v1, which its models tell once.
        const std::string base = base_of(r, &j->control);
        url_parts u;
        if (!parse_url(base + (claude ? "/messages" : "/chat/completions"), u))
        {
            fail("the relay has no valid address", "relay_unavailable");
            return;
        }
        std::string error, refusal;
        const bool offline = st.offline;
        std::shared_ptr<httplib::Client> client = relay_client(r, offline, u.origin, 300, error, refusal);
        if (!client)
        {
            // A refusal of the offline mode answers with the code offline, an engine out of reach with its own.
            const bool refused = refusal.compare(0, 13, "relay_offline") == 0;
            fail(error.empty() ? std::string("the external engine cannot be reached") : error,
                refused ? "offline" : "relay_unreachable");
            return;
        }
        /*
            The connection opens for an answer that goes on alone, whatever the moment of its
            cancellation; the cancellation of the downloads, which the offline mode holds, ends
            the answers online only.
        */
        const cancellable* control = &j->control;
        client->set_socket_options([control, offline](socket_t sock) {
            if (!(!offline && http_cancelled()) && !control->cancelled()) return;
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
        /*
            The blocks of an answer of Claude that its search of the web adds: the queries of its
            searches (server_tool_use), shown with the reasoning, and the sources a text cites,
            whose links follow the text, by the index of their block.
        */
        struct claude_block
        {
            std::string type;
            std::string input;
            std::vector<std::string> sources;
        };
        std::map<long long, claude_block> blocks;
        long long web_searches = 0;
        auto add_source = [](claude_block& b, const json& citation) {
            if (citation["url"].is_string() && b.sources.size() < 16) b.sources.push_back(citation["url"].as_string());
        };
        auto web_block = [&](const json& b) {
            const std::string kind = b["type"].as_string();
            if (kind == "server_tool_use" && b["name"].as_string() == "web_search" && b["input"]["query"].is_string())
                emit(ALBIREO_EVENT_REASONING, "\nweb_search: " + b["input"]["query"].as_string() + "\n");
            if (kind == "web_search_tool_result" && b["content"]["error_code"].is_string())
                emit(ALBIREO_EVENT_REASONING, "\nweb_search: " + b["content"]["error_code"].as_string() + "\n");
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
                else if (type == "content_block_start")
                {
                    const json& b = e["content_block"];
                    claude_block& block = blocks[e["index"].as_int(0)];
                    block.type = b["type"].as_string();
                    if (block.type == "text")
                        for (const json& c : b["citations"].elements()) add_source(block, c);
                    if (block.type == "web_search_tool_result") web_block(b);
                }
                else if (type == "content_block_delta")
                {
                    const json& d = e["delta"];
                    const std::string kind = d["type"].as_string();
                    if (kind == "text_delta") emit(ALBIREO_EVENT_CONTENT, d["text"].as_string());
                    else if (kind == "thinking_delta") emit(ALBIREO_EVENT_REASONING, d["thinking"].as_string());
                    else if (kind == "input_json_delta" || kind == "citations_delta")
                    {
                        claude_block& block = blocks[e["index"].as_int(0)];
                        if (kind == "citations_delta") add_source(block, d["citation"]);
                        else if (block.input.size() < 64 * 1024) block.input += d["partial_json"].as_string();
                    }
                }
                else if (type == "content_block_stop")
                {
                    auto it = blocks.find(e["index"].as_int(0));
                    if (it == blocks.end()) return;
                    if (it->second.type == "server_tool_use")
                    {
                        json b = json::object();
                        b["type"] = "server_tool_use";
                        b["name"] = "web_search";
                        b["input"] = json::parse_or_null(it->second.input);
                        web_block(b);
                    }
                    if (it->second.type == "text") emit(ALBIREO_EVENT_CONTENT, source_links(it->second.sources));
                    blocks.erase(it);
                }
                else if (type == "message_delta")
                {
                    if (e["delta"]["stop_reason"].is_string())
                        finish = finish_of_claude(e["delta"]["stop_reason"].as_string());
                    completion_tokens = e["usage"]["output_tokens"].as_int(completion_tokens);
                    web_searches = e["usage"]["server_tool_use"]["web_search_requests"].as_int(web_searches);
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
                    if (b["type"].as_string() == "text")
                    {
                        claude_block block;
                        for (const json& c : b["citations"].elements()) add_source(block, c);
                        emit(ALBIREO_EVENT_CONTENT, b["text"].as_string() + source_links(block.sources));
                    }
                    web_block(b);
                }
                web_searches = a["usage"]["server_tool_use"]["web_search_requests"].as_int(0);
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
        const bool downloads_cancelled = !offline && http_cancelled();
        if (j->control.cancelled() || downloads_cancelled)
        {
            fail(downloads_cancelled ? std::string("the request to the external engine was cancelled")
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
        // The searches of the web of Claude, which Anthropic bills apart.
        if (web_searches > 0) usage["web_search_requests"] = web_searches;
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
            + std::to_string(completion_tokens) + " written"
            + (web_searches > 0 ? ", " + std::to_string(web_searches) + (web_searches == 1 ? " search" : " searches")
                + " of the web" : std::string()));
        j->sink(ALBIREO_EVENT_DONE, done.dump());
        j->done = true;
    }

    std::string relay_service::base_of(const relay_settings& r, cancellable* control)
    {
        if (r.provider != "openai") return relay_base_url(r);
        const std::string typed = openai_bases(r).front();
        {
            std::lock_guard<std::mutex> lock(bases_m_);
            auto it = bases_.find(typed);
            if (it != bases_.end() && (it->second.until == 0 || now_unix() < it->second.until)) return it->second.base;
        }
        std::string base;
        const model_listing l = reach_interface(r, settings_().offline, 15, control, base);
        if (l.status == 200 && l.listed) remember_base(typed, base, 0);
        else if (l.status != 0) remember_base(typed, base, now_unix() + unlisted_seconds);
        return base;
    }

    void relay_service::remember_base(const std::string& typed, const std::string& base, int64_t until)
    {
        std::lock_guard<std::mutex> lock(bases_m_);
        if (bases_.size() >= max_bases && bases_.find(typed) == bases_.end()) bases_.clear();
        known_base& k = bases_[typed];
        k.base = base;
        k.until = until;
    }

    json relay_service::test(const relay_settings& r)
    {
        json o = json::object();
        o["ok"] = false;
        auto refuse = [&o](const std::string& error, const std::string& code) {
            o["error"] = error;
            if (!code.empty()) o["code"] = code;
            return o;
        };
        const std::string auth = auth_of(r);
        if (r.provider == "openai" && !valid_relay_url(r.url))
            return refuse("the address is not valid: http:// or https:// followed by the base of the interface",
                "relay_address");
        if (auth == "basic" && r.secret.empty()) return refuse("the password is missing", "relay_password_missing");
        if (auth != "none" && r.secret.empty()) return refuse("the key is missing", "relay_key_missing");
        const bool offline = settings_().offline;
        std::string why;
        if (offline && !relay_offline_ready(r, why))
            return refuse(why, r.provider == "openai" ? "relay_offline_remote" : "relay_offline");
        std::string base;
        const model_listing l = reach_interface(r, offline, 30, nullptr, base);
        if (!l.error.empty()) return refuse(l.error, l.code);
        if (l.status == 0)
        {
            if (l.transfer == httplib::Error::Canceled)
                return refuse("the request to the external engine was cancelled", "");
            return refuse(transfer_error(l.transfer), transfer_code(l.transfer));
        }
        if (l.status != 200)
        {
            refuse(provider_error(l.status, l.body, r), status_code(l.status));
            json args = json::object();
            args["s"] = std::to_string(l.status);
            o["args"] = args;
            // The message of the engine, in its language, which the tray adds to its own.
            const std::string message = provider_message(l.body, r);
            if (!message.empty()) o["detail"] = message;
            return o;
        }
        if (!l.listed)
            return refuse("the address answers, but not as an interface of OpenAI", "relay_not_interface");
        if (r.provider == "openai")
        {
            remember_base(openai_bases(r).front(), base, 0);
            // The base found when it differs from the address typed: with /v1, or without an operation.
            if (base != relay_base_url(r)) o["url"] = base;
        }
        o["ok"] = true;
        o["models"] = l.models;
        return o;
    }

}
