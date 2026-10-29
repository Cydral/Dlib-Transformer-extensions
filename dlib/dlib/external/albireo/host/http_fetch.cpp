// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include "http_fetch.h"

#include <cctype>
#include <cstdlib>
#include <mutex>

#include <dlib/external/httplib/httplib.h>

#if defined(_WIN32)
#include <winhttp.h>
#endif

#include "../common/platform.h"
#include "../common/version.h"

namespace albireo
{

    bool parse_url(const std::string& url, url_parts& out)
    {
        const size_t scheme_end = url.find("://");
        if (scheme_end == std::string::npos) return false;
        const std::string scheme = url.substr(0, scheme_end);
        if (scheme != "http" && scheme != "https") return false;
        const size_t host_start = scheme_end + 3;
        const size_t path_start = url.find('/', host_start);
        out.origin = url.substr(0, path_start);
        out.target = path_start == std::string::npos ? std::string("/") : url.substr(path_start);
        return out.origin.size() > host_start;
    }

    namespace
    {
        std::mutex proxy_m;
        std::string proxy_setting;

        struct proxy_address
        {
            std::string host;
            int port = 0;
            std::string user;
            std::string password;
        };

        std::string lower(std::string s)
        {
            for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        std::string percent_decode(const std::string& s)
        {
            std::string out;
            for (size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1]))
                    && std::isxdigit(static_cast<unsigned char>(s[i + 2])))
                {
                    out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
                    i += 2;
                }
                else
                {
                    out += s[i];
                }
            }
            return out;
        }

        // [http://][user[:password]@]host[:port][/]
        bool parse_proxy(std::string text, proxy_address& p)
        {
            p = proxy_address();
            const size_t scheme = text.find("://");
            if (scheme != std::string::npos)
            {
                if (lower(text.substr(0, scheme)) != "http") return false;
                text = text.substr(scheme + 3);
            }
            while (!text.empty() && text.back() == '/') text.pop_back();
            const size_t at = text.rfind('@');
            if (at != std::string::npos)
            {
                const std::string credentials = text.substr(0, at);
                text = text.substr(at + 1);
                const size_t colon = credentials.find(':');
                p.user = percent_decode(credentials.substr(0, colon));
                if (colon != std::string::npos) p.password = percent_decode(credentials.substr(colon + 1));
            }
            p.port = 80;
            const size_t colon = text.rfind(':');
            if (colon != std::string::npos && text.find(']', colon) == std::string::npos)
            {
                const std::string port = text.substr(colon + 1);
                if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos)
                    return false;
                p.port = std::atoi(port.c_str());
                text = text.substr(0, colon);
            }
            if (text.size() > 2 && text.front() == '[' && text.back() == ']') text = text.substr(1, text.size() - 2);
            p.host = text;
            if (p.host.empty() || p.port <= 0 || p.port > 65535) return false;
            for (char c : p.host)
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == ':' || c == '_'))
                    return false;
            return true;
        }

        std::string environment(const char* name)
        {
#if defined(_WIN32)
            const wchar_t* v = _wgetenv(widen(name).c_str());
            return v ? narrow(v) : std::string();
#else
            const char* v = std::getenv(name);
            return v ? std::string(v) : std::string();
#endif
        }

        std::string host_of(const std::string& origin)
        {
            std::string h = origin.substr(origin.find("://") + 3);
            if (!h.empty() && h[0] == '[') return lower(h.substr(1, h.find(']') - 1));
            return lower(h.substr(0, h.find(':')));
        }

        // Whether NO_PROXY, or the nature of the address, sends a host around the proxy.
        bool direct_host(const std::string& host)
        {
            if (host == "localhost" || host == "::1" || host.compare(0, 4, "127.") == 0) return true;
            std::string list = environment("NO_PROXY");
            if (list.empty()) list = environment("no_proxy");
            size_t start = 0;
            while (start <= list.size())
            {
                size_t end = list.find_first_of(", ", start);
                if (end == std::string::npos) end = list.size();
                std::string item = lower(list.substr(start, end - start));
                start = end + 1;
                if (item.empty()) continue;
                if (item == "*") return true;
                if (item.compare(0, 2, "*.") == 0) item = item.substr(2);
                else if (item[0] == '.') item = item.substr(1);
                if (host == item) return true;
                if (host.size() > item.size() && host.compare(host.size() - item.size(), item.size(), item) == 0
                    && host[host.size() - item.size() - 1] == '.')
                    return true;
            }
            return false;
        }

#if defined(_WIN32)
        // The proxy of WinHTTP for a scheme: "host:port", or "http=host:port;https=host:port".
        std::string winhttp_proxy(bool https)
        {
            WINHTTP_PROXY_INFO info = {};
            if (!WinHttpGetDefaultProxyConfiguration(&info)) return std::string();
            std::string list;
            if (info.dwAccessType == WINHTTP_ACCESS_TYPE_NAMED_PROXY && info.lpszProxy) list = narrow(info.lpszProxy);
            if (info.lpszProxy) GlobalFree(info.lpszProxy);
            if (info.lpszProxyBypass) GlobalFree(info.lpszProxyBypass);
            std::string fallback;
            size_t start = 0;
            while (start < list.size())
            {
                size_t end = list.find_first_of("; ", start);
                if (end == std::string::npos) end = list.size();
                const std::string item = list.substr(start, end - start);
                start = end + 1;
                if (item.empty()) continue;
                const size_t eq = item.find('=');
                if (eq == std::string::npos)
                {
                    if (fallback.empty()) fallback = item;
                    continue;
                }
                if (lower(item.substr(0, eq)) == (https ? "https" : "http")) return item.substr(eq + 1);
            }
            return fallback;
        }
#endif

        std::string proxy_for(const std::string& origin)
        {
            if (direct_host(host_of(origin))) return std::string();
            std::string setting;
            {
                std::lock_guard<std::mutex> lock(proxy_m);
                setting = proxy_setting;
            }
            if (setting == "none") return std::string();
            if (!setting.empty()) return setting;
            const bool https = origin.compare(0, 8, "https://") == 0;
            std::string p = environment(https ? "HTTPS_PROXY" : "HTTP_PROXY");
            if (p.empty()) p = environment(https ? "https_proxy" : "http_proxy");
            if (p.empty()) p = environment("ALL_PROXY");
#if defined(_WIN32)
            if (p.empty()) p = winhttp_proxy(https);
#endif
            return p;
        }

        std::unique_ptr<httplib::Client> make_client(const std::string& origin, int timeout_seconds)
        {
            std::unique_ptr<httplib::Client> c(new httplib::Client(origin));
            proxy_address p;
            const std::string proxy = proxy_for(origin);
            if (!proxy.empty() && parse_proxy(proxy, p))
            {
                c->set_proxy(p.host, p.port);
                if (!p.user.empty()) c->set_proxy_basic_auth(p.user, p.password);
            }
            c->set_follow_location(true);
            c->set_connection_timeout(timeout_seconds, 0);
            c->set_read_timeout(timeout_seconds, 0);
            c->set_write_timeout(timeout_seconds, 0);
            c->set_keep_alive(false);
            c->set_default_headers({ { "User-Agent", std::string(ALBIREO_PRODUCT_NAME) + "/" + ALBIREO_VERSION } });
            return c;
        }
    }

    void set_http_proxy(const std::string& setting)
    {
        std::lock_guard<std::mutex> lock(proxy_m);
        proxy_setting = setting;
    }

    bool valid_proxy_setting(const std::string& setting)
    {
        proxy_address p;
        return setting.empty() || setting == "none" || (setting.size() <= 512 && parse_proxy(setting, p));
    }

    bool http_get(const std::string& url, std::string& body, std::string& error, size_t max_bytes, int timeout_seconds)
    {
        url_parts u;
        if (!parse_url(url, u))
        {
            error = "invalid address " + url;
            return false;
        }
        auto c = make_client(u.origin, timeout_seconds);
        body.clear();
        bool too_big = false;
        auto r = c->Get(u.target, [&](const char* data, size_t n) {
            if (body.size() + n > max_bytes)
            {
                too_big = true;
                return false;
            }
            body.append(data, n);
            return true;
        });
        if (too_big)
        {
            error = url + " is larger than expected";
            return false;
        }
        if (!r)
        {
            error = url + ": " + httplib::to_string(r.error());
            return false;
        }
        if (r->status != 200)
        {
            error = url + ": status " + std::to_string(r->status);
            return false;
        }
        return true;
    }

    bool http_get_range(const std::string& url, uint64_t from, int64_t to,
        const std::function<bool(const char*, size_t)>& receiver, std::string& error, uint64_t* total)
    {
        url_parts u;
        if (!parse_url(url, u))
        {
            error = "invalid address " + url;
            return false;
        }
        auto c = make_client(u.origin, 60);
        httplib::Headers headers;
        if (from > 0 || to >= 0)
        {
            const std::string last = to >= 0 ? std::to_string(to) : std::string();
            headers.emplace("Range", "bytes=" + std::to_string(from) + "-" + last);
        }
        bool refused_range = false;
        bool stopped = false;
        int status = 0;
        auto r = c->Get(u.target, headers,
            [&](const httplib::Response& res) {
                status = res.status;
                if (total)
                {
                    *total = 0;
                    const std::string cr = res.get_header_value("Content-Range");
                    const size_t slash = cr.find('/');
                    if (slash != std::string::npos && cr.substr(slash + 1) != "*")
                        *total = std::stoull(cr.substr(slash + 1));
                    else if (res.status == 200 && res.has_header("Content-Length"))
                        *total = std::stoull(res.get_header_value("Content-Length"));
                }
                if (from > 0 && res.status == 200)
                {
                    refused_range = true;
                    return false;
                }
                return res.status == 200 || res.status == 206;
            },
            [&](const char* data, size_t n) {
                if (!receiver(data, n))
                {
                    stopped = true;
                    return false;
                }
                return true;
            });
        if (refused_range)
        {
            error = url + ": the server does not resume downloads";
            return false;
        }
        if (stopped)
        {
            error = "stopped";
            return false;
        }
        if (!r)
        {
            error = status ? url + ": status " + std::to_string(status) : url + ": " + httplib::to_string(r.error());
            return false;
        }
        return true;
    }

    uint64_t http_content_length(const std::string& url, std::string& error)
    {
        url_parts u;
        if (!parse_url(url, u))
        {
            error = "invalid address " + url;
            return 0;
        }
        auto c = make_client(u.origin, 30);
        auto r = c->Head(u.target);
        if (!r || r->status != 200)
        {
            error = url + ": " + (r ? "status " + std::to_string(r->status) : httplib::to_string(r.error()));
            return 0;
        }
        return r->has_header("Content-Length") ? std::stoull(r->get_header_value("Content-Length")) : 0;
    }

}
