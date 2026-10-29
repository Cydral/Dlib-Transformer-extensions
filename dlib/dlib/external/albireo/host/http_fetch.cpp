// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include "http_fetch.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

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

        // The cancellation of the downloads, its reason, and the clients of the requests
        // in progress.
        std::atomic<bool> cancelled{false};
        std::mutex active_m;
        std::string cancel_reason = "stopped";
        std::set<std::shared_ptr<httplib::Client>> active;

        std::string cancelled_error()
        {
            std::lock_guard<std::mutex> lock(active_m);
            return cancel_reason;
        }

        // Thrown from the socket of a client about to connect after a cancellation.
        struct connection_cancelled
        {
        };

#if defined(_WIN32)
        // Signalled while the downloads are cancelled, for the resolutions that wait.
        HANDLE cancel_event()
        {
            static const HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            return e;
        }
#endif

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

        std::shared_ptr<httplib::Client> make_client(const std::string& origin, const std::string& proxy,
            int timeout_seconds)
        {
            std::shared_ptr<httplib::Client> c = std::make_shared<httplib::Client>(origin);
            proxy_address p;
            if (!proxy.empty() && parse_proxy(proxy, p))
            {
                c->set_proxy(p.host, p.port);
                if (!p.user.empty()) c->set_proxy_basic_auth(p.user, p.password);
            }
            // The redirections are followed here, one request at a time, so that each one
            // is bounded and can be cancelled.
            c->set_follow_location(false);
            /*
                The last check before a connection opens.  A cancellation that comes after
                the client is registered and before it connects finds no request in flight
                to stop, so the socket checks it itself: no connection leaves the machine
                once the downloads are cancelled.  The exception leaves httplib before it
                connects, its locks released by their guards.
            */
            c->set_socket_options([](socket_t sock) {
                if (!cancelled) return;
                httplib::detail::close_socket(sock);
                throw connection_cancelled();
            });
            c->set_connection_timeout(connect_seconds, 0);
            c->set_read_timeout(timeout_seconds, 0);
            c->set_write_timeout(timeout_seconds, 0);
            c->set_keep_alive(false);
            c->set_default_headers({ { "User-Agent", std::string(ALBIREO_PRODUCT_NAME) + "/" + ALBIREO_VERSION } });
            return c;
        }

#if defined(_WIN32)
        typedef INT (WSAAPI* addrinfo_cancel_function)(LPHANDLE);
        typedef INT (WSAAPI* addrinfo_result_function)(LPOVERLAPPED);

        // A resolution in progress.
        struct lookup
        {
            OVERLAPPED overlapped = {};
            PADDRINFOEXW result = nullptr;
            HANDLE handle = nullptr;
        };

        /*
            Opens and closes a connection to one of the addresses of a resolution, waiting
            in slices of a tenth of a second so that a cancellation ends the wait at once;
            true when an address answers.  select() reports the refused connections, which
            WSAPoll misses on some versions of Windows.
        */
        bool probe(const ADDRINFOEXW* list, int port, std::string& error)
        {
            // One bound for every address of the name, so that a network that drops the
            // packets costs connect_seconds whatever the number of its addresses.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(connect_seconds);
            for (const ADDRINFOEXW* a = list; a && std::chrono::steady_clock::now() < deadline; a = a->ai_next)
            {
                if ((a->ai_family != AF_INET && a->ai_family != AF_INET6) || !a->ai_addr) continue;
                sockaddr_storage address = {};
                std::memcpy(&address, a->ai_addr, std::min(sizeof(address), static_cast<size_t>(a->ai_addrlen)));
                const u_short net_port = htons(static_cast<u_short>(port));
                if (a->ai_family == AF_INET) reinterpret_cast<sockaddr_in*>(&address)->sin_port = net_port;
                else reinterpret_cast<sockaddr_in6*>(&address)->sin6_port = net_port;
                const SOCKET sock = socket(a->ai_family, SOCK_STREAM, IPPROTO_TCP);
                if (sock == INVALID_SOCKET) continue;
                u_long nonblocking = 1;
                ioctlsocket(sock, FIONBIO, &nonblocking);
                bool open = connect(sock, reinterpret_cast<const sockaddr*>(&address),
                    static_cast<int>(a->ai_addrlen)) == 0;
                if (!open && WSAGetLastError() == WSAEWOULDBLOCK)
                {
                    while (!cancelled && std::chrono::steady_clock::now() < deadline)
                    {
                        fd_set writable, failed;
                        FD_ZERO(&writable);
                        FD_ZERO(&failed);
                        FD_SET(sock, &writable);
                        FD_SET(sock, &failed);
                        timeval slice = { 0, 100000 };
                        const int r = select(0, nullptr, &writable, &failed, &slice);
                        if (r == 0) continue;
                        open = r > 0 && FD_ISSET(sock, &writable) && !FD_ISSET(sock, &failed);
                        break;
                    }
                }
                closesocket(sock);
                if (open) return true;
                if (cancelled)
                {
                    error = cancelled_error();
                    return false;
                }
            }
            error = "no answer within " + std::to_string(connect_seconds) + " s";
            return false;
        }
#endif

        /*
            Resolves the name of a server and checks that it answers on its port, within
            connect_seconds for each, giving up at once when the downloads are cancelled.
            The answer of the resolution stays in the cache of the system, where the client
            that follows finds it, and its connection, which cannot be interrupted, finds a
            server that answers.  Elsewhere than on Windows, the client resolves and
            connects by itself, bounded by connect_seconds.
        */
        bool reach_bounded(const std::string& host, int port, std::string& error)
        {
#if defined(_WIN32)
            if (host == "localhost") return true;
            // Functions of Windows 8 and later, absent from the headers of MinGW.
            const HMODULE ws2 = GetModuleHandleW(L"ws2_32.dll");
            const auto cancel = ws2 ? reinterpret_cast<addrinfo_cancel_function>(
                reinterpret_cast<void*>(GetProcAddress(ws2, "GetAddrInfoExCancel"))) : nullptr;
            const auto overlapped_result = ws2 ? reinterpret_cast<addrinfo_result_function>(
                reinterpret_cast<void*>(GetProcAddress(ws2, "GetAddrInfoExOverlappedResult"))) : nullptr;
            if (!cancel || !overlapped_result) return true;
            std::unique_ptr<lookup> l(new lookup());
            l->overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!l->overlapped.hEvent) return true;
            ADDRINFOEXW hints = {};
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            const std::wstring name = widen(host);
            INT rc = GetAddrInfoExW(name.c_str(), nullptr, NS_DNS, nullptr, &hints, &l->result, nullptr,
                &l->overlapped, nullptr, &l->handle);
            if (rc == WSA_IO_PENDING)
            {
                const HANDLE waits[2] = { l->overlapped.hEvent, cancel_event() };
                const DWORD w = WaitForMultipleObjects(2, waits, FALSE, static_cast<DWORD>(connect_seconds) * 1000);
                if (w != WAIT_OBJECT_0)
                {
                    cancel(&l->handle);
                    // The system releases the request once it ends it; a request it does not
                    // end in time is left to it, with its memory.
                    if (WaitForSingleObject(l->overlapped.hEvent, 2000) != WAIT_OBJECT_0) l.release();
                    else
                    {
                        if (l->result) FreeAddrInfoExW(l->result);
                        CloseHandle(l->overlapped.hEvent);
                    }
                    error = w == WAIT_OBJECT_0 + 1 ? cancelled_error()
                        : "the name " + host + " was not resolved within " + std::to_string(connect_seconds) + " s";
                    return false;
                }
                rc = overlapped_result(&l->overlapped);
            }
            bool ok = rc == NO_ERROR;
            if (!ok) error = "the name " + host + " cannot be resolved (error " + std::to_string(rc) + ")";
            else if (!probe(l->result, port, error))
            {
                ok = false;
                error = host + ":" + std::to_string(port) + ": " + error;
            }
            if (l->result) FreeAddrInfoExW(l->result);
            CloseHandle(l->overlapped.hEvent);
            return ok;
#else
            (void)host;
            (void)port;
            (void)error;
            return true;
#endif
        }

        // Registers the client of a request, so that http_cancel_all() can stop it.
        class registration
        {
        public:
            explicit registration(const std::shared_ptr<httplib::Client>& c) : c_(c)
            {
                std::lock_guard<std::mutex> lock(active_m);
                active.insert(c_);
            }
            ~registration()
            {
                std::lock_guard<std::mutex> lock(active_m);
                active.erase(c_);
            }
            registration(const registration&) = delete;
            registration& operator=(const registration&) = delete;

        private:
            std::shared_ptr<httplib::Client> c_;
        };

        // The port of an origin, "https://host:port".
        int port_of(const std::string& origin)
        {
            const bool https = origin.compare(0, 8, "https://") == 0;
            const std::string h = origin.substr(origin.find("://") + 3);
            const size_t bracket = h.find(']');
            const size_t colon = h.find(':', bracket == std::string::npos ? 0 : bracket);
            const int port = colon == std::string::npos ? 0 : std::atoi(h.substr(colon + 1).c_str());
            return port > 0 && port < 65536 ? port : https ? 443 : 80;
        }

        // The client for one address, the server or the proxy reached first.
        std::shared_ptr<httplib::Client> open(const url_parts& u, int timeout_seconds, std::string& error)
        {
            const std::string proxy = proxy_for(u.origin);
            proxy_address p;
            const bool via_proxy = !proxy.empty() && parse_proxy(proxy, p);
            if (via_proxy ? !reach_bounded(lower(p.host), p.port, error)
                          : !reach_bounded(host_of(u.origin), port_of(u.origin), error))
                return nullptr;
            return make_client(u.origin, proxy, timeout_seconds);
        }

        // The address a redirection leads to, from the address that answered it.
        bool redirection(const std::string& from, const std::string& location, std::string& to, std::string& error)
        {
            url_parts u;
            parse_url(from, u);
            if (location.find("://") != std::string::npos) to = location;
            else if (location.compare(0, 2, "//") == 0) to = from.substr(0, from.find("://") + 1) + location;
            else if (!location.empty() && location[0] == '/') to = u.origin + location;
            else
            {
                const std::string path = u.target.substr(0, u.target.find('?'));
                to = u.origin + path.substr(0, path.rfind('/') + 1) + location;
            }
            if (from.compare(0, 8, "https://") == 0 && to.compare(0, 8, "https://") != 0)
            {
                error = from + " redirects to an address without https";
                return false;
            }
            return true;
        }

        const int max_redirections = 5;
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

    void http_cancel_all(const std::string& reason)
    {
        std::vector<std::shared_ptr<httplib::Client>> clients;
        {
            std::lock_guard<std::mutex> lock(active_m);
            cancel_reason = reason;
            cancelled = true;
            clients.assign(active.begin(), active.end());
        }
#if defined(_WIN32)
        SetEvent(cancel_event());
#endif
        // stop() waits for a connection being opened to be open: each client is stopped
        // from a thread of its own, which keeps it alive, so that no caller waits.
        for (const auto& c : clients) std::thread([c]() { c->stop(); }).detach();
    }

    void http_restore()
    {
        cancelled = false;
#if defined(_WIN32)
        ResetEvent(cancel_event());
#endif
    }

    bool http_cancelled()
    {
        return cancelled;
    }

    bool http_get(const std::string& url, std::string& body, std::string& error, size_t max_bytes, int timeout_seconds)
    {
        std::string current = url;
        for (int hop = 0; hop <= max_redirections; ++hop)
        {
            url_parts u;
            if (!parse_url(current, u))
            {
                error = "invalid address " + current;
                return false;
            }
            if (cancelled)
            {
                error = cancelled_error();
                return false;
            }
            const std::shared_ptr<httplib::Client> c = open(u, timeout_seconds, error);
            if (!c)
            {
                error = url + ": " + error;
                return false;
            }
            registration reg(c);
            // A cancellation between the resolution and the registration.
            if (cancelled)
            {
                error = cancelled_error();
                return false;
            }
            body.clear();
            bool too_big = false;
            int status = 0;
            std::string location;
            httplib::Result r;
            try
            {
                r = c->Get(u.target,
                    [&](const httplib::Response& res) {
                        status = res.status;
                        if (res.status >= 300 && res.status < 400) location = res.get_header_value("Location");
                        return res.status == 200 && !cancelled;
                    },
                    [&](const char* data, size_t n) {
                        if (cancelled) return false;
                        if (body.size() + n > max_bytes)
                        {
                            too_big = true;
                            return false;
                        }
                        body.append(data, n);
                        return true;
                    });
            }
            catch (const connection_cancelled&)
            {
            }
            if (cancelled)
            {
                error = cancelled_error();
                return false;
            }
            if (!location.empty())
            {
                if (!redirection(current, location, current, error)) return false;
                continue;
            }
            if (too_big)
            {
                error = url + " is larger than expected";
                return false;
            }
            if (status != 0 && status != 200)
            {
                error = url + ": status " + std::to_string(status);
                return false;
            }
            if (!r)
            {
                error = url + ": " + httplib::to_string(r.error());
                return false;
            }
            return true;
        }
        error = url + ": too many redirections";
        return false;
    }

    bool http_get_range(const std::string& url, uint64_t from, int64_t to,
        const std::function<bool(const char*, size_t)>& receiver, std::string& error, uint64_t* total)
    {
        httplib::Headers headers;
        if (from > 0 || to >= 0)
        {
            const std::string last = to >= 0 ? std::to_string(to) : std::string();
            headers.emplace("Range", "bytes=" + std::to_string(from) + "-" + last);
        }
        std::string current = url;
        for (int hop = 0; hop <= max_redirections; ++hop)
        {
            url_parts u;
            if (!parse_url(current, u))
            {
                error = "invalid address " + current;
                return false;
            }
            if (cancelled)
            {
                error = cancelled_error();
                return false;
            }
            const std::shared_ptr<httplib::Client> c = open(u, 30, error);
            if (!c)
            {
                error = url + ": " + error;
                return false;
            }
            registration reg(c);
            if (cancelled)
            {
                error = cancelled_error();
                return false;
            }
            bool refused_range = false;
            bool stopped = false;
            int status = 0;
            std::string location;
            httplib::Result r;
            try
            {
                r = c->Get(u.target, headers,
                    [&](const httplib::Response& res) {
                        status = res.status;
                        if (res.status >= 300 && res.status < 400)
                        {
                            location = res.get_header_value("Location");
                            return false;
                        }
                        if (total)
                        {
                            *total = 0;
                            const std::string cr = res.get_header_value("Content-Range");
                            const size_t slash = cr.find('/');
                            if (slash != std::string::npos && cr.substr(slash + 1) != "*")
                                *total = std::strtoull(cr.substr(slash + 1).c_str(), nullptr, 10);
                            else if (res.status == 200 && res.has_header("Content-Length"))
                                *total = std::strtoull(res.get_header_value("Content-Length").c_str(), nullptr, 10);
                        }
                        if (from > 0 && res.status == 200)
                        {
                            refused_range = true;
                            return false;
                        }
                        return (res.status == 200 || res.status == 206) && !cancelled;
                    },
                    [&](const char* data, size_t n) {
                        if (cancelled || !receiver(data, n))
                        {
                            stopped = true;
                            return false;
                        }
                        return true;
                    });
            }
            catch (const connection_cancelled&)
            {
            }
            if (cancelled || stopped)
            {
                error = cancelled ? cancelled_error() : std::string("stopped");
                return false;
            }
            if (!location.empty())
            {
                if (!redirection(current, location, current, error)) return false;
                continue;
            }
            if (refused_range)
            {
                error = url + ": the server does not resume downloads";
                return false;
            }
            if (!r)
            {
                error = url + ": " + (status ? "status " + std::to_string(status) : httplib::to_string(r.error()));
                return false;
            }
            return true;
        }
        error = url + ": too many redirections";
        return false;
    }

}
