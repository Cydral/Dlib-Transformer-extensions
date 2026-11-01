// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_HTTP_FETCH_H_
#define ALBIREO_HOST_HTTP_FETCH_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/*
    The downloads of the service: the update feed, the catalog of the models, the
    installers and the model files.  Every request runs on a thread of its own caller, a
    background task of the service, and is bounded in time at each stage: the resolution
    of the name of the server (on Windows), the connection, then each wait for data.  A
    service that stops, or that the settings put offline, calls http_cancel_all(), which
    ends the requests in progress at their next stage and makes the following ones fail at
    once, so that an unreachable network, as on a machine without Internet access, never
    delays the service.

    The web access of the agent mode (web_access.h) makes its own requests with the same
    proxy (http_proxy_for()), the same bounded resolution (http_resolve()) and the same
    cancellation (http_registration).
*/

namespace httplib { class Client; }

namespace albireo
{

    struct url_parts
    {
        std::string origin;   // "https://host:port"
        std::string target;   // "/path?query"
    };

    bool parse_url(const std::string& url, url_parts& out);

    /*!
        ensures
            - Sets the proxy of the downloads that follow, from the "proxy" setting:
                - "" uses the proxy of the system: the variables HTTPS_PROXY and HTTP_PROXY
                  (with NO_PROXY), then, on Windows, the proxy of WinHTTP that
                  "netsh winhttp set proxy" configures for the services;
                - "none" connects directly;
                - "http://[user:password@]host[:port]" names a proxy, reached with CONNECT.
            - The local addresses are always reached directly.
    !*/
    void set_http_proxy(const std::string& setting);

    // Whether a value of the "proxy" setting is valid.
    bool valid_proxy_setting(const std::string& setting);

    /*!
        ensures
            - Ends the downloads in progress: a transfer stops at once, and on Windows a
              name being resolved or a server being reached as well.  A connection that
              httplib is opening, its TLS handshake or the answer of a proxy included,
              ends within its bounds (connect_seconds, then the timeout of the request
              for the proxy).  A request about to connect opens nothing: its client checks
              the cancellation before its socket connects.
            - Makes the downloads that follow fail at once, until http_restore().
            - reason becomes the error of the downloads it ends or refuses.
    !*/
    void http_cancel_all(const std::string& reason);
    void http_restore();
    bool http_cancelled();

    // Seconds allowed to resolve the name of a server, and to connect to it.
    const int connect_seconds = 10;

    /*!
        ensures
            - Downloads url into body, following up to five redirections, verifying the
              certificates of https servers against those of the system.  Fails beyond
              max_bytes, or when no data arrives during timeout_seconds.
    !*/
    bool http_get(const std::string& url, std::string& body, std::string& error, size_t max_bytes = 16u << 20,
        int timeout_seconds = 20);

    /*!
        ensures
            - Downloads the bytes [from, to] of url (to < 0: up to the end), handing them to
              receiver as they arrive; receiver returns false to stop.  Redirections are
              followed with the same range.
            - total, when given, receives the size of the whole resource when the server
              tells it (Content-Range or Content-Length), 0 otherwise.
            - Fails when the server ignores the range of a request that does not start at
              0, or answers with a range that starts elsewhere, since writing its answer
              would corrupt the file.
            - With to >= 0, the receiver gets the bytes [from, to] at most: the reading ends
              once they have come, the answer of a server that ignores the range included.
    !*/
    bool http_get_range(const std::string& url, uint64_t from, int64_t to,
        const std::function<bool(const char*, size_t)>& receiver, std::string& error, uint64_t* total = nullptr);

    /*!
        ensures
            - Gives in host and port the proxy through which the downloads reach origin
              ("https://host:port"), as set_http_proxy() chooses it, with its user and its
              password when it has them.
            - Returns false when origin is reached directly: the setting "none", no proxy of
              the system, a local address or a host of NO_PROXY.
    !*/
    bool http_proxy_for(const std::string& origin, std::string& host, int& port, std::string& user,
        std::string& password);

    /*!
        ensures
            - Gives in addresses the IPv4 and IPv6 addresses of the name host, as text
              ("93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"), in the order of the
              resolver of the system.  On Windows the resolution takes connect_seconds at most
              and ends at once when the downloads are cancelled.
            - Returns false with error when the name is not resolved or when the downloads are
              cancelled.
    !*/
    bool http_resolve(const std::string& host, std::vector<std::string>& addresses, std::string& error);

    class http_registration
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The registration of the client of a request while the request runs, so that
                http_cancel_all() stops it: the downloads of this module register their
                clients, and so do the requests of the web access (web_access.h).
        !*/
    public:
        explicit http_registration(const std::shared_ptr<httplib::Client>& c);
        ~http_registration();
        http_registration(const http_registration&) = delete;
        http_registration& operator=(const http_registration&) = delete;

    private:
        std::shared_ptr<httplib::Client> c_;
    };

}

#endif // ALBIREO_HOST_HTTP_FETCH_H_
