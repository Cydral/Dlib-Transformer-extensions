// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_HTTP_FETCH_H_
#define ALBIREO_HOST_HTTP_FETCH_H_

#include <cstdint>
#include <functional>
#include <string>

/*
    The downloads of the service: the update feed, the catalog of the models, the
    installers and the model files.  Every request runs on a thread of its own caller, a
    background task of the service, and is bounded in time at each stage: the resolution
    of the name of the server (on Windows), the connection, then each wait for data.  A
    service that stops, or that the settings put offline, calls http_cancel_all(), which
    ends the requests in progress at their next stage and makes the following ones fail at
    once, so that an unreachable network, as on a machine without Internet access, never
    delays the service.
*/

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
              0, since appending its answer would corrupt the file.
    !*/
    bool http_get_range(const std::string& url, uint64_t from, int64_t to,
        const std::function<bool(const char*, size_t)>& receiver, std::string& error, uint64_t* total = nullptr);

}

#endif // ALBIREO_HOST_HTTP_FETCH_H_
