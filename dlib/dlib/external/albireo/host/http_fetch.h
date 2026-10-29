// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_HTTP_FETCH_H_
#define ALBIREO_HOST_HTTP_FETCH_H_

#include <cstdint>
#include <functional>
#include <string>

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
            - Downloads url into body, following redirects, verifying the certificates of
              https servers against those of the system.  Fails beyond max_bytes.
    !*/
    bool http_get(const std::string& url, std::string& body, std::string& error, size_t max_bytes = 16u << 20,
        int timeout_seconds = 30);

    /*!
        ensures
            - Downloads the bytes [from, to] of url (to < 0: up to the end), handing them to
              receiver as they arrive; receiver returns false to stop.
            - total, when given, receives the size of the whole resource when the server
              tells it (Content-Range or Content-Length), 0 otherwise.
            - Fails when the server ignores the range of a request that does not start at
              0, since appending its answer would corrupt the file.
    !*/
    bool http_get_range(const std::string& url, uint64_t from, int64_t to,
        const std::function<bool(const char*, size_t)>& receiver, std::string& error, uint64_t* total = nullptr);

    // Size of a resource from a HEAD request, 0 when unknown.
    uint64_t http_content_length(const std::string& url, std::string& error);

}

#endif // ALBIREO_HOST_HTTP_FETCH_H_
