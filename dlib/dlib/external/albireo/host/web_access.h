// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_WEB_ACCESS_H_
#define ALBIREO_HOST_WEB_ACCESS_H_

#include <string>

#include "../common/json.h"
#include "settings.h"

/*
    The web access.  The Web button of the composer of the web application gives the model
    two tools, web_search and fetch_page, which reach the web through two routes of the
    service:

        POST /albireo/v1/web/search   {"query", "count", "language"}
        POST /albireo/v1/web/fetch    {"url", "language"}

    The search goes through the provider of the settings (web_search): by default ("auto") the
    Brave Search API with the key built into the service when the build has one (the file of
    ALBIREO_WEB_SEARCH_KEY_FILE, cmake/web_search_key.cmake), else the search of Wikipedia
    through its REST interface; "brave" takes the key of the administrator (web_search_key), else
    the one built in; "searxng" a SearXNG instance through its JSON format.  Every installation
    of a build shares the monthly credit of the key built in.  The addresses of the providers
    are fixed or chosen by the administrator, and the rule of the public addresses does not
    apply to them.  The key built in never reaches a message, the log, a route or a document of
    the settings.

    A page is read at a public address only.  The name of its host is refused when it names
    this computer or a local network (localhost, *.localhost, *.local, *.internal, *.home.arpa,
    a name of a single label); otherwise it is resolved, and the page is refused when one of
    its addresses is not public (public_address()).  The connection goes to the address
    checked (Client::set_hostname_addr_map), so that the name is not resolved again between
    the check and the connection, while the Host header, the server name of TLS and the check
    of the certificate keep the name.  The redirections are followed here, five at most, each
    address checked as the first one.  Behind a proxy (http_fetch.h), the request goes through
    the proxy, which resolves the name again on its own: the check applies to the addresses
    this computer resolves, and a name it cannot resolve is refused.

    A text is read up to 2 MB and cut beyond, a PDF file up to 20 MB and refused beyond; the
    answer gives the bytes in base64, which the web application decodes.  A request takes
    10 s at most to connect, 15 s for each wait for data and 30 s in all, and keeps no cookie.
    Both routes are refused while the service works offline or once the administrator turned
    the web access off, and their requests end at once when the service stops
    (http_cancel_all()).  The log names the host read and the provider of a search, never a
    path, a query or the text searched.
*/

namespace albireo
{

    // The answer of a route of the web access: its body with the status 200, else the error
    // the server gives with send_error().
    struct web_reply
    {
        int status = 200;
        json body;
        std::string message;      // the error, in English
        std::string code;         // its code: invalid_url, address_refused, offline, web_access_off...
        std::string param;        // the member of the request it concerns, empty for none
        int retry_after = 0;      // seconds to wait before another request, 0 when unknown
    };

    /*!
        ensures
            - Returns true when numeric, an IPv4 address in dotted decimal or an IPv6 address,
              is public.  These are not: the IPv4 ranges 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10,
              127.0.0.0/8, 169.254.0.0/16, 172.16.0.0/12, 192.0.0.0/24, 192.0.2.0/24,
              192.88.99.0/24, 192.168.0.0/16, 198.18.0.0/15, 198.51.100.0/24, 203.0.113.0/24,
              224.0.0.0/4 and 240.0.0.0/4; the IPv6 ranges ::/96 (::, ::1 and the
              IPv4-compatible addresses), 64:ff9b:1::/48, 100::/64, 2001::/32, 2001:db8::/32,
              fc00::/7, fe80::/10, fec0::/10 and ff00::/8; the
              addresses of ::ffff:0:0/96, 64:ff9b::/96 and 2002::/16 whose embedded IPv4
              address is not public.
            - Returns false for a text that is not an address.
    !*/
    bool public_address(const std::string& numeric);

    // Whether the build carries a key of the Brave Search API (cmake/web_search_key.cmake).
    bool web_search_key_builtin();

    /*!
        ensures
            - Returns the provider of the search of s: for "auto", "brave" when the build
              carries a key, else "wikipedia"; "wikipedia"; "brave" with the key of the
              administrator, or without one with the key built in; "searxng" with the address
              of an instance; "none" otherwise.  A search of the Brave Search API sends the
              key of the administrator when there is one, else the key built in.
    !*/
    std::string web_search_provider(const host_settings& s);

    /*!
        ensures
            - Returns what /albireo/v1/status gives the web application as "web":
              {"available": true when web_access is set and the service is not offline,
               "search": web_search_provider(s) while available, "none" otherwise}.  The page
              reading is available with the web access, whatever the provider of the search.
    !*/
    json web_capability(const host_settings& s);

    // Whether url can be the setting web_search_url: empty, or an http or https address of
    // 512 characters at most, with a host and without user information, query or fragment.
    bool valid_web_search_url(const std::string& url);

    // Whether key can be the setting web_search_key: empty, or 256 printable ASCII
    // characters at most, without space.
    bool valid_web_search_key(const std::string& key);

    /*!
        ensures
            - Answers POST /albireo/v1/web/fetch: reads the page at request["url"], an http or
              https address, with the Accept-Language request["language"] when it is given.
            - The body of the reply is {"url", "final_url", "status", "content_type",
              "charset", "size", "truncated", "data"}: the address asked and the one that
              answered after the redirections, the status of the site (4xx and 5xx included,
              with their body), the media type of its Content-Type and its charset ("" without
              one), the number of bytes read, whether the text was cut at 2 MB, and the bytes
              in base64.
            - Refuses with 400 an invalid address; 403 an address that is not public, an
              offline service and a web access turned off; 413 a PDF file above 20 MB; 415 a
              type other than text/<subtype>, application/xhtml+xml, application/xml,
              application/<name>+xml, application/json, application/<name>+json and
              application/pdf; 502 a site that cannot be reached or whose TLS connection fails;
              503 a fourth request at once (code busy) and a request the stop of the service
              ended; 504 a site that does not answer in time.
    !*/
    web_reply web_fetch(const json& request, const host_settings& s);

    /*!
        ensures
            - Answers POST /albireo/v1/web/search: searches request["query"], 1 to 400
              characters, through web_search_provider(s), request["count"] results at most (1
              to 10, 5 by default), in the language request["language"] when it is given
              ([a-z]{2,3} with an optional region).
            - The body of the reply is {"provider", "results": [{"title", "url", "snippet"}]},
              the results whose address is http or https only.
            - Refuses with 400 an invalid request; 403 an offline service, a web access turned
              off and a search without provider (code no_search_provider); 429 a provider that
              limits the rate of the requests, with retry_after when it gives one; 502 a
              provider that refuses the key or the format, gives another answer than its
              interface, or cannot be reached; 503 as web_fetch(); 504 a provider that does not
              answer in time.
    !*/
    web_reply web_search(const json& request, const host_settings& s);

    /*
        Reserved for the tests (tests/test_host.cpp), which serve their pages and the answers
        of the providers from a local server.  No setting and no route reaches these functions.
    */
    // Lets web_fetch() read the loopback addresses, 127.0.0.0/8 and ::1.
    void web_allow_loopback_for_tests(bool allow);
    // Sends the requests of Wikipedia and of the Brave Search API to origin, "http://127.0.0.1:<port>";
    // an empty origin gives them back their own addresses.
    void web_provider_origin_for_tests(const std::string& origin);
    // Takes key in place of the key built into the service, "" for a build without one.
    void web_builtin_key_for_tests(const std::string& key);

}

#endif // ALBIREO_HOST_WEB_ACCESS_H_
