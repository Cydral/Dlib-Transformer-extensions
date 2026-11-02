// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_PUBLIC_LINK_H_
#define PSEUDALIS_HOST_PUBLIC_LINK_H_

#include <cstdint>
#include <string>

/*
    The public link of the published files of Pseudalis: a public folder of pCloud, read only,
    whose folder dist holds the update feed, its signature and the installers
    (pseudalis-feed.json, pseudalis-feed.json.sig, pseudalis-setup-<version>.exe).  The
    service reads these files before their official address, GitHub for the feed, and falls
    back to it when the link does not answer or does not hold the file.  The signature and
    the hashes check the files of both sources the same way, so that the link brings nothing
    the feed does not name.

    The link answers through the API of pCloud: showpublink lists the folder with its
    subfolders and the identifier and size of each file, and getpublinkdownload gives the
    address of a file on a content server, valid for a few hours.  The password of the link
    goes with each request.  It is built into the service from a file outside the repository,
    read at the configuration (PSEUDALIS_PUBLIC_LINK_PASSWORD_FILE in CMakeLists.txt), and kept
    masked in the program; the errors and the logs never show it, nor the query of an address
    of the API.  A service built without that file reaches the link without password.

    Every request goes through http_fetch.h: it follows the proxy of the settings, is bounded
    in time, and fails at once when the service stops or goes offline.
*/

namespace pseudalis
{

    /*!
        ensures
            - Returns true when link is the address of a public link of pCloud,
              https://u.pcloud.link/publink/show?code=<code> (or e.pcloud.link, my.pcloud.com,
              e.pcloud.com), or a local address of the same form for the tests
              (http://127.0.0.1:<port>/publink/show?code=<code>), whose API is then that origin.
    !*/
    bool valid_public_link(const std::string& link);

    /*!
        ensures
            - Uses link, from the "public_link" setting, for the reads that follow, and
              forgets the listing and the addresses of the previous link; an empty or invalid
              link turns the reads off (public_link_enabled() is false).
    !*/
    void set_public_link(const std::string& link);
    bool public_link_enabled();
    // Whether the program carries a password for the link (built from the password file).
    bool public_link_has_password();

    /*!
        requires
            - path is relative to the folder of the link: "dist/pseudalis-feed.json".
        ensures
            - Gives in url the address of the file on a content server of the link, and in
              size its size, from a listing of the folder read at most 10 minutes before and
              an address obtained at most 30 minutes before, unless fresh asks for both again.
            - Returns false, with error, when no link is set, when the link does not answer,
              or when it holds no such file.
    !*/
    bool public_link_file(const std::string& path, std::string& url, uint64_t& size, std::string& error,
        bool fresh = false);

    // Reads a file of the link of max_bytes at most, the feed or its signature.
    bool public_link_get(const std::string& path, std::string& body, std::string& error, size_t max_bytes);

    // The message with the query of each address it quotes left out: an error of a request to the API.
    std::string without_queries(const std::string& message);

    // The message, preceded by "public link: " when it does not name the link: a reason for the logs.
    std::string about_public_link(const std::string& message);

}

#endif // PSEUDALIS_HOST_PUBLIC_LINK_H_
