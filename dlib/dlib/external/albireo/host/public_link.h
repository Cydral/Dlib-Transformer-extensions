// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_PUBLIC_LINK_H_
#define ALBIREO_HOST_PUBLIC_LINK_H_

#include <cstdint>
#include <string>

/*
    The public link of the published files of Albireo: a public folder of pCloud, read only,
    whose folder dist holds the update feed, the catalog of the models, their signatures and
    the installers, and whose folder models holds the files of the models of the catalog, each
    in the folder of its identifier under the name of the file of its official source
    (models/<identifier>/<file name>).  It is the only source of the downloads of the service:
    the feed, the catalog, the installers and the model files come from it alone, and the
    official addresses the catalog keeps (the hub of each model) are never contacted.  A link
    that is not set, that does not answer or that does not hold a file makes the operation
    fail with a message that names the file in the link.  The signatures and the hashes check
    the files of the link, so that it brings nothing the catalog and the feed do not name.

    The link answers through the API of pCloud: showpublink lists the folder with its
    subfolders and the identifier and size of each file, and getpublinkdownload gives the
    address of a file on a content server, valid for a few hours.  The password of the link
    goes with each request.  It is built into the service from a file outside the repository,
    read at the configuration (ALBIREO_PUBLIC_LINK_PASSWORD_FILE in CMakeLists.txt), and kept
    masked in the program; the errors and the logs never show it, nor the query of an address
    of the API.  A service built without that file reaches the link without password.

    Every request goes through http_fetch.h: it follows the proxy of the settings, is bounded
    in time, and fails at once when the service stops or goes offline.
*/

namespace albireo
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

    // Why a file of the link could not be read.
    enum class link_status
    {
        ok,
        unset,      // no valid public link is set
        missing,    // the link answers and does not hold the file: it is not published there yet
        failed      // the link does not answer, refuses the request, or gives no address for the file
    };

    /*!
        requires
            - path is relative to the folder of the link: "dist/feed.json",
              "models/<identifier>/<file name>".
        ensures
            - Gives in url the address of the file on a content server of the link, and in
              size its size, from a listing of the folder read at most 10 minutes before and
              an address obtained at most 30 minutes before, unless fresh asks for both again.
            - Returns false, with error and status (when given), when no link is set, when
              the link does not answer, or when it holds no such file; error then names the
              file in the link.
    !*/
    bool public_link_file(const std::string& path, std::string& url, uint64_t& size, std::string& error,
        bool fresh = false, link_status* status = nullptr);

    // Reads a file of the link of max_bytes at most, a feed, a catalog or their signatures.
    bool public_link_get(const std::string& path, std::string& body, std::string& error, size_t max_bytes,
        link_status* status = nullptr);

    // The message of a file the link does not hold: "the publication space of Albireo does not hold <path> yet".
    std::string not_published(const std::string& path);

    // The path in the link of a file of the model id published at url: models/<id>/<file name of url>.
    std::string public_link_model_path(const std::string& id, const std::string& url);

    // The message with the query of each address it quotes left out: an error of a request to the API.
    std::string without_queries(const std::string& message);

}

#endif // ALBIREO_HOST_PUBLIC_LINK_H_
