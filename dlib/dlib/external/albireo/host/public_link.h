// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_PUBLIC_LINK_H_
#define ALBIREO_HOST_PUBLIC_LINK_H_

#include <cstdint>
#include <string>

/*
    The publication space of Albireo: a public folder, read only, whose folder dist holds the
    update feed, the catalog of the models, their signatures and the installers, and whose
    folder models holds the files of the models of the catalog, each in the folder of its
    identifier under the name of the file of its official source (models/<identifier>/<file
    name>).  It is the only source of the updates: the feed, its signature and the installers
    come from it alone, and a file it does not hold, or a space that does not answer, makes the
    update fail with a message that names the file in the publication space (updater.h).  It is
    the first source of the catalog and of the model files, and their official addresses the
    second: when the space is not set, does not answer, does not hold a file, or holds one that
    its size, its signature or its hash refuses, the catalog comes from its official address in
    the repository of Albireo, and each part of a model from the official address the catalog
    gives for it, its hub (model_manager in models.h).  The signatures and the hashes check
    every file, whatever its source, so that neither source brings anything the catalog and the
    feed do not name.

    The folder is a public link of pCloud, a constant of the build that no setting, route or
    document gives.  It answers through the interface of pCloud: showpublink lists the folder
    with its subfolders and the identifier and size of each file, and getpublinkdownload gives
    the address of a file on a content server, valid for a few hours.  The password of the link
    goes with each request.  It is built into the service from a file outside the repository,
    read at the configuration (ALBIREO_PUBLIC_LINK_PASSWORD_FILE in CMakeLists.txt), and kept
    masked in the program.  A service built without that file reaches the link without password.

    No message names the link, its code, its password, the address of its interface or of a
    content server, or a host of pCloud: a failure names the file in the publication space and
    the kind of failure ("the publication space of Albireo cannot be reached for dist/feed.json:
    the request is answered with the status 403").  When the official address fails as well,
    the message adds that address, which is public, and its failure.  The code and the password
    go to hide_in_messages() (common/log.h), which the log and the errors of the routes apply
    last.  An answer other than the JSON of the interface (a web page that a proxy or a filter
    of the network sends, a body cut short, values of other types) fails the same way, and so
    does an entry of the listing without a valid identifier or size.

    Every request goes through http_fetch.h: it follows the proxy of the settings, is bounded
    in time, and fails at once when the service stops or goes offline.
*/

namespace albireo
{

    // Whether the space is set: always in the service, whose link is a constant of the build.
    bool public_link_enabled();
    // Whether the program carries a password for the link (built from the password file).
    bool public_link_has_password();

    // Why a file of the publication space could not be read.
    enum class link_status
    {
        ok,
        unset,      // no publication space is set: the tests only (public_link_for_tests())
        missing,    // the space answers and does not hold the file: it is not published there yet
        failed      // the space does not answer, refuses the request, answers otherwise than its interface,
                    // or gives no address for the file
    };

    /*!
        requires
            - path is relative to the publication space: "dist/feed.json",
              "models/<identifier>/<file name>".
        ensures
            - Gives in url the address of the file on a content server, and in size its size,
              from a listing of the space read at most 10 minutes before and an address
              obtained at most 30 minutes before, unless fresh asks for both again.  url goes
              to http_fetch.h alone, and never to a message.
            - Returns false, with error and status (when given), when the space does not answer
              or holds no such file; error then names the file in the space.
    !*/
    bool public_link_file(const std::string& path, std::string& url, uint64_t& size, std::string& error,
        bool fresh = false, link_status* status = nullptr);

    // Reads a file of the space of max_bytes at most, a feed, a catalog or their signatures.
    bool public_link_get(const std::string& path, std::string& body, std::string& error, size_t max_bytes,
        link_status* status = nullptr);

    // The message of a file the space does not hold: "the publication space of Albireo does not hold <path> yet".
    std::string not_published(const std::string& path);

    // The path in the space of a file of the model id published at url: models/<id>/<file name of url>.
    std::string public_link_model_path(const std::string& id, const std::string& url);

    // The message with the query of each address it quotes left out.
    std::string without_queries(const std::string& message);

    /*!
        Reserved for the tests (tests/test_host.cpp), which serve the interface of pCloud from a
        local server.  No setting and no route reaches it.
        ensures
            - Uses link, "http://127.0.0.1:<port>/publink/show?code=<code>", whose interface is
              that origin, in place of the link of the build, and forgets the listing and the
              addresses read before; an empty or invalid link leaves no space set
              (public_link_enabled() is false).  The code goes to hide_in_messages().
    !*/
    void public_link_for_tests(const std::string& link);

}

#endif // ALBIREO_HOST_PUBLIC_LINK_H_
