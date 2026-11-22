// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_DISTRIBUTION_H_
#define ALBIREO_HOST_DISTRIBUTION_H_

#include <string>

/*
    The distribution of Albireo.  Its versions are published on the website of Cydral
    Technology, in one folder (ALBIREO_DISTRIBUTION_URL in common/version.h) that holds the
    update feed (feed.json), the catalog of the models (catalog.json), their detached signatures
    (feed.json.sig, catalog.json.sig) and the installers, the one the feed names among them.
    The service asks that folder alone for the feed, the catalog and the installer of an update
    (updater.h, models.h).  The files of the models come from the address the catalog gives for
    each of them, their official address on the hub of the models, and from nowhere else: the
    website holds no copy of them.

    The folder is public.  It asks for no account and no password, serves its files over HTTPS
    as they were published, byte for byte, and answers the ranges of bytes.  The signatures of
    the feed and of the catalog, checked against the key built into the programs, and the sizes
    and the hashes these two files give, check every file read: neither the website nor the hub
    brings anything the signed files do not name.

    Every request goes through http_fetch.h: it follows the proxy of the settings, is bounded in
    time, and fails at once when the service stops or goes offline.  The website serves the
    folder without a redirection; one it would answer is followed as for any download, over
    https alone, what comes being checked the same way.  The messages name the addresses, which
    are public.
*/

namespace albireo
{

    // The address of the file name of the folder of the versions: "feed.json", "catalog.json.sig",
    // "albireo-setup-online-1.0.29.exe".
    std::string distribution_address(const std::string& name);

    /*!
        Reserved for the tests (tests/test_host.cpp), which serve the folder of the versions from a
        local server.  No setting and no route reaches it.
        ensures
            - Reads the files of the versions under folder ("http://127.0.0.1:<port>/dist") in
              place of the folder of the website; an empty folder gives the website back.
    !*/
    void distribution_for_tests(const std::string& folder);

}

#endif // ALBIREO_HOST_DISTRIBUTION_H_
