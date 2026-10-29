// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_UPDATER_H_
#define ALBIREO_HOST_UPDATER_H_

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "../common/json.h"

namespace albireo
{

    // Compares versions "1.2.3": negative, zero or positive as a is older, equal or newer.
    int compare_versions(const std::string& a, const std::string& b);

    class updater
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The updates of the programs.  The feed is a JSON document published on
                GitHub with a detached signature (ECDSA P-256 over its exact bytes); a feed
                whose signature does not verify against the key built into the programs is
                ignored.  When it announces a newer version, its installer is downloaded in
                the background into the updates folder and checked against the hash the
                feed gives.  The update is applied when the service restarts: the service
                then starts the installer in its update mode, which replaces the programs
                and starts the service again.

                A check made from the menu of the tray downloads the update in the same
                way and leaves it for the next restart; "apply" restarts at once.

                status() gives the state ("idle", "checking", "up_to_date", "downloading",
                "ready" or "error"), the versions, and the release notes of the feed in
                every language it has, for the tray to show in its own.
        !*/
    public:
        updater(const std::string& feed_url, const std::string& updates_dir);
        ~updater();

        void check_async();
        json status() const;

        // The installer of a downloaded update newer than this program, or empty.
        std::string staged_installer() const;

        // Starts the installer of the staged update in its update mode.
        bool launch_staged(std::string& error);

    private:
        void check();

        std::string feed_url_;
        std::string dir_;
        mutable std::mutex m_;
        json status_;
        std::thread worker_;
        std::atomic<bool> busy_{false};
    };

}

#endif // ALBIREO_HOST_UPDATER_H_
