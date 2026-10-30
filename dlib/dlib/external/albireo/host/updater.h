// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_UPDATER_H_
#define ALBIREO_HOST_UPDATER_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "../common/json.h"

namespace albireo
{

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

                A machine without Internet access receives the same files on a removable
                medium: import_offline() reads the signed feed and the installer it names
                from a folder, checks them as a download would be checked, and stages the
                update the same way.

                A staged update is the installer of its version in the updates folder, the
                signed feed that announced it beside it (feed-<version>.json and its .sig),
                and staged.json, which names them.  The installer is checked against that
                feed when the service reads staged.json and again right before it starts,
                held open meanwhile so that no other program can change it.  An update
                older than the one staged never replaces it.

                status() gives the state ("idle", "checking", "up_to_date", "downloading",
                "ready" or "error"), the versions, and the release notes of the feed in
                every language it has, for the tray to show in its own.
        !*/
    public:
        updater(const std::string& feed_url, const std::string& updates_dir);
        ~updater();

        // Checks the feed and downloads the update on a thread of its own.
        void check_async();
        // Ends a check in progress and refuses the following ones: the service stops.
        void cancel();
        json status() const;

        /*!
            ensures
                - Stages the update found in folder: feed.json, feed.json.sig and the
                  installer the feed names, by its file name.
                - Returns {"state": "ready", "version", "installer"} when the update is
                  staged, {"state": "up_to_date", "version"} when the folder holds this
                  version or an older one, {"error"} otherwise; "installer" names the file
                  of the installer in folder once the feed is read.
        !*/
        json import_offline(const std::string& folder);

        // The installer of a downloaded update newer than this program, or empty.
        std::string staged_installer() const;

        // Starts the installer of the staged update in its update mode.
        bool launch_staged(std::string& error);

    private:
        enum class stage_result { staged, older, failed };

        void check();
        /*!
            ensures
                - Records the installer of version, checked against the signed feed (its
                  exact bytes and signature), as the update of the next start, with the feed
                  beside it.
                - Returns older, staging nothing, when a newer version is staged, and failed,
                  with error, when the files cannot be written.
        !*/
        stage_result stage(const std::string& version, const std::string& sha, const json& notes,
            const std::string& feed, const std::string& signature, std::string& error);
        std::string installer_path(const std::string& version) const;
        // The signed feed of the update of version, kept beside its installer.
        std::string feed_path(const std::string& version) const;
        // Reads the signed feed beside the installer of version: the size and the SHA-256 it
        // gives for the installer; false, with error, when it is missing or altered.
        bool staged_feed(const std::string& version, uint64_t& size, std::string& sha, std::string& error) const;
        // Checks staged.json, read at the start, against the signed feed and the installer.
        bool check_staged(const json& staged, std::string& error) const;
        // Forgets the staged update, whose files no longer match: the next check stages it again.
        void drop_staged(const std::string& error);

        std::string feed_url_;
        std::string dir_;
        mutable std::mutex m_;
        json status_;
        json staged_;   // the update applied at the next start: version, file, sha256, notes
        std::thread worker_;
        std::mutex worker_m_;               // the start of a check, from a route or the service
        std::atomic<bool> busy_{false};
        std::atomic<bool> cancelled_{false};
    };

}

#endif // ALBIREO_HOST_UPDATER_H_
