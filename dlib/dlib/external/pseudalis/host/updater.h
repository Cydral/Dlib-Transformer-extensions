// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_UPDATER_H_
#define PSEUDALIS_HOST_UPDATER_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "../common/json.h"
#include "user_error.h"

namespace pseudalis
{

    class updater
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The updates of the programs.  The feed is a JSON document published at
                feed_url, the folder dist/pseudalis of the repository on GitHub, with a
                detached signature (ECDSA P-256 over its exact bytes, feed_url + ".sig"); a
                feed whose signature does not verify against the key built into the programs
                is ignored.  When it announces a newer version, its installer is downloaded
                in the background into the updates folder, from the folder of the feed when
                the feed names it there ("file"), else from its address ("url"), and checked
                against the size and the hash the feed gives.  The update is applied when the
                service restarts: the service then starts the installer in its update mode,
                which replaces the programs and starts the service again.  The addresses of
                the publication appear in no message, answer or log.

                A check made from the menu of the tray downloads the update in the same
                way and leaves it for the next restart; "apply" restarts at once.

                A staged update is the installer of its version in the updates folder, the
                signed feed that announced it beside it (feed-<version>.json and its .sig),
                and staged.json, which names them.  The installer is checked against that
                feed when the service reads staged.json and again right before it starts,
                held open meanwhile so that no other program can change it.  An update
                older than the one staged never replaces it.

                status() gives the state ("idle", "checking", "up_to_date", "downloading",
                "ready" or "error"), the versions, and the release notes of the feed in
                every language it has, for the tray to show in its own.

            ERRORS
                Every error, in the state and in the answers, comes as its English text
                ("error"), its code ("code") and its values ("args", an object), which the
                tray gives in the language of the user: update_check {e}, update_record
                {version}, update_feed_missing {version}, update_signature {file},
                update_feed_mismatch {file, version}, update_outside {file},
                update_installer_mismatch {version}, update_feed_invalid, update_download
                {e}, update_not_ready, update_invalid {version}, update_start {code},
                update_package_manager and stopping.
        !*/
    public:
        updater(const std::string& feed_url, const std::string& updates_dir);
        ~updater();

        // Checks the feed and downloads the update on a thread of its own.
        void check_async();
        // Ends a check in progress and refuses the following ones: the service stops.
        void cancel();
        json status() const;

        // The installer of a downloaded update newer than this program, or empty.
        std::string staged_installer() const;

        /*!
            ensures
                - Returns true when a restart applies the staged update; false otherwise,
                  with update_not_ready when none is staged, and update_package_manager on
                  the systems whose package manager applies the updates.
        !*/
        bool can_apply(user_error& error) const;

        // Starts the installer of the staged update in its update mode; a failure stays in the
        // state with its code, unless the service stops.
        bool launch_staged(user_error& error);

    private:
        void check();
        /*!
            ensures
                - Records the installer of version, checked against the signed feed (its
                  exact bytes and signature), as the update of the next start, with the feed
                  beside it.
                - Returns false, with error, when the files cannot be written.
        !*/
        bool stage(const std::string& version, const std::string& sha, const json& notes,
            const std::string& feed, const std::string& signature, user_error& error);
        std::string installer_path(const std::string& version) const;
        // The signed feed of the update of version, kept beside its installer.
        std::string feed_path(const std::string& version) const;
        // Reads the signed feed beside the installer of version: the size and the SHA-256 it
        // gives for the installer; false, with error, when it is missing or altered.
        bool staged_feed(const std::string& version, uint64_t& size, std::string& sha, user_error& error) const;
        // Checks staged.json, read at the start, against the signed feed and the installer.
        bool check_staged(const json& staged, user_error& error) const;
        // Forgets the staged update, whose files no longer match: the next check stages it again.
        void drop_staged(const user_error& error);

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

#endif // PSEUDALIS_HOST_UPDATER_H_
