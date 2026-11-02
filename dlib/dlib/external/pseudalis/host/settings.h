// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_SETTINGS_H_
#define PSEUDALIS_HOST_SETTINGS_H_

#include <functional>
#include <mutex>
#include <string>

#include "../common/json.h"

namespace pseudalis
{
    // The public link of the published files of Pseudalis (host/public_link.h).
    extern const char* const default_public_link;
    // The update feed on GitHub, read when the public link does not answer.
    extern const char* const default_feed_url;

    struct host_settings
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of the service, kept in settings.json in the data folder,
                which the users of the machine read (the tray takes the port there) and the
                administrators change, through the administration interface or the
                installer.  The service restarts itself when the port, the drive, the order
                of the dates, the proxy, the public link or the address of the feed change.
        !*/
        int port = 8444;
        std::string language = "auto";          // auto, en, fr, de, it, es: the tray, the shell, the descriptions
        std::string drive_letter = "A";         // the letter of the drive, empty without drive
        std::string date_order = "dmy";         // dmy or mdy: the numeric dates the detector reads
        bool pdf_layout = true;                 // false: the PDF files are rebuilt as text only
        std::string albireo = "auto";           // auto: the pictures are described when Albireo answers; off
        int max_file_mb = 256;                  // the largest file processed
        int removed_days = 30;                  // the days an original stays kept once its file left the drive
        bool check_updates = true;              // a check every day
        bool trust_authority = true;            // the local authority in the trusted roots (the installer)
        std::string public_link = default_public_link;
        std::string feed_url = default_feed_url;
        std::string proxy;                      // "", "none" or http://host:port (http_fetch.h)

        json to_json() const;
        static host_settings from_json(const json& j);
    };

    // Whether a drive letter is one the drive may take: D to Z, or A and B, without colon.
    bool valid_drive_letter(const std::string& letter);

    class settings_store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                settings.json, read at start and written atomically at each change.  A file
                that is no longer valid JSON is replaced by the defaults, kept aside as
                settings.json.invalid.
        !*/
    public:
        explicit settings_store(const std::string& path);

        host_settings get() const;
        bool set(const host_settings& s);
        // Applies change to the current settings and writes them in one step.
        host_settings update(const std::function<void(host_settings&)>& change);
        const std::string& path() const { return path_; }

    private:
        bool write();
        std::string path_;
        mutable std::mutex m_;
        host_settings s_;
    };
}

#endif // PSEUDALIS_HOST_SETTINGS_H_
