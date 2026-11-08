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
    // The update feed, in the folder dist/pseudalis of the repository on GitHub, beside the installer it names.
    extern const char* const default_feed_url;

    struct host_settings
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of the service, kept in settings.json in the data folder,
                which the users of the machine read (the tray takes the port there) and the
                administrators change, through the administration interface or the
                installer.  The service restarts itself when the port, the drive, the order
                of the dates, the proxy or the address of the feed change.
        !*/
        int port = 8444;
        std::string language = "auto";          // auto, en, fr, de, it, es: the tray, the shell, the descriptions
        std::string drive_letter = "A";         // the letter of the drive, empty without drive
        std::string date_order = "dmy";         // dmy or mdy: the numeric dates the detector reads
        std::string albireo = "auto";           // auto: the pictures are described when Albireo answers; off
        int max_file_mb = 150;                  // the largest file processed, in MB
        /*
            The days an original stays kept once its file left the drive, 0 for always; whatever
            the days, the originals kept never take more than history_mb, nor leave the disk of
            the data folder with less than 256 MB free: the oldest go first (service::make_room()).
        */
        int removed_days = 45;
        static const int max_removed_days = 3650;
        int history_mb = 10240;                 // the space the originals kept may take, in MB, max_file_mb at least
        static const int max_history_mb = 1048576;
        /*
            Whether the values of the files take their tags, which give them back from any copy with the
            key; false writes their substitutes alone, which only the record of the file gives back while
            it lasts: the drive, the web application and the interface take it unless a request says.
        */
        bool tags = true;
        bool check_updates = true;              // a check every day
        bool trust_authority = true;            // the local authority in the trusted roots (the installer)
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
                settings.json, read at start and written atomically at each change, outside
                the lock that get() takes.  A file that is no longer valid JSON is replaced by
                the defaults, kept aside as settings.json.invalid; a file that cannot be read
                stays as it is.  In both cases the originals of the files removed from the
                drive are kept without limit of time (removed_days 0) until the settings are
                set again, within the space of history_mb.
        !*/
    public:
        explicit settings_store(const std::string& path);

        host_settings get() const;
        bool set(const host_settings& s);
        // Applies change to the current settings and writes them in one step: result receives
        // them; false when they could not be written.
        bool update(const std::function<void(host_settings&)>& change, host_settings& result);
        const std::string& path() const { return path_; }

    private:
        bool write(const host_settings& s);
        std::string path_;
        mutable std::mutex m_;
        std::mutex write_m_;
        host_settings s_;
    };
}

#endif // PSEUDALIS_HOST_SETTINGS_H_
