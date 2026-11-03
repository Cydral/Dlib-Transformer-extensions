// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "settings.h"

#include <cctype>

#include "../common/log.h"
#include "../common/platform.h"

namespace pseudalis
{
    const char* const default_feed_url =
        "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/pseudalis/feed.json";

    namespace
    {
        bool known_language(const std::string& l)
        {
            return l == "auto" || l == "en" || l == "fr" || l == "de" || l == "it" || l == "es";
        }
    }

    bool valid_drive_letter(const std::string& letter)
    {
        if (letter.size() != 1) return false;
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(letter[0])));
        return c == 'A' || c == 'B' || (c >= 'D' && c <= 'Z');
    }

    json host_settings::to_json() const
    {
        json j = json::object();
        j["port"] = port;
        j["language"] = language;
        j["drive_letter"] = drive_letter;
        j["date_order"] = date_order;
        j["albireo"] = albireo;
        j["max_file_mb"] = max_file_mb;
        j["removed_days"] = removed_days;
        j["check_updates"] = check_updates;
        j["trust_authority"] = trust_authority;
        // The place of the publication appears only when it differs from the one of the build.
        if (feed_url != default_feed_url) j["feed_url"] = feed_url;
        j["proxy"] = proxy;
        return j;
    }

    host_settings host_settings::from_json(const json& j)
    {
        host_settings d, s;
        const long long port = j["port"].as_int(d.port);
        s.port = port >= 1024 && port <= 65535 ? static_cast<int>(port) : d.port;
        s.language = known_language(j["language"].str(d.language)) ? j["language"].str(d.language) : d.language;
        // The letter of the drive, in upper case.
        std::string letter = j["drive_letter"].str(d.drive_letter);
        if (!letter.empty() && !valid_drive_letter(letter)) letter = d.drive_letter;
        if (!letter.empty())
            letter = std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(letter[0]))));
        s.drive_letter = letter;
        s.date_order = j["date_order"].str(d.date_order) == "mdy" ? "mdy" : "dmy";
        s.albireo = j["albireo"].str(d.albireo) == "off" ? "off" : "auto";
        const long long mb = j["max_file_mb"].as_int(d.max_file_mb);
        s.max_file_mb = mb >= 1 && mb <= 2048 ? static_cast<int>(mb) : d.max_file_mb;
        const long long days = j["removed_days"].as_int(d.removed_days);
        s.removed_days = days >= 1 && days <= max_removed_days ? static_cast<int>(days) : d.removed_days;
        s.check_updates = j["check_updates"].as_bool(d.check_updates);
        s.trust_authority = j["trust_authority"].as_bool(d.trust_authority);
        s.feed_url = j["feed_url"].str(d.feed_url);
        s.proxy = j["proxy"].str(d.proxy);
        return s;
    }

    settings_store::settings_store(const std::string& path) : path_(path)
    {
        std::string text;
        json j;
        if (read_file(path_, text, 1u << 20))
        {
            if (json::parse(text, j) && j.is_object())
            {
                s_ = host_settings::from_json(j);
                return;
            }
            /*
                The defaults replace a damaged file, kept aside, except for the days the originals
                of the files removed from the drive stay: the longest, so that none goes before an
                administrator sets them again.
            */
            log_error("settings: " + path_ + " is not valid JSON, kept as settings.json.invalid; the defaults "
                      "replace it, the originals of the files removed from the drive kept meanwhile");
            rename_file(path_, path_ + ".invalid");
            s_.removed_days = host_settings::max_removed_days;
        }
        else if (file_exists(path_))
        {
            // A file that cannot be read stays as it is: the defaults serve meanwhile, no original goes.
            log_error("settings: " + path_ + " cannot be read, the defaults serve meanwhile");
            s_.removed_days = host_settings::max_removed_days;
            return;
        }
        write(s_);
    }

    host_settings settings_store::get() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return s_;
    }

    bool settings_store::set(const host_settings& s)
    {
        // The file is written outside the lock that get() takes, in the order of the changes.
        std::lock_guard<std::mutex> order(write_m_);
        {
            std::lock_guard<std::mutex> lock(m_);
            s_ = s;
        }
        return write(s);
    }

    bool settings_store::update(const std::function<void(host_settings&)>& change, host_settings& result)
    {
        std::lock_guard<std::mutex> order(write_m_);
        {
            std::lock_guard<std::mutex> lock(m_);
            change(s_);
            result = s_;
        }
        if (write(result)) return true;
        log_error("settings: cannot write " + path_);
        return false;
    }

    bool settings_store::write(const host_settings& s)
    {
        return write_file_atomic(path_, s.to_json().dump_formatted(2) + "\n");
    }
}
