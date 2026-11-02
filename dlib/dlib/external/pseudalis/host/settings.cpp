// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "settings.h"

#include <cctype>

#include "../common/log.h"
#include "../common/platform.h"

namespace pseudalis
{
    const char* const default_public_link =
        "https://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV";
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
        j["pdf_layout"] = pdf_layout;
        j["albireo"] = albireo;
        j["max_file_mb"] = max_file_mb;
        j["removed_days"] = removed_days;
        j["check_updates"] = check_updates;
        j["trust_authority"] = trust_authority;
        j["public_link"] = public_link;
        j["feed_url"] = feed_url;
        j["proxy"] = proxy;
        return j;
    }

    host_settings host_settings::from_json(const json& j)
    {
        host_settings d, s;
        const long long port = j["port"].as_int(d.port);
        s.port = port >= 1024 && port <= 65535 ? static_cast<int>(port) : d.port;
        s.language = known_language(j["language"].str(d.language)) ? j["language"].str(d.language) : d.language;
        s.drive_letter = j["drive_letter"].str(d.drive_letter);
        if (!s.drive_letter.empty() && !valid_drive_letter(s.drive_letter)) s.drive_letter = d.drive_letter;
        if (!s.drive_letter.empty())
            s.drive_letter = std::string(1, static_cast<char>(
                std::toupper(static_cast<unsigned char>(s.drive_letter[0]))));
        s.date_order = j["date_order"].str(d.date_order) == "mdy" ? "mdy" : "dmy";
        s.pdf_layout = j["pdf_layout"].as_bool(d.pdf_layout);
        s.albireo = j["albireo"].str(d.albireo) == "off" ? "off" : "auto";
        const long long mb = j["max_file_mb"].as_int(d.max_file_mb);
        s.max_file_mb = mb >= 1 && mb <= 2048 ? static_cast<int>(mb) : d.max_file_mb;
        const long long days = j["removed_days"].as_int(d.removed_days);
        s.removed_days = days >= 1 && days <= 3650 ? static_cast<int>(days) : d.removed_days;
        s.check_updates = j["check_updates"].as_bool(d.check_updates);
        s.trust_authority = j["trust_authority"].as_bool(d.trust_authority);
        s.public_link = j.contains("public_link") ? j["public_link"].str() : d.public_link;
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
            log_warning("settings: " + path_ + " is not valid JSON, the defaults replace it");
            rename_file(path_, path_ + ".invalid");
        }
        write();
    }

    host_settings settings_store::get() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return s_;
    }

    bool settings_store::set(const host_settings& s)
    {
        std::lock_guard<std::mutex> lock(m_);
        s_ = s;
        return write();
    }

    host_settings settings_store::update(const std::function<void(host_settings&)>& change)
    {
        std::lock_guard<std::mutex> lock(m_);
        change(s_);
        if (!write()) log_error("settings: cannot write " + path_);
        return s_;
    }

    bool settings_store::write()
    {
        return write_file_atomic(path_, s_.to_json().dump_formatted(2) + "\n");
    }
}
