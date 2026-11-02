// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "settings.h"

#include <algorithm>

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    const char* const model_roles[3] = { "general", "code", "security" };
    const char* const embedding_role = "embedding";
    const char* const proxy_password_mask = "********";

    namespace
    {
        // Where the password of a proxy setting "http://user:password@host:port" lies, as
        // http_fetch.cpp reads the setting; false when it has none.
        bool password_span(const std::string& proxy, size_t& from, size_t& length)
        {
            const size_t scheme = proxy.find("://");
            const size_t start = scheme == std::string::npos ? 0 : scheme + 3;
            const size_t at = proxy.rfind('@');
            if (at == std::string::npos || at < start) return false;
            const size_t colon = proxy.find(':', start);
            if (colon == std::string::npos || colon > at) return false;
            from = colon + 1;
            length = at - from;
            return length > 0;
        }

        // The setting without its password, ":password" removed.
        std::string without_password(const std::string& proxy)
        {
            size_t from = 0, length = 0;
            if (!password_span(proxy, from, length)) return proxy;
            return proxy.substr(0, from - 1) + proxy.substr(from + length);
        }

        // The file of the proxy with its password, beside settings.json.
        std::string secret_path(const std::string& settings_path)
        {
            return join_path(parent_path(settings_path), "proxy.secret");
        }

        // The file of the key of the search (web_search_key), beside settings.json.
        std::string key_path(const std::string& settings_path)
        {
            return join_path(parent_path(settings_path), "web_search.secret");
        }

        // The text of a secret file without the ends of line and the spaces that follow it.
        std::string trimmed(std::string text)
        {
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
                text.pop_back();
            return text;
        }

        // The copy of the last settings written, read when settings.json is no longer valid.
        std::string backup_path(const std::string& settings_path)
        {
            return settings_path + ".bak";
        }

        bool read_settings(const std::string& path, std::string& text, json& j, std::string& error)
        {
            if (!read_file(path, text))
            {
                error = "unreadable";
                return false;
            }
            return json::parse(text, j, &error) && j.is_object();
        }
    }

    std::string masked_proxy(const std::string& proxy)
    {
        size_t from = 0, length = 0;
        if (!password_span(proxy, from, length)) return proxy;
        return proxy.substr(0, from) + proxy_password_mask + proxy.substr(from + length);
    }

    bool proxy_password_masked(const std::string& proxy)
    {
        size_t from = 0, length = 0;
        return password_span(proxy, from, length) && proxy.compare(from, length, proxy_password_mask) == 0;
    }

    json host_settings::to_json() const
    {
        json j = json::object();
        j["port"] = port;
        j["models_dir"] = models_dir;
        j["language"] = language;
        j["engine"] = engine;
        j["accelerator"] = accelerator;
        j["v2ram"] = v2ram;
        j["v2ram_max_mb"] = v2ram_max_mb;
        j["device_mb"] = device_mb;
        j["threads"] = threads;
        j["context"] = context;
        j["max_sequences"] = max_sequences;
        j["kv_mb"] = kv_mb;
        json a = json::object();
        for (const auto& kv : active) a[kv.first] = kv.second;
        j["active"] = a;
        j["check_updates"] = check_updates;
        j["default_model"] = default_model;
        j["offline"] = offline;
        j["proxy"] = masked_proxy(proxy);
        j["web_access"] = web_access;
        j["web_search"] = web_search;
        j["web_search_url"] = web_search_url;
        j["web_search_key_set"] = !web_search_key.empty();
        j["public_link"] = public_link;
        j["engine_benchmarks"] = engine_benchmarks;
        if (trust_authority >= 0) j["trust_authority"] = trust_authority > 0;
        return j;
    }

    void host_settings::from_json(const json& j)
    {
        const host_settings d;
        port = static_cast<int>(j["port"].as_int(d.port));
        if (port < 1 || port > 65535) port = d.port;
        models_dir = j["models_dir"].str(d.models_dir);
        language = j["language"].str(d.language);
        engine = j["engine"].str(d.engine);
        accelerator = j["accelerator"].str(d.accelerator);
        v2ram = j["v2ram"].as_bool(d.v2ram);
        v2ram_max_mb = j["v2ram_max_mb"].as_int(d.v2ram_max_mb);
        device_mb = std::max<long long>(0, j["device_mb"].as_int(d.device_mb));
        threads = static_cast<int>(j["threads"].as_int(d.threads));
        context = static_cast<long>(j["context"].as_int(d.context));
        max_sequences = static_cast<int>(j["max_sequences"].as_int(d.max_sequences));
        if (max_sequences < 1 || max_sequences > 64) max_sequences = d.max_sequences;
        kv_mb = j["kv_mb"].as_int(d.kv_mb);
        active.clear();
        for (const auto& kv : j["active"].items())
        {
            if (!kv.second.is_string() || kv.second.as_string().empty()) continue;
            for (const char* r : model_roles)
                if (kv.first == r) active[kv.first] = kv.second.as_string();
            if (kv.first == embedding_role) active[kv.first] = kv.second.as_string();
        }
        check_updates = j["check_updates"].as_bool(d.check_updates);
        default_model = j["default_model"].as_bool(d.default_model);
        offline = j["offline"].as_bool(d.offline);
        proxy = j["proxy"].str(d.proxy);
        web_access = j["web_access"].as_bool(d.web_access);
        web_search = j["web_search"].str(d.web_search);
        web_search_url = j["web_search_url"].str(d.web_search_url);
        // The key comes from the administration route or from web_search.secret: a document without it keeps it.
        if (j["web_search_key"].is_string()) web_search_key = j["web_search_key"].as_string();
        public_link = j["public_link"].str(d.public_link);
        engine_benchmarks = j["engine_benchmarks"].is_object() ? j["engine_benchmarks"] : json::object();
        trust_authority = j["trust_authority"].is_bool() ? (j["trust_authority"].as_bool() ? 1 : 0) : d.trust_authority;
    }

    std::string host_settings::effective_models_dir() const
    {
        return models_dir.empty() ? join_path(data_dir(), "models") : models_dir;
    }

    settings_store::settings_store(const std::string& path) : path_(path)
    {
        bool restored = false;
        // A key of the search written in the settings by hand, which moves to web_search.secret.
        bool key_in_document = false;
        if (file_exists(path_))
        {
            json j;
            std::string text, error;
            if (read_settings(path_, text, j, error))
            {
                s_.from_json(j);
                key_in_document = j["web_search_key"].is_string();
                // The copy follows a valid file, the one the installer or a hand wrote included.
                const std::string backup = backup_path(path_);
                std::string copy;
                if ((!read_file(backup, copy) || copy != text) && !write_file_atomic(backup, text))
                    log_warning("settings: the copy of " + path_ + " cannot be written");
            }
            else
            {
                /*
                    A file damaged on its disk, or by a hand: the copy of the last valid settings
                    serves, so that the offline mode, the port, the functions of the models and
                    the folders stay as they were, and settings.json is written again from it.
                */
                json saved;
                std::string saved_text, unused;
                restored = read_settings(backup_path(path_), saved_text, saved, unused);
                if (restored) s_.from_json(saved);
                key_in_document = restored && saved["web_search_key"].is_string();
                log_warning("settings: " + path_ + " is not valid JSON (" + error + "), "
                    + (restored ? "its last copy is used" : "defaults used"));
            }
        }
        std::lock_guard<std::mutex> lock(m_);
        // The key of the search is read before any write, which keeps web_search.secret for a key only.
        std::string key;
        if (!key_in_document && read_file(key_path(path_), key, 4096)) s_.web_search_key = trimmed(key);
        if (proxy_password_masked(s_.proxy))
        {
            // The password stored for this address and this user.
            std::string stored;
            if (read_file(secret_path(path_), stored, 4096)) stored = trimmed(stored);
            if (!stored.empty() && masked_proxy(stored) == s_.proxy)
            {
                s_.proxy = stored;
            }
            else
            {
                log_warning("settings: the password of the proxy is not stored, the proxy is used without it");
                s_.proxy = without_password(s_.proxy);
            }
        }
        else if (masked_proxy(s_.proxy) != s_.proxy && !write())
        {
            log_warning("settings: the password of the proxy cannot move from " + path_ + " to proxy.secret");
        }
        // A key written in the document moves to web_search.secret once the password of the proxy is back.
        if (key_in_document && !write())
            log_warning("settings: the key of the search cannot move from " + path_ + " to web_search.secret");
        // Written again from its copy once the password of the proxy is back in the settings.
        if (restored && !write()) log_warning("settings: " + path_ + " cannot be written again");
    }

    bool settings_store::write()
    {
        make_dirs(parent_path(path_));
        // The password first, so that settings.json masks no password that is not stored.
        const std::string secret = secret_path(path_);
        const bool password = masked_proxy(s_.proxy) != s_.proxy;
        if (password && !write_file_restricted(secret, s_.proxy)) return false;
        // The key of the search, which settings.json gives as web_search_key_set.
        const std::string key_file = key_path(path_);
        const bool key = !s_.web_search_key.empty();
        if (key && !write_file_restricted(key_file, s_.web_search_key)) return false;
        const std::string text = s_.to_json().dump_formatted(2) + "\n";
        if (!write_file_atomic(path_, text)) return false;
        // The copy read when settings.json is damaged, written once settings.json is in place.
        if (!write_file_atomic(backup_path(path_), text)) log_warning("settings: its copy cannot be written");
        if (!password && file_exists(secret)) remove_file(secret);
        if (!key && file_exists(key_file)) remove_file(key_file);
        return true;
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
        if (!write()) log_warning("settings: " + path_ + " cannot be written");
        return s_;
    }

}
