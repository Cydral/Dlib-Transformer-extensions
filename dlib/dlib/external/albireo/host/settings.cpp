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
        j["update_feed"] = update_feed;
        j["catalog_url"] = catalog_url;
        j["engine_benchmarks"] = engine_benchmarks;
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
        update_feed = j["update_feed"].str(d.update_feed);
        catalog_url = j["catalog_url"].str(d.catalog_url);
        engine_benchmarks = j["engine_benchmarks"].is_object() ? j["engine_benchmarks"] : json::object();
    }

    std::string host_settings::effective_models_dir() const
    {
        return models_dir.empty() ? join_path(data_dir(), "models") : models_dir;
    }

    settings_store::settings_store(const std::string& path) : path_(path)
    {
        std::string text;
        if (read_file(path_, text))
        {
            json j;
            std::string error;
            if (json::parse(text, j, &error)) s_.from_json(j);
            else log_warning("settings: " + path_ + " is not valid JSON (" + error + "), defaults used");
        }
        std::lock_guard<std::mutex> lock(m_);
        if (proxy_password_masked(s_.proxy))
        {
            // The password stored for this address and this user.
            std::string stored;
            if (read_file(secret_path(path_), stored, 4096))
                while (!stored.empty() && (stored.back() == '\n' || stored.back() == '\r' || stored.back() == ' '))
                    stored.pop_back();
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
    }

    bool settings_store::write()
    {
        make_dirs(parent_path(path_));
        // The password first, so that settings.json masks no password that is not stored.
        const std::string secret = secret_path(path_);
        const bool password = masked_proxy(s_.proxy) != s_.proxy;
        if (password && !write_file_restricted(secret, s_.proxy)) return false;
        if (!write_file_atomic(path_, s_.to_json().dump_formatted(2) + "\n")) return false;
        if (!password && file_exists(secret)) remove_file(secret);
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
