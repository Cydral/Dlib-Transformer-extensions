// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "settings.h"

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    const char* const model_roles[3] = { "general", "code", "security" };

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
        j["threads"] = threads;
        j["context"] = context;
        j["max_sequences"] = max_sequences;
        j["kv_mb"] = kv_mb;
        json a = json::object();
        for (const auto& kv : active) a[kv.first] = kv.second;
        j["active"] = a;
        j["check_updates"] = check_updates;
        j["default_model"] = default_model;
        j["proxy"] = proxy;
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
        threads = static_cast<int>(j["threads"].as_int(d.threads));
        context = static_cast<long>(j["context"].as_int(d.context));
        max_sequences = static_cast<int>(j["max_sequences"].as_int(d.max_sequences));
        if (max_sequences < 1 || max_sequences > 64) max_sequences = d.max_sequences;
        kv_mb = j["kv_mb"].as_int(d.kv_mb);
        active.clear();
        for (const auto& kv : j["active"].items())
            if (kv.second.is_string() && !kv.second.as_string().empty())
                for (const char* r : model_roles)
                    if (kv.first == r) active[kv.first] = kv.second.as_string();
        check_updates = j["check_updates"].as_bool(d.check_updates);
        default_model = j["default_model"].as_bool(d.default_model);
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
        make_dirs(parent_path(path_));
        return write_file_atomic(path_, s_.to_json().dump_formatted(2) + "\n");
    }

}
