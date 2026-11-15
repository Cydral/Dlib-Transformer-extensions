// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "settings.h"

#include <algorithm>
#include <initializer_list>

#include "../common/log.h"
#include "../common/platform.h"
#include "certificates.h"

namespace albireo
{

    const char* const model_roles[3] = { "general", "code", "security" };
    const char* const embedding_role = "embedding";
    const char* const proxy_password_mask = "********";
    const char* const relay_secret_mask = "********";

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

        // The file of the secret of the relay, with what it was given for, beside settings.json.
        std::string relay_secret_path(const std::string& settings_path)
        {
            return join_path(parent_path(settings_path), "relay.secret");
        }

        // A value of a setting that is one of values, else def.
        std::string one_of(const std::string& value, std::initializer_list<const char*> values, const char* def)
        {
            for (const char* v : values)
                if (value == v) return value;
            return def;
        }

        // The file of the key of a search provider that the earlier versions kept beside settings.json.
        std::string obsolete_key_path(const std::string& settings_path)
        {
            return join_path(parent_path(settings_path), "web_search.secret");
        }

        /*
            The version of the documents settings.json that write() gives them: 2 since Albireo 1.0.21,
            whose graph of OpenVINO is off by default.  A document without it is of version 1.
        */
        const long long settings_version = 2;

        /*
            Whether a document of the settings holds a setting this version lacks, or another provider of the
            search, or comes from an earlier version of the document.
        */
        bool obsolete_settings(const json& j)
        {
            for (const char* key : { "public_link", "web_search_url", "web_search_key", "web_search_key_set" })
                if (j.contains(key)) return true;
            const std::string search = j["web_search"].as_string();
            return (j.contains("web_search") && search != "wikipedia" && search != "none")
                || j["settings_version"].as_int(1) < settings_version;
        }

        // The settings of a document of settings.json, from the version that wrote it.
        void read_document(host_settings& s, const json& j)
        {
            s.from_json(j);
            /*
                The default auto of the graph of OpenVINO, which the versions before 1.0.21 wrote, gives
                way to off, unless the accelerator cpu, the graph on the processor, was chosen.
            */
            if (j["settings_version"].as_int(1) < settings_version && s.graph == "auto" && s.accelerator != "cpu")
                s.graph = host_settings().graph;
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

    std::string relay_settings::secret_scope() const
    {
        json j = json::object();
        j["provider"] = provider;
        j["url"] = url;
        j["auth"] = auth;
        j["user"] = user;
        return j.dump();
    }

    json relay_settings::to_json() const
    {
        json j = json::object();
        j["enabled"] = enabled;
        j["provider"] = provider;
        j["url"] = url;
        j["model"] = model;
        j["auth"] = auth;
        j["user"] = user;
        j["secret"] = secret.empty() ? std::string() : std::string(relay_secret_mask);
        j["context"] = static_cast<long long>(context);
        j["max_output"] = static_cast<long long>(max_output);
        j["images"] = images;
        j["reasoning"] = reasoning;
        j["pseudonymize"] = pseudonymize;
        j["tags"] = tags;
        return j;
    }

    void relay_settings::from_json(const json& j)
    {
        const relay_settings d;
        enabled = j["enabled"].as_bool(d.enabled);
        provider = one_of(j["provider"].str(d.provider), { "openai", "ollama", "claude" }, "openai");
        url = j["url"].str(d.url);
        model = j["model"].str(d.model);
        auth = one_of(j["auth"].str(d.auth), { "key", "basic", "none" }, "key");
        user = j["user"].str(d.user);
        secret = j["secret"].str(d.secret);
        const long long c = j["context"].as_int(d.context);
        context = c >= 1024 && c <= (1LL << 24) ? static_cast<long>(c) : d.context;
        const long long m = j["max_output"].as_int(d.max_output);
        max_output = m >= 16 && m <= (1LL << 20) ? static_cast<long>(m) : d.max_output;
        images = j["images"].as_bool(d.images);
        reasoning = j["reasoning"].as_bool(d.reasoning);
        pseudonymize = j["pseudonymize"].as_bool(d.pseudonymize);
        tags = j["tags"].as_bool(d.tags);
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
        j["kv_type"] = kv_type;
        j["graph"] = graph;
        j["draft_tokens"] = draft_tokens;
        j["integer_prompt"] = integer_prompt;
        json a = json::object();
        for (const auto& kv : active) a[kv.first] = kv.second;
        j["active"] = a;
        j["check_updates"] = check_updates;
        j["offline"] = offline;
        j["proxy"] = masked_proxy(proxy);
        j["web_access"] = web_access;
        j["web_search"] = web_search;
        j["engine_benchmarks"] = engine_benchmarks;
        if (trust_authority >= 0) j["trust_authority"] = trust_authority > 0;
        j["relay"] = relay.to_json();
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
        kv_type = j["kv_type"].str(d.kv_type);
        if (kv_type != "auto" && kv_type != "q8" && kv_type != "f16") kv_type = d.kv_type;
        graph = j["graph"].str(d.graph);
        if (graph != "auto" && graph != "on" && graph != "off") graph = d.graph;
        draft_tokens = static_cast<int>(j["draft_tokens"].as_int(d.draft_tokens));
        if (draft_tokens < 0 || draft_tokens > 6) draft_tokens = d.draft_tokens;
        integer_prompt = j["integer_prompt"].as_bool(d.integer_prompt);
        active.clear();
        for (const auto& kv : j["active"].items())
        {
            if (!kv.second.is_string() || kv.second.as_string().empty()) continue;
            for (const char* r : model_roles)
                if (kv.first == r) active[kv.first] = kv.second.as_string();
            if (kv.first == embedding_role) active[kv.first] = kv.second.as_string();
        }
        check_updates = j["check_updates"].as_bool(d.check_updates);
        offline = j["offline"].as_bool(d.offline);
        proxy = j["proxy"].str(d.proxy);
        web_access = j["web_access"].as_bool(d.web_access);
        // The providers of the earlier versions, "auto", "brave" and "searxng", give way to Wikipedia.
        web_search = j["web_search"].as_string() == "none" ? "none" : d.web_search;
        engine_benchmarks = j["engine_benchmarks"].is_object() ? j["engine_benchmarks"] : json::object();
        trust_authority = j["trust_authority"].is_bool() ? (j["trust_authority"].as_bool() ? 1 : 0) : d.trust_authority;
        relay.from_json(j["relay"]);
    }

    std::string host_settings::effective_models_dir() const
    {
        return models_dir.empty() ? join_path(data_dir(), "models") : models_dir;
    }

    settings_store::settings_store(const std::string& path) : path_(path)
    {
        bool restored = false;
        // A document of an earlier version, written again without what this version lacks.
        bool obsolete = false;
        if (file_exists(path_))
        {
            json j;
            std::string text, error;
            if (read_settings(path_, text, j, error))
            {
                read_document(s_, j);
                version_ = j["settings_version"].as_int(1);
                obsolete = obsolete_settings(j);
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
                if (restored)
                {
                    read_document(s_, saved);
                    version_ = saved["settings_version"].as_int(1);
                }
                log_warning("settings: " + path_ + " is not valid JSON (" + error + "), "
                    + (restored ? "its last copy is used" : "defaults used"));
            }
        }
        std::lock_guard<std::mutex> lock(m_);
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
        if (s_.relay.secret == relay_secret_mask)
        {
            // The secret stored for this provider, this address and this user.
            std::string sealed, record_text;
            json record;
            const bool read = read_file(relay_secret_path(path_), sealed, 64 * 1024) && open_secret(sealed, record_text)
                && json::parse(record_text, record) && record["scope"].as_string() == s_.relay.secret_scope();
            s_.relay.secret = read ? record["secret"].as_string() : std::string();
            if (!read) log_warning("settings: the secret of the relay is not stored, the relay goes without it");
            record = json();
            wipe_secret(sealed);
            wipe_secret(record_text);
        }
        else if (!s_.relay.secret.empty() && !write())
        {
            log_warning("settings: the secret of the relay cannot move from " + path_ + " to relay.secret");
        }
        // Written again once the password of the proxy is back in the settings: from its copy, or without the
        // settings of an earlier version and the key of its search provider.
        if ((restored || obsolete || file_exists(obsolete_key_path(path_))) && !write())
            log_warning("settings: " + path_ + " cannot be written again");
    }

    bool settings_store::write()
    {
        make_dirs(parent_path(path_));
        // The password first, so that settings.json masks no password that is not stored.
        const std::string secret = secret_path(path_);
        const bool password = masked_proxy(s_.proxy) != s_.proxy;
        if (password && !write_file_restricted(secret, s_.proxy)) return false;
        const std::string relay_secret = relay_secret_path(path_);
        const bool relay_key = !s_.relay.secret.empty();
        if (relay_key)
        {
            json record = json::object();
            record["scope"] = s_.relay.secret_scope();
            record["secret"] = s_.relay.secret;
            std::string text = record.dump(), sealed;
            record = json();
            const bool kept = seal_secret(text, L"Albireo relay secret", sealed)
                && write_file_restricted(relay_secret, sealed);
            wipe_secret(text);
            wipe_secret(sealed);
            if (!kept) return false;
        }
        json document = s_.to_json();
        document["settings_version"] = std::max(version_, settings_version);
        const std::string text = document.dump_formatted(2) + "\n";
        if (!write_file_atomic(path_, text)) return false;
        // The copy read when settings.json is damaged, written once settings.json is in place.
        if (!write_file_atomic(backup_path(path_), text)) log_warning("settings: its copy cannot be written");
        if (!password && file_exists(secret)) remove_file(secret);
        if (!relay_key && file_exists(relay_secret)) remove_file(relay_secret);
        if (file_exists(obsolete_key_path(path_))) remove_file(obsolete_key_path(path_));
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
