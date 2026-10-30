// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SETTINGS_H_
#define ALBIREO_HOST_SETTINGS_H_

#include <functional>
#include <map>
#include <mutex>
#include <string>

#include "../common/json.h"

namespace albireo
{

    // The roles a language model can be activated for: one model at most per role.
    extern const char* const model_roles[3];   // "general", "code", "security"
    // The role of the embedding model, which computes the vectors of the document search
    // of the web application; one is always active.
    extern const char* const embedding_role;   // "embedding"

    // The text that stands for the password of the proxy in settings.json and in the answers
    // of the administration interface; settings_store keeps the password itself apart.
    extern const char* const proxy_password_mask;   // "********"
    // The proxy setting with its password, when it has one, replaced by proxy_password_mask.
    std::string masked_proxy(const std::string& proxy);
    // Whether the password of a proxy setting is proxy_password_mask.
    bool proxy_password_masked(const std::string& proxy);

    struct host_settings
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of the service, kept in settings.json in the data folder.
                The tray application changes them through the administration interface,
                and the service applies them, restarting itself when a change requires it
                (the port, the engine, the memory settings).
        !*/
        int port = 8443;
        std::string models_dir;                 // empty: <data>/models
        std::string language = "auto";          // auto, en, fr, de, it, es
        std::string engine = "auto";            // auto, best, or a variant: cpu, cpu-avx2, cpu-avxvnni, ...
        std::string accelerator = "auto";       // auto, none, npu, gpu, both (offloading engines)
        bool v2ram = true;                      // extended device memory of the tensor engines
        long long v2ram_max_mb = 0;             // 0: automatic
        long long device_mb = 0;                // copies of the weights on the device, 0: automatic
        int threads = 0;                        // 0: physical cores
        long context = 0;                       // 0: automatic per model
        int max_sequences = 4;                  // requests computed together per model
        long long kv_mb = 0;                    // 0: automatic
        std::map<std::string, std::string> active;   // role -> model identifier
        bool check_updates = true;
        bool default_model = true;              // download the default model of the catalog at the first start
        // A machine without Internet access: no request leaves the machine, the catalog,
        // the updates and the models come from files (import folder, removable media).
        bool offline = false;
        std::string proxy;                      // "": the proxy of the system, "none", or http://host:port
        std::string update_feed =
            "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/feed.json";
        std::string catalog_url =
            "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/catalog.json";
        json engine_benchmarks = json::object();     // variant -> measured tokens per second

        // The settings as settings.json and the administration interface give them, the
        // password of the proxy masked.
        json to_json() const;
        void from_json(const json& j);
        std::string effective_models_dir() const;
    };

    class settings_store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings file, read at start and written atomically at each change.

                settings.json is readable by the users of the machine, and gives the
                password of the proxy as proxy_password_mask.  The proxy with its password
                is kept in proxy.secret beside it, which only the service and the
                administrators read.  A password found in settings.json, written by an
                earlier version or by hand, moves to proxy.secret at once.
        !*/
    public:
        explicit settings_store(const std::string& path);

        host_settings get() const;
        bool set(const host_settings& s);
        /*!
            ensures
                - Applies change to the current settings and writes them, in one step: two
                  changes made at once, by a route and a background task for instance,
                  both persist.
                - Returns the settings written.
        !*/
        host_settings update(const std::function<void(host_settings&)>& change);
        const std::string& path() const { return path_; }

    private:
        // Writes proxy.secret, then settings.json; m_ is held.
        bool write();

        std::string path_;
        mutable std::mutex m_;
        host_settings s_;
    };

}

#endif // ALBIREO_HOST_SETTINGS_H_
