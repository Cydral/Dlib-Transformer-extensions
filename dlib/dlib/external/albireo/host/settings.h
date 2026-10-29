// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SETTINGS_H_
#define ALBIREO_HOST_SETTINGS_H_

#include <map>
#include <mutex>
#include <string>

#include "../common/json.h"

namespace albireo
{

    // The roles a model can be activated for: one model at most per role.
    extern const char* const model_roles[3];   // "general", "code", "security"

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
        std::string engine = "auto";            // auto, best, cpu, cpu-avx2, openvino, cuda
        std::string accelerator = "auto";       // auto, none, npu, gpu, both (offloading engines)
        bool v2ram = true;                      // extended device memory of the tensor engines
        long long v2ram_max_mb = 0;             // 0: automatic
        int threads = 0;                        // 0: physical cores
        long context = 0;                       // 0: automatic per model
        int max_sequences = 4;                  // requests computed together per model
        long long kv_mb = 0;                    // 0: automatic
        std::map<std::string, std::string> active;   // role -> model identifier
        bool check_updates = true;
        bool default_model = true;              // download the default model of the catalog at the first start
        std::string proxy;                      // "": the proxy of the system, "none", or http://host:port
        std::string update_feed =
            "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/feed.json";
        std::string catalog_url =
            "https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/catalog.json";
        json engine_benchmarks = json::object();     // variant -> measured tokens per second

        json to_json() const;
        void from_json(const json& j);
        std::string effective_models_dir() const;
    };

    class settings_store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings file, read at start and written atomically at each change.
        !*/
    public:
        explicit settings_store(const std::string& path);

        host_settings get() const;
        bool set(const host_settings& s);
        const std::string& path() const { return path_; }

    private:
        std::string path_;
        mutable std::mutex m_;
        host_settings s_;
    };

}

#endif // ALBIREO_HOST_SETTINGS_H_
