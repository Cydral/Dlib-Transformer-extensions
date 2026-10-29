// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SERVICE_H_
#define ALBIREO_HOST_SERVICE_H_

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/json.h"
#include "engine_host.h"
#include "models.h"
#include "settings.h"
#include "store.h"
#include "updater.h"

namespace httplib { class SSLServer; class Server; }

namespace albireo
{

    // The engine a machine runs: a library and, for the engines with several devices,
    // the device ("NPU", "GPU").  key names it in the measures: "cpu-avx2",
    // "openvino:npu", "openvino:gpu", "cuda".
    struct engine_choice
    {
        const engine_candidate* candidate = nullptr;
        std::string device;
        std::string key;
    };

    /*!
        ensures
            - Returns the engine to run for the settings s among the candidates found
              next to the service:
                - engine names a variant: that variant, when usable;
                - accelerator "npu", "gpu" or "both" (the setting of the tray, offered
                  when the processor engine runs): the engine of that device, the
                  fastest measured one for "both";
                - otherwise the fastest measured engine, or, without measures, the
                  graphics processor engine (cuda) when usable, then the processor
                  engines (cpu-avx2, cpu).
            - The returned candidate is null when no engine is usable.
    !*/
    engine_choice choose_engine(const std::vector<engine_candidate>& candidates, const host_settings& s);

    // The engines a set of candidates offers, one per device: what the tray lists.
    std::vector<engine_choice> engine_options(const std::vector<engine_candidate>& candidates);

    class service
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The Albireo service: the engine and its models, the local HTTPS server
                with the web application, the OpenAI compatible interface and the
                administration interface of the tray, the catalog and the downloads of
                the models, and the updates.

                run() starts everything and blocks until stop() or a restart request; the
                caller (the Windows service, or the console on other systems) then starts
                a new service when restart_requested() says so, which is how a change of
                port or of engine is applied.  When an update downloaded earlier is ready,
                run() starts its installer instead of serving and returns with
                update_started() true: the installer stops the service, replaces the
                programs and starts the service again.
        !*/
    public:
        service();
        ~service();

        // Starts and serves until stopped; returns false when it could not start.
        bool run(std::string& error);
        void stop();
        bool stopping() const { return stopping_; }
        bool restart_requested() const { return restart_; }
        bool update_started() const { return update_started_; }
        void request_restart();

        // The programs and the data.
        std::string program_dir() const { return program_dir_; }
        std::string data_dir() const { return data_dir_; }
        std::string admin_token() const { return admin_token_; }
        int port() const { return port_; }

        // What the interfaces need.
        settings_store& settings() { return *settings_; }
        store& db() { return *db_; }
        model_manager& models() { return *models_; }
        updater& updates() { return *updater_; }
        engine_host& engine() { return engine_; }

        json status();
        json engines();
        // The engine model of a request: an identifier, a role or its alias, or auto.
        std::string resolve_model(const std::string& requested);
        bool activate(const std::string& id, const std::string& role, std::string& error);
        void deactivate(const std::string& id);
        // Loads the models of the roles of the settings into the engine.
        void load_active_models();

        // Counts the usage of a finished request.
        void record_usage(long prompt_tokens, long generated_tokens);

        // Measures every usable engine on this machine with the active general model
        // (or the smallest installed one) and records the measures in the settings.
        json benchmark_engines();

        // Entry points of the child processes started by the service itself.
        static int probe_engine(const std::string& path);
        static int benchmark_engine(const std::string& path, const std::string& model, const std::string& config);

    private:
        void background_loop();
        void ensure_default_model();
        void scan_inbox();
        void on_installed(const installed_model& m);
        bool load_model(const installed_model& m, std::string& error);
        json engine_config(const host_settings& s, const std::string& device) const;
        std::string engines_dir() const;

        std::string program_dir_;
        std::string data_dir_;
        std::string admin_token_;
        int port_ = 0;
        std::unique_ptr<settings_store> settings_;
        std::unique_ptr<store> db_;
        std::unique_ptr<model_manager> models_;
        std::unique_ptr<updater> updater_;
        engine_host engine_;
        std::vector<engine_candidate> candidates_;
        engine_choice choice_;
        std::string engine_error_;

        std::mutex server_m_;
        std::unique_ptr<httplib::SSLServer> server_;

        std::atomic<bool> stopping_{false};
        std::atomic<bool> restart_{false};
        std::atomic<bool> update_started_{false};
        std::mutex m_;
        std::condition_variable cv_;
        std::thread background_;
        std::mutex models_m_;   // serializes the activations and the loads
        std::map<std::string, std::string> load_state_;   // model -> "loading", "ready" or the error
        int64_t started_at_ = 0;
        std::atomic<long long> usage_prompt_{0};
        std::atomic<long long> usage_generated_{0};
        std::atomic<long long> usage_requests_{0};
    };

    // Installs the routes of the HTTPS server (server.cpp).
    void install_routes(httplib::Server& server, service& svc);

}

#endif // ALBIREO_HOST_SERVICE_H_
