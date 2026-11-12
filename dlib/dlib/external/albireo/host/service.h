// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SERVICE_H_
#define ALBIREO_HOST_SERVICE_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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
                - engine names a variant: that variant, when usable, with the device the
                  accelerator names for the engines that have several;
                - accelerator "npu", "gpu" or "both": the engine of that device, the
                  fastest measured one for "both";
                - accelerator "none": the processor engine, the fastest measured one,
                  else the one of the richest kernels (cpu-avxvnni, cpu-avx2, cpu);
                - engine "best": the fastest measured engine;
                - otherwise, the automatic choice: the accelerator of the computer, the
                  NVIDIA card (cuda), else the NPU and the graphics processor of Intel
                  together (openvino:both), else the one of them present, each engine
                  computing on the processor the products its device does not take; the
                  processor engine when no accelerator is usable, or when a measure
                  found the accelerator slower than the processor engine.
            - The returned candidate is null when no engine is usable.
    !*/
    engine_choice choose_engine(const std::vector<engine_candidate>& candidates, const host_settings& s);

    /*!
        ensures
            - Returns the engines a set of candidates offers, one per device: what the tray
              lists.  Among the OpenVINO engines, one per family of processor kernels
              (openvino-avxvnni, openvino), the richest usable one offers the NPU, the
              graphics processor and both.
    !*/
    std::vector<engine_choice> engine_options(const std::vector<engine_candidate>& candidates);
    bool is_openvino_variant(const std::string& variant);

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

                The server answers as soon as its port is open.  The engines are described
                and the chosen one starts in the background, then the active models load:
                until then, starting_phase() names the step in progress, the status says
                so, and the requests for a model answer 503 with the code "starting", so
                that the tray and the clients tell a slow start from a failure.

            NETWORK
                The catalog, the updates and the downloads run on threads of their own,
                each request bounded in time, so that the local work (the models, the
                imports, the requests of the users) never waits for the network.  With
                the "offline" setting, no request leaves the machine: the catalog, the
                updates and the models come from folders, the import folder of the data
                folder or a folder named by the tray (import_folder()).
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
        // The folder of the voices of the speech synthesis and its program, beside the service.
        std::string speech_dir() const;
        std::string speech_program() const;
        updater& updates() { return *updater_; }
        engine_host& engine() { return engine_; }

        json status();
        json engines();
        // The step of the start in progress: "engines" (described, then the chosen one
        // starts), "models" (the active models load), or empty once the service is ready.
        std::string starting_phase() const;
        // True while the model of a role, or the model named, loads into the engine.
        bool model_loading(const std::string& requested);
        // The engine model of a request: an identifier, a role or its alias, "background",
        // or auto.
        std::string resolve_model(const std::string& requested);
        /*!
            ensures
                - Returns the model of the work the applications do in the background
                  (titles, summaries, memory, descriptions of the images of the projects):
                  the language model that comes with Albireo, which stays loaded whatever the
                  functions, or empty without one, the requests for "background" then going
                  to the general model.  The complete installer carries it; after the
                  installer of the updates, the default language model of the catalog, which
                  the service downloads at its first start (ensure_default_model()), stands
                  for it.
        !*/
        std::string background_model() const;
        /*!
            ensures
                - Returns the model of each function: the one chosen for it, else the general
                  model, else the language model that comes with Albireo, so that the three
                  functions keep a model once every choice is undone; the embedding role keeps
                  the model chosen for it.
        !*/
        std::map<std::string, std::string> effective_models(const host_settings& st) const;
        // The embedding model of a request: a loaded embedding model named by its
        // identifier, otherwise the active one ("auto", "embedding", "albireo-embedding").
        std::string resolve_embedding_model(const std::string& requested);
        /*!
            ensures
                - Returns, for each model loaded in the engine, {"inputs", "publisher"}: the
                  inputs the engine reads for it as its "models" query reports them ("text",
                  and "image" for a model loaded with its projector), "text" alone when the
                  engine does not say, and the company that made the model, from its entry,
                  empty when unknown.
        !*/
        std::map<std::string, json> loaded_traits();
        /*!
            ensures
                - Gives the function role to the model id, loading it when needed.  The
                  model the function used before leaves the engine once its requests are
                  served, unless another function uses it.  A model activated while it
                  waits to leave stays loaded.
        !*/
        bool activate(const std::string& id, const std::string& role, user_error& error);
        /*!
            ensures
                - Takes the function role from the model id, or every function when role
                  is empty, and gives it to the model that comes with Albireo, which then
                  serves it, else leaves it without a model chosen for it; unloads the
                  model once its requests are served when no function uses it any more.
                  The model of the document search keeps its function: the search always
                  has its embedding model, which another embedding model replaces through
                  activate().
        !*/
        bool deactivate(const std::string& id, const std::string& role, user_error& error);
        /*!
            ensures
                - Takes every function from the model id, the document search included,
                  and deletes its file.  A model in the engine is deleted once its requests
                  are served and it left the engine: pending is then true, and the model
                  stays listed, in the state "removing", until then.
                - A model that comes with Albireo, or the default model of the catalog that
                  stands for it (background_model()), is refused (code "builtin_model"): it
                  leaves its functions to other models and stays installed.
        !*/
        bool remove_model(const std::string& id, user_error& error, bool& pending);
        // The models waiting for their requests before they leave: id -> "unloading" or
        // "removing".
        json leaving_models();
        // Loads the models of the roles of the settings into the engine.
        void load_active_models();

        /*!
            ensures
                - Installs what path holds, for the route of the imports by path.  The
                  service reads with its own rights, those of the system on Windows, so that
                  path is accepted in two cases only, a file of another account being never
                  read, hashed, listed nor revealed:
                    - a folder or a file of the import folder of the data folder, which only
                      SYSTEM and the administrators fill: its signed catalog, its models
                      (*.gguf, in the folder and in its "models" subfolder) and its signed
                      update, as scan_inbox() installs them, the files staying in place;
                    - a signed offline bundle (tools/make_offline_bundle.py) on a local
                      drive: a folder whose catalog.json or feed.json bears the signature
                      of Albireo.  Its catalog, the models that catalog lists (found by
                      their names, sizes and hashes, model_manager::import_bundle()) and its
                      update are installed; no other file of the folder is read.
                  Every file is read through no link.  A model file of the user goes to the
                  service through the upload route, read with the rights of the user.
                - Returns {"imported", "already", "pending", "errors"} as
                  model_manager::import_path(), with "catalog" and "update" when the folder
                  holds them.
                - Refuses path, with refusal set and an empty result, when it is relative
                  (code import_path_relative), on a network share or drive
                  (import_path_network), a device or a path of the system namespaces
                  (import_path_device), invalid (import_path_invalid: a stream, a wildcard, a
                  control character, a name "." or ".."), a model file outside the import
                  folder (import_use_upload), outside the import folder and no signed
                  offline bundle reached through no link (import_not_bundle: the same code
                  whether the folder is missing, unreadable, unsigned or reached through a
                  link, so that the answer reveals nothing of a folder of another account),
                  missing in the import folder (import_not_found), reached through a link
                  in the import folder (import_path_link), or when the import folder lets
                  other accounts than the administrators write in it and cannot be
                  secured (import_folder_insecure).  Each refusal names the path in its
                  arguments ("path").
        !*/
        json import_folder(const std::string& path, user_error& refusal);

        // Applies the network settings at once: the proxy, and the offline mode, which
        // ends the requests in progress.
        void apply_network_settings(const host_settings& s);

        // Counts the usage of a finished request.
        void record_usage(long prompt_tokens, long generated_tokens);

        /*
            Measures every usable engine on this machine with the active general model (or
            the smallest installed one) and records the measures in the settings.  One
            measure runs at a time: a call during another one returns at once with the code
            "benchmark_running".  Each engine is measured while the engine of the service
            serves no request: the measure waits for its requests to end, two minutes at
            most, and is made again once when the service received requests meanwhile, so
            that the engines measured do not share the processor with it.
        */
        json benchmark_engines();

        // Entry points of the child processes started by the service itself.
        static int probe_engine(const std::string& path);
        static int benchmark_engine(const std::string& path, const std::string& model, const std::string& config);

    private:
        void background_loop();
        // Describes the engines next to the service and starts the one of the settings, a
        // processor engine when it fails; then engines_ready_ becomes true.
        void start_engines();
        /*!
            ensures
                - Registers the models that come with Albireo, in the folder of the programs.
                  A new one that takes the place of a model of its kind (a language model,
                  or an embedding model) that no longer comes with Albireo receives the
                  functions of that model, as an update that changes the model that comes
                  with Albireo does; another new one takes the functions no model holds.  The
                  functions of a model that left without a successor are freed.
                - A new language model receives the functions of the language models of the
                  catalog made of the file it was made from (source_sha256 of its manifest),
                  so that the same weights do not load twice.
                - The first language model among them serves the work in the background.
        !*/
        void register_builtin_models();
        /*!
            ensures
                - Gives the document search the embedding model installed, else downloads
                  the default embedding model of the catalog.
                - Without a language model that comes with Albireo, downloads the default
                  language model of the catalog, whatever the options of the installation
                  and the other models installed, unless the computer is offline: it then
                  stands for the model that comes with Albireo (background_model()).
                - A download of these models in progress goes on, one cancelled waits for the
                  next start of the service, and one that failed starts again.
                - Gives the general function, when no model holds it, to a language model
                  installed.
        !*/
        void ensure_default_model();
        /*
            The model that comes with Albireo, which serves the work in the background: the
            language model the complete installer carries, else the installed language model
            of the default entry of the catalog, else the default model of an earlier catalog
            chosen last (recorded in the database); a model being deleted is left out, and the
            result is empty without one.
        */
        std::string choose_background_model();
        /*
            Gives the work in the background to choose_background_model(), which loads whatever its functions,
            under models_m_, so that a deletion either sees the model chosen or leaves it out of the choice, and
            records the default model of the catalog chosen.
        */
        void update_background_model();
        // Logs why a default model of the catalog could not be downloaded, which the Models
        // window shows as the error of its download until a download starts.
        void note_default_failure(const json& entry, const user_error& error, const std::string& what);
        void scan_inbox();
        // Whether the import folder lets only the administrators write in it: secured again, its content
        // checked, when it does not; false with error when it cannot be.
        bool secure_inbox(std::string& error);
        /*!
            ensures
                - Gives a model that arrives the functions its entry names that no model
                  holds, and the general one when it is free.  A model of the engine whose
                  files changed, a projector given to it say, loads again once its requests
                  are served (retire() with reload).
                - A language model that becomes the model of the work in the background
                  (update_background_model()) takes only the functions that no other model
                  serves (effective_models()).
        !*/
        void on_installed(const installed_model& m);
        static bool is_embedding(const installed_model& m);
        /*
            The model that comes with Albireo, which takes the language functions a deactivation
            or a deletion takes from the model id: empty when there is none, when its file is
            missing or when it is id itself.
        */
        std::string successor_of(const std::string& id);
        // Loads a model into the engine, with its projector when it has one (options "projector").
        bool load_model(const installed_model& m, std::string& error);
        /*!
            ensures
                - Unloads the model id on a thread of its own once its requests are served
                  (engine_host::unload_model), so that the function it held passes at once
                  to the model that replaces it.
                - With reload, the model loads again from its record once it left the engine,
                  when a function or the work in the background still uses it; the model of
                  the work in the background is then unloaded too.
                - With background, the model of the work in the background leaves the engine
                  without loading again, for its files to be replaced (replace_model_files(),
                  which loads it again); otherwise it stays loaded.
        !*/
        void retire(const std::string& id, bool reload = false, bool background = false);
        // Loads the model id again after its reload, when a function or the work in the background uses it.
        void reload_model(const std::string& id);
        // Waits for the model id to finish unloading, when it is.
        void wait_retired(const std::string& id);
        /*!
            ensures
                - Runs place, which renames new files over those of the installed model id,
                  while the engine holds no file of id, which Windows would refuse to replace:
                  a model in the engine leaves it first, once its requests are served, and loads
                  again from its new files once they are recorded (on_installed()), or from its
                  files as they were when place fails.
                - Returns what place returns; false without running it when the model is being
                  deleted, comes with Albireo (its installer replaces its files), or when the
                  service stops.
        !*/
        bool replace_model_files(const std::string& id, const std::function<bool()>& place);
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
        // Written once by start_engines(), then read by the routes once engines_ready_ is
        // true.
        std::vector<engine_candidate> candidates_;
        engine_choice choice_;
        std::string engine_error_;
        // The model of the work in the background (background_model()), set by update_background_model().
        mutable std::mutex background_m_;
        std::string background_model_;
        std::atomic<bool> engines_ready_{false};
        std::atomic<bool> models_ready_{false};
        // Set by run() when a damaged database was set aside and a new one started.
        bool database_restarted_ = false;

        /*
            Guards the server, the model manager and the updater while run() creates and
            releases them, for stop(), which a thread of the system may call at any time, and
            for the download workers that end after the stop (choose_background_model()).
        */
        std::mutex server_m_;
        std::unique_ptr<httplib::SSLServer> server_;

        std::atomic<bool> stopping_{false};
        std::atomic<bool> restart_{false};
        std::atomic<bool> update_started_{false};
        std::mutex m_;
        std::condition_variable cv_;
        std::thread background_;
        std::thread restarter_;
        std::mutex models_m_;   // serializes the activations and the loads
        std::mutex import_m_;   // serializes the imports of folders
        std::mutex benchmark_m_;   // one measure of the engines at a time
        // model -> "loading", "ready", "unloading", "removing" or the error
        std::map<std::string, std::string> load_state_;
        // The models leaving the engine, each on a thread of retirers_, and those whose
        // file is deleted once they left.
        std::mutex retire_m_;
        std::condition_variable retire_cv_;
        std::set<std::string> retiring_;
        std::set<std::string> removing_;
        // The models that load again once they left the engine (retire() with reload).
        std::set<std::string> reloading_;
        std::vector<std::thread> retirers_;
        // The retirers that ended, joined by the next retirement.
        std::vector<std::thread::id> ended_retirers_;
        int64_t started_at_ = 0;
        std::atomic<long long> usage_prompt_{0};
        std::atomic<long long> usage_generated_{0};
        std::atomic<long long> usage_requests_{0};
    };

    // Installs the routes of the HTTPS server (server.cpp).
    void install_routes(httplib::Server& server, service& svc);

    // Whether the server can open port on the loopback interface, with the options of its
    // own socket; the port is closed again at once.
    bool port_available(int port);

}

#endif // ALBIREO_HOST_SERVICE_H_
