// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <dlib/external/httplib/httplib.h>

#include "service.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>

#include "../common/builtin_models.h"
#include "../common/engine_loader.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "../common/system_info.h"
#include "../common/version.h"
#include "certificates.h"
#include "http_fetch.h"

namespace albireo
{

    namespace
    {
        const int64_t catalog_period = 6 * 3600;
        const int64_t update_period = 6 * 3600;
        const int64_t certificate_period = 24 * 3600;
        const int64_t inbox_period = 60;

        // The errors of the requests that the service ends.
        const char* const stopping_reason = "the service stops";
        const char* const offline_reason = "offline mode: no request leaves this computer";

        std::string trim(const std::string& s)
        {
            const size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return std::string();
            const size_t e = s.find_last_not_of(" \t\r\n");
            return s.substr(b, e - b + 1);
        }

        bool ends_with(const std::string& s, const std::string& suffix)
        {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool is_role(const std::string& r)
        {
            for (const char* m : model_roles)
                if (r == m) return true;
            return r == embedding_role;
        }

        // True when the catalog entry of a model names the role among its uses.
        bool has_role(const json& descriptor, const std::string& role)
        {
            for (const auto& r : descriptor["roles"].elements())
                if (r.as_string() == role) return true;
            return false;
        }
    }

    bool is_openvino_variant(const std::string& variant)
    {
        return variant.compare(0, 8, "openvino") == 0;
    }

    std::vector<engine_choice> engine_options(const std::vector<engine_candidate>& candidates)
    {
        // The OpenVINO engine with the richest processor kernels that run here: its devices
        // make the choices of the NPU and of the graphics processor.
        const engine_candidate* openvino = nullptr;
        for (const char* v : { "openvino-avx512", "openvino-avxvnni", "openvino" })
            for (const auto& c : candidates)
                if (!openvino && c.usable && c.variant == v) openvino = &c;
        std::vector<engine_choice> out;
        for (const auto& c : candidates)
        {
            if (!c.usable) continue;
            if (is_openvino_variant(c.variant))
            {
                if (&c != openvino) continue;
                // One choice per kind of device, and one for the NPU and the graphics
                // processor together, where dlib routes each product to the faster of
                // them.
                std::string npu, gpu;
                for (const auto& d : c.description["devices"].elements())
                {
                    // The processor plugin is listed only when DLIB_NPU_DEVICE names it.
                    const std::string type = d["type"].as_string();
                    if (type != "npu" && type != "gpu" && !(type == "cpu" && d["id"].as_string() == "CPU")) continue;
                    const std::string key = "openvino:" + type;
                    bool seen = false;
                    for (const auto& e : out) seen = seen || e.key == key;
                    if (seen) continue;
                    engine_choice e;
                    e.candidate = &c;
                    e.device = d["id"].str(type == "npu" ? "NPU" : type == "gpu" ? "GPU" : "CPU");
                    e.key = key;
                    out.push_back(e);
                    if (type == "npu") npu = e.device;
                    if (type == "gpu") gpu = e.device;
                }
                if (!npu.empty() && !gpu.empty())
                {
                    engine_choice e;
                    e.candidate = &c;
                    e.device = npu + "," + gpu;
                    e.key = "openvino:both";
                    out.push_back(e);
                }
                continue;
            }
            engine_choice e;
            e.candidate = &c;
            e.key = c.variant;
            out.push_back(e);
        }
        return out;
    }

    engine_choice choose_engine(const std::vector<engine_candidate>& candidates, const host_settings& s)
    {
        const std::vector<engine_choice> options = engine_options(candidates);
        auto rate = [&](const engine_choice& e) {
            return s.engine_benchmarks[e.key]["tokens_per_second"].as_double(0);
        };
        auto is_cpu = [](const engine_choice& e) { return e.key.compare(0, 3, "cpu") == 0; };
        auto is_npu = [](const engine_choice& e) { return e.key == "openvino:npu"; };
        auto is_gpu = [](const engine_choice& e) { return e.key == "cuda" || e.key == "openvino:gpu"; };
        auto is_accelerator = [](const engine_choice& e) {
            return e.key == "cuda" || e.key == "openvino:npu" || e.key == "openvino:gpu" || e.key == "openvino:both";
        };
        auto fastest = [&](const std::function<bool(const engine_choice&)>& keep) {
            const engine_choice* best = nullptr;
            for (const auto& e : options)
                if (keep(e) && rate(e) > 0 && (!best || rate(e) > rate(*best))) best = &e;
            return best;
        };
        auto first = [&](const std::vector<std::string>& keys) {
            for (const auto& k : keys)
                for (const auto& e : options)
                    if (e.key == k) return &e;
            return static_cast<const engine_choice*>(nullptr);
        };

        if (s.engine != "auto" && s.engine != "best")
        {
            // A variant named in the settings; for the engines with several devices, the
            // accelerator setting picks the device.  "openvino" names the OpenVINO engine
            // that engine_options() keeps.
            const engine_choice* named = nullptr;
            for (const auto& e : options)
            {
                const bool openvino = s.engine == "openvino" && is_openvino_variant(e.candidate->variant);
                if (e.key != s.engine && e.candidate->variant != s.engine && !openvino) continue;
                if (!named) named = &e;
                if ((s.accelerator == "npu" && is_npu(e)) || (s.accelerator == "gpu" && is_gpu(e))
                    || (s.accelerator == "both" && e.key == "openvino:both")) named = &e;
            }
            if (named) return *named;
            log_warning("engines: the engine '" + s.engine + "' of the settings is not usable, automatic selection");
        }
        const engine_choice* e = nullptr;
        if (s.accelerator == "npu")
        {
            e = fastest(is_npu);
            if (!e) e = first({ "openvino:npu" });
        }
        else if (s.accelerator == "gpu")
        {
            e = fastest(is_gpu);
            if (!e) e = first({ "cuda", "openvino:gpu" });
        }
        else if (s.accelerator == "both")
        {
            e = fastest(is_accelerator);
            if (!e) e = first({ "cuda", "openvino:both", "openvino:gpu", "openvino:npu" });
        }
        else if (s.accelerator == "none")
        {
            e = fastest(is_cpu);
            if (!e) e = first({ "cpu-avx512", "cpu-avxvnni", "cpu-avx2", "cpu" });
        }
        if (!e) e = fastest([](const engine_choice&) { return true; });
        if (!e) e = first({ "cuda", "cpu-avx512", "cpu-avxvnni", "cpu-avx2", "cpu" });
        if (!e && !options.empty()) e = &options.front();
        return e ? *e : engine_choice();
    }

    service::service() : program_dir_(executable_dir()), data_dir_(albireo::data_dir())
    {
    }

    service::~service()
    {
        stop();
        if (background_.joinable()) background_.join();
        if (restarter_.joinable()) restarter_.join();
    }

    std::string service::engines_dir() const
    {
        const std::string d = join_path(program_dir_, "engines");
        return dir_exists(d) ? d : program_dir_;
    }

    json service::engine_config(const host_settings& s, const std::string& device) const
    {
        json c = json::object();
        c["threads"] = s.threads;
        c["context"] = s.context;
        c["max_sequences"] = s.max_sequences;
        c["kv_mb"] = s.kv_mb;
        c["v2ram"] = s.v2ram;
        c["v2ram_max_mb"] = s.v2ram_max_mb;
        c["offload_mb"] = s.device_mb;
        c["device"] = device;
        c["cache_dir"] = join_path(data_dir_, "engine-cache");
        return c;
    }

    bool service::run(std::string& error)
    {
        started_at_ = now_unix();
        if (!make_dirs(data_dir_))
        {
            error = "cannot create the data folder " + data_dir_;
            return false;
        }
        const std::string logs = join_path(data_dir_, "logs");
        make_dirs(logs);
        log_file::instance().open(join_path(logs, "service.log"));
        log_info(std::string(ALBIREO_FULL_NAME) + " " + ALBIREO_VERSION + " starting");
        log_info("programs in " + program_dir_ + ", data in " + data_dir_);

        const std::string settings_path = join_path(data_dir_, "settings.json");
        const bool first_run = !file_exists(settings_path);
        settings_.reset(new settings_store(settings_path));
        const host_settings st = settings_->get();
        if (first_run) settings_->set(st);
        port_ = st.port;
        apply_network_settings(st);

        try
        {
            db_.reset(new store(join_path(data_dir_, "albireo.db")));
        }
        catch (const std::exception& e)
        {
            error = std::string("the database cannot be opened: ") + e.what();
            return false;
        }

        // The token of the administration interface, read by the tray.
        const std::string token_path = join_path(data_dir_, "admin.token");
        std::string token;
        if (!read_file(token_path, token, 4096) || trim(token).size() < 32)
        {
            token = random_hex(48);
            if (!write_file_restricted(token_path, token, true))
            {
                error = "cannot write " + token_path;
                return false;
            }
        }
        admin_token_ = trim(token);

        const std::string cert_dir = join_path(data_dir_, "certificates");
        if (!ensure_certificates(cert_dir, error)) return false;

        // An update downloaded earlier is applied before anything else starts.
        {
            std::unique_ptr<updater> u(new updater(st.update_feed, join_path(data_dir_, "updates")));
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) u->cancel();
            updater_ = std::move(u);
        }
        if (!updater_->staged_installer().empty() && !stopping_)
        {
            std::string e;
            if (updater_->launch_staged(e))
            {
                update_started_ = true;
                log_info("updates: the installer of version " + updater_->status()["latest"].as_string()
                    + " is started, the service stops for the update");
                return true;
            }
            if (!e.empty()) log_warning("updates: " + e);
        }

        {
            std::unique_ptr<model_manager> m(new model_manager(*db_, st.effective_models_dir(), st.catalog_url));
            m->on_installed([this](const installed_model& im) { on_installed(im); });
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) m->stop();
            models_ = std::move(m);
        }

        register_builtin_models();

        // The engines start in the background (start_engines()), once the server answers.
        const certificate_files certs = certificate_paths(cert_dir);
        {
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) return true;
            server_.reset(new httplib::SSLServer(certs.server_cert.c_str(), certs.server_key.c_str()));
        }
        if (!server_->is_valid())
        {
            error = "the TLS configuration of the server failed";
            return false;
        }
        // The limits of the bodies are those of each route (server.cpp).
        server_->set_payload_max_length(0);
        server_->set_keep_alive_timeout(30);
        server_->set_read_timeout(120);
        server_->set_write_timeout(120);
        server_->new_task_queue = []() { return new httplib::ThreadPool(8, 64); };
        server_->set_socket_options([](socket_t sock) {
#if defined(_WIN32)
            // No other process may bind the port of the service.
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
        });
        install_routes(*server_, *this);
        if (!server_->bind_to_port("127.0.0.1", port_))
        {
            error = "the port " + std::to_string(port_) + " is in use or cannot be opened";
            return false;
        }
        {
            // A stop that came before the bind found no socket to close: the bound one is
            // closed here, and listen_after_bind() returns at once.
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) server_->stop();
        }
        log_info("server: https://localhost:" + std::to_string(port_) + "/");

        background_ = std::thread([this]() {
            try
            {
                background_loop();
            }
            catch (const std::exception& e)
            {
                log_error("service: the background tasks stopped: " + std::string(e.what()));
            }
            // A start cut short by an error ends the starting state all the same.
            engines_ready_ = true;
            models_ready_ = true;
        });
        server_->listen_after_bind();
        const bool stopped = stopping_;

        // Shutdown: the requests to the network end first, so that nothing waits for
        // them, then the background tasks, the downloads (their progress is kept), the
        // engine.
        stop();
        if (background_.joinable()) background_.join();
        std::unique_ptr<model_manager> models;
        std::unique_ptr<updater> updates;
        {
            std::lock_guard<std::mutex> lock(server_m_);
            server_.reset();
            models = std::move(models_);
            updates = std::move(updater_);
        }
        models.reset();
        engine_.stop();
        {
            // The unloadings end with the engine.
            std::vector<std::thread> retirers;
            {
                std::lock_guard<std::mutex> lock(retire_m_);
                retirers.swap(retirers_);
            }
            retire_cv_.notify_all();
            for (auto& t : retirers)
                if (t.joinable()) t.join();
        }
        updates.reset();
        log_info("service stopped");
        if (!stopped)
        {
            error = "the server stopped unexpectedly";
            return false;
        }
        return true;
    }

    void service::stop()
    {
        {
            // Under the lock of the waits, so that a wait about to start sees the flag.
            std::lock_guard<std::mutex> lock(m_);
            stopping_ = true;
        }
        cv_.notify_all();
        {
            std::lock_guard<std::mutex> lock(retire_m_);
        }
        retire_cv_.notify_all();
        std::lock_guard<std::mutex> lock(server_m_);
        // The downloads, the imports and the checks end at once; their progress is kept.
        http_cancel_all(stopping_reason);
        if (models_) models_->stop();
        if (updater_) updater_->cancel();
        if (server_) server_->stop();
    }

    void service::apply_network_settings(const host_settings& s)
    {
        set_http_proxy(s.proxy);
        {
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) return;
            if (s.offline) http_cancel_all(offline_reason);
            else http_restore();
        }
        // The background tasks start or end their network work at once.
        cv_.notify_all();
    }

    void service::request_restart()
    {
        if (restart_.exchange(true)) return;
        log_info("service: restart requested");
        // From a handler of the server: the response is written before the server stops.
        restarter_ = std::thread([this]() {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait_for(lock, std::chrono::milliseconds(300), [this]() { return stopping_.load(); });
            lock.unlock();
            stop();
        });
    }

    void service::start_engines()
    {
        // The status reports the start until the end of this function, whatever happens.
        struct ready_at_exit
        {
            std::atomic<bool>& flag;
            ~ready_at_exit() { flag = true; }
        } ready{ engines_ready_ };
        // The engine of this machine; a processor engine when the chosen one fails.
        const host_settings st = settings_->get();
        candidates_ = engine_.discover(engines_dir(), executable_path());
        choice_ = choose_engine(candidates_, st);
        if (stopping_) return;
        if (choice_.candidate)
        {
            std::string e;
            if (!engine_.start(*choice_.candidate, engine_config(st, choice_.device), e))
            {
                engine_error_ = choice_.key + ": " + e;
                log_error("engines: " + engine_error_);
                for (const auto& o : engine_options(candidates_))
                {
                    if (o.key.compare(0, 3, "cpu") != 0) continue;
                    if (engine_.start(*o.candidate, engine_config(st, o.device), e))
                    {
                        choice_ = o;
                        break;
                    }
                    log_error("engines: " + o.key + ": " + e);
                }
            }
        }
        else
        {
            engine_error_ = "no usable engine in " + engines_dir();
            log_error("engines: " + engine_error_);
        }
    }

    std::string service::starting_phase() const
    {
        if (!engines_ready_) return "engines";
        if (!models_ready_) return "models";
        return std::string();
    }

    bool service::model_loading(const std::string& requested)
    {
        const host_settings st = settings_->get();
        std::string id = requested;
        std::string role = requested;
        if (role.compare(0, 8, "albireo-") == 0) role = role.substr(8);
        if (requested.empty() || requested == "auto" || requested == "albireo") role = "general";
        if (is_role(role))
        {
            auto it = st.active.find(role);
            if (it == st.active.end() && role != embedding_role) it = st.active.find("general");
            if (it == st.active.end()) return false;
            id = it->second;
        }
        std::lock_guard<std::mutex> lock(m_);
        const auto state = load_state_.find(id);
        return state != load_state_.end() && state->second == "loading";
    }

    void service::background_loop()
    {
        /*
            The network tasks run on threads of their own and start first; the local work
            follows at once and never waits for them: the engines, while the server already
            answers, the models of the roles, the files of the import folder, the default
            model among those installed.
        */
        bool online = false;
        int64_t next_catalog = 0;
        int64_t next_update = 0;
        auto network = [&](const host_settings& st) {
            if (st.offline)
            {
                online = false;
                return;
            }
            const int64_t now = now_unix();
            if (!online)
            {
                // At the start, and back online: the interrupted downloads resume, the
                // catalog and the updates are checked at once.
                online = true;
                models_->resume_pending();
                next_catalog = now;
                next_update = now;
            }
            if (now >= next_catalog)
            {
                next_catalog = now + catalog_period;
                models_->refresh_catalog_async();
            }
            if (st.check_updates && now >= next_update)
            {
                next_update = now + update_period;
                updater_->check_async();
            }
        };
        network(settings_->get());
        start_engines();
        load_active_models();
        models_ready_ = true;
        if (stopping_) return;
        scan_inbox();
        unsigned catalog_seen = models_->catalog_generation();
        ensure_default_model();

        int64_t next_certificate = now_unix() + certificate_period;
        int64_t next_inbox = now_unix() + inbox_period;
        std::unique_lock<std::mutex> lock(m_);
        while (!stopping_)
        {
            cv_.wait_for(lock, std::chrono::seconds(5));
            if (stopping_) break;
            lock.unlock();
            const bool was_online = online;
            network(settings_->get());
            // A new catalog, downloaded or imported, may name the default model, which
            // can be downloaded once back online.
            if (models_->catalog_generation() != catalog_seen || (online && !was_online))
            {
                catalog_seen = models_->catalog_generation();
                ensure_default_model();
            }
            const int64_t now = now_unix();
            if (now >= next_inbox)
            {
                next_inbox = now + inbox_period;
                scan_inbox();
            }
            if (now >= next_certificate)
            {
                next_certificate = now + certificate_period;
                const std::string cert_dir = join_path(data_dir_, "certificates");
                const std::string before = certificate_thumbprint(certificate_paths(cert_dir).server_cert);
                std::string error;
                if (ensure_certificates(cert_dir, error)
                    && certificate_thumbprint(certificate_paths(cert_dir).server_cert) != before)
                {
                    log_info("certificates: the server certificate was renewed, restart");
                    request_restart();
                }
            }
            lock.lock();
        }
    }

    void service::scan_inbox()
    {
        /*
            What is left in the import folder of the data folder, by the installer from its
            medium or by an administrator, with the layout of an offline bundle: a signed
            catalog, models (also in a "models" subfolder), a signed update with its
            installer.  What is installed leaves the folder; a file that cannot be installed
            is renamed with the suffix ".rejected", so that it is read only once.
        */
        const std::string inbox = join_path(data_dir_, "import");
        if (!dir_exists(inbox)) return;
        std::lock_guard<std::mutex> serial(import_m_);
        const std::string sub = join_path(inbox, "models");
        // A copy in progress: the folder waits for the next pass.
        for (const std::string& dir : { inbox, sub })
            for (const auto& n : list_dir(dir))
                if (file_being_written(join_path(dir, n))) return;
        auto set_aside = [](const std::string& f) {
            if (file_exists(f)) rename_file(f, f + ".rejected");
        };

        const std::string catalog = join_path(inbox, "catalog.json");
        if (file_exists(catalog) && file_exists(catalog + ".sig"))
        {
            const json r = models_->import_catalog(inbox);
            if (r.contains("error"))
            {
                log_warning("models: import folder: " + r["error"].as_string());
                set_aside(catalog);
                set_aside(catalog + ".sig");
            }
            else
            {
                remove_file(catalog);
                remove_file(catalog + ".sig");
            }
        }

        for (const std::string& dir : { inbox, sub })
            for (const auto& n : list_dir(dir))
            {
                if (stopping_) return;
                if (!ends_with(n, ".gguf")) continue;
                const std::string f = join_path(dir, n);
                const std::string side = f.substr(0, f.size() - 5) + ".json";
                const json r = models_->import_path(f, true);
                for (const auto& e : r["errors"].elements()) log_warning("models: import folder: " + e.as_string());
                if (!r["imported"].empty()) remove_file(side);
                else if (!stopping_)
                {
                    set_aside(f);
                    set_aside(side);
                }
            }

        const std::string feed = join_path(inbox, "feed.json");
        if (file_exists(feed) && file_exists(feed + ".sig") && !stopping_)
        {
            const json r = updater_->import_offline(inbox);
            const std::string installer = r["installer"].as_string();
            const std::string source = installer.empty() ? std::string() : join_path(inbox, installer);
            // The installer the feed names may still be on its way.
            if (r.contains("error") && !source.empty() && !file_exists(source)) return;
            if (r.contains("error"))
            {
                log_warning("updates: import folder: " + r["error"].as_string());
                set_aside(feed);
                set_aside(feed + ".sig");
                if (!source.empty()) set_aside(source);
            }
            else
            {
                remove_file(feed);
                remove_file(feed + ".sig");
                if (!source.empty()) remove_file(source);
            }
        }
    }

    json service::import_folder(const std::string& path)
    {
        std::lock_guard<std::mutex> serial(import_m_);
        json result = json::object();
        const bool folder = dir_exists(path);
        if (folder && file_exists(join_path(path, "catalog.json")))
            result["catalog"] = models_->import_catalog(path);
        const json r = models_->import_path(path);
        result["imported"] = r["imported"];
        result["errors"] = r["errors"];
        if (folder && file_exists(join_path(path, "feed.json")))
            result["update"] = updater_->import_offline(path);
        log_info("import: " + path + ", " + std::to_string(r["imported"].size()) + " models");
        return result;
    }

    bool service::is_embedding(const installed_model& m)
    {
        return m.descriptor["embedding"].as_bool(false) || has_role(m.descriptor, embedding_role);
    }

    void service::register_builtin_models()
    {
        std::vector<std::string> removed;
        const std::vector<installed_model> added =
            models_->register_builtin(join_path(program_dir_, builtin_models_folder), removed);
        if (!removed.empty())
        {
            settings_->update([&](host_settings& s) {
                for (auto it = s.active.begin(); it != s.active.end();)
                {
                    if (std::find(removed.begin(), removed.end(), it->second) != removed.end()) it = s.active.erase(it);
                    else ++it;
                }
            });
        }
        // The engine is not started yet: the functions are given now, the models load with
        // the others.
        for (const auto& m : added) on_installed(m);
    }

    void service::note_default_failure(const json& entry, const user_error& error, const std::string& what)
    {
        log_warning("models: " + what + " cannot be downloaded: " + error.text);
        models_->note_failure(entry["id"].as_string(), error);
    }

    void service::ensure_default_model()
    {
        const host_settings st = settings_->get();
        const std::vector<installed_model> installed = db_->models();
        const json catalog = models_->catalog();

        // The document search always has its embedding model: an installed one, else the
        // default one of the catalog, downloaded whatever the options of the installation
        // (the computers without network receive it in their offline bundle).
        if (!st.active.count(embedding_role))
        {
            const installed_model* pick = nullptr;
            for (const auto& m : installed)
                if (is_embedding(m) && (!pick || m.descriptor["default"].as_bool(false))) pick = &m;
            user_error error;
            if (pick)
            {
                if (!activate(pick->id, embedding_role, error)) log_warning("models: " + pick->id + ": " + error.text);
            }
            else if (!st.offline)
            {
                for (const auto& e : catalog["models"].elements())
                {
                    if (!has_role(e, embedding_role) || !e["default"].as_bool(false)) continue;
                    if (models_->start_download(e["id"].as_string(), error))
                        log_info("models: download of the embedding model " + e["id"].as_string());
                    else
                        note_default_failure(e, error, "the embedding model");
                    break;
                }
            }
        }

        if (st.active.count("general")) return;
        const installed_model* pick = nullptr;
        for (const auto& m : installed)
        {
            // The model of the catalog meant for general use, else the first language model.
            if (is_embedding(m)) continue;
            if (!pick || has_role(m.descriptor, "general") || m.descriptor["default"].as_bool(false)) pick = &m;
        }
        if (pick)
        {
            user_error error;
            if (!activate(pick->id, "general", error)) log_warning("models: " + pick->id + ": " + error.text);
            return;
        }
        // No language model installed: the default model of the catalog is downloaded, once,
        // unless the installation asked otherwise or the computer is offline.
        if (!st.default_model || st.offline || db_->value("default_model_requested") == "1") return;
        for (const auto& e : catalog["models"].elements())
        {
            if (!e["default"].as_bool(false) || has_role(e, embedding_role)) continue;
            user_error error;
            if (models_->start_download(e["id"].as_string(), error))
            {
                db_->set_value("default_model_requested", "1");
                log_info("models: download of the default model " + e["id"].as_string());
            }
            else
            {
                note_default_failure(e, error, "the default model");
            }
            return;
        }
    }

    void service::on_installed(const installed_model& m)
    {
        if (stopping_) return;
        if (is_embedding(m))
        {
            // The first embedding model serves the document search.
            user_error error;
            if (!settings_->get().active.count(embedding_role) && !activate(m.id, embedding_role, error))
                log_warning("models: " + m.id + ": " + error.text);
            return;
        }
        // A new model takes the functions its catalog entry names and no model holds yet,
        // and the general one when it is free, so that every function has a model.
        std::vector<std::string> roles;
        for (const auto& r : m.descriptor["roles"].elements())
        {
            const std::string role = r.as_string();
            if (role == "general" || role == "code" || role == "security") roles.push_back(role);
        }
        roles.push_back("general");
        for (const auto& role : roles)
        {
            if (settings_->get().active.count(role)) continue;
            user_error error;
            if (!activate(m.id, role, error)) log_warning("models: " + m.id + ": " + error.text);
        }
    }

    bool service::load_model(const installed_model& m, std::string& error)
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            load_state_[m.id] = "loading";
        }
        const host_settings st = settings_->get();
        json options = json::object();
        // The settings of the conversations apply to the language models; an embedding
        // model keeps the defaults of the engine for its short inputs.
        if (!is_embedding(m))
        {
            if (st.context > 0) options["context"] = st.context;
            options["max_sequences"] = st.max_sequences;
            if (st.kv_mb > 0) options["kv_mb"] = st.kv_mb;
        }
        const bool ok = engine_.load_model(m.id, m.file, options, error);
        std::lock_guard<std::mutex> lock(m_);
        load_state_[m.id] = ok ? std::string("ready") : error;
        if (ok) log_info("models: " + m.id + " loaded");
        else log_error("models: " + m.id + " not loaded: " + error);
        return ok;
    }

    void service::load_active_models()
    {
        std::lock_guard<std::mutex> serial(models_m_);
        if (!engine_.running()) return;
        const host_settings st = settings_->get();
        std::vector<std::string> done;
        for (const auto& kv : st.active)
        {
            if (stopping_) return;
            if (std::find(done.begin(), done.end(), kv.second) != done.end()) continue;
            done.push_back(kv.second);
            installed_model m;
            if (!db_->find_model(kv.second, m) || !file_exists(m.file))
            {
                log_warning("models: the active model " + kv.second + " is not installed");
                std::lock_guard<std::mutex> lock(m_);
                load_state_[kv.second] = "not installed";
                continue;
            }
            std::string error;
            load_model(m, error);
        }
    }

    bool service::activate(const std::string& id, const std::string& role, user_error& error)
    {
        if (!is_role(role))
        {
            error.set("unknown role '" + role + "' (general, code, security or embedding)", "unknown_role")
                .arg("role", role);
            return false;
        }
        std::lock_guard<std::mutex> serial(models_m_);
        installed_model m;
        if (!db_->find_model(id, m) || !file_exists(m.file))
        {
            error.set("the model " + id + " is not installed", "not_installed").arg("m", id);
            return false;
        }
        const std::string name = m.name.empty() ? id : m.name;
        {
            std::lock_guard<std::mutex> lock(retire_m_);
            if (removing_.count(id) != 0)
            {
                error.set("the model " + id + " is being deleted", "being_deleted").arg("m", name);
                return false;
            }
        }
        if ((role == embedding_role) != is_embedding(m))
        {
            if (role == embedding_role)
                error.set("the model " + id + " writes text; the document search needs an embedding model",
                    "needs_embedding_model").arg("m", name);
            else
                error.set("the model " + id + " computes the vectors of the document search and answers no "
                    "conversation", "embedding_model_only").arg("m", name);
            return false;
        }
        // A model waiting for its requests before it leaves stays.
        if (engine_.keep_model(id)) log_info("models: " + id + " stays loaded");
        wait_retired(id);
        {
            const host_settings st = settings_->get();
            const auto current = st.active.find(role);
            if (current != st.active.end() && current->second == id) return true;
        }
        const std::vector<std::string> loaded = engine_.loaded_models();
        std::string load_error;
        if (engine_.running() && std::find(loaded.begin(), loaded.end(), id) == loaded.end()
            && !load_model(m, load_error))
        {
            error.set(load_error, "load_failed").arg("m", name).arg("e", load_error);
            return false;
        }
        // The settings may have changed while the model loaded: the function is set on
        // the settings as they are now.
        std::string previous;
        const host_settings st = settings_->update([&](host_settings& s) {
            const auto current = s.active.find(role);
            if (current != s.active.end()) previous = current->second;
            s.active[role] = id;
        });
        log_info("models: " + id + " active for " + role);
        // The model the role used before is released when no other role uses it.
        if (!previous.empty() && previous != id)
        {
            bool used = false;
            for (const auto& kv : st.active) used = used || kv.second == previous;
            if (!used) retire(previous);
        }
        return true;
    }

    void service::retire(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(retire_m_);
        if (stopping_ || !retiring_.insert(id).second) return;
        {
            std::lock_guard<std::mutex> state(m_);
            load_state_[id] = removing_.count(id) != 0 ? "removing" : "unloading";
        }
        // No new request goes to the model; the thread waits for those it has.
        engine_.begin_unload(id);
        retirers_.emplace_back([this, id]() {
            bool unloaded = engine_.finish_unload(id);
            if (unloaded) log_info("models: " + id + " unloaded");
            for (;;)
            {
                bool remove = false;
                {
                    // The last check and the end of the retirement in one step, so that a
                    // deletion asked meanwhile is either seen here or done by remove_model.
                    std::lock_guard<std::mutex> l(retire_m_);
                    if (removing_.count(id) == 0)
                    {
                        std::lock_guard<std::mutex> state(m_);
                        if (unloaded) load_state_.erase(id);
                        else load_state_[id] = "ready";
                        retiring_.erase(id);
                        break;
                    }
                    remove = unloaded;
                    if (remove) removing_.erase(id);
                }
                if (!remove)
                {
                    // Kept by an activation, then deleted: it leaves after all.
                    engine_.begin_unload(id);
                    unloaded = engine_.finish_unload(id);
                    if (unloaded) log_info("models: " + id + " unloaded");
                    continue;
                }
                std::string error;
                std::lock_guard<std::mutex> server(server_m_);
                if (models_ && !models_->remove(id, error)) log_warning("models: " + error);
            }
            retire_cv_.notify_all();
        });
    }

    json service::leaving_models()
    {
        json o = json::object();
        std::lock_guard<std::mutex> state(m_);
        for (const auto& kv : load_state_)
            if (kv.second == "unloading" || kv.second == "removing") o[kv.first] = kv.second;
        return o;
    }

    void service::wait_retired(const std::string& id)
    {
        std::unique_lock<std::mutex> lock(retire_m_);
        retire_cv_.wait(lock, [&]() { return stopping_.load() || retiring_.count(id) == 0; });
    }

    bool service::deactivate(const std::string& id, user_error& error)
    {
        std::lock_guard<std::mutex> serial(models_m_);
        bool refused = false;
        bool changed = false;
        settings_->update([&](host_settings& st) {
            const auto embedding = st.active.find(embedding_role);
            if (embedding != st.active.end() && embedding->second == id)
            {
                refused = true;
                return;
            }
            for (auto it = st.active.begin(); it != st.active.end();)
            {
                if (it->second == id)
                {
                    it = st.active.erase(it);
                    changed = true;
                }
                else
                {
                    ++it;
                }
            }
        });
        if (refused)
        {
            installed_model m;
            const std::string name = db_->find_model(id, m) && !m.name.empty() ? m.name : id;
            error.set("the document search needs an embedding model: activate another one to replace " + id,
                "embedding_needed").arg("m", name);
            return false;
        }
        if (changed) log_info("models: " + id + " deactivated");
        retire(id);
        return true;
    }

    bool service::remove_model(const std::string& id, user_error& error, bool& pending)
    {
        std::lock_guard<std::mutex> serial(models_m_);
        pending = false;
        installed_model m;
        if (!db_->find_model(id, m))
        {
            error.set("the model '" + id + "' is not installed", "not_installed").arg("m", id);
            return false;
        }
        const std::string name = m.name.empty() ? id : m.name;
        if (m.source == "builtin")
        {
            error.set("the model " + id + " comes with Albireo: it can leave its functions to other models, it "
                "cannot be deleted", "builtin_model").arg("m", name);
            return false;
        }
        bool changed = false;
        settings_->update([&](host_settings& st) {
            for (auto it = st.active.begin(); it != st.active.end();)
            {
                if (it->second == id)
                {
                    it = st.active.erase(it);
                    changed = true;
                }
                else
                {
                    ++it;
                }
            }
        });
        if (changed) log_info("models: " + id + " deactivated");
        const std::vector<std::string> loaded = engine_.loaded_models();
        const bool in_engine = std::find(loaded.begin(), loaded.end(), id) != loaded.end();
        {
            std::lock_guard<std::mutex> lock(retire_m_);
            if (in_engine || retiring_.count(id) != 0)
            {
                // Deleted by its retirement, once its requests are served.
                removing_.insert(id);
                pending = true;
            }
        }
        if (!pending)
        {
            std::string why;
            if (models_->remove(id, why)) return true;
            error.set(why, "delete_failed").arg("m", name).arg("file", m.file);
            return false;
        }
        retire(id);
        std::lock_guard<std::mutex> state(m_);
        load_state_[id] = "removing";
        log_info("models: " + id + " deleted once its requests are served");
        return true;
    }

    std::string service::resolve_model(const std::string& requested)
    {
        const host_settings st = settings_->get();
        const std::vector<std::string> loaded = engine_.loaded_models();
        auto is_loaded = [&](const std::string& id) {
            return std::find(loaded.begin(), loaded.end(), id) != loaded.end();
        };
        std::string role = requested;
        if (role.compare(0, 8, "albireo-") == 0) role = role.substr(8);
        if (requested.empty() || requested == "auto" || requested == "albireo") role = "general";
        const auto embedding = st.active.find(embedding_role);
        const std::string embedder = embedding == st.active.end() ? std::string() : embedding->second;
        if (role == embedding_role || requested == embedder) return std::string();
        if (is_role(role))
        {
            auto it = st.active.find(role);
            if (it != st.active.end() && is_loaded(it->second)) return it->second;
            // A role without its own model is served by the general model, else by the
            // first language model loaded.
            it = st.active.find("general");
            if (it != st.active.end() && is_loaded(it->second)) return it->second;
            for (const auto& id : loaded)
                if (id != embedder) return id;
            return std::string();
        }
        return is_loaded(requested) ? requested : std::string();
    }

    std::string service::resolve_embedding_model(const std::string& requested)
    {
        const host_settings st = settings_->get();
        const std::vector<std::string> loaded = engine_.loaded_models();
        auto is_loaded = [&](const std::string& id) {
            return std::find(loaded.begin(), loaded.end(), id) != loaded.end();
        };
        if (!requested.empty() && requested != "auto" && requested != embedding_role
            && requested != std::string("albireo-") + embedding_role)
        {
            installed_model m;
            return is_loaded(requested) && db_->find_model(requested, m) && is_embedding(m) ? requested : std::string();
        }
        const auto it = st.active.find(embedding_role);
        return it != st.active.end() && is_loaded(it->second) ? it->second : std::string();
    }

    void service::record_usage(long prompt_tokens, long generated_tokens)
    {
        usage_prompt_ += prompt_tokens;
        usage_generated_ += generated_tokens;
        ++usage_requests_;
        try
        {
            db_->add_usage(prompt_tokens, generated_tokens, 1);
        }
        catch (const std::exception& e)
        {
            log_warning(std::string("usage: ") + e.what());
        }
    }

    json service::status()
    {
        const host_settings st = settings_->get();
        json s = json::object();
        s["product"] = ALBIREO_PRODUCT_NAME;
        s["edition"] = ALBIREO_EDITION;
        s["version"] = ALBIREO_VERSION;
        s["company"] = ALBIREO_COMPANY_NAME;
        s["author"] = ALBIREO_AUTHOR_NAME;
        s["copyright"] = ALBIREO_COPYRIGHT;
        s["project"] = ALBIREO_PROJECT_URL;
        s["started_at"] = static_cast<long long>(started_at_);
        s["uptime"] = static_cast<long long>(now_unix() - started_at_);
        const std::string phase = starting_phase();
        s["state"] = phase.empty() ? "ready" : "starting";
        if (!phase.empty()) s["starting"] = phase;
        s["port"] = port_;
        s["url"] = "https://localhost:" + std::to_string(port_) + "/";
        s["data_dir"] = data_dir_;
        s["models_dir"] = st.effective_models_dir();
        s["models_dir_free"] = static_cast<unsigned long long>(free_disk_space(st.effective_models_dir()));

        json host = json::object();
        const cpu_info cpu = read_cpu_info();
        const memory_info mem = read_memory_info();
        host["name"] = host_name();
        host["os"] = operating_system_name();
        json c = json::object();
        c["name"] = cpu.brand.empty() ? cpu.vendor : cpu.brand;
        c["cores"] = cpu.physical_cores;
        c["threads"] = cpu.logical_cores;
        json features = json::array();
        if (cpu.avx2) features.push_back("AVX2");
        if (cpu.fma) features.push_back("FMA");
        if (cpu.f16c) features.push_back("F16C");
        if (cpu.avx_vnni) features.push_back("AVX-VNNI");
        if (cpu.avx512f) features.push_back("AVX-512");
        if (cpu.avx512vnni) features.push_back("AVX512-VNNI");
        if (cpu.amx) features.push_back("AMX");
        c["features"] = features;
        host["cpu"] = c;
        json m = json::object();
        m["total"] = static_cast<unsigned long long>(mem.total);
        m["available"] = static_cast<unsigned long long>(mem.available);
        host["memory"] = m;
        s["host"] = host;

        json e = json::object();
        e["running"] = engine_.running();
        e["variant"] = engine_.variant();
        const bool chosen = engines_ready_;
        e["key"] = chosen ? choice_.key : std::string();
        e["device"] = chosen ? choice_.device : std::string();
        e["error"] = chosen ? engine_error_ : std::string();
        e["description"] = engine_.description();
        json query = json::object();
        query["op"] = "stats";
        e["stats"] = engine_.running() ? engine_.query(query) : json();
        s["engine"] = e;
        query["op"] = "models";
        s["loaded_models"] = engine_.running() ? engine_.query(query)["models"] : json::array();

        json active = json::object();
        for (const auto& kv : st.active) active[kv.first] = kv.second;
        s["active"] = active;
        json loading = json::object();
        {
            std::lock_guard<std::mutex> lock(m_);
            for (const auto& kv : load_state_) loading[kv.first] = kv.second;
        }
        s["models_state"] = loading;
        json usage = json::object();
        usage["requests"] = usage_requests_.load();
        usage["prompt_tokens"] = usage_prompt_.load();
        usage["completion_tokens"] = usage_generated_.load();
        s["usage"] = usage;
        s["downloads"] = models_->downloads();
        s["updates"] = updater_->status();
        s["catalog"] = models_->catalog_status();
        s["offline"] = st.offline;
        json settings = json::object();
        settings["engine"] = st.engine;
        settings["accelerator"] = st.accelerator;
        settings["v2ram"] = st.v2ram;
        settings["v2ram_max_mb"] = st.v2ram_max_mb;
        settings["device_mb"] = st.device_mb;
        settings["language"] = st.language;
        settings["check_updates"] = st.check_updates;
        settings["offline"] = st.offline;
        s["settings"] = settings;
        return s;
    }

    json service::engines()
    {
        const host_settings st = settings_->get();
        if (!engines_ready_)
        {
            // The engines are being described: nothing to list yet.
            json out = json::object();
            out["starting"] = true;
            out["engines"] = json::array();
            out["options"] = json::array();
            out["current"] = "";
            out["setting"] = st.engine;
            out["accelerator"] = st.accelerator;
            out["accelerators_available"] = false;
            return out;
        }
        json list = json::array();
        for (const auto& c : candidates_)
        {
            json j = json::object();
            j["variant"] = c.variant;
            j["path"] = c.path;
            j["usable"] = c.usable;
            j["version"] = c.description["version"];
            j["build"] = c.description["build"];
            j["devices"] = c.description["devices"];
            if (!c.usable)
            {
                j["reason"] = c.description["reason"];
                if (c.description.contains("reason_code"))
                {
                    j["reason_code"] = c.description["reason_code"];
                    j["reason_args"] = c.description["reason_args"];
                }
            }
            j["current"] = choice_.candidate && choice_.candidate->path == c.path;
            list.push_back(j);
        }
        json options = json::array();
        for (const auto& o : engine_options(candidates_))
        {
            json j = json::object();
            j["key"] = o.key;
            j["variant"] = o.candidate->variant;
            j["device"] = o.device;
            j["current"] = o.key == choice_.key;
            j["measure"] = st.engine_benchmarks[o.key];
            options.push_back(j);
        }
        json out = json::object();
        out["engines"] = list;
        out["options"] = options;
        out["current"] = choice_.key;
        out["setting"] = st.engine;
        out["accelerator"] = st.accelerator;
        // The tray offers the choice of the accelerator while a processor engine runs.
        bool accelerators = false;
        for (const auto& o : engine_options(candidates_))
            accelerators = accelerators || o.key == "cuda" || o.key == "openvino:npu" || o.key == "openvino:gpu";
        out["accelerators_available"] = accelerators;
        if (!accelerators)
        {
            // Why none is offered: no engine of a device in the installation, or none that
            // runs on this computer (the list of the engines gives the reasons).
            bool installed = false;
            for (const auto& c : candidates_)
                installed = installed || c.variant == "cuda" || is_openvino_variant(c.variant);
            out["accelerators_reason"] = installed ? "unusable" : "not_installed";
        }
        return out;
    }

    json service::benchmark_engines()
    {
        // The model of the measure: the general model, else the smallest one installed.
        const host_settings st = settings_->get();
        installed_model model;
        bool found = st.active.count("general") && db_->find_model(st.active.at("general"), model);
        if (!found)
        {
            for (const auto& m : db_->models())
                if (!found || m.size < model.size)
                {
                    model = m;
                    found = true;
                }
        }
        json out = json::object();
        if (!engines_ready_)
        {
            out["error"] = "the service is starting";
            out["code"] = "starting";
            return out;
        }
        if (!found)
        {
            out["error"] = "no model is installed";
            return out;
        }
        json results = json::object();
        for (const auto& o : engine_options(candidates_))
        {
            if (stopping_) break;
            json config = engine_config(st, o.device);
            config["context"] = 2048;
            // The graphs of a device compile before the products they serve, so that the
            // measure reflects the device.
            config["benchmark"] = true;
            std::string output;
            log_info("engines: measuring " + o.key + " with " + model.id);
            const std::vector<std::string> args = { "--bench-engine", o.candidate->path, model.file, config.dump() };
            const int code = run_process(executable_path(), args, &output, 600000);
            const size_t start = output.rfind("\n{");
            const json r = json::parse_or_null(start == std::string::npos ? output : output.substr(start + 1));
            if (code == 0 && r["tokens_per_second"].as_double(0) > 0)
            {
                json m = r;
                m["model"] = model.id;
                m["measured_at"] = static_cast<long long>(now_unix());
                results[o.key] = m;
                log_info("engines: " + o.key + " " + std::to_string(r["tokens_per_second"].as_double()) + " tokens/s");
            }
            else
            {
                json m = json::object();
                const std::string failed = "the measure failed (code " + std::to_string(code) + ")";
                m["error"] = r["error"].is_string() ? r["error"] : json(failed);
                results[o.key] = m;
                log_warning("engines: measure of " + o.key + " failed");
            }
        }
        // The measures join the settings as they are now, which may have changed meanwhile.
        const host_settings updated = settings_->update([&](host_settings& s) {
            for (const auto& kv : results.items())
                if (!kv.second.contains("error")) s.engine_benchmarks[kv.first] = kv.second;
        });
        out["model"] = model.id;
        out["results"] = results;
        // The engine "best", or an accelerator choice among several devices, follows the
        // new measures after a restart.
        const engine_choice next = choose_engine(candidates_, updated);
        out["restart"] = next.key != choice_.key;
        if (next.key != choice_.key) request_restart();
        return out;
    }

    int service::probe_engine(const std::string& path)
    {
        engine_library lib;
        std::string error;
        json d;
        if (!lib.load(path, error))
        {
            d = json::object();
            d["usable"] = false;
            d["reason"] = error;
            std::cout << d.dump() << std::endl;
            return 2;
        }
        d = lib.description();
        if (!d.is_object())
        {
            d = json::object();
            d["usable"] = false;
            d["reason"] = "the library gives no description";
        }
        std::cout << d.dump() << std::endl;
        return 0;
    }

    int service::benchmark_engine(const std::string& path, const std::string& model, const std::string& config)
    {
        auto fail = [](const std::string& message) {
            json r = json::object();
            r["error"] = message;
            std::cout << r.dump() << std::endl;
            return 2;
        };
        engine_library lib;
        std::string error;
        if (!lib.load(path, error)) return fail(error);
        char err[2048] = { 0 };
        albireo_engine* e = lib.create(config.c_str(), nullptr, nullptr, err, sizeof(err));
        if (!e) return fail(err);
        json options = json::object();
        options["context"] = 2048;
        options["max_sequences"] = 1;
        if (lib.load_model(e, "bench", model.c_str(), options.dump().c_str(), err, sizeof(err)) != ALBIREO_OK)
        {
            lib.destroy(e);
            return fail(err);
        }

        struct waiter
        {
            std::mutex m;
            std::condition_variable cv;
            bool done = false;
            std::string result;
            static void ALBIREO_ENGINE_CALL on_event(void* user, int kind, const char* data, size_t size)
            {
                waiter* w = static_cast<waiter*>(user);
                if (kind != ALBIREO_EVENT_DONE && kind != ALBIREO_EVENT_ERROR) return;
                std::lock_guard<std::mutex> lock(w->m);
                w->done = true;
                w->result.assign(data, size);
                w->cv.notify_all();
            }
        };
        auto run = [&](int tokens, json& done) {
            json req = json::object();
            req["model"] = "bench";
            json messages = json::array();
            json msg = json::object();
            msg["role"] = "user";
            msg["content"] = "Write a detailed explanation of how a refracting telescope forms an image, "
                "covering the objective lens, the focal length, the eyepiece and the magnification.";
            messages.push_back(msg);
            req["messages"] = messages;
            req["max_tokens"] = tokens;
            req["temperature"] = 0;
            req["reasoning"] = false;
            json stop_none = json::array();
            req["stop"] = stop_none;
            waiter w;
            uint64_t id = 0;
            if (lib.submit(e, req.dump().c_str(), &waiter::on_event, &w, &id, err, sizeof(err)) != ALBIREO_OK)
                return false;
            std::unique_lock<std::mutex> lock(w.m);
            w.cv.wait(lock, [&]() { return w.done; });
            done = json::parse_or_null(w.result);
            return done["usage"].is_object();
        };
        json warm, done;
        // A first short request warms the caches and lets a device compile its graphs.
        const bool ok = run(8, warm) && run(64, done);
        lib.unload_model(e, "bench");
        lib.destroy(e);
        if (!ok) return fail(done["message"].str("the measure did not complete"));
        const json& t = done["timings"];
        json r = json::object();
        r["tokens_per_second"] = t["generation_per_second"];
        r["prompt_tokens_per_second"] = t["prompt_per_second"];
        r["generated"] = done["usage"]["completion_tokens"];
        std::cout << r.dump() << std::endl;
        return 0;
    }

}
