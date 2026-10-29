// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "engine_host.h"

#include <algorithm>
#include <atomic>

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    namespace
    {
        struct sink_holder
        {
            request_sink sink;
        };

        const char* const variant_order[] = { "cuda", "openvino", "cpu-avx2", "cpu" };

        std::atomic<bool> pinned_started{false};
    }

    bool engine_host::pinned_engine_started()
    {
        return pinned_started.load();
    }

    engine_host::engine_host() {}

    engine_host::~engine_host()
    {
        stop();
    }

    std::vector<engine_candidate> engine_host::discover(const std::string& dir, const std::string& probe_program)
    {
        std::vector<engine_candidate> out;
        for (const char* v : variant_order)
        {
            const std::string path = join_path(dir, engine_file_name(v));
            if (!file_exists(path)) continue;
            engine_candidate c;
            c.variant = v;
            c.path = path;
            std::string output;
            const int code = run_process(probe_program, { "--probe-engine", path }, &output, 60000);
            // The description is the last line the probe prints.
            const size_t start = output.find('{');
            c.description = start == std::string::npos ? json() : json::parse_or_null(output.substr(start));
            c.usable = code == 0 && c.description["usable"].as_bool(false);
            if (!c.usable && !c.description.contains("reason"))
            {
                if (!c.description.is_object()) c.description = json::object();
                c.description["reason"] = code == 0 ? std::string("not usable")
                    : "the library could not be loaded (code " + std::to_string(code) + ")";
            }
            const std::string state = c.usable ? std::string("usable")
                : "not usable: " + c.description["reason"].as_string();
            log_info("engines: " + c.variant + " " + state);
            out.push_back(c);
        }
        return out;
    }

    const engine_candidate* engine_host::choose(const std::vector<engine_candidate>& candidates,
        const std::string& wanted, const json& benchmarks)
    {
        if (wanted != "auto" && wanted != "best")
        {
            for (const auto& c : candidates)
                if (c.variant == wanted && c.usable) return &c;
            log_warning("engines: the requested engine '" + wanted + "' is not usable, automatic selection");
        }
        // The fastest measured on this machine.
        const engine_candidate* best = nullptr;
        double best_rate = 0;
        for (const auto& c : candidates)
        {
            const double rate = benchmarks[c.variant]["tokens_per_second"].as_double(0);
            if (c.usable && rate > best_rate)
            {
                best = &c;
                best_rate = rate;
            }
        }
        if (best) return best;
        for (const char* v : variant_order)
            for (const auto& c : candidates)
                if (c.variant == v && c.usable) return &c;
        return nullptr;
    }

    bool engine_host::start(const engine_candidate& c, const json& config, std::string& error)
    {
        stop();
        std::lock_guard<std::mutex> lock(m_);
        std::unique_ptr<engine_library> lib(new engine_library());
        if (!lib->load(c.path, error)) return false;
        if (c.description["pinned"].as_bool(false)) pinned_started = true;
        char err[2048] = { 0 };
        albireo_engine* e = lib->create(config.dump().c_str(), &engine_host::on_log, nullptr, err, sizeof(err));
        if (!e)
        {
            error = err;
            return false;
        }
        lib_ = std::move(lib);
        engine_ = e;
        current_ = c;
        current_.description = lib_->description();
        log_info("engines: " + c.variant + " started");
        return true;
    }

    void engine_host::stop()
    {
        std::lock_guard<std::mutex> lock(m_);
        if (engine_ && lib_) lib_->destroy(engine_);
        engine_ = nullptr;
        lib_.reset();
        models_.clear();
    }

    bool engine_host::running() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return engine_ != nullptr;
    }

    std::string engine_host::variant() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return engine_ ? current_.variant : std::string();
    }

    json engine_host::description() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return current_.description;
    }

    bool engine_host::load_model(const std::string& id, const std::string& path, const json& options,
        std::string& error)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (!engine_)
        {
            error = "no engine is running";
            return false;
        }
        char err[2048] = { 0 };
        const int rc = lib_->load_model(engine_, id.c_str(), path.c_str(), options.dump().c_str(), err, sizeof(err));
        if (rc != ALBIREO_OK)
        {
            error = err;
            return false;
        }
        if (std::find(models_.begin(), models_.end(), id) == models_.end()) models_.push_back(id);
        return true;
    }

    void engine_host::unload_model(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (!engine_) return;
        lib_->unload_model(engine_, id.c_str());
        models_.erase(std::remove(models_.begin(), models_.end(), id), models_.end());
    }

    std::vector<std::string> engine_host::loaded_models() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return models_;
    }

    void ALBIREO_ENGINE_CALL engine_host::on_event(void* user, int kind, const char* data, size_t size)
    {
        sink_holder* h = static_cast<sink_holder*>(user);
        h->sink(kind, std::string(data, size));
        if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR) delete h;
    }

    void ALBIREO_ENGINE_CALL engine_host::on_log(void*, int level, const char* message)
    {
        const std::string m = std::string("engine: ") + message;
        if (level >= ALBIREO_LOG_ERROR) log_error(m);
        else if (level == ALBIREO_LOG_WARNING) log_warning(m);
        else if (level == ALBIREO_LOG_INFO) log_info(m);
        else log_debug(m);
    }

    bool engine_host::submit(const json& request, request_sink sink, uint64_t& id, std::string& error)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (!engine_)
        {
            error = "no engine is running";
            return false;
        }
        sink_holder* h = new sink_holder{ std::move(sink) };
        char err[2048] = { 0 };
        const int rc = lib_->submit(engine_, request.dump().c_str(), &engine_host::on_event, h, &id, err, sizeof(err));
        if (rc != ALBIREO_OK)
        {
            delete h;
            error = err;
            return false;
        }
        return true;
    }

    void engine_host::cancel(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (engine_) lib_->cancel(engine_, id);
    }

    json engine_host::query(const json& request)
    {
        std::lock_guard<std::mutex> lock(m_);
        json answer;
        if (!engine_)
        {
            answer = json::object();
            answer["error"] = "no engine is running";
            return answer;
        }
        lib_->run_query(engine_, request, answer);
        return answer;
    }

}
