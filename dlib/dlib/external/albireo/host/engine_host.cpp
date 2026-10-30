// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "engine_host.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "../common/log.h"
#include "../common/platform.h"
#include "../common/system_info.h"
#include "../common/version.h"

namespace albireo
{

    namespace
    {
        struct sink_holder
        {
            request_sink sink;
            engine_host* host;
            std::string model;
        };

        const char* const variant_order[] = { "cuda", "openvino-avx512", "openvino-avxvnni", "openvino", "cpu-avx512",
            "cpu-avxvnni", "cpu-avx2", "cpu" };

        bool ends_with(const std::string& s, const std::string& tail)
        {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }

        // The instruction sets the kernels of a variant use (CMakeLists.txt): the engines of
        // the devices are built for AVX2, as "cpu-avx2".
        unsigned variant_isa(const std::string& variant)
        {
            if (variant == "cpu") return 0;
            if (ends_with(variant, "-avx512")) return qk_isa_avx2 | qk_isa_avx512;
            if (ends_with(variant, "-avxvnni")) return qk_isa_avx2 | qk_isa_avx_vnni;
            return qk_isa_avx2;
        }

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
            out.push_back(c);
        }
        // A library whose kernels the processor cannot run is described without being
        // loaded: the runtimes of the devices take seconds to start, which a slow computer
        // feels at each start of the service.
        const cpu_info cpu = read_cpu_info();
        for (auto& c : out)
        {
            const unsigned lacks = qk_lacking_isa(cpu, variant_isa(c.variant));
            if (!lacks) continue;
            json dev = json::object();
            dev["type"] = "cpu";
            dev["name"] = cpu.brand.empty() ? cpu.vendor : cpu.brand;
            json devices = json::array();
            devices.push_back(dev);
            json d = json::object();
            d["variant"] = c.variant;
            d["version"] = ALBIREO_VERSION;
            d["devices"] = devices;
            d["usable"] = false;
            d["reason"] = std::string("the processor or the system lacks ") + qk_isa_text(lacks);
            d["reason_code"] = qk_isa_code(lacks);
            c.description = d;
        }
        // Each other library is described by a process of its own, all at the same time:
        // the start of the service waits for the slowest one only.
        std::vector<std::thread> probes;
        for (auto& c : out)
        {
            if (c.description.is_object()) continue;
            engine_candidate* p = &c;
            probes.emplace_back([p, &probe_program]() {
                std::string output;
                const int code = run_process(probe_program, { "--probe-engine", p->path }, &output, 60000);
                // The description is the last line the probe prints.
                const size_t start = output.find('{');
                p->description = start == std::string::npos ? json() : json::parse_or_null(output.substr(start));
                p->usable = code == 0 && p->description["usable"].as_bool(false);
                if (!p->usable && !p->description.contains("reason"))
                {
                    if (!p->description.is_object()) p->description = json::object();
                    p->description["reason"] = code == 0 ? std::string("not usable")
                        : "the library could not be loaded (code " + std::to_string(code) + ")";
                    if (code != 0)
                    {
                        p->description["reason_code"] = "library_failed";
                        json args = json::object();
                        args["code"] = std::to_string(code);
                        p->description["reason_args"] = args;
                    }
                }
            });
        }
        for (auto& t : probes) t.join();
        for (const auto& c : out)
            log_info("engines: " + c.variant + " "
                + (c.usable ? std::string("usable") : "not usable: " + c.description["reason"].as_string()));
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
        {
            std::lock_guard<std::mutex> lock(flight_m_);
            stopping_ = false;
            in_flight_.clear();
        }
        std::unique_lock<std::shared_timed_mutex> life(life_);
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
        {
            std::lock_guard<std::mutex> lock(m_);
            running_ = true;
            current_ = c;
            current_.description = lib_->description();
        }
        log_info("engines: " + c.variant + " started");
        return true;
    }

    void engine_host::stop()
    {
        {
            // The unloadings waiting for their requests end at once.
            std::lock_guard<std::mutex> lock(flight_m_);
            stopping_ = true;
        }
        flight_cv_.notify_all();
        std::unique_lock<std::shared_timed_mutex> life(life_);
        {
            std::lock_guard<std::mutex> lock(m_);
            running_ = false;
            models_.clear();
        }
        if (engine_ && lib_) lib_->destroy(engine_);
        engine_ = nullptr;
        lib_.reset();
    }

    bool engine_host::running() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return running_;
    }

    std::string engine_host::variant() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return running_ ? current_.variant : std::string();
    }

    json engine_host::description() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return current_.description;
    }

    bool engine_host::load_model(const std::string& id, const std::string& path, const json& options,
        std::string& error)
    {
        std::shared_lock<std::shared_timed_mutex> life(life_);
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
        std::lock_guard<std::mutex> lock(m_);
        if (std::find(models_.begin(), models_.end(), id) == models_.end()) models_.push_back(id);
        return true;
    }

    bool engine_host::unload_model(const std::string& id, int drain_seconds)
    {
        begin_unload(id);
        return finish_unload(id, drain_seconds);
    }

    void engine_host::begin_unload(const std::string& id)
    {
        // From now on, submit() refuses the requests for it, and it leaves the list of the
        // models, in one step for keep_model().
        std::lock_guard<std::mutex> lock(flight_m_);
        draining_.emplace(id, false);
        std::lock_guard<std::mutex> models(m_);
        models_.erase(std::remove(models_.begin(), models_.end(), id), models_.end());
    }

    bool engine_host::finish_unload(const std::string& id, int drain_seconds)
    {
        {
            // The requests in progress and waiting are served first, without holding the
            // engine, so that a stop never waits for them.
            std::unique_lock<std::mutex> lock(flight_m_);
            flight_cv_.wait_for(lock, std::chrono::seconds(drain_seconds), [&]() {
                return stopping_ || draining_.count(id) == 0 || in_flight_.find(id) == in_flight_.end();
            });
            const auto it = draining_.find(id);
            if (it == draining_.end()) return false;
            it->second = true;
        }
        {
            std::shared_lock<std::shared_timed_mutex> life(life_);
            if (engine_) lib_->unload_model(engine_, id.c_str());
        }
        std::lock_guard<std::mutex> lock(flight_m_);
        draining_.erase(id);
        return true;
    }

    bool engine_host::keep_model(const std::string& id)
    {
        {
            std::lock_guard<std::mutex> lock(flight_m_);
            const auto it = draining_.find(id);
            if (it == draining_.end() || it->second) return false;
            draining_.erase(it);
            std::lock_guard<std::mutex> models(m_);
            if (std::find(models_.begin(), models_.end(), id) == models_.end()) models_.push_back(id);
        }
        flight_cv_.notify_all();
        return true;
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
        if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR)
        {
            h->host->request_ended(h->model);
            delete h;
        }
    }

    void engine_host::request_ended(const std::string& model)
    {
        {
            std::lock_guard<std::mutex> lock(flight_m_);
            auto it = in_flight_.find(model);
            if (it != in_flight_.end() && --it->second <= 0) in_flight_.erase(it);
        }
        flight_cv_.notify_all();
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
        std::shared_lock<std::shared_timed_mutex> life(life_);
        if (!engine_)
        {
            error = "no engine is running";
            return false;
        }
        const std::string model = request["model"].as_string();
        {
            // Counted before the engine sees it, since its last event may come first.
            std::lock_guard<std::mutex> lock(flight_m_);
            if (draining_.count(model) != 0)
            {
                error = "the model " + model + " is being unloaded";
                return false;
            }
            ++in_flight_[model];
        }
        sink_holder* h = new sink_holder{ std::move(sink), this, model };
        char err[2048] = { 0 };
        const int rc = lib_->submit(engine_, request.dump().c_str(), &engine_host::on_event, h, &id, err, sizeof(err));
        if (rc != ALBIREO_OK)
        {
            delete h;
            request_ended(model);
            error = err;
            return false;
        }
        return true;
    }

    void engine_host::cancel(uint64_t id)
    {
        std::shared_lock<std::shared_timed_mutex> life(life_);
        if (engine_) lib_->cancel(engine_, id);
    }

    json engine_host::query(const json& request)
    {
        std::shared_lock<std::shared_timed_mutex> life(life_);
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
