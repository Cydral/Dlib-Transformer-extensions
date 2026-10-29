// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_ENGINE_HOST_H_
#define ALBIREO_HOST_ENGINE_HOST_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../common/engine_loader.h"
#include "../common/json.h"

namespace albireo
{

    struct engine_candidate
    {
        std::string variant;   // "cpu", "cpu-avx2", "openvino", "cuda"
        std::string path;
        json description;
        bool usable = false;
    };

    // Receives the events of a request (kinds of engine_api.h).
    typedef std::function<void(int kind, const std::string& data)> request_sink;

    class engine_host
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The engine the service runs: the libraries found next to the service,
                described each in a child process so that a library that cannot load
                (a missing graphics driver, say) never takes the service down, the one
                selected for this machine, and the models it holds.

            SELECTION
                "auto" takes the fastest variant measured on this machine when a
                measure exists, otherwise the first usable one of: cuda, openvino,
                cpu-avx2, cpu.  "best" is "auto" with a measure made when none exists.
                Any other value names a variant, used when it is usable.
        !*/
    public:
        engine_host();
        ~engine_host();

        // Lists and describes the engine libraries of a folder, through probe_program.
        std::vector<engine_candidate> discover(const std::string& dir, const std::string& probe_program);

        static const engine_candidate* choose(const std::vector<engine_candidate>& candidates,
            const std::string& wanted, const json& benchmarks);

        bool start(const engine_candidate& c, const json& config, std::string& error);
        void stop();

        /*!
            ensures
                - Returns true once this process started an engine whose library stays loaded
                  until the process ends ("pinned" in its description): the engines of the
                  devices, whose runtimes keep threads and a state of their own.  A change of
                  engine settings then takes a new process.
        !*/
        static bool pinned_engine_started();
        bool running() const;
        std::string variant() const;
        json description() const;

        bool load_model(const std::string& id, const std::string& path, const json& options, std::string& error);
        void unload_model(const std::string& id);
        std::vector<std::string> loaded_models() const;

        // Submits a request; the sink receives its events until the final one.
        bool submit(const json& request, request_sink sink, uint64_t& id, std::string& error);
        void cancel(uint64_t id);
        json query(const json& request);

    private:
        static void ALBIREO_ENGINE_CALL on_event(void* user, int kind, const char* data, size_t size);
        static void ALBIREO_ENGINE_CALL on_log(void* user, int level, const char* message);

        mutable std::mutex m_;
        std::unique_ptr<engine_library> lib_;
        albireo_engine* engine_ = nullptr;
        engine_candidate current_;
        std::vector<std::string> models_;
    };

}

#endif // ALBIREO_HOST_ENGINE_HOST_H_
