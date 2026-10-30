// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_ENGINE_HOST_H_
#define ALBIREO_HOST_ENGINE_HOST_H_

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "../common/engine_loader.h"
#include "../common/json.h"

namespace albireo
{

    struct engine_candidate
    {
        std::string variant;   // "cpu", "cpu-avx2", "cpu-avxvnni", "cpu-avx512", "openvino", "cuda"
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
                cpu-avx512, cpu-avxvnni, cpu-avx2, cpu.  "best" is "auto" with a measure
                made when none exists.  Any other value names a variant, used when it is
                usable.

            CONCURRENCY
                A model being loaded or unloaded never delays the other operations: the
                requests, the queries and the cancellations go on during a load, which
                may read several gigabytes.  Only start() and stop() wait for them.

            UNLOADING
                A model leaves the engine once its requests are served: from the start of
                unload_model(), submit() refuses the new requests for it, those in progress
                and those waiting are computed, possibly together with those of the other
                models, and the model is unloaded when the last one ends.  stop() ends the
                waits.
        !*/
    public:
        engine_host();
        ~engine_host();

        // Lists and describes the engine libraries of a folder, through probe_program,
        // probing them all at once.
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
        /*!
            ensures
                - Unloads the model id once its requests are served (see UNLOADING) and
                  returns true, or returns false when keep_model() cancelled the unloading
                  while it waited.  The requests still running after drain_seconds end
                  with the unloading.
        !*/
        bool unload_model(const std::string& id, int drain_seconds = 1800);
        /*!
            ensures
                - The two steps of unload_model(): begin_unload() returns at once, the model
                  taking no new request from then on, and finish_unload() waits for its
                  requests and unloads it, unless keep_model() was called in between.
        !*/
        void begin_unload(const std::string& id);
        bool finish_unload(const std::string& id, int drain_seconds = 1800);
        /*!
            ensures
                - Cancels the unloading of id while it waits for the requests of the model:
                  the model takes requests again, and finish_unload() returns false.
                - Returns false when id is not waiting to leave (not being unloaded, or its
                  unloading already started).
        !*/
        bool keep_model(const std::string& id);
        std::vector<std::string> loaded_models() const;

        // Submits a request; the sink receives its events until the final one.
        bool submit(const json& request, request_sink sink, uint64_t& id, std::string& error);
        void cancel(uint64_t id);
        json query(const json& request);

    private:
        static void ALBIREO_ENGINE_CALL on_event(void* user, int kind, const char* data, size_t size);
        void request_ended(const std::string& model);
        static void ALBIREO_ENGINE_CALL on_log(void* user, int level, const char* message);

        // The engine itself: held exclusively to start and to stop it, shared by the
        // operations that use it.
        mutable std::shared_timed_mutex life_;
        std::unique_ptr<engine_library> lib_;
        albireo_engine* engine_ = nullptr;
        // What the service asks often: the state, the description and the models.
        mutable std::mutex m_;
        bool running_ = false;
        engine_candidate current_;
        std::vector<std::string> models_;

        // The requests in progress of each model, which an unloading waits for, and the
        // models being unloaded, which take no new request: false while they wait for their
        // requests, true once they leave the engine.
        mutable std::mutex flight_m_;
        std::condition_variable flight_cv_;
        std::map<std::string, long> in_flight_;
        std::map<std::string, bool> draining_;
        bool stopping_ = false;
    };

}

#endif // ALBIREO_HOST_ENGINE_HOST_H_
