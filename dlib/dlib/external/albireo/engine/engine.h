// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_ENGINE_H_
#define ALBIREO_ENGINE_ENGINE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/json.h"

namespace albireo
{
    class thread_pool;
    class tensor_device;
    struct loaded_model;
    struct request_state;
    struct embedding_job;

    // Receives the events of a request: a kind of engine_api.h and its data.
    typedef std::function<void(int kind, const std::string& data)> event_sink;

    // Receives the messages of the engine: a severity of engine_api.h and a text.
    typedef std::function<void(int level, const std::string& message)> log_sink;

    class engine
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An inference engine: the models it loaded, the requests submitted against
                them and the thread that computes them.

                One scheduler thread drives every model.  At each step it takes, for one
                model, the next token of every request being generated and a slice of the
                prompts being read, up to a number of tokens per pass, and computes them in
                a single forward pass: the weights are read once for all the requests, so
                that serving several conversations costs little more than serving one, and
                a long prompt never stalls the conversations already answering.

                The keys and values of a finished request stay in memory with its tokens.
                A new request that starts with the same tokens, as the next turn of a
                conversation does, or as two requests sharing a system prompt do, takes
                them over or copies them instead of computing them again.  The cached
                sequences are dropped, least recently used first, when memory runs short.

                An embedding model answers requests whose "input" lists texts: each text
                is read as a prompt of its own, the texts of every request sharing the
                passes, and the normalized hidden state of its last token becomes its
                vector.  While a language model answers, the passes of the embedding
                models stay short, so that indexing documents never stalls a conversation.

            THREAD SAFETY
                Every member function may be called from any thread.  Event sinks run on
                the scheduler thread and must return quickly.
        !*/
    public:
        engine(const json& config, log_sink sink);
        ~engine();

        engine(const engine&) = delete;
        engine& operator=(const engine&) = delete;

        // Describes this build and the devices it would use (see engine_api.h).
        static json describe();

        void load_model(const std::string& id, const std::string& path, const json& options);
        void unload_model(const std::string& id);

        json query(const json& request);

        // Validates and queues a request; throws std::invalid_argument for a bad one.
        uint64_t submit(const json& request, event_sink sink);
        void cancel(uint64_t id);

    private:
        void run();
        bool step(const std::shared_ptr<loaded_model>& m, long budget);
        uint64_t submit_embeddings(const std::shared_ptr<loaded_model>& m, const json& request, event_sink sink);
        void complete_part(loaded_model& m, request_state& r, const float* hidden);
        void end_job(embedding_job& job, const std::string& code, const std::string& message);
        void admit(loaded_model& m);
        bool assign_sequence(loaded_model& m, request_state& r);
        bool reserve(loaded_model& m, int seq, long length);
        void handle_token(loaded_model& m, request_state& r, int token);
        void emit_text(loaded_model& m, request_state& r, const std::string& text);
        void finish(loaded_model& m, request_state& r, const std::string& reason, bool keep_cache = true);
        void fail(request_state& r, const std::string& code, const std::string& message);
        void log(int level, const std::string& message);
        json model_info(const loaded_model& m) const;
        json stats();

        json settings_;
        log_sink log_;
        std::unique_ptr<thread_pool> pool_;
#ifdef ALBIREO_TENSOR_BACKEND
        // The device of the engines built on the tensors of dlib, shared by the models.
        std::unique_ptr<tensor_device> device_;
#endif
        int threads_ = 1;

        std::mutex mutex_;
        std::condition_variable wake_;
        std::condition_variable unloaded_;
        std::map<std::string, std::shared_ptr<loaded_model>> models_;
        std::map<uint64_t, std::shared_ptr<request_state>> requests_;
        std::map<uint64_t, std::shared_ptr<embedding_job>> jobs_;
        uint64_t next_id_ = 1;
        bool stop_ = false;
        std::thread worker_;

        // Throughput over the recent past, for the statistics.
        std::deque<std::pair<double, long>> generated_window_;
        std::deque<std::pair<double, long>> prompt_window_;
        std::mutex stats_mutex_;
        uint64_t total_generated_ = 0;
        uint64_t total_prompt_ = 0;
        uint64_t total_requests_ = 0;
    };

}

#endif // ALBIREO_ENGINE_ENGINE_H_
