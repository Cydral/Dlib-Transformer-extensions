// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "../common/engine_api.h"
#include "../common/system_info.h"
#include "../common/version.h"
#include "chat_format.h"
#include "cpu_runtime.h"
#include "sampler.h"
#include "thread_pool.h"
#ifdef ALBIREO_TENSOR_BACKEND
#include "tensor_backend.h"
#endif

#ifndef ALBIREO_ENGINE_VARIANT
#if defined(__AVX2__)
#define ALBIREO_ENGINE_VARIANT "cpu-avx2"
#else
#define ALBIREO_ENGINE_VARIANT "cpu"
#endif
#endif

namespace albireo
{
    typedef std::chrono::steady_clock clock_type;

    namespace
    {
        double seconds_since(clock_type::time_point a, clock_type::time_point b)
        {
            return std::chrono::duration<double>(b - a).count();
        }

        double now_seconds()
        {
            return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
        }

        // Bytes of text that end on a complete UTF-8 character; an incomplete sequence at
        // the end waits for the token that completes it.
        size_t complete_utf8(const std::string& s)
        {
            const size_t n = s.size();
            size_t i = n;
            // Walk back over at most three continuation bytes to the lead byte.
            size_t back = 0;
            while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80)
            {
                --i;
                ++back;
            }
            if (i == 0) return back == n ? n : 0;
            const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
            size_t need = 1;
            if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else return n;   // ASCII or an invalid byte: nothing to wait for
            return back + 1 >= need ? n : i - 1;
        }

        // The base64 form of the bytes of an array of floats, as the embeddings travel.
        std::string base64_floats(const std::vector<float>& v)
        {
            static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            const size_t n = v.size() * sizeof(float);
            std::vector<unsigned char> bytes(n);
            if (n) std::memcpy(bytes.data(), v.data(), n);
            std::string out;
            out.reserve((n + 2) / 3 * 4);
            for (size_t i = 0; i < n; i += 3)
            {
                const unsigned b0 = bytes[i];
                const unsigned b1 = i + 1 < n ? bytes[i + 1] : 0;
                const unsigned b2 = i + 2 < n ? bytes[i + 2] : 0;
                out += table[b0 >> 2];
                out += table[((b0 & 3) << 4) | (b1 >> 4)];
                out += i + 1 < n ? table[((b1 & 15) << 2) | (b2 >> 6)] : '=';
                out += i + 2 < n ? table[b2 & 63] : '=';
            }
            return out;
        }

        long common_prefix(const std::vector<int>& a, const std::vector<int>& b)
        {
            const size_t n = std::min(a.size(), b.size());
            size_t i = 0;
            while (i < n && a[i] == b[i]) ++i;
            return static_cast<long>(i);
        }

        std::string type_name(dlib::ggml_type t)
        {
            switch (t)
            {
            case dlib::ggml_type::F32: return "F32";
            case dlib::ggml_type::F16: return "F16";
            case dlib::ggml_type::BF16: return "BF16";
            case dlib::ggml_type::Q4_0: return "Q4_0";
            case dlib::ggml_type::Q4_1: return "Q4_1";
            case dlib::ggml_type::Q5_0: return "Q5_0";
            case dlib::ggml_type::Q5_1: return "Q5_1";
            case dlib::ggml_type::Q8_0: return "Q8_0";
            case dlib::ggml_type::Q8_1: return "Q8_1";
            case dlib::ggml_type::Q2_K: return "Q2_K";
            case dlib::ggml_type::Q3_K: return "Q3_K";
            case dlib::ggml_type::Q4_K: return "Q4_K";
            case dlib::ggml_type::Q5_K: return "Q5_K";
            case dlib::ggml_type::Q6_K: return "Q6_K";
            case dlib::ggml_type::Q8_K: return "Q8_K";
            case dlib::ggml_type::IQ4_NL: return "IQ4_NL";
            }
            return "unknown";
        }

        // The text of a message content: a string, or the text parts of a list.
        std::string content_text(const json& c)
        {
            if (c.is_string()) return c.as_string();
            std::string out;
            for (const auto& part : c.elements())
            {
                if (part.is_string()) out += part.as_string();
                else if (part["type"].as_string() == "text") out += part["text"].as_string();
                else if (part["type"].as_string() == "image_url" || part["type"].as_string() == "input_audio")
                    throw std::invalid_argument("this model reads text only; images and audio are not supported");
            }
            return out;
        }

        std::vector<chat_message> parse_messages(const json& list)
        {
            if (!list.is_array() || list.empty()) throw std::invalid_argument("messages must be a non-empty array");
            std::vector<chat_message> out;
            for (const auto& m : list.elements())
            {
                chat_message c;
                c.role = m["role"].as_string();
                if (c.role == "developer") c.role = "system";
                if (c.role != "system" && c.role != "user" && c.role != "assistant" && c.role != "tool")
                    throw std::invalid_argument("unknown message role '" + c.role + "'");
                c.content = content_text(m["content"]);
                c.reasoning = m.contains("reasoning_content") ? m["reasoning_content"].as_string()
                    : m["reasoning"].as_string();
                c.tool_call_id = m["tool_call_id"].as_string();
                c.name = m["name"].as_string();
                for (const auto& t : m["tool_calls"].elements())
                {
                    tool_call_data d;
                    d.id = t["id"].as_string();
                    const json& f = t["function"];
                    d.name = f["name"].as_string();
                    d.arguments = f["arguments"].is_string() ? f["arguments"].as_string() : f["arguments"].dump();
                    c.tool_calls.push_back(d);
                }
                out.push_back(c);
            }
            return out;
        }
    }

    struct cache_entry
    {
        int seq = -1;
        std::vector<int> tokens;
        uint64_t stamp = 0;
    };

    struct loaded_model
    {
        std::string id;
        std::string path;
        std::shared_ptr<model_file> file;
        model_config cfg;
        dlib::hf_tokenizer tok;
        std::unique_ptr<chat_format> format;
        std::unique_ptr<model_runtime> runtime;
        long context = 0;
        long max_sequences = 8;
        long batch_tokens = 512;
        std::vector<int> stops;
        std::vector<cache_entry> idle;
        std::vector<std::shared_ptr<request_state>> active;
        std::deque<std::shared_ptr<request_state>> waiting;
        std::atomic<bool> unloading{false};
        uint64_t stamp = 0;
        forward_batch batch;
        std::vector<float> logits;
        uint64_t parameters = 0;
        uint64_t bytes = 0;
        std::string quantization;
        std::atomic<long> n_active{0};
        std::atomic<long> n_idle{0};
        std::atomic<uint64_t> generated{0};
        std::atomic<uint64_t> prompted{0};
    };

    // A request of an embedding model: one vector per input, every input computed as a
    // part of its own, a request_state that holds the job.
    struct embedding_job
    {
        uint64_t id = 0;
        event_sink sink;
        std::vector<std::vector<float>> vectors;
        long remaining = 0;
        long tokens = 0;
        long dimensions = 0;    // 0: the width of the model
        std::atomic<bool> cancelled{false};
        bool ended = false;
        clock_type::time_point submitted, started;
    };

    struct request_state
    {
        enum class phase { waiting, prefill, decode, done };

        uint64_t id = 0;
        std::string model_id;
        event_sink sink;
        std::vector<int> prompt;
        std::vector<int> kv_tokens;
        int seq = -1;
        long cached = 0;
        std::unique_ptr<sampler> smp;
        std::unique_ptr<output_parser> parser;
        long max_tokens = 0;
        std::vector<std::string> stops;
        std::string pending_bytes;
        std::string held;
        long generated = 0;
        int next_token = -1;
        int tool_calls = 0;
        std::atomic<bool> cancelled{false};
        phase ph = phase::waiting;
        clock_type::time_point submitted, started, prompt_end, finished;
        bool prompt_done = false;
        // The embedding job this request computes one input of, and the index of the input.
        std::shared_ptr<embedding_job> job;
        long job_index = -1;

        bool is_cancelled() const { return cancelled || (job && job->cancelled); }
    };

    namespace
    {
        bool request_done(const std::shared_ptr<request_state>& r)
        {
            return r->ph == request_state::phase::done;
        }
    }

    engine::engine(const json& config, log_sink sink) : settings_(config), log_(std::move(sink))
    {
        const cpu_info cpu = read_cpu_info();
        threads_ = static_cast<int>(settings_["threads"].as_int(0));
        if (threads_ <= 0) threads_ = cpu.physical_cores;
        threads_ = std::max(1, std::min(threads_, cpu.logical_cores));
        pool_.reset(new thread_pool(threads_));
        pool_->set_spin_time(std::chrono::microseconds(settings_["spin_us"].as_int(2000)));
        log(ALBIREO_LOG_INFO, std::string("engine ") + ALBIREO_ENGINE_VARIANT + " " + ALBIREO_VERSION + ", "
            + std::to_string(threads_) + " threads, kernels " + qk_isa());
#ifdef ALBIREO_TENSOR_BACKEND
        device_.reset(new tensor_device(settings_, *pool_, [this](int level, const std::string& m) { log(level, m); }));
#endif
        worker_ = std::thread([this]() { run(); });
    }

    engine::~engine()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            for (auto& r : requests_) r.second->cancelled = true;
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
        // Requests the worker had no step left to finish.
        std::vector<std::shared_ptr<request_state>> left;
        for (auto& r : requests_) left.push_back(r.second);
        for (auto& r : left)
            if (r->ph != request_state::phase::done) fail(*r, "cancelled", "the engine stopped");
        requests_.clear();
        std::vector<std::shared_ptr<embedding_job>> jobs;
        for (auto& j : jobs_) jobs.push_back(j.second);
        for (auto& j : jobs) end_job(*j, "cancelled", "the engine stopped");
        models_.clear();
    }

    void engine::log(int level, const std::string& message)
    {
        if (log_) log_(level, message);
    }

    json engine::describe()
    {
        const cpu_info cpu = read_cpu_info();
        const memory_info mem = read_memory_info();
        json d = json::object();
        d["variant"] = ALBIREO_ENGINE_VARIANT;
        d["version"] = ALBIREO_VERSION;
        d["backend"] = "processor";
        d["abi"] = static_cast<long long>(ALBIREO_ENGINE_ABI_VERSION);
#if defined(_MSC_VER)
        d["build"] = "MSVC " + std::to_string(_MSC_VER) + ", " + __DATE__;
#elif defined(__clang__)
        d["build"] = std::string("Clang ") + __clang_version__ + ", " + __DATE__;
#elif defined(__GNUC__)
        d["build"] = std::string("GCC ") + __VERSION__ + ", " + __DATE__;
#endif
        d["kernels"] = qk_isa();
        json dev = json::object();
        dev["type"] = "cpu";
        dev["name"] = cpu.brand.empty() ? cpu.vendor : cpu.brand;
        dev["cores"] = cpu.physical_cores;
        dev["threads"] = cpu.logical_cores;
        dev["memory"] = static_cast<unsigned long long>(mem.total);
        json features = json::array();
        if (cpu.avx2) features.push_back("AVX2");
        if (cpu.fma) features.push_back("FMA");
        if (cpu.f16c) features.push_back("F16C");
        if (cpu.avx_vnni) features.push_back("AVX-VNNI");
        if (cpu.avx512f) features.push_back("AVX-512");
        if (cpu.avx512vnni) features.push_back("AVX-512 VNNI");
        if (cpu.amx) features.push_back("AMX");
        dev["features"] = features;
        json devices = json::array();
        devices.push_back(dev);
        bool usable = true;
        std::string reason;
        // The kernels of the variant: the rest of the library uses AVX2 at most, so that
        // this description runs on the processors that lack the instructions of the kernels.
#if defined(__AVX2__)
        const unsigned lacks = qk_missing_isa(cpu, qk_isa_avx2);
#else
        const unsigned lacks = qk_missing_isa(cpu);
#endif
        // The reasons carry a code, with which the windows explain them in the language of
        // the user.
        std::string reason_code;
        json reason_args = json::object();
        if (lacks)
        {
            usable = false;
            reason = std::string("the processor or the system lacks ") + qk_isa_text(lacks);
            reason_code = qk_isa_code(lacks);
        }
#ifdef ALBIREO_TENSOR_BACKEND
        // The devices of dlib take the products; the library stays loaded once started,
        // since the runtimes of the devices keep threads in the process.
        std::string backend, runtime, why, why_code;
        json why_args = json::object();
        const json accelerators = tensor_device::describe(backend, runtime, why, why_code, why_args);
        d["backend"] = backend;
        if (!runtime.empty()) d["runtime"] = runtime;
        for (const auto& a : accelerators.elements()) devices.push_back(a);
        d["pinned"] = true;
        if (usable && accelerators.empty() && backend != "dlib")
        {
            usable = false;
            reason = why;
            reason_code = why_code;
            reason_args = why_args;
        }
#endif
        d["devices"] = devices;
        d["usable"] = usable;
        if (!reason.empty()) d["reason"] = reason;
        if (!reason_code.empty())
        {
            d["reason_code"] = reason_code;
            if (reason_args.size() > 0) d["reason_args"] = reason_args;
        }
        return d;
    }

    // Models

    void engine::load_model(const std::string& id, const std::string& path, const json& options)
    {
        if (id.empty()) throw std::invalid_argument("a model needs an identifier");
        auto m = std::make_shared<loaded_model>();
        m->id = id;
        m->path = path;
        m->file = std::make_shared<model_file>(path);
        m->cfg = read_model_config(*m->file);
        m->tok.load_from_gguf(m->file->reader());
        if (static_cast<long>(m->tok.size()) != m->cfg.vocab)
            throw std::runtime_error("the vocabulary and the output layer of the model disagree");
        m->format = chat_format::detect(m->tok, m->cfg.chat_template);

        // An embedding model reads passages and questions of a few hundred tokens, many
        // at a time, and keeps nothing of them: its defaults are its own, and its keys and
        // values come in small blocks, given back as soon as each input is read.
        const bool embedding = m->cfg.is_embedding();
        if (embedding && m->cfg.pooling != 3)
            throw std::runtime_error("this embedding model pools its hidden states by a method the engine does not "
                "implement (the engine takes the state of the last token)");
        long ctx = options["context"].as_int(embedding ? 0 : settings_["context"].as_int(0));
        if (ctx <= 0) ctx = std::min<long>(m->cfg.ctx_train, embedding ? 8192 : 32768);
        const long limit = m->cfg.ctx_train > 0 ? std::max<long>(m->cfg.ctx_train, 256) : ctx;
        m->context = std::max<long>(256, std::min<long>(ctx, limit));
        m->max_sequences = std::max<long>(1, options["max_sequences"].as_int(
            embedding ? 8 : settings_["max_sequences"].as_int(8)));
        m->batch_tokens = std::max<long>(16, options["batch_tokens"].as_int(
            embedding ? 256 : settings_["batch_tokens"].as_int(512)));

        const size_t per_token = 2 * static_cast<size_t>(m->cfg.n_layers) * m->cfg.kv_dim() * sizeof(uint16_t);
        size_t kv = static_cast<size_t>(options["kv_mb"].as_int(embedding ? 0 : settings_["kv_mb"].as_int(0))) << 20;
        if (kv == 0 && embedding)
        {
            // One input of the whole context.
            kv = per_token * static_cast<size_t>(m->context);
        }
        else if (kv == 0)
        {
            // Two full contexts, within a quarter of the memory available.
            const memory_info mem = read_memory_info();
            kv = per_token * static_cast<size_t>(m->context) * 2;
            if (mem.available > 0) kv = std::min<size_t>(kv, static_cast<size_t>(mem.available / 4));
            kv = std::max<size_t>(kv, per_token * static_cast<size_t>(std::min<long>(m->context, 4096)));
        }
        std::unique_ptr<cpu_runtime> runtime(new cpu_runtime(m->file, m->cfg, *pool_, kv, embedding ? 64 : 256));
#ifdef ALBIREO_TENSOR_BACKEND
        runtime->set_offload(device_->make_offload(id));
#endif
        m->runtime = std::move(runtime);

        if (m->tok.eos_id() >= 0) m->stops.push_back(m->tok.eos_id());
        for (int s : m->format->stop_tokens()) m->stops.push_back(s);
        std::sort(m->stops.begin(), m->stops.end());
        m->stops.erase(std::unique(m->stops.begin(), m->stops.end()), m->stops.end());

        // Size and dominant storage type of the weights.
        std::map<std::string, uint64_t> by_type;
        for (const auto& t : m->file->reader().tensors())
        {
            m->parameters += t.n_elements();
            by_type[type_name(t.type)] += t.nbytes;
        }
        m->bytes = m->file->size();
        uint64_t most = 0;
        for (const auto& kvp : by_type)
            if (kvp.second > most) { most = kvp.second; m->quantization = kvp.first; }

        if (settings_["warm_up"].as_bool(true)) m->file->warm_up(true);

        const std::string use = embedding ? "vectors of " + std::to_string(m->cfg.d_model) + " dimensions"
            : std::string("chat format ") + m->format->name();
        log(ALBIREO_LOG_INFO, "model '" + id + "': " + m->runtime->description() + ", " + use
            + ", context " + std::to_string(m->context) + ", keys and values up to "
            + std::to_string(m->runtime->kv_bytes_budget() >> 20) + " MB"
            + (m->tok.exact_pretokenizer() ? "" : ", pre-tokenizer '" + m->tok.pretokenizer() + "' approximated"));

        unload_model(id);
        std::lock_guard<std::mutex> lock(mutex_);
        models_[id] = m;
    }

    void engine::unload_model(const std::string& id)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = models_.find(id);
        if (it == models_.end()) return;
        it->second->unloading = true;
        wake_.notify_all();
        unloaded_.wait(lock, [&]() { return models_.find(id) == models_.end() || stop_; });
    }

    json engine::model_info(const loaded_model& m) const
    {
        json j = json::object();
        j["id"] = m.id;
        j["name"] = m.cfg.name;
        j["arch"] = m.cfg.arch;
        j["path"] = m.path;
        j["context"] = m.context;
        j["trained_context"] = m.cfg.ctx_train;
        j["format"] = m.format->name();
        j["tools"] = !m.cfg.is_embedding() && m.format->supports_tools();
        j["reasoning"] = !m.cfg.is_embedding() && m.format->supports_reasoning();
        j["parameters"] = static_cast<unsigned long long>(m.parameters);
        j["quantization"] = m.quantization;
        j["bytes"] = static_cast<unsigned long long>(m.bytes);
        j["kv_bytes_per_token"] = static_cast<unsigned long long>(m.runtime->kv_bytes_per_token());
        j["experts"] = m.cfg.n_experts;
        j["embedding"] = m.cfg.is_embedding();
        if (m.cfg.is_embedding()) j["dimensions"] = m.cfg.d_model;
        j["description"] = albireo::describe(m.cfg);
        return j;
    }

    // Requests

    uint64_t engine::submit(const json& req, event_sink sink)
    {
        const std::string model_id = req["model"].as_string();
        std::shared_ptr<loaded_model> m;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = models_.find(model_id);
            if (it == models_.end() || it->second->unloading)
                throw std::invalid_argument("the model '" + model_id + "' is not loaded");
            m = it->second;
        }
        if (req.contains("input"))
        {
            if (!m->cfg.is_embedding())
                throw std::invalid_argument("the model '" + model_id + "' writes text; vectors come from an embedding "
                    "model");
            return submit_embeddings(m, req, std::move(sink));
        }
        if (m->cfg.is_embedding())
            throw std::invalid_argument("the model '" + model_id + "' computes vectors: its requests give their texts "
                "in \"input\"");

        auto r = std::make_shared<request_state>();
        r->model_id = model_id;
        r->sink = std::move(sink);
        r->submitted = clock_type::now();

        render_options opt;
        current_dates(opt);
        const std::string effort = req["reasoning_effort"].as_string();
        bool reasoning = m->format->supports_reasoning();
        if (req.contains("reasoning")) reasoning = req["reasoning"].as_bool(reasoning);
        if (!effort.empty()) reasoning = effort != "none" && effort != "minimal";
        if (req["chat_template_kwargs"].contains("enable_thinking"))
            reasoning = req["chat_template_kwargs"]["enable_thinking"].as_bool(reasoning);
        opt.reasoning = reasoning && m->format->supports_reasoning();
        if (req["tool_choice"].as_string() != "none" && req["tools"].is_array() && !req["tools"].empty())
        {
            if (!m->format->supports_tools()) throw std::invalid_argument("this model does not support tools");
            opt.tools = req["tools"];
        }

        if (req.contains("messages"))
        {
            const std::vector<chat_message> messages = parse_messages(req["messages"]);
            std::vector<prompt_piece> pieces;
            m->format->render(messages, opt, pieces);
            r->prompt = chat_format::tokenize(m->tok, pieces, m->format->separate_pieces());
            r->parser = m->format->make_parser(opt);
        }
        else if (req["prompt"].is_string())
        {
            r->prompt = m->tok.encode(req["prompt"].as_string(), m->tok.add_bos_default(), false, true);
            // A raw completion is plain text: no markers to interpret.
            struct passthrough : output_parser
            {
                void feed(const std::string& t, std::vector<output_event>& e) override
                {
                    output_event ev;
                    ev.text = t;
                    e.push_back(ev);
                }
                void finish(std::vector<output_event>&) override {}
            };
            r->parser.reset(new passthrough());
        }
        else
        {
            throw std::invalid_argument("a request needs messages or a prompt");
        }
        if (r->prompt.empty()) throw std::invalid_argument("the prompt is empty");
        if (static_cast<long>(r->prompt.size()) >= m->context)
            throw std::invalid_argument("the prompt holds " + std::to_string(r->prompt.size())
                + " tokens, beyond the context of " + std::to_string(m->context) + " tokens of the model");

        // Sampling: the values of the request, or those advised for the family.
        sampling_params sp;
        const bool chatml_reasoning = opt.reasoning && std::string(m->format->name()) == "chatml";
        sp.temperature = static_cast<float>(req["temperature"].as_double(chatml_reasoning ? 0.6 : 0.7));
        sp.top_p = static_cast<float>(req["top_p"].as_double(chatml_reasoning ? 0.95 : 0.9));
        sp.top_k = static_cast<int>(req["top_k"].as_int(chatml_reasoning ? 20 : 40));
        sp.min_p = static_cast<float>(req["min_p"].as_double(0.0));
        sp.repeat_penalty =
            static_cast<float>(req["repeat_penalty"].as_double(req["repetition_penalty"].as_double(1.0)));
        sp.repeat_last_n = static_cast<int>(req["repeat_last_n"].as_int(64));
        sp.presence_penalty = static_cast<float>(req["presence_penalty"].as_double(0.0));
        sp.frequency_penalty = static_cast<float>(req["frequency_penalty"].as_double(0.0));
        sp.seed = static_cast<uint64_t>(req["seed"].as_int(0));
        for (const auto& b : req["logit_bias"].items())
            sp.logit_bias[std::atoi(b.first.c_str())] = static_cast<float>(b.second.as_double(0.0));
        if (sp.temperature < 0 || sp.top_p < 0 || sp.top_p > 1 || sp.min_p < 0 || sp.min_p > 1)
            throw std::invalid_argument("sampling parameters out of range");
        r->smp.reset(new sampler(sp));

        const long room = m->context - static_cast<long>(r->prompt.size());
        long max_tokens = req["max_completion_tokens"].as_int(req["max_tokens"].as_int(0));
        if (max_tokens <= 0 || max_tokens > room) max_tokens = room;
        r->max_tokens = max_tokens;

        if (req["stop"].is_string() && !req["stop"].as_string().empty()) r->stops.push_back(req["stop"].as_string());
        for (const auto& s : req["stop"].elements())
            if (s.is_string() && !s.as_string().empty()) r->stops.push_back(s.as_string());
        if (r->stops.size() > 16) throw std::invalid_argument("at most 16 stop sequences");

        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) throw std::runtime_error("the engine is stopping");
        // The model may have been unloaded while the prompt was prepared: its queue is no
        // longer read.
        if (m->unloading) throw std::invalid_argument("the model '" + model_id + "' is not loaded");
        r->id = next_id_++;
        requests_[r->id] = r;
        m->waiting.push_back(r);
        ++total_requests_;
        wake_.notify_all();
        return r->id;
    }

    uint64_t engine::submit_embeddings(const std::shared_ptr<loaded_model>& m, const json& req, event_sink sink)
    {
        // The inputs: a text, a list of texts, a list of token identifiers, or a list of
        // such lists.  A text receives the tokens the model adds around its inputs (the
        // final token whose state becomes the vector, for the decoders); token lists are
        // taken as they are.
        std::vector<std::vector<int>> inputs;
        const json& in = req["input"];
        auto text_tokens = [&](const std::string& text) {
            return m->tok.encode(text, m->tok.add_bos_default(), m->tok.add_eos_default(), false);
        };
        auto token_list = [&](const json& list) {
            std::vector<int> ids;
            for (const auto& t : list.elements())
            {
                if (!t.is_integer() || t.as_int() < 0 || t.as_int() >= m->cfg.vocab)
                    throw std::invalid_argument("input holds an invalid token identifier");
                ids.push_back(static_cast<int>(t.as_int()));
            }
            return ids;
        };
        if (in.is_string()) inputs.push_back(text_tokens(in.as_string()));
        else if (in.is_array() && !in.empty() && in[0].is_integer()) inputs.push_back(token_list(in));
        else if (in.is_array())
        {
            for (const auto& e : in.elements())
            {
                if (e.is_string()) inputs.push_back(text_tokens(e.as_string()));
                else if (e.is_array()) inputs.push_back(token_list(e));
                else throw std::invalid_argument("input must hold texts or lists of token identifiers");
            }
        }
        if (inputs.empty()) throw std::invalid_argument("input is empty");
        if (inputs.size() > 2048) throw std::invalid_argument("at most 2048 inputs per request");
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (inputs[i].empty()) throw std::invalid_argument("input " + std::to_string(i) + " is empty");
            if (static_cast<long>(inputs[i].size()) > m->context)
                throw std::invalid_argument("input " + std::to_string(i) + " holds " + std::to_string(inputs[i].size())
                    + " tokens, beyond the " + std::to_string(m->context) + " tokens the model reads at once");
        }
        const long dimensions = req["dimensions"].as_int(0);
        if (dimensions < 0 || dimensions > m->cfg.d_model)
            throw std::invalid_argument("dimensions must lie between 1 and " + std::to_string(m->cfg.d_model));

        auto job = std::make_shared<embedding_job>();
        job->sink = std::move(sink);
        job->vectors.resize(inputs.size());
        job->remaining = static_cast<long>(inputs.size());
        job->dimensions = dimensions;
        job->submitted = clock_type::now();
        std::vector<std::shared_ptr<request_state>> parts;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto r = std::make_shared<request_state>();
            r->model_id = m->id;
            r->job = job;
            r->job_index = static_cast<long>(i);
            r->submitted = job->submitted;
            job->tokens += static_cast<long>(inputs[i].size());
            r->prompt.swap(inputs[i]);
            parts.push_back(r);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) throw std::runtime_error("the engine is stopping");
        if (m->unloading) throw std::invalid_argument("the model '" + m->id + "' is not loaded");
        job->id = next_id_++;
        jobs_[job->id] = job;
        for (auto& r : parts)
        {
            r->id = next_id_++;
            m->waiting.push_back(r);
        }
        ++total_requests_;
        wake_.notify_all();
        return job->id;
    }

    void engine::complete_part(loaded_model& m, request_state& r, const float* hidden)
    {
        // The state of the last token, cut to the dimensions asked (the leading ones, for
        // the models trained to keep their meaning when shortened), at unit length.
        embedding_job& job = *r.job;
        const long n = job.dimensions > 0 ? job.dimensions : m.cfg.d_model;
        std::vector<float> v(hidden, hidden + n);
        double norm = 0;
        for (float x : v) norm += static_cast<double>(x) * x;
        const float scale = norm > 0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0f;
        for (float& x : v) x *= scale;
        job.vectors[static_cast<size_t>(r.job_index)].swap(v);
        r.ph = request_state::phase::done;
        r.finished = clock_type::now();
        if (r.seq >= 0) m.runtime->release_sequence(r.seq);
        r.seq = -1;
        if (job.started.time_since_epoch().count() == 0 || r.started < job.started) job.started = r.started;
        if (--job.remaining > 0 || job.ended) return;

        job.ended = true;
        json data = json::array();
        for (size_t i = 0; i < job.vectors.size(); ++i)
        {
            json e = json::object();
            e["index"] = static_cast<long long>(i);
            e["embedding"] = base64_floats(job.vectors[i]);
            data.push_back(e);
        }
        json done = json::object();
        done["data"] = data;
        done["dimensions"] = n;
        json usage = json::object();
        usage["prompt_tokens"] = job.tokens;
        usage["total_tokens"] = job.tokens;
        done["usage"] = usage;
        json timings = json::object();
        const clock_type::time_point end = clock_type::now();
        const double compute = seconds_since(job.started, end);
        timings["queued_ms"] = 1000.0 * seconds_since(job.submitted, job.started);
        timings["prompt_ms"] = 1000.0 * compute;
        timings["prompt_per_second"] = compute > 0 ? job.tokens / compute : 0.0;
        done["timings"] = timings;
        job.sink(ALBIREO_EVENT_DONE, done.dump());
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.erase(job.id);
    }

    void engine::end_job(embedding_job& job, const std::string& code, const std::string& message)
    {
        // The first failure or cancellation of an input ends the whole request; its other
        // inputs see the cancellation and stop.
        job.cancelled = true;
        if (job.ended) return;
        job.ended = true;
        json e = json::object();
        e["code"] = code;
        e["message"] = message;
        if (job.sink) job.sink(ALBIREO_EVENT_ERROR, e.dump());
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.erase(job.id);
    }

    void engine::cancel(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = requests_.find(id);
        if (it != requests_.end()) it->second->cancelled = true;
        auto j = jobs_.find(id);
        if (j != jobs_.end()) j->second->cancelled = true;
        wake_.notify_all();
    }

    // Scheduler

    void engine::run()
    {
        for (;;)
        {
            std::vector<std::shared_ptr<loaded_model>> models;
            // Whether a language model has requests: the embedding models then compute
            // short passes, the answers being what the users wait for.
            bool answering = false;
            std::vector<bool> busy;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&]() {
                    if (stop_) return true;
                    for (const auto& m : models_)
                        if (m.second->unloading || !m.second->waiting.empty() || !m.second->active.empty()) return true;
                    return false;
                });
                if (stop_) break;
                for (const auto& m : models_)
                {
                    models.push_back(m.second);
                    const bool has_requests = !m.second->waiting.empty() || !m.second->active.empty();
                    busy.push_back(has_requests);
                    if (!m.second->cfg.is_embedding() && has_requests) answering = true;
                }
            }
            // The device keeps the copies of the models with requests, and may give the
            // room of the others to them (tensor_backend.h).
            for (size_t i = 0; i < models.size(); ++i) models[i]->runtime->set_busy(busy[i]);
            for (const auto& m : models)
            {
                if (m->unloading)
                {
                    // Finish what the model still holds, then let it go.
                    std::deque<std::shared_ptr<request_state>> waiting;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        waiting.swap(m->waiting);
                    }
                    for (auto& r : waiting) fail(*r, "unloaded", "the model was unloaded");
                    for (auto& r : m->active)
                        if (r->ph != request_state::phase::done) fail(*r, "unloaded", "the model was unloaded");
                    m->active.clear();
                    std::lock_guard<std::mutex> lock(mutex_);
                    models_.erase(m->id);
                    unloaded_.notify_all();
                    continue;
                }
                try
                {
                    const bool short_pass = m->cfg.is_embedding() && answering;
                    step(m, short_pass ? std::min<long>(m->batch_tokens, 16) : m->batch_tokens);
                    // An embedding model keeps nothing between its requests: once idle, the
                    // memory of the inputs it read goes back to the system.
                    if (m->cfg.is_embedding() && m->active.empty())
                    {
                        bool idle = false;
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            idle = m->waiting.empty();
                        }
                        if (idle) m->runtime->trim_memory();
                    }
                }
                catch (const std::exception& e)
                {
                    log(ALBIREO_LOG_ERROR, "model '" + m->id + "': " + e.what());
                    for (auto& r : m->active)
                        if (r->ph != request_state::phase::done) fail(*r, "internal", e.what());
                    for (auto& r : m->active)
                        if (r->seq >= 0) { m->runtime->release_sequence(r->seq); r->seq = -1; }
                    m->active.clear();
                }
            }
        }
    }

    void engine::admit(loaded_model& m)
    {
        for (;;)
        {
            std::shared_ptr<request_state> r;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (m.waiting.empty() || static_cast<long>(m.active.size()) >= m.max_sequences) return;
                r = m.waiting.front();
                m.waiting.pop_front();
            }
            if (r->is_cancelled())
            {
                finish(m, *r, "cancelled");
                continue;
            }
            r->started = clock_type::now();
            if (!assign_sequence(m, *r))
            {
                fail(*r, "out_of_memory", "no memory left for the keys and values of this request");
                continue;
            }
            r->ph = request_state::phase::prefill;
            m.active.push_back(r);
            if (r->cached > 0)
            {
                json p = json::object();
                p["prompt_done"] = r->cached;
                p["prompt_total"] = static_cast<long>(r->prompt.size());
                p["cached"] = r->cached;
                r->sink(ALBIREO_EVENT_PROGRESS, p.dump());
            }
        }
    }

    bool engine::assign_sequence(loaded_model& m, request_state& r)
    {
        // The cached sequence sharing the longest prefix with the prompt, one prompt token
        // at least being left to compute: its logits begin the answer.
        const long limit = static_cast<long>(r.prompt.size()) - 1;
        long best_len = 0;
        size_t best = m.idle.size();
        for (size_t i = 0; i < m.idle.size(); ++i)
        {
            const long n = std::min(limit, common_prefix(m.idle[i].tokens, r.prompt));
            if (n > best_len)
            {
                best_len = n;
                best = i;
            }
        }
        if (best < m.idle.size())
        {
            cache_entry& e = m.idle[best];
            const bool take = best_len * 2 >= static_cast<long>(e.tokens.size());
            if (take)
            {
                // The rest of the cached sequence is of no further use: take it over.
                r.seq = e.seq;
                m.runtime->truncate_sequence(r.seq, best_len);
                m.idle.erase(m.idle.begin() + static_cast<std::ptrdiff_t>(best));
            }
            else
            {
                // Another conversation may still need it: copy the shared prefix.
                const int s = m.runtime->create_sequence();
                bool copied = m.runtime->copy_sequence(e.seq, s, best_len);
                while (!copied && m.idle.size() > 1)
                {
                    // Free the least recently used other sequence and try again.
                    size_t lru = m.idle.size();
                    for (size_t i = 0; i < m.idle.size(); ++i)
                        if (i != best && (lru == m.idle.size() || m.idle[i].stamp < m.idle[lru].stamp)) lru = i;
                    m.runtime->release_sequence(m.idle[lru].seq);
                    m.idle.erase(m.idle.begin() + static_cast<std::ptrdiff_t>(lru));
                    if (lru < best) --best;
                    copied = m.runtime->copy_sequence(m.idle[best].seq, s, best_len);
                }
                if (!copied)
                {
                    m.runtime->release_sequence(s);
                    r.seq = m.idle[best].seq;
                    m.runtime->truncate_sequence(r.seq, best_len);
                    m.idle.erase(m.idle.begin() + static_cast<std::ptrdiff_t>(best));
                }
                else
                {
                    r.seq = s;
                    m.idle[best].stamp = ++m.stamp;
                }
            }
            r.cached = best_len;
        }
        else
        {
            r.seq = m.runtime->create_sequence();
            r.cached = 0;
        }
        r.kv_tokens.assign(r.prompt.begin(), r.prompt.begin() + r.cached);
        return true;
    }

    bool engine::reserve(loaded_model& m, int seq, long length)
    {
        while (!m.runtime->reserve_sequence(seq, length))
        {
            if (m.idle.empty()) return false;
            size_t lru = 0;
            for (size_t i = 1; i < m.idle.size(); ++i)
                if (m.idle[i].stamp < m.idle[lru].stamp) lru = i;
            m.runtime->release_sequence(m.idle[lru].seq);
            m.idle.erase(m.idle.begin() + static_cast<std::ptrdiff_t>(lru));
        }
        return true;
    }

    bool engine::step(const std::shared_ptr<loaded_model>& mp, long budget_tokens)
    {
        loaded_model& m = *mp;
        for (auto& r : m.active)
            if (r->is_cancelled() && r->ph != request_state::phase::done) finish(m, *r, "cancelled");
        m.active.erase(std::remove_if(m.active.begin(), m.active.end(), request_done), m.active.end());
        admit(m);
        if (m.active.empty())
        {
            m.n_active = 0;
            return false;
        }

        // The next token of every request being generated, then slices of the prompts.
        struct row { request_state* r; long count; bool logits; };
        std::vector<row> rows;
        forward_batch& b = m.batch;
        b.clear();
        b.hidden_only = m.cfg.is_embedding();
        long budget = budget_tokens;
        for (auto& rp : m.active)
        {
            request_state& r = *rp;
            if (r.ph != request_state::phase::decode) continue;
            const long len = m.runtime->sequence_length(r.seq);
            if (len + 1 > m.context || !reserve(m, r.seq, len + 1))
            {
                finish(m, r, "length");
                continue;
            }
            b.segments.push_back({ r.seq, static_cast<long>(b.tokens.size()), 1 });
            b.logit_rows.push_back(static_cast<long>(b.tokens.size()));
            b.tokens.push_back(r.next_token);
            rows.push_back({ &r, 1, true });
            --budget;
        }
        for (auto& rp : m.active)
        {
            request_state& r = *rp;
            if (r.ph != request_state::phase::prefill || budget <= 0) continue;
            const long done = static_cast<long>(r.kv_tokens.size());
            const long n = std::min<long>(static_cast<long>(r.prompt.size()) - done, budget);
            if (!reserve(m, r.seq, done + n))
            {
                // Nothing else to give back: this prompt does not fit.
                if (rows.empty()) fail(r, "out_of_memory", "the keys and values of this prompt do not fit in memory");
                continue;
            }
            const bool last = done + n == static_cast<long>(r.prompt.size());
            b.segments.push_back({ r.seq, static_cast<long>(b.tokens.size()), n });
            if (last) b.logit_rows.push_back(static_cast<long>(b.tokens.size()) + n - 1);
            b.tokens.insert(b.tokens.end(), r.prompt.begin() + done, r.prompt.begin() + done + n);
            rows.push_back({ &r, n, last });
            budget -= n;
        }
        if (rows.empty())
        {
            m.active.erase(std::remove_if(m.active.begin(), m.active.end(), request_done), m.active.end());
            return false;
        }

        m.runtime->forward(b, m.logits);

        long prompt_tokens = 0, generated = 0;
        size_t logit_row = 0;
        const long V = m.cfg.vocab;
        for (const row& w : rows)
        {
            request_state& r = *w.r;
            if (r.ph == request_state::phase::decode)
            {
                r.kv_tokens.push_back(r.next_token);
            }
            else
            {
                const long done = static_cast<long>(r.kv_tokens.size());
                r.kv_tokens.insert(r.kv_tokens.end(), r.prompt.begin() + done, r.prompt.begin() + done + w.count);
                prompt_tokens += w.count;
                if (!r.job)
                {
                    json p = json::object();
                    p["prompt_done"] = static_cast<long>(r.kv_tokens.size());
                    p["prompt_total"] = static_cast<long>(r.prompt.size());
                    p["cached"] = r.cached;
                    r.sink(ALBIREO_EVENT_PROGRESS, p.dump());
                }
            }
            if (!w.logits) continue;
            const long this_row = static_cast<long>(logit_row++);
            if (r.job)
            {
                // An input of an embedding request, read to its end.
                complete_part(m, r, m.runtime->hidden().data() + this_row * m.cfg.d_model);
                continue;
            }
            float* lg = m.logits.data() + this_row * V;
            if (r.ph == request_state::phase::prefill)
            {
                r.prompt_end = clock_type::now();
                // The penalties see the end of the prompt as the reference servers do.
                const size_t tail = std::min<size_t>(r.prompt.size(), 64);
                for (size_t i = r.prompt.size() - tail; i < r.prompt.size(); ++i) r.smp->accept(r.prompt[i]);
                r.ph = request_state::phase::decode;
            }
            if (r.cancelled)
            {
                finish(m, r, "cancelled");
                continue;
            }
            const int token = r.smp->sample(lg, V);
            ++generated;
            handle_token(m, r, token);
        }

        m.prompted += static_cast<uint64_t>(prompt_tokens);
        m.generated += static_cast<uint64_t>(generated);
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            const double t = now_seconds();
            if (generated) generated_window_.emplace_back(t, generated);
            if (prompt_tokens) prompt_window_.emplace_back(t, prompt_tokens);
            total_generated_ += static_cast<uint64_t>(generated);
            total_prompt_ += static_cast<uint64_t>(prompt_tokens);
            while (!generated_window_.empty() && generated_window_.front().first < t - 5) generated_window_.pop_front();
            while (!prompt_window_.empty() && prompt_window_.front().first < t - 5) prompt_window_.pop_front();
        }

        m.active.erase(std::remove_if(m.active.begin(), m.active.end(), request_done), m.active.end());
        m.n_active = static_cast<long>(m.active.size());
        m.n_idle = static_cast<long>(m.idle.size());
        return true;
    }

    void engine::handle_token(loaded_model& m, request_state& r, int token)
    {
        if (std::binary_search(m.stops.begin(), m.stops.end(), token))
        {
            finish(m, r, "stop");
            return;
        }
        ++r.generated;
        r.smp->accept(token);
        // Control tokens are not text, except the markers the output parser reads.
        if (m.tok.token_type(token) != dlib::hf_tokenizer::TT_CONTROL
            || std::binary_search(m.format->marker_tokens().begin(), m.format->marker_tokens().end(), token))
            r.pending_bytes += m.tok.token_bytes(token);
        const size_t n = complete_utf8(r.pending_bytes);
        if (n > 0)
        {
            const std::string text = r.pending_bytes.substr(0, n);
            r.pending_bytes.erase(0, n);
            emit_text(m, r, text);
            if (r.ph == request_state::phase::done) return;
        }
        if (r.generated >= r.max_tokens || m.runtime->sequence_length(r.seq) + 1 >= m.context)
        {
            finish(m, r, "length");
            return;
        }
        r.next_token = token;
    }

    namespace
    {
        void deliver(request_state& r, std::vector<output_event>& events)
        {
            for (auto& e : events)
            {
                if (e.kind == output_kind::content) r.sink(ALBIREO_EVENT_CONTENT, e.text);
                else if (e.kind == output_kind::reasoning) r.sink(ALBIREO_EVENT_REASONING, e.text);
                else
                {
                    json c = json::object();
                    c["index"] = r.tool_calls++;
                    c["id"] = e.call.id;
                    c["name"] = e.call.name;
                    c["arguments"] = e.call.arguments;
                    r.sink(ALBIREO_EVENT_TOOL_CALL, c.dump());
                }
            }
            events.clear();
        }
    }

    void engine::emit_text(loaded_model& m, request_state& r, const std::string& text)
    {
        std::vector<output_event> events;
        if (r.stops.empty())
        {
            r.parser->feed(text, events);
            deliver(r, events);
            return;
        }
        r.held += text;
        size_t at = std::string::npos;
        for (const auto& s : r.stops)
        {
            const size_t p = r.held.find(s);
            if (p < at) at = p;
        }
        if (at != std::string::npos)
        {
            r.parser->feed(r.held.substr(0, at), events);
            r.held.clear();
            deliver(r, events);
            finish(m, r, "stop");
            return;
        }
        // Hold back what could begin a stop sequence.
        size_t keep = 0;
        for (const auto& s : r.stops)
            for (size_t len = std::min(s.size() - 1, r.held.size()); len > keep; --len)
                if (r.held.compare(r.held.size() - len, len, s, 0, len) == 0)
                {
                    keep = len;
                    break;
                }
        r.parser->feed(r.held.substr(0, r.held.size() - keep), events);
        r.held.erase(0, r.held.size() - keep);
        deliver(r, events);
    }

    void engine::finish(loaded_model& m, request_state& r, const std::string& reason, bool keep_cache)
    {
        if (r.ph == request_state::phase::done) return;
        if (r.job)
        {
            r.ph = request_state::phase::done;
            if (r.seq >= 0) m.runtime->release_sequence(r.seq);
            r.seq = -1;
            end_job(*r.job, reason, "the request was " + reason);
            return;
        }
        std::vector<output_event> events;
        if (r.parser)
        {
            if (!r.held.empty()) r.parser->feed(r.held, events);
            r.held.clear();
            r.parser->finish(events);
            deliver(r, events);
        }
        r.ph = request_state::phase::done;
        r.finished = clock_type::now();

        json done = json::object();
        done["finish_reason"] = reason == "stop" && r.tool_calls > 0 ? std::string("tool_calls") : reason;
        json usage = json::object();
        usage["prompt_tokens"] = static_cast<long>(r.prompt.size());
        usage["completion_tokens"] = r.generated;
        usage["total_tokens"] = static_cast<long>(r.prompt.size()) + r.generated;
        usage["cached_tokens"] = r.cached;
        done["usage"] = usage;
        json timings = json::object();
        const bool started = r.started.time_since_epoch().count() != 0;
        const bool prompted = r.prompt_end.time_since_epoch().count() != 0;
        timings["queued_ms"] = started ? 1000.0 * seconds_since(r.submitted, r.started) : 0.0;
        if (prompted)
        {
            const double pt = seconds_since(r.started, r.prompt_end);
            const double gt = seconds_since(r.prompt_end, r.finished);
            const long computed = static_cast<long>(r.prompt.size()) - r.cached;
            timings["prompt_ms"] = 1000.0 * pt;
            timings["generation_ms"] = 1000.0 * gt;
            timings["prompt_per_second"] = pt > 0 ? computed / pt : 0.0;
            timings["generation_per_second"] = gt > 0 && r.generated > 1 ? (r.generated - 1) / gt : 0.0;
        }
        done["timings"] = timings;
        r.sink(ALBIREO_EVENT_DONE, done.dump());

        if (r.seq >= 0)
        {
            if (keep_cache && !r.kv_tokens.empty())
            {
                cache_entry e;
                e.seq = r.seq;
                e.tokens.swap(r.kv_tokens);
                e.stamp = ++m.stamp;
                m.idle.push_back(std::move(e));
                // A bounded number of cached sequences, the least recently used dropped.
                const size_t limit = static_cast<size_t>(std::max<long>(4, 2 * m.max_sequences));
                while (m.idle.size() > limit)
                {
                    size_t lru = 0;
                    for (size_t i = 1; i < m.idle.size(); ++i)
                        if (m.idle[i].stamp < m.idle[lru].stamp) lru = i;
                    m.runtime->release_sequence(m.idle[lru].seq);
                    m.idle.erase(m.idle.begin() + static_cast<std::ptrdiff_t>(lru));
                }
            }
            else
            {
                m.runtime->release_sequence(r.seq);
            }
            r.seq = -1;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.erase(r.id);
    }

    void engine::fail(request_state& r, const std::string& code, const std::string& message)
    {
        if (r.ph == request_state::phase::done) return;
        r.ph = request_state::phase::done;
        json e = json::object();
        e["code"] = code;
        e["message"] = message;
        if (r.job) end_job(*r.job, code, message);
        else if (r.sink) r.sink(ALBIREO_EVENT_ERROR, e.dump());
        if (r.seq >= 0)
        {
            std::shared_ptr<loaded_model> m;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = models_.find(r.model_id);
                if (it != models_.end()) m = it->second;
            }
            if (m) m->runtime->release_sequence(r.seq);
            r.seq = -1;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.erase(r.id);
    }

    // Queries

    json engine::stats()
    {
        json s = json::object();
        s["variant"] = ALBIREO_ENGINE_VARIANT;
        s["version"] = ALBIREO_VERSION;
        s["threads"] = threads_;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            const double t = now_seconds();
            // The tokens of the last five seconds: none once the engine waits, rather than a
            // rate that falls slowly after the last pass.
            auto rate = [&](const std::deque<std::pair<double, long>>& w) {
                long total = 0;
                double first = t;
                for (const auto& e : w)
                {
                    if (e.first < t - 5) continue;
                    total += e.second;
                    first = std::min(first, e.first);
                }
                return total > 0 ? total / std::max(1.0, t - first) : 0.0;
            };
            s["generation_tokens_per_second"] = rate(generated_window_);
            s["prompt_tokens_per_second"] = rate(prompt_window_);
            s["generated_total"] = static_cast<unsigned long long>(total_generated_);
            s["prompt_total"] = static_cast<unsigned long long>(total_prompt_);
        }
        json models = json::array();
        long active = 0, waiting = 0;
        std::lock_guard<std::mutex> lock(mutex_);
        s["requests_total"] = static_cast<unsigned long long>(total_requests_);
        for (const auto& kv : models_)
        {
            const loaded_model& m = *kv.second;
            json j = json::object();
            j["id"] = m.id;
            j["active"] = m.n_active.load();
            j["waiting"] = static_cast<long>(m.waiting.size());
            j["cached_sequences"] = m.n_idle.load();
            j["kv_used"] = static_cast<unsigned long long>(m.runtime->kv_bytes_used());
            j["kv_budget"] = static_cast<unsigned long long>(m.runtime->kv_bytes_budget());
            j["generated_total"] = static_cast<unsigned long long>(m.generated.load());
            j["prompt_total"] = static_cast<unsigned long long>(m.prompted.load());
            active += m.n_active.load();
            waiting += static_cast<long>(m.waiting.size());
            models.push_back(j);
        }
        s["active"] = active;
        s["waiting"] = waiting;
        s["models"] = models;
#ifdef ALBIREO_TENSOR_BACKEND
        s["device"] = device_->stats();
        const json memory = device_->memory();
        if (!memory.is_null()) s["memory"] = memory;
#endif
        return s;
    }

    json engine::query(const json& req)
    {
        const std::string op = req["op"].as_string();
        if (op == "stats") return stats();
        if (op == "describe") return describe();
        if (op == "models")
        {
            json list = json::array();
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& kv : models_) list.push_back(model_info(*kv.second));
            json out = json::object();
            out["models"] = list;
            return out;
        }

        std::shared_ptr<loaded_model> m;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = models_.find(req["model"].as_string());
            if (it == models_.end())
                throw std::invalid_argument("the model '" + req["model"].as_string() + "' is not loaded");
            m = it->second;
        }
        json out = json::object();
        if (op == "tokenize")
        {
            const std::vector<int> ids = m->tok.encode(req["text"].as_string(), false, false,
                req["special"].as_bool(false));
            json t = json::array();
            for (int id : ids) t.push_back(id);
            out["tokens"] = t;
            return out;
        }
        if (op == "detokenize")
        {
            std::string text;
            for (const auto& t : req["tokens"].elements()) text += m->tok.token_bytes(static_cast<int>(t.as_int()));
            out["text"] = text;
            return out;
        }
        if (op == "count" || op == "render")
        {
            render_options opt;
            current_dates(opt);
            opt.reasoning = req["reasoning"].as_bool(m->format->supports_reasoning());
            if (req["tools"].is_array()) opt.tools = req["tools"];
            const std::vector<chat_message> messages = parse_messages(req["messages"]);
            std::vector<prompt_piece> pieces;
            m->format->render(messages, opt, pieces);
            const std::vector<int> ids = chat_format::tokenize(m->tok, pieces, m->format->separate_pieces());
            out["tokens"] = static_cast<long>(ids.size());
            out["context"] = m->context;
            if (op == "render")
            {
                std::string text;
                for (const auto& p : pieces) text += p.text;
                out["prompt"] = text;
            }
            return out;
        }
        throw std::invalid_argument("unknown query '" + op + "'");
    }

}
