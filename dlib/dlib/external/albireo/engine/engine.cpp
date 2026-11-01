// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <list>
#include <stdexcept>

#include "../common/engine_api.h"
#include "../common/system_info.h"
#include "../common/version.h"
#include "chat_format.h"
#include "cpu_runtime.h"
#include "image_input.h"
#include "sampler.h"
#include "thread_pool.h"
#include "vision_encoder.h"
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

        // An integer of a request within [lo, hi], read before any narrowing to a smaller type.
        long long bounded(const json& v, long long def, long long lo, long long hi)
        {
            const long long x = v.as_int(def);
            return x < lo ? lo : x > hi ? hi : x;
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

        /*
            The images of a request, in the order of their parts: the content of a message
            holds, at the place of each image, a mark made of two private-use characters around
            the index of the image, which the prompt builder turns into the tokens of the image.
            The text of the messages loses these two characters beforehand, so that no text can
            pose as an image.
        */
        struct image_part
        {
            std::string url;
            std::string detail;
        };

        const char* const mark_open = "\xEE\x80\x80";     // U+E000
        const char* const mark_close = "\xEE\x80\x81";    // U+E001

        std::string without_marks(const std::string& text)
        {
            if (text.find("\xEE\x80") == std::string::npos) return text;
            std::string out;
            out.reserve(text.size());
            for (size_t i = 0; i < text.size(); ++i)
            {
                if (text.compare(i, 3, mark_open) == 0 || text.compare(i, 3, mark_close) == 0)
                {
                    out += "\xEF\xBF\xBD";
                    i += 2;
                    continue;
                }
                out += text[i];
            }
            return out;
        }

        /*
            The text of a message content: a string, or the parts of a list, text parts as
            they are and image parts as marks when images receives them; a model that reads
            text only (images null) refuses the image and audio parts.
        */
        std::string content_text(const json& c, std::vector<image_part>* images)
        {
            if (c.is_string()) return images ? without_marks(c.as_string()) : c.as_string();
            std::string out;
            for (const auto& part : c.elements())
            {
                const std::string type = part["type"].as_string();
                if (part.is_string()) out += images ? without_marks(part.as_string()) : part.as_string();
                else if (type == "text")
                    out += images ? without_marks(part["text"].as_string()) : part["text"].as_string();
                else if (type == "image_url" && images)
                {
                    const json& u = part["image_url"];
                    image_part ip;
                    ip.url = u.is_string() ? u.as_string() : u["url"].as_string();
                    ip.detail = u.is_string() ? std::string() : u["detail"].as_string();
                    if (ip.url.empty()) throw std::invalid_argument("an image part has no url");
                    out += mark_open + std::to_string(images->size()) + mark_close;
                    images->push_back(ip);
                }
                else if (type == "image_url" || type == "input_audio")
                {
                    throw std::invalid_argument(type == "image_url"
                        ? "this model reads text only; images are not supported"
                        : "this model reads text and images only; audio is not supported");
                }
            }
            return out;
        }

        std::vector<chat_message> parse_messages(const json& list, std::vector<image_part>* images = nullptr)
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
                c.content = content_text(m["content"], images);
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
                    // An object is written as the templates write it, with a space after each separator.
                    d.arguments = f["arguments"].is_string() ? f["arguments"].as_string()
                        : f["arguments"].is_null() ? std::string("{}") : f["arguments"].dump_formatted();
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
        std::vector<int> tokens;    // the keys of the tokens of the sequence (request_state::keys)
        uint64_t stamp = 0;
    };

    // The encoding of an image, kept by the digest of its pixels.
    struct image_encoding
    {
        std::array<uint8_t, 32> digest{};
        std::shared_ptr<const std::vector<float>> features;
    };

    // An image of a request: its size in merged tokens, its pixels until they are encoded, then its encoding.
    struct request_image
    {
        std::array<uint8_t, 32> digest{};
        long grid_w = 0, grid_h = 0, tokens = 0;
        long start = 0;             // index of its first token in the prompt
        vision_encoder::image_plan plan;
        rgb_image pixels;
        std::shared_ptr<const std::vector<float>> features;
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
        // When the model last computed, and whether the memory of its free blocks went back
        // to the system since (engine::trim_idle).
        clock_type::time_point last_step;
        bool trimmed = true;
        forward_batch batch;
        std::vector<float> logits;
        uint64_t parameters = 0;
        uint64_t bytes = 0;
        std::string quantization;
        // The image encoder of a model that reads images, its tokens and its encodings.
        std::shared_ptr<model_file> projector;
        std::unique_ptr<vision_encoder> vision;
        int vision_start = -1, image_pad = -1, vision_end = -1;
        std::list<image_encoding> encodings;            // most recently used first
        size_t encoding_bytes = 0;
        std::unique_ptr<vision_encoder::job> vision_job;
        uint64_t vision_owner = 0;                      // the request whose image is being encoded
        size_t vision_image = 0;
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
        /*
            keys holds the prompt with the tokens of each image made distinct by the digest of
            the image, which the reuse of cached sequences compares; kv_tokens holds the keys of
            the tokens in the keys and values of the sequence.
        */
        std::vector<int> keys;
        std::vector<int> kv_tokens;
        std::vector<request_image> images;
        // Rotary positions of the prompt, three per token, for the models with multimodal positions,
        // and the position of the first token after the prompt.
        std::vector<long> positions;
        long next_position = 0;
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

        // The images of one request, at most.
        const size_t max_request_images = 16;
        // The memory the encodings of the images a model keeps for the following requests may take.
        const size_t encoding_budget = size_t(256) << 20;

        /*
            Reads the images of parts for the encoder of m: their bytes, their size, how the
            encoder reads them with the detail of the part (vision_encoder::plan()) and, with
            decode, their pixels at the size of the plan and the digest of these pixels, which
            for LLaVA-NeXT also names the tiles the image is read by.
        */
        void prepare_images(const loaded_model& m, const std::vector<image_part>& parts, bool decode,
            std::vector<request_image>& out)
        {
            if (parts.size() > max_request_images)
                throw std::invalid_argument("a request holds " + std::to_string(parts.size())
                    + " images; the engine reads at most " + std::to_string(max_request_images));
            const image_limits limits;
            for (const auto& part : parts)
            {
                const std::string bytes = decode_image_data_url(part.url, limits.max_bytes);
                const image_header header = read_image_header(bytes, limits);
                request_image im;
                im.plan = m.vision->plan(header.width, header.height, part.detail);
                im.grid_w = im.plan.grid_w;
                im.grid_h = im.plan.grid_h;
                im.tokens = im.plan.tokens;
                if (decode)
                {
                    im.pixels = decode_image(bytes, limits, im.plan.width, im.plan.height);
                    im.digest = image_digest(im.pixels);
                    if (m.vision->kind() == vision_encoder::family::llava_next)
                    {
                        std::array<uint8_t, 48> named{};
                        std::copy(im.digest.begin(), im.digest.end(), named.begin());
                        for (int b = 0; b < 8; ++b)
                        {
                            named[static_cast<size_t>(32 + b)] = static_cast<uint8_t>(im.plan.tiles_w >> (8 * b));
                            named[static_cast<size_t>(40 + b)] = static_cast<uint8_t>(im.plan.tiles_h >> (8 * b));
                        }
                        im.digest = sha256(named.data(), named.size());
                    }
                }
                out.push_back(std::move(im));
            }
        }

        /*
            The messages of a LLaVA-NeXT model place their images before their text, each one
            followed by a line break, as its chat template writes them.
        */
        void images_first(std::vector<chat_message>& messages)
        {
            for (auto& msg : messages)
            {
                if (msg.content.find(mark_open) == std::string::npos) continue;
                std::string marks, text;
                size_t at = 0;
                for (;;)
                {
                    const size_t open = msg.content.find(mark_open, at);
                    const size_t close = open == std::string::npos ? open : msg.content.find(mark_close, open + 3);
                    if (close == std::string::npos)
                    {
                        text += msg.content.substr(at);
                        break;
                    }
                    text += msg.content.substr(at, open - at);
                    marks += msg.content.substr(open, close + 3 - open) + "\n";
                    at = close + 3;
                }
                msg.content = marks + text;
            }
        }

        /*
            The tokens of a rendered prompt.  A mark of image in the text of a message becomes
            the markup <|vision_start|><|image_pad|><|vision_end|> (Qwen3-VL) or <image>
            (LLaVA-NeXT), so that the text around it is encoded apart, as the reference
            tokenizer splits on these tokens; the image token is then repeated for each token
            of the image, and images[i].start receives the index of its first token.
        */
        std::vector<int> prompt_tokens(const loaded_model& m, const std::vector<prompt_piece>& pieces,
            std::vector<request_image>& images)
        {
            if (images.empty()) return chat_format::tokenize(m.tok, pieces);
            std::vector<prompt_piece> parts;
            std::vector<size_t> order;
            for (const auto& p : pieces)
            {
                if (p.markup || p.text.find(mark_open) == std::string::npos)
                {
                    parts.push_back(p);
                    continue;
                }
                size_t at = 0;
                for (;;)
                {
                    const size_t open = p.text.find(mark_open, at);
                    const size_t close = open == std::string::npos ? open : p.text.find(mark_close, open + 3);
                    prompt_piece text;
                    if (close == std::string::npos)
                    {
                        text.text = p.text.substr(at);
                        if (!text.text.empty()) parts.push_back(text);
                        break;
                    }
                    text.text = p.text.substr(at, open - at);
                    if (!text.text.empty()) parts.push_back(text);
                    const std::string index = p.text.substr(open + 3, close - open - 3);
                    if (index.empty() || index.size() > 3 || index.find_first_not_of("0123456789") != std::string::npos
                        || std::stoul(index) >= images.size())
                        throw std::runtime_error("a mark of image names no image of the request");
                    prompt_piece mark;
                    mark.text = m.vision->kind() == vision_encoder::family::llava_next ? "<image>"
                        : "<|vision_start|><|image_pad|><|vision_end|>";
                    mark.markup = true;
                    parts.push_back(mark);
                    order.push_back(std::stoul(index));
                    at = close + 3;
                }
            }
            if (order.size() != images.size())
                throw std::invalid_argument("the chat format of the model gives no place to an image of this request");
            const std::vector<int> ids = chat_format::tokenize(m.tok, parts);
            std::vector<int> out;
            out.reserve(ids.size());
            size_t k = 0;
            for (int id : ids)
            {
                if (id != m.image_pad)
                {
                    out.push_back(id);
                    continue;
                }
                if (k >= order.size()) throw std::runtime_error("the prompt holds more image tokens than images");
                request_image& im = images[order[k++]];
                im.start = static_cast<long>(out.size());
                out.insert(out.end(), static_cast<size_t>(im.tokens), m.image_pad);
            }
            if (k != order.size()) throw std::runtime_error("an image of the prompt lost its tokens");
            return out;
        }

        /*
            The keys of the prompt of r, and, for a model with multimodal rotary positions, the
            positions of its tokens (the rule of the reference implementation of Qwen3-VL): a
            text token takes the next position in its three components; the token at row y and
            column x of an image starting at position p takes (p, p + y, p + x), and the text
            after the image starts at p plus the larger side of its grid.
        */
        void finish_prompt(const loaded_model& m, request_state& r)
        {
            r.keys = r.prompt;
            for (const request_image& im : r.images)
            {
                uint32_t h = 0;
                for (int b = 0; b < 4; ++b) h = (h << 8) | im.digest[static_cast<size_t>(b)];
                h ^= static_cast<uint32_t>(im.tokens) * 0x9e3779b9u;
                for (long j = 0; j < im.tokens; ++j)
                    r.keys[static_cast<size_t>(im.start + j)] =
                        -1 - static_cast<int>((h + static_cast<uint32_t>(j) * 2654435761u) & 0x7fffffffu);
            }
            r.positions.clear();
            if (!m.cfg.multimodal_rope()) return;
            const long n = static_cast<long>(r.prompt.size());
            r.positions.assign(static_cast<size_t>(3 * n), 0);
            std::vector<const request_image*> by_start;
            for (const request_image& im : r.images) by_start.push_back(&im);
            std::sort(by_start.begin(), by_start.end(),
                [](const request_image* a, const request_image* b) { return a->start < b->start; });
            long p = 0, t = 0;
            size_t next = 0;
            while (t < n)
            {
                if (next < by_start.size() && by_start[next]->start == t)
                {
                    const request_image& im = *by_start[next++];
                    for (long y = 0; y < im.grid_h; ++y)
                        for (long x = 0; x < im.grid_w; ++x)
                        {
                            long* q = &r.positions[static_cast<size_t>(3 * (t + y * im.grid_w + x))];
                            q[0] = p;
                            q[1] = p + y;
                            q[2] = p + x;
                        }
                    t += im.tokens;
                    p += std::max(im.grid_h, im.grid_w);
                    continue;
                }
                long* q = &r.positions[static_cast<size_t>(3 * t)];
                q[0] = q[1] = q[2] = p;
                ++p;
                ++t;
            }
            r.next_position = p;
        }

        // The encoding of the image of digest d that m keeps, made the most recently used, or null.
        std::shared_ptr<const std::vector<float>> kept_encoding(loaded_model& m, const std::array<uint8_t, 32>& d)
        {
            for (auto it = m.encodings.begin(); it != m.encodings.end(); ++it)
                if (it->digest == d)
                {
                    m.encodings.splice(m.encodings.begin(), m.encodings, it);
                    return m.encodings.front().features;
                }
            return nullptr;
        }

        void keep_encoding(loaded_model& m, const std::array<uint8_t, 32>& d,
            const std::shared_ptr<const std::vector<float>>& features)
        {
            image_encoding e;
            e.digest = d;
            e.features = features;
            m.encodings.push_front(e);
            m.encoding_bytes += features->size() * sizeof(float);
            while (m.encoding_bytes > encoding_budget && m.encodings.size() > 1)
            {
                m.encoding_bytes -= m.encodings.back().features->size() * sizeof(float);
                m.encodings.pop_back();
            }
        }
    }

    engine::engine(const json& config, log_sink sink) : settings_(config), log_(std::move(sink))
    {
        const cpu_info cpu = read_cpu_info();
        threads_ = static_cast<int>(settings_["threads"].as_int(0));
        if (threads_ <= 0) threads_ = cpu.physical_cores;
        threads_ = std::max(1, std::min(threads_, cpu.logical_cores));
        pool_.reset(new thread_pool(threads_));
        // A spin of a tenth of a second at most, since a longer one keeps the processor busy between passes.
        pool_->set_spin_time(std::chrono::microseconds(bounded(settings_["spin_us"], 2000, 0, 100000)));
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
        // Requests the worker had no step left to finish.  The destructor throws nothing: an
        // event that cannot be written for lack of memory is dropped.
        try
        {
            std::vector<std::shared_ptr<request_state>> left;
            for (auto& r : requests_) left.push_back(r.second);
            for (auto& r : left)
                if (r->ph != request_state::phase::done) fail(*r, "cancelled", "the engine stopped");
            requests_.clear();
            std::vector<std::shared_ptr<embedding_job>> jobs;
            for (auto& j : jobs_) jobs.push_back(j.second);
            for (auto& j : jobs) end_job(*j, "cancelled", "the engine stopped");
        }
        catch (...)
        {
        }
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
        // since the runtimes of the devices keep threads in the process.  An engine whose
        // kernels cannot run here leaves its devices alone: their runtimes take seconds to
        // start.
        d["backend"] = tensor_device::backend_name();
        d["pinned"] = true;
        if (usable)
        {
            std::string backend, runtime, why, why_code;
            json why_args = json::object();
            const json accelerators = tensor_device::describe(backend, runtime, why, why_code, why_args);
            if (!runtime.empty()) d["runtime"] = runtime;
            for (const auto& a : accelerators.elements()) devices.push_back(a);
            if (accelerators.empty() && backend != "dlib")
            {
                usable = false;
                reason = why;
                reason_code = why_code;
                reason_args = why_args;
            }
            else if (!why.empty())
            {
                d["missing"] = why;
            }
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
        // One load at a time: two loads of the same identifier would otherwise both replace
        // the model, the requests of the first one never ending.
        std::lock_guard<std::mutex> serial(loading_);
        auto m = std::make_shared<loaded_model>();
        m->id = id;
        m->path = path;
        m->file = std::make_shared<model_file>(path);
        m->cfg = read_model_config(*m->file);
        m->tok.load_from_gguf(m->file->reader());
        if (static_cast<long>(m->tok.size()) != m->cfg.vocab)
            throw std::runtime_error("the vocabulary and the output layer of the model disagree");
        // The tokens the engine adds or stops on must be tokens of the vocabulary.
        if (m->tok.bos_id() >= m->cfg.vocab || m->tok.eos_id() >= m->cfg.vocab)
            throw std::runtime_error("the tokenizer of the model names a special token outside its vocabulary");
        m->format = chat_format::detect(m->tok, m->cfg.chat_template);

        // The image encoder: the projector file of the options, or the vision tensors of the model file.
        const std::string projector = options["projector"].as_string();
        if (!projector.empty())
        {
            m->projector = std::make_shared<model_file>(projector);
            if (!vision_encoder::present(*m->projector))
                throw std::invalid_argument("the projector file holds no image encoder");
        }
        else if (vision_encoder::present(*m->file))
        {
            m->projector = m->file;
        }
        if (m->projector)
        {
            m->vision.reset(new vision_encoder(m->projector, *pool_));
            // Qwen3-VL places the tokens of an image by multimodal rotary positions, a LLaVA-NeXT
            // decoder by their order.
            const bool llava = m->vision->kind() == vision_encoder::family::llava_next;
            if (llava == m->cfg.multimodal_rope())
                throw std::invalid_argument(m->cfg.multimodal_rope() || m->cfg.n_deepstack > 0
                    ? "the projector is not the image encoder of this model"
                    : "this model reads text only and takes no projector");
            if (m->vision->output_width() != m->cfg.input_width())
                throw std::invalid_argument("the projector gives " + std::to_string(m->vision->output_width())
                    + " values per image token and the model reads " + std::to_string(m->cfg.input_width())
                    + ": they do not go together");
            if (llava)
            {
                m->image_pad = m->tok.token_id("<image>");
                if (m->image_pad < 0)
                    throw std::invalid_argument("the vocabulary of the model lacks the token <image>");
            }
            else
            {
                m->vision_start = m->tok.token_id("<|vision_start|>");
                m->image_pad = m->tok.token_id("<|image_pad|>");
                m->vision_end = m->tok.token_id("<|vision_end|>");
                if (m->vision_start < 0 || m->image_pad < 0 || m->vision_end < 0)
                    throw std::invalid_argument("the vocabulary of the model lacks the tokens of the images");
            }
        }

        // An embedding model reads passages and questions of a few hundred tokens, many
        // at a time, and keeps nothing of them: its defaults are its own, and its keys and
        // values come in small blocks, given back as soon as each input is read.
        const bool embedding = m->cfg.is_embedding();
        if (embedding && m->cfg.pooling != 3)
            throw std::runtime_error("this embedding model pools its hidden states by a method the engine does not "
                "implement (the engine takes the state of the last token)");
        /*
            The settings of the model, its options or those of the engine, within bounds read
            before any narrowing (long has 32 bits on Windows), which are a context of 2^26
            positions, 256 sequences, passes of 65536 tokens and 2^24 MB of keys and values.
        */
        auto setting = [&](const char* key, long long def, long long lo, long long hi) {
            const json& v = options.contains(key) ? options[key] : settings_[key];
            if (v.is_null()) return def;
            if (!v.is_number()) throw std::invalid_argument(std::string("the setting ") + key + " must be a number");
            const double x = v.as_double();
            if (!(x >= static_cast<double>(lo) && x <= static_cast<double>(hi)))
                throw std::invalid_argument(std::string("the setting ") + key + " must lie between "
                    + std::to_string(lo) + " and " + std::to_string(hi));
            return v.as_int(def);
        };
        long ctx = embedding && !options.contains("context") ? 0
            : static_cast<long>(setting("context", 0, 0, 1 << 26));
        if (ctx <= 0) ctx = std::min<long>(m->cfg.ctx_train, embedding ? 8192 : 32768);
        const long limit = m->cfg.ctx_train > 0 ? std::max<long>(m->cfg.ctx_train, 256) : ctx;
        m->context = std::max<long>(256, std::min<long>(ctx, limit));
        m->max_sequences = embedding && !options.contains("max_sequences") ? 8
            : std::max<long>(1, static_cast<long>(setting("max_sequences", 8, 0, 256)));
        m->batch_tokens = embedding && !options.contains("batch_tokens") ? 256
            : std::max<long>(16, static_cast<long>(setting("batch_tokens", 512, 0, 65536)));

        const size_t per_token = 2 * static_cast<size_t>(m->cfg.n_layers) * m->cfg.kv_dim() * sizeof(uint16_t);
        size_t kv = (embedding && !options.contains("kv_mb") ? size_t(0)
            : static_cast<size_t>(setting("kv_mb", 0, 0, 1 << 24))) << 20;
        if (kv == 0 && embedding)
        {
            // One input of the whole context.
            kv = per_token * static_cast<size_t>(m->context);
        }
        else if (kv == 0)
        {
            /*
                Two full contexts, within a quarter of the memory available once the other
                models have grown to their own budgets: the blocks are allocated as the
                sequences grow, so the memory free now still has to hold that growth.
            */
            const memory_info mem = read_memory_info();
            size_t growth = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const auto& o : models_)
                {
                    if (o.first == id) continue;
                    const size_t budget = o.second->runtime->kv_bytes_budget();
                    const size_t used = o.second->runtime->kv_bytes_used();
                    if (budget > used) growth += budget - used;
                }
            }
            kv = per_token * static_cast<size_t>(m->context) * 2;
            if (mem.available > 0)
            {
                const size_t available = static_cast<size_t>(mem.available);
                kv = std::min<size_t>(kv, (available > growth ? available - growth : 0) / 4);
            }
            kv = std::max<size_t>(kv, per_token * static_cast<size_t>(std::min<long>(m->context, 4096)));
        }
        std::unique_ptr<cpu_runtime> runtime(new cpu_runtime(m->file, m->cfg, *pool_, kv, embedding ? 64 : 256));
#ifdef ALBIREO_TENSOR_BACKEND
        runtime->set_offload(device_->make_offload(id));
#endif
        cpu_runtime* const processor = runtime.get();
        m->runtime = std::move(runtime);

        // The context never exceeds what the keys and values can hold: a prompt the engine
        // accepts can then always be read to its end, and a longer one is refused at once.
        const long room = static_cast<long>(m->runtime->kv_bytes_budget() / m->runtime->kv_bytes_per_token());
        if (room < m->context)
        {
            log(ALBIREO_LOG_INFO, "model '" + id + "': context reduced from " + std::to_string(m->context) + " to "
                + std::to_string(room) + " tokens, the positions its keys and values can hold in the memory given "
                "to them");
            m->context = room;
        }
        // The rotary divisors of a model with LongRoPE follow the context of its sequences.
        processor->set_context(m->context);
        select_rope_factors(m->cfg, m->context);

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
        if (m->vision) log(ALBIREO_LOG_INFO, "model '" + id + "': images read by " + m->vision->description());
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
        json inputs = json::array();
        inputs.push_back("text");
        if (m.vision)
        {
            inputs.push_back("image");
            json t = json::object();
            for (const char* level : { "min", "low", "auto", "high" }) t[level] = m.vision->most_tokens(level);
            j["image_tokens"] = t;
            j["images_per_request"] = static_cast<long>(max_request_images);
            j["projector_bytes"] = static_cast<unsigned long long>(m.projector == m.file ? 0 : m.projector->size());
        }
        j["inputs"] = inputs;
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
            std::vector<image_part> parts;
            std::vector<chat_message> messages = parse_messages(req["messages"], m->vision ? &parts : nullptr);
            if (!parts.empty() && m->vision->kind() == vision_encoder::family::llava_next) images_first(messages);
            std::vector<prompt_piece> pieces;
            m->format->render(messages, opt, pieces);
            if (!parts.empty()) prepare_images(*m, parts, true, r->images);
            r->prompt = prompt_tokens(*m, pieces, r->images);
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
        finish_prompt(*m, *r);
        if (static_cast<long>(r->prompt.size()) >= m->context)
            throw std::invalid_argument("the prompt holds " + std::to_string(r->prompt.size())
                + " tokens, beyond the context of " + std::to_string(m->context) + " tokens of the model");

        // Sampling: the values of the request, or those advised for the family.
        sampling_params sp;
        const bool chatml_reasoning = opt.reasoning && std::string(m->format->name()) == "chatml";
        sp.temperature = static_cast<float>(req["temperature"].as_double(chatml_reasoning ? 0.6 : 0.7));
        sp.top_p = static_cast<float>(req["top_p"].as_double(chatml_reasoning ? 0.95 : 0.9));
        // The candidates are selected in a heap: more than 1024 of them would cost time at every
        // token for a mass that temperature and top-p leave aside.
        sp.top_k = static_cast<int>(bounded(req["top_k"], chatml_reasoning ? 20 : 40, 0, 1024));
        sp.min_p = static_cast<float>(req["min_p"].as_double(0.0));
        const double repeat_penalty = req["repeat_penalty"].as_double(req["repetition_penalty"].as_double(1.0));
        sp.repeat_penalty = static_cast<float>(repeat_penalty);
        sp.repeat_last_n = static_cast<int>(bounded(req["repeat_last_n"], 64, -1, 1 << 20));
        const double presence = req["presence_penalty"].as_double(0.0);
        const double frequency = req["frequency_penalty"].as_double(0.0);
        sp.presence_penalty = static_cast<float>(presence);
        sp.frequency_penalty = static_cast<float>(frequency);
        sp.seed = static_cast<uint64_t>(req["seed"].as_int(0));
        for (const auto& b : req["logit_bias"].items())
        {
            char* end = nullptr;
            const long long token = std::strtoll(b.first.c_str(), &end, 10);
            if (end == b.first.c_str() || *end != '\0' || token < 0 || token >= m->cfg.vocab)
                throw std::invalid_argument("logit_bias holds an invalid token identifier");
            // The range of the OpenAI interface, -100 (banned) to 100 (forced).
            const double bias = b.second.as_double(0.0);
            if (!b.second.is_number() || !(bias >= -100.0 && bias <= 100.0))
                throw std::invalid_argument("logit_bias values must be numbers between -100 and 100");
            sp.logit_bias[static_cast<int>(token)] = static_cast<float>(bias);
        }
        /*
            The JSON numbers may be infinite, so every parameter is checked against a finite
            range, which keeps the logits finite.
        */
        if (!(sp.temperature >= 0 && sp.temperature <= 1e4) || !(sp.top_p >= 0 && sp.top_p <= 1)
            || !(sp.min_p >= 0 && sp.min_p <= 1) || !(repeat_penalty > 0 && repeat_penalty <= 100)
            || !(presence >= -100 && presence <= 100) || !(frequency >= -100 && frequency <= 100))
            throw std::invalid_argument("sampling parameters out of range");
        r->smp.reset(new sampler(sp));

        const long room = m->context - static_cast<long>(r->prompt.size());
        // Compared before any narrowing: long has 32 bits on Windows.
        const long long asked = req["max_completion_tokens"].as_int(req["max_tokens"].as_int(0));
        r->max_tokens = asked <= 0 || asked > room ? room : static_cast<long>(asked);

        if (req["stop"].is_string() && !req["stop"].as_string().empty()) r->stops.push_back(req["stop"].as_string());
        for (const auto& s : req["stop"].elements())
            if (s.is_string() && !s.as_string().empty()) r->stops.push_back(s.as_string());
        if (r->stops.size() > 16) throw std::invalid_argument("at most 16 stop sequences");
        // Each token compares the end of the answer with the stop sequences, whose length is bounded for that.
        for (const auto& s : r->stops)
            if (s.size() > 1024) throw std::invalid_argument("a stop sequence holds 1024 bytes at most");

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
        const long long dimensions = req["dimensions"].as_int(0);
        if (dimensions < 0 || dimensions > m->cfg.d_model)
            throw std::invalid_argument("dimensions must lie between 1 and " + std::to_string(m->cfg.d_model));

        auto job = std::make_shared<embedding_job>();
        job->sink = std::move(sink);
        job->vectors.resize(inputs.size());
        job->remaining = static_cast<long>(inputs.size());
        job->dimensions = static_cast<long>(dimensions);
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
            r->keys = r->prompt;
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
#ifdef ALBIREO_TENSOR_BACKEND
        // CUDA keeps one current device per thread: this thread computes on the card of the
        // engine.
        if (!device_->bind_thread()) log(ALBIREO_LOG_WARNING, "device: the card of the engine cannot be selected");
#endif
        for (;;)
        {
            std::vector<std::shared_ptr<loaded_model>> models;
            // Whether a language model has requests: the embedding models then compute
            // short passes, the answers being what the users wait for.
            bool answering = false;
            std::vector<bool> busy;
            bool woken = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                // Without requests, the thread still wakes now and then to give the memory of
                // the idle models back (trim_idle).
                woken = wake_.wait_for(lock, std::chrono::seconds(5), [&]() {
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
            if (!woken)
            {
                trim_idle(models);
                continue;
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
                    for (auto& r : waiting)
                    {
                        try
                        {
                            fail(*r, "unloaded", "the model was unloaded");
                        }
                        catch (...)
                        {
                            // Out of memory while the event was written: the request ends without it.
                        }
                    }
                    abandon(*m, "unloaded", "the model was unloaded");
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
                catch (const std::bad_alloc&)
                {
                    /*
                        The system ran out of memory during the pass.  Its requests end with the
                        code of a full memory of keys and values, and the free blocks go back to
                        the system, so that the next requests find room.
                    */
                    abandon(*m, "out_of_memory", "the system has no memory left for this request");
                    trim_after_failure(*m);
                }
                catch (const std::exception& e)
                {
                    try
                    {
                        log(ALBIREO_LOG_ERROR, "model '" + m->id + "': " + e.what());
                    }
                    catch (...)
                    {
                    }
                    abandon(*m, "internal", e.what());
                }
                catch (...)
                {
                    abandon(*m, "internal", "unknown error");
                }
            }
        }
    }

    void engine::trim_after_failure(loaded_model& m)
    {
        try
        {
            m.runtime->trim_memory();
        }
        catch (...)
        {
        }
    }

    void engine::abandon(loaded_model& m, const std::string& code, const std::string& message)
    {
        /*
            Ends every request the model computes with an error and gives their sequences
            back.  Writing an event may run out of memory: that request then ends without
            its event, and the others still end.
        */
        for (auto& r : m.active)
        {
            try
            {
                if (r->ph != request_state::phase::done) fail(*r, code, message);
            }
            catch (...)
            {
                r->ph = request_state::phase::done;
            }
            try
            {
                if (r->seq >= 0) m.runtime->release_sequence(r->seq);
            }
            catch (...)
            {
            }
            r->seq = -1;
        }
        m.active.clear();
    }

    void engine::trim_idle(const std::vector<std::shared_ptr<loaded_model>>& models)
    {
        // The free blocks of a language model without requests for a while go back to the
        // system; its cached conversations stay.
        const auto now = clock_type::now();
        for (const auto& m : models)
        {
            if (m->trimmed || m->unloading || !m->active.empty()) continue;
            if (seconds_since(m->last_step, now) < 30) continue;
            m->runtime->trim_memory();
            m->trimmed = true;
        }
    }

    void engine::drop_cancelled(loaded_model& m)
    {
        // The cancelled requests leave the queue at once, whether a slot is free or not.
        std::vector<std::shared_ptr<request_state>> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = m.waiting.begin(); it != m.waiting.end();)
            {
                if ((*it)->is_cancelled())
                {
                    dropped.push_back(*it);
                    it = m.waiting.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        for (auto& r : dropped) finish(m, *r, "cancelled");
    }

    bool engine::preempt(loaded_model& m, const request_state* keep)
    {
        /*
            Gives the memory of a request reading its prompt to the answers being written,
            the request admitted last first: it goes back to the front of the queue
            (requeue_preempted) and reads its prompt again once there is room.
        */
        for (auto it = m.active.rbegin(); it != m.active.rend(); ++it)
        {
            request_state& r = **it;
            if (&r == keep || r.ph != request_state::phase::prefill || r.job || r.seq < 0) continue;
            if (m.runtime->sequence_length(r.seq) == 0) continue;
            m.runtime->release_sequence(r.seq);
            r.seq = -1;
            r.kv_tokens.clear();
            r.cached = 0;
            r.ph = request_state::phase::waiting;
            log(ALBIREO_LOG_DEBUG, "model '" + m.id + "': a prompt being read gave its memory to the answers being "
                "written, and will be read again");
            return true;
        }
        return false;
    }

    void engine::requeue_preempted(loaded_model& m)
    {
        std::vector<std::shared_ptr<request_state>> back;
        for (const auto& r : m.active)
            if (r->ph == request_state::phase::waiting) back.push_back(r);
        if (back.empty()) return;
        // Back in the queue before they leave the active requests, so that a failure of the
        // insertion leaves them where abandon() finds them.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            m.waiting.insert(m.waiting.begin(), back.begin(), back.end());
        }
        m.active.erase(std::remove_if(m.active.begin(), m.active.end(),
            [](const std::shared_ptr<request_state>& r) { return r->ph == request_state::phase::waiting; }),
            m.active.end());
    }

    void engine::admit(loaded_model& m)
    {
        for (;;)
        {
            std::shared_ptr<request_state> r;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (m.waiting.empty() || static_cast<long>(m.active.size()) >= m.max_sequences) return;
                // Room for the request among the active ones before it leaves the queue, so that
                // an exception from here on ends it with the others (abandon).
                m.active.reserve(m.active.size() + 1);
                r = m.waiting.front();
                m.waiting.pop_front();
            }
            m.active.push_back(r);
            if (r->is_cancelled())
            {
                finish(m, *r, "cancelled");
                m.active.pop_back();
                continue;
            }
            r->started = clock_type::now();
            if (!assign_sequence(m, *r))
            {
                fail(*r, "out_of_memory", "no memory left for the keys and values of this request");
                m.active.pop_back();
                continue;
            }
            r->ph = request_state::phase::prefill;
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
            const long n = std::min(limit, common_prefix(m.idle[i].tokens, r.keys));
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
        r.kv_tokens.assign(r.keys.begin(), r.keys.begin() + r.cached);
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

    void engine::advance_encoding(loaded_model& m)
    {
        // The request whose image is being encoded: the encoding stops when it has left.
        request_state* owner = nullptr;
        bool answering = false;
        for (const auto& rp : m.active)
        {
            if (rp->id == m.vision_owner && rp->ph == request_state::phase::prefill) owner = rp.get();
            if (rp->ph == request_state::phase::decode) answering = true;
        }
        if (!owner || m.vision_image >= owner->images.size())
        {
            m.vision_job.reset();
            m.vision->release_work();
            return;
        }
        // A short slice while answers are being written, so that their tokens keep coming.
        if (!m.vision->advance(*m.vision_job, answering ? 0.1 : 0.5)) return;
        const std::shared_ptr<std::vector<float>> out = std::make_shared<std::vector<float>>();
        out->swap(m.vision_job->output());
        const std::shared_ptr<const std::vector<float>> features = out;
        const request_image& image = owner->images[m.vision_image];
        const std::array<uint8_t, 32> digest = image.digest;
        log(ALBIREO_LOG_DEBUG, "model '" + m.id + "': image of " + std::to_string(image.grid_w) + "x"
            + std::to_string(image.grid_h) + " tokens encoded in "
            + std::to_string(static_cast<long>(1000 * m.vision_job->seconds())) + " ms");
        keep_encoding(m, digest, features);
        for (const auto& rp : m.active)
            for (request_image& im : rp->images)
                if (!im.features && im.digest == digest)
                {
                    im.features = features;
                    im.pixels.clear();
                }
        m.vision_job.reset();
        m.vision->release_work();
    }

    bool engine::step(const std::shared_ptr<loaded_model>& mp, long budget_tokens)
    {
        loaded_model& m = *mp;
        for (auto& r : m.active)
            if (r->is_cancelled() && r->ph != request_state::phase::done) finish(m, *r, "cancelled");
        m.active.erase(std::remove_if(m.active.begin(), m.active.end(), request_done), m.active.end());
        drop_cancelled(m);
        admit(m);
        if (m.active.empty())
        {
            m.n_active = 0;
            return false;
        }
        if (m.vision_job) advance_encoding(m);

        // The next token of every request being generated, then slices of the prompts.
        struct row { request_state* r; long count; bool logits; };
        std::vector<row> rows;
        forward_batch& b = m.batch;
        b.clear();
        b.hidden_only = m.cfg.is_embedding();
        const bool multimodal = m.cfg.multimodal_rope();
        long budget = budget_tokens;
        for (auto& rp : m.active)
        {
            request_state& r = *rp;
            if (r.ph != request_state::phase::decode) continue;
            const long len = m.runtime->sequence_length(r.seq);
            if (len + 1 > m.context)
            {
                finish(m, r, "length");
                continue;
            }
            bool reserved = reserve(m, r.seq, len + 1);
            while (!reserved && preempt(m)) reserved = reserve(m, r.seq, len + 1);
            if (!reserved)
            {
                // The answers being written hold the whole memory of the keys and values.
                fail(r, "out_of_memory", "no memory left for the keys and values of this answer");
                continue;
            }
            b.segments.push_back({ r.seq, static_cast<long>(b.tokens.size()), 1 });
            b.logit_rows.push_back(static_cast<long>(b.tokens.size()));
            b.tokens.push_back(r.next_token);
            if (multimodal)
            {
                // A generated token follows the last position of the prompt.
                const long p = r.next_position + len - static_cast<long>(r.prompt.size());
                b.positions.insert(b.positions.end(), { p, p, p });
            }
            rows.push_back({ &r, 1, true });
            --budget;
        }
        requeue_preempted(m);
        for (auto& rp : m.active)
        {
            request_state& r = *rp;
            if (r.ph != request_state::phase::prefill || budget <= 0) continue;
            const long done = static_cast<long>(r.kv_tokens.size());
            long n = std::min<long>(static_cast<long>(r.prompt.size()) - done, budget);
            // The slice stops before the first image it reads that is not encoded yet.
            for (size_t i = 0; i < r.images.size(); ++i)
            {
                request_image& im = r.images[i];
                if (im.features || im.start + im.tokens <= done || im.start >= done + n) continue;
                im.features = kept_encoding(m, im.digest);
                if (im.features)
                {
                    im.pixels.clear();
                    continue;
                }
                if (!m.vision_job)
                {
                    m.vision_job = m.vision->start(im.pixels, im.plan);
                    m.vision_owner = r.id;
                    m.vision_image = i;
                }
                n = std::min<long>(n, std::max<long>(0, im.start - done));
            }
            if (n <= 0) continue;
            if (!reserve(m, r.seq, done + n))
            {
                /*
                    The other requests hold the memory.  While one of them computes in this
                    pass, this prompt waits for room; otherwise the prompts admitted after
                    this one give theirs, so that two prompts never wait for each other.
                */
                bool reserved = false;
                if (rows.empty())
                    while (!reserved && preempt(m, &r)) reserved = reserve(m, r.seq, done + n);
                if (!reserved)
                {
                    if (rows.empty())
                        fail(r, "out_of_memory", "the keys and values of this prompt do not fit in memory");
                    continue;
                }
            }
            const bool last = done + n == static_cast<long>(r.prompt.size());
            const long row0 = static_cast<long>(b.tokens.size());
            b.segments.push_back({ r.seq, row0, n });
            if (last) b.logit_rows.push_back(row0 + n - 1);
            b.tokens.insert(b.tokens.end(), r.prompt.begin() + done, r.prompt.begin() + done + n);
            if (multimodal)
                b.positions.insert(b.positions.end(), r.positions.begin() + 3 * done,
                    r.positions.begin() + 3 * (done + n));
            // The tokens of the images read their encodings: a row of input_width() values each.
            for (const request_image& im : r.images)
            {
                const long from = std::max(im.start, done), to = std::min(im.start + im.tokens, done + n);
                if (from >= to) continue;
                const size_t width = static_cast<size_t>(m.cfg.input_width());
                for (long t = from; t < to; ++t)
                {
                    b.embedding_rows.push_back(row0 + t - done);
                    const float* row = im.features->data() + static_cast<size_t>(t - im.start) * width;
                    b.embeddings.insert(b.embeddings.end(), row, row + width);
                }
            }
            rows.push_back({ &r, n, last });
            budget -= n;
        }
        requeue_preempted(m);
        if (rows.empty())
        {
            m.active.erase(std::remove_if(m.active.begin(), m.active.end(), request_done), m.active.end());
            return false;
        }

        m.runtime->forward(b, m.logits);
        m.last_step = clock_type::now();
        m.trimmed = false;

        long prompt_tokens = 0, generated = 0;
        size_t logit_row = 0;
        const long V = m.cfg.vocab;
        // The rows whose next token is drawn: each request has a sampler of its own, so the
        // draws run in parallel and the tokens are then handled in order.
        struct draw { request_state* r; float* logits; int token; };
        std::vector<draw> draws;
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
                r.kv_tokens.insert(r.kv_tokens.end(), r.keys.begin() + done, r.keys.begin() + done + w.count);
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
            draws.push_back({ &r, lg, -1 });
        }
        if (draws.size() == 1)
            draws[0].token = draws[0].r->smp->sample(draws[0].logits, V);
        else if (draws.size() > 1)
            pool_->parallel_for(static_cast<long>(draws.size()), [&](long i, int) {
                draw& d = draws[static_cast<size_t>(i)];
                d.token = d.r->smp->sample(d.logits, V);
            });
        for (const draw& d : draws)
        {
            ++generated;
            handle_token(m, *d.r, d.token);
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
            for (const auto& t : req["tokens"].elements())
            {
                const long long id = t.as_int(-1);
                if (!t.is_integer() || id < 0 || id >= m->cfg.vocab)
                    throw std::invalid_argument("tokens holds an invalid token identifier");
                text += m->tok.token_bytes(static_cast<int>(id));
            }
            out["text"] = text;
            return out;
        }
        if (op == "count" || op == "render")
        {
            render_options opt;
            current_dates(opt);
            opt.reasoning = req["reasoning"].as_bool(m->format->supports_reasoning());
            if (req["tools"].is_array()) opt.tools = req["tools"];
            std::vector<image_part> parts;
            std::vector<chat_message> messages = parse_messages(req["messages"], m->vision ? &parts : nullptr);
            if (!parts.empty() && m->vision->kind() == vision_encoder::family::llava_next) images_first(messages);
            std::vector<prompt_piece> pieces;
            m->format->render(messages, opt, pieces);
            // The images count by their size, read from their headers, without being decoded.
            std::vector<request_image> images;
            if (!parts.empty()) prepare_images(*m, parts, false, images);
            const std::vector<int> ids = prompt_tokens(*m, pieces, images);
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
