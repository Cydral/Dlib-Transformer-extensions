// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Chat endpoint shared by the inference paths of this library.
//
// There are two ways to run a model here: the shape-dynamic engine, which adapts to a
// container at run time, and a network type compiled from a generated header. They differ
// in how weights are held and how an image reaches the stream, and in nothing else that a
// chat endpoint cares about. Serving them used to mean two copies of the same request
// handling, which is how two endpoints answering the same question start to disagree.
//
// So the endpoint lives here once, parameterized by the engine, and each program supplies
// an adapter. What the adapter must provide is small, and deliberately so:
//
//     void set_context(long capacity, long keep);
//     const tensor& forward_prefill(const std::vector<int>& ids);
//     const tensor& step(int token);
//     bool vision_available() const;
//     long visual_tokens() const;
//     bool stage_image(const matrix<rgb_pixel>& img, std::string& why);
//     void commit_images(const std::vector<long>& positions);
//
// The last two are the only place the two paths genuinely diverge. Staging an image means
// something different on each: the shape-dynamic engine encodes it into vectors right
// away, since its tower sits outside the network, while a compiled model only normalizes
// the pixels, its tower being a layer that will run during the prefill. Committing then
// hands the vectors, or the pixels, to whichever mechanism that engine uses. Everything
// above that line, the splitting of the wire messages, the placement of the image block,
// the location of the reserved positions, the sampling and the streaming, is identical and
// is written once.

#ifndef DLIB_SERVER_CHAT_SERVICE_H_
#define DLIB_SERVER_CHAT_SERVICE_H_

#include "chat_service_abstract.h"

#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "server_chat.h"
#include "../base64.h"
#include "../image_io.h"
#include "../image_transforms.h"
#include "../dnn/text_generation.h"
#include "../tokenizer/chat_template.h"
#include "../tokenizer/hf_tokenizer.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /* Decodes one data URL into an image.

       The interface sends attachments inline rather than by reference, so an image arrives
       base64-encoded inside the JSON, twice removed from a picture: once by the transport
       encoding and once by the file format. Both are undone here.

       Only data URLs are handled. Fetching an http URL would make the server a client of
       whatever address a request names, which is not a capability a chat endpoint should
       acquire by accident. */
    inline bool decode_data_url(const std::string& url, matrix<rgb_pixel>& img,
        std::string& why)
    {
        if (url.compare(0, 5, "data:") != 0)
        { why = "only inline images are accepted, not remote URLs"; return false; }
        const size_t comma = url.find(',');
        if (comma == std::string::npos)
        { why = "the data URL carries no payload"; return false; }
        const std::string head = url.substr(0, comma);
        if (head.find(";base64") == std::string::npos)
        { why = "the data URL is not base64-encoded"; return false; }

        std::string bytes;
        try
        {
            std::istringstream in(url.substr(comma + 1));
            std::ostringstream out;
            dlib::base64().decode(in, out);
            bytes = out.str();
        }
        catch (const std::exception& e)
        { why = std::string("base64: ") + e.what(); return false; }
        if (bytes.size() < 8)
        { why = "the decoded attachment is too short to be an image"; return false; }

        /* The declared media type is a hint from the browser; the magic bytes are what
           the file actually is. Trusting the first over the second turns a mislabelled
           upload into an exception rather than a picture. */
        const unsigned char* p = reinterpret_cast<const unsigned char*>(bytes.data());
        try
        {
            if (p[0] == 0xFF && p[1] == 0xD8)
            {
#ifdef DLIB_JPEG_SUPPORT
                load_jpeg(img, bytes.data(), bytes.size());
                return true;
#else
                why = "this build has no JPEG support; rebuild dlib against libjpeg";
                return false;
#endif
            }
            if (p[0] == 0x89 && p[1] == 'P' && p[2] == 'N' && p[3] == 'G')
            {
#ifdef DLIB_PNG_SUPPORT
                load_png(img, bytes.data(), bytes.size());
                return true;
#else
                why = "this build has no PNG support; rebuild dlib against libpng";
                return false;
#endif
            }
            if (p[0] == 'B' && p[1] == 'M')
            {
                /* No memory loader for this one, so it goes through a temporary file. Rare
                   enough from a browser not to be worth more. */
                const std::string path = "/tmp/dlib_chat_upload.bmp";
                { std::ofstream f(path, std::ios::binary); f.write(bytes.data(), bytes.size()); }
                load_image(img, path);
                return true;
            }
        }
        catch (const std::exception& e)
        { why = std::string("decode: ") + e.what(); return false; }
        why = "the attachment is not a JPEG, a PNG or a BMP";
        return false;
    }

// ----------------------------------------------------------------------------------------

    /* One model served by the endpoint: an engine adapter, its tokenizer and the chat
       template detected for it. Several can be loaded side by side; the request's "model"
       field selects one, the first declared being the default.

       The engine and the tokenizer are held by pointer and owned elsewhere, by whatever
       loaded them, so that a program can keep them in whatever container suits it. */
    template <typename engine_type>
    struct served_model
    {
        std::string name;
        engine_type* engine = nullptr;
        hf_tokenizer* tok = nullptr;
        chat_template_formatter fmt;
    };

// ----------------------------------------------------------------------------------------

    /* The chat endpoint itself.

       One request is handled at a time: the service is stateless and server_chat
       serializes the calls, which matches the single-generation-thread assumption both
       engines make. */
    template <typename engine_type>
    class chat_service : public server_chat
    {
    public:

        chat_service(std::vector<served_model<engine_type>> models, long ctx,
            double forced_temp, bool temp_forced, bool deterministic, bool trace_prompt)
            : models_(std::move(models)), ctx_(ctx), temp_(forced_temp),
              temp_forced_(temp_forced), det_(deterministic), trace_prompt_(trace_prompt)
        {
            std::vector<chat_model_info> infos;
            for (const served_model<engine_type>& m : models_)
                infos.push_back(chat_model_info{ m.name, m.fmt.supports_reasoning() });
            set_models(infos);
        }

    private:

        served_model<engine_type>& select(const std::string& id)
        {
            for (served_model<engine_type>& m : models_)
                if (m.name == id) return m;
            return models_.front();   // first declared model is the default
        }

        /* Sampling resolved per request against the target model's own presets, never
           against another served model's; each request override applies on top. Unset
           overrides arrive negative. */
        sampling_params resolve_sampling(const chat_request& req,
            const chat_template_formatter& fmt) const
        {
            sampling_params sp;
            sp.temperature = req.temperature >= 0.0 ? req.temperature
                : (temp_forced_ ? temp_ : fmt.default_temperature());
            sp.top_k = req.top_k >= 0 ? static_cast<size_t>(req.top_k) : fmt.default_top_k();
            sp.top_p = req.top_p >= 0.0 ? static_cast<float>(req.top_p) : fmt.default_top_p();
            sp.min_p = req.min_p >= 0.0 ? static_cast<float>(req.min_p) : fmt.default_min_p();
            sp.repeat_penalty = req.repeat_penalty >= 0.0
                ? static_cast<float>(req.repeat_penalty) : fmt.default_repeat_penalty();
            sp.greedy = det_ || sp.temperature <= 0.0;
            return sp;
        }

        chat_result on_chat_completion(const chat_request& req,
            const std::function<void(const std::string&)>& emit) override
        {
            served_model<engine_type>& use = select(req.model);
            engine_type& engine = *use.engine;
            hf_tokenizer& tok = *use.tok;
            chat_template_formatter fmt = use.fmt;
            if (req.reasoning >= 0 && fmt.supports_reasoning())
                fmt.set_reasoning(req.reasoning == 1);

            /* Split the wire messages: every system part joins the system block, a user
               message opens a turn, an assistant message closes the turn before it. */
            std::string sys;
            std::vector<chat_turn> turns;
            bool has_image = false;
            for (const chat_message& m : req.messages)
            {
                if (m.role == "system")
                {
                    if (!sys.empty()) sys += "\n";
                    sys += m.content;
                }
                else if (m.role == "user")
                {
                    std::string text = m.content;
                    for (size_t k = 0; k < m.image_urls.size(); ++k)
                    {
                        if (!engine.vision_available())
                        {
                            text += "\n[attached image: not visible to this text-only model]";
                            continue;
                        }
                        /* Only the last user message may carry images: one prefill carries
                           one set of visual positions, and an earlier turn's picture would
                           have to be re-placed for every follow-up. A stateless service can
                           say so rather than pretend. */
                        if (&m != &req.messages.back())
                        {
                            text += "\n[image from an earlier turn: not carried over]";
                            continue;
                        }
                        matrix<rgb_pixel> img;
                        std::string why;
                        if (!decode_data_url(m.image_urls[k], img, why))
                        {
                            text += "\n[attached image ignored: " + why + "]";
                            continue;
                        }
                        if (!engine.stage_image(img, why))
                        {
                            text += "\n[attached image ignored: " + why + "]";
                            continue;
                        }
                        /* The block goes ahead of the text, which is the order the
                           reference template renders and the one the model was trained
                           on. */
                        text = idefics3_markers::image_block(engine.visual_tokens()) + text;
                        has_image = true;
                    }
                    turns.push_back(chat_turn{ text, std::string() });
                }
                else if (m.role == "assistant" && !turns.empty())
                {
                    turns.back().assistant = m.content;
                }
            }
            if (turns.empty())
                throw std::runtime_error("the conversation contains no user message");
            turns.back().assistant.clear();  // the turn being answered carries no reply yet

            const std::vector<int> ids = encode_conversation(tok, fmt, sys, turns);
            const long max_new = req.max_tokens > 0 ? req.max_tokens : 512;
            if (static_cast<long>(ids.size()) + 8 >= ctx_)
                throw std::runtime_error("prompt exceeds the context capacity; reduce the "
                    "context budget in the interface settings");

            const sampling_params sp = resolve_sampling(req, fmt);
            if (trace_prompt_) trace_stream(tok, ids, sp, "prompt to " + use.name);

            /* Same capacity and same pinned prefix as the interactive loop: an eviction
               mid-answer must drop the same rows on every path. */
            engine.set_context(ctx_, system_keep_length(tok, fmt, sys));

            /* Reserved positions are located by scanning for the placeholder rather than by
               counting characters: the tokenizer decides where they land, and a single
               token of drift would put the image on the wrong words. */
            if (has_image)
            {
                const std::vector<int> mark = tok.encode(
                    idefics3_markers::image_placeholder(), false, false, true, false);
                std::vector<long> positions;
                if (mark.size() == 1)
                    for (size_t i = 0; i < ids.size(); ++i)
                        if (ids[i] == mark[0]) positions.push_back(static_cast<long>(i));
                if (static_cast<long>(positions.size()) != engine.visual_tokens())
                    throw std::runtime_error("the reserved image positions do not match "
                        "what the vision tower produces");
                engine.commit_images(positions);
            }

            std::vector<int> recent;
            std::string streamed;
            generation_options opt;
            opt.max_new_tokens = max_new;
            opt.is_cancelled = [&req]() {
                return signal_handler::is_triggered()
                    || (req.is_cancelled && req.is_cancelled());
            };
            /* Stable suffixes only: the client never receives a fragment of the stop marker
               nor a trailing blank that the final answer will have trimmed. An open
               reasoning span still streams raw, so the interface keeps showing the trace
               live; the server's streaming path buffers any incomplete UTF-8 tail on
               top. */
            opt.on_token = [&](const generation_event& ev)
            {
                if (!ev.clean_delta.empty()) emit(ev.clean_delta);
                else if (ev.reasoning_open && ev.answer.size() > streamed.size())
                {
                    emit(ev.answer.substr(streamed.size()));
                    streamed = ev.answer;
                }
            };

            const generation_result gen = generate_reply(engine, tok, fmt,
                engine.forward_prefill(ids), sampler_, sp, recent, opt);

            chat_result res;
            res.prompt_tokens = static_cast<long>(ids.size());
            res.completion_tokens = static_cast<long>(gen.tokens.size());
            res.finish_reason = gen.truncated ? "length" : "stop";
            res.content = gen.text;
            return res;
        }

        /* Declaration order follows the constructor's initializer list. */
        std::vector<served_model<engine_type>> models_;
        long ctx_;
        double temp_;
        bool temp_forced_;
        bool det_;
        bool trace_prompt_;
        token_sampler sampler_;
    };
}

#endif // DLIB_SERVER_CHAT_SERVICE_H_
