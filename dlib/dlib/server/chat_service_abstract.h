// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SERVER_CHAT_SERVICE_ABSTRACT_H_
#ifdef DLIB_SERVER_CHAT_SERVICE_ABSTRACT_H_

#include <functional>
#include <string>
#include <vector>

#include "server_chat_abstract.h"
#include "../dnn/text_generation_abstract.h"
#include "../tokenizer/chat_template_abstract.h"
#include "../tokenizer/hf_tokenizer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    bool decode_data_url(
        const std::string& url,
        matrix<rgb_pixel>& img,
        std::string& why
    );
    /*!
        ensures
            - Decodes one inline data URL into img and returns true.
            - Returns false and sets why to a sentence naming the reason otherwise.
            - Only data URLs are accepted. Fetching an http URL would make the server a
              client of whatever address a request names, which is not a capability a chat
              endpoint should acquire by accident.
            - The format is decided by the magic bytes rather than by the declared media
              type: the declaration is a hint from the browser, the bytes are what the file
              actually is, and trusting the first turns a mislabelled upload into an
              exception deep in a decoder rather than a clear refusal here.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename engine_type>
    struct served_model
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One model behind the endpoint: an engine adapter, its tokenizer and the chat
                template detected for it.

                The engine and the tokenizer are held by pointer and owned elsewhere, so
                that a program can keep them in whatever container suits it. They must
                outlive the chat_service that names them.
        !*/

        std::string name;
        engine_type* engine;
        hf_tokenizer* tok;
        chat_template_formatter fmt;
    };

// ----------------------------------------------------------------------------------------

    template <typename engine_type>
    class chat_service : public server_chat
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An OpenAI-shaped chat endpoint over one or more models, parameterized by the
                engine that runs them.

                WHY IT IS A TEMPLATE

                    There are two ways to run a model in this library: the shape-dynamic
                    engine, which adapts to a container at run time, and a network type
                    compiled from a generated header. They differ in how weights are held
                    and how an image reaches the stream, and in nothing else an endpoint
                    cares about. Serving them separately meant two copies of the same
                    request handling, which is how two endpoints answering the same question
                    start to disagree.

                WHAT AN ENGINE MUST PROVIDE

                    void set_context(long capacity, long keep);
                    const tensor& forward_prefill(const std::vector<int>& ids);
                    const tensor& step(int token);
                    bool vision_available() const;
                    long visual_tokens() const;
                    bool stage_image(const matrix<rgb_pixel>& img, std::string& why);
                    void commit_images(const std::vector<long>& positions);

                    The last two are the only place the two paths genuinely diverge.
                    Staging means encoding for an engine whose tower sits outside the
                    network, and normalizing pixels for one whose tower is a layer that will
                    run during the prefill.

                THREADING

                    One request is handled at a time: the service is stateless and
                    server_chat serializes the calls, which matches the single generation
                    thread both engines assume.

                LIMITS

                    One image per turn, carried by the last user message. One prefill
                    carries one set of visual positions, and an earlier turn's picture would
                    have to be re-placed for every follow-up.
        !*/

    public:

        chat_service(
            std::vector<served_model<engine_type>> models,
            long ctx,
            double forced_temp,
            bool temp_forced,
            bool deterministic,
            bool trace_prompt
        );
        /*!
            requires
                - models is not empty.
                - Every engine and tokenizer named by models outlives this object.
            ensures
                - Requests naming no model, or an unknown one, are answered by the first
                  model declared.
                - ctx is the KV cache capacity offered to every request.
                - When temp_forced, forced_temp replaces the template's preset for requests
                  that do not carry a temperature of their own.
                - When deterministic, sampling is greedy whatever a request asks.
                - When trace_prompt, the token stream handed to each generation is printed.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SERVER_CHAT_SERVICE_ABSTRACT_H_
