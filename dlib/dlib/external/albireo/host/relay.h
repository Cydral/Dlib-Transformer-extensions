// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_RELAY_H_
#define ALBIREO_HOST_RELAY_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/json.h"
#include "pseudonyms.h"
#include "settings.h"

/*
    The relay of the inference: the model albireo-relay of the interface sends a discussion to
    an external engine, which the settings name (relay_settings), and gives its answer back as
    the engine of the computer gives its own.  The service thus serves as a proxy between the
    web application, or any client of its interface, and the external engine, whose address
    and key the clients never see; the other models keep the engine of the computer.

    Three providers are known: "openai", any engine of the interface of OpenAI (the base of its
    interface in the settings, a key, a user and a password, or nothing), "ollama", the cloud of
    Ollama (https://ollama.com/v1, interface of OpenAI, a key), and "claude", the interface of
    Anthropic (https://api.anthropic.com/v1/messages, a key).  The models of the last two are
    known with their context window, the tokens of their answers, their images and their
    reasoning; those of an "openai" engine come from the settings.

    With the pseudonymization, the texts of a request leave through Pseudalis (pseudonyms.h):
    each personal value is replaced by its substitute, or its tag, and the answer gets the
    values back as it streams.  The images then never leave: a model of the computer that reads
    images describes each of them, and the description goes in its place, protected as the
    texts are.  The same happens without pseudonymization when the external model reads no
    image.
*/

namespace albireo
{

    // The model of the interface that relays a discussion to the external engine.
    extern const char* const relay_model_id;   // "albireo-relay"

    struct relay_model
    {
        std::string id;                     // the model as the provider names it
        std::string name;                   // its name for people
        long context = 0;                   // tokens of its context window
        long max_output = 0;                // tokens of an answer at most, its reasoning included
        bool images = false;                // reads images
        /*
            The reasoning: "none", "toggle" (turned on or off by the request), "always" (always
            on), with the levels of effort the model takes, if any, and the level it takes when
            the request gives none.
        */
        std::string thinking = "none";
        std::vector<std::string> efforts;
        std::string default_effort;
    };

    // The models that relay.cpp knows for a provider ("claude", "ollama"), newest first; none for "openai".
    std::vector<relay_model> known_relay_models(const std::string& provider);

    // Whether url can be the base of the interface of an "openai" engine: http:// or https://, no query.
    bool valid_relay_url(const std::string& url);

    // The base of the interface of the provider of r: url for "openai", without its last '/'.
    std::string relay_base_url(const relay_settings& r);

    /*!
        ensures
            - Gives in m the model that r names, with its traits: those relay.cpp knows for
              "claude" and "ollama", those of the settings for "openai".
            - Returns false when r names no model, or a model its provider does not offer here.
    !*/
    bool relay_model_of(const relay_settings& r, relay_model& m);

    /*!
        ensures
            - Returns true when r is enabled and has what a request needs: a known provider, a
              model, a valid address for "openai", and its secret when its authentication asks
              for one.  Otherwise why tells what is missing.
    !*/
    bool relay_ready(const relay_settings& r, std::string& why);

    /*!
        ensures
            - Returns true when the relay of r can answer in the offline mode, which keeps this
              computer and its local network alone within reach: an "openai" engine whose
              address is "localhost" or a name under .localhost, reached at 127.0.0.1, an IP
              address of the local network (loopback, private or link-local), or a name of the
              network, which each request resolves, refuses when one of its addresses lies
              elsewhere, and reaches at an address it checked.  The providers "ollama" and
              "claude" need the Internet.  Otherwise why tells it.
    !*/
    bool relay_offline_ready(const relay_settings& r, std::string& why);

    // {"providers": [{"id", "name", "url", "auths", "models": [...]}]}: what the tray offers.
    json relay_catalog();

    // The model of r and its traits as the interface gives them, without the address and the secret.
    json relay_description(const relay_settings& r);

    class relay_service
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The answers of the external engine: each request runs on a thread of its own,
                max_jobs at most at once, and gives its events as the engine of the computer
                does (common/engine_api.h): the text, the reasoning, then the end of the
                answer with its usage, or its error.  The secret goes in the headers of the
                requests alone, and never in a log, an answer or an error.

                The requests follow the proxy and the cancellation of the downloads
                (http_fetch.h), and the stop of the service ends them.  In the offline mode,
                which cancels the downloads, the relay reaches an "openai" engine of this
                computer or of its local network alone, directly, without the proxy: each
                address of its name is checked before any connection, which goes to one of
                them (relay_offline_ready()).
        !*/
    public:
        typedef std::function<void(int kind, const std::string& data)> sink_type;
        typedef std::function<host_settings()> settings_source;
        /*
            Describes an image, a data URL, with a model of the computer that reads images:
            returns false with error when none can, or once cancelled turns true.
        */
        typedef std::function<bool(const std::string& data_url, const std::atomic<bool>& cancelled,
            std::string& description, std::string& error)> image_describer;

        relay_service(settings_source settings, image_describer describe);
        ~relay_service();
        relay_service(const relay_service&) = delete;
        relay_service& operator=(const relay_service&) = delete;

        /*!
            requires
                - request is the body of a chat completion of the interface of OpenAI, its
                  messages checked by the server (check_media()).
            ensures
                - Starts the answer of request with the external engine of the settings; its
                  events go to sink, from another thread, until the end or the error.  The
                  sampling passes as numbers alone, and the stop sequences only without
                  pseudonymization; the answers of Claude take 64000 tokens at most when the
                  request gives no bound.
                - Returns false, with error, the HTTP status and the code of the refusal, when
                  the relay refuses the request at once: turned off or incomplete, an engine
                  out of reach of the offline mode, too many answers in progress, tools, or
                  several choices.
        !*/
        bool submit(const json& request, sink_type sink, uint64_t& id, std::string& error, int& status,
            std::string& code);

        // Ends the answer id, whose last event follows.
        void cancel(uint64_t id);

        // Ends the answers in progress and waits for their threads; submit() refuses afterwards.
        void stop();

        /*!
            ensures
                - Asks the external engine of r for its models, which tells whether its address
                  and its secret are right: {"ok", "models": [...]}, with "url" when the
                  interface of an "openai" engine answers at another base than the address
                  typed (with /v1, or without the path of an operation), or {"ok": false,
                  "error", "code"}, the code naming the failure for the tray ("relay_address",
                  "relay_http_key", ...), with "args": {"s": status} and "detail", the message
                  of the engine, for an HTTP status.  The settings of r stand as they are, the
                  mask of the secret excepted, which the caller replaces by the stored secret.
        !*/
        json test(const relay_settings& r);

        // The state of the pseudonymization: {"found", "reason", "memory": {...}}.
        json pseudonymization_status();

        // Forgets the values kept by the pseudonymization (pseudonym_memory) and the descriptions of the images.
        void forget_pseudonyms();

        static const size_t max_jobs = 8;

    private:
        struct job;
        void run(const std::shared_ptr<job>& j);
        /*
            The base of the interface of the provider of r: for "openai", the address typed, or
            the same with /v1 when only there a list of models answers, found once per address
            and kept; an engine that answers without listing its models is asked again after
            unlisted_seconds, one out of reach at the next request.  The connection of the
            search is held by control.
        */
        std::string base_of(const relay_settings& r, cancellable* control);
        // Keeps base for the address typed, until the time until (Unix seconds), 0 for as long as the service runs.
        void remember_base(const std::string& typed, const std::string& base, int64_t until);
        // Joins the threads of the answers that ended; jobs_m_ is held.
        void reap();

        settings_source settings_;
        image_describer describe_;
        pseudalis_client pseudalis_;
        pseudonym_memory memory_;

        std::mutex jobs_m_;
        std::list<std::shared_ptr<job>> jobs_;
        uint64_t next_id_ = 1;
        bool stopped_ = false;

        // The bases found for the addresses of "openai" engines, max_bases at most, with the end of their use.
        struct known_base
        {
            std::string base;
            int64_t until = 0;
        };
        std::mutex bases_m_;
        std::map<std::string, known_base> bases_;
        static const size_t max_bases = 32;
        static const int unlisted_seconds = 600;

        // The descriptions of the images by the SHA-256 of their data URL, max_descriptions at most.
        std::mutex descriptions_m_;
        std::list<std::pair<std::string, std::string>> descriptions_;
        static const size_t max_descriptions = 64;
    };

    /*!
        Reserved for the tests (tests/test_host.cpp).  No setting and no route reaches it.
        ensures
            - Sends the requests of the provider "claude" or "ollama" to origin ("http://127.0.0.1:port"),
              where a test server answers as the provider; an empty origin gives them their address back.
    !*/
    void relay_origin_for_tests(const std::string& provider, const std::string& origin);

    /*!
        Reserved for the tests (tests/test_host.cpp).  No setting and no route reaches it.
        ensures
            - The offline mode resolves name, in lowercase, to addresses rather than through the
              resolver of the system; empty addresses give the name back to the resolver.
    !*/
    void relay_name_for_tests(const std::string& name, const std::vector<std::string>& addresses);

}

#endif // ALBIREO_HOST_RELAY_H_
