// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_SETTINGS_H_
#define ALBIREO_HOST_SETTINGS_H_

#include <functional>
#include <map>
#include <mutex>
#include <string>

#include "../common/json.h"

namespace albireo
{

    // The roles a language model can be activated for: one model at most per role.
    extern const char* const model_roles[3];   // "general", "code", "security"
    // The role of the embedding model, which computes the vectors of the document search
    // of the web application; one is always active.
    extern const char* const embedding_role;   // "embedding"

    // The text that stands for the password of the proxy in settings.json and in the answers
    // of the administration interface; settings_store keeps the password itself apart.
    extern const char* const proxy_password_mask;   // "********"
    // The proxy setting with its password, when it has one, replaced by proxy_password_mask.
    std::string masked_proxy(const std::string& proxy);
    // Whether the password of a proxy setting is proxy_password_mask.
    bool proxy_password_masked(const std::string& proxy);

    // The text that stands for the key or the password of the relay in settings.json and in the
    // answers of the administration interface; settings_store keeps the secret itself apart.
    extern const char* const relay_secret_mask;   // "********"

    struct relay_settings
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The relay of the inference (relay.h): the external engine to which the model
                albireo-relay sends the discussions that ask for it, the models of the
                computer staying in place for the others.

                provider is "openai", an engine of the interface of OpenAI at url (its base,
                "https://host/v1", where /chat/completions and /models lie, or the same without
                /v1, which relay.cpp then adds), "ollama", the
                cloud of Ollama, or "claude", the interface of Anthropic: the last two have their
                address and their models in relay.cpp.  auth is "key" (a bearer key, the only
                one of the last two), "basic" (user and secret) or "none".  context, max_output,
                images and reasoning describe the model of an "openai" engine, which the
                program cannot know; relay.cpp knows those of the others.

                pseudonymize sends the texts through Pseudalis, the anonymization service of
                the computer, before they leave (pseudonyms.h), with the tags of its values
                when tags is true and their substitutes alone otherwise.  keep_substitutes
                leaves the substitutes in the answers, without their values: a check of what
                left the computer.
        !*/
        bool enabled = false;
        std::string provider = "openai";        // openai, ollama or claude
        std::string url;                        // openai: the base of its interface
        std::string model;
        std::string auth = "key";               // key, basic or none
        std::string user;                       // basic
        std::string secret;                     // the key or the password, kept in relay.secret
        long context = 131072;                  // openai: the context window of the model
        long max_output = 16384;                // openai: the tokens of an answer at most
        bool images = false;                    // openai: the model reads images
        bool reasoning = false;                 // openai: the model takes reasoning_effort
        bool pseudonymize = false;
        bool tags = false;
        bool keep_substitutes = false;          // the answers keep the substitutes, a check

        // What the secret was given for: a secret never goes to another provider, address or user.
        std::string secret_scope() const;
        json to_json() const;                   // the secret masked
        void from_json(const json& j);
    };

    struct host_settings
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of the service, kept in settings.json in the data folder.
                The tray application changes them through the administration interface,
                and the service applies them, restarting itself when a change requires it
                (the port, the engine, the memory settings).
        !*/
        int port = 8443;
        std::string models_dir;                 // empty: <data>/models
        std::string language = "auto";          // auto, en, fr, de, it, es
        std::string engine = "auto";            // auto, best, or a variant: cpu, cpu-avx2, cpu-avxvnni, ...
        std::string accelerator = "auto";       // auto, none, npu, gpu, both (offloading engines), cpu (graph)
        bool v2ram = true;                      // extended device memory of the tensor engines
        long long v2ram_max_mb = 0;             // 0: automatic
        long long device_mb = 0;                // copies of the weights on the device, 0: automatic
        int threads = 0;                        // 0: physical cores
        long context = 0;                       // 0: automatic per model
        int max_sequences = 4;                  // requests computed together per model
        long long kv_mb = 0;                    // 0: automatic
        std::string kv_type = "auto";           // keys and values: auto (8 bits), q8 or f16
        std::string graph = "off";              // graph of OpenVINO for the models: off, auto or on
        int draft_tokens = 4;                   // tokens of the drafts a step verifies per answer, 0 to 6, 0: none
        bool integer_prompt = true;             // products of the prompts on 8-bit integers
        std::map<std::string, std::string> active;   // role -> model identifier
        bool check_updates = true;
        // A machine without Internet access: no request leaves the machine, the catalog,
        // the updates and the models come from files (import folder, removable media).
        bool offline = false;
        std::string proxy;                      // "": the proxy of the system, "none", or http://host:port
        /*
            The web access of the Web button (web_access.h): the routes through which the web
            application searches the web and reads its pages, turned off by web_access = false.
            web_search names the provider of the search: "wikipedia" or "none".
        */
        bool web_access = true;
        std::string web_search = "wikipedia";
        json engine_benchmarks = json::object();     // variant -> measured tokens per second
        /*
            The choice of the installation for the local authority in the trusted roots of
            the machine: 1 trusted, 0 not, -1 unknown (an installation before 1.0.8), where a
            new authority follows the trust of the one it replaces (certificates.h).
        */
        int trust_authority = -1;
        relay_settings relay;

        /*
            The settings as settings.json and the administration interface give them, the
            password of the proxy masked.  from_json() reads the documents of the earlier
            versions without error: web_search "auto", "brave" or "searxng" becomes
            "wikipedia", and the settings they had that this version lacks (public_link,
            web_search_url, web_search_key, web_search_key_set) are left aside.
        */
        json to_json() const;
        void from_json(const json& j);
        std::string effective_models_dir() const;
    };

    class settings_store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings file, read at start and written atomically at each change,
                with a copy, settings.json.bak, written with it and at each start that reads a
                valid settings.json.  The copy is read in its place when settings.json is no
                longer valid JSON (a disk that failed, an edit by hand), then written again.

                settings.json is readable by the users of the machine, and gives the
                password of the proxy as proxy_password_mask.  The proxy with its password
                is kept in proxy.secret beside it, which only the service and the
                administrators read.  A password found in settings.json, written by an
                earlier version or by hand, moves to proxy.secret at once.  The key or the
                password of the relay goes the same way to relay.secret, encrypted for the
                account of the service on Windows (seal_secret()) with the provider, the
                address and the user it was given for: settings.json that names others reads
                no secret.

                A settings.json of an earlier version that holds settings this version
                lacks, or another provider of the search, is written again at the start
                without them, and web_search.secret, the key of a search provider of those
                versions, is deleted.  The file carries the version of its document,
                settings_version, 3 since Albireo 1.0.23: a document without it, written
                before 1.0.21, reads the graph of OpenVINO "auto", the default of its version,
                as "off", the default since then, unless its accelerator is "cpu", the graph on
                the processor; a document before version 3 reads the 8192 tokens of an answer
                of the relay, the default of 1.0.22, as 16384, the default since then.  Both
                are written again with their version.  A document of a later version keeps
                its version when this one writes it.
        !*/
    public:
        explicit settings_store(const std::string& path);

        host_settings get() const;
        bool set(const host_settings& s);
        /*!
            ensures
                - Applies change to the current settings and writes them, in one step: two
                  changes made at once, by a route and a background task for instance,
                  both persist.
                - Returns the settings written.
        !*/
        host_settings update(const std::function<void(host_settings&)>& change);
        const std::string& path() const { return path_; }

    private:
        // Writes proxy.secret and relay.secret, then settings.json; m_ is held.
        bool write();

        std::string path_;
        mutable std::mutex m_;
        host_settings s_;
        // The version of the document read, which write() keeps when a later version wrote it.
        long long version_ = 0;
    };

}

#endif // ALBIREO_HOST_SETTINGS_H_
