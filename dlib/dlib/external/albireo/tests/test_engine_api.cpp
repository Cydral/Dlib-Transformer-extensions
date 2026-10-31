// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks an engine library through its C interface, on a tiny model it writes itself.

        test_engine_api engine_library [folder]

    The invalid arguments and documents of every function must be refused with their code,
    the settings and the sampling parameters beyond their ranges too.  Then several threads
    submit and cancel requests while the model is unloaded and loaded again, the memory of
    the keys and values runs short, and the engine is destroyed with requests running: every
    request accepted must receive exactly one final event, and nothing after it.  Returns 0
    when every check passes, and when the processor lacks the instructions of the engine.
*/

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../common/engine_loader.h"
#include "tiny_gguf.h"

using namespace albireo;

namespace
{
    int failures = 0;
    std::mutex print_mutex;

    void expect(bool ok, const std::string& what)
    {
        if (!ok)
        {
            std::lock_guard<std::mutex> lock(print_mutex);
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    // The events of one request.
    struct tracker
    {
        std::mutex m;
        std::condition_variable cv;
        int finals = 0;
        int after_final = 0;
        int kind = 0;
        std::string final_json;
    };

    void ALBIREO_ENGINE_CALL on_event(void* user, int kind, const char* data, size_t size)
    {
        tracker& t = *static_cast<tracker*>(user);
        std::lock_guard<std::mutex> lock(t.m);
        if (t.finals > 0) ++t.after_final;
        if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR)
        {
            ++t.finals;
            t.kind = kind;
            t.final_json.assign(data, size);
            t.cv.notify_all();
        }
    }

    bool wait_final(tracker& t, int seconds)
    {
        std::unique_lock<std::mutex> lock(t.m);
        return t.cv.wait_for(lock, std::chrono::seconds(seconds), [&]() { return t.finals > 0; });
    }

    std::string chat(const std::string& text, const std::string& extra = "")
    {
        return "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"" + text + "\"}]" + extra + "}";
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_engine_api engine_library [folder]\n");
        return 2;
    }
    const std::string folder = argc > 2 ? std::string(argv[2]) + "/" : std::string();
    const std::string path = folder + "test_engine_api.gguf";
    const std::string damaged_path = folder + "test_engine_api_damaged.gguf";
    const albireo_test::tiny_gguf model = albireo_test::tiny_gguf_writer().build();
    if (!model.save(path, model.bytes)
        || !model.save(damaged_path, model.with("kv tokenizer.ggml.eos_token_id", 1u << 20)))
    {
        std::printf("cannot write %s\n", path.c_str());
        return 2;
    }

    engine_library lib;
    std::string error;
    if (!lib.load(argv[1], error))
    {
        std::printf("cannot load %s: %s\n", argv[1], error.c_str());
        return 2;
    }
    const json description = lib.description();
    if (!description["usable"].as_bool(false))
    {
        std::printf("albireo engine interface: skipped, %s\n", description["reason"].as_string().c_str());
        return 0;
    }
    std::printf("albireo engine interface (%s)\n", description["variant"].as_string().c_str());

    // Arguments and documents.
    char err[1024] = { 0 };
    expect(lib.create("{\"threads\": ", nullptr, nullptr, err, sizeof(err)) == nullptr && err[0] != 0,
        "an invalid configuration is refused with a message");
    albireo_engine* e = lib.create("{\"threads\": 2, \"spin_us\": 1000000000}", nullptr, nullptr, err, sizeof(err));
    expect(e != nullptr, std::string("engine created: ") + err);
    if (!e) return 1;
    expect(lib.load_model(nullptr, "m", path.c_str(), nullptr, err, sizeof(err)) == ALBIREO_E_INVALID_ARGUMENT,
        "load without an engine");
    expect(lib.load_model(e, nullptr, path.c_str(), nullptr, err, sizeof(err)) == ALBIREO_E_INVALID_ARGUMENT,
        "load without an identifier");
    for (const char* options : { "{\"kv_mb\": -1}", "{\"max_sequences\": 100000}", "{\"context\": -5}",
        "{\"batch_tokens\": 1e999}", "{\"kv_mb\": \"lots\"}", "{\"context\": " })
        expect(lib.load_model(e, "m", path.c_str(), options, err, sizeof(err)) == ALBIREO_E_INVALID_ARGUMENT,
            std::string("load options refused: ") + options);
    expect(lib.load_model(e, "m", (folder + "missing.gguf").c_str(), nullptr, err, sizeof(err)) != ALBIREO_OK,
        "a missing file is refused");
    expect(lib.load_model(e, "m", damaged_path.c_str(), nullptr, err, sizeof(err)) != ALBIREO_OK,
        "a special token outside the vocabulary is refused");
    const char* options = "{\"context\": 1024, \"max_sequences\": 16, \"kv_mb\": 1}";
    expect(lib.load_model(e, "m", path.c_str(), options, err, sizeof(err)) == ALBIREO_OK,
        std::string("the model loads: ") + err);

    size_t needed = 0;
    expect(lib.query(e, "{\"op\":\"stats\"}", nullptr, 0, &needed) == ALBIREO_E_BUFFER_TOO_SMALL && needed > 1,
        "a query without a buffer gives its size");
    char answer[4096];
    expect(lib.query(e, "[1, 2]", answer, sizeof(answer), &needed) == ALBIREO_E_INVALID_ARGUMENT,
        "a query that is no object");
    expect(lib.query(nullptr, "{\"op\":\"stats\"}", answer, sizeof(answer), &needed) == ALBIREO_E_INVALID_ARGUMENT,
        "a query without an engine");

    tracker unused;
    uint64_t id = 0;
    expect(lib.submit(e, chat("hello").c_str(), nullptr, nullptr, &id, err, sizeof(err)) == ALBIREO_E_INVALID_ARGUMENT,
        "a request without a callback");
    const std::string long_stop(2000, 'x');
    const std::string refused[] = { "{\"model\":", "[]", "{\"model\":\"other\"}",
        chat("hello", ",\"temperature\":1e999"),
        chat("hello", ",\"repeat_penalty\":0"), chat("hello", ",\"presence_penalty\":-1e999"),
        chat("hello", ",\"logit_bias\":{\"5\":1e999}"), chat("hello", ",\"logit_bias\":{\"5\":\"high\"}"),
        chat("hello", ",\"top_p\":2"), chat("hello", ",\"stop\":[\"" + long_stop + "\"]"),
        chat("hello", ",\"logit_bias\":{\"999999\":1}"), "{\"model\":\"m\",\"messages\":[{\"role\":\"robot\"}]}" };
    for (const auto& r : refused)
        expect(lib.submit(e, r.c_str(), &on_event, &unused, &id, err, sizeof(err)) == ALBIREO_E_INVALID_ARGUMENT,
            "request refused: " + r.substr(0, 80));
    const std::string too_long = chat(std::string(5000, 'a'));
    expect(lib.submit(e, too_long.c_str(), &on_event, &unused, &id, err, sizeof(err)) == ALBIREO_E_CONTEXT_OVERFLOW,
        "a prompt beyond the context");
    expect(lib.cancel(e, 123456789) == ALBIREO_OK, "cancel of an unknown request");
    std::printf("  arguments, settings and parameters out of range refused\n");

    /*
        Three threads submit 30 requests each and cancel a third of them.  The keys and values
        of 1 MB hold 16 blocks of 256 positions for 16 sequences of up to 4 blocks: the answers
        run out of memory and the prompts give theirs back.  The first round runs to its end,
        the second one meets the unloading of the model, which is then loaded again.
    */
    for (int round = 0; round < 2; ++round)
    {
        std::vector<std::unique_ptr<tracker>> accepted;
        std::mutex accepted_mutex;
        std::vector<std::thread> clients;
        for (int c = 0; c < 3; ++c)
        {
            clients.emplace_back([&, c]() {
                std::mt19937 rng(static_cast<unsigned>(c + 1 + 10 * round));
                for (int i = 0; i < 30; ++i)
                {
                    std::unique_ptr<tracker> t(new tracker());
                    const int max_tokens = 1 + static_cast<int>(rng() % 150);
                    std::string extra = ",\"max_tokens\":" + std::to_string(max_tokens) + ",\"seed\":"
                        + std::to_string(i);
                    if (i % 4 == 1) extra += ",\"stop\":[\"ab\"]";
                    if (i % 5 == 2) extra += ",\"temperature\":0";
                    const std::string text(static_cast<size_t>(200 + rng() % 600), static_cast<char>('a' + c));
                    const std::string request = chat(text, extra);
                    uint64_t rid = 0;
                    char e2[512];
                    const int rc = lib.submit(e, request.c_str(), &on_event, t.get(), &rid, e2, sizeof(e2));
                    if (rc == ALBIREO_OK)
                    {
                        if (rng() % 3 == 0)
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(rng() % 20));
                            lib.cancel(e, rid);
                        }
                        std::lock_guard<std::mutex> lock(accepted_mutex);
                        accepted.push_back(std::move(t));
                    }
                    else
                    {
                        // Refused while the model was unloaded.
                        expect(rc == ALBIREO_E_INVALID_ARGUMENT, std::string("refusal of a request: ") + e2);
                    }
                }
            });
        }
        if (round == 1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            expect(lib.unload_model(e, "m") == ALBIREO_OK, "unload during the requests");
            expect(lib.load_model(e, "m", path.c_str(), options, err, sizeof(err)) == ALBIREO_OK, "load again");
        }
        for (auto& t : clients) t.join();
        long missing = 0;
        std::map<std::string, long> endings;
        for (auto& t : accepted)
        {
            if (!wait_final(*t, 120))
            {
                ++missing;
                continue;
            }
            std::lock_guard<std::mutex> lock(t->m);
            expect(t->finals == 1 && t->after_final == 0, "one final event and nothing after it");
            const json f = json::parse_or_null(t->final_json);
            ++endings[t->kind == ALBIREO_EVENT_DONE ? f["finish_reason"].as_string()
                : "error " + f["code"].as_string()];
        }
        expect(missing == 0, std::to_string(missing) + " requests never ended");
        std::string summary;
        for (const auto& x : endings)
            summary += (summary.empty() ? "" : ", ") + std::to_string(x.second) + " " + x.first;
        std::printf("  %zu requests from 3 threads with cancellations%s: %s\n", accepted.size(),
            round == 1 ? " and a reload" : "", summary.c_str());
    }

    // The engine destroyed with requests running: each one ends once, before destroy returns.
    std::vector<std::unique_ptr<tracker>> running;
    for (int i = 0; i < 6; ++i)
    {
        std::unique_ptr<tracker> t(new tracker());
        const std::string request = chat(std::string(50, static_cast<char>('k' + i)), ",\"max_tokens\":400");
        if (lib.submit(e, request.c_str(), &on_event, t.get(), &id, err, sizeof(err)) == ALBIREO_OK)
            running.push_back(std::move(t));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    lib.destroy(e);
    for (auto& t : running)
    {
        std::lock_guard<std::mutex> lock(t->m);
        expect(t->finals == 1, "a request running at the destruction ends once");
    }
    std::printf("  %zu requests running at the destruction of the engine ended\n", running.size());

    std::remove(path.c_str());
    std::remove(damaged_path.c_str());
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
