// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Drives an engine library through its C interface, as the host does.

        engine_chat engine_library model.gguf [options] "question" ["question"...]

    Options:
        --reasoning on|off   reasoning of the models that can reason
        --max N              tokens generated at most per answer
        --temperature T      0 for the most likely tokens
        --parallel           submits every question at once instead of one after the other
        --turns              chains the questions as the turns of one conversation
        --tools              offers a weather tool to the model
        --system TEXT        system message of the conversations
        --render FILE        prints the prompt the model receives for the conversation of FILE,
                             a JSON object {"messages", "tools", "reasoning"}, and exits
        --device NAME        device of the OpenVINO engine (NPU, GPU, NPU,GPU) or number of
                             the card of the CUDA engine
        --routing MODE       auto, device (every product on the device) or host
        --threads N, --context N

    Prints the reasoning in brackets, the answer, the tool calls, then the statistics of
    each request.  Returns 0 when every request ended normally.

    With an embedding model, the texts are embedded instead: in one request, or one request
    each with --parallel, the engine batching them either way.  The program prints the
    leading values of each vector and the cosine similarity of every pair, and with
    --reference FILE compares the vectors with those of FILE, the JSON output of another
    implementation ({"data": [{"embedding": [...]}, ...]}), in the same order.
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/engine_loader.h"

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>

#include "../common/platform.h"
#endif

using namespace albireo;

namespace
{
    struct pending
    {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
        bool failed = false;
        std::string content, reasoning, tools, final_json;
        int index = 0;
        bool parallel = false;
    };

    std::mutex print_mutex;

    void ALBIREO_ENGINE_CALL on_event(void* user, int kind, const char* data, size_t size)
    {
        pending& p = *static_cast<pending*>(user);
        const std::string s(data, size);
        std::lock_guard<std::mutex> lock(p.m);
        switch (kind)
        {
        case ALBIREO_EVENT_CONTENT:
            p.content += s;
            if (!p.parallel)
            {
                std::lock_guard<std::mutex> l(print_mutex);
                std::fputs(s.c_str(), stdout);
                std::fflush(stdout);
            }
            break;
        case ALBIREO_EVENT_REASONING:
            p.reasoning += s;
            if (!p.parallel)
            {
                std::lock_guard<std::mutex> l(print_mutex);
                std::printf("\x1b[2m%s\x1b[0m", s.c_str());
                std::fflush(stdout);
            }
            break;
        case ALBIREO_EVENT_TOOL_CALL:
            p.tools += s + "\n";
            break;
        case ALBIREO_EVENT_PROGRESS:
            break;
        case ALBIREO_EVENT_DONE:
        case ALBIREO_EVENT_ERROR:
            p.final_json = s;
            p.failed = kind == ALBIREO_EVENT_ERROR;
            p.done = true;
            p.cv.notify_all();
            break;
        }
    }

    void ALBIREO_ENGINE_CALL on_log(void*, int level, const char* message)
    {
        std::lock_guard<std::mutex> l(print_mutex);
        std::fprintf(stderr, "[engine %d] %s\n", level, message);
    }

    // The floats of the base64 form in which the engine returns the vectors.
    std::vector<float> decode_floats(const std::string& text)
    {
        std::string bytes;
        unsigned value = 0;
        int bits = 0;
        for (char c : text)
        {
            int d = -1;
            if (c >= 'A' && c <= 'Z') d = c - 'A';
            else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
            else if (c >= '0' && c <= '9') d = c - '0' + 52;
            else if (c == '+') d = 62;
            else if (c == '/') d = 63;
            if (d < 0) continue;
            value = (value << 6) | static_cast<unsigned>(d);
            bits += 6;
            if (bits >= 8)
            {
                bits -= 8;
                bytes += static_cast<char>((value >> bits) & 0xFF);
            }
        }
        std::vector<float> v(bytes.size() / sizeof(float));
        if (!v.empty()) std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
        return v;
    }

    double cosine(const std::vector<float>& a, const std::vector<float>& b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        {
            ab += static_cast<double>(a[i]) * b[i];
            aa += static_cast<double>(a[i]) * a[i];
            bb += static_cast<double>(b[i]) * b[i];
        }
        return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
    }

    // Embeds the texts and prints their vectors and similarities; 0 when everything worked.
    int embed(engine_library& lib, albireo_engine* e, const std::vector<std::string>& texts, bool parallel,
        const std::string& reference)
    {
        std::vector<std::vector<float>> vectors(texts.size());
        std::vector<std::unique_ptr<pending>> waits;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t requests = parallel ? texts.size() : 1;
        for (size_t q = 0; q < requests; ++q)
        {
            json req = json::object();
            req["model"] = "m";
            json input = json::array();
            if (parallel) input.push_back(texts[q]);
            else for (const auto& t : texts) input.push_back(t);
            req["input"] = input;
            std::unique_ptr<pending> p(new pending());
            p->parallel = true;
            uint64_t id = 0;
            char err[1024] = { 0 };
            if (lib.submit(e, req.dump().c_str(), on_event, p.get(), &id, err, sizeof(err)) != ALBIREO_OK)
            {
                std::printf("submit: %s\n", err);
                return 1;
            }
            waits.push_back(std::move(p));
        }
        int failures = 0;
        for (size_t q = 0; q < waits.size(); ++q)
        {
            pending& p = *waits[q];
            std::unique_lock<std::mutex> lock(p.m);
            p.cv.wait(lock, [&]() { return p.done; });
            const json done = json::parse_or_null(p.final_json);
            if (p.failed)
            {
                std::printf("error: %s\n", p.final_json.c_str());
                ++failures;
                continue;
            }
            for (const auto& d : done["data"].elements())
                vectors[parallel ? q : static_cast<size_t>(d["index"].as_int())] =
                    decode_floats(d["embedding"].as_string());
            std::printf("usage %s, timings %s\n", done["usage"].dump().c_str(), done["timings"].dump().c_str());
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%zu vectors in %.2f s\n", texts.size(), s);
        for (size_t i = 0; i < vectors.size(); ++i)
        {
            const auto& v = vectors[i];
            std::printf("[%zu] %zu dimensions:", i, v.size());
            for (size_t k = 0; k < v.size() && k < 6; ++k) std::printf(" %.5f", v[k]);
            std::printf("  %s\n", texts[i].substr(0, 60).c_str());
        }
        std::printf("cosine similarities:\n");
        for (size_t i = 0; i < vectors.size(); ++i)
        {
            for (size_t j = 0; j < vectors.size(); ++j) std::printf(" %.4f", cosine(vectors[i], vectors[j]));
            std::printf("\n");
        }
        if (!reference.empty())
        {
            std::FILE* f = std::fopen(reference.c_str(), "rb");
            std::string text;
            char chunk[4096];
            size_t n;
            while (f && (n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) text.append(chunk, n);
            if (f) std::fclose(f);
            const json ref = json::parse_or_null(text);
            double worst = 1.0;
            size_t k = 0;
            for (const auto& d : ref["data"].elements())
            {
                if (k >= vectors.size()) break;
                std::vector<float> r;
                for (const auto& x : d["embedding"].elements()) r.push_back(static_cast<float>(x.as_double()));
                const double c = cosine(vectors[k], r);
                std::printf("reference [%zu]: cosine %.6f\n", k, c);
                worst = std::min(worst, c);
                ++k;
            }
            if (k != vectors.size())
            {
                std::printf("the reference holds %zu vectors for %zu texts\n", k, vectors.size());
                ++failures;
            }
            else if (worst < 0.999)
            {
                std::printf("the vectors differ from the reference\n");
                ++failures;
            }
        }
        return failures ? 1 : 0;
    }
}

int main(int argc, char** argv)
{
#if defined(_WIN32)
    // The paths and the questions in UTF-8, as the engine reads them, and the answers shown
    // as such, whatever the code page of the console.
    std::vector<std::string> utf8_args;
    int wargc = 0;
    if (LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc))
    {
        for (int i = 0; i < wargc; ++i) utf8_args.push_back(narrow(wargv[i]));
        LocalFree(wargv);
    }
    std::vector<char*> utf8_argv;
    for (auto& a : utf8_args) utf8_argv.push_back(&a[0]);
    utf8_argv.push_back(nullptr);
    if (wargc == argc) argv = utf8_argv.data();
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc < 4)
    {
        std::printf("usage: engine_chat engine_library model.gguf [options] question...\n");
        return 2;
    }
    std::string reasoning = "";
    long max_tokens = 256;
    double temperature = 0.0;
    bool parallel = false, turns = false, tools = false;
    int threads = 0;
    long context = 4096;
    std::vector<std::string> questions;
    std::string system, device, routing, reference;
    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--reasoning" && i + 1 < argc) reasoning = argv[++i];
        else if (a == "--max" && i + 1 < argc) max_tokens = std::atol(argv[++i]);
        else if (a == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (a == "--parallel") parallel = true;
        else if (a == "--turns") turns = true;
        else if (a == "--tools") tools = true;
        else if (a == "--system" && i + 1 < argc) system = argv[++i];
        else if (a == "--render" && i + 1 < argc) ++i;
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--context" && i + 1 < argc) context = std::atol(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = argv[++i];
        else if (a == "--routing" && i + 1 < argc) routing = argv[++i];
        else if (a == "--reference" && i + 1 < argc) reference = argv[++i];
        else questions.push_back(a);
    }

    // The OpenVINO engine reads its device from the environment when it first looks for the
    // devices, which describe() below already does.
    if (!device.empty())
    {
#if defined(_WIN32)
        _putenv_s("DLIB_NPU_DEVICE", device.c_str());
#else
        setenv("DLIB_NPU_DEVICE", device.c_str(), 1);
#endif
    }

    engine_library lib;
    std::string error;
    if (!lib.load(argv[1], error))
    {
        std::printf("%s\n", error.c_str());
        return 1;
    }
    bool render_mode = false;
    for (int i = 3; i < argc; ++i) render_mode = render_mode || std::string(argv[i]) == "--render";
    if (!render_mode) std::printf("%s\n", lib.description().dump_formatted().c_str());

    json config = json::object();
    config["threads"] = threads;
    // The device engines: the device (NPU, GPU, NPU,GPU, or the number of a CUDA card) and the
    // routing of the products (auto, device, host).
    if (!device.empty()) config["device"] = device;
    if (!routing.empty()) config["routing"] = routing;
    char err[1024] = { 0 };
    albireo_engine* e = lib.create(config.dump().c_str(), render_mode ? nullptr : on_log, nullptr, err, sizeof(err));
    if (!e)
    {
        std::printf("create: %s\n", err);
        return 1;
    }
    json options = json::object();
    options["context"] = context;
    if (lib.load_model(e, "m", argv[2], options.dump().c_str(), err, sizeof(err)) != ALBIREO_OK)
    {
        std::printf("load: %s\n", err);
        lib.destroy(e);
        return 1;
    }
    json models;
    lib.run_query(e, json::parse_or_null("{\"op\":\"models\"}"), models);
    if (!render_mode) std::printf("%s\n", models.dump_formatted().c_str());
    const json& listed = models;
    if (listed["models"][0]["embedding"].as_bool(false))
    {
        const int rc = embed(lib, e, questions, parallel, reference);
        json stats;
        lib.run_query(e, json::parse_or_null("{\"op\":\"stats\"}"), stats);
        std::printf("\n%s\n", stats.dump_formatted().c_str());
        lib.destroy(e);
        return rc;
    }

    for (int i = 3; i + 1 < argc; ++i)
    {
        if (std::string(argv[i]) != "--render") continue;
        std::FILE* f = std::fopen(argv[i + 1], "rb");
        std::string text;
        char chunk[4096];
        size_t n;
        while (f && (n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) text.append(chunk, n);
        if (f) std::fclose(f);
        json req = json::parse_or_null(text);
        req["op"] = "render";
        req["model"] = "m";
        json answer;
        const int rc = lib.run_query(e, req, answer);
        std::printf("%s", rc == ALBIREO_OK ? answer["prompt"].as_string().c_str() : answer.dump().c_str());
        lib.destroy(e);
        return rc == ALBIREO_OK ? 0 : 1;
    }

    json tool_list = json::parse_or_null(R"([{"type":"function","function":{"name":"get_weather",
        "description":"Current weather of a city","parameters":{"type":"object","properties":{"city":{"type":"string",
        "description":"City name"}},"required":["city"]}}}])");

    int failures = 0;
    json conversation = json::array();
    if (!system.empty())
    {
        json sys = json::object();
        sys["role"] = "system";
        sys["content"] = system;
        conversation.push_back(sys);
    }
    std::vector<std::unique_ptr<pending>> all;
    std::vector<json> requests;
    for (size_t q = 0; q < questions.size(); ++q)
    {
        json msg = json::object();
        msg["role"] = "user";
        msg["content"] = questions[q];
        json req = json::object();
        req["model"] = "m";
        if (turns)
        {
            conversation.push_back(msg);
            req["messages"] = conversation;
        }
        else
        {
            json list = conversation;
            list.push_back(msg);
            req["messages"] = list;
        }
        req["max_tokens"] = max_tokens;
        req["temperature"] = temperature;
        if (!reasoning.empty()) req["reasoning"] = reasoning == "on";
        if (tools) req["tools"] = tool_list;

        std::unique_ptr<pending> p(new pending());
        p->index = static_cast<int>(q);
        p->parallel = parallel;
        uint64_t id = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (lib.submit(e, req.dump().c_str(), on_event, p.get(), &id, err, sizeof(err)) != ALBIREO_OK)
        {
            std::printf("submit: %s\n", err);
            ++failures;
            continue;
        }
        if (!parallel)
        {
            std::printf("\n=== %s\n", questions[q].c_str());
            std::unique_lock<std::mutex> lock(p->m);
            p->cv.wait(lock, [&]() { return p->done; });
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("\n%s--- %s (%.2f s)\n", p->tools.c_str(), p->final_json.c_str(), s);
            if (p->failed) ++failures;
            if (turns)
            {
                json answer = json::object();
                answer["role"] = "assistant";
                answer["content"] = p->content;
                conversation.push_back(answer);
            }
        }
        all.push_back(std::move(p));
    }
    if (parallel)
    {
        for (auto& p : all)
        {
            std::unique_lock<std::mutex> lock(p->m);
            p->cv.wait(lock, [&]() { return p->done; });
            std::printf("\n=== %s\n", questions[static_cast<size_t>(p->index)].c_str());
            if (!p->reasoning.empty()) std::printf("[%s]\n", p->reasoning.c_str());
            std::printf("%s\n%s--- %s\n", p->content.c_str(), p->tools.c_str(), p->final_json.c_str());
            if (p->failed) ++failures;
        }
    }
    json stats;
    lib.run_query(e, json::parse_or_null("{\"op\":\"stats\"}"), stats);
    std::printf("\n%s\n", stats.dump_formatted().c_str());
    lib.destroy(e);
    return failures ? 1 : 0;
}
