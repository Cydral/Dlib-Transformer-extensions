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

    // The texts of the answer and of the reasoning of one request, and its final event.
    struct collector
    {
        std::mutex m;
        std::condition_variable cv;
        std::string content;
        std::string reasoning;
        std::string final_json;
        int finals = 0;
    };

    void ALBIREO_ENGINE_CALL collect_event(void* user, int kind, const char* data, size_t size)
    {
        collector& c = *static_cast<collector*>(user);
        std::lock_guard<std::mutex> lock(c.m);
        if (kind == ALBIREO_EVENT_CONTENT) c.content.append(data, size);
        else if (kind == ALBIREO_EVENT_REASONING) c.reasoning.append(data, size);
        else if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR)
        {
            ++c.finals;
            c.final_json.assign(data, size);
            c.cv.notify_all();
        }
    }

    bool wait_collected(collector& c, int seconds)
    {
        std::unique_lock<std::mutex> lock(c.m);
        return c.cv.wait_for(lock, std::chrono::seconds(seconds), [&]() { return c.finals > 0; });
    }

    // The text without the white space at its ends.
    std::string trimmed(const std::string& s)
    {
        const size_t a = s.find_first_not_of(" \n");
        if (a == std::string::npos) return std::string();
        return s.substr(a, s.find_last_not_of(" \n") - a + 1);
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
        "{\"batch_tokens\": 1e999}", "{\"kv_mb\": \"lots\"}", "{\"context\": ", "{\"draft_tokens\": 7}",
        "{\"draft_tokens\": -1}" })
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
        A request whose reasoning and tool calls carry the private-use characters U+E000 and
        U+E001 around a digit, the marks the engine places images with: a model that reads
        images strips them from every rendered string so that no field can pose as an image,
        and a text model takes them as plain text.  Either way the request runs to its end and
        receives one final event, rather than faulting on a forged placeholder.
    */
    {
        const std::string e0 = "\xEE\x80\x80";   // U+E000
        const std::string e1 = "\xEE\x80\x81";   // U+E001
        const std::string mark = e0 + "0" + e1;
        json messages = json::array();
        json user = json::object();
        user["role"] = "user";
        user["content"] = "describe " + mark + " this";
        messages.push_back(user);
        json assistant = json::object();
        assistant["role"] = "assistant";
        assistant["content"] = "";
        assistant["reasoning"] = "think " + mark + " again";
        json call = json::object();
        call["id"] = "call" + mark;
        json fn = json::object();
        fn["name"] = "tool" + mark;
        fn["arguments"] = "{\"city\": \"" + mark + "\"}";
        call["function"] = fn;
        json calls = json::array();
        calls.push_back(call);
        assistant["tool_calls"] = calls;
        messages.push_back(assistant);
        json again = json::object();
        again["role"] = "user";
        again["content"] = "and " + mark + " now";
        messages.push_back(again);
        json req = json::object();
        req["model"] = "m";
        req["messages"] = messages;
        req["max_tokens"] = 4;
        tracker t;
        uint64_t rid = 0;
        const int rc = lib.submit(e, req.dump().c_str(), &on_event, &t, &rid, err, sizeof(err));
        expect(rc == ALBIREO_OK, std::string("a request with forged image marks is accepted: ") + err);
        if (rc == ALBIREO_OK)
        {
            expect(wait_final(t, 60), "the forged-mark request ends");
            std::lock_guard<std::mutex> lock(t.m);
            expect(t.finals == 1 && t.after_final == 0 && t.kind == ALBIREO_EVENT_DONE,
                "the forged-mark request ends once, without a fault");
        }
        std::printf("  forged image marks in the reasoning and the tool calls handled\n");
    }

    /*
        The history of an answer read ahead of the next question.  The answer, 40 letters
        among a, b, c and d (every other token banned), ends at its length; the engine then
        reads it as the next request renders it, without the empty reasoning that opened it.
        The next request finds those tokens cached and computes its question only: its
        uncached tokens are those of "more" and of the turns around it, fewer than 30, where
        the answer and its end would add 48 more.
    */
    {
        // The size of the vocabulary: the first identifier the model cannot detokenize.
        int vocab = 0;
        for (;; ++vocab)
        {
            json answer_json;
            const std::string q = "{\"op\":\"detokenize\",\"model\":\"m\",\"tokens\":[" + std::to_string(vocab) + "]}";
            if (lib.run_query(e, json::parse_or_null(q), answer_json) != ALBIREO_OK) break;
        }
        json bias = json::object();
        for (int t = 0; t < vocab; ++t)
            if (t < 97 || t > 100) bias[std::to_string(t)] = -100;
        auto ask = [&](const json& messages, int max_tokens, tracker& t, uint64_t& rid) {
            json req = json::object();
            req["model"] = "m";
            req["messages"] = messages;
            req["max_tokens"] = max_tokens;
            req["temperature"] = 0;
            req["reasoning"] = false;
            req["logit_bias"] = bias;
            return lib.submit(e, req.dump().c_str(), &on_event, &t, &rid, err, sizeof(err));
        };
        auto content = [](const std::string& role, const std::string& text) {
            json m = json::object();
            m["role"] = role;
            m["content"] = text;
            return m;
        };
        // The tokens the warm-ups read so far, and the warm-ups not ended.
        auto warmed_total = [&](long long& warming) {
            json stats;
            lib.run_query(e, json::parse_or_null("{\"op\":\"stats\"}"), stats);
            const json& s = stats;
            warming = s["models"][0]["warming"].as_int(1);
            return s["models"][0]["warmed_total"].as_int(0);
        };
        long long warming = 0;
        const long long before = warmed_total(warming);
        json messages = json::array();
        messages.push_back(content("user", "tell me"));
        tracker first;
        uint64_t rid = 0;
        std::string answer;
        struct collector
        {
            tracker* t;
            std::string* text;
            static void ALBIREO_ENGINE_CALL on(void* user, int kind, const char* data, size_t size)
            {
                collector& c = *static_cast<collector*>(user);
                if (kind == ALBIREO_EVENT_CONTENT)
                {
                    std::lock_guard<std::mutex> lock(c.t->m);
                    c.text->append(data, size);
                }
                on_event(c.t, kind, data, size);
            }
        } collect{ &first, &answer };
        json req = json::object();
        req["model"] = "m";
        req["messages"] = messages;
        req["max_tokens"] = 40;
        req["temperature"] = 0;
        req["reasoning"] = false;
        req["logit_bias"] = bias;
        expect(lib.submit(e, req.dump().c_str(), &collector::on, &collect, &rid, err, sizeof(err)) == ALBIREO_OK,
            std::string("the first question of a conversation is accepted: ") + err);
        expect(wait_final(first, 60), "the first answer ends");
        expect(answer.size() == 40 && answer.find_first_not_of("abcd") == std::string::npos,
            "the first answer holds 40 letters a to d: " + answer);
        // The warm-up reads in the background, and ends with the history.
        long long warmed = 0;
        for (int i = 0; i < 300; ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            warmed = warmed_total(warming) - before;
            if (warmed >= 40 && warming == 0) break;
        }
        expect(warmed >= 40, "the history of the answer is read ahead (" + std::to_string(warmed) + " tokens)");
        messages.push_back(content("assistant", answer));
        messages.push_back(content("user", "more"));
        tracker second;
        expect(ask(messages, 1, second, rid) == ALBIREO_OK, std::string("the next question is accepted: ") + err);
        expect(wait_final(second, 60), "the next answer ends");
        {
            std::lock_guard<std::mutex> lock(second.m);
            const json f = json::parse_or_null(second.final_json);
            const long long prompt = f["usage"]["prompt_tokens"].as_int(0);
            const long long cached = f["usage"]["cached_tokens"].as_int(0);
            expect(prompt > 48 && prompt - cached < 30, "the next question finds its history cached ("
                + std::to_string(cached) + " of " + std::to_string(prompt) + " tokens)");
        }
        // A question that follows its answer at once takes over the warm-up, and ends once.
        messages.push_back(content("assistant", "a"));
        messages.push_back(content("user", "again"));
        tracker third, fourth;
        expect(ask(messages, 30, third, rid) == ALBIREO_OK, "a third question is accepted");
        expect(wait_final(third, 60), "the third answer ends");
        messages.push_back(content("user", "and again"));
        expect(ask(messages, 2, fourth, rid) == ALBIREO_OK, "a question at once after an answer is accepted");
        expect(wait_final(fourth, 60), "the question at once after an answer ends");
        {
            std::lock_guard<std::mutex> lock(fourth.m);
            expect(fourth.finals == 1 && fourth.after_final == 0 && fourth.kind == ALBIREO_EVENT_DONE,
                "the question at once after an answer ends once");
        }
        std::printf("  the history of an answer read ahead of the next question (%lld tokens)\n", warmed);
    }

    /*
        The guard against loops and the budget of a reasoning, the token "a" favoured by a bias
        of 100 at temperature 0 so that every answer repeats it.  The engine stops an answer
        once nine tenths of its last 384 tokens end a passage of 16 tokens it wrote before.  A
        model whose template opens the reasoning of every answer, asked for no reasoning, has its
        reasoning ended at once, or after the tokens of the budget the request gives: the engine
        writes the sentence of the authors of Qwen3 and the marker that closes the reasoning,
        which the client does not receive as reasoning, and the answer then receives the tokens
        the request gave it.
    */
    {
        const std::string thinking_path = folder + "test_engine_api_thinking.gguf";
        const albireo_test::tiny_gguf thinking =
            albireo_test::tiny_gguf_writer().build(albireo_test::tiny_gguf_writer::thinking_template());
        expect(thinking.save(thinking_path, thinking.bytes), "the model that always reasons is written");
        expect(lib.load_model(e, "t", thinking_path.c_str(), options, err, sizeof(err)) == ALBIREO_OK,
            std::string("the model that always reasons loads: ") + err);
        auto ask = [&](const std::string& model, const std::string& extra, collector& c) {
            const std::string req = "{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hello\"}],\"temperature\":0,\"logit_bias\":{\"97\":100}" + extra + "}";
            uint64_t rid = 0;
            const int rc = lib.submit(e, req.c_str(), &collect_event, &c, &rid, err, sizeof(err));
            expect(rc == ALBIREO_OK, "request accepted: " + extra + " " + err);
            if (rc == ALBIREO_OK) expect(wait_collected(c, 120), "request ended: " + extra);
            std::lock_guard<std::mutex> lock(c.m);
            return json::parse_or_null(c.final_json);
        };
        const std::string sentence = "Considering the limited time by the user, I have to give the solution based on "
            "the thinking directly now.";

        collector looping;
        const json f = ask("m", ",\"reasoning\":false,\"max_tokens\":1000", looping);
        const long long n = f["usage"]["completion_tokens"].as_int(0);
        expect(f["finish_reason"].as_string() == "length" && f["repetition"].as_bool(false) && n >= 384 && n < 450,
            "an answer that repeats itself is stopped (" + std::to_string(n) + " tokens)");
        expect(looping.content == std::string(static_cast<size_t>(std::max(0LL, n)), 'a'),
            "the answer stopped keeps the tokens it wrote");

        json models;
        expect(lib.run_query(e, json::parse_or_null("{\"op\":\"models\"}"), models) == ALBIREO_OK, "models query");
        int always = 0, switched = 0;
        for (const auto& m : models["models"].elements())
        {
            if (m["id"].as_string() == "t" && m["reasoning_always"].as_bool(false)) ++always;
            if (m["id"].as_string() == "m" && !m["reasoning_always"].as_bool(true)) ++switched;
        }
        expect(always == 1 && switched == 1, "the description of a model tells whether it always reasons");

        // The tokens of the end of a reasoning the engine writes.
        json counted;
        json q = json::object();
        q["op"] = "tokenize";
        q["model"] = "t";
        q["text"] = "\n\n" + sentence + "\n</think>\n\n";
        q["special"] = true;
        expect(lib.run_query(e, q, counted) == ALBIREO_OK, "tokenize query");
        const long long closing = static_cast<long long>(counted["tokens"].size());

        /*
            The end of a reasoning that reached its budget, which the engine writes, counts among the
            tokens of the answer and stays out of the reasoning the client receives.
        */
        collector brief;
        const json g = ask("t", ",\"reasoning\":false,\"reasoning_budget\":24,\"max_tokens\":40", brief);
        expect(brief.reasoning == std::string(24, 'a') && brief.content.find(sentence) == std::string::npos,
            "a reasoning ends after its budget, its end left out of the reasoning received: " + brief.reasoning);
        expect(trimmed(brief.content) == std::string(40, 'a') && g["finish_reason"].as_string() == "length"
            && g["usage"]["completion_tokens"].as_int(0) == 24 + closing + 40 && !g["repetition"].as_bool(false),
            "the answer that follows a short reasoning receives its own tokens: " + g.dump());

        collector none;
        const json g0 = ask("t", ",\"reasoning\":false,\"reasoning_budget\":0,\"max_tokens\":8", none);
        expect(none.reasoning.empty() && trimmed(none.content) == std::string(8, 'a')
            && g0["usage"]["completion_tokens"].as_int(0) == closing + 8,
            "a budget of 0 ends the reasoning at once: " + g0.dump());

        // A model that always reasons, asked for no reasoning: the engine ends its reasoning at once.
        collector plain;
        const json g1 = ask("t", ",\"reasoning\":false,\"max_tokens\":8", plain);
        expect(plain.reasoning.empty() && trimmed(plain.content) == std::string(8, 'a')
            && g1["usage"]["completion_tokens"].as_int(0) == closing + 8,
            "a model that always reasons, asked for no reasoning, answers at once: " + g1.dump());

        // A budget given as null is no budget.
        collector unset;
        const json g2 = ask("t", ",\"reasoning\":false,\"reasoning_budget\":null,\"max_tokens\":8", unset);
        expect(unset.reasoning.empty() && trimmed(unset.content) == std::string(8, 'a')
            && g2["usage"]["completion_tokens"].as_int(0) == closing + 8,
            "a budget given as null ends the reasoning at once as no budget does: " + g2.dump());

        collector background;
        ask("t", ",\"reasoning\":false,\"priority\":\"background\",\"max_tokens\":4", background);
        expect(background.reasoning.empty() && trimmed(background.content) == std::string(4, 'a'),
            "the work in the background answers at once: " + background.reasoning);

        // The effort low gives a short reasoning, 256 tokens for the work in the background, within max_tokens.
        collector low;
        ask("t", ",\"reasoning_effort\":\"low\",\"priority\":\"background\",\"max_tokens\":"
            + std::to_string(256 + closing + 4), low);
        expect(low.reasoning == std::string(256, 'a') && trimmed(low.content) == std::string(4, 'a'),
            "the effort low of the work in the background reasons for 256 tokens: "
            + std::to_string(low.reasoning.size()));

        // A reasoning asked for has no budget by default, even for the work in the background.
        collector deep;
        const json h = ask("t", ",\"reasoning\":true,\"priority\":\"background\",\"max_tokens\":300", deep);
        expect(deep.reasoning == std::string(300, 'a') && deep.content.empty()
            && h["finish_reason"].as_string() == "length", "a reasoning asked for has no budget by default");

        // The repetitions of the reasoning and those of the answer are counted apart.
        collector parts;
        const json i = ask("t", ",\"reasoning\":false,\"reasoning_budget\":300,\"max_tokens\":390", parts);
        expect(trimmed(parts.content) == std::string(390, 'a') && !i["repetition"].as_bool(false)
            && i["usage"]["completion_tokens"].as_int(0) == 300 + closing + 390,
            "the answer after a reasoning of 300 tokens receives its 390 tokens: " + i.dump());

        /*
            A reasoning that goes round in circles ends as at its budget, and the answer then
            receives the tokens of the request, whatever is left of the budget.
        */
        collector circles;
        const json j = ask("t", ",\"reasoning\":false,\"reasoning_budget\":1000,\"max_tokens\":40", circles);
        const size_t looped = circles.reasoning.size();
        expect(looped >= 384 && looped < 450 && circles.reasoning == std::string(looped, 'a'),
            "a reasoning that repeats itself is ended: " + std::to_string(looped) + " tokens");
        expect(trimmed(circles.content) == std::string(40, 'a') && j["finish_reason"].as_string() == "length"
            && !j["repetition"].as_bool(false)
            && j["usage"]["completion_tokens"].as_int(0) == static_cast<long long>(looped) + closing + 40,
            "the answer after a reasoning ended early receives the tokens of the request: " + j.dump());

        // The same with a reasoning asked for, whose answer then loops in turn.
        collector both;
        const json k = ask("t", ",\"reasoning\":true,\"max_tokens\":1500", both);
        expect(both.reasoning.size() >= 384 && both.reasoning.find_first_not_of('a') == std::string::npos
            && trimmed(both.content).size() >= 384
            && trimmed(both.content).find_first_not_of('a') == std::string::npos && k["repetition"].as_bool(false)
            && k["finish_reason"].as_string() == "length", "a reasoning asked for that loops, then its answer: "
            + k.dump());

        // The stop sequences of a request apply to its answer, not to the short reasoning the engine gives it.
        collector stopped;
        const json l = ask("t", ",\"reasoning\":false,\"reasoning_budget\":24,\"max_tokens\":40,"
            "\"stop\":[\"\\n\\n\"]", stopped);
        expect(trimmed(stopped.content) == std::string(40, 'a') && l["finish_reason"].as_string() == "length",
            "the stop sequences apply to the answer: " + l.dump());

        /*
            The drafts of the answers: the answers that repeat "a" find their drafts in their own
            text, and the model keeps them.  The same model without drafts writes the same
            reasoning and the same answer, stopped at the same token.
        */
        const std::string undrafted = "{\"context\": 1024, \"max_sequences\": 16, \"kv_mb\": 1, \"draft_tokens\": 0}";
        expect(lib.load_model(e, "u", thinking_path.c_str(), undrafted.c_str(), err, sizeof(err)) == ALBIREO_OK,
            std::string("the model without drafts loads: ") + err);
        collector plainly;
        const json k0 = ask("u", ",\"reasoning\":true,\"max_tokens\":1500", plainly);
        expect(plainly.reasoning == both.reasoning && plainly.content == both.content
            && k0["usage"]["completion_tokens"].as_int(-1) == k["usage"]["completion_tokens"].as_int(-2)
            && k0["finish_reason"].as_string() == k["finish_reason"].as_string(),
            "the drafts leave the answer unchanged: " + k0.dump() + " against " + k.dump());
        json drafts;
        expect(lib.run_query(e, json::parse_or_null("{\"op\":\"stats\"}"), drafts) == ALBIREO_OK, "stats query");
        long long drafted = -1, kept = -1, undrafted_total = -1;
        for (const auto& m : drafts["models"].elements())
        {
            if (m["id"].as_string() == "t")
            {
                drafted = m["drafted_total"].as_int(-1);
                kept = m["draft_kept_total"].as_int(-1);
            }
            if (m["id"].as_string() == "u") undrafted_total = m["drafted_total"].as_int(-1);
        }
        expect(drafted > 0 && kept > 0 && kept <= drafted && undrafted_total == 0, "the answers that repeat are "
            "drafted (" + std::to_string(kept) + " of " + std::to_string(drafted) + " tokens kept)");
        expect(lib.unload_model(e, "u") == ALBIREO_OK, "the model without drafts unloads");

        expect(lib.unload_model(e, "t") == ALBIREO_OK, "the model that always reasons unloads");
        std::printf("  the answers that loop stopped, the reasoning of a model that always reasons bounded\n");
        std::printf("  drafts: %lld of %lld tokens kept, the answers those of the model without them\n", kept, drafted);
    }

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
