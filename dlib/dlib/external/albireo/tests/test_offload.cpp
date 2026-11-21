// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the products a device takes from the processor runtime.

        test_offload model.gguf [device] [threads] [budget]

    device is the device of the build: the index of a card with CUDA, "NPU", "GPU" or
    "CPU" with OpenVINO (the CPU plugin of OpenVINO runs the device code on a machine
    without NPU), ignored with the host code of dlib.  budget gives the megabytes of the
    copies of the weights of the first pass below, 0 (by default) for the automatic budget,
    a share of the free memory of the device.

    The program runs the same token sequence, a prompt of 130 tokens then 16 single steps,
    through the runtime alone and through the runtime with the device:
        1. every product on the device, against the processor kernels computing in
           single precision (qk_set_exact_products), so that the logits agree to the
           rounding of the device; with CUDA and the host code of dlib, the layers of a
           dense model run on the device (layer_pass) when the budget holds the matrices
           of its layers, the prompt through tiles of the weights expanded for cuBLAS and
           the steps through the blocks read in place, the device attending the steps itself;
        2. the same with a budget of 16 MB for the copies of the weights, so that most
           matrices cross the bus at each product of the prompt, or are left to the
           processor;
        3. with CUDA and the host code of dlib, keys and values in 8 bits, the device
           attending the steps over its copies of them, against the exact products with the
           same keys and values, compared as distributions;
        4. the measured routing with the usual kernels, or with the exact products when the
           device computes the layers whole, compared as distributions, and the routes it chose.

        test_offload synthetic [folder] [device]

    checks the attention of the device on the tiny model of tiny_gguf.h, written in folder,
    with keys and values in half precision and in 8 bits, without and with the biases of
    the attention: a script of passes (a prompt, single steps across blocks, two sequences
    stepping together, one of them sharing the start of the other, a sequence cut inside a
    block and continued by a pass of a few tokens and by a longer one) runs through the
    processor alone and through the device, whose layers and attention must run there and
    whose logits must agree with those of the processor; then the same with a budget of 1 MB,
    which holds the matrices but no copy of the keys and values: the layers run on the device,
    the processor attends; then on a card that runs out of memory (ALBIREO_TEST_CARD_FREE_MB),
    whose copies leave it, wait for the ceiling the release left to rise once the card has room
    again (ALBIREO_TEST_CARD_RAISE_MS gives the period of the rise), and come back, then leave
    again within the period of that rise, their bytes bounding the next rises and the period
    doubling.  The tiny mixture of experts and the tiny hybrid decoder of tiny_gguf.h follow,
    whose steps hand the products of a layer over by groups: a device that leaves them to the
    processor gives the logits of the runtime alone, bit for bit, and the measured routing
    and the device those of the processor to their rounding.

    The layers of the tiny hybrid decoder then run on the device (hybrid_pass), with experts
    in single precision and with experts in Q8_0 behind an exit of their latent space in
    BF16: a script of passes (chunks of a prompt that save the states of the Mamba-3 layers,
    one of them through the tiles of the weights expanded, single steps, sequences that step
    together, copies at a saved length and at the length of a sequence whose states the
    device alone holds, a sequence that goes back to a saved state) gives the logits of the
    processor.  A device that fails during a step (ALBIREO_TEST_HYBRID_FAILURE) gives the
    states back, the processor computing that step and the following passes; a device that
    lacks memory during a step keeps the layers afterwards; a device that fails without
    giving the states back loses them, the sequence then going back to its saved state.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../common/engine_api.h"
#include "../engine/cpu_runtime.h"
#include "../engine/tensor_backend.h"
#include "tiny_gguf.h"

using namespace albireo;

namespace
{
    int failures = 0;

    struct stats
    {
        double max_abs = 0;
        double scale = 0;
        double mean_kl = 0;
        double top1 = 0;
    };

    stats compare(const float* ref, const float* got, long rows, long vocab)
    {
        stats s;
        long same = 0;
        for (long r = 0; r < rows; ++r)
        {
            const float* a = ref + r * vocab;
            const float* b = got + r * vocab;
            double ma = -1e30, mb = -1e30;
            long ia = 0, ib = 0;
            for (long v = 0; v < vocab; ++v)
            {
                s.max_abs = std::max(s.max_abs, static_cast<double>(std::fabs(a[v] - b[v])));
                s.scale = std::max(s.scale, static_cast<double>(std::fabs(a[v])));
                if (a[v] > ma) { ma = a[v]; ia = v; }
                if (b[v] > mb) { mb = b[v]; ib = v; }
            }
            if (ia == ib) ++same;
            double za = 0, zb = 0;
            for (long v = 0; v < vocab; ++v)
            {
                za += std::exp(a[v] - ma);
                zb += std::exp(b[v] - mb);
            }
            double kl = 0;
            for (long v = 0; v < vocab; ++v)
            {
                const double pa = std::exp(a[v] - ma) / za;
                kl += pa * ((a[v] - ma - std::log(za)) - (b[v] - mb - std::log(zb)));
            }
            s.mean_kl += kl;
        }
        s.mean_kl /= std::max<long>(1, rows);
        s.top1 = static_cast<double>(same) / std::max<long>(1, rows);
        return s;
    }

    void report(const char* what, const stats& s)
    {
        std::printf("  %-36s max |diff| %.3g (logits up to %.3g), mean KL %.3g, top-1 %.1f%%\n",
            what, s.max_abs, s.scale, s.mean_kl, 100.0 * s.top1);
    }

    void expect(bool ok, const char* what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what);
        }
    }

    // A prompt of k tokens in one pass, then the other tokens one at a time; returns the
    // logits of every position.
    std::vector<float> run(cpu_runtime& rt, const std::vector<int>& tokens, long k, double& ms)
    {
        const long T = static_cast<long>(tokens.size());
        const long V = rt.config().vocab;
        std::vector<float> out(static_cast<size_t>(T * V));
        const int seq = rt.create_sequence();
        expect(rt.reserve_sequence(seq, T), "reserve");
        std::vector<float> logits;
        forward_batch b;
        b.tokens.assign(tokens.begin(), tokens.begin() + k);
        b.segments.push_back({ seq, 0, k });
        for (long i = 0; i < k; ++i) b.logit_rows.push_back(i);
        const auto t0 = std::chrono::steady_clock::now();
        rt.forward(b, logits);
        std::copy(logits.begin(), logits.end(), out.begin());
        for (long i = k; i < T; ++i)
        {
            b.clear();
            b.tokens.push_back(tokens[static_cast<size_t>(i)]);
            b.segments.push_back({ seq, 0, 1 });
            b.logit_rows.push_back(0);
            rt.forward(b, logits);
            std::copy(logits.begin(), logits.end(), out.begin() + i * V);
        }
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        rt.release_sequence(seq);
        return out;
    }

    json device_config(const std::string& device, long budget_mb, const std::string& routing)
    {
        json c = json::object();
        c["device"] = device;
        c["offload_mb"] = budget_mb;
        c["routing"] = routing;
        c["v2ram"] = false;
        return c;
    }

    void print_log(int level, const std::string& message)
    {
        static const char* names[] = { "debug", "info", "warning", "error" };
        std::printf("  [%s] %s\n", names[std::max(0, std::min(3, level))], message.c_str());
    }

    // Sets a variable of the environment, or removes it without value.
    void set_variable(const char* name, const char* value)
    {
#if defined(_WIN32)
        _putenv_s(name, value ? value : "");
#else
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
#endif
    }

    /*
        Gives the device a card of that many megabytes less its copies (ALBIREO_TEST_CARD_FREE_MB),
        or its own memory again without megabytes; then waits for the next reading of the memory of
        the card, which a model makes every quarter of a second at most.
    */
    void simulate_card(const char* megabytes)
    {
        set_variable("ALBIREO_TEST_CARD_FREE_MB", megabytes);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    // A deterministic token of the tiny vocabulary.
    int next_token(uint32_t& state)
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<int>((state >> 8) % 256);
    }

    // Runs a pass of count tokens on each of the given sequences and appends the logits of its rows to out.
    void pass(cpu_runtime& rt, const std::vector<int>& seqs, long count, uint32_t& state, std::vector<float>& out)
    {
        forward_batch b;
        for (int s : seqs)
        {
            const long start = static_cast<long>(b.tokens.size());
            for (long i = 0; i < count; ++i) b.tokens.push_back(next_token(state));
            b.segments.push_back({ s, start, count });
            expect(rt.reserve_sequence(s, rt.sequence_length(s) + count), "reserve");
        }
        for (long i = 0; i < static_cast<long>(b.tokens.size()); ++i) b.logit_rows.push_back(i);
        std::vector<float> logits;
        rt.forward(b, logits);
        out.insert(out.end(), logits.begin(), logits.end());
    }

    /*
        The passes of the synthetic check, the same for every runtime, with blocks of 16
        positions: a prompt of 40 tokens, 20 single steps, a second sequence holding the first
        37 positions of the first (two blocks shared, the third copied), both stepping together
        10 times, the first cut at 50 positions and continued by a pass of 5 tokens, 4 steps, a
        cut at 61, a pass of 12 tokens and 6 steps; returns the logits of every row.
    */
    std::vector<float> script(cpu_runtime& rt)
    {
        std::vector<float> out;
        uint32_t state = 99;
        const int a = rt.create_sequence();
        pass(rt, { a }, 40, state, out);
        for (int i = 0; i < 20; ++i) pass(rt, { a }, 1, state, out);
        const int b = rt.create_sequence();
        expect(rt.copy_sequence(a, b, 37), "copy of the start of a sequence");
        for (int i = 0; i < 10; ++i) pass(rt, { a, b }, 1, state, out);
        rt.truncate_sequence(a, 50);
        pass(rt, { a }, 5, state, out);
        for (int i = 0; i < 4; ++i) pass(rt, { a }, 1, state, out);
        rt.truncate_sequence(a, 61);
        pass(rt, { a }, 12, state, out);
        for (int i = 0; i < 6; ++i) pass(rt, { a, b }, 1, state, out);
        rt.release_sequence(b);
        rt.release_sequence(a);
        expect(rt.kv_bytes_used() == 0, "every block returned");
        return out;
    }

    /*
        The steps of a mixture of experts, or of a hybrid decoder, whose layers hand their
        products over by groups (matmul_offload::matmul_group()), after a prompt of 12 tokens:
        with the routing "host", the device leaves every group to the processor, which
        computes it in one parallel section and gives the logits of the runtime alone, bit for
        bit; with the measured routing, a group goes to the processor once the routes of its
        shapes are settled there; with the routing "device", every product runs on the device.
    */
    void grouped_products(const std::string& path, const std::string& device, bool hybrid, thread_pool& pool)
    {
        albireo_test::tiny_gguf_writer writer;
        const albireo_test::tiny_gguf model = hybrid ? writer.build_hybrid() : writer.experts(8, 3).build();
        if (!model.save(path, model.bytes))
        {
            expect(false, "write the model of the groups of products");
            return;
        }
        auto file = std::make_shared<model_file>(path);
        const model_config cfg = read_model_config(*file);
        std::printf("%s\n", describe(cfg).c_str());
        const long V = cfg.vocab;
        std::vector<int> tokens(40);
        uint32_t state = 5;
        for (auto& t : tokens) t = next_token(state);
        const long prompt = 12;
        const long T = static_cast<long>(tokens.size());
        double ms = 0;
        std::vector<float> reference;
        {
            cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
            reference = run(rt, tokens, prompt, ms);
        }
        const char* routings[3] = { "host", "auto", "device" };
        for (const char* routing : routings)
        {
            tensor_device dev(device_config(device, 0, routing), pool, print_log);
            std::vector<float> got;
            json st;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
                rt.set_offload(dev.make_offload());
                got = run(rt, tokens, prompt, ms);
                st = dev.stats();
            }
            const stats d = compare(reference.data(), got.data(), T, V);
            report((std::string("groups of products, routing ") + routing).c_str(), d);
            expect(st["failures"].as_int() == 0, "no device failure with groups of products");
            if (std::string(routing) == "host")
            {
                expect(d.max_abs == 0, "groups left to the processor against the runtime alone");
                expect(st["device_products"].as_int() == 0 && st["host_products"].as_int() > 0,
                    "every product of a device that routes to the processor on the processor");
            }
            else
            {
                expect(d.max_abs <= 2e-3 * std::max(1.0, d.scale) && d.top1 >= 0.97,
                    "groups of products through the device");
                if (std::string(routing) == "device")
                    expect(st["device_products"].as_int() > 0, "the products of the groups on the device");
            }
        }
    }

    /*
        A pass of count tokens on each of the given sequences of a hybrid decoder, the tokens of
        a sequence being a chunk of a prompt when they are several; the logits of its rows, or
        of the last row of each sequence alone, go to out.
    */
    void hybrid_step(cpu_runtime& rt, const std::vector<int>& seqs, long count, bool last_only, uint32_t& state,
        std::vector<float>& out)
    {
        forward_batch b;
        for (int s : seqs)
        {
            const long start = static_cast<long>(b.tokens.size());
            for (long i = 0; i < count; ++i) b.tokens.push_back(next_token(state));
            b.segments.push_back({ s, start, count });
            expect(rt.reserve_sequence(s, rt.sequence_length(s) + count), "reserve");
            if (last_only) b.logit_rows.push_back(start + count - 1);
        }
        if (!last_only)
            for (long i = 0; i < static_cast<long>(b.tokens.size()); ++i) b.logit_rows.push_back(i);
        std::vector<float> logits;
        rt.forward(b, logits);
        out.insert(out.end(), logits.begin(), logits.end());
    }

    /*
        The passes of the check of the layers of a hybrid decoder, with blocks of 16 positions: a
        chunk of 20 tokens, 10 steps, a second sequence copied from the first at its saved
        length, both stepping together 6 times, a chunk of 9 tokens of which the last row is
        asked, a chunk of 70 tokens of the second sequence, 4 steps of both, the first sequence
        going back to the 45 tokens of its saved state, a chunk of 3 tokens and 3 steps, a third
        sequence copied from the first at its length, and 3 steps of both.  mark is called
        before the fifth step and before the steps that follow the return to a saved state.
    */
    std::vector<float> hybrid_script(cpu_runtime& rt, const std::function<void(int)>& mark)
    {
        std::vector<float> out;
        uint32_t state = 7;
        const int a = rt.create_sequence();
        hybrid_step(rt, { a }, 20, false, state, out);
        for (int i = 0; i < 10; ++i)
        {
            if (i == 4) mark(1);
            hybrid_step(rt, { a }, 1, false, state, out);
        }
        const int b = rt.create_sequence();
        expect(rt.copy_sequence(a, b, 20), "copy of a sequence at its saved length");
        for (int i = 0; i < 6; ++i) hybrid_step(rt, { a, b }, 1, false, state, out);
        hybrid_step(rt, { a }, 9, true, state, out);
        hybrid_step(rt, { b }, 70, true, state, out);
        for (int i = 0; i < 4; ++i) hybrid_step(rt, { a, b }, 1, false, state, out);
        expect(rt.restorable_length(a, 48) == 45, "the length of the saved state of a sequence");
        rt.truncate_sequence(a, 45);
        hybrid_step(rt, { a }, 3, false, state, out);
        mark(2);
        for (int i = 0; i < 3; ++i) hybrid_step(rt, { a }, 1, false, state, out);
        const int c = rt.create_sequence();
        expect(rt.copy_sequence(a, c, rt.sequence_length(a)), "copy of a sequence at its length");
        for (int i = 0; i < 3; ++i) hybrid_step(rt, { a, c }, 1, false, state, out);
        rt.release_sequence(c);
        rt.release_sequence(b);
        rt.release_sequence(a);
        expect(rt.kv_bytes_used() == 0, "every block of the hybrid decoder returned");
        return out;
    }

    /*
        The layers of the tiny hybrid decoder on the device (hybrid_pass), against the processor
        alone computing in single precision: the script above; the script with a device that
        fails during a step, then with one that lacks memory during a step; and a sequence whose
        states are lost with the device.
    */
    void hybrid_layers(const std::string& path, const std::string& device, thread_pool& pool)
    {
        const auto nothing = [](int) {};
        for (int compact = 0; compact < 2; ++compact)
        {
            const albireo_test::tiny_gguf model = albireo_test::tiny_gguf_writer().latent(32)
                .compact_experts(compact != 0).build_hybrid();
            if (!model.save(path, model.bytes))
            {
                expect(false, "write the hybrid decoder of the layers of the device");
                return;
            }
            auto file = std::make_shared<model_file>(path);
            const model_config cfg = read_model_config(*file);
            std::printf("%s%s\n", describe(cfg).c_str(), compact ? ", experts in Q8_0" : "");
            const long V = cfg.vocab;
            qk_set_exact_products(true);
            std::vector<float> reference;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
                reference = hybrid_script(rt, nothing);
            }
            const long rows = static_cast<long>(reference.size()) / V;

            // Without a failure, then with a device that fails or lacks memory in the fifth step.
            const char* cases[3] = { "", "3", "memory:3" };
            for (const char* failure : cases)
            {
                const std::string asked = failure;
                tensor_device dev(device_config(device, 0, "auto"), pool, print_log);
                std::vector<float> got;
                json before, after;
                {
                    cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
                    rt.set_offload(dev.make_offload());
                    got = hybrid_script(rt, [&](int at) {
                        if (at == 1)
                        {
                            before = dev.stats();
                            if (!asked.empty()) set_variable("ALBIREO_TEST_HYBRID_FAILURE", failure);
                        }
                        if (at == 2) after = dev.stats();
                    });
                    set_variable("ALBIREO_TEST_HYBRID_FAILURE", nullptr);
                }
                const json st = dev.stats();
                expect(got.size() == reference.size(), "the same rows of the hybrid decoder");
                if (got.size() != reference.size()) continue;
                const stats d = compare(reference.data(), got.data(), rows, V);
                const std::string what = asked.empty() ? "layers of the hybrid decoder on the device"
                    : asked == "3" ? "device that fails in a step" : "device that lacks memory in a step";
                report(what.c_str(), d);
                expect(d.max_abs <= 2e-3 * std::max(1.0, d.scale) && d.top1 >= 0.97, what.c_str());
                std::printf("  passes on the device: %lld before the fifth step, %lld at the return to a saved state, "
                    "%lld in all; states left on the device %lld bytes\n", before["layer_passes"].as_int(),
                    after["layer_passes"].as_int(), st["layer_passes"].as_int(), st["weights_bytes"].as_int());
                expect(before["layer_passes"].as_int() >= 5, "the layers of the hybrid decoder ran on the device");
                if (asked == "3")
                    expect(after["layer_passes"].as_int() == before["layer_passes"].as_int() + 1,
                        "the processor computes the passes after a failure of the device");
                else
                    expect(st["layer_passes"].as_int() >= before["layer_passes"].as_int() + 25,
                        "the device keeps the layers of the hybrid decoder");
                expect(st["weights_bytes"].as_int() == 0, "the copies and the states left with the model");
            }

            /*
                A device that fails in a step without giving the states back: the pass ends with an
                error, the sequence goes back to the 20 tokens of its saved state, and the processor
                continues it as a runtime that never had a device.
            */
            std::vector<int> tokens(27);
            uint32_t state = 21;
            for (auto& t : tokens) t = next_token(state);
            auto feed = [&](cpu_runtime& rt, int seq, long first, long count, std::vector<float>& out) {
                forward_batch b;
                b.tokens.assign(tokens.begin() + first, tokens.begin() + first + count);
                b.segments.push_back({ seq, 0, count });
                b.logit_rows.push_back(count - 1);
                expect(rt.reserve_sequence(seq, rt.sequence_length(seq) + count), "reserve");
                std::vector<float> logits;
                rt.forward(b, logits);
                out.insert(out.end(), logits.begin(), logits.end());
            };
            std::vector<float> want, lost;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
                const int seq = rt.create_sequence();
                feed(rt, seq, 0, 20, want);
                for (long i = 23; i < 27; ++i) feed(rt, seq, i, 1, want);
                rt.release_sequence(seq);
            }
            {
                tensor_device dev(device_config(device, 0, "auto"), pool, print_log);
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16);
                rt.set_offload(dev.make_offload());
                const int seq = rt.create_sequence();
                feed(rt, seq, 0, 20, lost);
                std::vector<float> unused;
                for (long i = 20; i < 22; ++i) feed(rt, seq, i, 1, unused);
                set_variable("ALBIREO_TEST_HYBRID_FAILURE", "lost:3");
                bool thrown = false;
                try
                {
                    feed(rt, seq, 22, 1, unused);
                }
                catch (const std::runtime_error&)
                {
                    thrown = true;
                }
                set_variable("ALBIREO_TEST_HYBRID_FAILURE", nullptr);
                expect(thrown, "a pass whose states are lost with the device ends with an error");
                expect(rt.restorable_length(seq, 22) == 20,
                    "a sequence that lost its states goes back to its saved state");
                rt.truncate_sequence(seq, 20);
                for (long i = 23; i < 27; ++i) feed(rt, seq, i, 1, lost);
                rt.release_sequence(seq);
            }
            expect(lost.size() == want.size(), "the same rows after the loss of the states");
            if (lost.size() == want.size())
            {
                const stats d = compare(want.data(), lost.data(), static_cast<long>(want.size()) / V, V);
                report("states lost with the device", d);
                expect(d.max_abs <= 2e-3 * std::max(1.0, d.scale) && d.top1 >= 0.97, "states lost with the device");
            }
            qk_set_exact_products(false);
        }
    }

    int synthetic(const std::string& folder, const std::string& device)
    {
        std::string backend, runtime, reason, reason_code;
        json reason_args = json::object();
        tensor_device::describe(backend, runtime, reason, reason_code, reason_args);
        if (backend != "cuda" && backend != "dlib")
        {
            // The routing of OpenVINO takes products only: no layer and no attention to check.
            std::printf("backend %s: the device takes products only\nall checks passed\n", backend.c_str());
            return 0;
        }
        const std::string path = folder + "test_offload_synthetic.gguf";
        thread_pool pool(2);
        for (int with_biases = 0; with_biases < 2; ++with_biases)
        {
            const albireo_test::tiny_gguf model = albireo_test::tiny_gguf_writer().head_dim(32)
                .biases(with_biases != 0).build();
            if (!model.save(path, model.bytes))
            {
                std::printf("cannot write %s\n", path.c_str());
                return 2;
            }
            auto file = std::make_shared<model_file>(path);
            const model_config cfg = read_model_config(*file);
            std::printf("%s%s\n", describe(cfg).c_str(), with_biases ? ", biases" : "");
            const long V = cfg.vocab;
            for (kv_format format : { kv_format::f16, kv_format::q8 })
            {
                const bool q8 = format == kv_format::q8;
                qk_set_exact_products(true);
                std::vector<float> reference;
                {
                    cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16, format);
                    reference = script(rt);
                }
                tensor_device dev(device_config(device, 0, "device"), pool, print_log);
                std::vector<float> got;
                json st;
                {
                    cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 16, format);
                    rt.set_offload(dev.make_offload());
                    got = script(rt);
                    st = dev.stats();
                }
                qk_set_exact_products(false);
                expect(dev.stats()["kv_bytes"].as_int() == 0, "the copies of the keys and values left with the model");
                const long rows = static_cast<long>(reference.size()) / V;
                expect(got.size() == reference.size(), "the same rows");
                const stats d = compare(reference.data(), got.data(), rows, V);
                report(q8 ? "device, keys and values in 8 bits" : "device, keys and values in half precision", d);
                std::printf("  passes on the device %lld, attended there %lld, copies of keys and values %lld bytes\n",
                    st["layer_passes"].as_int(), st["attention_passes"].as_int(), st["kv_bytes"].as_int());
                expect(st["layer_passes"].as_int() > 0, "the layers ran on the device");
                // The 41 passes of 1 to 5 tokens of the script, and the probes of the attention.
                expect(st["attention_passes"].as_int() >= 41, "the device attended the steps");
                expect(st["kv_bytes"].as_int() > 0, "copies of the keys and values on the device");
                expect(st["failures"].as_int() == 0, "no device failure");
                // A key rounded to the other side of a step of the 8-bit format moves its scores a little more.
                if (q8)
                    expect(d.mean_kl < 1e-4 && d.top1 >= 0.97, "attention of the device, 8 bits");
                else
                    expect(d.max_abs <= 1e-3 * std::max(1.0, d.scale) && d.top1 >= 0.99, "attention of the device");
            }
            if (with_biases) continue;

            /*
                A device whose budget of 1 MB holds the matrices of the tiny model but not one page of
                4,096 positions: the layers run on the device, the processor attends the passes, and
                the attention of the device is probed again 64 passes later, the script running twice.
            */
            qk_set_exact_products(true);
            std::vector<float> reference;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 4096, kv_format::f16);
                reference = script(rt);
                const std::vector<float> again = script(rt);
                reference.insert(reference.end(), again.begin(), again.end());
            }
            tensor_device dev(device_config(device, 1, "device"), pool, print_log);
            std::vector<float> got;
            json st;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, 4096, kv_format::f16);
                rt.set_offload(dev.make_offload());
                got = script(rt);
                const std::vector<float> again = script(rt);
                got.insert(got.end(), again.begin(), again.end());
                st = dev.stats();
            }
            qk_set_exact_products(false);
            expect(got.size() == reference.size(), "the same rows, without room");
            const stats d = compare(reference.data(), got.data(), static_cast<long>(reference.size()) / V, V);
            report("device without room for its pages", d);
            expect(st["layer_passes"].as_int() > 0, "the layers ran on the device without room for the pages");
            expect(st["attention_passes"].as_int() == 0, "the processor attended the passes");
            expect(st["failures"].as_int() == 0, "no device failure without room");
            expect(d.max_abs <= 1e-3 * std::max(1.0, d.scale) && d.top1 >= 0.99, "layers of the device without room");

            /*
                A card that lacks memory, with a budget of 512 MB: with 8 GB, the layers and the
                attention run on the device; with nothing left, the copies leave the card at the next
                reading of its memory and the processor computes the script; with 1,792 MB, the copies
                keep to the ceiling the release left, nothing, while the period of the rise of the
                ceiling (ALBIREO_TEST_CARD_RAISE_MS, 10 minutes) runs; with a period of a millisecond,
                the ceiling rises by half the 768 MB the card keeps beside its reserve, and the copies
                come back; with nothing left again, within the period of 10 minutes of that rise, the
                copies leave, their bytes bound the next rises and the period doubles; once the model
                leaves, no copy is left and the ceiling and its bound go.  The logits stay those of the
                processor.
            */
            const long blocks = 16;
            qk_set_exact_products(true);
            std::vector<float> exact;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, blocks, kv_format::f16);
                exact = script(rt);
            }
            tensor_device card(device_config(device, 512, "device"), pool, print_log);
            json roomy, short_of, held, again, after;
            std::vector<float> phases[5];
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, blocks, kv_format::f16);
                rt.set_offload(card.make_offload());
                set_variable("ALBIREO_TEST_CARD_RAISE_MS", "600000");
                simulate_card("8192");
                phases[0] = script(rt);
                roomy = card.stats();
                simulate_card("0");
                phases[1] = script(rt);
                short_of = card.stats();
                simulate_card("1792");
                phases[2] = script(rt);
                held = card.stats();
                set_variable("ALBIREO_TEST_CARD_RAISE_MS", "1");
                simulate_card("1792");
                phases[3] = script(rt);
                again = card.stats();
                set_variable("ALBIREO_TEST_CARD_RAISE_MS", "600000");
                simulate_card("0");
                phases[4] = script(rt);
                after = card.stats();
            }
            const json gone = card.stats();
            simulate_card(nullptr);
            set_variable("ALBIREO_TEST_CARD_RAISE_MS", nullptr);
            qk_set_exact_products(false);
            const char* names[5] = { "card with room", "card without memory", "card with room, copies held",
                "card with room again", "card without memory after a rise" };
            for (int p = 0; p < 5; ++p)
            {
                expect(phases[p].size() == exact.size(), "the same rows on a card short of memory");
                if (phases[p].size() != exact.size()) continue;
                const stats e = compare(exact.data(), phases[p].data(), static_cast<long>(exact.size()) / V, V);
                report(names[p], e);
                expect(e.max_abs <= 1e-3 * std::max(1.0, e.scale) && e.top1 >= 0.99, names[p]);
            }
            std::printf("  layer passes %lld, %lld, %lld, %lld; copies %lld, %lld, %lld, %lld bytes; given back %lld "
                "times; ceiling %lld bytes, then %s\n", roomy["layer_passes"].as_int(),
                short_of["layer_passes"].as_int(), held["layer_passes"].as_int(), again["layer_passes"].as_int(),
                roomy["weights_bytes"].as_int(), short_of["weights_bytes"].as_int(), held["weights_bytes"].as_int(),
                again["weights_bytes"].as_int(), again["card_releases"].as_int(), again["card_cap_bytes"].as_int(-1),
                gone["card_cap_bytes"].is_null() ? "none" : "kept");
            expect(roomy["layer_passes"].as_int() > 0 && roomy["weights_bytes"].as_int() > 0,
                "the layers ran on the card with room");
            expect(roomy["card_cap_bytes"].is_null(), "no ceiling before the card lacked memory");
            expect(short_of["card_releases"].as_int() >= 1, "the copies left the card without memory");
            expect(short_of["weights_bytes"].as_int() == 0, "no copy on the card without memory");
            expect(short_of["layer_passes"].as_int() == roomy["layer_passes"].as_int(),
                "the processor computed the layers on the card without memory");
            expect(!short_of["card_cap_bytes"].is_null() && short_of["card_cap_bytes"].as_int() == 0,
                "the copies keep to what stayed on the card");
            expect(held["weights_bytes"].as_int() == 0
                && held["layer_passes"].as_int() == short_of["layer_passes"].as_int(),
                "the copies wait for the ceiling to rise");
            expect(again["weights_bytes"].as_int() > 0
                && again["layer_passes"].as_int() > held["layer_passes"].as_int(),
                "the layers ran on the card with room again");
            expect(again["card_cap_bytes"].as_int(-1) == (384LL << 20),
                "the ceiling rose by half the room of the card");
            expect(again["card_releases"].as_int() == short_of["card_releases"].as_int(),
                "no release while the card had room");
            expect(again["failures"].as_int() == 0, "no device failure on a card short of memory");
            std::printf("  after a rise: given back %lld times, bound %lld bytes, period %lld ms\n",
                after["card_releases"].as_int(), after["card_short_bytes"].as_int(-1),
                after["card_raise_ms"].as_int());
            expect(again["card_short_bytes"].is_null(), "no bound before a release that follows a rise");
            expect(after["card_releases"].as_int() == again["card_releases"].as_int() + 1
                && after["weights_bytes"].as_int() == 0, "the copies left the card without memory after a rise");
            expect(after["card_short_bytes"].as_int(-1) == again["weights_bytes"].as_int(),
                "the copies of a release that follows a rise bound the next rises");
            expect(after["card_raise_ms"].as_int() == 1200000, "the period of the rises doubled");
            expect(gone["weights_bytes"].as_int() == 0 && gone["card_cap_bytes"].is_null()
                && gone["card_short_bytes"].is_null(), "the ceiling and its bound left with the last copies");
        }
        std::remove(path.c_str());
        for (int hybrid = 0; hybrid < 2; ++hybrid) grouped_products(path, device, hybrid != 0, pool);
        std::remove(path.c_str());
        hybrid_layers(path, device, pool);
        std::remove(path.c_str());
        if (failures)
        {
            std::printf("%d checks FAILED\n", failures);
            return 1;
        }
        std::printf("all checks passed\n");
        return 0;
    }
}

static int run_checks(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_offload model.gguf [device] [threads] [budget]\n"
            "       test_offload synthetic [folder] [device]\n");
        return 2;
    }
    if (std::string(argv[1]) == "synthetic")
        return synthetic(argc > 2 ? std::string(argv[2]) + "/" : std::string(), argc > 3 ? argv[3] : "");
    const std::string device = argc > 2 ? argv[2] : "";
    const int threads = argc > 3 ? std::atoi(argv[3]) : 2;
    const long first_budget = argc > 4 ? std::max(0L, std::atol(argv[4])) : 0L;
    // The devices of OpenVINO are chosen once per process, before the first query.
    if (!device.empty())
    {
#if defined(_WIN32)
        _putenv_s("DLIB_NPU_DEVICE", device.c_str());
#else
        setenv("DLIB_NPU_DEVICE", device.c_str(), 1);
#endif
    }

    auto file = std::make_shared<model_file>(argv[1]);
    const model_config cfg = read_model_config(*file);
    std::printf("%s\n", describe(cfg).c_str());
    const long V = cfg.vocab;

    std::string backend, runtime, reason, reason_code;
    json reason_args = json::object();
    const json devices = tensor_device::describe(backend, runtime, reason, reason_code, reason_args);
    std::printf("backend %s (%s), devices %s%s\n", backend.c_str(), runtime.c_str(), devices.dump().c_str(),
        reason.empty() ? "" : (", " + reason).c_str());

    // A deterministic sequence of ordinary tokens.
    std::vector<int> tokens;
    unsigned s = 12345;
    for (int i = 0; i < 146; ++i)
    {
        s = s * 1103515245u + 12345u;
        tokens.push_back(static_cast<int>(100 + (s >> 8) % static_cast<unsigned>(std::min<long>(V - 100, 20000))));
    }
    const long T = static_cast<long>(tokens.size());
    const long k = 130;

    // The bytes of the matrices of the layers in their GGUF formats, which the layer pass keeps on the device.
    double layer_bytes = 0;
    for (const auto& t : file->reader().tensors())
    {
        const std::string& n = t.name;
        const size_t dot = n.rfind(".weight");
        if (n.compare(0, 4, "blk.") != 0 || dot == std::string::npos || dot + 7 != n.size()) continue;
        const std::string kind = n.substr(n.rfind('.', dot - 1) + 1, dot - n.rfind('.', dot - 1) - 1);
        if (kind == "attn_q" || kind == "attn_k" || kind == "attn_v" || kind == "attn_qkv" || kind == "attn_output"
            || kind == "ffn_gate" || kind == "ffn_up" || kind == "ffn_down")
            layer_bytes += static_cast<double>(t.nbytes);
    }

    thread_pool pool(threads);
    const size_t kv = size_t(256) << 20;
    double ms = 0;
    // Half precision arithmetic on an NPU or an Intel GPU: the comparisons are looser.
    const bool half = backend == "openvino" && device.find(":f32") == std::string::npos && device != "CPU";

    // 1 and 2: every product on the device, against exact processor products.
    qk_set_exact_products(true);
    std::vector<float> exact;
    {
        cpu_runtime rt(file, cfg, pool, kv, 16);
        exact = run(rt, tokens, k, ms);
        std::printf("processor, exact products: %.1f ms\n", ms);
    }
    for (int pass = 0; pass < 2; ++pass)
    {
        const bool small = pass == 1;
        const long budget = small ? 16 : first_budget;
        tensor_device dev(device_config(device, budget, "device"), pool, print_log);
        cpu_runtime rt(file, cfg, pool, kv, 16);
        rt.set_offload(dev.make_offload());
        std::printf("%s\n", rt.description().c_str());
        const std::vector<float> got = run(rt, tokens, k, ms);
        const json st = dev.stats();
        std::printf("device, every product, budget %s: %.1f ms, %s\n",
            budget ? (std::to_string(budget) + " MB").c_str() : "automatic", ms, st.dump().c_str());
        const stats d = compare(exact.data(), got.data(), T, V);
        report(small ? "device (small budget) vs exact" : "device vs exact", d);
        if (half)
            expect(d.mean_kl < 0.02 && d.top1 >= 0.9, "device products in half precision");
        else if (cfg.is_moe())
            // A rounding difference in the logits of the router can change the experts of a
            // token whose two candidates are close: the comparison bears on the distributions.
            expect(d.mean_kl < 0.01 && d.top1 >= 0.95, "device products");
        else
            expect(d.max_abs <= 2e-3 * std::max(1.0, d.scale) && d.top1 >= 0.97, "device products");
        expect(st["device_products"].as_int() > 0 || st["layer_passes"].as_int() > 0, "the device computed products");
        if ((backend == "cuda" || backend == "dlib") && !small && !cfg.is_moe())
        {
            if (st["weights_budget"].as_double(0) >= layer_bytes)
            {
                expect(st["layer_passes"].as_int() > 0, "the layers ran on the device");
                expect(st["attention_passes"].as_int() > 0, "the device attended the steps");
            }
            else
                std::printf("  the budget holds %.0f MB of the %.0f MB of the matrices of the layers: they stay "
                    "on the processor\n", st["weights_budget"].as_double(0) / 1048576.0, layer_bytes / 1048576.0);
        }
        if (backend != "openvino" && !small && !budget && !cfg.is_moe())
        {
            // Every product stays on the device when the automatic budget, a share of the free
            // memory, holds every matrix, even in single precision.
            const double matrices = 4.0 * cfg.d_model * (cfg.vocab + cfg.n_layers * (2.0 * cfg.q_dim()
                + 2.0 * cfg.kv_dim() + 3.0 * cfg.d_ffn));
            if (st["weights_budget"].as_double(0) >= matrices)
                expect(st["host_products"].as_int() == 0, "no product left to the processor");
            else
                std::printf("  the automatic budget holds %.0f MB of the %.0f MB of the matrices\n",
                    st["weights_budget"].as_double(0) / 1048576.0, matrices / 1048576.0);
        }
        expect(st["failures"].as_int() == 0, "no device failure");
    }

    /*
        3: keys and values in 8 bits, as the engine keeps them for a language model: the probes
        of the layers keep half precision, the device attends the steps over its copies in 8
        bits, against the processor with the same keys and values.
    */
    if ((backend == "cuda" || backend == "dlib") && !cfg.is_moe())
    {
        std::vector<float> exact8;
        {
            cpu_runtime rt(file, cfg, pool, kv, 16, kv_format::q8);
            exact8 = run(rt, tokens, k, ms);
        }
        tensor_device dev(device_config(device, first_budget, "device"), pool, print_log);
        cpu_runtime rt(file, cfg, pool, kv, 16, kv_format::q8);
        rt.set_offload(dev.make_offload());
        const std::vector<float> got = run(rt, tokens, k, ms);
        const json st = dev.stats();
        std::printf("device, keys and values in 8 bits: %.1f ms, %lld passes attended by the device\n", ms,
            st["attention_passes"].as_int());
        const stats d = compare(exact8.data(), got.data(), T, V);
        report("device vs exact, 8 bits", d);
        expect(d.mean_kl < 0.01 && d.top1 >= 0.9, "device products and attention, 8 bits");
        if (st["weights_budget"].as_double(0) >= layer_bytes)
        {
            expect(st["layer_passes"].as_int() > 0, "the layers ran on the device, 8 bits");
            expect(st["attention_passes"].as_int() > 0, "the device attended the steps, 8 bits");
        }
        expect(st["failures"].as_int() == 0, "no device failure, 8 bits");
    }

    // 4: the measured routing with the usual kernels.
    qk_set_exact_products(false);
    std::vector<float> usual;
    double host_ms = 0;
    {
        cpu_runtime rt(file, cfg, pool, kv, 16);
        usual = run(rt, tokens, k, host_ms);
        std::printf("processor, usual kernels: %.1f ms\n", host_ms);
        report("usual kernels vs exact", compare(exact.data(), usual.data(), T, V));
    }
    {
        tensor_device dev(device_config(device, 0, "auto"), pool, print_log);
        cpu_runtime rt(file, cfg, pool, kv, 16);
        rt.set_offload(dev.make_offload());
        // Several rounds, so that the routes are measured and chosen.
        std::vector<float> got;
        for (int round = 0; round < 6; ++round)
        {
            got = run(rt, tokens, k, ms);
            std::printf("measured routing, round %d: %.1f ms, device products %lld\n", round + 1, ms,
                dev.stats()["device_products"].as_int());
        }
        const json st = dev.stats();
        std::printf("routes: %s\n", st.dump().c_str());
        const stats d = compare(usual.data(), got.data(), T, V);
        report("measured routing vs processor", d);
        /*
            The layers the device computes whole take exact products: their logits then follow
            those of the exact products, which the usual kernels leave by the rounding of their
            activations.
        */
        const stats e = compare(exact.data(), got.data(), T, V);
        report("measured routing vs exact", e);
        expect((d.mean_kl < 0.05 && d.top1 >= 0.85) || (e.mean_kl < 0.01 && e.top1 >= 0.95), "measured routing");
        expect(st["failures"].as_int() == 0, "no device failure");
    }

    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

int main(int argc, char** argv)
{
    // An error, such as a file that cannot be opened or a device that fails, ends the program
    // with its message.
    try
    {
        return run_checks(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
}
