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
        4. the measured routing with the usual kernels, compared as distributions, and the
           routes it chose.

        test_offload synthetic [folder] [device]

    checks the attention of the device on the tiny model of tiny_gguf.h, written in folder,
    with keys and values in half precision and in 8 bits, without and with the biases of
    the attention: a script of passes (a prompt, single steps across blocks, two sequences
    stepping together, one of them sharing the start of the other, a sequence cut inside a
    block and continued by a pass of a few tokens and by a longer one) runs through the
    processor alone and through the device, whose layers and attention must run there and
    whose logits must agree with those of the processor; then the same with a budget of 1 MB,
    which holds the matrices but no copy of the keys and values: the layers run on the device,
    the processor attends.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
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

    /*
        Gives the device a card of that many megabytes less its copies (ALBIREO_TEST_CARD_FREE_MB),
        or its own memory again without megabytes; then waits for the next reading of the memory of
        the card, which a model makes every quarter of a second at most.
    */
    void simulate_card(const char* megabytes)
    {
#if defined(_WIN32)
        _putenv_s("ALBIREO_TEST_CARD_FREE_MB", megabytes ? megabytes : "");
#else
        if (megabytes)
            setenv("ALBIREO_TEST_CARD_FREE_MB", megabytes, 1);
        else
            unsetenv("ALBIREO_TEST_CARD_FREE_MB");
#endif
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
                A card that lacks memory: with 8 GB, the layers and the attention run on the device;
                with nothing left, the copies leave the card at the next reading of its memory and the
                processor computes the script; with 8 GB again, the copies come back.  The logits stay
                those of the processor.
            */
            const long blocks = 16;
            qk_set_exact_products(true);
            std::vector<float> exact;
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, blocks, kv_format::f16);
                exact = script(rt);
            }
            tensor_device card(device_config(device, 0, "device"), pool, print_log);
            json roomy, short_of, again;
            std::vector<float> phases[3];
            {
                cpu_runtime rt(file, cfg, pool, size_t(64) << 20, blocks, kv_format::f16);
                rt.set_offload(card.make_offload());
                simulate_card("8192");
                phases[0] = script(rt);
                roomy = card.stats();
                simulate_card("0");
                phases[1] = script(rt);
                short_of = card.stats();
                simulate_card("8192");
                phases[2] = script(rt);
                again = card.stats();
            }
            simulate_card(nullptr);
            qk_set_exact_products(false);
            const char* names[3] = { "card with room", "card without memory", "card with room again" };
            for (int p = 0; p < 3; ++p)
            {
                expect(phases[p].size() == exact.size(), "the same rows on a card short of memory");
                if (phases[p].size() != exact.size()) continue;
                const stats e = compare(exact.data(), phases[p].data(), static_cast<long>(exact.size()) / V, V);
                report(names[p], e);
                expect(e.max_abs <= 1e-3 * std::max(1.0, e.scale) && e.top1 >= 0.99, names[p]);
            }
            std::printf("  layer passes %lld, %lld, %lld; copies %lld, %lld, %lld bytes; given back %lld times\n",
                roomy["layer_passes"].as_int(), short_of["layer_passes"].as_int(), again["layer_passes"].as_int(),
                roomy["weights_bytes"].as_int(), short_of["weights_bytes"].as_int(), again["weights_bytes"].as_int(),
                short_of["card_releases"].as_int());
            expect(roomy["layer_passes"].as_int() > 0 && roomy["weights_bytes"].as_int() > 0,
                "the layers ran on the card with room");
            expect(short_of["card_releases"].as_int() >= 1, "the copies left the card without memory");
            expect(short_of["weights_bytes"].as_int() == 0, "no copy on the card without memory");
            expect(short_of["layer_passes"].as_int() == roomy["layer_passes"].as_int(),
                "the processor computed the layers on the card without memory");
            expect(again["weights_bytes"].as_int() > 0
                && again["layer_passes"].as_int() > short_of["layer_passes"].as_int(),
                "the layers ran on the card with room again");
            expect(again["failures"].as_int() == 0, "no device failure on a card short of memory");
        }
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
        expect(d.mean_kl < 0.05 && d.top1 >= 0.85, "measured routing");
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
