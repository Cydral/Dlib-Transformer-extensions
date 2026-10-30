// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the products a device takes from the processor runtime.

        test_offload model.gguf [device] [threads]

    device is the device of the build: the index of a card with CUDA, "NPU", "GPU" or
    "CPU" with OpenVINO (the CPU plugin of OpenVINO runs the device code on a machine
    without NPU), ignored with the host code of dlib.

    The program runs the same token sequence through the runtime alone and through the
    runtime with the device:
        1. every product on the device, against the processor kernels computing in
           single precision (qk_set_exact_products), so that the logits agree to the
           rounding of the device;
        2. the same with a budget of 16 MB for the copies of the weights, so that most
           matrices are expanded at each product, or left to the processor;
        3. the measured routing with the usual kernels, on a prompt followed by single
           steps, compared as distributions, and the routes it chose.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../common/engine_api.h"
#include "../engine/cpu_runtime.h"
#include "../engine/tensor_backend.h"

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
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_offload model.gguf [device] [threads]\n");
        return 2;
    }
    const std::string device = argc > 2 ? argv[2] : "";
    const int threads = argc > 3 ? std::atoi(argv[3]) : 2;
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
    for (int i = 0; i < 48; ++i)
    {
        s = s * 1103515245u + 12345u;
        tokens.push_back(static_cast<int>(100 + (s >> 8) % static_cast<unsigned>(std::min<long>(V - 100, 20000))));
    }
    const long T = static_cast<long>(tokens.size());
    const long k = 32;

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
    const long budgets[] = { 0, 16 };
    for (long budget : budgets)
    {
        tensor_device dev(device_config(device, budget, "device"), pool, print_log);
        cpu_runtime rt(file, cfg, pool, kv, 16);
        rt.set_offload(dev.make_offload());
        std::printf("%s\n", rt.description().c_str());
        const std::vector<float> got = run(rt, tokens, k, ms);
        const json st = dev.stats();
        std::printf("device, every product, budget %s: %.1f ms, %s\n", budget ? "16 MB" : "automatic", ms,
            st.dump().c_str());
        const stats d = compare(exact.data(), got.data(), T, V);
        report(budget ? "device (small budget) vs exact" : "device vs exact", d);
        if (half)
            expect(d.mean_kl < 0.02 && d.top1 >= 0.9, "device products in half precision");
        else if (cfg.is_moe())
            // A rounding difference in the logits of the router can change the experts of a
            // token whose two candidates are close: the comparison bears on the distributions.
            expect(d.mean_kl < 0.01 && d.top1 >= 0.95, "device products");
        else
            expect(d.max_abs <= 2e-3 * std::max(1.0, d.scale) && d.top1 >= 0.97, "device products");
        expect(st["device_products"].as_int() > 0, "the device computed products");
        if (backend != "openvino" && !budget && !cfg.is_moe())
        {
            // Every product stays on the device when the automatic budget, a share of the free
            // memory, holds every matrix in single precision.
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

    // 3: the measured routing with the usual kernels.
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
