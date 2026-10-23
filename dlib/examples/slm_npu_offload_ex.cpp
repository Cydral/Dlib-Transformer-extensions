// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    @file slm_npu_offload_ex.cpp
    @brief Routing the tensor operations of a language model to an NPU, and measuring the gain

    dlib built with DLIB_USE_NPU (see the build notes of the main README) offers every tensor
    operation it runs on the host to an inference device found when the program starts: the
    NPU of the processor by default, reached through the OpenVINO runtime. Each operation
    then runs where it was measured to be faster, with no change to the code that calls it.
    This program shows that mechanism at work on the machine it runs on:

    1) --info      : what was built in, which device was found and how it computes.
    2) --benchmark : the primitives of a transformer at the sizes of a small language model,
                     timed on the host and on the device, with the difference between the
                     two results.
    3) --train     : training steps of a small transformer on the built-in Shakespeare
                     extract, once with every operation on the host and once with automatic
                     routing, from the same initial weights, with the time per step, the
                     loss reached and the routing summary.

    Without option, the three run in turn.

    The device paths can be exercised on a machine without an NPU, on the OpenVINO CPU
    plugin, by setting DLIB_NPU_DEVICE=CPU. The timings then compare the host code of dlib
    with that of OpenVINO and say nothing about an NPU, while the results and the routing
    are those of a real device. DLIB_NPU_DEVICE=GPU selects an integrated GPU instead.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>

// Include internal dataset
#include "slm_data.h"

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

namespace
{
#ifdef DLIB_USE_CUDA
    const bool cuda_build = true;
#else
    const bool cuda_build = false;
#endif

    unsigned long long device_calls (const std::string& op)
    {
        for (const auto& s : npu::get_statistics())
            if (s.name == op)
                return s.device_calls;
        return 0;
    }

    // Largest difference relative to the largest magnitude of the reference.
    double relative_error (const tensor& reference, const tensor& other)
    {
        if (reference.size() != other.size())
            return std::numeric_limits<double>::infinity();
        const float* a = reference.host();
        const float* b = other.host();
        double scale = 0, err = 0;
        for (size_t i = 0; i < reference.size(); ++i)
        {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
                continue;
            scale = std::max(scale, static_cast<double>(std::abs(a[i])));
            err = std::max(err, static_cast<double>(std::abs(a[i] - b[i])));
        }
        return err == 0 ? 0 : err / std::max(scale, 1e-30);
    }

    resizable_tensor random_tensor (dlib::rand& rnd, long n, long k, long nr, long nc)
    {
        resizable_tensor t(n, k, nr, nc);
        float* p = t.host();
        for (size_t i = 0; i < t.size(); ++i)
            p[i] = 2 * rnd.get_random_float() - 1;
        return t;
    }

    /*
        Writable access to t, as the layer producing an activation has at every call. The
        device keeps a copy of the operands that no access of that kind reached since their
        last use, such as weights; the benchmark marks its activations this way so that they
        are converted at every call, as in a network.
    */
    void touch (tensor& t)
    {
        t.host();
    }

    // Median duration of one call, in milliseconds.
    template <typename F>
    double time_calls (F&& compute, int reps)
    {
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i)
        {
            const auto t0 = std::chrono::steady_clock::now();
            compute();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    }

    /*
        Times compute with every operation on the host, then forced on the device once its
        graph is compiled, and compares what it left in out both times. op is the name the
        routing statistics give to the operation.
    */
    template <typename F>
    void benchmark (const std::string& what, const std::string& op, const tensor& out, F compute, int reps)
    {
        const npu::routing initial = npu::get_routing();
        npu::set_routing(npu::routing::never);
        compute();
        const double host_ms = time_calls(compute, reps);
        resizable_tensor reference;
        reference.copy_size(out);
        memcpy(reference, out);

        npu::set_synchronous_compilation(true);
        npu::set_routing(npu::routing::always);
        const unsigned long long before = device_calls(op);
        compute();  // compiles the graph on first sight, then runs it once
        const double device_ms = time_calls(compute, reps);
        const bool ran = device_calls(op) > before;
        npu::set_routing(initial);
        npu::set_synchronous_compilation(false);

        cout << "  " << left << setw(44) << what << right << fixed << setprecision(3) << setw(10) << host_ms;
        if (ran)
            cout << setw(12) << device_ms << setw(9) << setprecision(2) << host_ms / device_ms << "x"
                 << setw(12) << scientific << setprecision(1) << relative_error(reference, out) << "\n";
        else
            cout << "      host only, the device declines it\n";
        cout.unsetf(std::ios::floatfield);
    }
}

// ----------------------------------------------------------------------------------------

void show_device ()
{
    cout << "=== DEVICE ===\n";
    if (!npu::is_compiled_in())
    {
        cout << "dlib was built without DLIB_USE_NPU: every operation runs on the "
             << (cuda_build ? "CUDA device.\n" : "host.\n");
        cout << "Configure with -DDLIB_USE_NPU=ON, OpenVINO installed, to route them to an NPU.\n\n";
        return;
    }
    npu::print_status(cout);
    if (npu::is_available())
        cout << "Arithmetic on the device: " << (npu::uses_half_precision() ? "half" : "single")
             << " precision. Operations that must not round stay on the host when it is half.\n";
    else
        cout << "No device was found, so every operation runs on the host. DLIB_NPU_DEVICE=CPU runs the\n"
             << "device paths on the OpenVINO CPU plugin to see them at work anyway.\n";
    cout << "\n";
}

// ----------------------------------------------------------------------------------------

void run_benchmark (int reps)
{
    cout << "=== BENCHMARK ===\n";
    if (!npu::is_available())
    {
        cout << "No device to compare the host with.\n\n";
        return;
    }
    if (cuda_build)
    {
        cout << "In a CUDA build, CUDA computes these operations and the NPU receives none.\n\n";
        return;
    }
    dlib::rand rnd(7);
    const long B = 8, T = 128, D = 256, H = 4, HD = D / H, F = 4 * D, V = 4096;
    cout << "Shapes of a small decoder: batch " << B << ", " << T << " tokens, width " << D << ", " << H
         << " heads, feed-forward " << F << ", vocabulary " << V << ". Median of " << reps << " calls.\n";
    cout << "  " << left << setw(44) << "operation" << right << setw(10) << "host ms" << setw(12) << "device ms"
         << setw(10) << "speed-up" << setw(12) << "difference" << "\n";

    // The products of the linear layers: the token rows times a weight matrix.
    resizable_tensor x = random_tensor(rnd, B * T, D, 1, 1);
    resizable_tensor w_qkv = random_tensor(rnd, D, 3 * D, 1, 1), qkv(B * T, 3 * D, 1, 1);
    benchmark("gemm, attention projection 1024x256x768", "gemm", qkv,
        [&]{ touch(x); tt::gemm(0, qkv, 1, x, false, w_qkv, false); }, reps);
    resizable_tensor w_up = random_tensor(rnd, D, F, 1, 1), up(B * T, F, 1, 1);
    benchmark("gemm, feed-forward 1024x256x1024", "gemm", up,
        [&]{ touch(x); tt::gemm(0, up, 1, x, false, w_up, false); }, reps);
    resizable_tensor w_down = random_tensor(rnd, F, D, 1, 1), down(B * T, D, 1, 1);
    benchmark("gemm, feed-forward back 1024x1024x256", "gemm", down,
        [&]{ touch(up); tt::gemm(0, down, 1, up, false, w_down, false); }, reps);

    // Token by token inference, at the width of a model of half a billion parameters: one or
    // a few rows against weights that do not change, which the device keeps after the first
    // calls instead of converting them at each one.
    const long DM = 896, FM = 4864;
    resizable_tensor w_ff = random_tensor(rnd, DM, FM, 1, 1), w_back = random_tensor(rnd, FM, DM, 1, 1);
    resizable_tensor one = random_tensor(rnd, 1, DM, 1, 1), one_up(1, FM, 1, 1), one_down(1, DM, 1, 1);
    benchmark("gemm, one token, 896x4864 weight", "gemm", one_up,
        [&]{ touch(one); tt::gemm(0, one_up, 1, one, false, w_ff, false); }, reps);
    benchmark("gemm, one token, 4864x896 weight", "gemm", one_down,
        [&]{ touch(one_up); tt::gemm(0, one_down, 1, one_up, false, w_back, false); }, reps);
    resizable_tensor few = random_tensor(rnd, 8, DM, 1, 1), few_up(8, FM, 1, 1);
    benchmark("gemm, 8 tokens, 896x4864 weight", "gemm", few_up,
        [&]{ touch(few); tt::gemm(0, few_up, 1, few, false, w_ff, false); }, reps);

    // Attention, one plane per sample and head.
    resizable_tensor q = random_tensor(rnd, B, H, T, HD), k = random_tensor(rnd, B, H, T, HD);
    resizable_tensor v = random_tensor(rnd, B, H, T, HD), scores(B, H, T, T), probs(B, H, T, T), att(B, H, T, HD);
    benchmark("gemm, attention scores", "gemm", scores,
        [&]{ touch(q); touch(k); tt::gemm(0, scores, 1.0f / std::sqrt(static_cast<float>(HD)), q, false, k, true,
            operation_mode::PLANE_WISE); }, reps);
    {
        float* s = scores.host();
        for (long p = 0; p < B * H; ++p)
            for (long r = 0; r < T; ++r)
                for (long c = r + 1; c < T; ++c)
                    s[(p * T + r) * T + c] = -std::numeric_limits<float>::infinity();
    }
    benchmark("softmax, causal scores", "softmax", probs,
        [&]{ tt::softmax(probs, scores, operation_mode::PLANE_WISE); }, reps);
    benchmark("gemm, attention output", "gemm", att,
        [&]{ touch(probs); touch(v); tt::gemm(0, att, 1, probs, false, v, false, operation_mode::PLANE_WISE); },
        reps);

    // Normalization and activations over the token rows.
    resizable_tensor rows = random_tensor(rnd, B, 1, T, D), gamma(1, 1, 1, D), normed, scale;
    gamma = 1;
    benchmark("rms_normalize", "rms_normalize", normed,
        [&]{ tt::rms_normalize(1e-6, normed, scale, rows, gamma, true); }, reps);
    resizable_tensor wide = random_tensor(rnd, B, 1, T, F), act(B, 1, T, F);
    benchmark("gelu, feed-forward width", "gelu", act, [&]{ tt::gelu(act, wide); }, reps);
    benchmark("silu, feed-forward width", "silu", act, [&]{ tt::silu(act, wide); }, reps);

    // The loss over the vocabulary.
    resizable_tensor logits = random_tensor(rnd, B, 1, T, V), grad(B, 1, T, V), ids(B, 1, T, 1);
    std::vector<matrix<unsigned long, 0, 1>> targets(B, matrix<unsigned long, 0, 1>(T));
    for (long i = 0; i < B; ++i)
        for (long t = 0; t < T; ++t)
        {
            ids.host()[i * T + t] = static_cast<float>(rnd.get_random_32bit_number() % V);
            targets[i](t) = rnd.get_random_32bit_number() % V;
        }
    cpu::compute_loss_cross_entropy_per_token cross_entropy;
    benchmark("cross entropy over the vocabulary", "cross_entropy", grad, [&]{
        grad = 0;
        double loss;
        cross_entropy(targets.begin(), ids, logits, grad, loss, -1, 0.0, -1, 0.0);
    }, reps);

    // A convolution of a vision tower.
    resizable_tensor image = random_tensor(rnd, 4, 32, 32, 32), filters = random_tensor(rnd, 64, 32, 3, 3), maps;
    tt::tensor_conv conv;
    conv.setup(image, filters, 1, 1, 1, 1);
    benchmark("conv 3x3, 32 to 64 channels, 32x32", "conv", maps,
        [&]{ touch(image); conv(false, maps, image, filters); }, reps);
    cout << "\n";
}

// ----------------------------------------------------------------------------------------

namespace
{
    // A byte-level decoder: 256 bytes and a padding token.
    using model_config = canonical_transformer_config<257, 2, 4, 128>;
    using train_net = model_config::network_type<true>;

    struct training_run
    {
        double ms_per_step = 0;        // over every step
        double ms_per_late_step = 0;   // over the second half, once graphs are compiled
        double loss = 0;
    };

    training_run train (
        train_net& net,
        const std::vector<matrix<int, 0, 1>>& samples,
        const std::vector<matrix<unsigned long, 0, 1>>& targets,
        long steps,
        long batch_size,
        npu::routing mode
    )
    {
        dnn_trainer<train_net, adam> trainer(net, adam(0.004, 0.9, 0.999));
        trainer.set_learning_rate(1e-3);
        trainer.be_quiet();
        npu::set_routing(mode);
        training_run run;
        const auto start = std::chrono::steady_clock::now();
        auto half = start;
        for (long s = 0; s < steps; ++s)
        {
            if (s == steps / 2)
            {
                trainer.get_net();
                half = std::chrono::steady_clock::now();
            }
            const size_t first = static_cast<size_t>(s * batch_size) % (samples.size() - batch_size);
            const std::vector<matrix<int, 0, 1>> x(samples.begin() + first, samples.begin() + first + batch_size);
            const std::vector<matrix<unsigned long, 0, 1>> y(targets.begin() + first,
                targets.begin() + first + batch_size);
            trainer.train_one_step(x, y);
        }
        trainer.get_net();
        const auto end = std::chrono::steady_clock::now();
        run.ms_per_step = std::chrono::duration<double, std::milli>(end - start).count() / steps;
        run.ms_per_late_step = std::chrono::duration<double, std::milli>(end - half).count() / (steps - steps / 2);
        run.loss = trainer.get_average_loss();
        return run;
    }
}

void run_training (long steps)
{
    cout << "=== TRAINING ===\n";
    const long window = 64, batch_size = 32;
    std::vector<int> tokens;
    for (const unsigned char c : get_dataset_as_text(dataset_id::SHAKESPEARE_EXTRACT))
        tokens.push_back(c);
    std::vector<matrix<int, 0, 1>> samples;
    std::vector<matrix<unsigned long, 0, 1>> targets;
    build_single_token_prediction_dataset(std::vector<std::vector<int>>{ tokens }, window, 256, false,
        samples, targets);
    if (samples.size() <= static_cast<size_t>(batch_size))
    {
        cout << "The dataset is too small for this run.\n\n";
        return;
    }
    cout << model_config::model_info::describe() << "\n";
    cout << samples.size() << " windows of " << window << " bytes, batches of " << batch_size << ", "
         << steps << " steps per run.\n";

    // Both runs start from the same weights.
    const npu::routing initial = npu::get_routing();
    npu::set_routing(npu::routing::never);
    train_net host_net;
    host_net(samples[0]);
    train_net routed_net = host_net;

    const training_run host = train(host_net, samples, targets, steps, batch_size, npu::routing::never);
    npu::reset_statistics();
    const training_run routed = train(routed_net, samples, targets, steps, batch_size, npu::routing::automatic);

    cout << fixed << setprecision(1);
    cout << "  host only        : " << setw(8) << host.ms_per_step << " ms per step, " << setw(8)
         << host.ms_per_late_step << " ms over the second half, loss " << setprecision(4) << host.loss << "\n";
    cout << setprecision(1);
    cout << "  automatic routing: " << setw(8) << routed.ms_per_step << " ms per step, " << setw(8)
         << routed.ms_per_late_step << " ms over the second half, loss " << setprecision(4) << routed.loss << "\n";
    cout.unsetf(std::ios::floatfield);
    cout << "The first steps of the routed run also pay for compiling graphs. With the graph cache on\n"
         << "(DLIB_NPU_CACHE_DIR), the next run starts from them. Routing summary of that run:\n";
    npu::print_status(cout);
    npu::set_routing(initial);
    cout << "\n";
}

// ----------------------------------------------------------------------------------------

int main (int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("info", "Report the build, the device found and how it computes");
        parser.add_option("benchmark", "Time the primitives of a transformer on the host and on the device");
        parser.add_option("train", "Compare training steps with and without routing");
        parser.add_option("reps", "Calls timed per primitive (default: 10)", 1);
        parser.add_option("steps", "Training steps per run (default: 60)", 1);
        parser.add_option("h", "Display this help");
        parser.parse(argc, argv);
        if (parser.option("h"))
        {
            parser.print_options();
            return 0;
        }

        const int reps = get_option(parser, "reps", 10);
        const long steps = get_option(parser, "steps", 60);
        const bool all = !parser.option("info") && !parser.option("benchmark") && !parser.option("train");

        if (all || parser.option("info"))
            show_device();
        if (all || parser.option("benchmark"))
            run_benchmark(std::max(1, reps));
        if (all || parser.option("train"))
            run_training(std::max(2L, steps));
        return 0;
    }
    catch (exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}
