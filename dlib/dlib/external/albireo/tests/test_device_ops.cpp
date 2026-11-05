// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the operations of dlib on weight matrices kept in their GGUF formats
    (dlib/cuda/gguf_blocks.h) against the processor kernels of the engine.

        test_device_ops [card]

    With CUDA, the operations run on the card given, or on the one with the most memory;
    otherwise on the processor, through the host code of dlib.  For every format:
        - tt::gguf_dequantize() against qk_dequantize_row(), row by row;
        - tt::gguf_matvec() for 1 to 20 tokens and for 130 tokens, against products in double
          precision with the rows qk_dequantize_row() expands, with activations that start on
          16 bytes and with activations that do not;
    then tt::rms_norm_rows() and tt::silu_multiply() against their definitions.  The program
    ends with the time of the products of a generated token with the matrices of a model of
    2 billion parameters, and the rate at which they read the weights.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include <dlib/cuda/gguf_blocks.h>
#include <dlib/cuda/tensor_tools.h>
#ifdef DLIB_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "../engine/qkernels.h"

using namespace albireo;
namespace gb = dlib::gguf_blocks;

namespace
{
    int failures = 0;

    void expect(bool ok, const std::string& what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    struct format
    {
        int type;
        const char* name;
        // The offsets of the scales in half precision within a block, and the bytes of a block.
        std::vector<size_t> halves;
        size_t block_bytes;
        long block_values;
    };

    const std::vector<format>& formats()
    {
        static const std::vector<format> list = {
            { gb::f32, "F32", {}, 4, 1 },
            { gb::f16, "F16", {}, 2, 1 },
            { gb::bf16, "BF16", {}, 2, 1 },
            { gb::q4_0, "Q4_0", { 0 }, 18, 32 },
            { gb::q4_1, "Q4_1", { 0, 2 }, 20, 32 },
            { gb::q5_0, "Q5_0", { 0 }, 22, 32 },
            { gb::q5_1, "Q5_1", { 0, 2 }, 24, 32 },
            { gb::q8_0, "Q8_0", { 0 }, 34, 32 },
            { gb::iq4_nl, "IQ4_NL", { 0 }, 18, 32 },
            { gb::q4_K, "Q4_K", { 0, 2 }, 144, 256 },
            { gb::q5_K, "Q5_K", { 0, 2 }, 176, 256 },
            { gb::q6_K, "Q6_K", { 208 }, 210, 256 }
        };
        return list;
    }

    void put_u16(uint8_t* p, uint16_t v)
    {
        p[0] = static_cast<uint8_t>(v & 0xFF);
        p[1] = static_cast<uint8_t>(v >> 8);
    }

    /*
        A row of cols values in format f: random bytes, the values of the floating formats
        within [-1, 1], and the scales of the blocks finite and small enough for products of
        a few units.
    */
    void random_row(const format& f, uint8_t* row, long cols, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        std::uniform_int_distribution<int> byte(0, 255);
        if (f.type == gb::f32)
        {
            for (long i = 0; i < cols; ++i)
            {
                const float v = unit(rng);
                std::memcpy(row + 4 * i, &v, 4);
            }
            return;
        }
        if (f.type == gb::f16 || f.type == gb::bf16)
        {
            for (long i = 0; i < cols; ++i)
            {
                const float v = unit(rng);
                uint32_t bits;
                std::memcpy(&bits, &v, 4);
                put_u16(row + 2 * i, f.type == gb::f16 ? qk_float_to_half(v) : static_cast<uint16_t>(bits >> 16));
            }
            return;
        }
        const long blocks = cols / f.block_values;
        for (size_t i = 0; i < static_cast<size_t>(blocks) * f.block_bytes; ++i)
            row[i] = static_cast<uint8_t>(byte(rng));
        std::uniform_real_distribution<float> scale(0.004f, 0.012f);
        for (long b = 0; b < blocks; ++b)
            for (size_t at : f.halves)
            {
                const float s = scale(rng) * (byte(rng) < 128 ? -1.0f : 1.0f);
                put_u16(row + static_cast<size_t>(b) * f.block_bytes + at, qk_float_to_half(s));
            }
    }

    void random_floats(float* p, size_t n, float range, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> d(-range, range);
        for (size_t i = 0; i < n; ++i) p[i] = d(rng);
    }

    // The tensor that holds the bytes of a matrix: dlib tensors hold floats.
    void load_weights(dlib::resizable_tensor& w, const std::vector<uint8_t>& bytes)
    {
        w.set_size(static_cast<long long>((bytes.size() + 3) / 4));
        float* p = w.host_write_only();
        std::memset(p, 0, w.size() * sizeof(float));
        std::memcpy(p, bytes.data(), bytes.size());
    }

    void check_format(const format& f, long rows, long cols, std::mt19937& rng)
    {
        const size_t rb = gb::row_bytes(f.type, cols);
        const std::string what = std::string(f.name) + " " + std::to_string(rows) + "x" + std::to_string(cols);
        expect(rb == qk_row_bytes(static_cast<dlib::ggml_type>(f.type), cols), what + ": row bytes");
        std::vector<uint8_t> bytes(static_cast<size_t>(rows) * rb);
        for (long r = 0; r < rows; ++r) random_row(f, bytes.data() + static_cast<size_t>(r) * rb, cols, rng);
        dlib::resizable_tensor w;
        load_weights(w, bytes);

        // The reference rows.
        std::vector<float> ref(static_cast<size_t>(rows * cols));
        for (long r = 0; r < rows; ++r)
            qk_dequantize_row(static_cast<dlib::ggml_type>(f.type), bytes.data() + static_cast<size_t>(r) * rb,
                ref.data() + r * cols, cols);

        // Expansion of all the rows, then of rows in the middle.
        const long firsts[] = { 0, 5 };
        for (long first : firsts)
        {
            const long count = first == 0 ? rows : std::min<long>(17, rows - first);
            dlib::resizable_tensor out(count, cols);
            dlib::tt::gguf_dequantize(out, w, f.type, first, count, cols, rb);
            const float* got = out.host();
            double worst = 0;
            for (long i = 0; i < count * cols; ++i)
            {
                const double a = ref[static_cast<size_t>(first * cols + i)];
                worst = std::max(worst, std::fabs(got[i] - a) / std::max(1.0, std::fabs(a)));
            }
            expect(worst <= 1e-5, what + ": expansion from row " + std::to_string(first) + " (relative error "
                + std::to_string(worst) + ")");
        }

        // Products, with activations on 16 bytes and with activations one float further.
        const long tokens[] = { 1, 2, 3, 4, 5, 8, 9, 16, 20, 130 };
        double worst = 0;
        for (long n : tokens)
            for (int shift = 0; shift < 2; ++shift)
            {
                dlib::resizable_tensor x(n * cols + shift);
                random_floats(x.host_write_only(), x.size(), 1.0f, rng);
                dlib::alias_tensor xa(n, cols);
                const dlib::alias_tensor_instance xs = xa(x, static_cast<size_t>(shift));
                dlib::resizable_tensor y(n, rows);
                dlib::tt::gguf_matvec(y, xs, w, f.type, rows, cols, rb);
                const float* got = y.host();
                const float* xp = x.host() + shift;
                for (long t = 0; t < n; ++t)
                    for (long r = 0; r < rows; ++r)
                    {
                        double sum = 0, magnitude = 0;
                        for (long i = 0; i < cols; ++i)
                        {
                            const double term = static_cast<double>(ref[static_cast<size_t>(r * cols + i)])
                                * xp[t * cols + i];
                            sum += term;
                            magnitude += std::fabs(term);
                        }
                        const double err = std::fabs(got[t * rows + r] - sum) / (magnitude + 1e-6);
                        worst = std::max(worst, err);
                    }
            }
        expect(worst <= 2e-5, what + ": products (relative error " + std::to_string(worst) + ")");
        std::printf("  %-7s %4ldx%-5ld products within %.2g of their magnitude\n", f.name, rows, cols, worst);
    }

    void check_row_operations(std::mt19937& rng)
    {
        const long rows = 7, width = 96;
        dlib::resizable_tensor x(rows, width), gamma(width), y(rows, width);
        random_floats(x.host_write_only(), x.size(), 3.0f, rng);
        random_floats(gamma.host_write_only(), gamma.size(), 2.0f, rng);
        const float eps = 1e-6f;
        std::vector<float> expected(x.size());
        {
            const float* xp = x.host();
            const float* g = gamma.host();
            for (long r = 0; r < rows; ++r)
            {
                double s = 0;
                for (long i = 0; i < width; ++i) s += static_cast<double>(xp[r * width + i]) * xp[r * width + i];
                const double scale = 1.0 / std::sqrt(s / width + eps);
                for (long i = 0; i < width; ++i)
                    expected[static_cast<size_t>(r * width + i)] = static_cast<float>(xp[r * width + i] * scale * g[i]);
            }
        }
        dlib::tt::rms_norm_rows(y, x, gamma, eps);
        double worst = 0;
        for (size_t i = 0; i < y.size(); ++i)
            worst = std::max<double>(worst,
                std::fabs(y.host()[i] - expected[i]) / std::max(1.0f, std::fabs(expected[i])));
        expect(worst <= 1e-5, "rms_norm_rows (relative error " + std::to_string(worst) + ")");
        // In place.
        dlib::tt::rms_norm_rows(x, x, gamma, eps);
        double worst_in_place = 0;
        for (size_t i = 0; i < x.size(); ++i)
            worst_in_place = std::max<double>(worst_in_place,
                std::fabs(x.host()[i] - expected[i]) / std::max(1.0f, std::fabs(expected[i])));
        expect(worst_in_place <= 1e-5, "rms_norm_rows in place (relative error " + std::to_string(worst_in_place)
            + ")");

        const long n = 4099;
        dlib::resizable_tensor g(n), u(n);
        random_floats(g.host_write_only(), g.size(), 8.0f, rng);
        random_floats(u.host_write_only(), u.size(), 2.0f, rng);
        std::vector<float> want(static_cast<size_t>(n));
        for (long i = 0; i < n; ++i)
        {
            const double a = g.host()[i];
            want[static_cast<size_t>(i)] = static_cast<float>(a / (1.0 + std::exp(-a)) * u.host()[i]);
        }
        dlib::tt::silu_multiply(g, u);
        double worst_silu = 0;
        for (long i = 0; i < n; ++i)
            worst_silu = std::max<double>(worst_silu, std::fabs(g.host()[i] - want[static_cast<size_t>(i)])
                / std::max(1.0f, std::fabs(want[static_cast<size_t>(i)])));
        expect(worst_silu <= 1e-5, "silu_multiply (relative error " + std::to_string(worst_silu) + ")");
        std::printf("  rms_norm_rows %.2g, in place %.2g, silu_multiply %.2g\n", worst, worst_in_place, worst_silu);
    }

    /*
        The products of one generated token with the matrices of a layer of a model of 2
        billion parameters in Q4_K and Q6_K, and with its output matrix: the time of each
        and the rate at which they read the weights.
    */
    void measure(std::mt19937& rng)
    {
        struct shape
        {
            const char* name;
            int type;
            long rows, cols;
        };
        const shape shapes[] = {
            { "attn_q   Q4_K", gb::q4_K, 2048, 2048 },
            { "ffn_up   Q4_K", gb::q4_K, 6144, 2048 },
            { "ffn_down Q6_K", gb::q6_K, 2048, 6144 },
            { "output   Q6_K", gb::q6_K, 151936, 2048 }
        };
#ifdef DLIB_USE_CUDA
        const int repeats = 50;
#else
        const int repeats = 1;
#endif
        for (const shape& s : shapes)
        {
#ifndef DLIB_USE_CUDA
            if (s.rows > 10000) continue;
#endif
            const format* f = nullptr;
            for (const format& c : formats())
                if (c.type == s.type) f = &c;
            const size_t rb = gb::row_bytes(s.type, s.cols);
            std::vector<uint8_t> bytes(static_cast<size_t>(s.rows) * rb);
            for (long r = 0; r < s.rows; ++r) random_row(*f, bytes.data() + static_cast<size_t>(r) * rb, s.cols, rng);
            dlib::resizable_tensor w;
            load_weights(w, bytes);
            std::vector<uint8_t>().swap(bytes);
            for (long n : { 1L, 8L })
            {
                dlib::resizable_tensor x(n, s.cols), y(n, s.rows);
                random_floats(x.host_write_only(), x.size(), 1.0f, rng);
                // The first product moves the weights to the device.
                dlib::tt::gguf_matvec(y, x, w, s.type, s.rows, s.cols, rb);
                y.host();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < repeats; ++i) dlib::tt::gguf_matvec(y, x, w, s.type, s.rows, s.cols, rb);
                y.host();
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                    / repeats;
                std::printf("  %s %6ldx%-5ld %ld token%s: %8.1f us, %7.1f GB/s of weights\n", s.name, s.rows, s.cols,
                    n, n > 1 ? "s" : " ", seconds * 1e6, static_cast<double>(s.rows) * rb / seconds / 1e9);
            }
        }
    }
}

static int run_checks(int argc, char** argv)
{
#ifdef DLIB_USE_CUDA
    int card = argc > 1 ? std::atoi(argv[1]) : -1;
    if (card < 0)
    {
        // The card with the most memory, as the engine chooses it.
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
        size_t most = 0;
        for (int i = 0; i < count; ++i)
        {
            cudaDeviceProp p;
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess && (card < 0 || p.totalGlobalMem > most))
            {
                most = p.totalGlobalMem;
                card = i;
            }
        }
    }
    if (card < 0)
    {
        std::printf("no CUDA device\n");
        return 2;
    }
    dlib::cuda::set_device(card);
    std::printf("device %d: %s\n", card, dlib::cuda::get_device_name(card).c_str());
#else
    (void)argc;
    (void)argv;
    std::printf("host code of dlib\n");
#endif
    std::mt19937 rng(20261104u);
    const long shapes[][2] = { { 37, 256 }, { 9, 768 }, { 70, 2304 } };
    for (const format& f : formats())
        for (const auto& s : shapes) check_format(f, s[0], s[1], rng);
    check_row_operations(rng);
    measure(rng);
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
