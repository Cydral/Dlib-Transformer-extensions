// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/cuda/avx2_dlib.h>
#include <dlib/cuda/cpu_dlib.h>
#include <dlib/cuda/cpu_gguf.h>
#include <dlib/cuda/gguf_blocks.h>
#include <dlib/cuda/tensor_tools.h>
#include <dlib/rand.h>
#include <dlib/simd/cpu_features.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.avx2");

    // The largest difference between two tensors, each difference divided by 1 + |value|.
    float worst (const tensor& a, const tensor& b)
    {
        float w = 0;
        const float* x = a.host();
        const float* y = b.host();
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (std::isnan(x[i]) || std::isnan(y[i]))
            {
                if (std::isnan(x[i]) != std::isnan(y[i])) return std::numeric_limits<float>::infinity();
                continue;
            }
            w = std::max(w, std::abs(x[i] - y[i]) / (1 + std::abs(x[i])));
        }
        return w;
    }

    // Values of every size, with the bounds of the approximations and values that are not numbers.
    void fill (tensor& t, dlib::rand& rnd, bool with_nan)
    {
        const float marks[] = { 0.0f, -0.0f, 1e-8f, -1e-8f, 0.624f, 0.626f, -0.625f, 5.0f, -5.0f, 20.0f, -20.0f, 44.0f,
            -44.0f, 88.0f, -88.0f, 100.0f, -100.0f, 1e6f, -1e6f };
        float* p = t.host_write_only();
        for (size_t i = 0; i < t.size(); ++i)
        {
            const double u = rnd.get_random_double();
            if (u < 0.1) p[i] = marks[rnd.get_random_32bit_number() % (sizeof(marks) / sizeof(marks[0]))];
            else if (u < 0.5) p[i] = static_cast<float>(rnd.get_random_gaussian());
            else p[i] = static_cast<float>(8 * rnd.get_random_gaussian());
        }
        if (with_nan && t.size() > 3)
        {
            p[1] = std::numeric_limits<float>::quiet_NaN();
            p[t.size() - 2] = std::numeric_limits<float>::infinity();
            p[t.size() / 2] = -std::numeric_limits<float>::infinity();
        }
    }

    void test_activations (
    )
    {
        dlib::rand rnd;
        // Sizes on both sides of a vector of eight values.
        const long sizes[] = { 1, 7, 8, 9, 31, 64, 1001 };
        for (const long n : sizes)
        {
            resizable_tensor src(1, n), a(1, n), b(1, n);
            fill(src, rnd, n > 8);
            DLIB_TEST(avx2::tanh(a, src));
            cpu::tanh(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "tanh " << n << ": " << worst(b, a));
            DLIB_TEST(avx2::sigmoid(a, src));
            cpu::sigmoid(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "sigmoid " << n << ": " << worst(b, a));
            DLIB_TEST(avx2::silu(a, src));
            cpu::silu(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "silu " << n << ": " << worst(b, a));
            DLIB_TEST(avx2::gelu(a, src));
            cpu::gelu(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "gelu " << n << ": " << worst(b, a));
            // In place, as the layers call them.
            a = src;
            DLIB_TEST(avx2::tanh(a, a));
            cpu::tanh(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "tanh in place " << n << ": " << worst(b, a));
        }
    }

    void test_softmax (
    )
    {
        dlib::rand rnd;
        const long shapes[][4] = { { 2, 3, 4, 5 }, { 3, 17, 1, 1 }, { 1, 2, 3, 70 }, { 4, 1, 1, 9 }, { 2, 5, 2, 1 } };
        for (const auto& s : shapes)
        {
            resizable_tensor src(s[0], s[1], s[2], s[3]), a, b;
            a.copy_size(src);
            b.copy_size(src);
            float* p = src.host_write_only();
            for (size_t i = 0; i < src.size(); ++i) p[i] = static_cast<float>(6 * rnd.get_random_gaussian());
            /*
                A row far below its largest value, a row with a value that is not a number, and
                rows that hold minus infinity alone, as those of a masked attention, or plus
                infinity.
            */
            if (src.size() >= 12)
            {
                p[0] = 200;
                p[src.size() - 1] = std::numeric_limits<float>::quiet_NaN();
                for (long c = 0; c < s[3]; ++c) p[s[3] + c] = -std::numeric_limits<float>::infinity();
                p[2 * s[3]] = std::numeric_limits<float>::infinity();
            }
            DLIB_TEST(avx2::softmax(a, src, operation_mode::PLANE_WISE));
            cpu::softmax(b, src, operation_mode::PLANE_WISE);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "softmax over the columns: " << worst(b, a));
            const bool lines = s[2] * s[3] == 1;
            DLIB_TEST(avx2::softmax(a, src, operation_mode::CHANNEL_WISE) == lines);
            if (lines)
            {
                cpu::softmax(b, src, operation_mode::CHANNEL_WISE);
                DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "softmax over the channels: " << worst(b, a));
            }
            DLIB_TEST(avx2::softmax_all(a, src));
            cpu::softmax_all(b, src);
            DLIB_TEST_MSG(worst(b, a) <= 2e-6f, "softmax over a sample: " << worst(b, a));
            // The tensor tools choose the same functions.
            tt::softmax(b, src, operation_mode::PLANE_WISE);
            avx2::softmax(a, src, operation_mode::PLANE_WISE);
#if !defined(DLIB_USE_CUDA) && !defined(DLIB_USE_NPU)
            DLIB_TEST(worst(b, a) == 0);
#endif
        }
    }

    void test_gguf_matvec (
    )
    {
        namespace gb = dlib::gguf_blocks;
        dlib::rand rnd;
        const int types[] = { gb::f32, gb::f16, gb::bf16, gb::q4_0, gb::q4_1, gb::q5_0, gb::q5_1, gb::q8_0, gb::iq4_nl,
            gb::q4_K, gb::q5_K, gb::q6_K };
        for (const int type : types)
        {
            // A small product on the calling thread, then one that the threads share.
            const long shapes[][3] = { { 5, 256, 1 }, { 3, 512, 70 }, { 600, 768, 2 } };
            for (const auto& s : shapes)
            {
                const long rows = s[0], cols = s[1], n = s[2];
                const size_t rb = gb::row_bytes(type, cols);
                std::vector<uint8_t> bytes(static_cast<size_t>(rows) * rb);
                if (type == gb::f32)
                {
                    std::vector<float> values(static_cast<size_t>(rows) * cols);
                    for (float& v : values) v = static_cast<float>(rnd.get_random_gaussian());
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                }
                else if (type == gb::f16 || type == gb::bf16)
                {
                    // Halves and upper halves of floats with small exponents.
                    for (size_t i = 0; i < bytes.size(); i += 2)
                    {
                        bytes[i] = static_cast<uint8_t>(rnd.get_random_8bit_number());
                        bytes[i + 1] = static_cast<uint8_t>((rnd.get_random_8bit_number() & 0x83) | 0x3C);
                    }
                }
                else
                {
                    for (auto& b : bytes) b = static_cast<uint8_t>(rnd.get_random_8bit_number());
                    // The scales of the blocks are halves: their exponents stay small.
                    const size_t block = type == gb::q4_K ? 144 : type == gb::q5_K ? 176 : type == gb::q6_K ? 210
                        : rb * 32 / cols;
                    for (size_t at = 0; at + block <= bytes.size(); at += block)
                    {
                        const size_t scale = type == gb::q6_K ? at + 208 : at;
                        bytes[scale + 1] = static_cast<uint8_t>((bytes[scale + 1] & 0x83) | 0x3C);
                        if (type == gb::q4_K || type == gb::q5_K || type == gb::q4_1 || type == gb::q5_1)
                            bytes[scale + 3] = static_cast<uint8_t>((bytes[scale + 3] & 0x83) | 0x3C);
                    }
                }
                resizable_tensor w(1, static_cast<long long>((bytes.size() + 3) / 4));
                std::memcpy(w.host_write_only(), bytes.data(), bytes.size());
                resizable_tensor x(n, cols), a(n, rows), b(n, rows);
                for (float& v : x) v = static_cast<float>(rnd.get_random_gaussian());
                DLIB_TEST(avx2::gguf_matvec(a, x, w, type, rows, cols, rb));
                cpu::gguf_matvec(b, x, w, type, rows, cols, rb);
                float scale = 1;
                for (const float v : b) scale = std::max(scale, std::abs(v));
                float diff = 0;
                for (size_t i = 0; i < a.size(); ++i) diff = std::max(diff, std::abs(a.host()[i] - b.host()[i]));
                DLIB_TEST_MSG(diff <= 1e-4f * scale, "gguf_matvec, format " << type << ", " << rows << "x" << cols
                    << ", " << n << " vectors: " << diff << " for " << scale);
            }
        }
    }

    class avx2_tester : public tester
    {
    public:
        avx2_tester (
        ) :
            tester ("test_avx2",
                    "Runs tests on the functions of the tensor tools written for AVX2 against those of cpu_dlib.h.")
        {}

        void perform_test (
        )
        {
            if (!avx2::is_available())
            {
                dlog << LINFO << "no AVX2, FMA and F16C on this processor or in this build ("
                    << describe(get_cpu_features()) << "): the functions decline";
                resizable_tensor a(1, 8), b(1, 8);
                a = 1;
                b = 0;
                DLIB_TEST(!avx2::tanh(b, a) && !avx2::gelu(b, a) && !avx2::softmax_all(b, a));
                DLIB_TEST(sum(mat(b)) == 0);
                return;
            }
            print_spinner();
            test_activations();
            print_spinner();
            test_softmax();
            print_spinner();
            test_gguf_matvec();
        }
    } a;

}
