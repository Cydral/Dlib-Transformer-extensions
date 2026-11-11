// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the quantized kernels of the engine against the reference dequantization of
    dlib (dlib/data_io/gguf_dequantize.h).

    For every weight format, random blocks are generated, a row of activations is
    quantized in the format the kernel expects, and the integer dot product is compared
    with the double precision dot product of the dequantized weights and the
    dequantized activations.  The two only differ by rounding, so the tolerance is tight:
    any error in the bit layout of a format shows up as a gross mismatch.

    The products, the conversions and the vector helpers of the attention are checked
    the same way.  The program returns 0 when every check passes.
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <dlib/data_io/gguf_dequantize.h>

#include "../engine/qkernels.h"
#include "../engine/thread_pool.h"

using namespace albireo;
using dlib::ggml_type;

namespace
{
    int failures = 0;

    void check(bool ok, const char* what, double a = 0, double b = 0)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s (%.8g vs %.8g)\n", what, a, b);
        }
    }

    struct fmt
    {
        ggml_type type;
        const char* name;
        long block;
        long bytes;
        std::vector<long> half_offsets;   // scale fields that must hold finite halves
    };

    std::vector<fmt> formats()
    {
        return {
            { ggml_type::Q4_0, "Q4_0", 32, 18, { 0 } },
            { ggml_type::Q4_1, "Q4_1", 32, 20, { 0, 2 } },
            { ggml_type::Q5_0, "Q5_0", 32, 22, { 0 } },
            { ggml_type::Q5_1, "Q5_1", 32, 24, { 0, 2 } },
            { ggml_type::Q8_0, "Q8_0", 32, 34, { 0 } },
            { ggml_type::IQ4_NL, "IQ4_NL", 32, 18, { 0 } },
            { ggml_type::Q2_K, "Q2_K", 256, 84, { 80, 82 } },
            { ggml_type::Q3_K, "Q3_K", 256, 110, { 108 } },
            { ggml_type::Q4_K, "Q4_K", 256, 144, { 0, 2 } },
            { ggml_type::Q5_K, "Q5_K", 256, 176, { 0, 2 } },
            { ggml_type::Q6_K, "Q6_K", 256, 210, { 208 } },
            { ggml_type::F16, "F16", 1, 2, {} },
            { ggml_type::BF16, "BF16", 1, 2, {} },
            { ggml_type::F32, "F32", 1, 4, {} },
        };
    }

    std::vector<uint8_t> random_row(const fmt& f, long n, std::mt19937& rng)
    {
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> scale(0.002f, 0.05f);
        std::uniform_real_distribution<float> value(-1.0f, 1.0f);
        const long blocks = n / f.block;
        std::vector<uint8_t> row(static_cast<size_t>(blocks * f.bytes));
        if (f.type == ggml_type::F32)
        {
            for (long i = 0; i < n; ++i)
            {
                const float v = value(rng);
                std::memcpy(&row[static_cast<size_t>(4 * i)], &v, 4);
            }
            return row;
        }
        if (f.type == ggml_type::F16 || f.type == ggml_type::BF16)
        {
            for (long i = 0; i < n; ++i)
            {
                const float v = value(rng);
                uint16_t h;
                if (f.type == ggml_type::F16)
                {
                    h = qk_float_to_half(v);
                }
                else
                {
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    h = static_cast<uint16_t>(bits >> 16);
                }
                std::memcpy(&row[static_cast<size_t>(2 * i)], &h, 2);
            }
            return row;
        }
        for (auto& b : row) b = static_cast<uint8_t>(byte(rng));
        for (long b = 0; b < blocks; ++b)
            for (long off : f.half_offsets)
            {
                const uint16_t h = qk_float_to_half(scale(rng));
                std::memcpy(&row[static_cast<size_t>(b * f.bytes + off)], &h, 2);
            }
        return row;
    }

    // The activation values a quantized activation row stands for.
    std::vector<double> activation_values(act_format f, const std::vector<uint8_t>& act, const std::vector<float>& x)
    {
        const long n = static_cast<long>(x.size());
        std::vector<double> v(static_cast<size_t>(n));
        if (f == act_format::f32)
        {
            for (long i = 0; i < n; ++i) v[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
        }
        else if (f == act_format::q8_0)
        {
            // float d, float s, int8 qs[32]
            for (long b = 0; b < n / 32; ++b)
            {
                float d;
                std::memcpy(&d, &act[static_cast<size_t>(b * 40)], 4);
                for (long j = 0; j < 32; ++j)
                {
                    const int8_t q = static_cast<int8_t>(act[static_cast<size_t>(b * 40 + 8 + j)]);
                    v[static_cast<size_t>(b * 32 + j)] = d * q;
                }
            }
        }
        else
        {
            // float d, int8 qs[256], int16 bsums[16]
            for (long b = 0; b < n / 256; ++b)
            {
                float d;
                std::memcpy(&d, &act[static_cast<size_t>(b * 292)], 4);
                for (long j = 0; j < 256; ++j)
                {
                    const int8_t q = static_cast<int8_t>(act[static_cast<size_t>(b * 292 + 4 + j)]);
                    v[static_cast<size_t>(b * 256 + j)] = d * q;
                }
            }
        }
        return v;
    }

    void test_dots(std::mt19937& rng)
    {
        std::uniform_real_distribution<float> value(-2.0f, 2.0f);
        for (const fmt& f : formats())
        {
            double worst = 0;
            for (int trial = 0; trial < 50; ++trial)
            {
                // An odd number of blocks, or a length that no vector divides, every other
                // trial: the kernels that take two blocks or 16 values at a time end alone.
                const long n = trial % 2 == 0 ? 1024 : f.block == 256 ? 1280 : f.block == 32 ? 1056 : 1003;
                const std::vector<uint8_t> w = random_row(f, n, rng);
                std::vector<float> x(static_cast<size_t>(n));
                for (auto& e : x) e = value(rng);
                const act_format af = qk_activation_format(f.type);
                std::vector<uint8_t> act(qk_activation_row_bytes(af, n));
                qk_quantize_row(af, x.data(), act.data(), n);
                const std::vector<double> a = activation_values(af, act, x);

                std::vector<float> wf(static_cast<size_t>(n));
                qk_dequantize_row(f.type, w.data(), wf.data(), n);
                double ref = 0, mag = 0;
                for (long i = 0; i < n; ++i)
                {
                    ref += static_cast<double>(wf[static_cast<size_t>(i)]) * a[static_cast<size_t>(i)];
                    mag += std::fabs(static_cast<double>(wf[static_cast<size_t>(i)]) * a[static_cast<size_t>(i)]);
                }
                const double got = qk_dot(f.type, w.data(), act.data(), n);
                const double err = std::fabs(got - ref) / (mag + 1e-12);
                worst = std::max(worst, err);
                check(err < 2e-5, f.name, got, ref);
                const double port = qk_dot_portable(f.type, w.data(), act.data(), n);
                check(std::fabs(port - ref) / (mag + 1e-12) < 2e-5, "portable kernel", port, ref);
            }
            std::printf("  dot %-7s worst relative error %.3g\n", f.name, worst);
        }
    }

    // The expansion of the formats with a vector kernel of their own, bit for bit that of dlib.
    void test_dequantization(std::mt19937& rng)
    {
        for (const fmt& f : formats())
        {
            if (f.type != ggml_type::Q6_K) continue;
            long differ = 0;
            for (int trial = 0; trial < 50; ++trial)
            {
                const long n = trial % 2 == 0 ? 1024 : 1280;
                const std::vector<uint8_t> w = random_row(f, n, rng);
                std::vector<float> got(static_cast<size_t>(n)), ref(static_cast<size_t>(n));
                qk_dequantize_row(f.type, w.data(), got.data(), n);
                for (long b = 0; b < n / f.block; ++b)
                    dlib::gguf_dq_q6_K(w.data() + b * f.bytes, ref.data() + b * f.block);
                if (std::memcmp(got.data(), ref.data(), sizeof(float) * got.size()) != 0) ++differ;
            }
            check(differ == 0, "expansion bit for bit with dlib", static_cast<double>(differ), 0);
            std::printf("  expansion %-7s compared with dlib on 50 rows\n", f.name);
        }
    }

    void test_activation_quantization(std::mt19937& rng)
    {
        std::uniform_real_distribution<float> value(-3.0f, 3.0f);
        const long n = 512;
        std::vector<float> x(static_cast<size_t>(n));
        for (auto& e : x) e = value(rng);
        for (act_format af : { act_format::q8_0, act_format::q8_k })
        {
            std::vector<uint8_t> act(qk_activation_row_bytes(af, n));
            qk_quantize_row(af, x.data(), act.data(), n);
            const std::vector<double> a = activation_values(af, act, x);
            double worst = 0;
            for (long i = 0; i < n; ++i)
                worst = std::max(worst, std::fabs(a[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]));
            // Half a quantum of a block whose largest value is 3.
            const double bound = 3.0 / 127.0 * 0.501;
            check(worst <= bound, "activation quantization error", worst, bound);
            if (af == act_format::q8_0)
            {
                // The scaled sum of the quanta, used by the asymmetric formats.
                for (long b = 0; b < n / 32; ++b)
                {
                    float s;
                    std::memcpy(&s, &act[static_cast<size_t>(b * 40 + 4)], 4);
                    double sum = 0;
                    for (long j = 0; j < 32; ++j) sum += a[static_cast<size_t>(b * 32 + j)];
                    check(std::fabs(s - sum) <= 1e-4 * (std::fabs(sum) + 1), "q8_0 block sum", s, sum);
                }
            }
            else
            {
                for (long b = 0; b < n / 256; ++b)
                    for (long g = 0; g < 16; ++g)
                    {
                        int16_t bs;
                        std::memcpy(&bs, &act[static_cast<size_t>(b * 292 + 260 + 2 * g)], 2);
                        int sum = 0;
                        for (long j = 0; j < 16; ++j)
                            sum += static_cast<int8_t>(act[static_cast<size_t>(b * 292 + 4 + g * 16 + j)]);
                        check(bs == sum, "q8_K group sum", bs, sum);
                    }
            }
        }
    }

    void test_matmul_case(const fmt& f, long cols, std::mt19937& rng, thread_pool& pool)
    {
        std::uniform_real_distribution<float> value(-1.0f, 1.0f);
        const long rows = 77;
        std::vector<uint8_t> data;
        for (long r = 0; r < rows; ++r)
        {
            const std::vector<uint8_t> row = random_row(f, cols, rng);
            data.insert(data.end(), row.begin(), row.end());
        }
        weight_matrix W;
        W.type = f.type;
        W.data = data.data();
        W.rows = rows;
        W.cols = cols;
        W.row_bytes = qk_row_bytes(f.type, cols);
        check(W.row_bytes * rows == data.size(), "row bytes");

        for (long tokens : { 1L, 2L, 3L, 4L, 5L, 7L, 8L, 37L, 100L, 150L, 300L })
        {
            const long xs = cols + 5, ys = rows + 3;
            std::vector<float> X(static_cast<size_t>(tokens * xs));
            for (auto& e : X) e = value(rng);
            std::vector<float> Y(static_cast<size_t>(tokens * ys), 7.0f);
            qk_scratch scratch;
            qk_matmul(W, X.data(), tokens, xs, Y.data(), ys, pool, scratch);
            std::vector<float> Y2 = Y;
            qk_matmul(W, X.data(), tokens, xs, Y2.data(), ys, pool, scratch, true);

            /*
                Few tokens go through the integer kernels of the generation, many through the
                kernels of the prompt: 8-bit panels for the formats of qk_integer_prompt(),
                whose activations are rounded as act_format::q8_0, and floats for the others.
            */
            const bool prompt = tokens >= qk_float_product_tokens;
            const bool panels = prompt && qk_integer_prompt(f.type, cols);
            const act_format af = panels ? act_format::q8_0 : qk_activation_format(f.type);
            std::vector<uint8_t> act(qk_activation_row_bytes(af, cols));
            std::vector<float> wf(static_cast<size_t>(cols));
            // A pass without the first token, whose tokens get the results of the whole pass.
            std::vector<float> Y3;
            if (panels && tokens - 1 >= qk_float_product_tokens)
            {
                Y3.assign(static_cast<size_t>((tokens - 1) * ys), 7.0f);
                qk_matmul(W, X.data() + xs, tokens - 1, xs, Y3.data(), ys, pool, scratch);
            }
            for (long t = 0; t < tokens; ++t)
            {
                qk_quantize_row(af, X.data() + t * xs, act.data(), cols);
                const std::vector<float> xt(X.begin() + t * xs, X.begin() + t * xs + cols);
                const std::vector<double> a = activation_values(af, act, xt);
                for (long r = 0; r < rows; ++r)
                {
                    float ref;
                    // Single precision sums: their rounding follows the magnitude of
                    // the terms rather than that of the result.
                    double mag = 0;
                    if (prompt)
                    {
                        qk_dequantize_row(f.type, W.data + r * W.row_bytes, wf.data(), cols);
                        double d = 0;
                        for (long i = 0; i < cols; ++i)
                        {
                            const double v = panels ? a[static_cast<size_t>(i)] : X[static_cast<size_t>(t * xs + i)];
                            const double p = static_cast<double>(wf[static_cast<size_t>(i)]) * v;
                            d += p;
                            mag += std::fabs(p);
                        }
                        ref = static_cast<float>(d);
                    }
                    else
                    {
                        ref = qk_dot(f.type, W.data + r * W.row_bytes, act.data(), cols);
                    }
                    const float got = Y[static_cast<size_t>(t * ys + r)];
                    const double tol = 1e-5 * (std::fabs(ref) + 1) + 2e-6 * mag;
                    check(std::fabs(got - ref) <= tol, panels ? "matmul on 8-bit panels" : "matmul", got, ref);
                    // Each token of a pass gets the result it gets alone, whatever the other tokens.
                    if (!prompt) check(got == ref, "integer product independent of the other tokens", got, ref);
                    if (t > 0 && !Y3.empty())
                    {
                        const float other = Y3[static_cast<size_t>((t - 1) * ys + r)];
                        check(got == other, "8-bit product independent of the other tokens", got, other);
                    }
                    check(std::fabs(Y2[static_cast<size_t>(t * ys + r)] - 2 * ref) <= 2 * tol,
                        "matmul accumulate", Y2[static_cast<size_t>(t * ys + r)], 2 * ref);
                }
                for (long r = rows; r < ys; ++r)
                    check(Y[static_cast<size_t>(t * ys + r)] == 7.0f, "matmul writes outside its rows");
            }
        }
    }

    void test_matmul(std::mt19937& rng, thread_pool& pool)
    {
        for (const fmt& f : formats())
        {
            test_matmul_case(f, 512, rng, pool);
            // The formats of single values also take a length that no vector divides.
            if (f.block == 1) test_matmul_case(f, 523, rng, pool);
            // Rows of several slices of the 8-bit panels.
            if (qk_integer_prompt(f.type, 2304)) test_matmul_case(f, 2304, rng, pool);
        }
        // The kernels of the prompt on floats for every format.
        qk_set_integer_prompt(false);
        for (const fmt& f : formats())
            if (f.block != 1) test_matmul_case(f, 512, rng, pool);
        qk_set_integer_prompt(true);
        std::printf("  matmul: every format, 1 to 300 tokens, integer, float and packed paths, strided, accumulated\n");
        std::printf("  integer products: each token as computed alone\n");
        std::string panels = "none in this build";
        if (qk_integer_prompt(ggml_type::Q4_K, 512))
            panels = std::string("Q4_0, Q4_K, Q5_K, Q6_K") + (qk_integer_prompt(ggml_type::Q8_0, 512) ? ", Q8_0" : "");
        std::printf("  8-bit products of the prompt (%s): each token as computed in another pass\n", panels.c_str());
    }

    void test_helpers(std::mt19937& rng)
    {
        // Every finite half converts to float and back unchanged.
        int bad = 0;
        for (uint32_t h = 0; h < 65536; ++h)
        {
            const uint16_t hh = static_cast<uint16_t>(h);
            if ((hh & 0x7C00u) == 0x7C00u && (hh & 0x03FFu)) continue;   // NaN
            if (qk_float_to_half(qk_half_to_float(hh)) != hh) ++bad;
        }
        check(bad == 0, "half round trip", bad, 0);

        // Rounding to nearest even, and the row versions against the scalar ones.
        std::uniform_real_distribution<float> value(-70000.0f, 70000.0f);
        std::vector<float> f(1000);
        for (auto& e : f) e = value(rng) * (rng() % 2 ? 1e-6f : 1.0f);
        std::vector<uint16_t> h(f.size());
        qk_float_to_half_row(f.data(), h.data(), static_cast<long>(f.size()));
        for (size_t i = 0; i < f.size(); ++i)
            check(h[i] == qk_float_to_half(f[i]), "half row conversion", h[i], qk_float_to_half(f[i]));
        std::vector<float> back(f.size());
        qk_half_to_float_row(h.data(), back.data(), static_cast<long>(f.size()));
        for (size_t i = 0; i < f.size(); ++i) check(back[i] == qk_half_to_float(h[i]), "float row conversion");

        std::uniform_real_distribution<float> bounded(-4.0f, 4.0f);
        for (long n : { 1L, 7L, 8L, 33L, 128L, 1000L })
        {
            std::vector<float> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
            for (long i = 0; i < n; ++i)
            {
                a[static_cast<size_t>(i)] = bounded(rng);
                b[static_cast<size_t>(i)] = bounded(rng);
            }
            double ref = 0, mag = 0;
            for (long i = 0; i < n; ++i)
            {
                ref += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
                mag += std::fabs(static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)]);
            }
            check(std::fabs(qk_dot_f32(a.data(), b.data(), n) - ref) <= 1e-5 * (mag + 1), "dot f32");

            std::vector<float> s = a;
            qk_scale(s.data(), -1.5f, n);
            for (long i = 0; i < n; ++i) check(s[static_cast<size_t>(i)] == a[static_cast<size_t>(i)] * -1.5f, "scale");

            float mx = -INFINITY;
            for (float e : a) mx = std::max(mx, e);
            check(qk_max(a.data(), n) == mx, "max");

            std::vector<float> e = a;
            const float sum = qk_exp_shift_sum(e.data(), mx, n);
            double rs = 0;
            for (long i = 0; i < n; ++i)
            {
                const double r = std::exp(static_cast<double>(a[static_cast<size_t>(i)]) - mx);
                rs += r;
                const double got = e[static_cast<size_t>(i)];
                check(std::fabs(got - r) <= 2e-6 * r + 1e-30, "exp", got, r);
            }
            check(std::fabs(sum - rs) <= 1e-5 * rs, "exp sum", sum, rs);
        }

        // Very negative arguments underflow to zero rather than to garbage.
        std::vector<float> neg = { -1000.0f, -120.0f, -88.0f, -87.0f, 0.0f, -1e30f, -50.0f, -10.0f, -3.0f };
        const float s = qk_exp_shift_sum(neg.data(), 0.0f, static_cast<long>(neg.size()));
        check(neg[0] == 0.0f && neg[1] >= 0.0f && neg[1] < 1e-37f && neg[5] == 0.0f, "exp underflow", neg[0], neg[5]);
        check(std::fabs(s - (1.0 + std::exp(-3.0) + std::exp(-10.0) + std::exp(-50.0))) < 1e-6, "exp underflow sum", s);

        // The gate of the SwiGLU blocks over the range of the activations, and beyond it.
        {
            std::uniform_real_distribution<float> wide(-30.0f, 30.0f), unit(-1.0f, 1.0f);
            const float edges[] = { -200.0f, 200.0f, 0.0f, -1e-20f };
            const size_t n = 1003;
            std::vector<float> g(n), u(n);
            for (size_t i = 0; i < n; ++i)
            {
                g[i] = i < 4 ? edges[i] : wide(rng);
                u[i] = unit(rng);
            }
            std::vector<float> got = g;
            qk_silu_mul(got.data(), u.data(), static_cast<long>(n));
            for (size_t i = 0; i < n; ++i)
            {
                const double x = g[i];
                const double r = x / (1.0 + std::exp(-x)) * u[i];
                check(std::fabs(got[i] - r) <= 2e-6 * std::fabs(r) + 1e-30, "silu", got[i], r);
            }
        }
        std::printf("  conversions and vector helpers\n");
    }

    void test_attention_tiles(std::mt19937& rng)
    {
        // Rows of queries against blocks of keys and values in half precision, with sizes
        // around the groups and the vector widths of the kernels.
        std::uniform_real_distribution<float> value(-2.0f, 2.0f);
        const long cases[][3] = { { 1, 1, 64 }, { 2, 5, 128 }, { 3, 37, 80 }, { 64, 256, 128 }, { 5, 100, 256 },
            { 2, 33, 72 }, { 7, 70, 64 }, { 1, 256, 128 }, { 4, 9, 40 } };
        for (const auto& c : cases)
        {
            const long rows = c[0], m = c[1], hd = c[2];
            const long lds = m + 3;
            std::vector<float> Q(static_cast<size_t>(rows * hd)), O(static_cast<size_t>(rows * hd));
            std::vector<float> P(static_cast<size_t>(rows * lds));
            std::vector<uint16_t> K(static_cast<size_t>(m * hd)), V(static_cast<size_t>(m * hd));
            for (auto& e : Q) e = value(rng);
            for (auto& e : O) e = value(rng);
            for (auto& e : P) e = value(rng) * 0.5f + 0.5f;
            for (auto& e : K) e = qk_float_to_half(value(rng));
            for (auto& e : V) e = qk_float_to_half(value(rng));
            std::vector<float> scratch(static_cast<size_t>(qk_attention_scratch(hd)));

            std::vector<float> S(static_cast<size_t>(rows * lds), 1234.0f);
            qk_attention_scores(Q.data(), rows, K.data(), m, hd, S.data(), lds, scratch.data());
            double worst = 0;
            for (long r = 0; r < rows; ++r)
            {
                for (long j = 0; j < m; ++j)
                {
                    double ref = 0, mag = 0;
                    for (long i = 0; i < hd; ++i)
                    {
                        const double t = static_cast<double>(Q[static_cast<size_t>(r * hd + i)])
                            * qk_half_to_float(K[static_cast<size_t>(j * hd + i)]);
                        ref += t;
                        mag += std::fabs(t);
                    }
                    worst = std::max(worst, std::fabs(S[static_cast<size_t>(r * lds + j)] - ref) / (mag + 1));
                }
                for (long j = m; j < lds; ++j)
                    check(S[static_cast<size_t>(r * lds + j)] == 1234.0f, "attention scores stay in their rows");
            }
            check(worst <= 1e-6, "attention scores", worst, static_cast<double>(rows * 1000000 + m * 1000 + hd));

            std::vector<float> got = O;
            qk_attention_values(P.data(), lds, rows, V.data(), m, hd, got.data(), scratch.data());
            worst = 0;
            for (long r = 0; r < rows; ++r)
                for (long i = 0; i < hd; ++i)
                {
                    double ref = O[static_cast<size_t>(r * hd + i)], mag = std::fabs(ref);
                    for (long j = 0; j < m; ++j)
                    {
                        const double t = static_cast<double>(P[static_cast<size_t>(r * lds + j)])
                            * qk_half_to_float(V[static_cast<size_t>(j * hd + i)]);
                        ref += t;
                        mag += std::fabs(t);
                    }
                    worst = std::max(worst, std::fabs(got[static_cast<size_t>(r * hd + i)] - ref) / (mag + 1));
                }
            check(worst <= 1e-6, "attention values", worst, static_cast<double>(rows * 1000000 + m * 1000 + hd));
        }
        std::printf("  attention by tiles\n");
    }

    void test_attention_q8(std::mt19937& rng)
    {
        // The rows of 8-bit keys and values: groups of 32 values with a scale in half precision.
        std::uniform_real_distribution<float> value(-3.0f, 3.0f);
        for (long n : { 32L, 64L, 128L, 256L })
        {
            std::vector<float> x(static_cast<size_t>(n));
            for (long i = 0; i < n; ++i)
            {
                // Groups of different magnitudes, a group of zeros, and a group with one large value.
                const long g = i / 32;
                const float f = g % 4 == 1 ? 0.0f : g % 4 == 2 ? 1e-3f : 1.0f;
                x[static_cast<size_t>(i)] = value(rng) * f;
            }
            if (n >= 128) x[100] = 50.0f;
            // A group of tiny values, whose scale is a subnormal half.
            if (n >= 256)
                for (long i = 192; i < 224; ++i) x[static_cast<size_t>(i)] = 1e-6f * static_cast<float>(i - 207);
            std::vector<int8_t> q(static_cast<size_t>(n));
            std::vector<uint16_t> scales(static_cast<size_t>(n / 32));
            qk_float_to_q8_row(x.data(), q.data(), scales.data(), n);
            std::vector<float> back(static_cast<size_t>(n));
            qk_q8_to_float_row(q.data(), scales.data(), back.data(), n);
            for (long g = 0; g < n / 32; ++g)
            {
                float amax = 0.0f;
                for (long i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[static_cast<size_t>(g * 32 + i)]));
                uint16_t stored = qk_float_to_half(amax / 127.0f);
                if (qk_half_to_float(stored) * 127.5f < amax) ++stored;
                check(scales[static_cast<size_t>(g)] == stored, "q8 scale", scales[static_cast<size_t>(g)], stored);
                const float d = qk_half_to_float(stored);
                const float inv = d > 0.0f ? 1.0f / d : 0.0f;
                for (long i = 0; i < 32; ++i)
                {
                    const size_t k = static_cast<size_t>(g * 32 + i);
                    const float r = std::max(-127.0f, std::min(127.0f, std::nearbyint(x[k] * inv)));
                    check(q[k] == static_cast<int8_t>(r), "q8 value", q[k], r);
                    check(back[k] == d * static_cast<float>(q[k]), "q8 dequantized value", back[k], d * q[k]);
                    check(std::fabs(back[k] - x[k]) <= 0.5f * d * 1.01f + 1e-30f, "q8 rounding", back[k], x[k]);
                }
            }
        }

        // The attention kernels over 8-bit rows against the products of the dequantized rows.
        const long cases[][3] = { { 1, 1, 64 }, { 2, 5, 128 }, { 3, 37, 32 }, { 64, 256, 128 }, { 5, 100, 256 },
            { 7, 70, 64 }, { 1, 256, 128 }, { 4, 9, 96 } };
        std::uniform_real_distribution<float> unit(-2.0f, 2.0f);
        for (const auto& c : cases)
        {
            const long rows = c[0], m = c[1], hd = c[2];
            const long lds = m + 3;
            const long per_row = hd / 32;
            std::vector<float> Q(static_cast<size_t>(rows * hd)), O(static_cast<size_t>(rows * hd));
            std::vector<float> P(static_cast<size_t>(rows * lds));
            std::vector<float> Kf(static_cast<size_t>(m * hd)), Vf(static_cast<size_t>(m * hd));
            for (auto& e : Q) e = unit(rng);
            for (auto& e : O) e = unit(rng);
            for (auto& e : P) e = unit(rng) * 0.25f + 0.5f;
            for (auto& e : Kf) e = unit(rng);
            for (auto& e : Vf) e = unit(rng);
            std::vector<int8_t> K(static_cast<size_t>(m * hd)), V(static_cast<size_t>(m * hd));
            std::vector<uint16_t> Ks(static_cast<size_t>(m * per_row)), Vs(static_cast<size_t>(m * per_row));
            std::vector<float> Kd(Kf.size()), Vd(Vf.size());
            for (long j = 0; j < m; ++j)
            {
                qk_float_to_q8_row(Kf.data() + j * hd, K.data() + j * hd, Ks.data() + j * per_row, hd);
                qk_float_to_q8_row(Vf.data() + j * hd, V.data() + j * hd, Vs.data() + j * per_row, hd);
                qk_q8_to_float_row(K.data() + j * hd, Ks.data() + j * per_row, Kd.data() + j * hd, hd);
                qk_q8_to_float_row(V.data() + j * hd, Vs.data() + j * per_row, Vd.data() + j * hd, hd);
            }
            std::vector<float> scratch(static_cast<size_t>(qk_attention_scratch(hd)));

            std::vector<float> S(static_cast<size_t>(rows * lds), 1234.0f);
            qk_attention_scores_q8(Q.data(), rows, K.data(), Ks.data(), m, hd, S.data(), lds, scratch.data());
            double worst = 0;
            for (long r = 0; r < rows; ++r)
            {
                for (long j = 0; j < m; ++j)
                {
                    double ref = 0, mag = 0;
                    for (long i = 0; i < hd; ++i)
                    {
                        const double t = static_cast<double>(Q[static_cast<size_t>(r * hd + i)])
                            * Kd[static_cast<size_t>(j * hd + i)];
                        ref += t;
                        mag += std::fabs(t);
                    }
                    worst = std::max(worst, std::fabs(S[static_cast<size_t>(r * lds + j)] - ref) / (mag + 1));
                }
                for (long j = m; j < lds; ++j)
                    check(S[static_cast<size_t>(r * lds + j)] == 1234.0f, "q8 attention scores stay in their rows");
            }
            check(worst <= 1e-6, "q8 attention scores", worst, static_cast<double>(rows * 1000000 + m * 1000 + hd));

            std::vector<float> got = O;
            qk_attention_values_q8(P.data(), lds, rows, V.data(), Vs.data(), m, hd, got.data(), scratch.data());
            worst = 0;
            for (long r = 0; r < rows; ++r)
                for (long i = 0; i < hd; ++i)
                {
                    double ref = O[static_cast<size_t>(r * hd + i)], mag = std::fabs(ref);
                    for (long j = 0; j < m; ++j)
                    {
                        const double t = static_cast<double>(P[static_cast<size_t>(r * lds + j)])
                            * Vd[static_cast<size_t>(j * hd + i)];
                        ref += t;
                        mag += std::fabs(t);
                    }
                    worst = std::max(worst, std::fabs(got[static_cast<size_t>(r * hd + i)] - ref) / (mag + 1));
                }
            check(worst <= 1e-6, "q8 attention values", worst, static_cast<double>(rows * 1000000 + m * 1000 + hd));
        }
        std::printf("  8-bit keys and values\n");
    }

    void test_pool(thread_pool& pool)
    {
        // Many small sections in a row, the pattern of a generation step.
        std::vector<long> hits(1000, 0);
        for (int rep = 0; rep < 2000; ++rep)
        {
            const long n = 1 + rep % 13;
            pool.parallel_for(n, [&](long t, int thread) {
                check(thread >= 0 && thread < pool.size(), "thread index");
                hits[static_cast<size_t>(t)] += 1;
            });
        }
        long total = 0;
        for (long h : hits) total += h;
        long expected = 0;
        for (int rep = 0; rep < 2000; ++rep) expected += 1 + rep % 13;
        check(total == expected, "pool tasks", static_cast<double>(total), static_cast<double>(expected));

        // Sections separated by pauses long enough for the workers to sleep.
        pool.set_spin_time(std::chrono::microseconds(50));
        for (int rep = 0; rep < 20; ++rep)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::atomic<long> count{0};
            pool.parallel_for(64, [&](long, int) { count.fetch_add(1); });
            check(count.load() == 64, "pool after sleeping");
        }
        pool.set_spin_time(std::chrono::microseconds(2000));
        std::printf("  thread pool: %d threads\n", pool.size());
    }

    void test_new_pools()
    {
        /*
            A section opened as soon as the pool exists, before the system has run its
            workers, in which every worker takes part.  A worker that missed it would leave
            the section waiting for ever, which the watchdog reports.
        */
        std::atomic<bool> finished{false};
        std::thread watchdog([&]() {
            for (int i = 0; i < 600 && !finished.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!finished.load())
            {
                std::printf("FAILED: a section opened at the creation of a pool never ended\n");
                std::fflush(stdout);
                std::_Exit(1);
            }
        });
        for (int rep = 0; rep < 500; ++rep)
        {
            thread_pool fresh(2 + rep % 3);
            std::atomic<long> count{0};
            fresh.parallel_for(16, [&](long, int) { count.fetch_add(1); });
            check(count.load() == 16, "section of a new pool");
        }
        finished = true;
        watchdog.join();
        std::printf("  new pools: a section at creation\n");
    }
}

int main()
{
    const unsigned lacks = qk_missing_isa(read_cpu_info());
    if (lacks)
    {
        std::printf("albireo quantized kernels (%s): skipped, the processor lacks %s\n", qk_isa(), qk_isa_text(lacks));
        return 0;
    }
    std::printf("albireo quantized kernels (%s)\n", qk_isa());
    std::mt19937 rng(1234);
    thread_pool pool(4);
    test_activation_quantization(rng);
    test_dequantization(rng);
    test_dots(rng);
    test_matmul(rng, pool);
    test_helpers(rng);
    test_attention_tiles(rng);
    test_attention_q8(rng);
    test_pool(pool);
    test_new_pools();
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
