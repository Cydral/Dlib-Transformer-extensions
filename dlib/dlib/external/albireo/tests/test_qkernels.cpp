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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

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

        for (long tokens : { 1L, 3L, 7L, 8L, 37L, 100L })
        {
            const long xs = cols + 5, ys = rows + 3;
            std::vector<float> X(static_cast<size_t>(tokens * xs));
            for (auto& e : X) e = value(rng);
            std::vector<float> Y(static_cast<size_t>(tokens * ys), 7.0f);
            qk_scratch scratch;
            qk_matmul(W, X.data(), tokens, xs, Y.data(), ys, pool, scratch);
            std::vector<float> Y2 = Y;
            qk_matmul(W, X.data(), tokens, xs, Y2.data(), ys, pool, scratch, true);

            // Few tokens go through the integer kernels, many through floats.
            const bool floats = tokens >= qk_float_product_tokens;
            const act_format af = qk_activation_format(f.type);
            std::vector<uint8_t> act(qk_activation_row_bytes(af, cols));
            std::vector<float> wf(static_cast<size_t>(cols));
            for (long t = 0; t < tokens; ++t)
            {
                qk_quantize_row(af, X.data() + t * xs, act.data(), cols);
                for (long r = 0; r < rows; ++r)
                {
                    float ref;
                    // Single precision sums: their rounding follows the magnitude of
                    // the terms rather than that of the result.
                    double mag = 0;
                    if (floats)
                    {
                        qk_dequantize_row(f.type, W.data + r * W.row_bytes, wf.data(), cols);
                        double d = 0;
                        for (long i = 0; i < cols; ++i)
                        {
                            const double p = static_cast<double>(wf[static_cast<size_t>(i)])
                                * X[static_cast<size_t>(t * xs + i)];
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
                    check(std::fabs(got - ref) <= tol, "matmul", got, ref);
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
        }
        std::printf("  matmul: every format, 1 to 100 tokens, integer and float paths, strided, accumulated\n");
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
            std::vector<uint16_t> bh(static_cast<size_t>(n));
            for (long i = 0; i < n; ++i)
            {
                a[static_cast<size_t>(i)] = bounded(rng);
                b[static_cast<size_t>(i)] = bounded(rng);
                bh[static_cast<size_t>(i)] = qk_float_to_half(b[static_cast<size_t>(i)]);
            }
            double ref = 0, ref_h = 0, mag = 0;
            for (long i = 0; i < n; ++i)
            {
                ref += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
                ref_h += static_cast<double>(a[static_cast<size_t>(i)]) * qk_half_to_float(bh[static_cast<size_t>(i)]);
                mag += std::fabs(static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)]);
            }
            check(std::fabs(qk_dot_f32(a.data(), b.data(), n) - ref) <= 1e-5 * (mag + 1), "dot f32");
            check(std::fabs(qk_dot_f32_f16(a.data(), bh.data(), n) - ref_h) <= 1e-5 * (mag + 1), "dot f32 f16");

            std::vector<float> y = a;
            qk_axpy_f16(0.75f, bh.data(), y.data(), n);
            for (long i = 0; i < n; ++i)
            {
                const float e = a[static_cast<size_t>(i)] + 0.75f * qk_half_to_float(bh[static_cast<size_t>(i)]);
                check(std::fabs(y[static_cast<size_t>(i)] - e) <= 1e-6f * (std::fabs(e) + 1), "axpy f16");
            }

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
    test_dots(rng);
    test_matmul(rng, pool);
    test_helpers(rng);
    test_attention_tiles(rng);
    test_pool(pool);
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
