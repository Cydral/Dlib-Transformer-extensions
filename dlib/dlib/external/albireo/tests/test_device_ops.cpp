// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the operations of dlib on weight matrices kept in their GGUF formats
    (dlib/cuda/gguf_blocks.h) against the processor kernels of the engine.

        test_device_ops [card]

    With CUDA, the operations run on the card given, or on the one with the most memory;
    otherwise on the processor, through the host code of dlib.  First, on the processor, the
    lanes of the kernel of the device for Q4_K and Q6_K (gguf_blocks::lane_products()), against
    the values qk_dequantize_row() gives.  Then, for every format:
        - tt::gguf_dequantize() against qk_dequantize_row(), row by row;
        - tt::gguf_matvec() for 1 to 20 tokens and for 130 tokens, against products in double
          precision with the rows qk_dequantize_row() expands, with activations that start on
          16 bytes and with activations that do not;
    then tt::rms_norm_rows() and tt::silu_multiply() against their definitions, and tt::add()
    with one row added to each row of a pass, as the biases of the attention are;
    tt::rope_rows() against the rotation in double precision; tt::paged_kv_store() and
    tt::paged_kv_gather() against the rows the processor writes (qk_float_to_half_row(),
    qk_rotate_row() and qk_float_to_q8_row()), for heads of 32 to 256 values in half
    precision and in 8 bits; tt::paged_attention() against the attention in double precision
    over pages that hold the rows the processor wrote, for tokens near and far from the start
    of their sequences, with several query heads per key/value head.  The program ends with
    the time of the products of one and of eight tokens with the matrices of a model of 2
    billion parameters, and the rate at which they read the weights, and with the time of the
    attention of a generated token over 4,096 positions.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include <dlib/cuda/gguf_blocks.h>
#include <dlib/cuda/kv_pages.h>
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

    /*
        The lanes of the kernel of the device for Q4_K and Q6_K (gb::lane_products()), run here one
        after the other: their sums over the 32 lanes of a super-block against the products of the
        values qk_dequantize_row() gives, for 1 to 8 rows of activations.
    */
    template <int N>
    void check_lanes(const format& f, std::mt19937& rng)
    {
        const long cols = 1024;
        const long supers = cols / 256;
        std::vector<uint8_t> row(gb::row_bytes(f.type, cols));
        random_row(f, row.data(), cols, rng);
        std::vector<float> values(static_cast<size_t>(cols)), x(static_cast<size_t>(N * cols));
        qk_dequantize_row(static_cast<dlib::ggml_type>(f.type), row.data(), values.data(), cols);
        random_floats(x.data(), x.size(), 1.0f, rng);
        for (int n = 1; n <= N; ++n)
        {
            float acc[N] = {};
            for (long b = 0; b < supers; ++b)
                for (int lane = 0; lane < 32; ++lane)
                {
                    const uint8_t* p = row.data() + static_cast<size_t>(b) * f.block_bytes;
                    if (f.type == gb::q4_K) gb::lane_products<gb::q4_K, N>(p, x.data() + b * 256, cols, n, lane, acc);
                    else gb::lane_products<gb::q6_K, N>(p, x.data() + b * 256, cols, n, lane, acc);
                }
            for (int t = 0; t < n; ++t)
            {
                double want = 0, size = 0;
                for (long i = 0; i < cols; ++i)
                {
                    want += static_cast<double>(values[static_cast<size_t>(i)]) * x[static_cast<size_t>(t * cols + i)];
                    size += std::fabs(static_cast<double>(values[static_cast<size_t>(i)])
                        * x[static_cast<size_t>(t * cols + i)]);
                }
                expect(std::fabs(acc[t] - want) <= 1e-5 * size + 1e-6, std::string(f.name) + " lanes, row "
                    + std::to_string(t) + " of " + std::to_string(n) + ": " + std::to_string(acc[t]) + " against "
                    + std::to_string(want));
            }
        }
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

        /*
            The biases of the queries, keys and values added to each row of a pass, as the attention
            of the device does: tt::add() with one row added to all, which dlib computes with a
            kernel of its own rather than with cuDNN, whose engines the CUDA engine does not carry.
        */
        const long tokens = 5, dims = 160;
        dlib::resizable_tensor bias(1, dims);
        random_floats(bias.host_write_only(), bias.size(), 1.0f, rng);
        double worst_bias = 0;
        const float betas[] = { 1.0f, 0.0f, 0.5f };
        for (float beta : betas)
        {
            dlib::resizable_tensor q(tokens, dims);
            random_floats(q.host_write_only(), q.size(), 2.0f, rng);
            std::vector<float> sum(q.size());
            for (long t = 0; t < tokens; ++t)
                for (long i = 0; i < dims; ++i)
                    sum[static_cast<size_t>(t * dims + i)] = beta * q.host()[t * dims + i] + 0.75f * bias.host()[i];
            dlib::tt::add(beta, q, 0.75f, bias);
            for (size_t i = 0; i < q.size(); ++i)
                worst_bias = std::max<double>(worst_bias, std::fabs(q.host()[i] - sum[i]));
        }
        expect(worst_bias <= 1e-6, "biases added to each row (error " + std::to_string(worst_bias) + ")");
        std::printf("  biases added to each row %.2g\n", worst_bias);
    }

    void check_rope(std::mt19937& rng)
    {
        const long n = 5, heads = 3, hd = 64, dims = 48;
        double worst = 0;
        for (int neox = 0; neox < 2; ++neox)
        {
            dlib::resizable_tensor rows(n * heads, hd), angles(n, dims);
            random_floats(rows.host_write_only(), rows.size(), 2.0f, rng);
            std::uniform_real_distribution<float> theta(-3.0f, 3.0f);
            float* a = angles.host_write_only();
            for (long t = 0; t < n; ++t)
                for (long i = 0; i < dims / 2; ++i)
                {
                    const float th = theta(rng);
                    a[t * dims + i] = 1.1f * std::cos(th);
                    a[t * dims + dims / 2 + i] = 1.1f * std::sin(th);
                }
            const std::vector<float> before(rows.host(), rows.host() + rows.size());
            dlib::tt::rope_rows(rows, angles, heads, hd, dims, neox != 0);
            const float* got = rows.host();
            for (long t = 0; t < n; ++t)
                for (long h = 0; h < heads; ++h)
                {
                    const float* x = before.data() + (t * heads + h) * hd;
                    const float* y = got + (t * heads + h) * hd;
                    std::vector<double> want(x, x + hd);
                    for (long i = 0; i < dims / 2; ++i)
                    {
                        const long i0 = neox ? i : 2 * i, i1 = neox ? i + dims / 2 : 2 * i + 1;
                        const double c = a[t * dims + i], sn = a[t * dims + dims / 2 + i];
                        want[static_cast<size_t>(i0)] = x[i0] * c - x[i1] * sn;
                        want[static_cast<size_t>(i1)] = x[i0] * sn + x[i1] * c;
                    }
                    for (long i = 0; i < hd; ++i)
                        worst = std::max(worst, std::fabs(y[i] - want[static_cast<size_t>(i)]));
                }
        }
        expect(worst <= 1e-5, "rope_rows (error " + std::to_string(worst) + ")");
        std::printf("  rope_rows %.2g\n", worst);
    }

    // Writes the row of keys (kind 0) or of values (1) of a head at row r of a page, as the processor runtime does.
    void host_write_row(uint8_t* page, const dlib::kv_page_layout& L, long kind, long layer, long head, long r,
        const float* x)
    {
        const long hd = L.head_dim;
        uint8_t* slab = page + L.slab_offset(kind, layer, head);
        if (!L.q8)
        {
            qk_float_to_half_row(x, reinterpret_cast<uint16_t*>(slab + static_cast<size_t>(r) * L.data_bytes()), hd);
            return;
        }
        std::vector<float> row(x, x + hd);
        if (kind == 0) qk_rotate_row(row.data(), hd);
        qk_float_to_q8_row(row.data(), reinterpret_cast<int8_t*>(slab + static_cast<size_t>(r) * L.data_bytes()),
            reinterpret_cast<uint16_t*>(slab + static_cast<size_t>(L.block_positions) * L.data_bytes()
                + static_cast<size_t>(r) * L.scale_bytes()), hd);
    }

    // The values of row r of a slab, as floats.
    void host_read_row(const uint8_t* slab, const dlib::kv_page_layout& L, long r, float* x)
    {
        const long hd = L.head_dim;
        if (!L.q8)
        {
            qk_half_to_float_row(reinterpret_cast<const uint16_t*>(slab + static_cast<size_t>(r) * L.data_bytes()), x,
                hd);
            return;
        }
        qk_q8_to_float_row(reinterpret_cast<const int8_t*>(slab + static_cast<size_t>(r) * L.data_bytes()),
            reinterpret_cast<const uint16_t*>(slab + static_cast<size_t>(L.block_positions) * L.data_bytes()
                + static_cast<size_t>(r) * L.scale_bytes()), x, hd);
    }

    void put_integers(dlib::resizable_tensor& t, const std::vector<int32_t>& values)
    {
        t.set_size(static_cast<long long>(values.size()));
        std::memcpy(t.host_write_only(), values.data(), sizeof(int32_t) * values.size());
    }

    /*
        The rows tt::paged_kv_store() writes against those of the processor, byte for byte but
        for the keys of the 8-bit format, whose rotation may round a value to its neighbour: a
        few of their integers may then move by one step.  The other rows of the pages keep their
        bytes, and tt::paged_kv_gather() returns the rows written.
    */
    void check_kv_store(std::mt19937& rng)
    {
        for (int q8 = 0; q8 < 2; ++q8)
            for (long hd : { 32L, 64L, 96L, 128L, 256L })
            {
                dlib::kv_page_layout L;
                L.layers = 3;
                L.heads = 2;
                L.head_dim = hd;
                L.block_positions = 16;
                L.q8 = q8 != 0;
                const long pages = 5, n = 7, layer = 1;
                const size_t bytes = static_cast<size_t>(pages) * L.page_bytes();
                std::vector<uint8_t> before(bytes);
                for (auto& b : before) b = static_cast<uint8_t>(rng() & 0xFF);
                dlib::resizable_tensor p;
                load_weights(p, before);
                dlib::resizable_tensor k(n, L.heads * hd), v(n, L.heads * hd), slots;
                random_floats(k.host_write_only(), k.size(), 3.0f, rng);
                random_floats(v.host_write_only(), v.size(), 3.0f, rng);
                // A few large channels, as in the keys of some models.
                for (long t = 0; t < n; ++t) k.host()[t * L.heads * hd + 5] *= 9.0f;
                std::vector<int32_t> s;
                while (static_cast<long>(s.size()) < n)
                {
                    const int32_t slot = static_cast<int32_t>(rng() % static_cast<uint32_t>(pages * L.block_positions));
                    if (std::find(s.begin(), s.end(), slot) == s.end()) s.push_back(slot);
                }
                put_integers(slots, s);
                dlib::tt::paged_kv_store(p, L, layer, k, v, slots);
                std::vector<uint8_t> want = before;
                for (long t = 0; t < n; ++t)
                    for (long kind = 0; kind < 2; ++kind)
                        for (long h = 0; h < L.heads; ++h)
                            host_write_row(want.data() + static_cast<size_t>(s[t] / L.block_positions) * L.page_bytes(),
                                L, kind, layer, h, s[t] % L.block_positions,
                                (kind == 0 ? k.host() : v.host()) + (t * L.heads + h) * hd);
                const uint8_t* got = reinterpret_cast<const uint8_t*>(p.host());
                long differ = 0, far = 0;
                for (size_t i = 0; i < bytes; ++i)
                {
                    if (got[i] == want[i]) continue;
                    ++differ;
                    // An integer of a rotated key one step away is a rounding of the rotation.
                    const int step = static_cast<int>(static_cast<int8_t>(got[i]))
                        - static_cast<int>(static_cast<int8_t>(want[i]));
                    if (!L.q8 || std::abs(step) != 1) ++far;
                }
                const long values = n * L.heads * hd;
                const char* format = q8 ? "8 bits" : "half precision";
                const std::string what = std::string(format) + ", heads of " + std::to_string(hd);
                expect(far == 0 && differ * 100 <= values, "paged_kv_store, " + what + " (" + std::to_string(differ)
                    + " bytes differ, " + std::to_string(far) + " by more than a step)");

                const size_t gathered_bytes = static_cast<size_t>(n) * 2 * L.layers * L.heads * L.row_bytes();
                dlib::resizable_tensor gathered(static_cast<long long>((gathered_bytes + 3) / 4));
                dlib::tt::paged_kv_gather(gathered, p, L, slots);
                const uint8_t* g = reinterpret_cast<const uint8_t*>(gathered.host());
                got = reinterpret_cast<const uint8_t*>(p.host());
                bool same = true;
                for (long t = 0; t < n; ++t)
                {
                    const uint8_t* page = got + static_cast<size_t>(s[t] / L.block_positions) * L.page_bytes();
                    const size_t r = static_cast<size_t>(s[t] % L.block_positions);
                    for (long kind = 0; kind < 2; ++kind)
                        for (long l = 0; l < L.layers; ++l)
                            for (long h = 0; h < L.heads; ++h)
                            {
                                const uint8_t* slab = page + L.slab_offset(kind, l, h);
                                same = same && std::memcmp(g, slab + r * L.data_bytes(), L.data_bytes()) == 0;
                                g += L.data_bytes();
                                if (!L.q8) continue;
                                same = same && std::memcmp(g, slab + static_cast<size_t>(L.block_positions)
                                    * L.data_bytes() + r * L.scale_bytes(), L.scale_bytes()) == 0;
                                g += L.scale_bytes();
                            }
                }
                expect(same, "paged_kv_gather, " + what);
                std::printf("  paged_kv_store %-14s heads of %3ld: %ld bytes differ by a step of a rotated key\n",
                    format, hd, differ);
            }
    }

    /*
        tt::paged_attention() against the attention in double precision over the same pages:
        tokens at positions 0, 17, 300 and 2999 of sequences whose blocks lie in pages taken in
        a shuffled order, queries scaled, and rotated for the 8-bit keys as the processor rotates
        them.  The long sequence splits the positions among parts on a card.
    */
    void check_attention(std::mt19937& rng)
    {
        struct case_t
        {
            long hd, kv_heads, q_heads;
        };
        const case_t cases[] = { { 32, 2, 4 }, { 64, 1, 6 }, { 96, 2, 2 }, { 128, 2, 8 }, { 256, 1, 3 } };
        const long positions[] = { 0, 17, 300, 2999 };
        const long n = 4;
        for (int q8 = 0; q8 < 2; ++q8)
            for (const case_t& c : cases)
            {
                dlib::kv_page_layout L;
                L.layers = 2;
                L.heads = c.kv_heads;
                L.head_dim = c.hd;
                L.block_positions = 64;
                L.q8 = q8 != 0;
                const long layer = 1, hd = c.hd, width = c.q_heads * c.hd;
                // The pages of each token: its own sequence, its blocks in shuffled pages.
                std::vector<int32_t> tables, tokens;
                long pages = 0;
                for (long t = 0; t < n; ++t)
                {
                    tokens.push_back(static_cast<int32_t>(positions[t]));
                    tokens.push_back(static_cast<int32_t>(tables.size()));
                    for (long b = 0; b <= positions[t] / L.block_positions; ++b)
                        tables.push_back(static_cast<int32_t>(pages++));
                }
                std::vector<int32_t> order(static_cast<size_t>(pages));
                for (long i = 0; i < pages; ++i) order[static_cast<size_t>(i)] = static_cast<int32_t>(i);
                std::shuffle(order.begin(), order.end(), rng);
                for (auto& e : tables) e = order[static_cast<size_t>(e)];
                std::vector<uint8_t> bytes(static_cast<size_t>(pages) * L.page_bytes());
                // The slab of kind of position j of token t.
                auto slab = [&](long t, long j, long kind, long g) {
                    const size_t page = static_cast<size_t>(tables[static_cast<size_t>(tokens[2 * t + 1]
                        + j / L.block_positions)]);
                    return bytes.data() + page * L.page_bytes() + L.slab_offset(kind, layer, g);
                };
                std::vector<float> row(static_cast<size_t>(hd));
                for (long t = 0; t < n; ++t)
                    for (long j = 0; j <= positions[t]; ++j)
                        for (long kind = 0; kind < 2; ++kind)
                            for (long h = 0; h < L.heads; ++h)
                            {
                                random_floats(row.data(), row.size(), 2.0f, rng);
                                if (kind == 0) row[3] *= 8.0f;
                                host_write_row(slab(t, j, kind, h) - L.slab_offset(kind, layer, h), L, kind, layer, h,
                                    j % L.block_positions, row.data());
                            }
                dlib::resizable_tensor p, tk, tb, q(n, width), out(n, width);
                load_weights(p, bytes);
                put_integers(tk, tokens);
                put_integers(tb, tables);
                random_floats(q.host_write_only(), q.size(), 1.0f, rng);
                const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
                dlib::tt::paged_attention(out, q, c.q_heads, scale, p, L, layer, tk, tb, 3000);
                const float* got = out.host();
                const long rep = c.q_heads / c.kv_heads;
                double worst = 0;
                std::vector<float> k(static_cast<size_t>(hd)), val(static_cast<size_t>(hd));
                std::vector<float> qr(static_cast<size_t>(hd));
                for (long t = 0; t < n; ++t)
                    for (long h = 0; h < c.q_heads; ++h)
                    {
                        const long g = h / rep;
                        const float* src = q.host() + t * width + h * hd;
                        for (long i = 0; i < hd; ++i) qr[static_cast<size_t>(i)] = src[i] * scale;
                        if (L.q8) qk_rotate_row(qr.data(), hd);
                        const long P = positions[t] + 1;
                        std::vector<double> a(static_cast<size_t>(P));
                        double mx = -1e300;
                        for (long j = 0; j < P; ++j)
                        {
                            host_read_row(slab(t, j, 0, g), L, j % L.block_positions, k.data());
                            double d = 0;
                            for (long i = 0; i < hd; ++i)
                                d += static_cast<double>(qr[static_cast<size_t>(i)]) * k[static_cast<size_t>(i)];
                            a[static_cast<size_t>(j)] = d;
                            mx = std::max(mx, d);
                        }
                        double sum = 0;
                        for (auto& x : a) sum += (x = std::exp(x - mx));
                        std::vector<double> want(static_cast<size_t>(hd), 0.0);
                        for (long j = 0; j < P; ++j)
                        {
                            host_read_row(slab(t, j, 1, g), L, j % L.block_positions, val.data());
                            const double w = a[static_cast<size_t>(j)] / sum;
                            for (size_t i = 0; i < want.size(); ++i) want[i] += w * val[i];
                        }
                        const float* o = got + t * width + h * hd;
                        for (long i = 0; i < hd; ++i)
                            worst = std::max(worst, std::fabs(o[i] - want[static_cast<size_t>(i)]));
                    }
                const char* format = q8 ? "8 bits" : "half precision";
                const std::string what = std::string(format) + ", heads of " + std::to_string(hd) + ", "
                    + std::to_string(c.q_heads) + "/" + std::to_string(c.kv_heads);
                expect(worst <= 2e-5, "paged_attention, " + what + " (error " + std::to_string(worst) + ")");
                std::printf("  paged_attention %-14s heads of %3ld, %ld/%ld: error %.2g\n", format, hd, c.q_heads,
                    c.kv_heads, worst);
            }
    }

    /*
        The attention of one generated token over 4,096 positions with the heads of a model of 4
        billion parameters (32 query heads, 8 key/value heads of 128 values), in half precision
        and in 8 bits: the time of a layer, and the rate at which it reads the keys and values.
    */
    void measure_attention(std::mt19937& rng)
    {
#ifdef DLIB_USE_CUDA
        const long positions = 4096;
        const int repeats = 50;
#else
        const long positions = 512;
        const int repeats = 1;
#endif
        for (int q8 = 0; q8 < 2; ++q8)
        {
            dlib::kv_page_layout L;
            L.layers = 1;
            L.heads = 8;
            L.head_dim = 128;
            L.block_positions = 256;
            L.q8 = q8 != 0;
            const long pages = positions / L.block_positions;
            std::vector<uint8_t> bytes(static_cast<size_t>(pages) * L.page_bytes());
            for (auto& b : bytes) b = static_cast<uint8_t>(rng() & 0x3F);
            dlib::resizable_tensor p, tk, tb, q(1, 32 * 128), out(1, 32 * 128);
            load_weights(p, bytes);
            std::vector<int32_t> tables;
            for (long i = 0; i < pages; ++i) tables.push_back(static_cast<int32_t>(i));
            put_integers(tk, { static_cast<int32_t>(positions - 1), 0 });
            put_integers(tb, tables);
            random_floats(q.host_write_only(), q.size(), 1.0f, rng);
            dlib::tt::paged_attention(out, q, 32, 0.088f, p, L, 0, tk, tb, positions);
            out.host();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) dlib::tt::paged_attention(out, q, 32, 0.088f, p, L, 0, tk, tb, positions);
            out.host();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                / repeats;
            std::printf("  attention of a token over %ld positions, %s: %8.1f us per layer, %7.1f GB/s of keys and "
                "values\n", positions, q8 ? "8 bits" : "half precision", seconds * 1e6,
                static_cast<double>(bytes.size()) / seconds / 1e9);
        }
    }

    /*
        The products of one generated token, and of the eight tokens of a pass that checks a
        draft, with the matrices of a layer of a model of 2 billion parameters in Q4_K and Q6_K,
        and with its output matrix: the time of each and the rate at which they read the
        weights.
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
        if (f.type == gb::q4_K || f.type == gb::q6_K)
        {
            check_lanes<1>(f, rng);
            check_lanes<8>(f, rng);
        }
    for (const format& f : formats())
        for (const auto& s : shapes) check_format(f, s[0], s[1], rng);
    check_row_operations(rng);
    check_rope(rng);
    check_kv_store(rng);
    check_attention(rng);
    measure(rng);
    measure_attention(rng);
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
