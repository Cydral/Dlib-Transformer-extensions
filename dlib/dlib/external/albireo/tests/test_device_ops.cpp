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
    of their sequences, with several query heads per key/value head.  The operations of the
    hybrid decoders follow: tt::expert_routes() against a sort of the scores, experts of
    equal rank included; tt::gguf_matvec_picked() for every format against products in
    double precision with the rows of the matrices its picks name; tt::squared_relu() and
    tt::mix_rows() against their definitions; tt::mamba3_step() and tt::geodesic_update()
    against the processor code of dlib, with the geometry of the layers of Olala 1.0 mini,
    token by token and several tokens in one call.
    The program ends with the time of the products of one and of eight tokens with the
    matrices of a model of 2 billion parameters, and the rate at which they read the
    weights, with the time of the attention of a generated token over 4,096 positions, and
    with the time of one token in a block of a hybrid decoder of the size of those of
    Olala 1.0 mini and in a block of a dense decoder of 4 billion parameters, operation by
    operation and as the passes of the layers chain them.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
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
        The operations of the mixtures of experts of the hybrid decoders: the routes of tokens
        over 8 to 1,000 experts against a sort of their ranked scores, with experts of equal
        rank; the products through picked matrices, for every format, with matrices whose rows
        fill their blocks of warps or not, activations on 16 bytes or one float further, one
        row of activations for several matrices and one row each; the squares and the weighted
        sums.
    */
    void check_experts(std::mt19937& rng)
    {
        const long routes[4][3] = { { 1, 8, 2 }, { 3, 256, 6 }, { 5, 1000, 64 }, { 2, 300, 1 } };
        for (const auto& c : routes)
        {
            const long n = c[0], experts = c[1], top_k = c[2];
            dlib::resizable_tensor logits(n, experts), bias(1, experts), picks(n, top_k), weights(n, top_k);
            random_floats(logits.host_write_only(), logits.size(), 4.0f, rng);
            random_floats(bias.host_write_only(), bias.size(), 0.1f, rng);
            // The second half of the experts of the first token share one rank.
            for (long e = experts / 2; e < experts; ++e)
            {
                logits.host()[e] = 2.5f;
                bias.host()[e] = 0.0f;
            }
            for (int normalize = 0; normalize < 2; ++normalize)
            {
                dlib::tt::expert_routes(picks, weights, logits, bias, top_k, normalize != 0, 2.45f);
                bool same = true;
                double worst = 0;
                for (long t = 0; t < n; ++t)
                {
                    std::vector<float> score(static_cast<size_t>(experts)), ranked(static_cast<size_t>(experts));
                    std::vector<long> order(static_cast<size_t>(experts));
                    for (long e = 0; e < experts; ++e)
                    {
                        const size_t at = static_cast<size_t>(e);
                        score[at] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(
                            logits.host()[t * experts + e]))));
                        ranked[at] = score[at] + bias.host()[e];
                        order[at] = e;
                    }
                    std::stable_sort(order.begin(), order.end(), [&](long a, long b) {
                        return ranked[static_cast<size_t>(a)] > ranked[static_cast<size_t>(b)];
                    });
                    float selected = 0;
                    for (long k = 0; k < top_k; ++k)
                        selected += score[static_cast<size_t>(order[static_cast<size_t>(k)])];
                    for (long k = 0; k < top_k; ++k)
                    {
                        int32_t pick;
                        std::memcpy(&pick, picks.host() + t * top_k + k, sizeof(pick));
                        const long expert = order[static_cast<size_t>(k)];
                        // Scores the device and the processor round apart may exchange two neighbours.
                        if (pick != expert && !(pick >= 0 && pick < experts
                            && std::fabs(ranked[static_cast<size_t>(pick)] - ranked[static_cast<size_t>(expert)])
                                < 1e-6f))
                            same = false;
                        if (pick < 0 || pick >= experts) continue;
                        float want = score[static_cast<size_t>(pick)];
                        if (normalize) want /= selected + 1e-20f;
                        worst = std::max<double>(worst, std::fabs(weights.host()[t * top_k + k] - want * 2.45f));
                    }
                }
                expect(same, "experts of " + std::to_string(experts) + ", " + std::to_string(top_k) + " a token");
                expect(worst <= 1e-5, "weights of the experts (error " + std::to_string(worst) + ")");
            }
        }

        double worst = 0;
        for (const format& f : formats())
        {
            const bool blocks_of_256 = f.block_values == 256;
            for (int shape = 0; shape < 2; ++shape)
            {
                const long cols = blocks_of_256 ? (shape ? 1536 : 256) : (shape ? 1120 : 32);
                const long rows = shape ? 21 : 16, matrices = 9;
                const size_t rb = gb::row_bytes(f.type, cols);
                std::vector<uint8_t> bytes(static_cast<size_t>(rows * matrices) * rb);
                for (long r = 0; r < rows * matrices; ++r)
                    random_row(f, bytes.data() + static_cast<size_t>(r) * rb, cols, rng);
                dlib::resizable_tensor w;
                load_weights(w, bytes);
                std::vector<float> ref(static_cast<size_t>(rows * matrices * cols));
                for (long r = 0; r < rows * matrices; ++r)
                    qk_dequantize_row(static_cast<dlib::ggml_type>(f.type), bytes.data() + static_cast<size_t>(r) * rb,
                        ref.data() + r * cols, cols);
                for (long per_row : { 1L, 4L })
                {
                    std::vector<int32_t> chosen(12);
                    for (auto& e : chosen) e = static_cast<int32_t>(rng() % static_cast<unsigned>(matrices));
                    const long p = static_cast<long>(chosen.size());
                    dlib::resizable_tensor picks, y(p, rows);
                    put_integers(picks, chosen);
                    for (int shift = 0; shift < 2; ++shift)
                    {
                        dlib::resizable_tensor x((p / per_row) * cols + shift);
                        random_floats(x.host_write_only(), x.size(), 1.0f, rng);
                        dlib::alias_tensor xa(p / per_row, cols);
                        const dlib::alias_tensor_instance xs = xa(x, static_cast<size_t>(shift));
                        dlib::tt::gguf_matvec_picked(y, xs, w, f.type, rows, cols, rb, picks, per_row);
                        const float* got = y.host();
                        const float* xp = x.host() + shift;
                        for (long j = 0; j < p; ++j)
                            for (long r = 0; r < rows; ++r)
                            {
                                const float* row = ref.data() + (chosen[static_cast<size_t>(j)] * rows + r) * cols;
                                double sum = 0, magnitude = 0;
                                for (long i = 0; i < cols; ++i)
                                {
                                    const double term = static_cast<double>(row[i]) * xp[(j / per_row) * cols + i];
                                    sum += term;
                                    magnitude += std::fabs(term);
                                }
                                worst = std::max(worst, std::fabs(got[j * rows + r] - sum) / (magnitude + 1e-6));
                            }
                    }
                }
            }
        }
        expect(worst <= 2e-5, "products through picked matrices (relative error " + std::to_string(worst) + ")");

        dlib::resizable_tensor x(5, 333), start;
        random_floats(x.host_write_only(), x.size(), 3.0f, rng);
        start = x;
        dlib::tt::squared_relu(x);
        double worst_square = 0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            const float r = std::max(start.host()[i], 0.0f);
            worst_square = std::max<double>(worst_square, std::fabs(x.host()[i] - r * r));
        }
        expect(worst_square <= 1e-6, "squared_relu (error " + std::to_string(worst_square) + ")");
        const long n = 5, per_row = 6, width = 384;
        dlib::resizable_tensor src(n * per_row, width), weights(n, per_row), mixed(n, width);
        random_floats(src.host_write_only(), src.size(), 2.0f, rng);
        random_floats(weights.host_write_only(), weights.size(), 1.0f, rng);
        dlib::tt::mix_rows(mixed, src, weights, per_row);
        double worst_mix = 0;
        for (long t = 0; t < n; ++t)
            for (long i = 0; i < width; ++i)
            {
                double sum = 0;
                for (long j = 0; j < per_row; ++j)
                    sum += static_cast<double>(weights.host()[t * per_row + j])
                        * src.host()[(t * per_row + j) * width + i];
                worst_mix = std::max(worst_mix, std::fabs(mixed.host()[t * width + i] - sum));
            }
        expect(worst_mix <= 1e-5, "mix_rows (error " + std::to_string(worst_mix) + ")");
        std::printf("  experts: products through picked matrices within %.2g of their magnitude, squares %.2g, "
            "weighted sums %.2g\n", worst, worst_square, worst_mix);
    }

    // The tensors of a Mamba-3 layer of H heads of P channels, states of N values, R inputs and S turned pairs.
    struct mamba_layer
    {
        long H, P, N, R, S;
        dlib::resizable_tensor norms, bias, misc, mxz, mo;

        mamba_layer(long h, long p, long n, long r, long s, std::mt19937& rng) : H(h), P(p), N(n), R(r), S(s)
        {
            norms.set_size(1, 2 * N);
            bias.set_size(1, H * 2 * R * N);
            misc.set_size(1, 2 * H);
            mxz.set_size(1, H * 2 * R * P);
            mo.set_size(1, H * R * P);
            std::normal_distribution<float> g(0.0f, 1.0f);
            for (float& v : norms) v = 1 + 0.2f * g(rng);
            for (float& v : bias) v = 1 + 0.3f * g(rng);
            for (float& v : misc) v = 0.5f * g(rng);
            for (float& v : mxz) v = 0.7f + 0.3f * g(rng);
            for (float& v : mo) v = 0.5f + 0.2f * g(rng);
        }

        long state_floats() const { return H * S + H * P * N + H * R * N + H * P; }
    };

    /*
        tt::mamba3_step() and tt::geodesic_update() against the processor code of dlib over the
        tokens of a sequence, with a small geometry and with that of the layers of Olala 1.0 mini
        (48 heads of 64 channels, states of 128 values, 4 inputs and 32 turned pairs).
    */
    void check_mamba(std::mt19937& rng)
    {
        std::normal_distribution<float> g(0.0f, 1.0f);
        const long geometries[2][5] = { { 3, 5, 8, 2, 2 }, { 48, 64, 128, 4, 32 } };
        double worst_out = 0, worst_state = 0;
        for (const auto& c : geometries)
        {
            mamba_layer L(c[0], c[1], c[2], c[3], c[4], rng);
            dlib::resizable_tensor s1(1, L.state_floats()), s2;
            s1 = 0;
            s2 = s1;
            dlib::resizable_tensor o1(1, L.H * L.P), o2(1, L.H * L.P), in(1, L.H * (2 * L.P + 3));
            dlib::resizable_tensor dyn(1, 2 * L.R * L.N + L.S);
            for (int t = 0; t < 6; ++t)
            {
                for (float& v : in) v = g(rng);
                for (float& v : dyn) v = 1.5f * g(rng);
                dlib::cpu::mamba3_step(o1, s1, in, dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
                dlib::tt::mamba3_step(o2, s2, in, dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
                double scale_out = 1, scale_state = 1;
                for (size_t i = 0; i < o1.size(); ++i) scale_out = std::max<double>(scale_out, std::fabs(o1.host()[i]));
                for (size_t i = 0; i < s1.size(); ++i)
                    scale_state = std::max<double>(scale_state, std::fabs(s1.host()[i]));
                for (size_t i = 0; i < o1.size(); ++i)
                    worst_out = std::max(worst_out, std::fabs(o1.host()[i] - o2.host()[i]) / scale_out);
                for (size_t i = 0; i < s1.size(); ++i)
                    worst_state = std::max(worst_state, std::fabs(s1.host()[i] - s2.host()[i]) / scale_state);
            }
            // Then several tokens in one call, as the tokens of a prompt.
            const long T = 5;
            dlib::resizable_tensor many_in(T, L.H * (2 * L.P + 3)), many_dyn(T, 2 * L.R * L.N + L.S);
            dlib::resizable_tensor m1(T, L.H * L.P), m2(T, L.H * L.P);
            for (float& v : many_in) v = g(rng);
            for (float& v : many_dyn) v = 1.5f * g(rng);
            dlib::cpu::mamba3_step(m1, s1, many_in, many_dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
            dlib::tt::mamba3_step(m2, s2, many_in, many_dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
            double scale_out = 1, scale_state = 1;
            for (size_t i = 0; i < m1.size(); ++i) scale_out = std::max<double>(scale_out, std::fabs(m1.host()[i]));
            for (size_t i = 0; i < s1.size(); ++i) scale_state = std::max<double>(scale_state, std::fabs(s1.host()[i]));
            for (size_t i = 0; i < m1.size(); ++i)
                worst_out = std::max(worst_out, std::fabs(m1.host()[i] - m2.host()[i]) / scale_out);
            for (size_t i = 0; i < s1.size(); ++i)
                worst_state = std::max(worst_state, std::fabs(s1.host()[i] - s2.host()[i]) / scale_state);
        }
        expect(worst_out <= 1e-3 && worst_state <= 1e-3, "mamba3_step against the processor (outputs "
            + std::to_string(worst_out) + ", states " + std::to_string(worst_state) + ")");
        dlib::resizable_tensor x(3, 1, 1, 1536), y, turn(3, 1, 1, 1536);
        for (float& v : x) v = g(rng);
        for (float& v : turn) v = 3 * g(rng);
        y = x;
        dlib::cpu::geodesic_update(x, turn, 1.2f, 0.01f, 2);
        dlib::tt::geodesic_update(y, turn, 1.2f, 0.01f, 2);
        double worst_turn = 0;
        for (size_t i = 0; i < x.size(); ++i)
            worst_turn = std::max<double>(worst_turn, std::fabs(x.host()[i] - y.host()[i]));
        expect(worst_turn <= 1e-5, "geodesic_update against the processor (error " + std::to_string(worst_turn) + ")");
        std::printf("  mamba3_step within %.2g of the processor, its states %.2g, geodesic_update %.2g\n", worst_out,
            worst_state, worst_turn);
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

    // A matrix of rows rows and cols columns of random weights in a format, on the device.
    void random_matrix(dlib::resizable_tensor& w, int type, long rows, long cols, std::mt19937& rng)
    {
        const format* f = nullptr;
        for (const format& c : formats())
            if (c.type == type) f = &c;
        const size_t rb = gb::row_bytes(type, cols);
        std::vector<uint8_t> bytes(static_cast<size_t>(rows) * rb);
        // The rows repeat a few random ones: the time of a product does not depend on their values.
        const long distinct = std::min<long>(rows, 64);
        for (long r = 0; r < distinct; ++r) random_row(*f, bytes.data() + static_cast<size_t>(r) * rb, cols, rng);
        for (long r = distinct; r < rows; ++r)
            std::memcpy(bytes.data() + static_cast<size_t>(r) * rb,
                bytes.data() + static_cast<size_t>(r % distinct) * rb, rb);
        load_weights(w, bytes);
    }

    /*
        One token in a block of a hybrid decoder of the size of those of Olala 1.0 mini (a width
        of 1,536, a Mamba-3 layer of 48 heads of 64 channels, 256 experts of 768 in a latent
        space of 384, 6 a token, beside a shared expert of 1,536), with the formats of its Q6_K
        file: the time of each operation alone, the device waited for after each, then the time
        of the block as the hybrid pass chains its operations, the device waited for once.
    */
    void measure_hybrid(std::mt19937& rng)
    {
#ifdef DLIB_USE_CUDA
        const int repeats = 50;
#else
        const int repeats = 1;
#endif
        const long d = 1536, E = 256, K = 6, dl = 384, df = 768, ds = 1536;
        mamba_layer L(48, 64, 128, 4, 32, rng);
        const long in_w = L.H * (2 * L.P + 3), dyn_w = 2 * L.R * L.N + L.S, inner = L.H * L.P;
        dlib::resizable_tensor ssm_in, ssm_dyn, wo, router, latent_down, shared_up, experts_up, experts_down, latent_up;
        dlib::resizable_tensor shared_down;
        random_matrix(ssm_in, gb::q6_K, in_w, d, rng);
        random_matrix(ssm_dyn, gb::q6_K, dyn_w, d, rng);
        random_matrix(wo, gb::q6_K, d, inner, rng);
        random_matrix(router, gb::f32, E, d, rng);
        random_matrix(latent_down, gb::q6_K, dl, d, rng);
        random_matrix(shared_up, gb::q6_K, ds, d, rng);
        random_matrix(experts_up, gb::q8_0, E * df, dl, rng);
        random_matrix(experts_down, gb::q6_K, E * dl, df, rng);
        random_matrix(latent_up, gb::bf16, d, dl, rng);
        random_matrix(shared_down, gb::q6_K, d, ds, rng);
        const size_t rb_q6 = gb::row_bytes(gb::q6_K, d);
        dlib::resizable_tensor x(1, 1, 1, d), g(1, 1, 1, d), up(1, 1, 1, d);
        dlib::resizable_tensor proj(1, in_w), dyn(1, dyn_w), mixed(1, inner);
        dlib::resizable_tensor state(1, L.state_floats()), scores(1, E), bias(1, E), picks(1, K), weights(1, K);
        dlib::resizable_tensor latent(1, dl), raised(K, df), lowered(K, dl), routed(1, dl), shared(1, ds);
        random_floats(x.host_write_only(), x.size(), 0.03f, rng);
        random_floats(bias.host_write_only(), bias.size(), 0.1f, rng);
        state = 0;

        struct step
        {
            const char* name;
            std::function<void()> run;
            const dlib::tensor* result;
        };
        const std::vector<step> steps = {
            { "ssm_in      Q6_K 6288x1536", [&]() {
                dlib::tt::gguf_matvec(proj, x, ssm_in, gb::q6_K, in_w, d, rb_q6); }, &proj },
            { "ssm_in_dyn  Q6_K 1056x1536", [&]() {
                dlib::tt::gguf_matvec(dyn, x, ssm_dyn, gb::q6_K, dyn_w, d, rb_q6); }, &dyn },
            { "step of the Mamba-3 layer ", [&]() {
                dlib::tt::mamba3_step(mixed, state, proj, dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f); },
                &mixed },
            { "attn_output Q6_K 1536x3072", [&]() {
                dlib::tt::gguf_matvec(g, mixed, wo, gb::q6_K, d, inner, gb::row_bytes(gb::q6_K, inner)); }, &g },
            { "geodesic update           ", [&]() { dlib::tt::geodesic_update(x, g, 1.0f, 0.02f, 3); }, &x },
            { "router      F32  256x1536 ", [&]() {
                dlib::tt::gguf_matvec(scores, x, router, gb::f32, E, d, gb::row_bytes(gb::f32, d)); }, &scores },
            { "experts of the token      ", [&]() {
                dlib::tt::expert_routes(picks, weights, scores, bias, K, true, 2.45f); }, &weights },
            { "latent_down Q6_K 384x1536 ", [&]() {
                dlib::tt::gguf_matvec(latent, x, latent_down, gb::q6_K, dl, d, rb_q6); }, &latent },
            { "shared_up   Q6_K 1536x1536", [&]() {
                dlib::tt::gguf_matvec(shared, x, shared_up, gb::q6_K, ds, d, rb_q6); }, &shared },
            { "experts up  Q8_0 6x768x384", [&]() {
                dlib::tt::gguf_matvec_picked(raised, latent, experts_up, gb::q8_0, df, dl, gb::row_bytes(gb::q8_0, dl),
                    picks, K); }, &raised },
            { "squares of the experts    ", [&]() { dlib::tt::squared_relu(raised); }, &raised },
            { "experts down Q6_K 6x384x768", [&]() {
                dlib::tt::gguf_matvec_picked(lowered, raised, experts_down, gb::q6_K, dl, df,
                    gb::row_bytes(gb::q6_K, df), picks, 1); }, &lowered },
            { "weighted sum of the experts", [&]() { dlib::tt::mix_rows(routed, lowered, weights, K); }, &routed },
            { "latent_up   BF16 1536x384 ", [&]() {
                dlib::tt::gguf_matvec(g, routed, latent_up, gb::bf16, d, dl, gb::row_bytes(gb::bf16, dl)); }, &g },
            { "squares of the shared expert", [&]() { dlib::tt::squared_relu(shared); }, &shared },
            { "shared_down Q6_K 1536x1536", [&]() {
                dlib::tt::gguf_matvec(up, shared, shared_down, gb::q6_K, d, ds, gb::row_bytes(gb::q6_K, ds)); }, &up },
            { "sum of the two outputs    ", [&]() { dlib::tt::add(1, g, 1, up); }, &g },
            { "geodesic update           ", [&]() { dlib::tt::geodesic_update(x, g, 1.0f, 0.02f, 3); }, &x }
        };
        // Once through, so that the weights are on the device and every buffer holds values.
        for (const step& s : steps) s.run();
        x.host();
        double sum = 0;
        for (const step& s : steps)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i)
            {
                s.run();
                s.result->host();
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                / repeats;
            sum += seconds;
            std::printf("  hybrid block, %s: %8.1f us\n", s.name, seconds * 1e6);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            for (const step& s : steps) s.run();
        x.host();
        const double chained = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / repeats;
        std::printf("  hybrid block, one token: %8.1f us with the device waited for once (%.1f us operation by "
            "operation); 36 such blocks, %.1f ms\n", chained * 1e6, sum * 1e6, chained * 36e3);

        // The step of the Mamba-3 layer over the 64 tokens of a part of a prompt, in one call.
        const long T = 64;
        dlib::resizable_tensor many_in(T, in_w), many_dyn(T, dyn_w), many_out(T, inner);
        random_floats(many_in.host_write_only(), many_in.size(), 1.0f, rng);
        random_floats(many_dyn.host_write_only(), many_dyn.size(), 1.5f, rng);
        dlib::tt::mamba3_step(many_out, state, many_in, many_dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
        many_out.host();
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
        {
            dlib::tt::mamba3_step(many_out, state, many_in, many_dyn, L.norms, L.bias, L.misc, L.mxz, L.mo, 1e-6f);
            many_out.host();
        }
        const double many = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / repeats;
        std::printf("  hybrid block, step of the Mamba-3 layer over %ld tokens in one call: %8.1f us, %.1f us a "
            "token\n", T, many * 1e6, many * 1e6 / T);
    }

    /*
        One token in a block of a dense decoder of the size of those of a model of 4 billion
        parameters (a width of 2,560, 32 query heads and 8 key/value heads of 128 values, a
        feed-forward block of 9,728), its matrices in Q4_K, the token attending 1,024 positions
        in half precision: the time of each operation alone, the device waited for after each,
        then the time of the block as the layer pass chains its operations.
    */
    void measure_dense(std::mt19937& rng)
    {
#ifdef DLIB_USE_CUDA
        const int repeats = 50;
#else
        const int repeats = 1;
#endif
        const long d = 2560, H = 32, KVH = 8, hd = 128, qd = H * hd, kvd = KVH * hd, df = 9728, positions = 1024;
        dlib::resizable_tensor wq, wk, wv, wo, w_gate, w_up, w_down;
        random_matrix(wq, gb::q4_K, qd, d, rng);
        random_matrix(wk, gb::q4_K, kvd, d, rng);
        random_matrix(wv, gb::q4_K, kvd, d, rng);
        random_matrix(wo, gb::q4_K, d, qd, rng);
        random_matrix(w_gate, gb::q4_K, df, d, rng);
        random_matrix(w_up, gb::q4_K, df, d, rng);
        random_matrix(w_down, gb::q4_K, d, df, rng);
        const size_t rb_d = gb::row_bytes(gb::q4_K, d);
        dlib::kv_page_layout L;
        L.layers = 1;
        L.heads = KVH;
        L.head_dim = hd;
        L.block_positions = 256;
        L.q8 = false;
        const long pages = positions / L.block_positions;
        std::vector<uint8_t> bytes(static_cast<size_t>(pages) * L.page_bytes());
        for (auto& b : bytes) b = static_cast<uint8_t>(rng() & 0x3F);
        dlib::resizable_tensor p, slots, tk, tb;
        load_weights(p, bytes);
        std::vector<int32_t> tables;
        for (long i = 0; i < pages; ++i) tables.push_back(static_cast<int32_t>(i));
        // The token writes the last row of the pages and attends every position.
        put_integers(slots, { static_cast<int32_t>(positions - 1) });
        put_integers(tk, { static_cast<int32_t>(positions - 1), 0 });
        put_integers(tb, tables);
        dlib::resizable_tensor x(1, d), xn(1, d), tmp(1, d), gamma(1, d), q(1, qd), k(1, kvd), v(1, kvd), ctx(1, qd);
        dlib::resizable_tensor gate(1, df), up(1, df), angles(1, hd);
        random_floats(x.host_write_only(), x.size(), 1.0f, rng);
        gamma = 1;
        float* a = angles.host_write_only();
        for (long i = 0; i < hd / 2; ++i)
        {
            a[i] = std::cos(0.01f * static_cast<float>(i));
            a[hd / 2 + i] = std::sin(0.01f * static_cast<float>(i));
        }

        struct step
        {
            const char* name;
            std::function<void()> run;
            const dlib::tensor* result;
        };
        const std::vector<step> steps = {
            { "norm of the hidden state  ", [&]() { dlib::tt::rms_norm_rows(xn, x, gamma, 1e-6f); }, &xn },
            { "attn_q      Q4_K 4096x2560", [&]() { dlib::tt::gguf_matvec(q, xn, wq, gb::q4_K, qd, d, rb_d); }, &q },
            { "attn_k      Q4_K 1024x2560", [&]() { dlib::tt::gguf_matvec(k, xn, wk, gb::q4_K, kvd, d, rb_d); }, &k },
            { "attn_v      Q4_K 1024x2560", [&]() { dlib::tt::gguf_matvec(v, xn, wv, gb::q4_K, kvd, d, rb_d); }, &v },
            { "rotation of the queries   ", [&]() { dlib::tt::rope_rows(q, angles, H, hd, hd, true); }, &q },
            { "rotation of the keys      ", [&]() { dlib::tt::rope_rows(k, angles, KVH, hd, hd, true); }, &k },
            { "keys and values to pages  ", [&]() { dlib::tt::paged_kv_store(p, L, 0, k, v, slots); }, &k },
            { "attention, 1,024 positions", [&]() {
                dlib::tt::paged_attention(ctx, q, H, 0.088f, p, L, 0, tk, tb, positions); }, &ctx },
            { "attn_output Q4_K 2560x4096", [&]() {
                dlib::tt::gguf_matvec(tmp, ctx, wo, gb::q4_K, d, qd, gb::row_bytes(gb::q4_K, qd)); }, &tmp },
            { "residual sum              ", [&]() { dlib::tt::add(1, x, 0.01f, tmp); }, &x },
            { "norm of the hidden state  ", [&]() { dlib::tt::rms_norm_rows(xn, x, gamma, 1e-6f); }, &xn },
            { "ffn_gate    Q4_K 9728x2560", [&]() {
                dlib::tt::gguf_matvec(gate, xn, w_gate, gb::q4_K, df, d, rb_d); }, &gate },
            { "ffn_up      Q4_K 9728x2560", [&]() {
                dlib::tt::gguf_matvec(up, xn, w_up, gb::q4_K, df, d, rb_d); }, &up },
            { "gate times the raised rows", [&]() { dlib::tt::silu_multiply(gate, up); }, &gate },
            { "ffn_down    Q4_K 2560x9728", [&]() {
                dlib::tt::gguf_matvec(tmp, gate, w_down, gb::q4_K, d, df, gb::row_bytes(gb::q4_K, df)); }, &tmp },
            { "residual sum              ", [&]() { dlib::tt::add(1, x, 0.01f, tmp); }, &x }
        };
        // Once through, so that the weights are on the device and every buffer holds values.
        for (const step& s : steps) s.run();
        x.host();
        double sum = 0;
        for (const step& s : steps)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i)
            {
                s.run();
                s.result->host();
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                / repeats;
            sum += seconds;
            std::printf("  dense block, %s: %8.1f us\n", s.name, seconds * 1e6);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            for (const step& s : steps) s.run();
        x.host();
        const double chained = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / repeats;
        std::printf("  dense block, one token: %8.1f us with the device waited for once (%.1f us operation by "
            "operation); 36 such blocks, %.1f ms\n", chained * 1e6, sum * 1e6, chained * 36e3);
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
    check_experts(rng);
    check_mamba(rng);
    measure(rng);
    measure_attention(rng);
    measure_hybrid(rng);
    measure_dense(rng);
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
