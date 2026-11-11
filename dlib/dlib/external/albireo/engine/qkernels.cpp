// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "qkernels.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <dlib/data_io/gguf_dequantize.h>

#include "thread_pool.h"

#if defined(__AVX2__)
#define ALBIREO_QK_AVX2 1
#include <immintrin.h>
#endif

/*
    The instructions beyond AVX2, the kernels of the processor engine cpu-avxvnni:
        ALBIREO_QK_VNNI     the dot products of bytes and of 16-bit integers in one
                            instruction, on 256-bit vectors (AVX-VNNI).
    MSVC defines no macro for VNNI, so the build names it (ALBIREO_AVX_VNNI).
*/
#if ALBIREO_QK_AVX2 && (defined(__AVXVNNI__) || defined(ALBIREO_AVX_VNNI))
#define ALBIREO_QK_VNNI 1
#endif

namespace albireo
{
    using dlib::ggml_type;

    // Half precision

    namespace
    {
#if !ALBIREO_QK_AVX2
        // Conversion table of the portable build, filled before main() runs.
        struct half_table
        {
            float v[65536];
            half_table()
            {
                for (uint32_t h = 0; h < 65536; ++h)
                    v[h] = dlib::gguf_half_to_float(static_cast<uint16_t>(h));
            }
        };
        const half_table half_values;
#endif

        inline float h2f(uint16_t h)
        {
#if ALBIREO_QK_AVX2
            // _cvtsh_ss is missing from the headers of MSVC; the F16C instruction is reached
            // through its vector form.
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
            return half_values.v[h];
#endif
        }

        inline float h2f(const uint8_t* p)
        {
            uint16_t h;
            std::memcpy(&h, p, 2);
            return h2f(h);
        }

        inline float bf2f(uint16_t h)
        {
            const uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
    }

    float qk_half_to_float(uint16_t h) { return h2f(h); }

    uint16_t qk_float_to_half(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        const uint32_t sign = (x >> 16) & 0x8000u;
        uint32_t mant = x & 0x007FFFFFu;
        const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF);
        if (exp == 0xFF)
            return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
        int32_t e = exp - 127 + 15;
        if (e >= 0x1F)
            return static_cast<uint16_t>(sign | 0x7C00u);
        if (e <= 0)
        {
            if (e < -10) return static_cast<uint16_t>(sign);
            mant |= 0x00800000u;
            const uint32_t shift = static_cast<uint32_t>(14 - e);
            uint32_t half_mant = mant >> shift;
            const uint32_t rem = mant & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1);
            if (rem > halfway || (rem == halfway && (half_mant & 1u))) ++half_mant;
            return static_cast<uint16_t>(sign | half_mant);
        }
        uint32_t half = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
        return static_cast<uint16_t>(half);
    }

    void qk_half_to_float_row(const uint16_t* src, float* dst, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX2
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
#endif
        for (; i < n; ++i) dst[i] = h2f(src[i]);
    }

    void qk_float_to_half_row(const float* src, uint16_t* dst, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX2
        for (; i + 8 <= n; i += 8)
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#endif
        for (; i < n; ++i) dst[i] = qk_float_to_half(src[i]);
    }

    // 8-bit keys and values

    void qk_float_to_q8_row(const float* src, int8_t* dst, uint16_t* scales, long n)
    {
        for (long b = 0; b * 32 < n; ++b)
        {
            const float* x = src + b * 32;
            int8_t* q = dst + b * 32;
            float amax = 0.0f;
#if ALBIREO_QK_AVX2
            const __m256 sign = _mm256_set1_ps(-0.0f);
            const __m256 x0 = _mm256_loadu_ps(x), x1 = _mm256_loadu_ps(x + 8);
            const __m256 x2 = _mm256_loadu_ps(x + 16), x3 = _mm256_loadu_ps(x + 24);
            __m256 m = _mm256_max_ps(_mm256_andnot_ps(sign, x0), _mm256_andnot_ps(sign, x1));
            m = _mm256_max_ps(m, _mm256_max_ps(_mm256_andnot_ps(sign, x2), _mm256_andnot_ps(sign, x3)));
            __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            h = _mm_max_ps(h, _mm_movehl_ps(h, h));
            h = _mm_max_ss(h, _mm_movehdup_ps(h));
            amax = _mm_cvtss_f32(h);
#else
            for (long i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
#endif
            /*
                The values are quantized with the scale as stored, in half precision, raised to the next half
                when its rounding brought it so far below amax / 127 that the largest value would round beyond
                127 (the subnormal halves of the tiny groups), the values then kept within [-127, 127].
            */
            uint16_t stored = qk_float_to_half(amax / 127.0f);
            if (h2f(stored) * 127.5f < amax) ++stored;
            const float d = h2f(stored);
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            scales[b] = stored;
#if ALBIREO_QK_AVX2
            const __m256 vi = _mm256_set1_ps(inv);
            const __m256 hi = _mm256_set1_ps(127.0f), lo = _mm256_set1_ps(-127.0f);
            // The mode of the rounding is written out: a compiler may require a literal constant there.
            auto level = [&](__m256 x) {
                const __m256 r = _mm256_round_ps(_mm256_mul_ps(x, vi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                return _mm256_cvtps_epi32(_mm256_max_ps(lo, _mm256_min_ps(hi, r)));
            };
            __m256i i0 = level(x0);
            __m256i i1 = level(x1);
            __m256i i2 = level(x2);
            __m256i i3 = level(x3);
            // The packs work on each half of the registers: the order is restored at the end.
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(q), i0);
#else
            for (long i = 0; i < 32; ++i)
            {
                const float r = std::nearbyint(x[i] * inv);
                q[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, r)));
            }
#endif
        }
    }

    void qk_q8_to_float_row(const int8_t* src, const uint16_t* scales, float* dst, long n)
    {
        for (long b = 0; b * 32 < n; ++b)
        {
            const float d = h2f(scales[b]);
            const int8_t* q = src + b * 32;
            float* y = dst + b * 32;
#if ALBIREO_QK_AVX2
            const __m256 vd = _mm256_set1_ps(d);
            const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
            const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));
            // Eight values at a time, each widened to 32 bits, converted and scaled.
            const __m128i eights[4] = { lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8) };
            for (int k = 0; k < 4; ++k)
                _mm256_storeu_ps(y + 8 * k, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eights[k]))));
#else
            for (long i = 0; i < 32; ++i) y[i] = d * static_cast<float>(q[i]);
#endif
        }
    }

    void qk_rotate_row(float* x, long n)
    {
        for (long i = 0; i < n; ++i)
            if ((static_cast<uint32_t>(i) * 2654435761u) >> 31) x[i] = -x[i];
        const long block = n & -n;
        const float norm = 1.0f / std::sqrt(static_cast<float>(block));
        for (long b0 = 0; b0 < n; b0 += block)
        {
            float* y = x + b0;
            for (long len = 1; len < block; len <<= 1)
                for (long i = 0; i < block; i += 2 * len)
                    for (long j = i; j < i + len; ++j)
                    {
                        const float a = y[j], c = y[j + len];
                        y[j] = a + c;
                        y[j + len] = a - c;
                    }
            for (long i = 0; i < block; ++i) y[i] *= norm;
        }
    }

    const char* qk_isa()
    {
#if ALBIREO_QK_VNNI
        return "AVX2+FMA+F16C+AVX-VNNI";
#elif ALBIREO_QK_AVX2
        return "AVX2+FMA+F16C";
#elif defined(__SSE2__) || defined(_M_X64)
        return "SSE2";
#else
        return "portable";
#endif
    }

    unsigned qk_required_isa()
    {
        unsigned isa = 0;
#if ALBIREO_QK_AVX2
        isa |= qk_isa_avx2;
#endif
#if ALBIREO_QK_VNNI
        isa |= qk_isa_avx_vnni;
#endif
        return isa;
    }

    // Activation formats

    namespace
    {
        // 32 activations with their scale and the scaled sum of their quanta, which the
        // asymmetric weight formats (_1) need for their minimum.
        struct block_q8_0x
        {
            float d;
            float s;
            int8_t qs[32];
        };
        static_assert(sizeof(block_q8_0x) == 40, "unexpected padding");

        // 256 activations with their scale and the sums of each group of 16, which the
        // k-quant kernels use for the minimums and the offsets of their weights.
        struct block_q8_K
        {
            float d;
            int8_t qs[256];
            int16_t bsums[16];
        };
        static_assert(sizeof(block_q8_K) == 292, "unexpected padding");

        inline int nearest_int(float f)
        {
            return static_cast<int>(std::lrintf(f));
        }

        void quantize_row_q8_0x(const float* x, block_q8_0x* y, long n)
        {
            const long nb = n / 32;
            for (long b = 0; b < nb; ++b, x += 32)
            {
#if ALBIREO_QK_AVX2
                const __m256 sign = _mm256_set1_ps(-0.0f);
                __m256 v0 = _mm256_loadu_ps(x);
                __m256 v1 = _mm256_loadu_ps(x + 8);
                __m256 v2 = _mm256_loadu_ps(x + 16);
                __m256 v3 = _mm256_loadu_ps(x + 24);
                __m256 m = _mm256_andnot_ps(sign, v0);
                m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v1));
                m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v2));
                m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v3));
                __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
                m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
                m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
                const float amax = _mm_cvtss_f32(m4);
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 127.0f / amax : 0.0f;
                y[b].d = d;
                const __m256 mul = _mm256_set1_ps(id);
                v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST);
                v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST);
                v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST);
                v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST);
                __m256i i0 = _mm256_cvtps_epi32(v0);
                __m256i i1 = _mm256_cvtps_epi32(v1);
                __m256i i2 = _mm256_cvtps_epi32(v2);
                __m256i i3 = _mm256_cvtps_epi32(v3);
                // Sum of the quanta, for the asymmetric formats.
                __m256i sum = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
                __m128i s4 = _mm_add_epi32(_mm256_extracti128_si256(sum, 1), _mm256_castsi256_si128(sum));
                s4 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 1));
                y[b].s = d * static_cast<float>(_mm_cvtsi128_si32(s4));
                i0 = _mm256_packs_epi32(i0, i1);
                i2 = _mm256_packs_epi32(i2, i3);
                i0 = _mm256_packs_epi16(i0, i2);
                // The packs work within 128-bit lanes; this permutation restores the order.
                const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                i0 = _mm256_permutevar8x32_epi32(i0, perm);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(y[b].qs), i0);
#else
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 127.0f / amax : 0.0f;
                y[b].d = d;
                int sum = 0;
                for (int j = 0; j < 32; ++j)
                {
                    const int q = nearest_int(x[j] * id);
                    y[b].qs[j] = static_cast<int8_t>(q);
                    sum += q;
                }
                y[b].s = d * static_cast<float>(sum);
#endif
            }
        }

        void quantize_row_q8_K(const float* x, block_q8_K* y, long n)
        {
            const long nb = n / 256;
            for (long b = 0; b < nb; ++b, x += 256)
            {
                float max = 0.0f, amax = 0.0f;
                for (int j = 0; j < 256; ++j)
                {
                    const float ax = std::fabs(x[j]);
                    if (ax > amax) { amax = ax; max = x[j]; }
                }
                if (amax == 0.0f)
                {
                    y[b].d = 0.0f;
                    std::memset(y[b].qs, 0, sizeof(y[b].qs));
                    std::memset(y[b].bsums, 0, sizeof(y[b].bsums));
                    continue;
                }
                // The extreme value maps to -127, as in the reference implementation
                // of the format, which keeps the quanta symmetric.
                const float iscale = -127.0f / max;
                for (int j = 0; j < 256; ++j)
                {
                    const int v = nearest_int(iscale * x[j]);
                    y[b].qs[j] = static_cast<int8_t>(std::min(127, v));
                }
                for (int j = 0; j < 16; ++j)
                {
                    int sum = 0;
                    for (int k = 0; k < 16; ++k) sum += y[b].qs[j * 16 + k];
                    y[b].bsums[j] = static_cast<int16_t>(sum);
                }
                y[b].d = 1.0f / iscale;
            }
        }
    }

    act_format qk_activation_format(ggml_type type)
    {
        switch (type)
        {
        case ggml_type::Q4_0: case ggml_type::Q4_1: case ggml_type::Q5_0: case ggml_type::Q5_1:
        case ggml_type::Q8_0: case ggml_type::IQ4_NL:
            return act_format::q8_0;
        case ggml_type::Q2_K: case ggml_type::Q3_K: case ggml_type::Q4_K: case ggml_type::Q5_K:
        case ggml_type::Q6_K:
            return act_format::q8_k;
        default:
            return act_format::f32;
        }
    }

    bool qk_supported(ggml_type type)
    {
        switch (type)
        {
        case ggml_type::F32: case ggml_type::F16: case ggml_type::BF16:
        case ggml_type::Q4_0: case ggml_type::Q4_1: case ggml_type::Q5_0: case ggml_type::Q5_1:
        case ggml_type::Q8_0: case ggml_type::IQ4_NL:
        case ggml_type::Q2_K: case ggml_type::Q3_K: case ggml_type::Q4_K: case ggml_type::Q5_K:
        case ggml_type::Q6_K:
            return true;
        default:
            return false;
        }
    }

    size_t qk_row_bytes(ggml_type type, long cols)
    {
        const dlib::ggml_type_traits tr = dlib::get_ggml_type_traits(type);
        return static_cast<size_t>(cols) / tr.block_size * tr.type_size;
    }

    size_t qk_activation_row_bytes(act_format f, long n)
    {
        switch (f)
        {
        case act_format::q8_0: return static_cast<size_t>(n / 32) * sizeof(block_q8_0x);
        case act_format::q8_k: return static_cast<size_t>(n / 256) * sizeof(block_q8_K);
        default: return static_cast<size_t>(n) * sizeof(float);
        }
    }

    void qk_quantize_row(act_format f, const float* src, void* dst, long n)
    {
        switch (f)
        {
        case act_format::q8_0: quantize_row_q8_0x(src, static_cast<block_q8_0x*>(dst), n); break;
        case act_format::q8_k: quantize_row_q8_K(src, static_cast<block_q8_K*>(dst), n); break;
        default: std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float)); break;
        }
    }

    // Portable dot products

    namespace
    {
        const int8_t iq4nl_values[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };

        float dot_q4_0_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 18)
            {
                const uint8_t* qs = w + 2;
                int s = 0;
                for (int j = 0; j < 16; ++j)
                    s += ((qs[j] & 0x0F) - 8) * y[b].qs[j] + ((qs[j] >> 4) - 8) * y[b].qs[j + 16];
                sum += h2f(w) * y[b].d * static_cast<float>(s);
            }
            return sum;
        }

        float dot_q4_1_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 20)
            {
                const uint8_t* qs = w + 4;
                int s = 0;
                for (int j = 0; j < 16; ++j)
                    s += (qs[j] & 0x0F) * y[b].qs[j] + (qs[j] >> 4) * y[b].qs[j + 16];
                sum += h2f(w) * y[b].d * static_cast<float>(s) + h2f(w + 2) * y[b].s;
            }
            return sum;
        }

        float dot_q5_0_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 22)
            {
                uint32_t qh;
                std::memcpy(&qh, w + 2, 4);
                const uint8_t* qs = w + 6;
                int s = 0;
                for (int j = 0; j < 16; ++j)
                {
                    const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10u);
                    const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10u);
                    s += (((qs[j] & 0x0F) | h0) - 16) * y[b].qs[j] + (((qs[j] >> 4) | h1) - 16) * y[b].qs[j + 16];
                }
                sum += h2f(w) * y[b].d * static_cast<float>(s);
            }
            return sum;
        }

        float dot_q5_1_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 24)
            {
                uint32_t qh;
                std::memcpy(&qh, w + 4, 4);
                const uint8_t* qs = w + 8;
                int s = 0;
                for (int j = 0; j < 16; ++j)
                {
                    const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10u);
                    const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10u);
                    s += ((qs[j] & 0x0F) | h0) * y[b].qs[j] + ((qs[j] >> 4) | h1) * y[b].qs[j + 16];
                }
                sum += h2f(w) * y[b].d * static_cast<float>(s) + h2f(w + 2) * y[b].s;
            }
            return sum;
        }

        float dot_q8_0_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 34)
            {
                const int8_t* qs = reinterpret_cast<const int8_t*>(w + 2);
                int s = 0;
                for (int j = 0; j < 32; ++j) s += qs[j] * y[b].qs[j];
                sum += h2f(w) * y[b].d * static_cast<float>(s);
            }
            return sum;
        }

        float dot_iq4_nl_ref(const uint8_t* w, const block_q8_0x* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 18)
            {
                const uint8_t* qs = w + 2;
                int s = 0;
                for (int j = 0; j < 16; ++j)
                    s += iq4nl_values[qs[j] & 0x0F] * y[b].qs[j] + iq4nl_values[qs[j] >> 4] * y[b].qs[j + 16];
                sum += h2f(w) * y[b].d * static_cast<float>(s);
            }
            return sum;
        }

        inline void get_scale_min_k4(int j, const uint8_t* q, int& d, int& m)
        {
            if (j < 4)
            {
                d = q[j] & 63;
                m = q[j + 4] & 63;
            }
            else
            {
                d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            }
        }

        float dot_q2_K_ref(const uint8_t* w, const block_q8_K* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 84)
            {
                const uint8_t* scales = w;
                const uint8_t* q = w + 16;
                const float d = h2f(w + 80) * y[b].d;
                const float dmin = h2f(w + 82) * y[b].d;
                // Minimums against the group sums of the activations.
                int summs = 0;
                for (int j = 0; j < 16; ++j) summs += y[b].bsums[j] * (scales[j] >> 4);
                const int8_t* q8 = y[b].qs;
                int isum = 0;
                int is = 0;
                for (int k = 0; k < 2; ++k)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        int d1 = scales[is++] & 0x0F;
                        int s1 = 0;
                        for (int l = 0; l < 16; ++l) s1 += q8[l] * ((q[l] >> shift) & 3);
                        int d2 = scales[is++] & 0x0F;
                        int s2 = 0;
                        for (int l = 16; l < 32; ++l) s2 += q8[l] * ((q[l] >> shift) & 3);
                        isum += d1 * s1 + d2 * s2;
                        shift += 2;
                        q8 += 32;
                    }
                    q += 32;
                }
                sum += d * static_cast<float>(isum) - dmin * static_cast<float>(summs);
            }
            return sum;
        }

        float dot_q3_K_ref(const uint8_t* w, const block_q8_K* y, long n)
        {
            const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
            float sum = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 110)
            {
                const uint8_t* hm = w;
                const uint8_t* q = w + 32;
                const float d = h2f(w + 108) * y[b].d;
                uint32_t aux[4];
                std::memcpy(aux, w + 96, 12);
                const uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                int8_t scales[16];
                std::memcpy(scales, aux, 16);
                const int8_t* q8 = y[b].qs;
                int isum = 0;
                int is = 0;
                uint8_t m = 1;
                for (int k = 0; k < 2; ++k)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        int s1 = 0;
                        for (int l = 0; l < 16; ++l)
                            s1 += q8[l] * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        int s2 = 0;
                        for (int l = 16; l < 32; ++l)
                            s2 += q8[l] * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        isum += (scales[is] - 32) * s1 + (scales[is + 1] - 32) * s2;
                        is += 2;
                        shift += 2;
                        m <<= 1;
                        q8 += 32;
                    }
                    q += 32;
                }
                sum += d * static_cast<float>(isum);
            }
            return sum;
        }

        float dot_q4_K_ref(const uint8_t* w, const block_q8_K* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 144)
            {
                const float d = h2f(w) * y[b].d;
                const float dmin = h2f(w + 2) * y[b].d;
                const uint8_t* scales = w + 4;
                const uint8_t* qs = w + 16;
                const int8_t* q8 = y[b].qs;
                int isum = 0, summ = 0;
                for (int j = 0; j < 4; ++j)
                {
                    int sc1, m1, sc2, m2;
                    get_scale_min_k4(2 * j, scales, sc1, m1);
                    get_scale_min_k4(2 * j + 1, scales, sc2, m2);
                    int s1 = 0, s2 = 0;
                    for (int l = 0; l < 32; ++l)
                    {
                        s1 += (qs[l] & 0x0F) * q8[l];
                        s2 += (qs[l] >> 4) * q8[l + 32];
                    }
                    isum += sc1 * s1 + sc2 * s2;
                    summ += m1 * (y[b].bsums[4 * j] + y[b].bsums[4 * j + 1])
                          + m2 * (y[b].bsums[4 * j + 2] + y[b].bsums[4 * j + 3]);
                    qs += 32;
                    q8 += 64;
                }
                sum += d * static_cast<float>(isum) - dmin * static_cast<float>(summ);
            }
            return sum;
        }

        float dot_q5_K_ref(const uint8_t* w, const block_q8_K* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 176)
            {
                const float d = h2f(w) * y[b].d;
                const float dmin = h2f(w + 2) * y[b].d;
                const uint8_t* scales = w + 4;
                const uint8_t* qh = w + 16;
                const uint8_t* qs = w + 48;
                const int8_t* q8 = y[b].qs;
                int isum = 0, summ = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 4; ++j)
                {
                    int sc1, m1, sc2, m2;
                    get_scale_min_k4(2 * j, scales, sc1, m1);
                    get_scale_min_k4(2 * j + 1, scales, sc2, m2);
                    int s1 = 0, s2 = 0;
                    for (int l = 0; l < 32; ++l)
                    {
                        s1 += ((qs[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) * q8[l];
                        s2 += ((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) * q8[l + 32];
                    }
                    isum += sc1 * s1 + sc2 * s2;
                    summ += m1 * (y[b].bsums[4 * j] + y[b].bsums[4 * j + 1])
                          + m2 * (y[b].bsums[4 * j + 2] + y[b].bsums[4 * j + 3]);
                    qs += 32;
                    q8 += 64;
                    u1 <<= 2;
                    u2 <<= 2;
                }
                sum += d * static_cast<float>(isum) - dmin * static_cast<float>(summ);
            }
            return sum;
        }

        float dot_q6_K_ref(const uint8_t* w, const block_q8_K* y, long n)
        {
            float sum = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const uint8_t* ql = w;
                const uint8_t* qh = w + 128;
                const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
                const float d = h2f(w + 208) * y[b].d;
                const int8_t* q8 = y[b].qs;
                int isum = 0;
                for (int k = 0; k < 2; ++k)
                {
                    int s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int l = 0; l < 32; ++l)
                    {
                        const int is = l / 16;
                        const int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        s[is + 0] += q1 * q8[l];
                        s[is + 2] += q2 * q8[l + 32];
                        s[is + 4] += q3 * q8[l + 64];
                        s[is + 6] += q4 * q8[l + 96];
                    }
                    for (int j = 0; j < 8; ++j) isum += sc[j] * s[j];
                    ql += 64;
                    qh += 32;
                    sc += 8;
                    q8 += 128;
                }
                sum += d * static_cast<float>(isum);
            }
            return sum;
        }

        float dot_f32_ref(const float* a, const float* b, long n)
        {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
            }
            for (; i < n; ++i) s0 += a[i] * b[i];
            return (s0 + s1) + (s2 + s3);
        }

        float dot_f16_ref(const uint8_t* w, const float* x, long n)
        {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(w);
            float s0 = 0, s1 = 0;
            long i = 0;
            for (; i + 2 <= n; i += 2)
            {
                s0 += h2f(h[i]) * x[i];
                s1 += h2f(h[i + 1]) * x[i + 1];
            }
            for (; i < n; ++i) s0 += h2f(h[i]) * x[i];
            return s0 + s1;
        }

        float dot_bf16_ref(const uint8_t* w, const float* x, long n)
        {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(w);
            float s0 = 0, s1 = 0;
            long i = 0;
            for (; i + 2 <= n; i += 2)
            {
                s0 += bf2f(h[i]) * x[i];
                s1 += bf2f(h[i + 1]) * x[i + 1];
            }
            for (; i < n; ++i) s0 += bf2f(h[i]) * x[i];
            return s0 + s1;
        }
    }

// AVX2 dot products

#if ALBIREO_QK_AVX2
    namespace
    {
        inline float hsum_ps(__m256 x)
        {
            __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
            r = _mm_add_ps(r, _mm_movehl_ps(r, r));
            r = _mm_add_ss(r, _mm_movehdup_ps(r));
            return _mm_cvtss_f32(r);
        }

        inline int hsum_epi32_128(__m128i x)
        {
            x = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
            x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 1));
            return _mm_cvtsi128_si32(x);
        }

        // The 32 nibbles of 16 bytes as 32 bytes: low nibbles first, then high nibbles.
        inline __m256i nibbles_to_bytes(const uint8_t* p)
        {
            const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m256i b = _mm256_set_m128i(_mm_srli_epi16(t, 4), t);
            return _mm256_and_si256(_mm256_set1_epi8(0x0F), b);
        }

        // Sum of the products of signed bytes, as eight float partial sums.
        inline __m256 mul_sum_i8_pairs(__m256i x, __m256i y)
        {
            const __m256i ax = _mm256_sign_epi8(x, x);
            const __m256i sy = _mm256_sign_epi8(y, x);
            const __m256i dot = _mm256_maddubs_epi16(ax, sy);
            const __m256i sum = _mm256_madd_epi16(dot, _mm256_set1_epi16(1));
            return _mm256_cvtepi32_ps(sum);
        }

        float dot_q4_0_avx2(const uint8_t* w, const block_q8_0x* y, long n)
        {
            __m256 acc = _mm256_setzero_ps();
            const __m256i off = _mm256_set1_epi8(8);
            for (long b = 0; b < n / 32; ++b, w += 18)
            {
                const __m256 d = _mm256_set1_ps(h2f(w) * y[b].d);
                const __m256i qx = _mm256_sub_epi8(nibbles_to_bytes(w + 2), off);
                const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs));
                acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs(qx, qy), acc);
            }
            return hsum_ps(acc);
        }

        float dot_q8_0_avx2(const uint8_t* w, const block_q8_0x* y, long n)
        {
            __m256 acc = _mm256_setzero_ps();
            for (long b = 0; b < n / 32; ++b, w += 34)
            {
                const __m256 d = _mm256_set1_ps(h2f(w) * y[b].d);
                const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2));
                const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs));
                acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs(qx, qy), acc);
            }
            return hsum_ps(acc);
        }

        // Unpacks the 8 scales and 8 minimums of a Q4_K or Q5_K super-block into
        // 16 bytes: scales in bytes 0..7, minimums in bytes 8..15.
        inline void unpack_k4_scales(const uint8_t* packed, uint32_t utmp[4])
        {
            const uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
            std::memcpy(utmp, packed, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
        }

        // Shuffle mask broadcasting the 16-bit lane i of each 128-bit half.
        inline __m256i scale_shuffle_k4(int i)
        {
            return _mm256_set1_epi16(static_cast<short>(((2 * i + 1) << 8) | (2 * i)));
        }

        // Shuffle mask giving byte i to the low eight lanes and byte i+1 to the high
        // eight lanes of a 128-bit vector.
        inline __m128i scale_shuffle_q6(int i)
        {
            return _mm_setr_epi8(
                static_cast<char>(i), static_cast<char>(i), static_cast<char>(i), static_cast<char>(i),
                static_cast<char>(i), static_cast<char>(i), static_cast<char>(i), static_cast<char>(i),
                static_cast<char>(i + 1), static_cast<char>(i + 1), static_cast<char>(i + 1), static_cast<char>(i + 1),
                static_cast<char>(i + 1), static_cast<char>(i + 1), static_cast<char>(i + 1), static_cast<char>(i + 1));
        }

        float dot_q4_K_avx2(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            __m256 acc = _mm256_setzero_ps();
            __m128 acc_m = _mm_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 144)
            {
                const float d = y[b].d * h2f(w);
                const float dmin = -y[b].d * h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].bsums));
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);

                const uint8_t* q4 = w + 16;
                const int8_t* q8 = y[b].qs;
                __m256i sumi = _mm256_setzero_si256();
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
                    const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
                    const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
                    q4 += 32;
                    const __m256i q4l = _mm256_and_si256(q4bits, m4);
                    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                    q8 += 32;
                    __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
                    p16l = _mm256_madd_epi16(scale_l, p16l);
                    const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                    q8 += 32;
                    __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
                    p16h = _mm256_madd_epi16(scale_h, p16h);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            return hsum_ps(acc) + _mm_cvtss_f32(acc_m);
        }

        float dot_q5_K_avx2(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i mone = _mm256_set1_epi8(1);
            __m256 acc = _mm256_setzero_ps();
            float summs = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 176)
            {
                const float d = y[b].d * h2f(w);
                const float dmin = -y[b].d * h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].bsums));
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                summs += dmin * static_cast<float>(hsum_epi32_128(prod));

                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);
                const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
                __m256i hmask = mone;

                const uint8_t* q5 = w + 48;
                const int8_t* q8 = y[b].qs;
                __m256i sumi = _mm256_setzero_si256();
                int bit = 0;
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i scale_0 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
                    const __m256i scale_1 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
                    const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5));
                    q5 += 32;
                    const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                    const __m256i q5h_0 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit)), 4);
                    ++bit;
                    const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                    hmask = _mm256_slli_epi16(hmask, 1);
                    const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    const __m256i q5h_1 = _mm256_slli_epi16(
                        _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit)), 4);
                    ++bit;
                    const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                    hmask = _mm256_slli_epi16(hmask, 1);

                    const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                    q8 += 32;
                    const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                    q8 += 32;
                    const __m256i p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
                    const __m256i p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            return hsum_ps(acc) + summs;
        }

        float dot_q6_K_avx2(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i m2 = _mm256_set1_epi8(3);
            const __m256i m32s = _mm256_set1_epi8(32);
            __m256 acc = _mm256_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const float d = y[b].d * h2f(w + 208);
                const uint8_t* q4 = w;
                const uint8_t* qh = w + 128;
                const int8_t* q8 = y[b].qs;
                const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 192));
                __m256i sumi = _mm256_setzero_si256();
                int is = 0;
                for (int j = 0; j < 2; ++j)
                {
                    const __m128i scale_0 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 0));
                    const __m128i scale_1 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 2));
                    const __m128i scale_2 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 4));
                    const __m128i scale_3 = _mm_shuffle_epi8(scales, scale_shuffle_q6(is + 6));
                    is += 8;

                    const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
                    q4 += 32;
                    const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
                    q4 += 32;
                    const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
                    qh += 32;

                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
                    const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
                    const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
                    const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                    const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                    const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                    const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64));
                    const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 96));
                    q8 += 128;

                    // (q - 32) * a computed as q * a - 32 * a, keeping the first operand of
                    // maddubs unsigned.
                    __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_0),
                        _mm256_maddubs_epi16(m32s, q8_0));
                    __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_1),
                        _mm256_maddubs_epi16(m32s, q8_1));
                    __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_2),
                        _mm256_maddubs_epi16(m32s, q8_2));
                    __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_3),
                        _mm256_maddubs_epi16(m32s, q8_3));

                    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                    p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                    p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            return hsum_ps(acc);
        }

        /*
            The products of one row of weights with NT rows of activations, the blocks of
            weights unpacked once for all of them.  Each row of activations goes through the
            operations of the kernel of one row in the same order: its result is the one of
            that kernel, bit for bit.
        */
        template <int NT>
        void dot_q4_K_avx2_rows(const uint8_t* w, const block_q8_K* const* y, long n, float* out)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            __m256 acc[NT];
            __m128 acc_m[NT];
            for (int t = 0; t < NT; ++t)
            {
                acc[t] = _mm256_setzero_ps();
                acc_m[t] = _mm_setzero_ps();
            }
            for (long b = 0; b < n / 256; ++b, w += 144)
            {
                const float wd = h2f(w), wmin = h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);
                __m256i sumi[NT];
                for (int t = 0; t < NT; ++t)
                {
                    const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][b].bsums));
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                       _mm256_extracti128_si256(q8sums, 1));
                    const __m128i prod = _mm_madd_epi16(mins, q8s);
                    acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(-y[t][b].d * wmin), _mm_cvtepi32_ps(prod), acc_m[t]);
                    sumi[t] = _mm256_setzero_si256();
                }
                const uint8_t* q4 = w + 16;
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
                    const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
                    const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 32 * j));
                    const __m256i q4l = _mm256_and_si256(q4bits, m4);
                    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    for (int t = 0; t < NT; ++t)
                    {
                        const int8_t* q8 = y[t][b].qs + 64 * j;
                        const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                        const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
                        const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                        const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16l, p16h));
                    }
                }
                for (int t = 0; t < NT; ++t)
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y[t][b].d * wd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
            for (int t = 0; t < NT; ++t)
            {
                __m128 m = _mm_add_ps(acc_m[t], _mm_movehl_ps(acc_m[t], acc_m[t]));
                m = _mm_add_ss(m, _mm_movehdup_ps(m));
                out[t] = hsum_ps(acc[t]) + _mm_cvtss_f32(m);
            }
        }

        template <int NT>
        void dot_q6_K_avx2_rows(const uint8_t* w, const block_q8_K* const* y, long n, float* out)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i m2 = _mm256_set1_epi8(3);
            const __m256i m32s = _mm256_set1_epi8(32);
            __m256 acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const float wd = h2f(w + 208);
                const uint8_t* q4 = w;
                const uint8_t* qh = w + 128;
                const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 192));
                __m256i sumi[NT];
                for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
                for (int j = 0; j < 2; ++j)
                {
                    const int is = 8 * j;
                    const __m256i sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 0)));
                    const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 2)));
                    const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 4)));
                    const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 6)));
                    const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j));
                    const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j + 32));
                    const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh + 32 * j));
                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
                    const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
                    const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
                    const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);
                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);
                    for (int t = 0; t < NT; ++t)
                    {
                        const int8_t* q8 = y[t][b].qs + 128 * j;
                        const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                        const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                        const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64));
                        const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 96));
                        const __m256i p16_0 = _mm256_madd_epi16(sc0, _mm256_sub_epi16(_mm256_maddubs_epi16(q4_0, q8_0),
                            _mm256_maddubs_epi16(m32s, q8_0)));
                        const __m256i p16_1 = _mm256_madd_epi16(sc1, _mm256_sub_epi16(_mm256_maddubs_epi16(q4_1, q8_1),
                            _mm256_maddubs_epi16(m32s, q8_1)));
                        const __m256i p16_2 = _mm256_madd_epi16(sc2, _mm256_sub_epi16(_mm256_maddubs_epi16(q4_2, q8_2),
                            _mm256_maddubs_epi16(m32s, q8_2)));
                        const __m256i p16_3 = _mm256_madd_epi16(sc3, _mm256_sub_epi16(_mm256_maddubs_epi16(q4_3, q8_3),
                            _mm256_maddubs_epi16(m32s, q8_3)));
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16_0, p16_1));
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16_2, p16_3));
                    }
                }
                for (int t = 0; t < NT; ++t)
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y[t][b].d * wd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
            for (int t = 0; t < NT; ++t) out[t] = hsum_ps(acc[t]);
        }

        float dot_f32_avx2(const float* a, const float* b, long n)
        {
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            long i = 0;
            for (; i + 32 <= n; i += 32)
            {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
            }
            for (; i + 8 <= n; i += 8)
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
            float s = hsum_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            for (; i < n; ++i) s += a[i] * b[i];
            return s;
        }

        float dot_f16_avx2(const uint8_t* w, const float* x, long n)
        {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(w);
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + i)));
                const __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + i + 8)));
                s0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), s0);
                s1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), s1);
            }
            for (; i + 8 <= n; i += 8)
            {
                const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + i)));
                s0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), s0);
            }
            float s = hsum_ps(_mm256_add_ps(s0, s1));
            for (; i < n; ++i) s += h2f(h[i]) * x[i];
            return s;
        }
    }
#endif

// VNNI dot products: one instruction for the sums of four products of bytes, or of two
// products of 16-bit integers, on 256-bit vectors

#if ALBIREO_QK_VNNI
    namespace
    {
        // acc + the sums of the products of four unsigned bytes of u by the four signed bytes
        // of s at the same places.
        inline __m256i dpbusd(__m256i acc, __m256i u, __m256i s)
        {
            return _mm256_dpbusd_avx_epi32(acc, u, s);
        }

        // acc + the sums of the products of two signed 16-bit integers of a by those of b.
        inline __m256i dpwssd(__m256i acc, __m256i a, __m256i b)
        {
            return _mm256_dpwssd_avx_epi32(acc, a, b);
        }

        /*
            The weights of Q4_0 enter the unsigned operand as they are stored, 0 to 15, and
            their offset of 8 comes from the sums of the activations that their blocks carry
            (s = d * sum(q)).
        */
        float dot_q4_0_vnni(const uint8_t* w, const block_q8_0x* y, long n)
        {
            __m256 acc = _mm256_setzero_ps();
            float offset = 0.0f;
            for (long b = 0; b < n / 32; ++b, w += 18)
            {
                const float dw = h2f(w);
                const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs));
                const __m256i dot = dpbusd(_mm256_setzero_si256(), nibbles_to_bytes(w + 2), qy);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[b].d), _mm256_cvtepi32_ps(dot), acc);
                offset += dw * y[b].s;
            }
            return hsum_ps(acc) - 8.0f * offset;
        }

        float dot_q8_0_vnni(const uint8_t* w, const block_q8_0x* y, long n)
        {
            __m256 acc = _mm256_setzero_ps();
            for (long b = 0; b < n / 32; ++b, w += 34)
            {
                const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2));
                const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs));
                // |x| times the activations with the signs of x.
                const __m256i dot = dpbusd(_mm256_setzero_si256(), _mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w) * y[b].d), _mm256_cvtepi32_ps(dot), acc);
            }
            return hsum_ps(acc);
        }

        // The kernels of AVX2 with the scales applied and summed by dpwssd.
        float dot_q4_K_vnni(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            __m256 acc = _mm256_setzero_ps();
            __m128 acc_m = _mm_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 144)
            {
                const float d = y[b].d * h2f(w);
                const float dmin = -y[b].d * h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].bsums));
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);
                const uint8_t* q4 = w + 16;
                const int8_t* q8 = y[b].qs;
                __m256i sumi = _mm256_setzero_si256();
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 32 * j));
                    const __m256i q4l = _mm256_and_si256(q4bits, m4);
                    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j));
                    const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j + 32));
                    sumi = dpwssd(sumi, _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j)),
                        _mm256_maddubs_epi16(q4l, q8l));
                    sumi = dpwssd(sumi, _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1)),
                        _mm256_maddubs_epi16(q4h, q8h));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            return hsum_ps(acc) + _mm_cvtss_f32(acc_m);
        }

        float dot_q5_K_vnni(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i mone = _mm256_set1_epi8(1);
            __m256 acc = _mm256_setzero_ps();
            float summs = 0.0f;
            for (long b = 0; b < n / 256; ++b, w += 176)
            {
                const float d = y[b].d * h2f(w);
                const float dmin = -y[b].d * h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].bsums));
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                summs += dmin * static_cast<float>(hsum_epi32_128(prod));

                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);
                const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
                const uint8_t* q5 = w + 48;
                const int8_t* q8 = y[b].qs;
                __m256i sumi = _mm256_setzero_si256();
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5 + 32 * j));
                    // Bit 2j of each byte of hbits is the fifth bit of the first sub-block,
                    // bit 2j+1 that of the second.
                    const __m256i h0 = _mm256_and_si256(_mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j)), mone);
                    const __m256i h1 = _mm256_and_si256(_mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j + 1)), mone);
                    const __m256i q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), _mm256_slli_epi16(h0, 4));
                    const __m256i q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4),
                        _mm256_slli_epi16(h1, 4));
                    const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j));
                    const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 64 * j + 32));
                    sumi = dpwssd(sumi, _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j)),
                        _mm256_maddubs_epi16(q5_0, q8_0));
                    sumi = dpwssd(sumi, _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1)),
                        _mm256_maddubs_epi16(q5_1, q8_1));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            return hsum_ps(acc) + summs;
        }

        /*
            Q6_K: the weights enter as stored, 0 to 63, and their offset of 32 comes from the
            sums of the activations by groups of 16, which share a scale:
            sum sc_g (q - 32) a = sum sc_g q a - 32 sum_g sc_g bsums_g.
        */
        float dot_q6_K_vnni(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i m2 = _mm256_set1_epi8(3);
            __m256 acc = _mm256_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const float d = y[b].d * h2f(w + 208);
                const uint8_t* q4 = w;
                const uint8_t* qh = w + 128;
                const int8_t* q8 = y[b].qs;
                const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 192));
                const __m256i offset = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scales),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].bsums)));
                __m256i sumi = _mm256_setzero_si256();
                for (int j = 0; j < 2; ++j)
                {
                    const int is = 8 * j;
                    const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j));
                    const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j + 32));
                    const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh + 32 * j));
                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
                    const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
                    const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
                    const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);
                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);
                    const int8_t* a = q8 + 128 * j;
                    sumi = dpwssd(sumi, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 0))),
                        _mm256_maddubs_epi16(q4_0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))));
                    sumi = dpwssd(sumi, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 2))),
                        _mm256_maddubs_epi16(q4_1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32))));
                    sumi = dpwssd(sumi, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 4))),
                        _mm256_maddubs_epi16(q4_2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 64))));
                    sumi = dpwssd(sumi, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 6))),
                        _mm256_maddubs_epi16(q4_3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 96))));
                }
                sumi = _mm256_sub_epi32(sumi, _mm256_slli_epi32(offset, 5));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            return hsum_ps(acc);
        }

        /*
            The products of one row of weights with NT rows of activations, the blocks of
            weights unpacked once for all of them.  Each row of activations goes through the
            operations of the kernel of one row above in the same order: its result is the
            one of that kernel, bit for bit.
        */
        template <int NT>
        void dot_q4_K_vnni_rows(const uint8_t* w, const block_q8_K* const* y, long n, float* out)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            __m256 acc[NT];
            __m128 acc_m[NT];
            for (int t = 0; t < NT; ++t)
            {
                acc[t] = _mm256_setzero_ps();
                acc_m[t] = _mm_setzero_ps();
            }
            for (long b = 0; b < n / 256; ++b, w += 144)
            {
                const float wd = h2f(w), wmin = h2f(w + 2);
                uint32_t utmp[4];
                unpack_k4_scales(w + 4, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                    _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                  static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
                const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                const __m256i scales = _mm256_set_m128i(sc128, sc128);
                __m256i sumi[NT];
                for (int t = 0; t < NT; ++t)
                {
                    const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][b].bsums));
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                       _mm256_extracti128_si256(q8sums, 1));
                    const __m128i prod = _mm_madd_epi16(mins, q8s);
                    acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(-y[t][b].d * wmin), _mm_cvtepi32_ps(prod), acc_m[t]);
                    sumi[t] = _mm256_setzero_si256();
                }
                const uint8_t* q4 = w + 16;
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j));
                    const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
                    const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 32 * j));
                    const __m256i q4l = _mm256_and_si256(q4bits, m4);
                    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    for (int t = 0; t < NT; ++t)
                    {
                        const int8_t* q8 = y[t][b].qs + 64 * j;
                        const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
                        const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32));
                        sumi[t] = dpwssd(sumi[t], scale_l, _mm256_maddubs_epi16(q4l, q8l));
                        sumi[t] = dpwssd(sumi[t], scale_h, _mm256_maddubs_epi16(q4h, q8h));
                    }
                }
                for (int t = 0; t < NT; ++t)
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y[t][b].d * wd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
            for (int t = 0; t < NT; ++t)
            {
                __m128 m = _mm_add_ps(acc_m[t], _mm_movehl_ps(acc_m[t], acc_m[t]));
                m = _mm_add_ss(m, _mm_movehdup_ps(m));
                out[t] = hsum_ps(acc[t]) + _mm_cvtss_f32(m);
            }
        }

        template <int NT>
        void dot_q6_K_vnni_rows(const uint8_t* w, const block_q8_K* const* y, long n, float* out)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i m2 = _mm256_set1_epi8(3);
            __m256 acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const float wd = h2f(w + 208);
                const uint8_t* q4 = w;
                const uint8_t* qh = w + 128;
                const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 192));
                __m256i sumi[NT];
                for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
                for (int j = 0; j < 2; ++j)
                {
                    const int is = 8 * j;
                    const __m256i sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 0)));
                    const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 2)));
                    const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 4)));
                    const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle_q6(is + 6)));
                    const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j));
                    const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4 + 64 * j + 32));
                    const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh + 32 * j));
                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
                    const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
                    const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
                    const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);
                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                    const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);
                    for (int t = 0; t < NT; ++t)
                    {
                        const int8_t* a = y[t][b].qs + 128 * j;
                        sumi[t] = dpwssd(sumi[t], sc0,
                            _mm256_maddubs_epi16(q4_0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))));
                        sumi[t] = dpwssd(sumi[t], sc1,
                            _mm256_maddubs_epi16(q4_1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32))));
                        sumi[t] = dpwssd(sumi[t], sc2,
                            _mm256_maddubs_epi16(q4_2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 64))));
                        sumi[t] = dpwssd(sumi[t], sc3,
                            _mm256_maddubs_epi16(q4_3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 96))));
                    }
                }
                for (int t = 0; t < NT; ++t)
                {
                    const __m256i offset = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scales),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t][b].bsums)));
                    sumi[t] = _mm256_sub_epi32(sumi[t], _mm256_slli_epi32(offset, 5));
                    acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y[t][b].d * wd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
                }
            }
            for (int t = 0; t < NT; ++t) out[t] = hsum_ps(acc[t]);
        }
    }
#endif

    float qk_dot_portable(ggml_type t, const uint8_t* w, const void* act, long n)
    {
        switch (t)
        {
        case ggml_type::Q4_0: return dot_q4_0_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q4_1: return dot_q4_1_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q5_0: return dot_q5_0_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q5_1: return dot_q5_1_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q8_0: return dot_q8_0_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::IQ4_NL: return dot_iq4_nl_ref(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q2_K: return dot_q2_K_ref(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q3_K: return dot_q3_K_ref(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q4_K: return dot_q4_K_ref(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q5_K: return dot_q5_K_ref(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q6_K: return dot_q6_K_ref(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::F32: return dot_f32_ref(reinterpret_cast<const float*>(w), static_cast<const float*>(act), n);
        case ggml_type::F16: return dot_f16_ref(w, static_cast<const float*>(act), n);
        case ggml_type::BF16: return dot_bf16_ref(w, static_cast<const float*>(act), n);
        default:
            throw std::runtime_error("qk_dot: unsupported tensor type "
                + std::to_string(static_cast<unsigned>(t)));
        }
    }

    float qk_dot(ggml_type t, const uint8_t* w, const void* act, long n)
    {
#if ALBIREO_QK_VNNI
        switch (t)
        {
        case ggml_type::Q4_0: return dot_q4_0_vnni(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q8_0: return dot_q8_0_vnni(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q4_K: return dot_q4_K_vnni(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q5_K: return dot_q5_K_vnni(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q6_K: return dot_q6_K_vnni(w, static_cast<const block_q8_K*>(act), n);
        default: break;
        }
#endif
#if ALBIREO_QK_AVX2
        switch (t)
        {
        case ggml_type::Q4_0: return dot_q4_0_avx2(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q8_0: return dot_q8_0_avx2(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q4_K: return dot_q4_K_avx2(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q5_K: return dot_q5_K_avx2(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q6_K: return dot_q6_K_avx2(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::F32: return dot_f32_avx2(reinterpret_cast<const float*>(w), static_cast<const float*>(act), n);
        case ggml_type::F16: return dot_f16_avx2(w, static_cast<const float*>(act), n);
        default: break;
        }
#endif
        return qk_dot_portable(t, w, act, n);
    }

    namespace
    {
        /*
            The dot products of one row of weights with the nt <= 4 rows of activations act,
            written to out: the kernels of several rows for the k-quants of this build, which
            unpack the weights once, and qk_dot() row by row for the other formats.  Each
            result is the one of qk_dot(), bit for bit, so that an answer never depends on the
            requests computed with it.
        */
        void dot_rows(ggml_type t, const uint8_t* w, const uint8_t* const* act, long nt, long n, float* out)
        {
#if ALBIREO_QK_AVX2
            if (nt >= 2 && (t == ggml_type::Q4_K || t == ggml_type::Q6_K))
            {
                const block_q8_K* y[4];
                for (long i = 0; i < nt; ++i) y[i] = reinterpret_cast<const block_q8_K*>(act[i]);
#if ALBIREO_QK_VNNI
#define ALBIREO_QK_ROWS(type, k) dot_##type##_vnni_rows<k>(w, y, n, out)
#else
#define ALBIREO_QK_ROWS(type, k) dot_##type##_avx2_rows<k>(w, y, n, out)
#endif
                if (t == ggml_type::Q4_K)
                {
                    if (nt == 2) ALBIREO_QK_ROWS(q4_K, 2);
                    else if (nt == 3) ALBIREO_QK_ROWS(q4_K, 3);
                    else ALBIREO_QK_ROWS(q4_K, 4);
                }
                else
                {
                    if (nt == 2) ALBIREO_QK_ROWS(q6_K, 2);
                    else if (nt == 3) ALBIREO_QK_ROWS(q6_K, 3);
                    else ALBIREO_QK_ROWS(q6_K, 4);
                }
#undef ALBIREO_QK_ROWS
                return;
            }
#endif
            for (long i = 0; i < nt; ++i) out[i] = qk_dot(t, w, act[i], n);
        }

#if ALBIREO_QK_AVX2
        // The eight bytes of v from byte 8 * part, signed, times s.
        inline __m256 scaled_bytes(__m256i v, int part, __m256 s)
        {
            const __m128i half = part < 2 ? _mm256_castsi256_si128(v) : _mm256_extracti128_si256(v, 1);
            const __m128i bytes = part % 2 == 0 ? half : _mm_srli_si128(half, 8);
            return _mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)));
        }

        /*
            A block of Q6_K expanded to floats, 32 values per vector of bytes: the values of
            gguf_dq_q6_K(), bit for bit, each one the product of the scale of its group of 16,
            d times the signed scale, by its signed 6-bit value.
        */
        void dequantize_q6_K_avx2(const uint8_t* p, float* y)
        {
            const uint8_t* ql = p;
            const uint8_t* qh = p + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
            const float d = h2f(p + 208);
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m256i m2 = _mm256_set1_epi8(0x03);
            const __m256i offset = _mm256_set1_epi8(32);
            for (int half = 0; half < 2; ++half)
            {
                const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
                const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
                // The low four bits, then the two high bits moved to bits 4 and 5.
                __m256i q[4];
                q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(h, m2), 4));
                q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m2), 4));
                q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m2), 4));
                q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m2), 4));
                for (int k = 0; k < 4; ++k)
                {
                    const __m256i v = _mm256_sub_epi8(q[k], offset);
                    const __m256 s0 = _mm256_set1_ps(d * sc[2 * k]);
                    const __m256 s1 = _mm256_set1_ps(d * sc[2 * k + 1]);
                    float* o = y + 128 * half + 32 * k;
                    _mm256_storeu_ps(o, scaled_bytes(v, 0, s0));
                    _mm256_storeu_ps(o + 8, scaled_bytes(v, 1, s0));
                    _mm256_storeu_ps(o + 16, scaled_bytes(v, 2, s1));
                    _mm256_storeu_ps(o + 24, scaled_bytes(v, 3, s1));
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
#endif
    }

    void qk_dequantize_row(ggml_type t, const uint8_t* w, float* dst, long n)
    {
        typedef void (*block_fn)(const uint8_t*, float*);
        block_fn fn = nullptr;
        long qk = 0, bytes = 0;
        switch (t)
        {
        case ggml_type::F32: std::memcpy(dst, w, static_cast<size_t>(n) * 4); return;
        case ggml_type::F16: qk_half_to_float_row(reinterpret_cast<const uint16_t*>(w), dst, n); return;
        case ggml_type::BF16:
            for (long i = 0; i < n; ++i)
            {
                uint16_t h;
                std::memcpy(&h, w + 2 * i, 2);
                dst[i] = bf2f(h);
            }
            return;
        case ggml_type::Q4_0: fn = dlib::gguf_dq_q4_0; qk = 32; bytes = 18; break;
        case ggml_type::Q4_1: fn = dlib::gguf_dq_q4_1; qk = 32; bytes = 20; break;
        case ggml_type::Q5_0: fn = dlib::gguf_dq_q5_0; qk = 32; bytes = 22; break;
        case ggml_type::Q5_1: fn = dlib::gguf_dq_q5_1; qk = 32; bytes = 24; break;
        case ggml_type::Q8_0: fn = dlib::gguf_dq_q8_0; qk = 32; bytes = 34; break;
        case ggml_type::IQ4_NL: fn = dlib::gguf_dq_iq4_nl; qk = 32; bytes = 18; break;
        case ggml_type::Q2_K: fn = dlib::gguf_dq_q2_K; qk = 256; bytes = 84; break;
        case ggml_type::Q3_K: fn = dlib::gguf_dq_q3_K; qk = 256; bytes = 110; break;
        case ggml_type::Q4_K: fn = dlib::gguf_dq_q4_K; qk = 256; bytes = 144; break;
        case ggml_type::Q5_K: fn = dlib::gguf_dq_q5_K; qk = 256; bytes = 176; break;
        case ggml_type::Q6_K:
#if ALBIREO_QK_AVX2
            // The reference loop reads its scales lane by lane and expands ten times slower.
            fn = dequantize_q6_K_avx2;
#else
            fn = dlib::gguf_dq_q6_K;
#endif
            qk = 256;
            bytes = 210;
            break;
        default:
            throw std::runtime_error("qk_dequantize_row: unsupported tensor type "
                + std::to_string(static_cast<unsigned>(t)));
        }
        for (long b = 0; b < n / qk; ++b) fn(w + b * bytes, dst + b * qk);
    }

    // Products

    namespace
    {
        std::atomic<bool> exact_products{false};

        void matmul_exact(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, thread_pool& pool, bool accumulate)
        {
            const long rows = W.rows, n = W.cols;
            const long group = 16;
            pool.parallel_for((rows + group - 1) / group, [&](long g, int) {
                std::vector<float> w(static_cast<size_t>(n));
                const long r1 = std::min(rows, (g + 1) * group);
                for (long r = g * group; r < r1; ++r)
                {
                    qk_dequantize_row(W.type, W.data + static_cast<size_t>(r) * W.row_bytes, w.data(), n);
                    for (long t = 0; t < n_tokens; ++t)
                    {
                        const float v = qk_dot_f32(w.data(), X + t * x_stride, n);
                        float& out = Y[t * y_stride + r];
                        out = accumulate ? out + v : v;
                    }
                }
            });
        }
    }

    void qk_set_exact_products(bool on) { exact_products.store(on); }

    // Products of several tokens on floats

    namespace
    {
        inline void store_result(float* out, float v, bool accumulate)
        {
            *out = accumulate ? *out + v : v;
        }

#if ALBIREO_QK_AVX2
        /*
            Y[t][r] (+)= sum_k X[t][k] * W[r][k] for 3 tokens and 4 rows, the rows of W
            being contiguous floats of length K.  Every pair keeps its own accumulator,
            so each vector of X serves 4 products and each vector of W serves 3: twelve
            multiply-adds for seven loads, in the sixteen vector registers of AVX2.  The
            accumulators are named variables rather than an array so that no compiler
            leaves them in memory.
        */
        void gemm_3x4(const float* X, long xs, const float* W, long K, float* Y, long ys, bool accumulate)
        {
            const float* x0 = X;
            const float* x1 = X + xs;
            const float* x2 = X + 2 * xs;
            const float* w0 = W;
            const float* w1 = W + K;
            const float* w2 = W + 2 * K;
            const float* w3 = W + 3 * K;
            const __m256 zero = _mm256_setzero_ps();
            __m256 a00 = zero, a01 = zero, a02 = zero, a03 = zero;
            __m256 a10 = zero, a11 = zero, a12 = zero, a13 = zero;
            __m256 a20 = zero, a21 = zero, a22 = zero, a23 = zero;
            long k = 0;
            for (; k + 8 <= K; k += 8)
            {
                const __m256 v0 = _mm256_loadu_ps(x0 + k);
                const __m256 v1 = _mm256_loadu_ps(x1 + k);
                const __m256 v2 = _mm256_loadu_ps(x2 + k);
                __m256 w = _mm256_loadu_ps(w0 + k);
                a00 = _mm256_fmadd_ps(v0, w, a00); a10 = _mm256_fmadd_ps(v1, w, a10); a20 = _mm256_fmadd_ps(v2, w, a20);
                w = _mm256_loadu_ps(w1 + k);
                a01 = _mm256_fmadd_ps(v0, w, a01); a11 = _mm256_fmadd_ps(v1, w, a11); a21 = _mm256_fmadd_ps(v2, w, a21);
                w = _mm256_loadu_ps(w2 + k);
                a02 = _mm256_fmadd_ps(v0, w, a02); a12 = _mm256_fmadd_ps(v1, w, a12); a22 = _mm256_fmadd_ps(v2, w, a22);
                w = _mm256_loadu_ps(w3 + k);
                a03 = _mm256_fmadd_ps(v0, w, a03); a13 = _mm256_fmadd_ps(v1, w, a13); a23 = _mm256_fmadd_ps(v2, w, a23);
            }
            float s[3][4] = {
                { hsum_ps(a00), hsum_ps(a01), hsum_ps(a02), hsum_ps(a03) },
                { hsum_ps(a10), hsum_ps(a11), hsum_ps(a12), hsum_ps(a13) },
                { hsum_ps(a20), hsum_ps(a21), hsum_ps(a22), hsum_ps(a23) } };
            for (; k < K; ++k)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j) s[i][j] += X[i * xs + k] * W[j * K + k];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j) store_result(Y + i * ys + j, s[i][j], accumulate);
        }

        // One token against 4 rows, for the tokens left over by gemm_3x4.
        void gemm_1x4(const float* X, const float* W, long K, float* Y, bool accumulate)
        {
            const __m256 zero = _mm256_setzero_ps();
            __m256 a0 = zero, a1 = zero, a2 = zero, a3 = zero;
            long k = 0;
            for (; k + 8 <= K; k += 8)
            {
                const __m256 v = _mm256_loadu_ps(X + k);
                a0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(W + k), a0);
                a1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(W + K + k), a1);
                a2 = _mm256_fmadd_ps(v, _mm256_loadu_ps(W + 2 * K + k), a2);
                a3 = _mm256_fmadd_ps(v, _mm256_loadu_ps(W + 3 * K + k), a3);
            }
            float s[4] = { hsum_ps(a0), hsum_ps(a1), hsum_ps(a2), hsum_ps(a3) };
            for (; k < K; ++k)
                for (int j = 0; j < 4; ++j) s[j] += X[k] * W[j * K + k];
            for (int j = 0; j < 4; ++j) store_result(Y + j, s[j], accumulate);
        }
#endif


        // The products of n_rows contiguous float rows of W with n_tokens rows of X,
        // written to the first n_rows columns of Y.
        void gemm_rows(const float* W, long n_rows, long K, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, bool accumulate)
        {
            long t = 0;
            long r4 = 0;
#if ALBIREO_QK_AVX2
            r4 = n_rows / 4 * 4;
            for (; t + 3 <= n_tokens; t += 3)
                for (long r = 0; r < r4; r += 4)
                    gemm_3x4(X + t * x_stride, x_stride, W + r * K, K, Y + t * y_stride + r, y_stride, accumulate);
            for (long u = t; u < n_tokens; ++u)
                for (long r = 0; r < r4; r += 4)
                    gemm_1x4(X + u * x_stride, W + r * K, K, Y + u * y_stride + r, accumulate);
            t = 0;
#endif
            // What the blocks above leave: every token against the last rows, or all of
            // it in the portable build.
            for (; t < n_tokens; ++t)
                for (long r = r4; r < n_rows; ++r)
                    store_result(Y + t * y_stride + r, qk_dot_f32(W + r * K, X + t * x_stride, K), accumulate);
        }

#if ALBIREO_QK_AVX2
        /*
            The products of the long prompt passes as a packed matrix product, the method of
            the BLAS libraries.  Each task takes a tile of weight rows and expands it to floats
            by slices of packed_depth columns, laid out in panels of packed_rows rows: for each
            column, the values of the rows of a panel lie side by side.  The tokens are laid
            out once per product in panels of packed_tokens tokens, column by column.  The
            kernel keeps packed_tokens x packed_rows results in registers and, for each column,
            multiplies one load of weights by each token: every value read serves several
            multiply-adds, and the panels of a slice stay in the caches while they are read.
        */
        const long packed_rows = 16;    // two vectors of 8 floats
        const long packed_tokens = 6;
        const long packed_depth = 256;  // a multiple of the block of every quantization type
        const long packed_tile = 64;    // rows of weights per task
        /*
            Below these passes, expanding and laying out the weights costs more than the
            packed kernel saves (measured with two threads: the packed kernel is as fast as the
            kernel of the rows at 128 tokens, and 1.05 to 1.25 times faster at 512).
        */
        const long packed_min_tokens = 256;
        // The floats of the packed tokens a thread keeps from one product to the next (16 MB).
        const size_t packed_tokens_kept = size_t(1) << 22;

        // n floats of v, 64-byte aligned.
        float* aligned_floats(std::vector<float>& v, size_t n)
        {
            if (v.size() < n + 16) v.resize(n + 16);
            const uintptr_t p = reinterpret_cast<uintptr_t>(v.data());
            return reinterpret_cast<float*>((p + 63) & ~static_cast<uintptr_t>(63));
        }

        // c (packed_tokens x packed_rows) = the products of depth columns of a panel of
        // weights w and a panel of tokens x.
        void packed_kernel(const float* w, const float* x, long depth, float* c)
        {
            const __m256 zero = _mm256_setzero_ps();
            __m256 c00 = zero, c01 = zero, c10 = zero, c11 = zero, c20 = zero, c21 = zero;
            __m256 c30 = zero, c31 = zero, c40 = zero, c41 = zero, c50 = zero, c51 = zero;
            for (long k = 0; k < depth; ++k, w += 16, x += 6)
            {
                const __m256 w0 = _mm256_load_ps(w);
                const __m256 w1 = _mm256_load_ps(w + 8);
                __m256 b = _mm256_broadcast_ss(x);
                c00 = _mm256_fmadd_ps(w0, b, c00); c01 = _mm256_fmadd_ps(w1, b, c01);
                b = _mm256_broadcast_ss(x + 1);
                c10 = _mm256_fmadd_ps(w0, b, c10); c11 = _mm256_fmadd_ps(w1, b, c11);
                b = _mm256_broadcast_ss(x + 2);
                c20 = _mm256_fmadd_ps(w0, b, c20); c21 = _mm256_fmadd_ps(w1, b, c21);
                b = _mm256_broadcast_ss(x + 3);
                c30 = _mm256_fmadd_ps(w0, b, c30); c31 = _mm256_fmadd_ps(w1, b, c31);
                b = _mm256_broadcast_ss(x + 4);
                c40 = _mm256_fmadd_ps(w0, b, c40); c41 = _mm256_fmadd_ps(w1, b, c41);
                b = _mm256_broadcast_ss(x + 5);
                c50 = _mm256_fmadd_ps(w0, b, c50); c51 = _mm256_fmadd_ps(w1, b, c51);
            }
            _mm256_store_ps(c, c00); _mm256_store_ps(c + 8, c01);
            _mm256_store_ps(c + 16, c10); _mm256_store_ps(c + 24, c11);
            _mm256_store_ps(c + 32, c20); _mm256_store_ps(c + 40, c21);
            _mm256_store_ps(c + 48, c30); _mm256_store_ps(c + 56, c31);
            _mm256_store_ps(c + 64, c40); _mm256_store_ps(c + 72, c41);
            _mm256_store_ps(c + 80, c50); _mm256_store_ps(c + 88, c51);
        }

        // Expands rows [r0, r0 + n) of W, columns [k0, k0 + depth), into panels of packed_rows
        // rows; the rows past n are zeros.
        void pack_weights(const weight_matrix& W, long r0, long n, long k0, long depth, float* panels, float* row)
        {
            if (n % packed_rows != 0)
            {
                float* last = panels + n / packed_rows * packed_rows * depth;
                std::fill(last, last + packed_rows * depth, 0.0f);
            }
            const size_t offset = qk_row_bytes(W.type, k0);
            for (long r = 0; r < n; ++r)
            {
                const uint8_t* src = W.data + static_cast<size_t>(r0 + r) * W.row_bytes + offset;
                const float* v = row;
                if (W.type == ggml_type::F32) v = reinterpret_cast<const float*>(src);
                else qk_dequantize_row(W.type, src, row, depth);
                float* dst = panels + r / packed_rows * packed_rows * depth + r % packed_rows;
                for (long k = 0; k < depth; ++k) dst[k * packed_rows] = v[k];
            }
        }

        // Panel q of the tokens: K columns of packed_tokens tokens, the tokens past n being zeros.
        void pack_tokens(const float* X, long n, long x_stride, long K, long q, float* panels)
        {
            float* dst = panels + q * K * packed_tokens;
            for (long i = 0; i < packed_tokens; ++i)
            {
                const long t = q * packed_tokens + i;
                if (t < n)
                {
                    const float* src = X + t * x_stride;
                    for (long k = 0; k < K; ++k) dst[k * packed_tokens + i] = src[k];
                }
                else
                {
                    for (long k = 0; k < K; ++k) dst[k * packed_tokens + i] = 0.0f;
                }
            }
        }

        void matmul_packed(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, thread_pool& pool, bool accumulate)
        {
            const long rows = W.rows, K = W.cols;
            const long token_panels = (n_tokens + packed_tokens - 1) / packed_tokens;
            static thread_local std::vector<float> token_buffer;
            float* xp = aligned_floats(token_buffer, static_cast<size_t>(token_panels * K * packed_tokens));
            pool.parallel_for(token_panels, [&](long q, int) { pack_tokens(X, n_tokens, x_stride, K, q, xp); });

            const long n_tiles = (rows + packed_tile - 1) / packed_tile;
            pool.parallel_for(n_tiles, [&](long tile, int) {
                static thread_local std::vector<float> weight_buffer, row_buffer;
                const long r0 = tile * packed_tile;
                const long n = std::min(packed_tile, rows - r0);
                const long panels = (n + packed_rows - 1) / packed_rows;
                float* wp = aligned_floats(weight_buffer, static_cast<size_t>(panels * packed_rows * packed_depth));
                float* row = aligned_floats(row_buffer, static_cast<size_t>(packed_depth));
                alignas(64) float c[packed_tokens * packed_rows];
                for (long k0 = 0; k0 < K; k0 += packed_depth)
                {
                    const long depth = std::min(packed_depth, K - k0);
                    pack_weights(W, r0, n, k0, depth, wp, row);
                    // The first slice writes the results, the next ones add to them.
                    const bool add = accumulate || k0 > 0;
                    for (long q = 0; q < token_panels; ++q)
                    {
                        const float* xq = xp + (q * K + k0) * packed_tokens;
                        const long t0 = q * packed_tokens;
                        const long nt = std::min(packed_tokens, n_tokens - t0);
                        for (long p = 0; p < panels; ++p)
                        {
                            packed_kernel(wp + p * packed_rows * depth, xq, depth, c);
                            const long nr = std::min(packed_rows, n - p * packed_rows);
                            for (long i = 0; i < nt; ++i)
                            {
                                float* y = Y + (t0 + i) * y_stride + r0 + p * packed_rows;
                                const float* ci = c + i * packed_rows;
                                if (add)
                                    for (long j = 0; j < nr; ++j) y[j] += ci[j];
                                else
                                    for (long j = 0; j < nr; ++j) y[j] = ci[j];
                            }
                        }
                    }
                }
            });
            // A larger buffer goes back to the system: the thread would otherwise keep the largest one it needed.
            if (token_buffer.size() > packed_tokens_kept) std::vector<float>().swap(token_buffer);
        }
#endif

        void matmul_float(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, thread_pool& pool, bool accumulate)
        {
#if ALBIREO_QK_AVX2
            if (n_tokens >= packed_min_tokens)
            {
                matmul_packed(W, X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
                return;
            }
#endif
            const long rows = W.rows, K = W.cols;
            // A tile of rows is expanded once to floats and serves every token: its
            // floats stay in the second level cache while the tokens go by, in groups
            // small enough for their rows to stay in the first level.
            const long tile_rows = 16;
            const long token_group = 48;
            const long n_tiles = (rows + tile_rows - 1) / tile_rows;
            pool.parallel_for(n_tiles, [&](long tile, int) {
                static thread_local std::vector<float> buffer;
                const long r0 = tile * tile_rows;
                const long n = std::min(tile_rows, rows - r0);
                const float* w = nullptr;
                if (W.type == ggml_type::F32)
                {
                    w = reinterpret_cast<const float*>(W.data + static_cast<size_t>(r0) * W.row_bytes);
                }
                else
                {
                    buffer.resize(static_cast<size_t>(n * K));
                    for (long r = 0; r < n; ++r)
                        qk_dequantize_row(W.type, W.data + static_cast<size_t>(r0 + r) * W.row_bytes,
                            buffer.data() + r * K, K);
                    w = buffer.data();
                }
                for (long t0 = 0; t0 < n_tokens; t0 += token_group)
                    gemm_rows(w, n, K, X + t0 * x_stride, std::min(token_group, n_tokens - t0), x_stride,
                        Y + t0 * y_stride + r0, y_stride, accumulate);
            });
        }
    }

    // Products of several tokens on 8-bit integers

    namespace
    {
        std::atomic<bool> integer_prompt{true};

#if ALBIREO_QK_AVX2
        /*
            The products of the passes of many tokens on 8-bit integers, for the formats whose
            weights are 8-bit integers times a scale, plus an offset for some of them: an
            instruction of the products of bytes then computes 32 products where a vector of
            floats computes 8.

            The input features go by groups of 256, a group by 8 chunks of 32 features.  A token
            is quantized by chunks as for act_format::q8_0: a scale sx, the largest magnitude of
            the chunk over 127, and integers xq = round(x / sx) within [-127, 127], kept as the
            unsigned bytes xq + 128.  A weight is the product of a signed 8-bit integer s by the
            scale A of its chunk, or of its half of a chunk for the formats with a scale per 16
            features, plus the offset B of its chunk for the formats with minimums: w = A s + B,
            the value of its format.

            A panel holds the 256 bytes of a group transposed by units of 4 bytes: vector k of
            the group holds, in its lane of 32 bits j, the bytes 4k to 4k + 3 of chunk j.  The
            products of the bytes of a lane, which the instruction adds together, then belong to
            one chunk, and the 8 vectors of a group give the dot products of its 8 chunks in one
            register; the formats with a scale per 16 features read them as two halves of 4
            vectors.  As the sum over a lane of (xq + 128) s is dot(xq, s) + 128 sum(s),
                sum over a chunk of w x = sx (A dot + Bw) + B c
            with Bw = -128 A sum(s), a constant of the weights, and c = sx sum(xq), a constant
            of the token.  The products of the bytes are exact, and the results differ from the
            products on floats by the rounding of the activations to 8 bits, as those of the
            integer kernels of the generation.
        */
        enum class panel_kind { none, q4_0, q8_0, q4_K, q5_K, q6_K };

        panel_kind panel_kind_of(ggml_type t)
        {
            switch (t)
            {
            case ggml_type::Q4_0: return panel_kind::q4_0;
            case ggml_type::Q4_K: return panel_kind::q4_K;
            case ggml_type::Q5_K: return panel_kind::q5_K;
            case ggml_type::Q6_K: return panel_kind::q6_K;
#if ALBIREO_QK_VNNI
            // Its signed bytes times the bytes of the activations overflow the 16-bit sums of AVX2.
            case ggml_type::Q8_0: return panel_kind::q8_0;
#endif
            default: return panel_kind::none;
            }
        }

        // A scale per 16 features, read as two halves of the vectors of a group.
        inline bool panel_halves(panel_kind k) { return k == panel_kind::q6_K; }
        // An offset per chunk, times the sums of the activations.
        inline bool panel_minimums(panel_kind k) { return k == panel_kind::q4_K || k == panel_kind::q5_K; }
        /*
            The floats of a group of a row: A then Bw of its chunks, then B for the formats with
            minimums; A and Bw of the first halves, then of the second ones, for the formats with
            a scale per 16 features.
        */
        inline long panel_floats(panel_kind k) { return panel_halves(k) ? 32 : panel_minimums(k) ? 24 : 16; }

        /*
            Transposes the 8 x 8 units of 32 bits of v: unit j of vector i becomes unit i of
            vector j.
        */
        inline void transpose_units(__m256i v[8])
        {
            const __m256i t0 = _mm256_unpacklo_epi32(v[0], v[1]);
            const __m256i t1 = _mm256_unpackhi_epi32(v[0], v[1]);
            const __m256i t2 = _mm256_unpacklo_epi32(v[2], v[3]);
            const __m256i t3 = _mm256_unpackhi_epi32(v[2], v[3]);
            const __m256i t4 = _mm256_unpacklo_epi32(v[4], v[5]);
            const __m256i t5 = _mm256_unpackhi_epi32(v[4], v[5]);
            const __m256i t6 = _mm256_unpacklo_epi32(v[6], v[7]);
            const __m256i t7 = _mm256_unpackhi_epi32(v[6], v[7]);
            // Units 0 and 1, 2 and 3 of vectors 0 to 3 in each half, then of vectors 4 to 7.
            const __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
            const __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
            const __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
            const __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
            const __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
            const __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
            const __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
            const __m256i u7 = _mm256_unpackhi_epi64(t5, t7);
            v[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
            v[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
            v[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
            v[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
            v[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
            v[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
            v[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
            v[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
        }

        /*
            The panel of a token: its groups quantized by chunks, each group as its 256 bytes
            xq + 128 transposed, written to bytes, and as the scales sx and the sums c of its 8
            chunks, written to floats.
        */
        void quantize_panel(const float* x, long groups, uint8_t* bytes, float* floats)
        {
            const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
            block_q8_0x b[8];
            for (long g = 0; g < groups; ++g, x += 256, bytes += 256, floats += 16)
            {
                quantize_row_q8_0x(x, b, 256);
                __m256i v[8];
                for (int j = 0; j < 8; ++j)
                {
                    v[j] = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[j].qs)), flip);
                    floats[j] = b[j].d;
                    floats[8 + j] = b[j].s;
                }
                transpose_units(v);
                for (int k = 0; k < 8; ++k) _mm256_store_si256(reinterpret_cast<__m256i*>(bytes + 32 * k), v[k]);
            }
        }

        /*
            The 8 chunks of the group of weights at p as signed bytes, in the order of the
            features, and the floats of the group that come from its format: A in f[0..7], and
            in f[16..23] B, or A of the second halves.
        */
        void weight_chunks(panel_kind k, const uint8_t* p, __m256i c[8], float* f)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            switch (k)
            {
            case panel_kind::q4_0:
                for (int j = 0; j < 8; ++j, p += 18)
                {
                    f[j] = h2f(p);
                    c[j] = _mm256_sub_epi8(nibbles_to_bytes(p + 2), _mm256_set1_epi8(8));
                }
                break;
            case panel_kind::q8_0:
                for (int j = 0; j < 8; ++j, p += 34)
                {
                    f[j] = h2f(p);
                    c[j] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 2));
                }
                break;
            case panel_kind::q4_K:
            case panel_kind::q5_K:
            {
                // w = d sc q - dmin m for a chunk of scale sc and minimum m.
                uint32_t utmp[4];
                unpack_k4_scales(p + 4, utmp);
                const __m128i sm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(utmp));
                const __m256 sc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sm));
                const __m256 mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(sm, 8)));
                _mm256_store_ps(f, _mm256_mul_ps(_mm256_set1_ps(h2f(p)), sc));
                _mm256_store_ps(f + 16, _mm256_mul_ps(_mm256_set1_ps(-h2f(p + 2)), mn));
                if (k == panel_kind::q4_K)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16 + 32 * j));
                        c[2 * j] = _mm256_and_si256(v, m4);
                        c[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                    }
                }
                else
                {
                    // Bit 2j of each byte of the high bits is the fifth bit of chunk 2j, bit 2j + 1 that of 2j + 1.
                    const __m256i hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16));
                    const __m256i one = _mm256_set1_epi8(1);
                    for (int j = 0; j < 4; ++j)
                    {
                        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 48 + 32 * j));
                        const __m256i h0 = _mm256_and_si256(_mm256_srl_epi16(hb, _mm_cvtsi32_si128(2 * j)), one);
                        const __m256i h1 = _mm256_and_si256(_mm256_srl_epi16(hb, _mm_cvtsi32_si128(2 * j + 1)), one);
                        c[2 * j] = _mm256_or_si256(_mm256_and_si256(v, m4), _mm256_slli_epi16(h0, 4));
                        c[2 * j + 1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4),
                            _mm256_slli_epi16(h1, 4));
                    }
                }
                break;
            }
            case panel_kind::q6_K:
            {
                // w = d sc (q - 32), sc the signed scale of 16 features: the even ones scale the first halves.
                const __m256 d = _mm256_set1_ps(h2f(p + 208));
                const __m128i sc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 192));
                const __m128i even = _mm_shuffle_epi8(sc, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                    0, 2, 4, 6, 8, 10, 12, 14));
                const __m128i odd = _mm_shuffle_epi8(sc, _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,
                    1, 3, 5, 7, 9, 11, 13, 15));
                _mm256_store_ps(f, _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(even))));
                _mm256_store_ps(f + 16, _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(odd))));
                const __m256i m2 = _mm256_set1_epi8(3);
                const __m256i offset = _mm256_set1_epi8(32);
                for (int half = 0; half < 2; ++half)
                {
                    // The low four bits, then the two high bits moved to bits 4 and 5, as dequantize_q6_K_avx2().
                    const uint8_t* ql = p + 64 * half;
                    const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
                    const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
                    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 128 + 32 * half));
                    __m256i* q = c + 4 * half;
                    q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(h, m2), 4));
                    q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m2), 4));
                    q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m2), 4));
                    q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m2), 4));
                    for (int j = 0; j < 4; ++j) q[j] = _mm256_sub_epi8(q[j], offset);
                }
                break;
            }
            default:
                // No format of the panels: zeros.
                for (int j = 0; j < 8; ++j) c[j] = _mm256_setzero_si256();
                _mm256_store_ps(f, _mm256_setzero_ps());
                _mm256_store_ps(f + 16, _mm256_setzero_ps());
                break;
            }
        }

        // Lays out the group of weights at p in a panel: its 256 bytes, then its floats (panel_floats()).
        void pack_panel_group(panel_kind k, const uint8_t* p, uint8_t* bytes, float* f)
        {
            __m256i c[8];
            weight_chunks(k, p, c, f);
            transpose_units(c);
            // The sums of the bytes s of each lane over the first and over the second half of the vectors.
            const __m256i ones8 = _mm256_set1_epi8(1);
            const __m256i ones16 = _mm256_set1_epi16(1);
            __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j)
            {
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, c[j]), ones16));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, c[j + 4]), ones16));
            }
            for (int j = 0; j < 8; ++j) _mm256_store_si256(reinterpret_cast<__m256i*>(bytes + 32 * j), c[j]);
            const __m256 m128 = _mm256_set1_ps(-128.0f);
            if (panel_halves(k))
            {
                _mm256_store_ps(f + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(f), m128), _mm256_cvtepi32_ps(s0)));
                _mm256_store_ps(f + 24, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(f + 16), m128),
                    _mm256_cvtepi32_ps(s1)));
            }
            else
            {
                _mm256_store_ps(f + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(f), m128),
                    _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1))));
            }
        }

        // A block of rows of weights and of tokens in their panels, from one group on.
        struct panel_block
        {
            const uint8_t* w = nullptr;   // the bytes of the first row
            const float* wf = nullptr;    // the floats of the first row
            long w_stride = 0;            // bytes from a row to the next
            long wf_stride = 0;           // floats from a row to the next
            long wf_group = 0;            // floats of a group of a row
            const uint8_t* x = nullptr;   // the bytes of the first token
            const float* xf = nullptr;    // the floats of the first token
            long x_stride = 0;            // bytes from a token to the next
            long xf_stride = 0;           // floats from a token to the next
            long groups = 0;
        };

        // acc + the products of the scaled dot products of the lanes of a half: sx (A dot + Bw).
        inline void panel_fold(__m256& acc, __m256i dot, const float* f, const float* sx)
        {
            const __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_load_ps(f), _mm256_load_ps(f + 8));
            acc = _mm256_fmadd_ps(v, _mm256_load_ps(sx), acc);
        }

        /*
            The kernels: R <= 2 rows of weights against T <= 4 tokens over the groups of a
            block, each pair with its own accumulators, written to out[r * T + t].  The pairs go
            through the same operations whatever R and T, so that a token gets the same results
            in every block.  With VNNI, an instruction adds the four products of the bytes of
            each lane to its sum.  With AVX2, the products of pairs of bytes add up in 16 bits
            over RUN vectors, as many as the bounds of the format allow without overflow, before
            their sums in 32 bits.  The accumulators are named variables rather than arrays so
            that no compiler leaves them in memory.
        */
        template <int R, int T, bool HALVES, bool MINS, int RUN>
        void panel_kernel(const panel_block& b, float* out)
        {
            static_assert(R >= 1 && R <= 2 && T >= 1 && T <= 4, "blocks of 2 rows and 4 tokens at most");
            static_assert((HALVES ? 4 : 8) % RUN == 0, "the runs of 16-bit sums divide the halves");
            const __m256 zf = _mm256_setzero_ps();
            __m256 a00 = zf, a01 = zf, a02 = zf, a03 = zf, a10 = zf, a11 = zf, a12 = zf, a13 = zf;
#if !ALBIREO_QK_VNNI
            const __m256i ones16 = _mm256_set1_epi16(1);
#endif
            for (long g = 0; g < b.groups; ++g)
            {
                const uint8_t* w0 = b.w + 256 * g;
                const uint8_t* w1 = R > 1 ? w0 + b.w_stride : w0;
                const uint8_t* x0 = b.x + 256 * g;
                const uint8_t* x1 = T > 1 ? x0 + b.x_stride : x0;
                const uint8_t* x2 = T > 2 ? x1 + b.x_stride : x0;
                const uint8_t* x3 = T > 3 ? x2 + b.x_stride : x0;
                const float* f0 = b.wf + b.wf_group * g;
                const float* f1 = R > 1 ? f0 + b.wf_stride : f0;
                const float* s0 = b.xf + 16 * g;
                const float* s1 = T > 1 ? s0 + b.xf_stride : s0;
                const float* s2 = T > 2 ? s1 + b.xf_stride : s0;
                const float* s3 = T > 3 ? s2 + b.xf_stride : s0;
                for (int h = 0; h < (HALVES ? 2 : 1); ++h)
                {
                    const __m256i zi = _mm256_setzero_si256();
                    __m256i d00 = zi, d01 = zi, d02 = zi, d03 = zi, d10 = zi, d11 = zi, d12 = zi, d13 = zi;
                    const int first = HALVES ? 4 * h : 0;
                    const int last = HALVES ? 4 * h + 4 : 8;
#if ALBIREO_QK_VNNI
                    for (int k = first; k < last; ++k)
                    {
                        const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(w0 + 32 * k));
                        const __m256i v1 = R > 1
                            ? _mm256_load_si256(reinterpret_cast<const __m256i*>(w1 + 32 * k)) : zi;
                        __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x0 + 32 * k));
                        d00 = dpbusd(d00, x, v0);
                        if (R > 1) d10 = dpbusd(d10, x, v1);
                        if (T > 1)
                        {
                            x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x1 + 32 * k));
                            d01 = dpbusd(d01, x, v0);
                            if (R > 1) d11 = dpbusd(d11, x, v1);
                        }
                        if (T > 2)
                        {
                            x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x2 + 32 * k));
                            d02 = dpbusd(d02, x, v0);
                            if (R > 1) d12 = dpbusd(d12, x, v1);
                        }
                        if (T > 3)
                        {
                            x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x3 + 32 * k));
                            d03 = dpbusd(d03, x, v0);
                            if (R > 1) d13 = dpbusd(d13, x, v1);
                        }
                    }
#else
                    for (int k = first; k < last; k += RUN)
                    {
                        __m256i e00 = zi, e01 = zi, e02 = zi, e03 = zi, e10 = zi, e11 = zi, e12 = zi, e13 = zi;
                        for (int j = k; j < k + RUN; ++j)
                        {
                            const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(w0 + 32 * j));
                            const __m256i v1 = R > 1
                                ? _mm256_load_si256(reinterpret_cast<const __m256i*>(w1 + 32 * j)) : zi;
                            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x0 + 32 * j));
                            e00 = _mm256_add_epi16(e00, _mm256_maddubs_epi16(x, v0));
                            if (R > 1) e10 = _mm256_add_epi16(e10, _mm256_maddubs_epi16(x, v1));
                            if (T > 1)
                            {
                                x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x1 + 32 * j));
                                e01 = _mm256_add_epi16(e01, _mm256_maddubs_epi16(x, v0));
                                if (R > 1) e11 = _mm256_add_epi16(e11, _mm256_maddubs_epi16(x, v1));
                            }
                            if (T > 2)
                            {
                                x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x2 + 32 * j));
                                e02 = _mm256_add_epi16(e02, _mm256_maddubs_epi16(x, v0));
                                if (R > 1) e12 = _mm256_add_epi16(e12, _mm256_maddubs_epi16(x, v1));
                            }
                            if (T > 3)
                            {
                                x = _mm256_load_si256(reinterpret_cast<const __m256i*>(x3 + 32 * j));
                                e03 = _mm256_add_epi16(e03, _mm256_maddubs_epi16(x, v0));
                                if (R > 1) e13 = _mm256_add_epi16(e13, _mm256_maddubs_epi16(x, v1));
                            }
                        }
                        d00 = _mm256_add_epi32(d00, _mm256_madd_epi16(e00, ones16));
                        if (R > 1) d10 = _mm256_add_epi32(d10, _mm256_madd_epi16(e10, ones16));
                        if (T > 1) d01 = _mm256_add_epi32(d01, _mm256_madd_epi16(e01, ones16));
                        if (T > 1 && R > 1) d11 = _mm256_add_epi32(d11, _mm256_madd_epi16(e11, ones16));
                        if (T > 2) d02 = _mm256_add_epi32(d02, _mm256_madd_epi16(e02, ones16));
                        if (T > 2 && R > 1) d12 = _mm256_add_epi32(d12, _mm256_madd_epi16(e12, ones16));
                        if (T > 3) d03 = _mm256_add_epi32(d03, _mm256_madd_epi16(e03, ones16));
                        if (T > 3 && R > 1) d13 = _mm256_add_epi32(d13, _mm256_madd_epi16(e13, ones16));
                    }
#endif
                    const float* g0 = f0 + 16 * h;
                    const float* g1 = f1 + 16 * h;
                    panel_fold(a00, d00, g0, s0);
                    if (R > 1) panel_fold(a10, d10, g1, s0);
                    if (T > 1) panel_fold(a01, d01, g0, s1);
                    if (T > 1 && R > 1) panel_fold(a11, d11, g1, s1);
                    if (T > 2) panel_fold(a02, d02, g0, s2);
                    if (T > 2 && R > 1) panel_fold(a12, d12, g1, s2);
                    if (T > 3) panel_fold(a03, d03, g0, s3);
                    if (T > 3 && R > 1) panel_fold(a13, d13, g1, s3);
                }
                if (MINS)
                {
                    // The offsets of the chunks times the sums of the activations: B c.
                    const __m256 m0 = _mm256_load_ps(f0 + 16);
                    const __m256 m1 = _mm256_load_ps(f1 + 16);
                    a00 = _mm256_fmadd_ps(m0, _mm256_load_ps(s0 + 8), a00);
                    if (R > 1) a10 = _mm256_fmadd_ps(m1, _mm256_load_ps(s0 + 8), a10);
                    if (T > 1) a01 = _mm256_fmadd_ps(m0, _mm256_load_ps(s1 + 8), a01);
                    if (T > 1 && R > 1) a11 = _mm256_fmadd_ps(m1, _mm256_load_ps(s1 + 8), a11);
                    if (T > 2) a02 = _mm256_fmadd_ps(m0, _mm256_load_ps(s2 + 8), a02);
                    if (T > 2 && R > 1) a12 = _mm256_fmadd_ps(m1, _mm256_load_ps(s2 + 8), a12);
                    if (T > 3) a03 = _mm256_fmadd_ps(m0, _mm256_load_ps(s3 + 8), a03);
                    if (T > 3 && R > 1) a13 = _mm256_fmadd_ps(m1, _mm256_load_ps(s3 + 8), a13);
                }
            }
            out[0] = hsum_ps(a00);
            if (T > 1) out[1] = hsum_ps(a01);
            if (T > 2) out[2] = hsum_ps(a02);
            if (T > 3) out[3] = hsum_ps(a03);
            if (R > 1)
            {
                out[T] = hsum_ps(a10);
                if (T > 1) out[T + 1] = hsum_ps(a11);
                if (T > 2) out[T + 2] = hsum_ps(a12);
                if (T > 3) out[T + 3] = hsum_ps(a13);
            }
        }

        typedef void (*panel_fn)(const panel_block&, float*);

        // The kernels of a format, for 1 or 2 rows and 1 to 4 tokens.
        template <bool HALVES, bool MINS, int RUN>
        void panel_kernels(panel_fn fns[2][4])
        {
            fns[0][0] = panel_kernel<1, 1, HALVES, MINS, RUN>;
            fns[0][1] = panel_kernel<1, 2, HALVES, MINS, RUN>;
            fns[0][2] = panel_kernel<1, 3, HALVES, MINS, RUN>;
            fns[0][3] = panel_kernel<1, 4, HALVES, MINS, RUN>;
            fns[1][0] = panel_kernel<2, 1, HALVES, MINS, RUN>;
            fns[1][1] = panel_kernel<2, 2, HALVES, MINS, RUN>;
            fns[1][2] = panel_kernel<2, 3, HALVES, MINS, RUN>;
            fns[1][3] = panel_kernel<2, 4, HALVES, MINS, RUN>;
        }

        /*
            The runs of 16-bit sums of AVX2: (xq + 128) s summed over two bytes stays within
            2 x 255 x max |s|, and RUN of these sums within 32767: 8 for Q4_0 (s within [-8, 7]),
            4 for Q4_K (s within [0, 15]), 2 for Q5_K ([0, 31]) and Q6_K ([-32, 31]).
        */
        void panel_kernels(panel_kind k, panel_fn fns[2][4])
        {
            switch (k)
            {
            case panel_kind::q4_0: panel_kernels<false, false, 8>(fns); break;
            case panel_kind::q8_0: panel_kernels<false, false, 8>(fns); break;
            case panel_kind::q4_K: panel_kernels<false, true, 4>(fns); break;
            case panel_kind::q5_K: panel_kernels<false, true, 2>(fns); break;
            case panel_kind::q6_K: panel_kernels<true, false, 2>(fns); break;
            default: break;
            }
        }

        // The tokens of a block of the kernels, whose accumulators the 16 vector registers of AVX2 hold.
        const long panel_tokens = 3;
        const long panel_tile = 32;    // rows of weights per task
        const long panel_slice = 8;    // groups of 256 features per slice of the weights of a task
        // The bytes of the panels of the tokens a thread keeps from one product to the next (16 MB).
        const size_t panel_bytes_kept = size_t(1) << 24;

        /*
            The memory of the panels: n elements on 64 bytes, left as they are, a buffer that
            grows losing its contents.  Its functions belong to this file alone, so that each
            family of kernels keeps its own copy of them, compiled with its own instructions.
        */
        template <typename T>
        class panel_buffer
        {
        public:
            panel_buffer() {}
            panel_buffer(const panel_buffer&) = delete;
            panel_buffer& operator=(const panel_buffer&) = delete;
            ~panel_buffer() { release(); }

            T* get(size_t n)
            {
                if (n > size_)
                {
                    release();
                    if (n > (static_cast<size_t>(-1) - 128) / sizeof(T)) throw std::bad_alloc();
                    raw_ = static_cast<char*>(::operator new(n * sizeof(T) + 64));
                    const uintptr_t at = reinterpret_cast<uintptr_t>(raw_);
                    data_ = reinterpret_cast<T*>(raw_ + (64 - at % 64) % 64);
                    size_ = n;
                }
                return data_;
            }

            size_t size() const { return size_; }

            void release()
            {
                ::operator delete(raw_);
                raw_ = nullptr;
                data_ = nullptr;
                size_ = 0;
            }

        private:
            char* raw_ = nullptr;
            T* data_ = nullptr;
            size_t size_ = 0;
        };

        void matmul_panels(const weight_matrix& W, panel_kind kind, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, thread_pool& pool, bool accumulate)
        {
            const long rows = W.rows;
            const long groups = W.cols / 256;
            const long fg = panel_floats(kind);
            const size_t group_bytes = qk_row_bytes(W.type, 256);
            panel_fn fns[2][4];
            panel_kernels(kind, fns);

            // The panels of the tokens, once for all the rows.
            static thread_local panel_buffer<uint8_t> token_bytes;
            static thread_local panel_buffer<float> token_floats;
            uint8_t* const xb = token_bytes.get(static_cast<size_t>(n_tokens * groups) * 256);
            float* const xf = token_floats.get(static_cast<size_t>(n_tokens * groups) * 16);
            pool.parallel_for(n_tokens, [&](long t, int) {
                const size_t at = static_cast<size_t>(t) * static_cast<size_t>(groups);
                quantize_panel(X + static_cast<size_t>(t) * static_cast<size_t>(x_stride), groups, xb + at * 256,
                    xf + at * 16);
            });

            /*
                A task lays out a tile of rows by slices of groups, then runs every block of
                tokens against the rows of the slice, 2 rows at a time: the panels of a block of
                tokens stay in the first level cache while the rows of the slice go by.
            */
            const long n_tiles = (rows + panel_tile - 1) / panel_tile;
            pool.parallel_for(n_tiles, [&](long tile, int) {
                static thread_local panel_buffer<uint8_t> weight_bytes;
                static thread_local panel_buffer<float> weight_floats;
                const long r0 = tile * panel_tile;
                const long nr = std::min(panel_tile, rows - r0);
                const long slice = std::min(panel_slice, groups);
                uint8_t* const wb = weight_bytes.get(static_cast<size_t>(nr * slice) * 256);
                float* const wf = weight_floats.get(static_cast<size_t>(nr * slice * fg));
                for (long g0 = 0; g0 < groups; g0 += panel_slice)
                {
                    const long ng = std::min(panel_slice, groups - g0);
                    for (long r = 0; r < nr; ++r)
                    {
                        const uint8_t* src = W.data + static_cast<size_t>(r0 + r) * W.row_bytes
                            + static_cast<size_t>(g0) * group_bytes;
                        for (long g = 0; g < ng; ++g)
                            pack_panel_group(kind, src + static_cast<size_t>(g) * group_bytes,
                                wb + (r * ng + g) * 256, wf + (r * ng + g) * fg);
                    }
                    // The first slice writes the results, the next ones add to them.
                    const bool add = accumulate || g0 > 0;
                    panel_block b;
                    b.w_stride = ng * 256;
                    b.wf_stride = ng * fg;
                    b.wf_group = fg;
                    b.x_stride = groups * 256;
                    b.xf_stride = groups * 16;
                    b.groups = ng;
                    for (long t0 = 0; t0 < n_tokens; t0 += panel_tokens)
                    {
                        const long nt = std::min(panel_tokens, n_tokens - t0);
                        const size_t at = static_cast<size_t>(t0) * static_cast<size_t>(groups)
                            + static_cast<size_t>(g0);
                        b.x = xb + at * 256;
                        b.xf = xf + at * 16;
                        for (long r = 0; r < nr; r += 2)
                        {
                            const long rr = std::min<long>(2, nr - r);
                            b.w = wb + r * ng * 256;
                            b.wf = wf + r * ng * fg;
                            float out[8];
                            fns[rr - 1][nt - 1](b, out);
                            for (long i = 0; i < rr; ++i)
                                for (long t = 0; t < nt; ++t)
                                {
                                    float& y = Y[static_cast<size_t>(t0 + t) * static_cast<size_t>(y_stride)
                                        + static_cast<size_t>(r0 + r + i)];
                                    y = add ? y + out[i * nt + t] : out[i * nt + t];
                                }
                        }
                    }
                }
            });
            // Larger buffers go back to the system: the thread would otherwise keep the largest ones it needed.
            if (token_bytes.size() > panel_bytes_kept)
            {
                token_bytes.release();
                token_floats.release();
            }
        }
#endif
    }

    bool qk_integer_prompt(ggml_type t, long cols)
    {
#if ALBIREO_QK_AVX2
        return integer_prompt.load(std::memory_order_relaxed) && cols % 256 == 0
            && panel_kind_of(t) != panel_kind::none;
#else
        (void)t;
        (void)cols;
        return false;
#endif
    }

    void qk_set_integer_prompt(bool on) { integer_prompt.store(on); }

    void qk_matmul(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
        float* Y, long y_stride, thread_pool& pool, qk_scratch& scratch, bool accumulate, qk_path path)
    {
        if (!W.valid() || n_tokens <= 0) return;
        if (exact_products.load(std::memory_order_relaxed))
        {
            matmul_exact(W, X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
            return;
        }
        if (path == qk_path::prompt || (path == qk_path::automatic && n_tokens >= qk_float_product_tokens))
        {
#if ALBIREO_QK_AVX2
            if (qk_integer_prompt(W.type, W.cols))
            {
                matmul_panels(W, panel_kind_of(W.type), X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
                return;
            }
#endif
            matmul_float(W, X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
            return;
        }
        if (path == qk_path::floats)
        {
            matmul_float(W, X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
            return;
        }
        const act_format f = qk_activation_format(W.type);
        const long n = W.cols;
        const size_t act_bytes = qk_activation_row_bytes(f, n);

        // The activations, quantized once for all the rows of the matrix.
        const uint8_t* act = nullptr;
        if (f == act_format::f32 && x_stride == n)
        {
            act = reinterpret_cast<const uint8_t*>(X);
        }
        else
        {
            scratch.act.resize(act_bytes * static_cast<size_t>(n_tokens));
            uint8_t* dst = scratch.act.data();
            if (n_tokens >= 8)
            {
                pool.parallel_for(n_tokens, [&](long t, int) {
                    qk_quantize_row(f, X + t * x_stride, dst + static_cast<size_t>(t) * act_bytes, n);
                });
            }
            else
            {
                for (long t = 0; t < n_tokens; ++t)
                    qk_quantize_row(f, X + t * x_stride, dst + static_cast<size_t>(t) * act_bytes, n);
            }
            act = dst;
        }

        // Groups of output rows, enough of them to balance the threads, few enough for
        // each group to amortize its hand-over.
        const long rows = W.rows;
        const long threads = pool.size();
        long group = std::max<long>(1, rows / (threads * 8));
        group = std::min<long>(group, 128);
        if (n_tokens > 1) group = std::max<long>(1, std::min<long>(group, 32));
        const long n_groups = (rows + group - 1) / group;
        const ggml_type type = W.type;
        const uint8_t* data = W.data;
        const size_t row_bytes = W.row_bytes;

        // With several tokens, the tokens go by tiles: the rows of a group and the
        // activations of a tile then stay in the caches while every pair is computed.
        const long tile = 16;
        pool.parallel_for(n_groups, [&](long g, int) {
            const long r0 = g * group;
            const long r1 = std::min(rows, r0 + group);
            for (long t0 = 0; t0 < n_tokens; t0 += tile)
            {
                const long t1 = std::min(n_tokens, t0 + tile);
                for (long r = r0; r < r1; ++r)
                {
                    const uint8_t* wrow = data + static_cast<size_t>(r) * row_bytes;
                    // Up to four tokens at a time, for which the weights of the row are unpacked
                    // once; a token left alone goes to qk_dot() directly.
                    long t = t0;
                    for (; t + 1 < t1; t += 4)
                    {
                        const long nt = std::min<long>(4, t1 - t);
                        const uint8_t* rows[4];
                        float v[4];
                        for (long i = 0; i < nt; ++i) rows[i] = act + static_cast<size_t>(t + i) * act_bytes;
                        dot_rows(type, wrow, rows, nt, n, v);
                        for (long i = 0; i < nt; ++i)
                        {
                            float& out = Y[(t + i) * y_stride + r];
                            out = accumulate ? out + v[i] : v[i];
                        }
                    }
                    for (; t < t1; ++t)
                    {
                        const float v = qk_dot(type, wrow, act + static_cast<size_t>(t) * act_bytes, n);
                        float& out = Y[t * y_stride + r];
                        out = accumulate ? out + v : v;
                    }
                }
            }
        });
    }

    // Vector helpers

    float qk_dot_f32(const float* a, const float* b, long n)
    {
#if ALBIREO_QK_AVX2
        return dot_f32_avx2(a, b, n);
#else
        return dot_f32_ref(a, b, n);
#endif
    }

    void qk_scale(float* x, float s, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX2
        const __m256 sv = _mm256_set1_ps(s);
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), sv));
#endif
        for (; i < n; ++i) x[i] *= s;
    }

    float qk_max(const float* x, long n)
    {
        float m = -INFINITY;
        long i = 0;
#if ALBIREO_QK_AVX2
        if (n >= 8)
        {
            __m256 mv = _mm256_loadu_ps(x);
            for (i = 8; i + 8 <= n; i += 8) mv = _mm256_max_ps(mv, _mm256_loadu_ps(x + i));
            __m128 r = _mm_max_ps(_mm256_extractf128_ps(mv, 1), _mm256_castps256_ps128(mv));
            r = _mm_max_ps(r, _mm_movehl_ps(r, r));
            r = _mm_max_ss(r, _mm_movehdup_ps(r));
            m = _mm_cvtss_f32(r);
        }
#endif
        for (; i < n; ++i) m = std::max(m, x[i]);
        return m;
    }

#if ALBIREO_QK_AVX2
    namespace
    {
        // exp(x) for eight floats, accurate to a few units in the last place over the
        // range a softmax uses (x <= 0), after the classic Cephes reduction.
        inline __m256 exp256_ps(__m256 x)
        {
            const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
            const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
            const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
            const __m256 c1 = _mm256_set1_ps(0.693359375f);
            const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
            x = _mm256_min_ps(x, exp_hi);
            x = _mm256_max_ps(x, exp_lo);
            __m256 fx = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
            fx = _mm256_floor_ps(fx);
            x = _mm256_fnmadd_ps(fx, c1, x);
            x = _mm256_fnmadd_ps(fx, c2, x);
            const __m256 z = _mm256_mul_ps(x, x);
            __m256 y = _mm256_set1_ps(1.9875691500e-4f);
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
            y = _mm256_fmadd_ps(y, z, x);
            y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
            __m256i emm0 = _mm256_cvttps_epi32(fx);
            emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
            emm0 = _mm256_slli_epi32(emm0, 23);
            return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
        }
    }
#endif

    float qk_exp_shift_sum(float* x, float m, long n)
    {
        float sum = 0.0f;
        long i = 0;
#if ALBIREO_QK_AVX2
        const __m256 mv = _mm256_set1_ps(m);
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8)
        {
            const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), mv));
            _mm256_storeu_ps(x + i, e);
            acc = _mm256_add_ps(acc, e);
        }
        sum = hsum_ps(acc);
#endif
        for (; i < n; ++i)
        {
            x[i] = std::exp(x[i] - m);
            sum += x[i];
        }
        return sum;
    }

    void qk_silu_mul(float* g, const float* u, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX2
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 sign = _mm256_set1_ps(-0.0f);
        for (; i + 8 <= n; i += 8)
        {
            const __m256 x = _mm256_loadu_ps(g + i);
            const __m256 e = exp256_ps(_mm256_xor_ps(x, sign));
            _mm256_storeu_ps(g + i, _mm256_mul_ps(_mm256_div_ps(x, _mm256_add_ps(one, e)), _mm256_loadu_ps(u + i)));
        }
#endif
        for (; i < n; ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    }

    // Attention by tiles

    namespace
    {
        // The keys or the values converted to floats at a time: a few kilobytes, which stay
        // in the first level cache while every row of the tile reads them.
        inline long attention_group(long hd)
        {
            return std::max<long>(4, std::min<long>(64, 4096 / std::max<long>(1, hd)) / 4 * 4);
        }

#if ALBIREO_QK_AVX2
        // The sums of four vectors, as the four floats of one vector.
        inline __m128 hsum4_ps(__m256 a, __m256 b, __m256 c, __m256 d)
        {
            const __m256 s = _mm256_hadd_ps(_mm256_hadd_ps(a, b), _mm256_hadd_ps(c, d));
            return _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        }
#endif

        /*
            The dot products of R rows of queries (R being 1 or 2) with four consecutive keys
            of hd floats: s[r][c] = dot(q[r], k + c * hd).
        */
        template <int R>
        void dot_rows_x4(const float* const* q, const float* k, long hd, float* const* s)
        {
            long i = 0;
#if ALBIREO_QK_AVX2
            __m256 w[R][4];
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c) w[r][c] = _mm256_setzero_ps();
            for (; i + 8 <= hd; i += 8)
            {
                __m256 x[R];
                for (int r = 0; r < R; ++r) x[r] = _mm256_loadu_ps(q[r] + i);
                for (int c = 0; c < 4; ++c)
                {
                    const __m256 kv = _mm256_loadu_ps(k + c * hd + i);
                    for (int r = 0; r < R; ++r) w[r][c] = _mm256_fmadd_ps(x[r], kv, w[r][c]);
                }
            }
            for (int r = 0; r < R; ++r) _mm_storeu_ps(s[r], hsum4_ps(w[r][0], w[r][1], w[r][2], w[r][3]));
            if (i < hd)
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < 4; ++c)
                    {
                        float t = 0;
                        for (long e = i; e < hd; ++e) t += q[r][e] * k[c * hd + e];
                        s[r][c] += t;
                    }
#else
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c)
                {
                    float t = 0;
                    for (i = 0; i < hd; ++i) t += q[r][i] * k[c * hd + i];
                    s[r][c] = t;
                }
#endif
        }

        /*
            The dot products of R rows of queries with one key, s[r][0] = dot(q[r], k), through
            the operations of dot_rows_x4() for each of its keys: the score of a key never
            depends on the keys computed with it, so that a token gets the scores of a step of
            its own in a pass of several tokens of its sequence.
        */
        template <int R>
        void dot_rows_x1(const float* const* q, const float* k, long hd, float* const* s)
        {
            long i = 0;
#if ALBIREO_QK_AVX2
            __m256 w[R];
            for (int r = 0; r < R; ++r) w[r] = _mm256_setzero_ps();
            for (; i + 8 <= hd; i += 8)
            {
                const __m256 kv = _mm256_loadu_ps(k + i);
                for (int r = 0; r < R; ++r) w[r] = _mm256_fmadd_ps(_mm256_loadu_ps(q[r] + i), kv, w[r]);
            }
            for (int r = 0; r < R; ++r) s[r][0] = _mm_cvtss_f32(hsum4_ps(w[r], w[r], w[r], w[r]));
            if (i < hd)
                for (int r = 0; r < R; ++r)
                {
                    float t = 0;
                    for (long e = i; e < hd; ++e) t += q[r][e] * k[e];
                    s[r][0] += t;
                }
#else
            for (int r = 0; r < R; ++r)
            {
                float t = 0;
                for (i = 0; i < hd; ++i) t += q[r][i] * k[i];
                s[r][0] = t;
            }
#endif
        }

        /*
            The values of g keys weighted for R rows (R being 1 or 2) and added to the rows of
            the output: o[r][i] += sum over j < g of p[r][j] * v[j * hd + i].
        */
        template <int R>
        void add_weighted_rows(const float* const* p, const float* v, long g, long hd, float* const* o)
        {
            long i = 0;
#if ALBIREO_QK_AVX2
            for (; i + 32 <= hd; i += 32)
            {
                __m256 a[R][4];
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < 4; ++c) a[r][c] = _mm256_loadu_ps(o[r] + i + 8 * c);
                for (long j = 0; j < g; ++j)
                {
                    const float* vr = v + j * hd + i;
                    __m256 wr[R];
                    for (int r = 0; r < R; ++r) wr[r] = _mm256_set1_ps(p[r][j]);
                    for (int c = 0; c < 4; ++c)
                    {
                        const __m256 x = _mm256_loadu_ps(vr + 8 * c);
                        for (int r = 0; r < R; ++r) a[r][c] = _mm256_fmadd_ps(wr[r], x, a[r][c]);
                    }
                }
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < 4; ++c) _mm256_storeu_ps(o[r] + i + 8 * c, a[r][c]);
            }
            for (; i + 8 <= hd; i += 8)
            {
                __m256 a[R];
                for (int r = 0; r < R; ++r) a[r] = _mm256_loadu_ps(o[r] + i);
                for (long j = 0; j < g; ++j)
                {
                    const __m256 x = _mm256_loadu_ps(v + j * hd + i);
                    for (int r = 0; r < R; ++r) a[r] = _mm256_fmadd_ps(_mm256_set1_ps(p[r][j]), x, a[r]);
                }
                for (int r = 0; r < R; ++r) _mm256_storeu_ps(o[r] + i, a[r]);
            }
#endif
            for (; i < hd; ++i)
                for (int r = 0; r < R; ++r)
                {
                    float t = o[r][i];
                    for (long j = 0; j < g; ++j) t += p[r][j] * v[j * hd + i];
                    o[r][i] = t;
                }
        }
    }

    long qk_attention_scratch(long hd)
    {
        return attention_group(hd) * hd;
    }

    namespace
    {
        // The scores of the rows of queries over m keys, which expand(j0, g, to) gives as floats g at a time.
        template <typename Expand>
        void attention_scores(const float* Q, long rows, long m, long hd, float* S, long lds, float* scratch,
            const Expand& expand)
        {
            const long G = attention_group(hd);
            for (long j0 = 0; j0 < m; j0 += G)
            {
                const long g = std::min(G, m - j0);
                expand(j0, g, scratch);
                long r = 0;
                for (; r + 2 <= rows; r += 2)
                {
                    const float* q[2] = { Q + r * hd, Q + (r + 1) * hd };
                    long j = 0;
                    for (; j + 4 <= g; j += 4)
                    {
                        float* s[2] = { S + r * lds + j0 + j, S + (r + 1) * lds + j0 + j };
                        dot_rows_x4<2>(q, scratch + j * hd, hd, s);
                    }
                    for (; j < g; ++j)
                    {
                        float* s[2] = { S + r * lds + j0 + j, S + (r + 1) * lds + j0 + j };
                        dot_rows_x1<2>(q, scratch + j * hd, hd, s);
                    }
                }
                for (; r < rows; ++r)
                {
                    const float* q[1] = { Q + r * hd };
                    long j = 0;
                    for (; j + 4 <= g; j += 4)
                    {
                        float* s[1] = { S + r * lds + j0 + j };
                        dot_rows_x4<1>(q, scratch + j * hd, hd, s);
                    }
                    for (; j < g; ++j)
                    {
                        float* s[1] = { S + r * lds + j0 + j };
                        dot_rows_x1<1>(q, scratch + j * hd, hd, s);
                    }
                }
            }
        }

        // The values of m keys weighted and added to the rows of the output, expand giving them as above.
        template <typename Expand>
        void attention_values(const float* P, long lds, long rows, long m, long hd, float* O, float* scratch,
            const Expand& expand)
        {
            const long G = attention_group(hd);
            for (long j0 = 0; j0 < m; j0 += G)
            {
                const long g = std::min(G, m - j0);
                expand(j0, g, scratch);
                long r = 0;
                for (; r + 2 <= rows; r += 2)
                {
                    const float* p[2] = { P + r * lds + j0, P + (r + 1) * lds + j0 };
                    float* o[2] = { O + r * hd, O + (r + 1) * hd };
                    add_weighted_rows<2>(p, scratch, g, hd, o);
                }
                for (; r < rows; ++r)
                {
                    const float* p[1] = { P + r * lds + j0 };
                    float* o[1] = { O + r * hd };
                    add_weighted_rows<1>(p, scratch, g, hd, o);
                }
            }
        }
    }

    void qk_attention_scores(const float* Q, long rows, const uint16_t* K, long m, long hd, float* S, long lds,
        float* scratch)
    {
        attention_scores(Q, rows, m, hd, S, lds, scratch,
            [=](long j0, long g, float* to) { qk_half_to_float_row(K + j0 * hd, to, g * hd); });
    }

    void qk_attention_values(const float* P, long lds, long rows, const uint16_t* V, long m, long hd, float* O,
        float* scratch)
    {
        attention_values(P, lds, rows, m, hd, O, scratch,
            [=](long j0, long g, float* to) { qk_half_to_float_row(V + j0 * hd, to, g * hd); });
    }

    void qk_attention_scores_q8(const float* Q, long rows, const int8_t* K, const uint16_t* K_scales, long m, long hd,
        float* S, long lds, float* scratch)
    {
        attention_scores(Q, rows, m, hd, S, lds, scratch, [=](long j0, long g, float* to) {
            qk_q8_to_float_row(K + j0 * hd, K_scales + j0 * (hd / 32), to, g * hd);
        });
    }

    void qk_attention_values_q8(const float* P, long lds, long rows, const int8_t* V, const uint16_t* V_scales, long m,
        long hd, float* O, float* scratch)
    {
        attention_values(P, lds, rows, m, hd, O, scratch, [=](long j0, long g, float* to) {
            qk_q8_to_float_row(V + j0 * hd, V_scales + j0 * (hd / 32), to, g * hd);
        });
    }

    // Encoders

    void qk_layer_norm(const float* x, const float* gamma, const float* beta, float* y, long n, float eps)
    {
        float sum = 0.0f;
        long i = 0;
#if ALBIREO_QK_AVX2
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (; i + 16 <= n; i += 16)
        {
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x + i));
            s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x + i + 8));
        }
        sum = hsum_ps(_mm256_add_ps(s0, s1));
#endif
        for (; i < n; ++i) sum += x[i];
        const float mean = sum / static_cast<float>(n);
        float squares = 0.0f;
        i = 0;
#if ALBIREO_QK_AVX2
        const __m256 mv = _mm256_set1_ps(mean);
        s0 = _mm256_setzero_ps();
        s1 = _mm256_setzero_ps();
        for (; i + 16 <= n; i += 16)
        {
            const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(x + i), mv);
            const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(x + i + 8), mv);
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
        }
        squares = hsum_ps(_mm256_add_ps(s0, s1));
#endif
        for (; i < n; ++i) squares += (x[i] - mean) * (x[i] - mean);
        const float inv = 1.0f / std::sqrt(squares / static_cast<float>(n) + eps);
        i = 0;
#if ALBIREO_QK_AVX2
        const __m256 iv = _mm256_set1_ps(inv);
        for (; i + 8 <= n; i += 8)
        {
            const __m256 d = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), mv), iv);
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(d, _mm256_loadu_ps(gamma + i), _mm256_loadu_ps(beta + i)));
        }
#endif
        for (; i < n; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }

    void qk_gelu_tanh(float* x, long n)
    {
        const float c = 0.7978845608028654f;   // sqrt(2 / pi)
        const float a = 0.044715f;
        long i = 0;
#if ALBIREO_QK_AVX2
        // -2u == x (-2c - 2ca x^2)
        const __m256 c1 = _mm256_set1_ps(-2.0f * c), c3 = _mm256_set1_ps(-2.0f * c * a);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8 <= n; i += 8)
        {
            const __m256 v = _mm256_loadu_ps(x + i);
            const __m256 t = _mm256_mul_ps(v, _mm256_fmadd_ps(c3, _mm256_mul_ps(v, v), c1));
            _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(one, exp256_ps(t))));
        }
#endif
        for (; i < n; ++i)
        {
            const float v = x[i];
            x[i] = v / (1.0f + std::exp(-2.0f * c * (v + a * v * v * v)));
        }
    }

    namespace
    {
        // erf(z) for z >= 0 after Abramowitz and Stegun 7.1.26: 1 - t P(t) exp(-z^2) with
        // t = 1 / (1 + p z), within 1.5e-7.
        const float erf_p = 0.3275911f;
        const float erf_a1 = 0.254829592f, erf_a2 = -0.284496736f, erf_a3 = 1.421413741f;
        const float erf_a4 = -1.453152027f, erf_a5 = 1.061405429f;

        inline float erf_positive(float z)
        {
            const float t = 1.0f / (1.0f + erf_p * z);
            const float poly = t * (erf_a1 + t * (erf_a2 + t * (erf_a3 + t * (erf_a4 + t * erf_a5))));
            return 1.0f - poly * std::exp(-z * z);
        }
    }

    void qk_gelu_erf(float* x, long n)
    {
        const float s = 0.7071067811865476f;   // 1 / sqrt(2)
        long i = 0;
#if ALBIREO_QK_AVX2
        const __m256 sign = _mm256_set1_ps(-0.0f);
        const __m256 one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f), sv = _mm256_set1_ps(s);
        const __m256 pv = _mm256_set1_ps(erf_p);
        for (; i + 8 <= n; i += 8)
        {
            const __m256 v = _mm256_loadu_ps(x + i);
            const __m256 z = _mm256_mul_ps(_mm256_andnot_ps(sign, v), sv);
            const __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(pv, z, one));
            __m256 poly = _mm256_fmadd_ps(_mm256_set1_ps(erf_a5), t, _mm256_set1_ps(erf_a4));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(erf_a3));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(erf_a2));
            poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(erf_a1));
            poly = _mm256_mul_ps(poly, t);
            const __m256 e = exp256_ps(_mm256_xor_ps(_mm256_mul_ps(z, z), sign));
            // erf(|v| / sqrt(2)), then the sign of v.
            const __m256 erf_abs = _mm256_fnmadd_ps(poly, e, one);
            const __m256 erf_v = _mm256_or_ps(erf_abs, _mm256_and_ps(v, sign));
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, erf_v)));
        }
#endif
        for (; i < n; ++i)
        {
            const float v = x[i];
            const float e = erf_positive(std::fabs(v) * s);
            x[i] = 0.5f * v * (1.0f + (v < 0.0f ? -e : e));
        }
    }

    void qk_add(float* y, const float* x, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX2
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i)));
#endif
        for (; i < n; ++i) y[i] += x[i];
    }

    namespace
    {
        // The keys of the encoder attention read at a time: the scores of a tile of queries
        // against them take 32 kilobytes.
        const long encoder_block_keys = 128;


        /*
            S[r * lds + j] = sum over d < hd of Q[r * hd + d] * Kt[d * ldk + j], for R rows and the
            n keys of a group: exactly 16 with AVX2, any number otherwise.
            Each key of Kt is read once for the R rows, without any horizontal sum.
        */
        template <int R>
        void encoder_scores_group(const float* Q, long hd, const float* Kt, long ldk, long n, float* S, long lds)
        {
#if ALBIREO_QK_AVX2
            (void)n;
            __m256 a0[R], a1[R];
            for (int r = 0; r < R; ++r) a0[r] = a1[r] = _mm256_setzero_ps();
            for (long d = 0; d < hd; ++d)
            {
                const float* k = Kt + d * ldk;
                const __m256 k0 = _mm256_loadu_ps(k);
                const __m256 k1 = _mm256_loadu_ps(k + 8);
                for (int r = 0; r < R; ++r)
                {
                    const __m256 q = _mm256_broadcast_ss(Q + r * hd + d);
                    a0[r] = _mm256_fmadd_ps(q, k0, a0[r]);
                    a1[r] = _mm256_fmadd_ps(q, k1, a1[r]);
                }
            }
            for (int r = 0; r < R; ++r)
            {
                _mm256_storeu_ps(S + r * lds, a0[r]);
                _mm256_storeu_ps(S + r * lds + 8, a1[r]);
            }
#else
            for (int r = 0; r < R; ++r)
                for (long j = 0; j < n; ++j)
                {
                    float t = 0.0f;
                    for (long d = 0; d < hd; ++d) t += Q[r * hd + d] * Kt[d * ldk + j];
                    S[r * lds + j] = t;
                }
#endif
        }

        void encoder_scores(const float* Q, long rows, long hd, const float* Kt, long ldk, long n, float* S, long lds)
        {
            const long G = 16;
            for (long r = 0; r < rows; r += 6)
            {
                const long R = std::min<long>(6, rows - r);
                const float* q = Q + r * hd;
                for (long j = 0; j < n; j += G)
                {
                    const long g = std::min(G, n - j);
                    float* s = S + r * lds + j;
#if ALBIREO_QK_AVX2
                    if (g < G)
                    {
                        // The last keys of a block, fewer than a vector pair.
                        for (long i = 0; i < R; ++i)
                            for (long c = 0; c < g; ++c)
                            {
                                float t = 0.0f;
                                for (long d = 0; d < hd; ++d) t += q[i * hd + d] * Kt[d * ldk + j + c];
                                s[i * lds + c] = t;
                            }
                        continue;
                    }
#endif
                    switch (R)
                    {
                    case 6: encoder_scores_group<6>(q, hd, Kt + j, ldk, g, s, lds); break;
                    case 5: encoder_scores_group<5>(q, hd, Kt + j, ldk, g, s, lds); break;
                    case 4: encoder_scores_group<4>(q, hd, Kt + j, ldk, g, s, lds); break;
                    case 3: encoder_scores_group<3>(q, hd, Kt + j, ldk, g, s, lds); break;
                    case 2: encoder_scores_group<2>(q, hd, Kt + j, ldk, g, s, lds); break;
                    default: encoder_scores_group<1>(q, hd, Kt + j, ldk, g, s, lds); break;
                    }
                }
            }
        }

        /*
            O[r * ldo + i] += sum over j < n of P[r * lds + j] * V[j * ldv + i], for R rows and
            the w dimensions of a group: exactly 32 with AVX2, any number otherwise.  Each row of
            V is read once for the R rows.
        */
        template <int R>
        void encoder_values_group(const float* P, long lds, const float* V, long ldv, long n, long w, float* O,
            long ldo)
        {
#if ALBIREO_QK_AVX2
            (void)w;
            __m256 a[R][4];
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c) a[r][c] = _mm256_loadu_ps(O + r * ldo + 8 * c);
            for (long j = 0; j < n; ++j)
            {
                const float* v = V + j * ldv;
                __m256 x[4];
                for (int c = 0; c < 4; ++c) x[c] = _mm256_loadu_ps(v + 8 * c);
                for (int r = 0; r < R; ++r)
                {
                    const __m256 p = _mm256_broadcast_ss(P + r * lds + j);
                    for (int c = 0; c < 4; ++c) a[r][c] = _mm256_fmadd_ps(p, x[c], a[r][c]);
                }
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c) _mm256_storeu_ps(O + r * ldo + 8 * c, a[r][c]);
#else
            for (int r = 0; r < R; ++r)
                for (long j = 0; j < n; ++j)
                {
                    const float p = P[r * lds + j];
                    for (long i = 0; i < w; ++i) O[r * ldo + i] += p * V[j * ldv + i];
                }
#endif
        }

        void encoder_values(const float* P, long lds, long rows, const float* V, long ldv, long n, long hd, float* O,
            long ldo)
        {
#if ALBIREO_QK_AVX2
            const long RG = 2, W = 32;
#else
            const long RG = 1, W = hd;
#endif
            for (long r = 0; r < rows; r += RG)
            {
                const long R = std::min(RG, rows - r);
                for (long i = 0; i < hd; i += W)
                {
                    const long w = std::min(W, hd - i);
                    const float* p = P + r * lds;
                    float* o = O + r * ldo + i;
#if ALBIREO_QK_AVX2
                    if (w < W)
                    {
                        for (long e = 0; e < R; ++e)
                            for (long j = 0; j < n; ++j)
                                for (long c = 0; c < w; ++c) o[e * ldo + c] += p[e * lds + j] * V[j * ldv + i + c];
                        continue;
                    }
#endif
#if ALBIREO_QK_AVX2
                    switch (R)
                    {
                    case 2: encoder_values_group<2>(p, lds, V + i, ldv, n, w, o, ldo); break;
                    default: encoder_values_group<1>(p, lds, V + i, ldv, n, w, o, ldo); break;
                    }
#else
                    // The portable build has no vector form for several rows: one row at a time.
                    (void)R;
                    encoder_values_group<1>(p, lds, V + i, ldv, n, w, o, ldo);
#endif
                }
            }
        }
    }

    long qk_encoder_attention_work(long)
    {
        return qk_encoder_tile_rows * encoder_block_keys + 2 * qk_encoder_tile_rows;
    }

    void qk_encoder_attention(const float* Q, long rows, const float* Kt, long ldk, const float* V, long ldv, long m,
        long hd, float* O, long ldo, float* work)
    {
        const long B = encoder_block_keys;
        float* S = work;
        float* top = work + qk_encoder_tile_rows * B;
        float* sum = top + qk_encoder_tile_rows;
        for (long r = 0; r < rows; ++r)
        {
            std::fill(O + r * ldo, O + r * ldo + hd, 0.0f);
            top[r] = -INFINITY;
            sum[r] = 0.0f;
        }
        for (long j0 = 0; j0 < m; j0 += B)
        {
            const long b = std::min(B, m - j0);
            encoder_scores(Q, rows, hd, Kt + j0, ldk, b, S, B);
            for (long r = 0; r < rows; ++r)
            {
                float* s = S + r * B;
                const float mx = std::max(top[r], qk_max(s, b));
                if (mx > top[r] && sum[r] > 0.0f)
                {
                    // A larger maximum: what the row gathered so far shrinks.
                    const float shrink = std::exp(top[r] - mx);
                    sum[r] *= shrink;
                    qk_scale(O + r * ldo, shrink, hd);
                }
                top[r] = mx;
                sum[r] += qk_exp_shift_sum(s, mx, b);
            }
            encoder_values(S, B, rows, V + j0 * ldv, ldv, b, hd, O, ldo);
        }
        for (long r = 0; r < rows; ++r) qk_scale(O + r * ldo, 1.0f / sum[r], hd);
    }

}
