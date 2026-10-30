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
    The instructions beyond AVX2, each the kernels of a processor engine:
        ALBIREO_QK_AVX512   AVX-512 F, BW, DQ and VL with VNNI (cpu-avx512): the products on
                            512-bit vectors, and the 256-bit kernels of ALBIREO_QK_VNNI;
        ALBIREO_QK_VNNI     the dot products of bytes and of 16-bit integers in one
                            instruction, on 256-bit vectors: AVX-VNNI (cpu-avxvnni), or its
                            AVX-512 form (ALBIREO_QK_VNNI_EVEX, which tests these kernels on
                            the processors that have AVX-512 VNNI and lack AVX-VNNI).
    MSVC defines no macro for VNNI: the build names it (ALBIREO_AVX_VNNI, ALBIREO_AVX512_VNNI).
*/
#if ALBIREO_QK_AVX2 && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) \
    && defined(__AVX512VL__) && (defined(__AVX512VNNI__) || defined(ALBIREO_AVX512_VNNI))
#define ALBIREO_QK_AVX512 1
#define ALBIREO_QK_VNNI 1
#define ALBIREO_QK_VNNI_FORM_EVEX 1
#elif ALBIREO_QK_AVX2 && defined(ALBIREO_QK_VNNI_EVEX) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define ALBIREO_QK_VNNI 1
#define ALBIREO_QK_VNNI_FORM_EVEX 1
#elif ALBIREO_QK_AVX2 && (defined(__AVXVNNI__) || defined(ALBIREO_AVX_VNNI))
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

    const char* qk_isa()
    {
#if ALBIREO_QK_AVX512
        return "AVX-512+VNNI";
#elif ALBIREO_QK_VNNI && ALBIREO_QK_VNNI_FORM_EVEX
        return "AVX2+FMA+F16C+VNNI (AVX-512 form)";
#elif ALBIREO_QK_VNNI
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
#if ALBIREO_QK_AVX512 || ALBIREO_QK_VNNI_FORM_EVEX
        isa |= qk_isa_avx512;
#elif ALBIREO_QK_VNNI
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
#if ALBIREO_QK_VNNI_FORM_EVEX
            return _mm256_dpbusd_epi32(acc, u, s);
#else
            return _mm256_dpbusd_avx_epi32(acc, u, s);
#endif
        }

#if !ALBIREO_QK_AVX512
        // acc + the sums of the products of two signed 16-bit integers of a by those of b.
        inline __m256i dpwssd(__m256i acc, __m256i a, __m256i b)
        {
#if ALBIREO_QK_VNNI_FORM_EVEX
            return _mm256_dpwssd_epi32(acc, a, b);
#else
            return _mm256_dpwssd_avx_epi32(acc, a, b);
#endif
        }
#endif

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

#if !ALBIREO_QK_AVX512
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
#endif
    }
#endif

// AVX-512 dot products and products of floats

#if ALBIREO_QK_AVX512
    namespace
    {
        inline float hsum512_ps(__m512 x)
        {
            return hsum_ps(_mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1)));
        }

        // Two vectors of 256 bits as one of 512, a in the low half.
        inline __m512i join256(__m256i a, __m256i b)
        {
            return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
        }

        inline __m512 join256_ps(float a, float b)
        {
            return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(a)), _mm256_set1_ps(b), 1);
        }

        inline __m256i load256(const void* p)
        {
            return _mm256_loadu_si256(static_cast<const __m256i*>(p));
        }

        // Sixteen 16-bit lanes of value a, then sixteen of value b.
        inline __m512i halves_epi16(short a, short b)
        {
            return join256(_mm256_set1_epi16(a), _mm256_set1_epi16(b));
        }

        // Eight 16-bit lanes of each of a, a+1, a+2 and a+3.
        inline __m512i quarters_epi16(short a)
        {
            const __m256i lo = _mm256_set_m128i(_mm_set1_epi16(static_cast<short>(a + 1)), _mm_set1_epi16(a));
            const __m256i hi = _mm256_set_m128i(_mm_set1_epi16(static_cast<short>(a + 3)),
                _mm_set1_epi16(static_cast<short>(a + 2)));
            return join256(lo, hi);
        }

        // Two blocks of Q8_0 at a time, the last odd one by the 256-bit kernel.
        float dot_q8_0_avx512(const uint8_t* w, const block_q8_0x* y, long n)
        {
            const long nb = n / 32;
            __m512 acc = _mm512_setzero_ps();
            long b = 0;
            for (; b + 2 <= nb; b += 2, w += 68)
            {
                const __m512i qx = join256(load256(w + 2), load256(w + 36));
                const __m512i qy = join256(load256(y[b].qs), load256(y[b + 1].qs));
                // |x| times the activations with the signs of x.
                const __m512i sy = _mm512_mask_sub_epi8(qy, _mm512_movepi8_mask(qx), _mm512_setzero_si512(), qy);
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(qx), sy);
                acc = _mm512_fmadd_ps(join256_ps(h2f(w) * y[b].d, h2f(w + 34) * y[b + 1].d),
                    _mm512_cvtepi32_ps(dot), acc);
            }
            float s = hsum512_ps(acc);
            if (b < nb) s += dot_q8_0_vnni(w, y + b, 32);
            return s;
        }

        float dot_q4_0_avx512(const uint8_t* w, const block_q8_0x* y, long n)
        {
            const long nb = n / 32;
            __m512 acc = _mm512_setzero_ps();
            float offset = 0.0f;
            long b = 0;
            for (; b + 2 <= nb; b += 2, w += 36)
            {
                const float d0 = h2f(w), d1 = h2f(w + 18);
                const __m512i qx = join256(nibbles_to_bytes(w + 2), nibbles_to_bytes(w + 20));
                const __m512i qy = join256(load256(y[b].qs), load256(y[b + 1].qs));
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qx, qy);
                acc = _mm512_fmadd_ps(join256_ps(d0 * y[b].d, d1 * y[b + 1].d), _mm512_cvtepi32_ps(dot), acc);
                offset += d0 * y[b].s + d1 * y[b + 1].s;
            }
            float s = hsum512_ps(acc) - 8.0f * offset;
            if (b < nb) s += dot_q4_0_vnni(w, y + b, 32);
            return s;
        }

        /*
            Q4_K and Q5_K: two sub-blocks of 32 weights per vector, the one of the low nibbles
            in the low half and the one of the high nibbles in the high half, which is the
            order of their 64 activations.
        */
        float dot_q4_K_avx512(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            __m512 acc = _mm512_setzero_ps();
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
                const __m256i q8sums = load256(y[b].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                const __m512i scales = _mm512_castsi256_si512(mins_and_scales);
                const uint8_t* q4 = w + 16;
                const int8_t* q8 = y[b].qs;
                __m512i sumi = _mm512_setzero_si512();
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i bits = load256(q4 + 32 * j);
                    const __m512i q = join256(_mm256_and_si256(bits, m4),
                        _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4));
                    const __m512i a = _mm512_loadu_si512(q8 + 64 * j);
                    const __m512i sc = _mm512_permutexvar_epi16(
                        halves_epi16(static_cast<short>(2 * j), static_cast<short>(2 * j + 1)), scales);
                    sumi = _mm512_dpwssd_epi32(sumi, sc, _mm512_maddubs_epi16(q, a));
                }
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }
            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            return hsum512_ps(acc) + _mm_cvtss_f32(acc_m);
        }

        float dot_q5_K_avx512(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m256i m4 = _mm256_set1_epi8(0x0F);
            const __m512i mone = _mm512_set1_epi8(1);
            __m512 acc = _mm512_setzero_ps();
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
                const __m256i q8sums = load256(y[b].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                   _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                summs += dmin * static_cast<float>(hsum_epi32_128(prod));

                const __m512i scales = _mm512_castsi256_si512(mins_and_scales);
                const __m256i hbits = load256(w + 16);
                const __m512i h = join256(hbits, hbits);
                const uint8_t* q5 = w + 48;
                const int8_t* q8 = y[b].qs;
                __m512i sumi = _mm512_setzero_si512();
                for (int j = 0; j < 4; ++j)
                {
                    const __m256i bits = load256(q5 + 32 * j);
                    const __m512i pair = halves_epi16(static_cast<short>(2 * j), static_cast<short>(2 * j + 1));
                    // Bit 2j of the bytes of hbits for the low half, bit 2j+1 for the high half.
                    const __m512i high = _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(h, pair), mone), 4);
                    const __m512i q = _mm512_or_si512(join256(_mm256_and_si256(bits, m4),
                        _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4)), high);
                    const __m512i a = _mm512_loadu_si512(q8 + 64 * j);
                    sumi = _mm512_dpwssd_epi32(sumi, _mm512_permutexvar_epi16(pair, scales),
                        _mm512_maddubs_epi16(q, a));
                }
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }
            return hsum512_ps(acc) + summs;
        }

        /*
            Q6_K on 64 weights per vector, in the order of their activations: the low nibbles
            of 64 bytes, then their high nibbles, with the two bits of qh that each takes.
            The offset of 32 comes from the sums of the activations by 16, as in the 256-bit
            kernel.
        */
        float dot_q6_K_avx512(const uint8_t* w, const block_q8_K* y, long n)
        {
            const __m512i m4 = _mm512_set1_epi8(0x0F);
            const __m512i m2 = _mm512_set1_epi8(3);
            __m256 acc = _mm256_setzero_ps();
            for (long b = 0; b < n / 256; ++b, w += 210)
            {
                const float d = y[b].d * h2f(w + 208);
                const uint8_t* q4 = w;
                const uint8_t* qh = w + 128;
                const int8_t* q8 = y[b].qs;
                const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 192)));
                const __m512i scales = _mm512_castsi256_si512(sc16);
                const __m256i offset = _mm256_madd_epi16(sc16, load256(y[b].bsums));
                __m512i sumi = _mm512_setzero_si512();
                for (int j = 0; j < 2; ++j)
                {
                    const __m512i bits = _mm512_loadu_si512(q4 + 64 * j);
                    const __m256i h256 = load256(qh + 32 * j);
                    const __m512i h = join256(h256, h256);
                    const __m512i qa = _mm512_or_si512(_mm512_and_si512(bits, m4), _mm512_slli_epi16(
                        _mm512_and_si512(_mm512_srlv_epi16(h, halves_epi16(0, 2)), m2), 4));
                    const __m512i qb = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits, 4), m4),
                        _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(h, halves_epi16(4, 6)), m2), 4));
                    const __m512i aa = _mm512_loadu_si512(q8 + 128 * j);
                    const __m512i ab = _mm512_loadu_si512(q8 + 128 * j + 64);
                    sumi = _mm512_dpwssd_epi32(sumi,
                        _mm512_permutexvar_epi16(quarters_epi16(static_cast<short>(8 * j)), scales),
                        _mm512_maddubs_epi16(qa, aa));
                    sumi = _mm512_dpwssd_epi32(sumi,
                        _mm512_permutexvar_epi16(quarters_epi16(static_cast<short>(8 * j + 4)), scales),
                        _mm512_maddubs_epi16(qb, ab));
                }
                __m256i s = _mm256_add_epi32(_mm512_castsi512_si256(sumi), _mm512_extracti64x4_epi64(sumi, 1));
                s = _mm256_sub_epi32(s, _mm256_slli_epi32(offset, 5));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s), acc);
            }
            return hsum_ps(acc);
        }

        // The first lanes of a vector of 16 floats, left of them or all 16.
        inline __mmask16 tail_mask(long left)
        {
            return left >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << left) - 1u);
        }

        float dot_f32_avx512(const float* a, const float* b, long n)
        {
            __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
            long i = 0;
            for (; i + 64 <= n; i += 64)
            {
                s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
                s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
                s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
                s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
            }
            for (; i < n; i += 16)
            {
                const __mmask16 m = tail_mask(n - i);
                s0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), s0);
            }
            return hsum512_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
        }

        float dot_f16_avx512(const uint8_t* w, const float* x, long n)
        {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(w);
            __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
            long i = 0;
            for (; i + 32 <= n; i += 32)
            {
                s0 = _mm512_fmadd_ps(_mm512_cvtph_ps(load256(h + i)), _mm512_loadu_ps(x + i), s0);
                s1 = _mm512_fmadd_ps(_mm512_cvtph_ps(load256(h + i + 16)), _mm512_loadu_ps(x + i + 16), s1);
            }
            for (; i + 16 <= n; i += 16)
                s0 = _mm512_fmadd_ps(_mm512_cvtph_ps(load256(h + i)), _mm512_loadu_ps(x + i), s0);
            float s = hsum512_ps(_mm512_add_ps(s0, s1));
            for (; i < n; ++i) s += h2f(h[i]) * x[i];
            return s;
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
#if ALBIREO_QK_AVX512
        switch (t)
        {
        case ggml_type::Q4_0: return dot_q4_0_avx512(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q8_0: return dot_q8_0_avx512(w, static_cast<const block_q8_0x*>(act), n);
        case ggml_type::Q4_K: return dot_q4_K_avx512(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q5_K: return dot_q5_K_avx512(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::Q6_K: return dot_q6_K_avx512(w, static_cast<const block_q8_K*>(act), n);
        case ggml_type::F32:
            return dot_f32_avx512(reinterpret_cast<const float*>(w), static_cast<const float*>(act), n);
        case ggml_type::F16: return dot_f16_avx512(w, static_cast<const float*>(act), n);
        default: break;
        }
#elif ALBIREO_QK_VNNI
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
        case ggml_type::Q6_K: fn = dlib::gguf_dq_q6_K; qk = 256; bytes = 210; break;
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

#if ALBIREO_QK_AVX2 && !ALBIREO_QK_AVX512
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

#if ALBIREO_QK_AVX512
        /*
            The products of 4 tokens and 4 rows on vectors of 16 floats: 16 accumulators,
            4 vectors of X and one of W, among the 32 registers of AVX-512.  The last columns,
            fewer than 16, are read through a mask.
        */
        void gemm_4x4_avx512(const float* X, long xs, const float* W, long K, float* Y, long ys, bool accumulate)
        {
            const float* x0 = X;
            const float* x1 = X + xs;
            const float* x2 = X + 2 * xs;
            const float* x3 = X + 3 * xs;
            const float* w0 = W;
            const float* w1 = W + K;
            const float* w2 = W + 2 * K;
            const float* w3 = W + 3 * K;
            const __m512 zero = _mm512_setzero_ps();
            __m512 a00 = zero, a01 = zero, a02 = zero, a03 = zero;
            __m512 a10 = zero, a11 = zero, a12 = zero, a13 = zero;
            __m512 a20 = zero, a21 = zero, a22 = zero, a23 = zero;
            __m512 a30 = zero, a31 = zero, a32 = zero, a33 = zero;
            for (long k = 0; k < K; k += 16)
            {
                const __mmask16 m = tail_mask(K - k);
                const __m512 v0 = _mm512_maskz_loadu_ps(m, x0 + k);
                const __m512 v1 = _mm512_maskz_loadu_ps(m, x1 + k);
                const __m512 v2 = _mm512_maskz_loadu_ps(m, x2 + k);
                const __m512 v3 = _mm512_maskz_loadu_ps(m, x3 + k);
                __m512 w = _mm512_maskz_loadu_ps(m, w0 + k);
                a00 = _mm512_fmadd_ps(v0, w, a00); a10 = _mm512_fmadd_ps(v1, w, a10);
                a20 = _mm512_fmadd_ps(v2, w, a20); a30 = _mm512_fmadd_ps(v3, w, a30);
                w = _mm512_maskz_loadu_ps(m, w1 + k);
                a01 = _mm512_fmadd_ps(v0, w, a01); a11 = _mm512_fmadd_ps(v1, w, a11);
                a21 = _mm512_fmadd_ps(v2, w, a21); a31 = _mm512_fmadd_ps(v3, w, a31);
                w = _mm512_maskz_loadu_ps(m, w2 + k);
                a02 = _mm512_fmadd_ps(v0, w, a02); a12 = _mm512_fmadd_ps(v1, w, a12);
                a22 = _mm512_fmadd_ps(v2, w, a22); a32 = _mm512_fmadd_ps(v3, w, a32);
                w = _mm512_maskz_loadu_ps(m, w3 + k);
                a03 = _mm512_fmadd_ps(v0, w, a03); a13 = _mm512_fmadd_ps(v1, w, a13);
                a23 = _mm512_fmadd_ps(v2, w, a23); a33 = _mm512_fmadd_ps(v3, w, a33);
            }
            store_result(Y, hsum512_ps(a00), accumulate);
            store_result(Y + 1, hsum512_ps(a01), accumulate);
            store_result(Y + 2, hsum512_ps(a02), accumulate);
            store_result(Y + 3, hsum512_ps(a03), accumulate);
            store_result(Y + ys, hsum512_ps(a10), accumulate);
            store_result(Y + ys + 1, hsum512_ps(a11), accumulate);
            store_result(Y + ys + 2, hsum512_ps(a12), accumulate);
            store_result(Y + ys + 3, hsum512_ps(a13), accumulate);
            store_result(Y + 2 * ys, hsum512_ps(a20), accumulate);
            store_result(Y + 2 * ys + 1, hsum512_ps(a21), accumulate);
            store_result(Y + 2 * ys + 2, hsum512_ps(a22), accumulate);
            store_result(Y + 2 * ys + 3, hsum512_ps(a23), accumulate);
            store_result(Y + 3 * ys, hsum512_ps(a30), accumulate);
            store_result(Y + 3 * ys + 1, hsum512_ps(a31), accumulate);
            store_result(Y + 3 * ys + 2, hsum512_ps(a32), accumulate);
            store_result(Y + 3 * ys + 3, hsum512_ps(a33), accumulate);
        }

        // One token against 4 rows, for the tokens left over by gemm_4x4_avx512.
        void gemm_1x4_avx512(const float* X, const float* W, long K, float* Y, bool accumulate)
        {
            const __m512 zero = _mm512_setzero_ps();
            __m512 a0 = zero, a1 = zero, a2 = zero, a3 = zero;
            for (long k = 0; k < K; k += 16)
            {
                const __mmask16 m = tail_mask(K - k);
                const __m512 v = _mm512_maskz_loadu_ps(m, X + k);
                a0 = _mm512_fmadd_ps(v, _mm512_maskz_loadu_ps(m, W + k), a0);
                a1 = _mm512_fmadd_ps(v, _mm512_maskz_loadu_ps(m, W + K + k), a1);
                a2 = _mm512_fmadd_ps(v, _mm512_maskz_loadu_ps(m, W + 2 * K + k), a2);
                a3 = _mm512_fmadd_ps(v, _mm512_maskz_loadu_ps(m, W + 3 * K + k), a3);
            }
            store_result(Y, hsum512_ps(a0), accumulate);
            store_result(Y + 1, hsum512_ps(a1), accumulate);
            store_result(Y + 2, hsum512_ps(a2), accumulate);
            store_result(Y + 3, hsum512_ps(a3), accumulate);
        }
#endif

        // The products of n_rows contiguous float rows of W with n_tokens rows of X,
        // written to the first n_rows columns of Y.
        void gemm_rows(const float* W, long n_rows, long K, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, bool accumulate)
        {
            long t = 0;
            long r4 = 0;
#if ALBIREO_QK_AVX512
            r4 = n_rows / 4 * 4;
            for (; t + 4 <= n_tokens; t += 4)
                for (long r = 0; r < r4; r += 4)
                    gemm_4x4_avx512(X + t * x_stride, x_stride, W + r * K, K, Y + t * y_stride + r, y_stride,
                        accumulate);
            for (long u = t; u < n_tokens; ++u)
                for (long r = 0; r < r4; r += 4)
                    gemm_1x4_avx512(X + u * x_stride, W + r * K, K, Y + u * y_stride + r, accumulate);
            t = 0;
#elif ALBIREO_QK_AVX2
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
#if ALBIREO_QK_AVX512
        const long packed_rows = 32;    // two vectors of 16 floats
#else
        const long packed_rows = 16;    // two vectors of 8 floats
#endif
        const long packed_tokens = 6;
        const long packed_depth = 256;  // a multiple of the block of every quantization type
        const long packed_tile = 64;    // rows of weights per task
        /*
            Below these passes, expanding and laying out the weights costs more than the
            packed kernel saves (measured on Qwen3 1.7B with two threads: with AVX-512, 1.05
            to 1.3 times faster from 128 tokens and 1.4 to 2 times at 512; with AVX2, even at
            128 tokens and 1.05 to 1.25 times faster at 512).
        */
#if ALBIREO_QK_AVX512
        const long packed_min_tokens = 128;
#else
        const long packed_min_tokens = 256;
#endif

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
#if ALBIREO_QK_AVX512
            const __m512 zero = _mm512_setzero_ps();
            __m512 c00 = zero, c01 = zero, c10 = zero, c11 = zero, c20 = zero, c21 = zero;
            __m512 c30 = zero, c31 = zero, c40 = zero, c41 = zero, c50 = zero, c51 = zero;
            for (long k = 0; k < depth; ++k, w += 32, x += 6)
            {
                const __m512 w0 = _mm512_load_ps(w);
                const __m512 w1 = _mm512_load_ps(w + 16);
                __m512 b = _mm512_set1_ps(x[0]);
                c00 = _mm512_fmadd_ps(w0, b, c00); c01 = _mm512_fmadd_ps(w1, b, c01);
                b = _mm512_set1_ps(x[1]);
                c10 = _mm512_fmadd_ps(w0, b, c10); c11 = _mm512_fmadd_ps(w1, b, c11);
                b = _mm512_set1_ps(x[2]);
                c20 = _mm512_fmadd_ps(w0, b, c20); c21 = _mm512_fmadd_ps(w1, b, c21);
                b = _mm512_set1_ps(x[3]);
                c30 = _mm512_fmadd_ps(w0, b, c30); c31 = _mm512_fmadd_ps(w1, b, c31);
                b = _mm512_set1_ps(x[4]);
                c40 = _mm512_fmadd_ps(w0, b, c40); c41 = _mm512_fmadd_ps(w1, b, c41);
                b = _mm512_set1_ps(x[5]);
                c50 = _mm512_fmadd_ps(w0, b, c50); c51 = _mm512_fmadd_ps(w1, b, c51);
            }
            _mm512_store_ps(c, c00); _mm512_store_ps(c + 16, c01);
            _mm512_store_ps(c + 32, c10); _mm512_store_ps(c + 48, c11);
            _mm512_store_ps(c + 64, c20); _mm512_store_ps(c + 80, c21);
            _mm512_store_ps(c + 96, c30); _mm512_store_ps(c + 112, c31);
            _mm512_store_ps(c + 128, c40); _mm512_store_ps(c + 144, c41);
            _mm512_store_ps(c + 160, c50); _mm512_store_ps(c + 176, c51);
#else
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
#endif
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

    void qk_matmul(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
        float* Y, long y_stride, thread_pool& pool, qk_scratch& scratch, bool accumulate, qk_path path)
    {
        if (!W.valid() || n_tokens <= 0) return;
        if (exact_products.load(std::memory_order_relaxed))
        {
            matmul_exact(W, X, n_tokens, x_stride, Y, y_stride, pool, accumulate);
            return;
        }
        if (path == qk_path::floats || (path == qk_path::automatic && n_tokens >= qk_float_product_tokens))
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
                    for (long t = t0; t < t1; ++t)
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
#if ALBIREO_QK_AVX512
        return dot_f32_avx512(a, b, n);
#elif ALBIREO_QK_AVX2
        return dot_f32_avx2(a, b, n);
#else
        return dot_f32_ref(a, b, n);
#endif
    }

    float qk_dot_f32_f16(const float* a, const uint16_t* b, long n)
    {
#if ALBIREO_QK_AVX512
        return dot_f16_avx512(reinterpret_cast<const uint8_t*>(b), a, n);
#elif ALBIREO_QK_AVX2
        return dot_f16_avx2(reinterpret_cast<const uint8_t*>(b), a, n);
#else
        return dot_f16_ref(reinterpret_cast<const uint8_t*>(b), a, n);
#endif
    }

    void qk_axpy_f16(float alpha, const uint16_t* x, float* y, long n)
    {
        long i = 0;
#if ALBIREO_QK_AVX512
        const __m512 a16 = _mm512_set1_ps(alpha);
        for (; i + 16 <= n; i += 16)
        {
            const __m512 xv = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)));
            _mm512_storeu_ps(y + i, _mm512_fmadd_ps(a16, xv, _mm512_loadu_ps(y + i)));
        }
#endif
#if ALBIREO_QK_AVX2
        const __m256 a = _mm256_set1_ps(alpha);
        for (; i + 8 <= n; i += 8)
        {
            const __m256 xv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(a, xv, _mm256_loadu_ps(y + i)));
        }
#endif
        for (; i < n; ++i) y[i] += alpha * h2f(x[i]);
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
#if ALBIREO_QK_AVX512
        inline __m256 fold512(__m512 a)
        {
            return _mm256_add_ps(_mm512_castps512_ps256(a), _mm512_extractf32x8_ps(a, 1));
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
#if ALBIREO_QK_AVX512
            __m512 w[R][4];
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < 4; ++c) w[r][c] = _mm512_setzero_ps();
            for (; i + 16 <= hd; i += 16)
            {
                __m512 x[R];
                for (int r = 0; r < R; ++r) x[r] = _mm512_loadu_ps(q[r] + i);
                for (int c = 0; c < 4; ++c)
                {
                    const __m512 kv = _mm512_loadu_ps(k + c * hd + i);
                    for (int r = 0; r < R; ++r) w[r][c] = _mm512_fmadd_ps(x[r], kv, w[r][c]);
                }
            }
            if (i < hd)
            {
                const __mmask16 mk = tail_mask(hd - i);
                __m512 x[R];
                for (int r = 0; r < R; ++r) x[r] = _mm512_maskz_loadu_ps(mk, q[r] + i);
                for (int c = 0; c < 4; ++c)
                {
                    const __m512 kv = _mm512_maskz_loadu_ps(mk, k + c * hd + i);
                    for (int r = 0; r < R; ++r) w[r][c] = _mm512_fmadd_ps(x[r], kv, w[r][c]);
                }
            }
            for (int r = 0; r < R; ++r)
                _mm_storeu_ps(s[r], hsum4_ps(fold512(w[r][0]), fold512(w[r][1]), fold512(w[r][2]), fold512(w[r][3])));
#elif ALBIREO_QK_AVX2
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
            The values of g keys weighted for R rows (R being 1 or 2) and added to the rows of
            the output: o[r][i] += sum over j < g of p[r][j] * v[j * hd + i].
        */
        template <int R>
        void add_weighted_rows(const float* const* p, const float* v, long g, long hd, float* const* o)
        {
            long i = 0;
#if ALBIREO_QK_AVX512
            for (; i + 64 <= hd; i += 64)
            {
                __m512 a[R][4];
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < 4; ++c) a[r][c] = _mm512_loadu_ps(o[r] + i + 16 * c);
                for (long j = 0; j < g; ++j)
                {
                    const float* vr = v + j * hd + i;
                    __m512 wr[R];
                    for (int r = 0; r < R; ++r) wr[r] = _mm512_set1_ps(p[r][j]);
                    for (int c = 0; c < 4; ++c)
                    {
                        const __m512 x = _mm512_loadu_ps(vr + 16 * c);
                        for (int r = 0; r < R; ++r) a[r][c] = _mm512_fmadd_ps(wr[r], x, a[r][c]);
                    }
                }
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < 4; ++c) _mm512_storeu_ps(o[r] + i + 16 * c, a[r][c]);
            }
            for (; i < hd; i += 16)
            {
                const __mmask16 mk = tail_mask(hd - i);
                __m512 a[R];
                for (int r = 0; r < R; ++r) a[r] = _mm512_maskz_loadu_ps(mk, o[r] + i);
                for (long j = 0; j < g; ++j)
                {
                    const __m512 x = _mm512_maskz_loadu_ps(mk, v + j * hd + i);
                    for (int r = 0; r < R; ++r) a[r] = _mm512_fmadd_ps(_mm512_set1_ps(p[r][j]), x, a[r]);
                }
                for (int r = 0; r < R; ++r) _mm512_mask_storeu_ps(o[r] + i, mk, a[r]);
            }
#elif ALBIREO_QK_AVX2
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

    void qk_attention_scores(const float* Q, long rows, const uint16_t* K, long m, long hd, float* S, long lds,
        float* scratch)
    {
        const long G = attention_group(hd);
        for (long j0 = 0; j0 < m; j0 += G)
        {
            const long g = std::min(G, m - j0);
            qk_half_to_float_row(K + j0 * hd, scratch, g * hd);
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
                    for (int e = 0; e < 2; ++e) S[(r + e) * lds + j0 + j] = qk_dot_f32(q[e], scratch + j * hd, hd);
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
                for (; j < g; ++j) S[r * lds + j0 + j] = qk_dot_f32(q[0], scratch + j * hd, hd);
            }
        }
    }

    void qk_attention_values(const float* P, long lds, long rows, const uint16_t* V, long m, long hd, float* O,
        float* scratch)
    {
        const long G = attention_group(hd);
        for (long j0 = 0; j0 < m; j0 += G)
        {
            const long g = std::min(G, m - j0);
            qk_half_to_float_row(V + j0 * hd, scratch, g * hd);
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
