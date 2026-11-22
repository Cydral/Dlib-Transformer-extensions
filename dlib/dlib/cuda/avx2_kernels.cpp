// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AVX2_KERNELS_CPP_
#define DLIB_AVX2_KERNELS_CPP_

#include "avx2_kernels.h"

/*
    This file is compiled for AVX2, FMA and F16C (dlib/CMakeLists.txt gives it these options
    alone).  A build that compiles the whole library in one translation unit (all/source.cpp)
    compiles its functions only when that unit has the instructions.  MSVC defines no macro for
    FMA, which /arch:AVX2 brings.
*/
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) \
    && defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define DLIB_AVX2_KERNELS_BUILT 1
#else
#define DLIB_AVX2_KERNELS_BUILT 0
#endif

#if DLIB_AVX2_KERNELS_BUILT

#include <immintrin.h>

/*
    The decoders of the GGUF formats are the inline functions of gguf_blocks.h.  They are read
    here under another namespace, so that the copies compiled for AVX2 never replace those of
    the files compiled for every processor.  In a single translation unit the header is already
    read, with the instructions of that unit, and its own namespace serves.
*/
#ifdef DLIB_GGUF_BLOCKS_H_
namespace dlib { namespace avx2 { namespace kernels { namespace blocks = ::dlib::gguf_blocks; } } }
#else
#define gguf_blocks gguf_blocks_avx2
#include "gguf_blocks.h"
#undef gguf_blocks
namespace dlib { namespace avx2 { namespace kernels { namespace blocks = ::dlib::gguf_blocks_avx2; } } }
#endif

namespace dlib
{
    namespace avx2
    {
        namespace kernels
        {
            namespace
            {
                inline __m256 exp256 (__m256 x)
                {
                    /*
                        exp(x) = 2^k exp(r), k the nearest integer to x/ln 2 and r the rest, in
                        [-ln 2/2, ln 2/2], where a polynomial of degree 6 gives exp(r) (the
                        coefficients of the Cephes library).  2^k is applied in two factors,
                        each a normal number: the result then overflows to infinity above
                        88.72 and goes down to zero through the subnormal numbers under -87.3,
                        as the exponential of the math library does.
                    */
                    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-104.0f)), _mm256_set1_ps(88.73f));
                    const __m256 k = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    x = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.693359375f), x);
                    x = _mm256_fnmadd_ps(k, _mm256_set1_ps(-2.12194440e-4f), x);
                    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
                    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
                    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
                    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
                    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
                    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
                    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
                    const __m256i whole = _mm256_cvtps_epi32(k);
                    const __m256i half = _mm256_srai_epi32(whole, 1);
                    const __m256i bias = _mm256_set1_epi32(127);
                    const __m256 first = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(half, bias), 23));
                    const __m256 second = _mm256_castsi256_ps(
                        _mm256_slli_epi32(_mm256_add_epi32(_mm256_sub_epi32(whole, half), bias), 23));
                    return _mm256_mul_ps(_mm256_mul_ps(y, first), second);
                }

                // The values of x that are not numbers, in the place of those of y.
                inline __m256 keep_nan (__m256 y, __m256 x)
                {
                    return _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
                }

                inline __m256 tanh256 (__m256 x)
                {
                    /*
                        Under 0.625, x + x^3 P(x^2), the polynomial of the Cephes library; above,
                        1 - 2/(exp(2|x|) + 1) with the sign of x.
                    */
                    const __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));
                    const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
                    const __m256 z = _mm256_mul_ps(x, x);
                    __m256 p = _mm256_set1_ps(-5.70498872745e-3f);
                    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(2.06390887954e-2f));
                    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-5.37397155531e-2f));
                    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.33314422036e-1f));
                    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(-3.33332819422e-1f));
                    const __m256 below = _mm256_fmadd_ps(_mm256_mul_ps(p, z), x, x);
                    const __m256 one = _mm256_set1_ps(1.0f);
                    const __m256 e = exp256(_mm256_add_ps(a, a));
                    const __m256 above = _mm256_or_ps(
                        _mm256_sub_ps(one, _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, one))), sign);
                    const __m256 small = _mm256_cmp_ps(a, _mm256_set1_ps(0.625f), _CMP_LT_OQ);
                    return keep_nan(_mm256_blendv_ps(above, below, small), x);
                }

                inline __m256 sigmoid256 (__m256 x)
                {
                    const __m256 one = _mm256_set1_ps(1.0f);
                    const __m256 e = exp256(_mm256_xor_ps(x, _mm256_set1_ps(-0.0f)));
                    return keep_nan(_mm256_div_ps(one, _mm256_add_ps(one, e)), x);
                }

                inline __m256 silu256 (__m256 x)
                {
                    return _mm256_mul_ps(x, sigmoid256(x));
                }

                inline __m256 gelu256 (__m256 x)
                {
                    /*
                        x (1 + erf(x/sqrt 2))/2.  With z = |x|/sqrt 2 and t = 1/(1 + p z), the
                        complement of the error function at z is t (a1 + t (a2 + t (a3 + t (a4
                        + t a5)))) exp(-z^2) within 1.5e-7 (Abramowitz and Stegun, 7.1.26), and
                        1 + erf(x/sqrt 2) is that complement under zero, two minus it above.
                    */
                    const __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
                    const __m256 z = _mm256_mul_ps(a, _mm256_set1_ps(0.70710678118654752f));
                    const __m256 one = _mm256_set1_ps(1.0f);
                    const __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), z, one));
                    __m256 p = _mm256_set1_ps(1.061405429f);
                    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027f));
                    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
                    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
                    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
                    const __m256 minus_zz = _mm256_xor_ps(_mm256_mul_ps(z, z), _mm256_set1_ps(-0.0f));
                    const __m256 c = _mm256_mul_ps(_mm256_mul_ps(p, t), exp256(minus_zz));
                    const __m256 above = _mm256_sub_ps(_mm256_set1_ps(2.0f), c);
                    const __m256 both = _mm256_blendv_ps(above, c, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));
                    return keep_nan(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), both), x);
                }

                // f over the n values of s, the last ones through a vector filled with zeros.
                template <typename F>
                inline void map (float* d, const float* s, size_t n, F f)
                {
                    size_t i = 0;
                    for (; i + 8 <= n; i += 8)
                        _mm256_storeu_ps(d + i, f(_mm256_loadu_ps(s + i)));
                    if (i < n)
                    {
                        float tail[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                        for (size_t j = i; j < n; ++j) tail[j - i] = s[j];
                        _mm256_storeu_ps(tail, f(_mm256_loadu_ps(tail)));
                        for (size_t j = i; j < n; ++j) d[j] = tail[j - i];
                    }
                }

                // The half precision number of two bytes, by the conversion of the processor.
                inline float half_value (const uint8_t* p)
                {
                    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(p[0] | (p[1] << 8))));
                }

                // The 32 unsigned bytes of b as four vectors of eight floats.
                inline void bytes_to_floats (__m256i b, __m256 v[4])
                {
                    const __m128i low = _mm256_castsi256_si128(b);
                    const __m128i high = _mm256_extracti128_si256(b, 1);
                    v[0] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low));
                    v[1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low, 8)));
                    v[2] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high));
                    v[3] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high, 8)));
                }

                /*
                    The 32 values of group g of a row, as gguf_blocks::decode_group() gives
                    them.  The formats met most often are decoded eight values at a time; the
                    others go through the decoders of gguf_blocks.h.
                */
                inline void decode (int type, const uint8_t* row, long g, __m256 v[4])
                {
                    switch (type)
                    {
                    case blocks::f32:
                    {
                        const float* p = reinterpret_cast<const float*>(row) + g * 32;
                        for (int i = 0; i < 4; ++i) v[i] = _mm256_loadu_ps(p + 8 * i);
                        break;
                    }
                    case blocks::f16:
                    {
                        const uint8_t* p = row + g * 64;
                        for (int i = 0; i < 4; ++i)
                            v[i] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 * i)));
                        break;
                    }
                    case blocks::bf16:
                    {
                        // The upper half of the bits of a float.
                        const uint8_t* p = row + g * 64;
                        for (int i = 0; i < 4; ++i)
                            v[i] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 * i))), 16));
                        break;
                    }
                    case blocks::q8_0:
                    {
                        const uint8_t* p = row + g * 34;
                        const __m256 d = _mm256_set1_ps(half_value(p));
                        for (int i = 0; i < 4; ++i)
                            v[i] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 2 + 8 * i)))), d);
                        break;
                    }
                    case blocks::q4_K:
                    {
                        const uint8_t* p = row + (g >> 3) * 144;
                        const int j = static_cast<int>(g & 7);
                        int sc, mn;
                        blocks::scale_min_k4(j, p + 4, sc, mn);
                        const __m256 scale = _mm256_set1_ps(half_value(p) * static_cast<float>(sc));
                        const __m256 minimum = _mm256_set1_ps(half_value(p + 2) * static_cast<float>(mn));
                        // The low or the high four bits of the 32 bytes of the pair of groups.
                        __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16 + (j >> 1) * 32));
                        if (j & 1) q = _mm256_srli_epi16(q, 4);
                        q = _mm256_and_si256(q, _mm256_set1_epi8(0x0F));
                        bytes_to_floats(q, v);
                        for (int i = 0; i < 4; ++i) v[i] = _mm256_fmsub_ps(scale, v[i], minimum);
                        break;
                    }
                    case blocks::q6_K:
                    {
                        const uint8_t* p = row + (g >> 3) * 210;
                        const int j = static_cast<int>(g & 7);
                        const int half = j >> 2;
                        const int k = j & 3;
                        const uint8_t* sc = p + 192 + half * 8 + 2 * k;
                        const float d = half_value(p + 208);
                        // Four low bits and two high bits of each value, less 32.
                        const uint8_t* ql = p + half * 64 + (k & 1) * 32;
                        __m256i low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
                        if (k >> 1) low = _mm256_srli_epi16(low, 4);
                        low = _mm256_and_si256(low, _mm256_set1_epi8(0x0F));
                        __m256i high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 128 + half * 32));
                        high = _mm256_and_si256(_mm256_srl_epi16(high, _mm_cvtsi32_si128(2 * k)), _mm256_set1_epi8(3));
                        bytes_to_floats(_mm256_or_si256(low, _mm256_slli_epi16(high, 4)), v);
                        const __m256 offset = _mm256_set1_ps(32.0f);
                        const __m256 first = _mm256_set1_ps(d * static_cast<float>(static_cast<int8_t>(sc[0])));
                        const __m256 second = _mm256_set1_ps(d * static_cast<float>(static_cast<int8_t>(sc[1])));
                        v[0] = _mm256_mul_ps(first, _mm256_sub_ps(v[0], offset));
                        v[1] = _mm256_mul_ps(first, _mm256_sub_ps(v[1], offset));
                        v[2] = _mm256_mul_ps(second, _mm256_sub_ps(v[2], offset));
                        v[3] = _mm256_mul_ps(second, _mm256_sub_ps(v[3], offset));
                        break;
                    }
                    default:
                    {
                        float values[32];
                        blocks::decode_group(type, row, g, values);
                        for (int i = 0; i < 4; ++i) v[i] = _mm256_loadu_ps(values + 8 * i);
                        break;
                    }
                    }
                }

                inline float sum8 (__m256 v)
                {
                    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
                    return _mm_cvtss_f32(s);
                }

                inline float max8 (__m256 v)
                {
                    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
                    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
                    return _mm_cvtss_f32(m);
                }
            }

            bool compiled (
            )
            {
                return true;
            }

            void tanh (float* d, const float* s, size_t n)
            {
                map(d, s, n, [](__m256 v) { return tanh256(v); });
            }

            void sigmoid (float* d, const float* s, size_t n)
            {
                map(d, s, n, [](__m256 v) { return sigmoid256(v); });
            }

            void silu (float* d, const float* s, size_t n)
            {
                map(d, s, n, [](__m256 v) { return silu256(v); });
            }

            void gelu (float* d, const float* s, size_t n)
            {
                map(d, s, n, [](__m256 v) { return gelu256(v); });
            }

            void softmax_rows (float* d, const float* s, size_t rows, size_t n, bool zero_empty_rows)
            {
                if (n == 0) return;
                const float infinity = _mm_cvtss_f32(_mm_castsi128_ps(_mm_set1_epi32(0x7f800000)));
                const float none = _mm_cvtss_f32(_mm_castsi128_ps(_mm_set1_epi32(0x7fc00000)));
                for (size_t r = 0; r < rows; ++r, d += n, s += n)
                {
                    /*
                        The largest value of the row, the values that are not numbers left
                        aside as cpu::softmax() leaves them, and whether the row holds one.
                    */
                    __m256 largest = _mm256_set1_ps(-infinity);
                    __m256 unordered = _mm256_setzero_ps();
                    size_t i = 0;
                    for (; i + 8 <= n; i += 8)
                    {
                        const __m256 v = _mm256_loadu_ps(s + i);
                        largest = _mm256_max_ps(v, largest);
                        unordered = _mm256_or_ps(unordered, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
                    }
                    float top = max8(largest);
                    bool lost = _mm256_movemask_ps(unordered) != 0;
                    for (; i < n; ++i)
                    {
                        if (s[i] != s[i]) lost = true;
                        else if (s[i] > top) top = s[i];
                    }
                    if (top == -infinity && zero_empty_rows)
                    {
                        for (i = 0; i < n; ++i) d[i] = 0.0f;
                        continue;
                    }
                    // The differences with an infinite largest value are not numbers, and neither is their sum.
                    if (lost || top == -infinity || top == infinity)
                    {
                        for (i = 0; i < n; ++i) d[i] = none;
                        continue;
                    }
                    const __m256 shift = _mm256_set1_ps(top);
                    __m256 total = _mm256_setzero_ps();
                    for (i = 0; i + 8 <= n; i += 8)
                    {
                        const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(s + i), shift));
                        _mm256_storeu_ps(d + i, e);
                        total = _mm256_add_ps(total, e);
                    }
                    float sum = sum8(total);
                    if (i < n)
                    {
                        // The last values, the vector filled with the shift itself: its extra values are not added.
                        float tail[8];
                        for (size_t j = 0; j < 8; ++j) tail[j] = i + j < n ? s[i + j] : top;
                        _mm256_storeu_ps(tail, exp256(_mm256_sub_ps(_mm256_loadu_ps(tail), shift)));
                        for (size_t j = i; j < n; ++j)
                        {
                            d[j] = tail[j - i];
                            sum += tail[j - i];
                        }
                    }
                    const __m256 scale = _mm256_set1_ps(1.0f / sum);
                    for (i = 0; i + 8 <= n; i += 8)
                        _mm256_storeu_ps(d + i, _mm256_mul_ps(_mm256_loadu_ps(d + i), scale));
                    for (; i < n; ++i) d[i] *= 1.0f / sum;
                }
            }

            void gguf_matvec_rows (
                float* y,
                const float* x,
                const uint8_t* w,
                int type,
                long rows,
                long cols,
                size_t row_bytes,
                long n,
                long first,
                long last
            )
            {
                // The vectors of a pass: a row is decoded once for all of them.
                const long pass = 64;
                const long groups = cols / blocks::group_size;
                __m256 v[4];
                __m256 acc[pass];
                for (long r = first; r < last; ++r)
                {
                    const uint8_t* wr = w + static_cast<size_t>(r) * row_bytes;
                    for (long t0 = 0; t0 < n; t0 += pass)
                    {
                        const long count = n - t0 < pass ? n - t0 : pass;
                        for (long t = 0; t < count; ++t) acc[t] = _mm256_setzero_ps();
                        for (long g = 0; g < groups; ++g)
                        {
                            decode(type, wr, g, v);
                            const __m256 v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
                            const float* xg = x + static_cast<size_t>(t0) * static_cast<size_t>(cols) + g * 32;
                            for (long t = 0; t < count; ++t, xg += cols)
                            {
                                __m256 a = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xg), acc[t]);
                                __m256 b = _mm256_mul_ps(v1, _mm256_loadu_ps(xg + 8));
                                a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(xg + 16), a);
                                b = _mm256_fmadd_ps(v3, _mm256_loadu_ps(xg + 24), b);
                                acc[t] = _mm256_add_ps(a, b);
                            }
                        }
                        for (long t = 0; t < count; ++t)
                            y[static_cast<size_t>(t0 + t) * static_cast<size_t>(rows) + r] = sum8(acc[t]);
                    }
                }
            }
        }
    }
}

#else // DLIB_AVX2_KERNELS_BUILT

namespace dlib
{
    namespace avx2
    {
        namespace kernels
        {
            bool compiled (
            )
            {
                return false;
            }

            void tanh (float*, const float*, size_t) {}
            void sigmoid (float*, const float*, size_t) {}
            void silu (float*, const float*, size_t) {}
            void gelu (float*, const float*, size_t) {}
            void softmax_rows (float*, const float*, size_t, size_t, bool) {}
            void gguf_matvec_rows (float*, const float*, const uint8_t*, int, long, long, size_t, long, long, long) {}
        }
    }
}

#endif // DLIB_AVX2_KERNELS_BUILT

#endif // DLIB_AVX2_KERNELS_CPP_
