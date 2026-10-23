// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_NPU_DLIB_CPP_
#define DLIB_DNN_NPU_DLIB_CPP_

/*
    Device paths of npu_dlib.h, on the OpenVINO runtime.

    This is the only file of dlib that includes OpenVINO. Its headers need C++17, so the build
    compiles this file alone in that mode; nothing OpenVINO declares crosses into the headers
    the rest of dlib, which stays C++14, includes.

    Organization
        - half precision conversions and scans, with AVX2/F16C kernels chosen at run time;
        - the runtime: device detection, settings, compiled graphs and their cache, the
          background compiler, statistics;
        - the executor, which moves operands between dlib tensors and the device buffers of a
          compiled graph, with the scaling and the guards described in npu_dlib.h;
        - one graph builder and one entry point per operation, grouped by family.

    A graph is built for a signature: the operation and the dimensions of its operands, some
    of them rounded up to a bucket, plus the structural attributes. Scalars that vary from call
    to call, such as eps or the coefficients of an affine transform, are graph inputs rather
    than constants, so that changing them does not recompile anything.
*/

#include "npu_dlib.h"

#ifdef DLIB_USE_NPU

#include "tensor.h"
#include "../geometry/rectangle.h"
#include "../threads.h"

#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DLIB_NPU_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DLIB_NPU_SIMD __attribute__((target("avx2,f16c")))
#else
#define DLIB_NPU_SIMD
#endif
#endif

namespace dlib
{
    namespace npu
    {
        // The block behind a tensor, whose copy mark tells whether its values changed.
        struct storage_access
        {
            static const gpu_data& of (const tensor& t) { return t.data(); }
        };

        namespace
        {

    // ----------------------------------------------------------------------------------------
    //                                  Small utilities
    // ----------------------------------------------------------------------------------------

            long long now_ns (
            )
            {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
            }

            std::string read_env (
                const char* name,
                bool& present
            )
            {
#ifdef _MSC_VER
                char* buf = nullptr;
                size_t len = 0;
                if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
                {
                    std::string value(buf);
                    free(buf);
                    present = true;
                    return value;
                }
                present = false;
                return std::string();
#else
                const char* value = std::getenv(name);
                present = value != nullptr;
                return value ? std::string(value) : std::string();
#endif
            }

#ifdef _WIN32
            std::string to_utf8 (
                const std::filesystem::path& p
            )
            {
                const auto s = p.u8string();
                return std::string(s.begin(), s.end());
            }
#endif

            // An environment variable holding a path, as UTF-8 whatever the code page.
            std::string read_env_path (
                const char* name,
                bool& present
            )
            {
#ifdef _WIN32
                const std::wstring wname(name, name + std::strlen(name));
#ifdef _MSC_VER
                wchar_t* buf = nullptr;
                size_t len = 0;
                if (_wdupenv_s(&buf, &len, wname.c_str()) == 0 && buf != nullptr)
                {
                    const std::string value = to_utf8(std::filesystem::path(buf));
                    free(buf);
                    present = true;
                    return value;
                }
                present = false;
                return std::string();
#else
                const wchar_t* value = _wgetenv(wname.c_str());
                present = value != nullptr;
                return value ? to_utf8(std::filesystem::path(value)) : std::string();
#endif
#else
                return read_env(name, present);
#endif
            }

            std::string lowercase (
                std::string s
            )
            {
                for (auto& c : s)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return s;
            }

            bool means_off (
                const std::string& value
            )
            {
                const std::string v = lowercase(value);
                return v == "0" || v == "off" || v == "false" || v == "no" || v == "never";
            }

            // 2^e as a float, for -126 <= e <= 127.
            inline float exp2i (
                int e
            )
            {
                return std::ldexp(1.0f, e);
            }

            // Smallest e with |x| <= 2^e, for finite x > 0.
            inline int ceil_log2 (
                float x
            )
            {
                int e = 0;
                const float m = std::frexp(x, &e); // x = m * 2^e with 0.5 <= m < 1
                return (m == 0.5f) ? e - 1 : e;
            }

    // ----------------------------------------------------------------------------------------
    //                         Half precision conversions and scans
    // ----------------------------------------------------------------------------------------

            /*
                Software conversions, used when the processor has no F16C instructions and for
                the tails of the vector loops. Both round to nearest even, as F16C does, so the
                two paths produce the same bits.
            */
            inline uint16_t half_from_float (
                float f
            )
            {
                uint32_t u;
                std::memcpy(&u, &f, 4);
                const uint32_t sign = u & 0x80000000u;
                u ^= sign;
                uint16_t h;
                if (u >= 0x47800000u)                   // 2^16 and above, infinities and NaNs
                {
                    h = (u > 0x7f800000u) ? 0x7e00 : 0x7c00;
                }
                else if (u < 0x38800000u)               // below the smallest normal half
                {
                    float magic;
                    const uint32_t magic_bits = 0x3f000000u; // 0.5: aligns the mantissa bits
                    std::memcpy(&magic, &magic_bits, 4);
                    float v;
                    std::memcpy(&v, &u, 4);
                    v += magic;
                    uint32_t vb;
                    std::memcpy(&vb, &v, 4);
                    h = static_cast<uint16_t>(vb - magic_bits);
                }
                else
                {
                    const uint32_t odd = (u >> 13) & 1u;
                    u += 0xc8000fffu + odd;              // rebias the exponent, round to even
                    h = static_cast<uint16_t>(u >> 13);
                }
                return static_cast<uint16_t>(h | (sign >> 16));
            }

            inline float float_from_half (
                uint16_t h
            )
            {
                const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
                uint32_t e = (h >> 10) & 0x1fu;
                uint32_t m = h & 0x3ffu;
                uint32_t bits;
                if (e == 0)
                {
                    if (m == 0)
                    {
                        bits = sign;
                    }
                    else
                    {
                        int shift = -1;
                        do { ++shift; m <<= 1; } while ((m & 0x400u) == 0);
                        bits = sign | (static_cast<uint32_t>(112 - shift) << 23) | ((m & 0x3ffu) << 13);
                    }
                }
                else if (e == 31)
                {
                    bits = sign | 0x7f800000u | (m << 13);
                }
                else
                {
                    bits = sign | ((e + 112u) << 23) | (m << 13);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                return f;
            }

            bool detect_simd (
            )
            {
#if defined(DLIB_NPU_X86)
#if defined(_MSC_VER)
                int r[4];
                __cpuid(r, 1);
                const bool f16c = ((r[2] >> 29) & 1) != 0;
                const bool avx = ((r[2] >> 28) & 1) != 0;
                const bool osxsave = ((r[2] >> 27) & 1) != 0;
                if (!(f16c && avx && osxsave))
                    return false;
                if ((_xgetbv(0) & 6) != 6)
                    return false;
                __cpuidex(r, 7, 0);
                return ((r[1] >> 5) & 1) != 0;
#else
                __builtin_cpu_init();
                return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
#endif
#else
                return false;
#endif
            }

            bool have_simd (
            )
            {
                static const bool value = detect_simd();
                return value;
            }

            struct scan_result
            {
                float maxabs = 0;   // largest magnitude among the finite values
                float maxval = -std::numeric_limits<float>::infinity(); // largest finite value
                bool nan = false;
                bool posinf = false;
                bool neginf = false;

                void merge (const scan_result& o)
                {
                    maxabs = std::max(maxabs, o.maxabs);
                    maxval = std::max(maxval, o.maxval);
                    nan = nan || o.nan;
                    posinf = posinf || o.posinf;
                    neginf = neginf || o.neginf;
                }
            };

            void scan_scalar (
                const float* p,
                size_t n,
                scan_result& r
            )
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const float x = p[i];
                    if (x != x)
                    {
                        r.nan = true;
                    }
                    else if (std::isinf(x))
                    {
                        if (x > 0) r.posinf = true;
                        else r.neginf = true;
                    }
                    else
                    {
                        const float a = std::abs(x);
                        if (a > r.maxabs) r.maxabs = a;
                        if (x > r.maxval) r.maxval = x;
                    }
                }
            }

            // dst = half((src - offset) * scale), raised to the lowest finite half when clamp is set.
            void pack_scalar (
                const float* src,
                uint16_t* dst,
                size_t n,
                float scale,
                float offset,
                bool clamp
            )
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const float v = (src[i] - offset) * scale;
                    dst[i] = half_from_float(clamp ? std::max(v, -65504.0f) : v);
                }
            }

            // dst = half((src - offsets) * scale), with one offset per value.
            void pack_columns_scalar (
                const float* src,
                const float* offsets,
                uint16_t* dst,
                size_t n,
                float scale
            )
            {
                for (size_t i = 0; i < n; ++i)
                    dst[i] = half_from_float((src[i] - offsets[i]) * scale);
            }

            // mode 0: dst = s*x, mode 1: dst += s*x, mode 2: dst = beta*dst + s*x
            void unpack_scalar (
                const uint16_t* src,
                float* dst,
                size_t n,
                float s,
                float beta,
                int mode
            )
            {
                if (mode == 0)
                    for (size_t i = 0; i < n; ++i) dst[i] = s * float_from_half(src[i]);
                else if (mode == 1)
                    for (size_t i = 0; i < n; ++i) dst[i] += s * float_from_half(src[i]);
                else
                    for (size_t i = 0; i < n; ++i) dst[i] = beta * dst[i] + s * float_from_half(src[i]);
            }

            size_t count_nonfinite_scalar (
                const uint16_t* p,
                size_t n
            )
            {
                size_t count = 0;
                for (size_t i = 0; i < n; ++i)
                    count += ((p[i] & 0x7c00u) == 0x7c00u) ? 1 : 0;
                return count;
            }

#if defined(DLIB_NPU_X86)

            DLIB_NPU_SIMD void scan_simd (
                const float* p,
                size_t n,
                scan_result& r
            )
            {
                const __m256 sign = _mm256_set1_ps(-0.0f);
                const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
                __m256 vmaxabs = _mm256_setzero_ps();
                __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                __m256 vnan = _mm256_setzero_ps();
                __m256 vinf = _mm256_setzero_ps();
                size_t i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    const __m256 x = _mm256_loadu_ps(p + i);
                    const __m256 a = _mm256_andnot_ps(sign, x);
                    const __m256 isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
                    const __m256 isinf = _mm256_cmp_ps(a, inf, _CMP_EQ_OQ);
                    const __m256 finite = _mm256_cmp_ps(a, inf, _CMP_LT_OQ);
                    vnan = _mm256_or_ps(vnan, isnan);
                    vinf = _mm256_or_ps(vinf, isinf);
                    vmaxabs = _mm256_max_ps(vmaxabs, _mm256_and_ps(finite, a));
                    vmax = _mm256_max_ps(vmax, _mm256_blendv_ps(vmax, x, finite));
                }
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, vmaxabs);
                for (float v : lanes) r.maxabs = std::max(r.maxabs, v);
                _mm256_store_ps(lanes, vmax);
                for (float v : lanes) r.maxval = std::max(r.maxval, v);
                if (_mm256_movemask_ps(vnan) != 0)
                    r.nan = true;
                if (_mm256_movemask_ps(vinf) != 0)
                {
                    // Rare: find which signs the infinities carry.
                    scan_result tail;
                    scan_scalar(p, i, tail);
                    r.posinf = r.posinf || tail.posinf;
                    r.neginf = r.neginf || tail.neginf;
                }
                scan_scalar(p + i, n - i, r);
            }

            DLIB_NPU_SIMD void pack_simd (
                const float* src,
                uint16_t* dst,
                size_t n,
                float scale,
                float offset,
                bool clamp
            )
            {
                const __m256 vs = _mm256_set1_ps(scale);
                const __m256 vo = _mm256_set1_ps(offset);
                const __m256 vl = _mm256_set1_ps(-65504.0f);
                size_t i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    __m256 x = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src + i), vo), vs);
                    if (clamp)
                        x = _mm256_max_ps(x, vl);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                        _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
                }
                pack_scalar(src + i, dst + i, n - i, scale, offset, clamp);
            }

            DLIB_NPU_SIMD void pack_columns_simd (
                const float* src,
                const float* offsets,
                uint16_t* dst,
                size_t n,
                float scale
            )
            {
                const __m256 vs = _mm256_set1_ps(scale);
                size_t i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    const __m256 x = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src + i),
                        _mm256_loadu_ps(offsets + i)), vs);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                        _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
                }
                pack_columns_scalar(src + i, offsets + i, dst + i, n - i, scale);
            }

            DLIB_NPU_SIMD void unpack_simd (
                const uint16_t* src,
                float* dst,
                size_t n,
                float s,
                float beta,
                int mode
            )
            {
                const __m256 vs = _mm256_set1_ps(s);
                const __m256 vb = _mm256_set1_ps(beta);
                size_t i = 0;
                if (mode == 0)
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
                        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, vs));
                    }
                }
                else if (mode == 1)
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
                        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_mul_ps(x, vs)));
                    }
                }
                else
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        const __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
                        const __m256 d = _mm256_mul_ps(_mm256_loadu_ps(dst + i), vb);
                        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, _mm256_mul_ps(x, vs)));
                    }
                }
                unpack_scalar(src + i, dst + i, n - i, s, beta, mode);
            }

            DLIB_NPU_SIMD size_t count_nonfinite_simd (
                const uint16_t* p,
                size_t n
            )
            {
                const __m256i mask = _mm256_set1_epi16(0x7c00);
                size_t count = 0;
                size_t i = 0;
                for (; i + 16 <= n; i += 16)
                {
                    const __m256i v =
                        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)), mask);
                    const unsigned bits = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, mask)));
                    if (bits != 0)
                    {
                        // Two mask bits per 16-bit lane.
                        unsigned b = bits;
                        while (b) { count += 1; b &= b - 1; }
                    }
                }
                return count / 2 + count_nonfinite_scalar(p + i, n - i);
            }

#endif // DLIB_NPU_X86

            void scan (
                const float* p,
                size_t n,
                scan_result& r
            )
            {
#if defined(DLIB_NPU_X86)
                if (have_simd()) { scan_simd(p, n, r); return; }
#endif
                scan_scalar(p, n, r);
            }

            void pack_half (
                const float* src,
                uint16_t* dst,
                size_t n,
                float scale,
                float offset,
                bool clamp
            )
            {
#if defined(DLIB_NPU_X86)
                if (have_simd()) { pack_simd(src, dst, n, scale, offset, clamp); return; }
#endif
                pack_scalar(src, dst, n, scale, offset, clamp);
            }

            void pack_half_columns (
                const float* src,
                const float* offsets,
                uint16_t* dst,
                size_t n,
                float scale
            )
            {
#if defined(DLIB_NPU_X86)
                if (have_simd()) { pack_columns_simd(src, offsets, dst, n, scale); return; }
#endif
                pack_columns_scalar(src, offsets, dst, n, scale);
            }

            void unpack_half (
                const uint16_t* src,
                float* dst,
                size_t n,
                float s,
                float beta,
                int mode
            )
            {
#if defined(DLIB_NPU_X86)
                if (have_simd()) { unpack_simd(src, dst, n, s, beta, mode); return; }
#endif
                unpack_scalar(src, dst, n, s, beta, mode);
            }

            size_t count_nonfinite_half (
                const uint16_t* p,
                size_t n
            )
            {
#if defined(DLIB_NPU_X86)
                if (have_simd()) return count_nonfinite_simd(p, n);
#endif
                return count_nonfinite_scalar(p, n);
            }

            void pack_float (
                const float* src,
                float* dst,
                size_t n,
                float scale,
                float offset
            )
            {
                if (scale == 1 && offset == 0)
                    std::memcpy(dst, src, n * sizeof(float));
                else
                    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] - offset) * scale;
            }

            void pack_float_columns (
                const float* src,
                const float* offsets,
                float* dst,
                size_t n,
                float scale
            )
            {
                for (size_t i = 0; i < n; ++i) dst[i] = (src[i] - offsets[i]) * scale;
            }

            void unpack_float (
                const float* src,
                float* dst,
                size_t n,
                float s,
                float beta,
                int mode
            )
            {
                if (mode == 0)
                {
                    if (s == 1) std::memcpy(dst, src, n * sizeof(float));
                    else for (size_t i = 0; i < n; ++i) dst[i] = s * src[i];
                }
                else if (mode == 1)
                {
                    for (size_t i = 0; i < n; ++i) dst[i] += s * src[i];
                }
                else
                {
                    for (size_t i = 0; i < n; ++i) dst[i] = beta * dst[i] + s * src[i];
                }
            }

            size_t count_nonfinite_float (
                const float* p,
                size_t n
            )
            {
                size_t count = 0;
                for (size_t i = 0; i < n; ++i)
                    count += std::isfinite(p[i]) ? 0 : 1;
                return count;
            }

            /*
                Runs f(begin, end) over [0, n) in blocks of grain items, which callers size to a
                few tens of thousands of values, split across dlib's thread pool once there are
                enough blocks to repay the dispatch.
            */
            template <typename F>
            void for_blocks (
                size_t n,
                size_t grain,
                F&& f
            )
            {
                grain = std::max<size_t>(grain, 1);
                if (n < 4 * grain)
                {
                    f(size_t(0), n);
                    return;
                }
                const size_t blocks = (n + grain - 1) / grain;
                dlib::parallel_for(0, static_cast<long>(blocks), [&](long b)
                {
                    const size_t begin = static_cast<size_t>(b) * grain;
                    f(begin, std::min(n, begin + grain));
                });
            }

    // ----------------------------------------------------------------------------------------
    //                                The operation table
    // ----------------------------------------------------------------------------------------

            enum class op_class
            {
                compute,  // arithmetic, routed by measurement
                movement, // copies, permutations and gathers: exact, hence on a device that
                          // computes in single precision only, and only when routing is forced
                exact     // must not round: persistent state, comparisons against a threshold,
                          // identities documented as exact, matrix inversion. On a device that
                          // computes in single precision only
            };

            /*
                One line per operation: its name, which DLIB_NPU_OPS and print_status() use, its
                class, and the size from which automatic routing considers it, in elements, or
                in multiply-accumulates for the products and the convolutions.
            */
#define DLIB_NPU_OPERATIONS(X) \
            X(inverse_norms,                   compute,  1 << 16) \
            X(inverse_sqrt,                    compute,  1 << 16) \
            X(divide_by_sqrt,                  exact,    1 << 16) \
            X(dot_prods,                       compute,  1 << 16) \
            X(scale_columns,                   compute,  1 << 16) \
            X(scale_rows,                      compute,  1 << 16) \
            X(scale_rows2,                     compute,  1 << 16) \
            X(exp,                             compute,  1 << 16) \
            X(log,                             compute,  1 << 16) \
            X(log10,                           compute,  1 << 16) \
            X(gemm,                            compute,  1 << 21) \
            X(multiply,                        compute,  1 << 16) \
            X(scale_channels,                  compute,  1 << 16) \
            X(multiply_conv,                   compute,  1 << 16) \
            X(multiply_zero_padded,            compute,  1 << 16) \
            X(affine_transform,                compute,  1 << 16) \
            X(affine_transform_conv,           compute,  1 << 16) \
            X(adam,                            exact,    1 << 16) \
            X(adam_atan2,                      exact,    1 << 16) \
            X(batch_normalize_inference,       compute,  1 << 16) \
            X(batch_normalize,                 compute,  1 << 16) \
            X(batch_normalize_gradient,        compute,  1 << 16) \
            X(batch_normalize_conv_inference,  compute,  1 << 16) \
            X(batch_normalize_conv,            compute,  1 << 16) \
            X(batch_normalize_conv_gradient,   compute,  1 << 16) \
            X(layer_normalize,                 compute,  1 << 16) \
            X(layer_normalize_gradient,        compute,  1 << 16) \
            X(rms_normalize,                   compute,  1 << 16) \
            X(rms_normalize_gradient,          compute,  1 << 16) \
            X(threshold,                       exact,    1 << 16) \
            X(dot,                             compute,  1 << 16) \
            X(add,                             compute,  1 << 16) \
            X(assign_bias_gradient,            compute,  1 << 16) \
            X(assign_conv_bias_gradient,       compute,  1 << 16) \
            X(softmax,                         compute,  1 << 16) \
            X(softmax_gradient,                compute,  1 << 16) \
            X(softmax_all,                     compute,  1 << 16) \
            X(softmax_all_gradient,            compute,  1 << 16) \
            X(sigmoid,                         compute,  1 << 16) \
            X(sigmoid_gradient,                compute,  1 << 16) \
            X(mish,                            compute,  1 << 16) \
            X(mish_gradient,                   compute,  1 << 16) \
            X(relu,                            compute,  1 << 16) \
            X(relu_gradient,                   compute,  1 << 16) \
            X(prelu,                           compute,  1 << 16) \
            X(prelu_gradient,                  compute,  1 << 16) \
            X(leaky_relu,                      compute,  1 << 16) \
            X(leaky_relu_gradient,             compute,  1 << 16) \
            X(tanh,                            compute,  1 << 16) \
            X(tanh_gradient,                   compute,  1 << 16) \
            X(clipped_relu,                    compute,  1 << 16) \
            X(clipped_relu_gradient,           compute,  1 << 16) \
            X(elu,                             compute,  1 << 16) \
            X(elu_gradient,                    compute,  1 << 16) \
            X(gelu,                            compute,  1 << 16) \
            X(gelu_gradient,                   compute,  1 << 16) \
            X(smelu,                           compute,  1 << 16) \
            X(smelu_gradient,                  compute,  1 << 16) \
            X(silu,                            compute,  1 << 16) \
            X(silu_gradient,                   compute,  1 << 16) \
            X(resize_bilinear,                 compute,  1 << 16) \
            X(resize_bilinear_gradient,        compute,  1 << 16) \
            X(reorg,                           movement, 1 << 16) \
            X(reorg_gradient,                  movement, 1 << 16) \
            X(copy_tensor,                     movement, 1 << 16) \
            X(inv,                             exact,    1 << 21) \
            X(transpose,                       movement, 1 << 16) \
            X(embeddings,                      movement, 1 << 16) \
            X(embeddings_gradient,             exact,    1 << 16) \
            X(puzzle_embeddings,               movement, 1 << 16) \
            X(puzzle_embeddings_gradient,      exact,    1 << 16) \
            X(repeat_channels,                 movement, 1 << 16) \
            X(accumulate_repeated_channels,    movement, 1 << 16) \
            X(compute_halt_logits,             compute,  1 << 16) \
            X(compute_halt_gradient,           compute,  1 << 16) \
            X(accumulate_weighted_state,       compute,  1 << 16) \
            X(apply_act_depth_scaling,         compute,  1 << 16) \
            X(broadcast_row,                   movement, 1 << 16) \
            X(act_mark_active,                 movement, 1 << 16) \
            X(act_update_halting,              exact,    1 << 16) \
            X(act_gather_final_state,          movement, 1 << 16) \
            X(act_accumulate_output_grad,      compute,  1 << 16) \
            X(rotary_embedding,                compute,  1 << 16) \
            X(split_heads,                     movement, 1 << 16) \
            X(merge_heads,                     movement, 1 << 16) \
            X(conv,                            compute,  1 << 21) \
            X(conv_gradient_for_data,          compute,  1 << 21) \
            X(conv_gradient_for_filters,       compute,  1 << 21) \
            X(pooling,                         compute,  1 << 16) \
            X(pooling_gradient,                compute,  1 << 16) \
            X(cross_entropy,                   compute,  1 << 16) \
            X(mean_squared,                    compute,  1 << 16)

            enum op_id : int
            {
#define DLIB_NPU_ENUM(name, cls, work) OP_##name,
                DLIB_NPU_OPERATIONS(DLIB_NPU_ENUM)
#undef DLIB_NPU_ENUM
                OP_COUNT
            };

            struct op_info
            {
                const char* name;
                op_class cls;
                double min_work;
            };

            const op_info op_table[OP_COUNT] = {
#define DLIB_NPU_INFO(name, cls, work) { #name, op_class::cls, static_cast<double>(work) },
                DLIB_NPU_OPERATIONS(DLIB_NPU_INFO)
#undef DLIB_NPU_INFO
            };

    // ----------------------------------------------------------------------------------------
    //                                     Graphs
    // ----------------------------------------------------------------------------------------

            /*
                The parameters of a graph: the dimensions and attributes that determine it.
                Fixed capacity, so that building a signature on every call allocates nothing.
            */
            struct params
            {
                int64_t v[32];
                int n = 0;

                params& operator<< (int64_t x)
                {
                    DLIB_CASSERT(n < 32);
                    v[n++] = x;
                    return *this;
                }

                int64_t operator[] (int i) const { return v[i]; }

                bool operator== (const params& o) const
                {
                    return n == o.n && std::equal(v, v + n, o.v);
                }
            };

            using node = ov::Output<ov::Node>;

            /*
                A thin layer over the OpenVINO operation set, enough to write each graph in a
                few lines. Floating point values use the exchange type of the runtime, f16 or
                f32; integer inputs are i32.
            */
            class graph
            {
            public:
                explicit graph (ov::element::Type t) : et(t) {}

                const ov::element::Type et;
                ov::ParameterVector parameters;
                ov::ResultVector results;

                node input (const std::vector<int64_t>& dims)
                {
                    auto p = std::make_shared<ov::op::v0::Parameter>(et, to_shape(dims));
                    parameters.push_back(p);
                    return p;
                }

                node input_i32 (const std::vector<int64_t>& dims)
                {
                    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, to_shape(dims));
                    parameters.push_back(p);
                    return p;
                }

                // A scalar fed at every call, as a one element input.
                node scalar ()
                {
                    return input({1});
                }

                void output (const node& x)
                {
                    node y = x;
                    if (y.get_element_type() != et)
                        y = std::make_shared<ov::op::v0::Convert>(y, et);
                    results.push_back(std::make_shared<ov::op::v0::Result>(y));
                }

                node c (double value) const
                {
                    return ov::op::v0::Constant::create(et, ov::Shape{}, {value});
                }

                node constant (const std::vector<int64_t>& dims, const std::vector<float>& values) const
                {
                    return ov::op::v0::Constant::create(et, to_shape(dims), values);
                }

                static node ints (const std::vector<int64_t>& values)
                {
                    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
                }

                static node add (const node& a, const node& b) { return std::make_shared<ov::op::v1::Add>(a, b); }
                static node sub (const node& a, const node& b) { return std::make_shared<ov::op::v1::Subtract>(a, b); }
                static node mul (const node& a, const node& b) { return std::make_shared<ov::op::v1::Multiply>(a, b); }
                static node div (const node& a, const node& b) { return std::make_shared<ov::op::v1::Divide>(a, b); }
                static node max (const node& a, const node& b) { return std::make_shared<ov::op::v1::Maximum>(a, b); }
                static node min (const node& a, const node& b) { return std::make_shared<ov::op::v1::Minimum>(a, b); }
                static node exp (const node& a) { return std::make_shared<ov::op::v0::Exp>(a); }
                static node log (const node& a) { return std::make_shared<ov::op::v0::Log>(a); }
                static node sqrt (const node& a) { return std::make_shared<ov::op::v0::Sqrt>(a); }
                static node tanh (const node& a) { return std::make_shared<ov::op::v0::Tanh>(a); }
                static node sigmoid (const node& a) { return std::make_shared<ov::op::v0::Sigmoid>(a); }
                static node erf (const node& a) { return std::make_shared<ov::op::v0::Erf>(a); }
                static node abs (const node& a) { return std::make_shared<ov::op::v0::Abs>(a); }
                static node atan (const node& a) { return std::make_shared<ov::op::v0::Atan>(a); }
                static node sign (const node& a) { return std::make_shared<ov::op::v0::Sign>(a); }
                static node relu (const node& a) { return std::make_shared<ov::op::v0::Relu>(a); }
                static node greater (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::Greater>(a, b); }
                static node greater_equal (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::GreaterEqual>(a, b); }
                static node less (const node& a, const node& b) { return std::make_shared<ov::op::v1::Less>(a, b); }
                static node less_equal (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::LessEqual>(a, b); }
                static node equal (const node& a, const node& b) { return std::make_shared<ov::op::v1::Equal>(a, b); }
                static node not_equal (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::NotEqual>(a, b); }
                static node logical_and (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::LogicalAnd>(a, b); }
                static node logical_or (const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::LogicalOr>(a, b); }
                static node select (const node& c, const node& a, const node& b)
                    { return std::make_shared<ov::op::v1::Select>(c, a, b); }

                node to_value (const node& boolean) const
                {
                    return std::make_shared<ov::op::v0::Convert>(boolean, et);
                }

                node square (const node& a) const { return mul(a, a); }

                static node reduce_sum (const node& x, const std::vector<int64_t>& axes, bool keep)
                {
                    return std::make_shared<ov::op::v1::ReduceSum>(x, ints(axes), keep);
                }

                static node reduce_mean (const node& x, const std::vector<int64_t>& axes, bool keep)
                {
                    return std::make_shared<ov::op::v1::ReduceMean>(x, ints(axes), keep);
                }

                static node reduce_max (const node& x, const std::vector<int64_t>& axes, bool keep)
                {
                    return std::make_shared<ov::op::v1::ReduceMax>(x, ints(axes), keep);
                }

                static node matmul (const node& a, const node& b, bool ta = false, bool tb = false)
                {
                    return std::make_shared<ov::op::v0::MatMul>(a, b, ta, tb);
                }

                static node reshape (const node& x, const std::vector<int64_t>& dims)
                {
                    return std::make_shared<ov::op::v1::Reshape>(x, ints(dims), false);
                }

                static node transpose (const node& x, const std::vector<int64_t>& perm)
                {
                    return std::make_shared<ov::op::v1::Transpose>(x, ints(perm));
                }

                static node softmax (const node& x, int64_t axis)
                {
                    return std::make_shared<ov::op::v8::Softmax>(x, axis);
                }

                static node broadcast (const node& x, const std::vector<int64_t>& dims)
                {
                    return std::make_shared<ov::op::v3::Broadcast>(x, ints(dims));
                }

                static node slice (const node& x, const std::vector<int64_t>& start,
                    const std::vector<int64_t>& stop, const std::vector<int64_t>& axes)
                {
                    std::vector<int64_t> step(start.size(), 1);
                    return std::make_shared<ov::op::v8::Slice>(x, ints(start), ints(stop), ints(step), ints(axes));
                }

                static node concat (const std::vector<node>& parts, int64_t axis)
                {
                    ov::OutputVector v(parts.begin(), parts.end());
                    return std::make_shared<ov::op::v0::Concat>(v, axis);
                }

                static node gather (const node& data, const node& indices, int64_t axis)
                {
                    return std::make_shared<ov::op::v8::Gather>(data, indices, ints({axis}), 0);
                }

                static ov::Shape to_shape (const std::vector<int64_t>& dims)
                {
                    ov::Shape s;
                    for (auto d : dims)
                        s.push_back(static_cast<size_t>(d));
                    return s;
                }
            };

            using builder = void (*)(graph&, const params&);

    // ----------------------------------------------------------------------------------------
    //                            Exchanges with the device buffers
    // ----------------------------------------------------------------------------------------

            /*
                Where a device buffer meets host memory. A device buffer is a dense array of
                p[0]*p[1]*p[2]*p[3] values, whatever shape the graph gives it. The host values
                occupy its leading corner of extents v[0..3], found at strides s[0..3] from the
                host pointer; the rest of the buffer is padding. When rows is set, device row i
                of p[3] values instead comes from host row rows[i], at stride s[2].
            */
            struct geometry
            {
                int64_t v[4] = {1, 1, 1, 1};
                int64_t p[4] = {1, 1, 1, 1};
                int64_t s[4] = {0, 0, 0, 1};
                const long* rows = nullptr;
                size_t row_count = 0;

                size_t padded_size () const { return static_cast<size_t>(p[0] * p[1] * p[2] * p[3]); }
                size_t valid_size () const
                {
                    if (rows) return row_count * static_cast<size_t>(v[3]);
                    return static_cast<size_t>(v[0] * v[1] * v[2] * v[3]);
                }
                size_t padded_rows () const { return static_cast<size_t>(p[0] * p[1] * p[2]); }

                // Host offset of device row r, or -1 when that row is padding.
                int64_t host_row (size_t r) const
                {
                    if (rows)
                        return r < row_count ? rows[r] * s[2] : -1;
                    const int64_t n = static_cast<int64_t>(r) / (p[1] * p[2]);
                    const int64_t k = (static_cast<int64_t>(r) / p[2]) % p[1];
                    const int64_t i = static_cast<int64_t>(r) % p[2];
                    if (n >= v[0] || k >= v[1] || i >= v[2])
                        return -1;
                    return n * s[0] + k * s[1] + i * s[2];
                }
            };

            // A dense 4-D view of a whole tensor, with the given padded extents.
            geometry dense (
                int64_t n, int64_t k, int64_t r, int64_t c,
                int64_t pn, int64_t pk, int64_t pr, int64_t pc
            )
            {
                geometry g;
                g.v[0] = n; g.v[1] = k; g.v[2] = r; g.v[3] = c;
                g.p[0] = pn; g.p[1] = pk; g.p[2] = pr; g.p[3] = pc;
                g.s[0] = k * r * c; g.s[1] = r * c; g.s[2] = c; g.s[3] = 1;
                return g;
            }

            geometry dense (
                int64_t n, int64_t k, int64_t r, int64_t c
            )
            {
                return dense(n, k, r, c, n, k, r, c);
            }

            // n contiguous values, padded to padded_n.
            geometry flat (
                size_t n,
                size_t padded_n
            )
            {
                return dense(1, 1, 1, static_cast<int64_t>(n), 1, 1, 1, static_cast<int64_t>(padded_n));
            }

            // rows x cols matrix with row stride ld, padded to prows x pcols.
            geometry matrix (
                int64_t rows, int64_t cols, int64_t ld,
                int64_t prows, int64_t pcols
            )
            {
                geometry g = dense(1, 1, rows, cols, 1, 1, prows, pcols);
                g.s[2] = ld;
                g.s[1] = rows * ld;
                g.s[0] = rows * ld;
                return g;
            }

            /*
                An [outer, n, inner] view padded to [po, pn, inner], laid out so that the rows
                the exchange copies are as long as possible.
            */
            geometry view3 (
                int64_t outer, int64_t n, int64_t inner,
                int64_t po, int64_t pn
            )
            {
                if (inner == 1)
                {
                    if (n == 1 && pn == 1)
                        return flat(static_cast<size_t>(outer), static_cast<size_t>(po));
                    return dense(1, 1, outer, n, 1, 1, po, pn);
                }
                return dense(1, outer, n, inner, 1, po, pn, inner);
            }

            enum class in_kind
            {
                values,  // finite, within the half range
                scaled,  // finite, brought to a target magnitude by a power of two
                scores,  // softmax inputs: -inf and large negative values are allowed
                indices, // integers given as i32 by the caller
                scalar   // one value
            };

            struct input
            {
                in_kind kind = in_kind::values;
                const float* host = nullptr;
                geometry g;
                float pad = 0;
                float value = 0;                        // scalar
                const std::vector<int32_t>* ints = nullptr; // indices, padded_size() entries
                int exp = 0;                            // values are multiplied by 2^exp on the way
                int target = 4;                         // scaled: largest magnitude ~ 2^target
                std::vector<unsigned char>* row_flags = nullptr; // scores: device rows all -inf
                /*
                    Rows of a softmax argument, contiguous on the host: in half precision each
                    row is brought to a maximum of zero before it is converted, which leaves the
                    result unchanged and keeps the resolution of the values near the maximum.
                    row_shift, when set, receives the amount subtracted from each device row.
                */
                bool shift_rows = false;
                std::vector<float>* row_shift = nullptr;
                /*
                    Values subtracted while packing, before any scaling: offsets[c] from the values
                    of channel c, c being the column when offset_by_column is set and
                    (row / offset_div) % offset_mod otherwise. Taken from the data of a
                    normalization, they bring its values near zero, where both exchange types
                    keep the precision of the deviations from the mean whatever the mean.
                */
                const float* offsets = nullptr;
                bool offset_by_column = false;
                int64_t offset_div = 1;
                int64_t offset_mod = 1;
                /*
                    The block holding a scaled operand that may keep its values from one call to
                    the next, such as the weights of a product: its device copy is then reused
                    for as long as the copy mark of the block shows no write.
                */
                const gpu_data* storage = nullptr;
                scan_result seen;
            };

            enum class out_kind
            {
                assign,  // host = f * result
                add,     // host += f * result
                blend,   // host = beta * host + f * result
                staged,  // copied, times f, into a host vector the caller processes
                ignore
            };

            struct output
            {
                out_kind kind = out_kind::assign;
                float* host = nullptr;
                geometry g;
                float alpha = 1;
                float beta = 0;
                int exp = 0;                            // f = alpha * 2^exp
                std::vector<float>* staged = nullptr;   // staged: valid_size() values
                const std::vector<unsigned char>* zero_rows = nullptr; // device rows written as 0
                bool unit_rows = false;                 // valid rows must sum to one
            };

            struct call
            {
                // Reserved so that the reference add_input() or add_output() returns stays valid
                // while the next operands are added.
                call () { in.reserve(16); out.reserve(8); }

                std::vector<input> in;
                std::vector<output> out;
                // After the scaled inputs are scanned and before anything is written: may set
                // scalars and output exponents; returning false sends the call to the host.
                std::function<bool(call&)> finalize;
                // Set by the executor: the call reused a device copy of an operand, or it saw an
                // operand for the first time or made its first copy, which makes it unfit as a
                // timing sample.
                bool reused = false;
                bool warming = false;

                input& add_input (in_kind kind, const float* host, const geometry& g, float pad = 0)
                {
                    in.emplace_back();
                    in.back().kind = kind;
                    in.back().host = host;
                    in.back().g = g;
                    in.back().pad = pad;
                    return in.back();
                }

                input& add_scalar (float value)
                {
                    in.emplace_back();
                    in.back().kind = in_kind::scalar;
                    in.back().value = value;
                    in.back().g = flat(1, 1);
                    return in.back();
                }

                input& add_indices (const std::vector<int32_t>& ints, size_t padded)
                {
                    in.emplace_back();
                    in.back().kind = in_kind::indices;
                    in.back().ints = &ints;
                    in.back().g = flat(padded, padded);
                    return in.back();
                }

                output& add_output (out_kind kind, float* host, const geometry& g)
                {
                    out.emplace_back();
                    out.back().kind = kind;
                    out.back().host = host;
                    out.back().g = g;
                    return out.back();
                }
            };

            void scan_input (
                input& x
            )
            {
                const geometry& g = x.g;
                const size_t rows = g.padded_rows();
                const size_t len = static_cast<size_t>(g.v[3]);
                std::mutex m;
                scan_result total;
                for_blocks(rows, std::max<size_t>(1, (size_t(1) << 16) / std::max<size_t>(len, 1)),
                    [&](size_t b, size_t e)
                    {
                        scan_result local;
                        for (size_t r = b; r < e; ++r)
                        {
                            const int64_t off = g.host_row(r);
                            if (off >= 0)
                                scan(x.host + off, len, local);
                        }
                        std::lock_guard<std::mutex> lock(m);
                        total.merge(local);
                    });
                x.seen = total;
            }

            // Fills a device buffer from the host. The scan of values and scores inputs is
            // done in the same pass and judged afterwards.
            void write_input (
                ov::Tensor& t,
                input& x,
                bool half
            )
            {
                const geometry& g = x.g;
                DLIB_CASSERT(t.get_size() == g.padded_size(), "device buffer and geometry disagree: "
                    << t.get_size() << " != " << g.padded_size());
                if (x.kind == in_kind::indices)
                {
                    std::memcpy(t.data(), x.ints->data(), t.get_size() * sizeof(int32_t));
                    return;
                }
                if (x.kind == in_kind::scalar)
                {
                    if (half) *static_cast<uint16_t*>(t.data()) = half_from_float(x.value);
                    else *static_cast<float*>(t.data()) = x.value;
                    return;
                }
                const bool fused_scan = x.kind != in_kind::scaled;
                const bool clamp_low = half && x.kind == in_kind::scores;
                const float f = exp2i(x.exp);
                const size_t rows = g.padded_rows();
                const size_t len = static_cast<size_t>(g.v[3]);
                const size_t plen = static_cast<size_t>(g.p[3]);
                std::mutex m;
                scan_result total;
                const size_t grain = std::max<size_t>(1, (size_t(1) << 16) / std::max<size_t>(plen, 1));
                if (x.row_flags)
                    x.row_flags->assign(rows, 0);
                const bool shift = half && x.shift_rows;
                DLIB_CASSERT(!shift || x.exp == 0, "shifted rows are not scaled");
                DLIB_CASSERT(!shift || !x.offsets, "rows are shifted or offset, not both");
                const bool by_column = x.offsets && x.offset_by_column;
                auto row_offset = [&](size_t r) -> float
                {
                    if (!x.offsets || x.offset_by_column)
                        return 0.0f;
                    return x.offsets[(static_cast<int64_t>(r) / x.offset_div) % x.offset_mod];
                };
                if (shift && x.row_shift)
                    x.row_shift->assign(rows, 0.0f);
                /*
                    Scans one row, flags it when it holds nothing but -inf, and returns the amount
                    to subtract from it: its largest finite value when its rows are shifted.
                */
                auto scan_row = [&](const float* src, size_t r, scan_result& local) -> float
                {
                    if (!x.row_flags && !shift)
                    {
                        scan(src, len, local);
                        return 0.0f;
                    }
                    scan_result one;
                    scan(src, len, one);
                    const float lowest = -std::numeric_limits<float>::infinity();
                    if (x.row_flags && one.neginf && !one.nan && !one.posinf && one.maxval == lowest)
                        (*x.row_flags)[r] = 1;
                    local.merge(one);
                    const float offset = (shift && one.maxval > lowest) ? one.maxval : 0.0f;
                    if (shift && x.row_shift)
                        (*x.row_shift)[r] = offset;
                    return offset;
                };
                if (half)
                {
                    uint16_t* dst = static_cast<uint16_t*>(t.data());
                    const uint16_t pad = half_from_float(x.pad);
                    for_blocks(rows, grain, [&](size_t b, size_t e)
                    {
                        scan_result local;
                        for (size_t r = b; r < e; ++r)
                        {
                            uint16_t* row = dst + r * plen;
                            const int64_t off = g.host_row(r);
                            size_t done = 0;
                            if (off >= 0)
                            {
                                const float* src = x.host + off;
                                const float top = fused_scan ? scan_row(src, r, local) : 0.0f;
                                if (by_column)
                                    pack_half_columns(src, x.offsets, row, len, f);
                                else
                                    pack_half(src, row, len, f, shift ? top : row_offset(r), clamp_low || shift);
                                done = len;
                            }
                            std::fill(row + done, row + plen, pad);
                        }
                        if (fused_scan)
                        {
                            std::lock_guard<std::mutex> lock(m);
                            total.merge(local);
                        }
                    });
                }
                else
                {
                    float* dst = static_cast<float*>(t.data());
                    for_blocks(rows, grain, [&](size_t b, size_t e)
                    {
                        scan_result local;
                        for (size_t r = b; r < e; ++r)
                        {
                            float* row = dst + r * plen;
                            const int64_t off = g.host_row(r);
                            size_t done = 0;
                            if (off >= 0)
                            {
                                const float* src = x.host + off;
                                if (fused_scan) scan_row(src, r, local);
                                if (by_column)
                                    pack_float_columns(src, x.offsets, row, len, f);
                                else
                                    pack_float(src, row, len, f, row_offset(r));
                                done = len;
                            }
                            std::fill(row + done, row + plen, x.pad);
                        }
                        if (fused_scan)
                        {
                            std::lock_guard<std::mutex> lock(m);
                            total.merge(local);
                        }
                    });
                }
                if (fused_scan)
                    x.seen = total;
            }

            // Whether the values an input carried are acceptable for the exchange type.
            bool acceptable (
                const input& x,
                bool half
            )
            {
                const scan_result& s = x.seen;
                switch (x.kind)
                {
                    case in_kind::values:
                    case in_kind::scaled:
                        if (s.nan || s.posinf || s.neginf) return false;
                        return !half || x.shift_rows || s.maxabs * exp2i(x.exp) <= 65504.0f;
                    case in_kind::scores:
                        if (s.nan || s.posinf) return false;
                        return !half || x.shift_rows || s.maxval * exp2i(x.exp) <= 65504.0f;
                    default:
                        return true;
                }
            }

            bool check_output (
                const ov::Tensor& t,
                const output& y,
                bool half
            )
            {
                if (y.kind == out_kind::ignore)
                    return true;
                const geometry& g = y.g;
                DLIB_CASSERT(t.get_size() == g.padded_size(), "device buffer and geometry disagree: "
                    << t.get_size() << " != " << g.padded_size());
                const size_t rows = g.padded_rows();
                const size_t len = static_cast<size_t>(g.v[3]);
                const size_t plen = static_cast<size_t>(g.p[3]);
                std::atomic<bool> bad(false);
                const float tolerance = half ? 2e-2f : 1e-4f;
                for_blocks(rows, std::max<size_t>(1, (size_t(1) << 16) / std::max<size_t>(plen, 1)),
                    [&](size_t b, size_t e)
                    {
                        for (size_t r = b; r < e && !bad.load(std::memory_order_relaxed); ++r)
                        {
                            if (g.host_row(r) < 0) continue;
                            if (y.zero_rows && (*y.zero_rows)[r]) continue;
                            if (half)
                            {
                                const uint16_t* row = static_cast<const uint16_t*>(t.data()) + r * plen;
                                if (count_nonfinite_half(row, len) != 0) { bad = true; break; }
                                if (y.unit_rows)
                                {
                                    double sum = 0;
                                    for (size_t i = 0; i < len; ++i) sum += float_from_half(row[i]);
                                    if (std::abs(sum - 1) > tolerance) { bad = true; break; }
                                }
                            }
                            else
                            {
                                const float* row = static_cast<const float*>(t.data()) + r * plen;
                                if (count_nonfinite_float(row, len) != 0) { bad = true; break; }
                                if (y.unit_rows)
                                {
                                    double sum = 0;
                                    for (size_t i = 0; i < len; ++i) sum += row[i];
                                    if (std::abs(sum - 1) > tolerance) { bad = true; break; }
                                }
                            }
                        }
                    });
                return !bad.load();
            }

            void read_output (
                const ov::Tensor& t,
                output& y,
                bool half
            )
            {
                if (y.kind == out_kind::ignore)
                    return;
                const geometry& g = y.g;
                const size_t rows = g.padded_rows();
                const size_t len = static_cast<size_t>(g.v[3]);
                const size_t plen = static_cast<size_t>(g.p[3]);
                const float f = y.alpha * exp2i(y.exp);
                // Valid rows in device order, which is host order: staging addresses them by
                // their rank.
                std::vector<size_t> rank;
                if (y.kind == out_kind::staged)
                {
                    rank.assign(rows, size_t(-1));
                    size_t valid = 0;
                    for (size_t r = 0; r < rows; ++r)
                        if (g.host_row(r) >= 0) rank[r] = valid++;
                    if (y.kind == out_kind::staged)
                        y.staged->assign(valid * len, 0.0f);
                }
                const int mode = (y.kind == out_kind::add) ? 1 : (y.kind == out_kind::blend ? 2 : 0);
                for_blocks(rows, std::max<size_t>(1, (size_t(1) << 16) / std::max<size_t>(plen, 1)),
                    [&](size_t b, size_t e)
                    {
                        for (size_t r = b; r < e; ++r)
                        {
                            const int64_t off = g.host_row(r);
                            if (off < 0) continue;
                            float* dst = (y.kind == out_kind::staged) ? y.staged->data() + rank[r] * len : y.host + off;
                            if (y.zero_rows && (*y.zero_rows)[r])
                            {
                                if (mode == 0) std::fill(dst, dst + len, 0.0f);
                                else if (mode == 2) for (size_t i = 0; i < len; ++i) dst[i] *= y.beta;
                                continue;
                            }
                            if (half)
                                unpack_half(static_cast<const uint16_t*>(t.data()) + r * plen, dst, len, f, y.beta,
                                    mode);
                            else
                                unpack_float(static_cast<const float*>(t.data()) + r * plen, dst, len, f, y.beta, mode);
                        }
                    });
            }

    // ----------------------------------------------------------------------------------------
    //                                     The runtime
    // ----------------------------------------------------------------------------------------

            struct signature
            {
                int op = 0;
                builder build = nullptr;
                params p;

                bool operator== (const signature& o) const
                {
                    return op == o.op && build == o.build && p == o.p;
                }
            };

            struct signature_hash
            {
                size_t operator() (const signature& s) const
                {
                    uint64_t h = 1469598103934665603ull;
                    auto mix = [&](uint64_t x) { h ^= x; h *= 1099511628211ull; };
                    mix(static_cast<uint64_t>(s.op));
                    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(s.build)));
                    for (int i = 0; i < s.p.n; ++i)
                        mix(static_cast<uint64_t>(s.p.v[i]));
                    return static_cast<size_t>(h);
                }
            };

            /*
                A view of a tensor that a product or a convolution reads: the block it lies in,
                where it starts, its geometry and how it is converted. Two calls with the same
                key read the same values when the copy mark of the block did not change between
                them.
            */
            struct operand_key
            {
                int op = 0;
                const gpu_data* storage = nullptr;
                const float* host = nullptr;
                int64_t v[4] = {0, 0, 0, 0};
                int64_t p[4] = {0, 0, 0, 0};
                int64_t s[4] = {0, 0, 0, 0};
                int target = 0;
                bool half = false;

                bool operator== (const operand_key& o) const
                {
                    return op == o.op && storage == o.storage && host == o.host && target == o.target &&
                        half == o.half && std::equal(v, v + 4, o.v) && std::equal(p, p + 4, o.p) &&
                        std::equal(s, s + 4, o.s);
                }
            };

            struct operand_key_hash
            {
                size_t operator() (const operand_key& k) const
                {
                    uint64_t h = 1469598103934665603ull;
                    auto mix = [&](uint64_t x) { h ^= x; h *= 1099511628211ull; };
                    mix(static_cast<uint64_t>(k.op));
                    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(k.storage)));
                    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(k.host)));
                    for (int i = 0; i < 4; ++i)
                    {
                        mix(static_cast<uint64_t>(k.v[i]));
                        mix(static_cast<uint64_t>(k.p[i]));
                        mix(static_cast<uint64_t>(k.s[i]));
                    }
                    mix(static_cast<uint64_t>(k.target) * 2 + (k.half ? 1 : 0));
                    return static_cast<size_t>(h);
                }
            };

            /*
                What the runtime knows of an operand seen before. Without a tensor, the mark its
                block carried when it was last converted, which tells at the next call whether
                its values changed in between. With a tensor, the converted values themselves,
                current while mark equals the copy mark of the block.
            */
            struct operand_copy
            {
                uint64_t mark = 0;               // 0 while the tensor is being written
                ov::Tensor tensor;
                scan_result seen;
                int exp = 0;
                size_t bytes = 0;
                std::atomic<int> users{0};       // calls that bound the tensor and have not ended
                long long last_use = 0;          // now_ns() of the last call that looked it up
            };

            enum entry_state : int { fresh = 0, queued = 1, ready = 2, failed = 3 };
            enum decision_t : int { undecided = 0, on_device = 1, on_host = 2 };

            struct entry
            {
                signature sig;
                std::atomic<int> state{fresh};
                int seen = 0;                    // calls so far, under the cache mutex
                std::mutex exec;                 // the request and its buffers
                ov::CompiledModel compiled;
                ov::InferRequest request;
                std::vector<ov::Tensor> own_inputs;                 // the buffers of the request
                std::vector<std::shared_ptr<operand_copy>> bound;   // the copy bound to an input
                size_t io_bytes = 0;
                std::atomic<unsigned long long> last_use{0};
                std::mutex calib;                // the measurements below
                std::atomic<int> device_samples{0};
                std::atomic<int> host_samples{0};
                long long device_best = std::numeric_limits<long long>::max();
                long long host_best = std::numeric_limits<long long>::max();
                std::atomic<int> decision{undecided};
                std::atomic<int> rejects_in_row{0};
            };

            // The name cpu_measure, whose member is called entry, uses for the type.
            typedef entry impl_entry;

            struct op_counters
            {
                std::atomic<unsigned long long> device_calls{0};
                std::atomic<unsigned long long> host_calls{0};
                std::atomic<unsigned long long> rejected{0};
                std::atomic<unsigned long long> graphs{0};
                std::atomic<unsigned long long> failures{0};
                std::atomic<unsigned long long> reused{0};
                std::atomic<long long> device_ns{0};
                std::atomic<long long> compile_ns{0};
                std::atomic<bool> touched{false};
            };

            const int calibration_samples = 3;
            const size_t max_graphs = 512;
            const size_t max_signatures = 16384;
            // Records of operands kept before those that carry no copy are forgotten.
            const size_t max_operands = 8192;
            // A copy unused for that long gives its memory to a new one when the budget is full.
            const long long idle_copy_ns = 30000000000LL;

            class runtime
            {
            public:
                bool disabled = false;
                bool available = false;
                std::string requested_device = "NPU";
                std::string device;
                std::string full_name;
                std::string why_unavailable;
                bool half = true;                // operands exchanged in half precision
                bool single = false;             // the device computes in single precision
                bool cache_enabled = true;
                std::string cache_dir;
                size_t budget = size_t(2048) << 20;
                int verbose = 0;
                std::atomic<int> mode{static_cast<int>(routing::automatic)};
                std::atomic<bool> sync{false};
                int overrides[OP_COUNT] = {};
                bool enabled_auto[OP_COUNT] = {};
                bool enabled_always[OP_COUNT] = {};

                std::unique_ptr<ov::Core> core;
                ov::AnyMap config;

                std::mutex cache_mutex;
                std::unordered_map<signature, std::shared_ptr<entry>, signature_hash> cache;
                size_t resident = 0;
                size_t compiled_graphs = 0;
                std::atomic<unsigned long long> clock{0};

                std::mutex queue_mutex;
                std::condition_variable queue_cv;
                std::deque<std::shared_ptr<entry>> queue;
                std::thread worker;
                bool stopping = false;

                // Operands kept for the device between calls, under operands_mutex.
                size_t operand_budget = size_t(2048) << 20;
                std::mutex operands_mutex;
                std::unordered_map<operand_key, std::shared_ptr<operand_copy>, operand_key_hash> operands;
                size_t operand_bytes = 0;
                size_t operand_copies = 0;
                unsigned long long operand_conversions = 0;
                std::atomic<uint64_t> next_mark{1};
                // Memory the device reads without a copy, when its plugin offers it.
                bool has_context = false;
                ov::RemoteContext context;

                op_counters counters[OP_COUNT];

                void derive_policies ()
                {
                    for (int op = 0; op < OP_COUNT; ++op)
                    {
                        bool a = false, w = false;
                        switch (op_table[op].cls)
                        {
                            case op_class::compute: a = true; w = true; break;
                            case op_class::movement: a = false; w = single; break;
                            case op_class::exact: a = single; w = single; break;
                        }
                        if (overrides[op] > 0) { a = true; w = true; }
                        if (overrides[op] < 0) { a = false; w = false; }
                        enabled_auto[op] = a;
                        enabled_always[op] = w;
                    }
                }
            };

            void shutdown_runtime ();
            void print_report (std::ostream& out);

            std::string default_cache_dir (
            )
            {
                bool present = false;
#ifdef _WIN32
                std::string base = read_env_path("LOCALAPPDATA", present);
                if (!present || base.empty()) base = read_env_path("TEMP", present);
                if (!present || base.empty()) return std::string();
                return base + "\\dlib\\npu_cache";
#else
                std::string base = read_env_path("XDG_CACHE_HOME", present);
                if (present && !base.empty()) return base + "/dlib/npu_cache";
                base = read_env_path("HOME", present);
                if (!present || base.empty()) return std::string();
                return base + "/.cache/dlib/npu_cache";
#endif
            }

            runtime* make_runtime (
            )
            {
                runtime* r = new runtime;
                bool present = false;

                std::string v = read_env("DLIB_NPU", present);
                if (present && means_off(v))
                {
                    r->disabled = true;
                    r->why_unavailable = "disabled by DLIB_NPU";
                    return r;
                }

                v = read_env("DLIB_NPU_VERBOSE", present);
                if (present) r->verbose = std::max(0, std::atoi(v.c_str()));

                v = read_env("DLIB_NPU_DEVICE", present);
                if (present && !v.empty()) r->requested_device = v;

                v = lowercase(read_env("DLIB_NPU_ROUTING", present));
                if (present)
                {
                    if (v == "always" || v == "force") r->mode = static_cast<int>(routing::always);
                    else if (means_off(v)) r->mode = static_cast<int>(routing::never);
                    else r->mode = static_cast<int>(routing::automatic);
                }

                v = read_env("DLIB_NPU_SYNC", present);
                if (present && !means_off(v) && !v.empty()) r->sync = true;

                v = read_env("DLIB_NPU_MEMORY_MB", present);
                if (present && std::atoll(v.c_str()) > 0)
                    r->budget = static_cast<size_t>(std::atoll(v.c_str())) << 20;

                v = read_env("DLIB_NPU_RESIDENT_MB", present);
                if (present && !v.empty())
                    r->operand_budget = static_cast<size_t>(std::max(0LL, std::atoll(v.c_str()))) << 20;

                v = read_env_path("DLIB_NPU_CACHE_DIR", present);
                if (present)
                {
                    r->cache_dir = v;
                    r->cache_enabled = !v.empty();
                }
                else
                {
                    r->cache_dir = default_cache_dir();
                    r->cache_enabled = !r->cache_dir.empty();
                }

                v = read_env("DLIB_NPU_OPS", present);
                if (present)
                {
                    std::stringstream ss(v);
                    std::string item;
                    while (std::getline(ss, item, ','))
                    {
                        int sign = 1;
                        while (!item.empty() &&
                            std::isspace(static_cast<unsigned char>(item.front()))) item.erase(item.begin());
                        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back()))) item.pop_back();
                        if (!item.empty() && (item[0] == '-' || item[0] == '+'))
                        {
                            sign = item[0] == '-' ? -1 : 1;
                            item.erase(item.begin());
                        }
                        for (int op = 0; op < OP_COUNT; ++op)
                            if (item == op_table[op].name)
                                r->overrides[op] = sign;
                    }
                }

                try
                {
                    r->core.reset(new ov::Core);
                }
                catch (const std::exception& e)
                {
                    r->why_unavailable = std::string("OpenVINO could not start: ") + e.what();
                    return r;
                }

                // Only the plugin of the requested device is loaded, so that a host without an
                // NPU pays for one plugin query and nothing else.
                const std::string family = r->requested_device.substr(0, r->requested_device.find('.'));
                const bool wants_id = family.size() < r->requested_device.size();
                std::vector<std::string> ids;
                try
                {
                    ids = r->core->get_property(family, ov::available_devices);
                }
                catch (const std::exception& e)
                {
                    r->why_unavailable = "no " + family + " plugin could be used: " + e.what();
                }
                for (const auto& id : ids)
                {
                    const std::string name = (ids.size() > 1 && !id.empty()) ? family + "." + id : family;
                    if (!wants_id || name == r->requested_device || family + "." + id == r->requested_device)
                    {
                        r->device = wants_id ? r->requested_device : family;
                        break;
                    }
                }
                if (r->device.empty())
                {
                    if (r->why_unavailable.empty())
                    {
                        std::ostringstream sout;
                        sout << "no " << r->requested_device << " device found";
                        if (!ids.empty())
                        {
                            sout << " (" << family << " devices:";
                            for (const auto& id : ids)
                                sout << " " << (id.empty() ? family : family + "." + id);
                            sout << ")";
                        }
                        r->why_unavailable = sout.str();
                    }
                    r->core.reset();
                    if (r->verbose > 0)
                        std::cerr << "dlib: NPU routing inactive, " << r->why_unavailable << std::endl;
                    return r;
                }

                try
                {
                    r->full_name = r->core->get_property(r->device, ov::device::full_name);
                }
                catch (...)
                {
                    r->full_name = r->device;
                }

                const bool cpu_device = r->device.rfind("CPU", 0) == 0;
                const bool gpu_device = r->device.rfind("GPU", 0) == 0;
                r->half = !cpu_device;
                v = lowercase(read_env("DLIB_NPU_PRECISION", present));
                if (present && (v == "f32" || v == "fp32" || v == "float"))
                    r->half = false;
                else if (present && (v == "f16" || v == "fp16" || v == "half"))
                    r->half = true;

                // An NPU computes in reduced precision whatever the exchange type.
                r->single = !r->half && (cpu_device || gpu_device);

                r->config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
                if (cpu_device || gpu_device)
                    r->config.insert(ov::hint::inference_precision(r->half ? ov::element::f16 : ov::element::f32));

                if (r->cache_enabled)
                {
                    std::error_code ec;
                    std::filesystem::create_directories(std::filesystem::u8path(r->cache_dir), ec);
                    try
                    {
                        r->core->set_property(ov::cache_dir(r->cache_dir));
                    }
                    catch (...)
                    {
                        r->cache_enabled = false;
                    }
                }

                // Copies of operands go in memory the device reads directly when the plugin
                // provides it, as the NPU and GPU plugins do; elsewhere in ordinary memory.
                try
                {
                    r->context = r->core->get_default_context(r->device);
                    r->has_context = true;
                }
                catch (...)
                {
                    r->has_context = false;
                }

                r->derive_policies();
                r->available = true;
                // Registered after the OpenVINO runtime started, so that it runs before the
                // runtime's own static objects are destroyed.
                std::atexit(shutdown_runtime);
                if (r->verbose > 0)
                    std::cerr << "dlib: NPU routing active on " << r->device << " (" << r->full_name << "), "
                        << (r->half ? "f16" : "f32") << " exchange, "
                        << (r->single ? "single" : "half") << " precision arithmetic" << std::endl;
                return r;
            }

            runtime& rt (
            )
            {
                // Never destroyed: shutdown_runtime() releases what it holds at exit, while
                // OpenVINO is still alive, and later calls see an unavailable runtime.
                static runtime* r = make_runtime();
                return *r;
            }

            thread_local int exact_depth = 0;
            thread_local entry* pending_host = nullptr;

            void record_sample (
                entry& e,
                bool device,
                long long ns
            )
            {
                std::lock_guard<std::mutex> lock(e.calib);
                if (e.decision.load() != undecided)
                    return;
                if (device)
                {
                    ++e.device_samples;
                    e.device_best = std::min(e.device_best, ns);
                }
                else
                {
                    ++e.host_samples;
                    e.host_best = std::min(e.host_best, ns);
                }
                if (e.device_samples >= calibration_samples && e.host_samples >= calibration_samples)
                    e.decision = (e.device_best < 0.9 * e.host_best) ? on_device : on_host;
            }

            void release (
                entry& e
            )
            {
                e.bound.clear();
                e.own_inputs.clear();
                e.request = ov::InferRequest();
                e.compiled = ov::CompiledModel();
            }

            void compile_entry (
                runtime& r,
                const std::shared_ptr<entry>& e
            )
            {
                const long long t0 = now_ns();
                const op_info& info = op_table[e->sig.op];
                try
                {
                    graph g(r.half ? ov::element::f16 : ov::element::f32);
                    e->sig.build(g, e->sig.p);
                    auto model = std::make_shared<ov::Model>(g.results, g.parameters, std::string("dlib_") + info.name);
                    ov::CompiledModel compiled = r.core->compile_model(model, r.device, r.config);
                    ov::InferRequest request = compiled.create_infer_request();
                    size_t bytes = 0;
                    std::vector<ov::Tensor> own;
                    for (size_t i = 0; i < compiled.inputs().size(); ++i)
                    {
                        ov::Tensor t = request.get_input_tensor(i);
                        std::memset(t.data(), 0, t.get_byte_size());
                        bytes += t.get_byte_size();
                        own.push_back(t);
                    }
                    for (size_t i = 0; i < compiled.outputs().size(); ++i)
                        bytes += request.get_output_tensor(i).get_byte_size();
                    if (bytes > r.budget)
                        throw std::runtime_error("graph buffers exceed DLIB_NPU_MEMORY_MB");
                    request.infer(); // the first inference pays one-time costs; not a sample

                    std::vector<std::shared_ptr<entry>> evicted;
                    {
                        std::lock_guard<std::mutex> lock(e->exec);
                        e->compiled = std::move(compiled);
                        e->request = std::move(request);
                        e->bound.assign(own.size(), nullptr);
                        e->own_inputs = std::move(own);
                        e->io_bytes = bytes;
                    }
                    {
                        std::lock_guard<std::mutex> lock(r.cache_mutex);
                        r.resident += bytes;
                        ++r.compiled_graphs;
                        e->last_use = ++r.clock;
                        e->state = ready;
                        // Least recently used graphs give their buffers back first, past the memory
                        // budget or the number of graphs kept.
                        while (r.resident > r.budget || r.compiled_graphs > max_graphs)
                        {
                            std::shared_ptr<entry> victim;
                            for (auto& kv : r.cache)
                            {
                                auto& c = kv.second;
                                if (c == e || c->state.load() != ready) continue;
                                if (!victim || c->last_use.load() < victim->last_use.load()) victim = c;
                            }
                            if (!victim) break;
                            victim->state = fresh;
                            victim->seen = 0;
                            r.resident -= victim->io_bytes;
                            --r.compiled_graphs;
                            evicted.push_back(victim);
                        }
                    }
                    for (auto& victim : evicted)
                    {
                        std::lock_guard<std::mutex> lock(victim->exec);
                        release(*victim);
                        victim->io_bytes = 0;
                    }
                    const long long dt = now_ns() - t0;
                    r.counters[e->sig.op].graphs++;
                    r.counters[e->sig.op].compile_ns += dt;
                    if (r.verbose > 1)
                        std::cerr << "dlib: NPU compiled " << info.name << " in " << dt / 1e6 << " ms ("
                            << bytes / 1024 << " KiB of buffers)" << std::endl;
                }
                catch (const std::exception& ex)
                {
                    e->state = failed;
                    r.counters[e->sig.op].failures++;
                    r.counters[e->sig.op].compile_ns += now_ns() - t0;
                    if (r.verbose > 0)
                        std::cerr << "dlib: NPU could not compile " << info.name << ", the host keeps it: "
                            << std::string(ex.what()).substr(0, 400) << std::endl;
                }
            }

            void worker_loop (
                runtime* r
            )
            {
                for (;;)
                {
                    std::shared_ptr<entry> e;
                    {
                        std::unique_lock<std::mutex> lock(r->queue_mutex);
                        r->queue_cv.wait(lock, [&] { return r->stopping || !r->queue.empty(); });
                        if (r->stopping)
                            return;
                        e = r->queue.front();
                        r->queue.pop_front();
                    }
                    compile_entry(*r, e);
                }
            }

            void enqueue_compile (
                runtime& r,
                const std::shared_ptr<entry>& e
            )
            {
                std::lock_guard<std::mutex> lock(r.queue_mutex);
                if (r.stopping)
                    return;
                if (!r.worker.joinable())
                    r.worker = std::thread(worker_loop, &r);
                r.queue.push_back(e);
                r.queue_cv.notify_one();
            }

            void shutdown_runtime (
            )
            {
                runtime& r = rt();
                {
                    std::lock_guard<std::mutex> lock(r.queue_mutex);
                    r.stopping = true;
                    r.queue.clear();
                }
                r.queue_cv.notify_all();
                if (r.worker.joinable())
                    r.worker.join();
                if (r.verbose > 0 && r.available)
                    print_report(std::cerr);
                r.available = false;
                std::lock_guard<std::mutex> lock(r.cache_mutex);
                for (auto& kv : r.cache)
                {
                    std::lock_guard<std::mutex> elock(kv.second->exec);
                    kv.second->state = failed;
                    release(*kv.second);
                }
                r.cache.clear();
                {
                    // The copies of operands may live in memory of the plugin, released with it.
                    std::lock_guard<std::mutex> olock(r.operands_mutex);
                    r.operands.clear();
                    r.operand_bytes = 0;
                    r.operand_copies = 0;
                    r.context = ov::RemoteContext();
                    r.has_context = false;
                }
                r.core.reset();
            }

            /*
                Whether an operation of this size may go to the device at all. Cheap: it is
                called by every qualifying tensor operation before anything else is done.
            */
            bool admit (
                int op,
                double work
            )
            {
                pending_host = nullptr;
                runtime& r = rt();
                if (!r.available || exact_depth > 0)
                    return false;
                const int m = r.mode.load(std::memory_order_relaxed);
                if (m == static_cast<int>(routing::never))
                    return false;
                if (m == static_cast<int>(routing::always))
                    return r.enabled_always[op];
                return r.enabled_auto[op] && work >= op_table[op].min_work;
            }

    // ----------------------------------------------------------------------------------------
    //                         Operands kept for the device between calls
    // ----------------------------------------------------------------------------------------

            /*
                The use a call makes of what the runtime knows of one of its operands. Without
                a copy, the operand is converted into the buffer of the graph, as any other.
            */
            struct operand_use
            {
                std::shared_ptr<operand_copy> copy;
                bool hit = false;        // the copy holds the current values: nothing to convert
                bool warming = false;    // first sight of the operand or first copy of it
                uint64_t mark = 0;       // a copy to write: the mark it carries once written
            };

            // A buffer the device reads without a further copy where the plugin allows it.
            ov::Tensor device_tensor (
                runtime& r,
                const ov::Shape& shape
            )
            {
                const ov::element::Type t = r.half ? ov::element::f16 : ov::element::f32;
                if (r.has_context)
                {
                    try
                    {
                        return r.context.create_host_tensor(t, shape);
                    }
                    catch (...)
                    {
                        r.has_context = false;
                    }
                }
                return ov::Tensor(t, shape);
            }

            // Under operands_mutex. Forgets the records without a copy, and the copies no call
            // looked up for idle_copy_ns.
            void prune_operands (
                runtime& r,
                long long now
            )
            {
                for (auto it = r.operands.begin(); it != r.operands.end();)
                {
                    operand_copy& c = *it->second;
                    const bool idle = now - c.last_use > idle_copy_ns && c.users.load() == 0;
                    if (!c.tensor || idle)
                    {
                        if (c.tensor)
                        {
                            r.operand_bytes -= c.bytes;
                            --r.operand_copies;
                        }
                        it = r.operands.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }

            // Under operands_mutex. Whether a copy of that size fits the budget, once the idle
            // copies are let go if it does not.
            bool make_room (
                runtime& r,
                size_t bytes,
                long long now
            )
            {
                if (r.operand_bytes + bytes <= r.operand_budget)
                    return true;
                for (auto it = r.operands.begin(); it != r.operands.end();)
                {
                    operand_copy& c = *it->second;
                    if (c.tensor && c.users.load() == 0 && now - c.last_use > idle_copy_ns)
                    {
                        r.operand_bytes -= c.bytes;
                        --r.operand_copies;
                        it = r.operands.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                return r.operand_bytes + bytes <= r.operand_budget;
            }

            /*
                Looks up what is known of the operand x of input i of e, whose request the
                caller holds. Returns a copy to bind when the block kept its values since that
                copy was written; a copy to write when the operand proved unchanged between two
                calls, or when its copy is outdated and no call reads it; nothing otherwise.
            */
            operand_use acquire_operand (
                runtime& r,
                entry& e,
                size_t i,
                const input& x
            )
            {
                operand_use u;
                if (!x.storage || r.operand_budget == 0)
                    return u;
                operand_key key;
                key.op = e.sig.op;
                key.storage = x.storage;
                key.host = x.host;
                key.target = x.target;
                key.half = r.half;
                for (int d = 0; d < 4; ++d)
                {
                    key.v[d] = x.g.v[d];
                    key.p[d] = x.g.p[d];
                    key.s[d] = x.g.s[d];
                }
                const uint64_t current = x.storage->copy_mark();
                const long long now = now_ns();
                std::lock_guard<std::mutex> lock(r.operands_mutex);
                auto it = r.operands.find(key);
                if (it != r.operands.end())
                {
                    operand_copy& c = *it->second;
                    c.last_use = now;
                    if (c.tensor && c.mark != 0 && c.mark == current)
                    {
                        ++c.users;
                        u.copy = it->second;
                        u.hit = true;
                        return u;
                    }
                }
                // The block is marked before its values are read, so that any write from now
                // on clears the mark.
                const uint64_t mark = x.storage->set_copy_mark(r.next_mark++);
                if (it == r.operands.end())
                {
                    if (r.operands.size() >= max_operands)
                        prune_operands(r, now);
                    auto c = std::make_shared<operand_copy>();
                    c->mark = mark;
                    c->last_use = now;
                    r.operands.emplace(key, std::move(c));
                    u.warming = true;
                    return u;
                }
                const std::shared_ptr<operand_copy>& c = it->second;
                if (!c->tensor)
                {
                    const bool unchanged = c->mark != 0 && c->mark == current;
                    const ov::Shape shape = e.compiled.input(i).get_shape();
                    const size_t bytes = ov::shape_size(shape) * (r.half ? 2 : 4);
                    if (unchanged && make_room(r, bytes, now))
                    {
                        c->tensor = device_tensor(r, shape);
                        c->bytes = bytes;
                        r.operand_bytes += bytes;
                        ++r.operand_copies;
                        c->mark = 0;
                        ++c->users;
                        u.copy = c;
                        u.mark = mark;
                        u.warming = true;
                        return u;
                    }
                    c->mark = mark;
                    return u;
                }
                // A copy of earlier values, written again unless a call still reads it.
                if (c->users.load() == 0)
                {
                    c->mark = 0;
                    ++c->users;
                    u.copy = c;
                    u.mark = mark;
                }
                return u;
            }

            bool execute (
                runtime& r,
                entry& e,
                call& c
            )
            {
                const bool half = r.half;
                std::lock_guard<std::mutex> lock(e.exec);
                if (e.state.load() != ready)
                    return false;
                ov::InferRequest& request = e.request;
                DLIB_CASSERT(e.compiled.inputs().size() == c.in.size() && e.compiled.outputs().size() == c.out.size(),
                    "graph and call disagree on " << op_table[e.sig.op].name);

                // The copies a call holds are given back however the call ends.
                std::vector<operand_use> uses(c.in.size());
                struct release_uses
                {
                    std::vector<operand_use>& uses;
                    ~release_uses () { for (auto& u : uses) if (u.copy) --u.copy->users; }
                } releaser{uses};

                for (size_t i = 0; i < c.in.size(); ++i)
                {
                    input& x = c.in[i];
                    if (x.storage)
                        uses[i] = acquire_operand(r, e, i, x);
                    if (uses[i].warming)
                        c.warming = true;
                    if (uses[i].hit)
                    {
                        x.seen = uses[i].copy->seen;
                        x.exp = uses[i].copy->exp;
                        c.reused = true;
                    }
                    else if (x.kind == in_kind::scaled)
                    {
                        scan_input(x);
                        if (x.seen.nan || x.seen.posinf || x.seen.neginf)
                            return false;
                        x.exp = 0;
                        if (half && x.seen.maxabs > 0)
                            x.exp = std::max(-60, std::min(60, x.target - ceil_log2(x.seen.maxabs)));
                    }
                }
                if (c.finalize && !c.finalize(c))
                    return false;

                for (size_t i = 0; i < c.in.size(); ++i)
                {
                    operand_use& u = uses[i];
                    if (u.copy)
                    {
                        if (!u.hit)
                            write_input(u.copy->tensor, c.in[i], half);
                        if (e.bound[i] != u.copy)
                        {
                            request.set_input_tensor(i, u.copy->tensor);
                            e.bound[i] = u.copy;
                        }
                    }
                    else
                    {
                        if (e.bound[i])
                        {
                            request.set_input_tensor(i, e.own_inputs[i]);
                            e.bound[i].reset();
                        }
                        write_input(e.own_inputs[i], c.in[i], half);
                    }
                }
                for (const auto& x : c.in)
                    if (!acceptable(x, half))
                        return false;
                {
                    // The copies this call wrote hold the values their blocks carry.
                    std::lock_guard<std::mutex> olock(r.operands_mutex);
                    for (size_t i = 0; i < c.in.size(); ++i)
                    {
                        operand_use& u = uses[i];
                        if (u.copy && !u.hit)
                        {
                            u.copy->seen = c.in[i].seen;
                            u.copy->exp = c.in[i].exp;
                            u.copy->mark = u.mark;
                            ++r.operand_conversions;
                        }
                    }
                }
                request.infer();
                for (size_t i = 0; i < c.out.size(); ++i)
                    if (!check_output(request.get_output_tensor(i), c.out[i], half))
                        return false;
                for (size_t i = 0; i < c.out.size(); ++i)
                    read_output(request.get_output_tensor(i), c.out[i], half);
                return true;
            }

            /*
                Routes one call. bind(c) describes the operands and is only invoked when the
                device is to compute. Returns true when it did.
            */
            template <typename BIND>
            bool route (
                int op,
                builder build,
                const params& p,
                BIND&& bind
            )
            {
                runtime& r = rt();
                op_counters& counters = r.counters[op];
                counters.touched = true;
                const bool always = r.mode.load(std::memory_order_relaxed) == static_cast<int>(routing::always);

                signature sig;
                sig.op = op;
                sig.build = build;
                sig.p = p;
                std::shared_ptr<entry> e;
                bool compile_now = false, compile_later = false;
                {
                    std::lock_guard<std::mutex> lock(r.cache_mutex);
                    auto it = r.cache.find(sig);
                    if (it == r.cache.end())
                    {
                        // Shapes seen once and never compiled are forgotten first.
                        if (r.cache.size() >= max_signatures)
                        {
                            for (auto i = r.cache.begin(); i != r.cache.end();)
                            {
                                if (i->second->state.load() == fresh) i = r.cache.erase(i);
                                else ++i;
                            }
                        }
                        e = std::make_shared<entry>();
                        e->sig = sig;
                        r.cache.emplace(sig, e);
                    }
                    else
                    {
                        e = it->second;
                    }
                    if (e->state.load() == fresh)
                    {
                        // A shape seen once may never come back: compile from the second call.
                        const bool sync = r.sync.load();
                        const int needed = (sync || always) ? 1 : 2;
                        if (++e->seen >= needed)
                        {
                            e->state = queued;
                            if (sync) compile_now = true;
                            else compile_later = true;
                        }
                    }
                }
                if (compile_now) compile_entry(r, e);
                else if (compile_later) enqueue_compile(r, e);

                if (e->state.load() != ready)
                {
                    counters.host_calls++;
                    return false;
                }

                bool timed = false;
                if (!always)
                {
                    const int d = e->decision.load();
                    if (d == on_host)
                    {
                        counters.host_calls++;
                        return false;
                    }
                    if (d == undecided)
                    {
                        if (e->device_samples <= e->host_samples)
                        {
                            timed = true;
                        }
                        else
                        {
                            pending_host = e.get();
                            counters.host_calls++;
                            return false;
                        }
                    }
                }

                call c;
                bind(c);
                const long long t0 = now_ns();
                bool ok = false;
                try
                {
                    ok = execute(r, *e, c);
                }
                catch (const std::exception& ex)
                {
                    e->state = failed;
                    counters.failures++;
                    if (r.verbose > 0)
                        std::cerr << "dlib: NPU failed to run " << op_table[op].name << ", the host keeps it: "
                            << std::string(ex.what()).substr(0, 400) << std::endl;
                    ok = false;
                }
                const long long dt = now_ns() - t0;
                if (!ok)
                {
                    counters.rejected++;
                    if (++e->rejects_in_row >= 8 && !always)
                        e->decision = on_host;
                    return false;
                }
                e->rejects_in_row = 0;
                e->last_use = ++r.clock;
                counters.device_calls++;
                counters.device_ns += dt;
                if (c.reused)
                    counters.reused++;
                // A call that met an operand for the first time, or made its first copy, costs
                // more than the calls that follow: the next one is timed instead.
                if (timed && !c.warming)
                    record_sample(*e, true, dt);
                return true;
            }

    // ----------------------------------------------------------------------------------------
    //                                   Shapes and buckets
    // ----------------------------------------------------------------------------------------

            /*
                Rounds a dimension that changes from call to call up to one of eight sizes per
                octave, so that at most an eighth of the work is padding and the number of
                graphs a varying dimension produces stays small.
            */
            int64_t bucket (
                int64_t n
            )
            {
                if (n <= 8)
                    return std::max<int64_t>(n, 1);
                if (n <= 16)
                    return 16;
                int64_t high = 1;
                while ((high << 1) <= n) high <<= 1;
                const int64_t step = std::max<int64_t>(high >> 3, 1);
                return (n + step - 1) / step * step;
            }

            // Flat layout of n values for the elementwise graphs: rows of width W.
            const int64_t flat_width = 256;

            int64_t flat_rows (
                size_t n
            )
            {
                return bucket(static_cast<int64_t>((n + flat_width - 1) / flat_width));
            }

            inline int64_t sz (long long x) { return static_cast<int64_t>(x); }

            // A gradient computed from gradient_input is assigned when the two are the same
            // object and added otherwise, as in every host gradient of cpu_dlib.cpp.
            inline out_kind accumulate_unless_same (const tensor& grad, const tensor& gradient_input)
            {
                return is_same_object(grad, gradient_input) ? out_kind::assign : out_kind::add;
            }

            // Linear operands of a product are brought to a magnitude that keeps a sum of K
            // products within the half range.
            inline int product_target (int64_t k)
            {
                return k <= 1024 ? 4 : (k <= 16384 ? 3 : 2);
            }

    // ----------------------------------------------------------------------------------------
    //                               Elementwise operations
    // ----------------------------------------------------------------------------------------

            /*
                Unary operations dest = f(src, scalars) over a flat view. Their graphs depend on
                the operation and on the padded length only.
            */
            node unary_math (
                graph& g,
                int op,
                const node& x,
                const std::vector<node>& s
            )
            {
                switch (op)
                {
                    case OP_sigmoid: return g.sigmoid(x);
                    case OP_tanh: return g.tanh(x);
                    case OP_relu: return g.relu(x);
                    case OP_mish: return std::make_shared<ov::op::v4::Mish>(x);
                    case OP_gelu: return std::make_shared<ov::op::v7::Gelu>(x, ov::op::GeluApproximationMode::ERF);
                    case OP_silu: return g.mul(x, g.sigmoid(x));
                    case OP_elu: return g.select(g.greater(x, g.c(0)), x, g.mul(s[0], g.sub(g.exp(x), g.c(1))));
                    case OP_leaky_relu:
                    case OP_prelu: return g.select(g.greater(x, g.c(0)), x, g.mul(s[0], x));
                    case OP_clipped_relu: return g.min(g.max(x, g.c(0)), s[0]);
                    case OP_smelu:
                    {
                        const node b = s[0];
                        const node quad = g.div(g.square(g.add(x, b)), g.mul(g.c(4), b));
                        return g.select(g.greater_equal(x, b), x,
                            g.select(g.less_equal(x, g.mul(b, g.c(-1))), g.c(0), quad));
                    }
                    case OP_exp: return g.exp(x);
                    case OP_log: return g.log(x);
                    case OP_log10: return g.mul(g.log(x), g.c(1.0 / std::log(10.0)));
                    case OP_threshold: return g.to_value(g.greater(x, s[0]));
                    case OP_inverse_sqrt: return g.div(g.c(1), g.sqrt(g.add(x, s[0])));
                    case OP_affine_transform: return g.add(g.mul(x, s[0]), s[1]);
                    default: break;
                }
                throw std::logic_error("unknown unary operation");
            }

            // p: op, rows, width, number of scalars
            void build_unary (graph& g, const params& p)
            {
                const node x = g.input({p[1], p[2]});
                std::vector<node> s;
                for (int i = 0; i < p[3]; ++i) s.push_back(g.scalar());
                g.output(unary_math(g, static_cast<int>(p[0]), x, s));
            }

            /*
                True when a linear combination assigns one of its own operands while the device
                computes in half precision. Such a call updates a tensor in place, as the solvers
                do with the parameters and their moments, and the device result would round the
                values that tensor carries from one call to the next, so the host keeps it.
            */
            bool rounds_in_place (
                const float* dest,
                std::initializer_list<const float*> srcs
            )
            {
                if (rt().single)
                    return false;
                for (const float* s : srcs)
                    if (s == dest)
                        return true;
                return false;
            }

            bool unary (
                int op,
                tensor& dest,
                const tensor& src,
                std::initializer_list<float> scalars = {}
            )
            {
                const size_t n = src.size();
                if (n == 0 || dest.size() != n || !admit(op, static_cast<double>(n)))
                    return false;
                if (op == OP_affine_transform && rounds_in_place(dest.host(), {src.host()}))
                    return false;
                const int64_t rows = flat_rows(n);
                const size_t padded = static_cast<size_t>(rows * flat_width);
                const std::vector<float> sv(scalars);
                params p;
                p << op << rows << flat_width << static_cast<int64_t>(sv.size());
                return route(op, &build_unary, p, [&](call& c)
                {
                    c.add_input(in_kind::values, src.host(), flat(n, padded));
                    for (float s : sv) c.add_scalar(s);
                    c.add_output(out_kind::assign, dest.host(), flat(n, padded));
                });
            }

            /*
                Gradients grad (+)= gradient_input * f'(x, scalars), x being the input or the
                output of the forward operation depending on the operation.
            */
            node gradient_math (
                graph& g,
                int op,
                const node& x,
                const node& gi,
                const std::vector<node>& s
            )
            {
                switch (op)
                {
                    case OP_sigmoid_gradient: return g.mul(gi, g.mul(x, g.sub(g.c(1), x)));
                    case OP_tanh_gradient: return g.mul(gi, g.sub(g.c(1), g.square(x)));
                    case OP_relu_gradient: return g.select(g.greater(x, g.c(0)), gi, g.c(0));
                    case OP_mish_gradient:
                    {
                        // mish'(x) = (1 - 2u) + x sigmoid(x) (4u - 4u^2), u = 1/(e^2x + 2e^x + 2),
                        // the host formula rewritten so that no intermediate overflows: u falls
                        // to zero where e^2x leaves the half range, which is its limit.
                        const node e = g.exp(x);
                        const node u = g.div(g.c(1), g.add(g.add(g.square(e), g.mul(g.c(2), e)), g.c(2)));
                        const node d = g.add(g.sub(g.c(1), g.mul(g.c(2), u)),
                            g.mul(g.mul(x, g.sigmoid(x)), g.mul(g.c(4), g.sub(u, g.square(u)))));
                        const node clipped = g.select(g.greater_equal(x, g.c(8)), g.c(1),
                            g.select(g.less_equal(x, g.c(-8)), g.c(0), d));
                        return g.mul(gi, clipped);
                    }
                    case OP_gelu_gradient:
                    {
                        const node cdf = g.mul(g.c(0.5), g.add(g.c(1), g.erf(g.mul(x, g.c(1.0 / std::sqrt(2.0))))));
                        const node pdf = g.mul(g.c(1.0 / std::sqrt(2.0 * 3.14159265358979323846)),
                            g.exp(g.mul(g.c(-0.5), g.square(x))));
                        return g.mul(gi, g.add(cdf, g.mul(x, pdf)));
                    }
                    case OP_silu_gradient:
                    {
                        const node sg = g.sigmoid(x);
                        return g.mul(gi, g.mul(sg, g.add(g.c(1), g.mul(x, g.sub(g.c(1), sg)))));
                    }
                    case OP_elu_gradient: return g.select(g.greater(x, g.c(0)), gi, g.mul(g.add(s[0], x), gi));
                    case OP_leaky_relu_gradient: return g.select(g.greater(x, g.c(0)), gi, g.mul(s[0], gi));
                    case OP_clipped_relu_gradient:
                        return g.select(g.logical_and(g.greater(x, g.c(0)), g.less(x, s[0])), gi, g.c(0));
                    case OP_smelu_gradient:
                    {
                        const node b = s[0];
                        const node mid = g.mul(g.div(g.sqrt(g.mul(b, x)), b), gi);
                        return g.select(g.greater_equal(x, b), gi, g.select(g.equal(x, g.c(0)), g.c(0), mid));
                    }
                    default: break;
                }
                throw std::logic_error("unknown gradient operation");
            }

            // p: op, rows, width, number of scalars
            void build_gradient (graph& g, const params& p)
            {
                const node x = g.input({p[1], p[2]});
                const node gi = g.input({p[1], p[2]});
                std::vector<node> s;
                for (int i = 0; i < p[3]; ++i) s.push_back(g.scalar());
                g.output(gradient_math(g, static_cast<int>(p[0]), x, gi, s));
            }

            bool gradient (
                int op,
                tensor& grad,
                const tensor& x,
                const tensor& gradient_input,
                std::initializer_list<float> scalars = {}
            )
            {
                const size_t n = x.size();
                if (n == 0 || grad.size() != n || gradient_input.size() != n || !admit(op, static_cast<double>(n)))
                    return false;
                const int64_t rows = flat_rows(n);
                const size_t padded = static_cast<size_t>(rows * flat_width);
                const std::vector<float> sv(scalars);
                params p;
                p << op << rows << flat_width << static_cast<int64_t>(sv.size());
                const out_kind kind = accumulate_unless_same(grad, gradient_input);
                return route(op, &build_gradient, p, [&](call& c)
                {
                    c.add_input(in_kind::values, x.host(), flat(n, padded));
                    c.add_input(in_kind::scaled, gradient_input.host(), flat(n, padded));
                    for (float s : sv) c.add_scalar(s);
                    c.add_output(kind, grad.host(), flat(n, padded));
                    c.finalize = [](call& c) { c.out[0].exp = -c.in[1].exp; return true; };
                });
            }

            /*
                Elementwise operations of several same-sized operands over a flat view.
            */
            enum nary_kind : int
            {
                NK_add,            // s1 + s2
                NK_multiply,       // s1 * s2
                NK_affine2,        // A s1 + B s2 + C
                NK_affine3,        // A s1 + B s2 + C s3 + D
                NK_divide_by_sqrt  // s1 / sqrt(max(s2, eps))
            };

            // p: kind, rows, width, number of inputs, number of scalars
            void build_nary (graph& g, const params& p)
            {
                std::vector<node> x, s;
                for (int i = 0; i < p[3]; ++i) x.push_back(g.input({p[1], p[2]}));
                for (int i = 0; i < p[4]; ++i) s.push_back(g.scalar());
                node y;
                switch (p[0])
                {
                    case NK_add: y = g.add(x[0], x[1]); break;
                    case NK_multiply: y = g.mul(x[0], x[1]); break;
                    case NK_affine2: y = g.add(g.add(g.mul(s[0], x[0]), g.mul(s[1], x[1])), s[2]); break;
                    case NK_affine3:
                        y = g.add(g.add(g.add(g.mul(s[0], x[0]), g.mul(s[1], x[1])), g.mul(s[2], x[2])), s[3]);
                        break;
                    case NK_divide_by_sqrt: y = g.div(x[0], g.sqrt(g.max(x[1], s[0]))); break;
                    default: throw std::logic_error("unknown elementwise operation");
                }
                g.output(y);
            }

            /*
                dest[begin, end) (op)= f(srcs[begin, end), scalars). Multiplications scale their
                operands, everything else carries them as they are.
            */
            bool nary (
                int op,
                int kind,
                out_kind mode,
                float* dest,
                std::vector<const float*> srcs,
                size_t n,
                std::initializer_list<float> scalars = {}
            )
            {
                if (n == 0 || !admit(op, static_cast<double>(n)))
                    return false;
                if (mode == out_kind::assign)
                    for (const float* s : srcs)
                        if (rounds_in_place(dest, {s}))
                            return false;
                const int64_t rows = flat_rows(n);
                const size_t padded = static_cast<size_t>(rows * flat_width);
                const std::vector<float> sv(scalars);
                params p;
                p << kind << rows << flat_width << static_cast<int64_t>(srcs.size()) << static_cast<int64_t>(sv.size());
                const bool product = kind == NK_multiply;
                return route(op, &build_nary, p, [&](call& c)
                {
                    for (const float* s : srcs)
                        c.add_input(product ? in_kind::scaled : in_kind::values, s, flat(n, padded));
                    for (float s : sv) c.add_scalar(s);
                    c.add_output(mode, dest, flat(n, padded));
                    if (product)
                        c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
                });
            }

    // ----------------------------------------------------------------------------------------
    //                                Matrix products
    // ----------------------------------------------------------------------------------------

            /*
                op(A) op(B), A stored as a [P_a, ra, ca] batch, B as [P_b, rb, cb], with batch
                broadcasting when one of them holds a single matrix.
                p: Pa, ra, ca, Pb, rb, cb, trans_a, trans_b
            */
            void build_matmul (graph& g, const params& p)
            {
                const node a = g.input({p[0], p[1], p[2]});
                const node b = g.input({p[3], p[4], p[5]});
                g.output(g.matmul(a, b, p[6] != 0, p[7] != 0));
            }

            struct plane_layout
            {
                long rows, cols;   // of one matrix as stored
                bool matrix;       // a single matrix shared by every plane
            };

            /*
                dest = alpha op(lhs) op(rhs) + beta dest. A, B and dest are views of host memory
                described by their stored matrix sizes; M and K, which change with the batch and
                the sequence length, are bucketed.
            */
            bool matmul (
                float beta, tensor& dest, float alpha,
                const tensor& lhs, bool trans_lhs, plane_layout L,
                const tensor& rhs, bool trans_rhs, plane_layout R,
                plane_layout D,
                long planes,
                bool bucket_n
            )
            {
                const long M = trans_lhs ? L.cols : L.rows;
                const long K = trans_lhs ? L.rows : L.cols;
                const long N = trans_rhs ? R.rows : R.cols;
                if ((trans_rhs ? R.cols : R.rows) != K || D.rows != M || D.cols != N)
                    return false;
                const double work = static_cast<double>(M) * N * K * planes;
                if (work == 0 || !admit(OP_gemm, work))
                    return false;
                const int64_t Mp = bucket(M), Kp = bucket(K), Np = bucket_n ? bucket(N) : N;
                const int64_t ar = trans_lhs ? Kp : Mp, ac = trans_lhs ? Mp : Kp;
                const int64_t br = trans_rhs ? Np : Kp, bc = trans_rhs ? Kp : Np;
                const int64_t pa = L.matrix ? 1 : planes, pb = R.matrix ? 1 : planes;
                params p;
                p << pa << ar << ac << pb << br << bc << (trans_lhs ? 1 : 0) << (trans_rhs ? 1 : 0);
                const int target = product_target(K);
                return route(OP_gemm, &build_matmul, p, [&](call& c)
                {
                    geometry ga = dense(pa, 1, L.rows, L.cols, pa, 1, ar, ac);
                    geometry gb = dense(pb, 1, R.rows, R.cols, pb, 1, br, bc);
                    // Either operand may be a weight, kept on the device while it does not change.
                    input& a = c.add_input(in_kind::scaled, lhs.host(), ga);
                    a.target = target;
                    a.storage = &storage_access::of(lhs);
                    input& b = c.add_input(in_kind::scaled, rhs.host(), gb);
                    b.target = target;
                    b.storage = &storage_access::of(rhs);
                    output& o = c.add_output(beta == 0 ? out_kind::assign : out_kind::blend, dest.host(),
                        dense(planes, 1, M, N, planes, 1, Mp, Np));
                    o.alpha = alpha;
                    o.beta = beta;
                    c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
                });
            }

    // ----------------------------------------------------------------------------------------
    //                                     Softmax
    // ----------------------------------------------------------------------------------------

            // p: rows, axis length, inner, axis (1 = middle of [rows, n, inner])
            void build_softmax (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                g.output(g.softmax(x, 1));
            }

            // p: rows, axis length, inner
            void build_softmax_gradient (graph& g, const params& p)
            {
                const node d = g.input({p[0], p[1], p[2]});
                const node gi = g.input({p[0], p[1], p[2]});
                const node dot = g.reduce_sum(g.mul(d, gi), {1}, true);
                g.output(g.mul(d, g.sub(gi, dot)));
            }

            /*
                Softmax over the middle axis of a [outer, n, inner] view. PLANE_WISE inputs, the
                attention scores, may carry masks: -inf, or large negative values that the half
                exchange clamps to its lowest finite value, which leaves their probabilities at
                zero. A row whose values are all -inf comes out as zeros, as on the host.
            */
            bool softmax_view (
                int op,
                tensor& dest,
                const tensor& src,
                long outer,
                long n,
                long inner,
                bool masked_rows
            )
            {
                const size_t total = src.size();
                if (total == 0 || !admit(op, static_cast<double>(total)))
                    return false;
                // The axis is padded only in the masked layout, where inner == 1 and padding
                // with -inf contributes nothing.
                const int64_t po = bucket(outer);
                const int64_t pn = masked_rows ? bucket(n) : n;
                params p;
                p << po << pn << inner;
                std::vector<unsigned char> masked;
                return route(op, &build_softmax, p, [&](call& c)
                {
                    const geometry gm = view3(outer, n, inner, po, pn);
                    input& x = c.add_input(masked_rows ? in_kind::scores : in_kind::values, src.host(), gm,
                        masked_rows ? -std::numeric_limits<float>::infinity() : 0.0f);
                    x.shift_rows = inner == 1;
                    output& o = c.add_output(out_kind::assign, dest.host(), gm);
                    if (masked_rows)
                    {
                        // Filled while the scores are packed: the rows that are all -inf.
                        x.row_flags = &masked;
                        o.zero_rows = &masked;
                        o.unit_rows = true;
                    }
                });
            }

            bool softmax_gradient_view (
                int op,
                tensor& grad,
                const tensor& dest,
                const tensor& gradient_input,
                long outer,
                long n,
                long inner,
                bool pad_axis
            )
            {
                const size_t total = dest.size();
                if (total == 0 || !admit(op, static_cast<double>(total)))
                    return false;
                const int64_t po = bucket(outer);
                const int64_t pn = pad_axis ? bucket(n) : n;
                params p;
                p << po << pn << inner;
                const out_kind kind = accumulate_unless_same(grad, gradient_input);
                return route(op, &build_softmax_gradient, p, [&](call& c)
                {
                    const geometry gm = view3(outer, n, inner, po, pn);
                    c.add_input(in_kind::values, dest.host(), gm);
                    c.add_input(in_kind::scaled, gradient_input.host(), gm);
                    c.add_output(kind, grad.host(), gm);
                    c.finalize = [](call& c) { c.out[0].exp = -c.in[1].exp; return true; };
                });
            }

    // ----------------------------------------------------------------------------------------
    //                                  RMS normalization
    // ----------------------------------------------------------------------------------------

            /*
                RMS normalization over the middle axis of an [outer, n, inner] view, gamma being
                indexed by that axis. The input is scaled by 2^e and eps by 2^2e, which leaves the
                output unchanged and divides the stored inverse RMS by 2^e.
                p: outer, n, inner
            */
            void build_rms_normalize (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                const node gamma = g.input({1, p[1], 1});
                const node eps = g.scalar();
                const node inv = g.div(g.c(1), g.sqrt(g.add(g.reduce_mean(g.square(x), {1}, true), eps)));
                g.output(g.mul(g.mul(x, inv), gamma));
                g.output(inv);
            }

            /*
                p: outer, n, inner. Inputs gi (scaled), x, inv, gamma; outputs the gradient to
                add to src_grad, gamma_grad, and S = sum over the axis of dx * x_hat, from which
                the host derives dscale.
            */
            void build_rms_normalize_gradient (graph& g, const params& p)
            {
                const node gi = g.input({p[0], p[1], p[2]});
                const node x = g.input({p[0], p[1], p[2]});
                const node inv = g.input({p[0], 1, p[2]});
                const node gamma = g.input({1, p[1], 1});
                const node xhat = g.mul(x, inv);
                const node dx = g.mul(gi, gamma);
                // Means rather than sums, so that no reduction leaves the half range; the host
                // multiplies them back by their counts.
                const node M = g.reduce_mean(g.mul(dx, xhat), {1}, true);
                g.output(g.mul(inv, g.sub(dx, g.mul(xhat, M))));
                g.output(g.reduce_mean(g.mul(gi, xhat), {0, 2}, false));
                g.output(M);
            }

            bool rms_normalize_view (
                const double eps,
                resizable_tensor& dest,
                resizable_tensor& scale,
                const tensor& src,
                const tensor& gamma,
                long outer,
                long n,
                long inner
            )
            {
                const size_t total = src.size();
                if (total == 0 || !admit(OP_rms_normalize, static_cast<double>(total)))
                    return false;
                const int64_t po = bucket(outer);
                params p;
                p << po << n << inner;
                return route(OP_rms_normalize, &build_rms_normalize, p, [&](call& c)
                {
                    const geometry gx = view3(outer, n, inner, po, n);
                    const geometry gs = view3(outer, 1, inner, po, 1);
                    input& x = c.add_input(in_kind::scaled, src.host(), gx);
                    x.target = 6;
                    c.add_input(in_kind::values, gamma.host(), flat(static_cast<size_t>(n), static_cast<size_t>(n)));
                    c.add_scalar(static_cast<float>(eps));
                    c.add_output(out_kind::assign, dest.host(), gx);
                    c.add_output(out_kind::assign, scale.host(), gs);
                    c.finalize = [eps](call& c)
                    {
                        // eps * 2^2e must stay a normal half: lower e if it would not.
                        int e = c.in[0].exp;
                        while (e > 0 && eps * std::ldexp(1.0, 2 * e) > 1e4) --e;
                        c.in[0].exp = e;
                        c.in[2].value = static_cast<float>(eps * std::ldexp(1.0, 2 * e));
                        c.out[1].exp = e;
                        return true;
                    };
                });
            }

            bool rms_normalize_gradient_view (
                const tensor& gradient_input,
                const tensor& scale,
                const tensor& src,
                const tensor& gamma,
                tensor& src_grad,
                tensor& gamma_grad,
                resizable_tensor& dscale,
                long outer,
                long n,
                long inner
            )
            {
                const size_t total = src.size();
                if (total == 0 || !admit(OP_rms_normalize_gradient, static_cast<double>(total)))
                    return false;
                const int64_t po = bucket(outer);
                params p;
                p << po << n << inner;
                std::vector<float> S;
                const bool ok = route(OP_rms_normalize_gradient, &build_rms_normalize_gradient, p, [&](call& c)
                {
                    const geometry gx = view3(outer, n, inner, po, n);
                    const geometry gs = view3(outer, 1, inner, po, 1);
                    c.add_input(in_kind::scaled, gradient_input.host(), gx);
                    c.add_input(in_kind::values, src.host(), gx);
                    c.add_input(in_kind::values, scale.host(), gs);
                    c.add_input(in_kind::values, gamma.host(), flat(static_cast<size_t>(n), static_cast<size_t>(n)));
                    c.add_output(out_kind::add, src_grad.host(), gx);
                    c.add_output(out_kind::assign, gamma_grad.host(), dense(1, 1, 1, n)).alpha =
                        static_cast<float>(po * inner);
                    c.add_output(out_kind::staged, nullptr, gs).staged = &S;
                    c.out.back().alpha = static_cast<float>(n);
                    c.finalize = [](call& c)
                    {
                        for (auto& o : c.out) o.exp = -c.in[0].exp;
                        return true;
                    };
                });
                if (!ok)
                    return false;
                // dscale = -0.5 inv^2 S, where the host formula has -0.5 inv^3 sum(dx x).
                dscale.copy_size(scale);
                float* ds = dscale.host();
                const float* inv = scale.host();
                for (size_t i = 0; i < scale.size(); ++i)
                    ds[i] = -0.5f * inv[i] * inv[i] * S[i];
                return true;
            }

    // ----------------------------------------------------------------------------------------
    //                                  Cross entropy
    // ----------------------------------------------------------------------------------------

            /*
                p: rows, vocabulary. Outputs the probabilities, the row maxima and the row sums
                of exp(x - max), divided by sqrt(vocabulary) to stay within the half range.
            */
            void build_cross_entropy (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1]});
                const node m = g.reduce_max(x, {1}, true);
                // Each term is divided by sqrt(vocabulary) before the sum, which then stays
                // within the half range whatever the vocabulary.
                const node e = g.mul(g.exp(g.sub(x, m)), g.c(1.0 / std::sqrt(static_cast<double>(p[1]))));
                const node s = g.reduce_sum(e, {1}, true);
                g.output(g.div(e, s));
                g.output(m);
                g.output(s);
            }

            void print_report (
                std::ostream& out
            )
            {
                runtime& r = rt();
                const auto flags = out.flags();
                out << "dlib NPU routing\n";
                out << "  build    : DLIB_USE_NPU, OpenVINO " << ov::get_openvino_version().buildNumber
#ifdef DLIB_USE_CUDA
                    << ", primary target CUDA\n";
#else
                    << ", primary target host\n";
#endif
                if (!r.available)
                {
                    out << "  device   : none ("
                        << (r.why_unavailable.empty() ? std::string("shut down") : r.why_unavailable) << ")\n";
                    out.flags(flags);
                    return;
                }
                const int m = r.mode.load();
                out << "  device   : " << r.device << ", " << r.full_name << ", " << (r.half ? "f16" : "f32")
                    << " exchange, " << (r.single ? "single" : "half") << " precision arithmetic\n";
                out << "  routing  : " << (m == static_cast<int>(routing::always) ? "always" :
                                           m == static_cast<int>(routing::never) ? "never" : "automatic")
                    << ", " << (r.sync.load() ? "synchronous" : "background") << " compilation, cache "
                    << (r.cache_enabled ? r.cache_dir : std::string("off")) << "\n";
                size_t resident = 0, graphs = 0;
                {
                    std::lock_guard<std::mutex> lock(r.cache_mutex);
                    resident = r.resident;
                    for (auto& kv : r.cache)
                        if (kv.second->state.load() == ready) ++graphs;
                }
                out << "  memory   : " << resident / (1 << 20) << " MiB held by " << graphs << " graphs, budget "
                    << r.budget / (1 << 20) << " MiB\n";
                size_t operand_bytes = 0, operand_copies = 0;
                unsigned long long conversions = 0;
                {
                    std::lock_guard<std::mutex> lock(r.operands_mutex);
                    operand_bytes = r.operand_bytes;
                    operand_copies = r.operand_copies;
                    conversions = r.operand_conversions;
                }
                out << "  operands : " << operand_bytes / (1 << 20) << " MiB held by " << operand_copies
                    << " copies, written " << conversions << " times, budget " << r.operand_budget / (1 << 20)
                    << " MiB\n";
                out << "  " << std::left << std::setw(32) << "operation" << std::right
                    << std::setw(10) << "device" << std::setw(10) << "reused" << std::setw(10) << "host"
                    << std::setw(10) << "rejected" << std::setw(8) << "graphs" << std::setw(8) << "failed"
                    << std::setw(12) << "device ms" << std::setw(12) << "compile ms" << "\n";
                for (int op = 0; op < OP_COUNT; ++op)
                {
                    const op_counters& c = r.counters[op];
                    if (!c.touched.load()) continue;
                    out << "  " << std::left << std::setw(32) << op_table[op].name << std::right
                        << std::setw(10) << c.device_calls.load() << std::setw(10) << c.reused.load()
                        << std::setw(10) << c.host_calls.load()
                        << std::setw(10) << c.rejected.load() << std::setw(8) << c.graphs.load()
                        << std::setw(8) << c.failures.load()
                        << std::setw(12) << std::fixed << std::setprecision(1) << c.device_ns.load() / 1e6
                        << std::setw(12) << c.compile_ns.load() / 1e6 << "\n";
                }
                out.flags(flags);
            }

        } // anonymous namespace

    // ----------------------------------------------------------------------------------------
    //                                   Run-time control
    // ----------------------------------------------------------------------------------------

        bool is_compiled_in () { return true; }

        bool is_available () { return rt().available; }

        std::string device_name ()
        {
            runtime& r = rt();
            return r.available ? r.device + ", " + r.full_name : std::string();
        }

        bool uses_half_precision ()
        {
            runtime& r = rt();
            return r.available && !r.single;
        }

        std::string runtime_version ()
        {
            return ov::get_openvino_version().buildNumber;
        }

        void set_routing (routing mode) { rt().mode = static_cast<int>(mode); }

        routing get_routing () { return static_cast<routing>(rt().mode.load()); }

        void set_synchronous_compilation (bool synchronous) { rt().sync = synchronous; }

        exact_scope::exact_scope () { ++exact_depth; }

        exact_scope::~exact_scope () { --exact_depth; }

        std::vector<op_statistics> get_statistics ()
        {
            runtime& r = rt();
            std::vector<op_statistics> result;
            for (int op = 0; op < OP_COUNT; ++op)
            {
                const op_counters& c = r.counters[op];
                if (!c.touched.load()) continue;
                op_statistics s;
                s.name = op_table[op].name;
                s.device_calls = c.device_calls.load();
                s.host_calls = c.host_calls.load();
                s.rejected = c.rejected.load();
                s.graphs = c.graphs.load();
                s.compile_failures = c.failures.load();
                s.reused_calls = c.reused.load();
                s.device_seconds = c.device_ns.load() / 1e9;
                s.compile_seconds = c.compile_ns.load() / 1e9;
                result.push_back(s);
            }
            return result;
        }

        void reset_statistics ()
        {
            runtime& r = rt();
            for (auto& c : r.counters)
            {
                c.device_calls = 0; c.host_calls = 0; c.rejected = 0; c.graphs = 0; c.failures = 0;
                c.reused = 0; c.device_ns = 0; c.compile_ns = 0; c.touched = false;
            }
            {
                std::lock_guard<std::mutex> lock(r.operands_mutex);
                r.operand_conversions = 0;
            }
            std::lock_guard<std::mutex> lock(r.cache_mutex);
            for (auto& kv : r.cache)
            {
                entry& e = *kv.second;
                std::lock_guard<std::mutex> elock(e.calib);
                e.decision = undecided;
                e.device_samples = e.host_samples = 0;
                e.device_best = e.host_best = std::numeric_limits<long long>::max();
                e.rejects_in_row = 0;
            }
        }

        void print_status (std::ostream& out) { print_report(out); }

        cpu_measure::cpu_measure () : entry(pending_host), start(0)
        {
            if (entry)
            {
                pending_host = nullptr;
                start = now_ns();
            }
        }

        cpu_measure::~cpu_measure ()
        {
            if (entry)
                record_sample(*static_cast<impl_entry*>(entry), false, now_ns() - start);
        }

    // ----------------------------------------------------------------------------------------
    //                              Elementwise entry points
    // ----------------------------------------------------------------------------------------

        bool exp (tensor& dest, const tensor& src) { return unary(OP_exp, dest, src); }
        bool log (tensor& dest, const tensor& src) { return unary(OP_log, dest, src); }
        bool log10 (tensor& dest, const tensor& src) { return unary(OP_log10, dest, src); }

        bool inverse_sqrt (tensor& dest, const tensor& src, const double eps)
        {
            return unary(OP_inverse_sqrt, dest, src, {static_cast<float>(eps)});
        }

        bool threshold (tensor& data, float thresh)
        {
            return unary(OP_threshold, data, data, {thresh});
        }

        bool sigmoid (tensor& dest, const tensor& src) { return unary(OP_sigmoid, dest, src); }
        bool sigmoid_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input)
        {
            return gradient(OP_sigmoid_gradient, grad, dest, gradient_input);
        }

        bool mish (tensor& dest, const tensor& src) { return unary(OP_mish, dest, src); }
        bool mish_gradient (tensor& grad, const tensor& src, const tensor& gradient_input)
        {
            return gradient(OP_mish_gradient, grad, src, gradient_input);
        }

        bool relu (tensor& dest, const tensor& src) { return unary(OP_relu, dest, src); }
        bool relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input)
        {
            return gradient(OP_relu_gradient, grad, dest, gradient_input);
        }

        bool prelu (tensor& dest, const tensor& src, const tensor& param)
        {
            return unary(OP_prelu, dest, src, {param.host()[0]});
        }

        bool leaky_relu (tensor& dest, const tensor& src, const float alpha)
        {
            return unary(OP_leaky_relu, dest, src, {alpha});
        }

        bool leaky_relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input, const float alpha)
        {
            return gradient(OP_leaky_relu_gradient, grad, dest, gradient_input, {alpha});
        }

        bool tanh (tensor& dest, const tensor& src) { return unary(OP_tanh, dest, src); }
        bool tanh_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input)
        {
            return gradient(OP_tanh_gradient, grad, dest, gradient_input);
        }

        bool clipped_relu (tensor& dest, const tensor& src, const float ceiling)
        {
            return unary(OP_clipped_relu, dest, src, {ceiling});
        }

        bool clipped_relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input, const float ceiling)
        {
            return gradient(OP_clipped_relu_gradient, grad, dest, gradient_input, {ceiling});
        }

        bool elu (tensor& dest, const tensor& src, const float alpha)
        {
            return unary(OP_elu, dest, src, {alpha});
        }

        bool elu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input, const float alpha)
        {
            return gradient(OP_elu_gradient, grad, dest, gradient_input, {alpha});
        }

        bool gelu (tensor& dest, const tensor& src) { return unary(OP_gelu, dest, src); }
        bool gelu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input)
        {
            return gradient(OP_gelu_gradient, grad, src, gradient_input);
        }

        bool smelu (tensor& dest, const tensor& src, const float beta)
        {
            return unary(OP_smelu, dest, src, {beta});
        }

        bool smelu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input, const float beta)
        {
            return gradient(OP_smelu_gradient, grad, dest, gradient_input, {beta});
        }

        bool silu (tensor& dest, const tensor& src) { return unary(OP_silu, dest, src); }
        bool silu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input)
        {
            return gradient(OP_silu_gradient, grad, src, gradient_input);
        }

        bool affine_transform (tensor& dest, const tensor& src, const float A, const float B)
        {
            return unary(OP_affine_transform, dest, src, {A, B});
        }

        bool affine_transform (tensor& dest, const tensor& src, const float A)
        {
            return unary(OP_affine_transform, dest, src, {A, 0.0f});
        }

        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const float A, const float B, const float C)
        {
            if (dest.size() != src1.size() || dest.size() != src2.size()) return false;
            return nary(OP_affine_transform, NK_affine2, out_kind::assign, dest.host(),
                {src1.host(), src2.host()}, dest.size(), {A, B, C});
        }

        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const float A, const float B)
        {
            return affine_transform(dest, src1, src2, A, B, 0.0f);
        }

        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const tensor& src3, const float A, const float B, const float C, const float D)
        {
            if (dest.size() != src1.size() || dest.size() != src2.size() || dest.size() != src3.size()) return false;
            return nary(OP_affine_transform, NK_affine3, out_kind::assign, dest.host(),
                {src1.host(), src2.host(), src3.host()}, dest.size(), {A, B, C, D});
        }

        bool affine_transform_range (size_t begin, size_t end, tensor& dest, const tensor& src1,
            const tensor& src2, const tensor& src3, const float A, const float B, const float C)
        {
            if (dest.size() != src1.size() || dest.size() != src2.size() || dest.size() != src3.size()) return false;
            if (begin >= end || end > dest.size()) return false;
            return nary(OP_affine_transform, NK_affine3, out_kind::assign, dest.host() + begin,
                {src1.host() + begin, src2.host() + begin, src3.host() + begin}, end - begin, {A, B, C, 0.0f});
        }

        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const tensor& src3, const float A, const float B, const float C)
        {
            return affine_transform_range(0, dest.size(), dest, src1, src2, src3, A, B, C);
        }

        bool divide_by_sqrt (tensor& dest, const tensor& num, const tensor& sqnorm, const double eps)
        {
            if (dest.size() != num.size() || dest.size() != sqnorm.size()) return false;
            return nary(OP_divide_by_sqrt, NK_divide_by_sqrt, out_kind::assign, dest.host(),
                {num.host(), sqnorm.host()}, dest.size(), {static_cast<float>(eps)});
        }

    // ----------------------------------------------------------------------------------------
    //                           Products, softmax, normalization
    // ----------------------------------------------------------------------------------------

        bool gemm (float beta, tensor& dest, float alpha, const tensor& lhs, bool trans_lhs,
            const tensor& rhs, bool trans_rhs, operation_mode mode)
        {
            if (lhs.size() == 0 || rhs.size() == 0 || dest.size() == 0)
                return false;
            if (mode == operation_mode::CHANNEL_WISE)
            {
                const plane_layout L{static_cast<long>(lhs.num_samples()),
                    static_cast<long>(lhs.size() / lhs.num_samples()), true};
                const plane_layout R{static_cast<long>(rhs.num_samples()),
                    static_cast<long>(rhs.size() / rhs.num_samples()), true};
                const plane_layout D{static_cast<long>(dest.num_samples()),
                    static_cast<long>(dest.size() / dest.num_samples()), true};
                return matmul(beta, dest, alpha, lhs, trans_lhs, L, rhs, trans_rhs, R, D, 1, false);
            }

            // The disambiguation of tt::gemm for PLANE_WISE operands, reproduced as is.
            auto plane_batched = [](const tensor& t) {
                return t.num_samples() * t.k() > 1 && t.nr() * t.nc() > 1;
            };
            auto shares_batch = [&](const tensor& t) {
                auto same = [&](const tensor& o) {
                    return plane_batched(o) && o.num_samples() == t.num_samples() && o.k() == t.k();
                };
                return same(lhs) || same(rhs) || same(dest);
            };
            auto is_matrix = [&](const tensor& t) {
                if (t.num_samples() * t.k() == 1) return t.nr() * t.nc() > 1;
                return t.nr() * t.nc() == 1 && !shares_batch(t);
            };
            long num_samples = static_cast<long>(std::min({lhs.num_samples(), rhs.num_samples(), dest.num_samples()}));
            long num_channels = static_cast<long>(std::min({lhs.k(), rhs.k(), dest.k()}));
            const bool lm = is_matrix(lhs), rm = is_matrix(rhs), dm = is_matrix(dest);
            if (lm && rm && dm)
                num_samples = num_channels = 1;
            const long planes = num_samples * num_channels;
            if (dm && planes > 1)
                return false;
            auto layout = [](const tensor& t, bool m) {
                plane_layout l;
                l.rows = static_cast<long>((m && t.num_samples() > 1) ? t.num_samples() : t.nr());
                l.cols = static_cast<long>((m && t.k() > 1) ? t.k() : t.nc());
                l.matrix = m;
                return l;
            };
            return matmul(beta, dest, alpha, lhs, trans_lhs, layout(lhs, lm), rhs, trans_rhs, layout(rhs, rm),
                layout(dest, dm), planes, true);
        }

        bool softmax (tensor& dest, const tensor& src, operation_mode mode)
        {
            if (dest.size() != src.size() || src.size() == 0)
                return false;
            if (mode == operation_mode::CHANNEL_WISE)
                return softmax_view(OP_softmax, dest, src, static_cast<long>(src.num_samples()),
                    static_cast<long>(src.k()), static_cast<long>(src.nr() * src.nc()), false);
            return softmax_view(OP_softmax, dest, src, static_cast<long>(src.num_samples() * src.k() * src.nr()),
                static_cast<long>(src.nc()), 1, true);
        }

        bool softmax_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input, operation_mode mode)
        {
            if (grad.size() != dest.size() || gradient_input.size() != dest.size() || dest.size() == 0)
                return false;
            if (mode == operation_mode::CHANNEL_WISE)
                return softmax_gradient_view(OP_softmax_gradient, grad, dest, gradient_input,
                    static_cast<long>(dest.num_samples()), static_cast<long>(dest.k()),
                    static_cast<long>(dest.nr() * dest.nc()), false);
            return softmax_gradient_view(OP_softmax_gradient, grad, dest, gradient_input,
                static_cast<long>(dest.num_samples() * dest.k() * dest.nr()), static_cast<long>(dest.nc()), 1, true);
        }

        bool softmax_all (tensor& dest, const tensor& src)
        {
            if (dest.size() != src.size() || src.size() == 0)
                return false;
            return softmax_view(OP_softmax_all, dest, src, static_cast<long>(src.num_samples()),
                static_cast<long>(src.size() / src.num_samples()), 1, false);
        }

        bool softmax_all_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input)
        {
            if (grad.size() != dest.size() || gradient_input.size() != dest.size() || dest.size() == 0)
                return false;
            return softmax_gradient_view(OP_softmax_all_gradient, grad, dest, gradient_input,
                static_cast<long>(dest.num_samples()), static_cast<long>(dest.size() / dest.num_samples()), 1, false);
        }

        bool rms_normalize (const double eps, resizable_tensor& dest, resizable_tensor& scale,
            const tensor& src, const tensor& gamma, bool axis_is_nc)
        {
            const long ns = static_cast<long>(src.num_samples()), ks = static_cast<long>(src.k());
            const long nrs = static_cast<long>(src.nr()), ncs = static_cast<long>(src.nc());
            if (src.size() == 0 || eps <= 0)
                return false;
            if (axis_is_nc)
            {
                if (gamma.size() != static_cast<size_t>(ncs)) return false;
                if (!admit(OP_rms_normalize, static_cast<double>(src.size()))) return false;
                dest.copy_size(src);
                scale.set_size(ns, ks, nrs, 1);
                return rms_normalize_view(eps, dest, scale, src, gamma, ns * ks * nrs, ncs, 1);
            }
            if (gamma.size() != static_cast<size_t>(ks)) return false;
            if (!admit(OP_rms_normalize, static_cast<double>(src.size()))) return false;
            dest.copy_size(src);
            scale.set_size(ns, 1, nrs, ncs);
            return rms_normalize_view(eps, dest, scale, src, gamma, ns, ks, nrs * ncs);
        }

        bool rms_normalize_gradient (const tensor& gradient_input, const tensor& scale,
            const tensor& src, const tensor& gamma, tensor& src_grad, tensor& gamma_grad,
            resizable_tensor& dscale, bool axis_is_nc)
        {
            const long ns = static_cast<long>(src.num_samples()), ks = static_cast<long>(src.k());
            const long nrs = static_cast<long>(src.nr()), ncs = static_cast<long>(src.nc());
            if (src.size() == 0 || gradient_input.size() != src.size() || src_grad.size() != src.size())
                return false;
            if (axis_is_nc)
            {
                if (gamma.size() != static_cast<size_t>(ncs) || gamma_grad.size() != gamma.size()) return false;
                return rms_normalize_gradient_view(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale,
                    ns * ks * nrs, ncs, 1);
            }
            if (gamma.size() != static_cast<size_t>(ks) || gamma_grad.size() != gamma.size()) return false;
            return rms_normalize_gradient_view(gradient_input, scale, src, gamma, src_grad, gamma_grad, dscale,
                ns, ks, nrs * ncs);
        }

        bool cross_entropy_rows (const tensor& logits, tensor& grad,
            const std::vector<long>& rows, const std::vector<unsigned long>& targets,
            double label_smoothing, double z_loss_weight, double& loss)
        {
            const long V = static_cast<long>(logits.nc());
            const size_t count = rows.size();
            if (count == 0 || logits.k() != 1 || targets.size() != count || grad.size() != logits.size())
                return false;
            if (!admit(OP_cross_entropy, static_cast<double>(count) * V))
                return false;
            const int64_t rp = bucket(static_cast<int64_t>(count));
            params p;
            p << rp << V;
            std::vector<float> maxima, sums, shifts;
            const bool ok = route(OP_cross_entropy, &build_cross_entropy, p, [&](call& c)
            {
                geometry gx = dense(1, 1, static_cast<int64_t>(count), V, 1, 1, rp, V);
                gx.rows = rows.data();
                gx.row_count = count;
                input& x = c.add_input(in_kind::values, logits.host(), gx);
                x.shift_rows = true;
                x.row_shift = &shifts;
                c.add_output(out_kind::assign, grad.host(), gx).unit_rows = true;
                const geometry gs = flat(count, static_cast<size_t>(rp));
                c.add_output(out_kind::staged, nullptr, gs).staged = &maxima;
                c.add_output(out_kind::staged, nullptr, gs).staged = &sums;
            });
            if (!ok)
                return false;

            // The device left the probabilities of each scored row in grad; the host finishes
            // what compute_loss_cross_entropy_per_token computes from them.
            // The log-probabilities come from the single precision logits and the row
            // log-sum-exp, log p = x - lse, floored at log(1e-10) as on the host, rather than
            // from the half precision probabilities.
            float* g = grad.host();
            const float* x = logits.host();
            const float smooth_target = (label_smoothing > 0) ? 1.0f - static_cast<float>(label_smoothing) : 1.0f;
            const float smooth_other = (label_smoothing > 0) ? static_cast<float>(label_smoothing / (V - 1)) : 0.0f;
            const float zw = static_cast<float>(z_loss_weight);
            const float inv_valid = 1.0f / count;
            const float root = std::sqrt(static_cast<float>(V));
            const float floor_log = std::log(1e-10f);
            // One partial sum per block of rows, added in order, so that the loss does not
            // depend on the scheduling of the threads.
            const size_t grain = std::max<size_t>(1, (size_t(1) << 16) / static_cast<size_t>(std::max<long>(V, 1)));
            std::vector<double> partial((count + grain - 1) / grain, 0.0);
            for_blocks(count, grain, [&](size_t b, size_t e)
            {
                double local = 0;
                for (size_t i = b; i < e; ++i)
                {
                    float* row = g + static_cast<size_t>(rows[i]) * V;
                    const float* logit = x + static_cast<size_t>(rows[i]) * V;
                    const unsigned long target = targets[i];
                    const float shift = shifts.empty() ? 0.0f : shifts[i];
                    const float lse = shift + maxima[i] + std::log(sums[i] * root);
                    const float zs = (zw > 0.0f) ? 2.0f * zw * lse : 0.0f;
                    if (smooth_other > 0)
                    {
                        for (long c = 0; c < V; ++c)
                        {
                            const float q = (static_cast<unsigned long>(c) == target) ? smooth_target : smooth_other;
                            local -= q * std::max(logit[c] - lse, floor_log);
                            row[c] = (row[c] - q + zs * row[c]) * inv_valid;
                        }
                    }
                    else
                    {
                        local -= smooth_target * std::max(logit[target] - lse, floor_log);
                        const float s = (1.0f + zs) * inv_valid;
                        for (long c = 0; c < V; ++c)
                            row[c] *= s;
                        row[target] -= smooth_target * inv_valid;
                    }
                    if (zw > 0.0f)
                        local += zw * lse * lse;
                }
                partial[b / grain] += local;
            });
            double total = 0;
            for (double v : partial)
                total += v;
            loss = total * inv_valid;
            return true;
        }

    // ----------------------------------------------------------------------------------------
    //                       Rows, columns, broadcasts and reductions
    // ----------------------------------------------------------------------------------------

        namespace
        {
            /*
                Products of an [rows, cols] matrix with a vector broadcast along one side.
                p: rows, cols, kind (0: scale_columns, 1: scale_rows)
            */
            void build_scale_matrix (graph& g, const params& p)
            {
                const node m = g.input({p[0], p[1]});
                const node v = (p[2] == 0) ? g.input({1, p[1]}) : g.input({p[0], 1});
                g.output(g.mul(m, v));
            }

            // p: rows, cols. out = (m1 - m2 v1) v2, v1 and v2 being row vectors.
            void build_scale_rows2 (graph& g, const params& p)
            {
                const node m1 = g.input({p[0], p[1]});
                const node m2 = g.input({p[0], p[1]});
                const node v1 = g.input({p[0], 1});
                const node v2 = g.input({p[0], 1});
                g.output(g.mul(g.sub(m1, g.mul(m2, v1)), v2));
            }

            /*
                Row reductions of an [rows, cols] matrix, as means that the host multiplies by
                cols. p: rows, cols, kind (0: sum of squares then rsqrt with eps, 1: dot of two)
            */
            void build_row_reduce (graph& g, const params& p)
            {
                if (p[2] == 0)
                {
                    const node x = g.input({p[0], p[1]});
                    const node eps = g.scalar();
                    g.output(g.div(g.c(1), g.sqrt(g.add(g.reduce_mean(g.square(x), {1}, false), eps))));
                }
                else
                {
                    const node a = g.input({p[0], p[1]});
                    const node b = g.input({p[0], p[1]});
                    g.output(g.reduce_mean(g.mul(a, b), {1}, false));
                }
            }

            /*
                a * b with numpy broadcasting over the leading axis, optionally summed over it.
                p: rows of a, rows of b, rows of the product, cols, sum over the rows
            */
            void build_broadcast_multiply (graph& g, const params& p)
            {
                const node a = g.input({p[0], p[3]});
                const node b = g.input({p[1], p[3]});
                node y = g.mul(a, b);
                if (p[4] != 0)
                    y = g.reduce_sum(y, {0}, false);
                g.output(y);
            }

            /*
                Per channel products and sums over an [n, k, inner] view.
                p: n, k, inner, kind (0: x * s[k], 1: sum over n and inner of a * b,
                2: x * s[n, k], 3: sum over n and inner of x as a mean)
            */
            void build_channels (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                switch (p[3])
                {
                    case 0: g.output(g.mul(x, g.input({1, p[1], 1}))); break;
                    case 1: g.output(g.reduce_mean(g.mul(x, g.input({p[0], p[1], p[2]})), {0, 2}, false)); break;
                    case 2: g.output(g.mul(x, g.input({p[0], p[1], 1}))); break;
                    case 3: g.output(g.reduce_mean(x, {0, 2}, false)); break;
                    default: throw std::logic_error("unknown channel operation");
                }
            }

            // p: dims of src (4), dims of dest (4). dest = broadcast(src)
            void build_broadcast (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2], p[3]});
                g.output(g.broadcast(x, {p[4], p[5], p[6], p[7]}));
            }

            // p: rows, cols, broadcast of A and B over the rows. dest = A x + B
            void build_affine_tensors (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1]});
                const int64_t ar = p[2] ? 1 : p[0];
                const node a = g.input({ar, p[1]});
                const node b = g.input({ar, p[1]});
                g.output(g.add(g.mul(a, x), b));
            }

            // p: n, k, inner. dest = A[k] x + B[k]
            void build_affine_channels (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                const node a = g.input({1, p[1], 1});
                const node b = g.input({1, p[1], 1});
                g.output(g.add(g.mul(a, x), b));
            }

            bool scale_matrix (int kind, tensor& out, const tensor& m, const tensor& v)
            {
                const int op = kind == 0 ? OP_scale_columns : OP_scale_rows;
                const int64_t rows = sz(m.num_samples());
                if (m.size() == 0 || rows == 0 || out.size() != m.size() || !admit(op, static_cast<double>(m.size())) ||
                    rounds_in_place(out.host(), {m.host(), v.host()}))
                    return false;
                const int64_t cols = static_cast<int64_t>(m.size()) / rows;
                if ((kind == 0 && static_cast<int64_t>(v.size()) != cols) ||
                    (kind == 1 && static_cast<int64_t>(v.size()) != rows))
                    return false;
                const int64_t pr = bucket(rows);
                params p;
                p << pr << cols << kind;
                return route(op, &build_scale_matrix, p, [&](call& c)
                {
                    const geometry gm = dense(1, 1, rows, cols, 1, 1, pr, cols);
                    c.add_input(in_kind::scaled, m.host(), gm);
                    const geometry gv = kind == 0 ? flat(static_cast<size_t>(cols), static_cast<size_t>(cols))
                                                  : flat(static_cast<size_t>(rows), static_cast<size_t>(pr));
                    c.add_input(in_kind::scaled, v.host(), gv);
                    c.add_output(out_kind::assign, out.host(), gm);
                    c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
                });
            }

            /*
                The zero padded corner of t inside a tensor of the dimensions of d, as the
                bounds checked loops of add() and multiply_zero_padded() read it.
            */
            geometry corner_of (const tensor& t, const tensor& d)
            {
                geometry g;
                g.v[0] = std::min(sz(t.num_samples()), sz(d.num_samples()));
                g.v[1] = std::min(sz(t.k()), sz(d.k()));
                g.v[2] = std::min(sz(t.nr()), sz(d.nr()));
                g.v[3] = std::min(sz(t.nc()), sz(d.nc()));
                g.p[0] = sz(d.num_samples()); g.p[1] = sz(d.k()); g.p[2] = sz(d.nr()); g.p[3] = sz(d.nc());
                g.s[0] = sz(t.k() * t.nr() * t.nc()); g.s[1] = sz(t.nr() * t.nc()); g.s[2] = sz(t.nc()); g.s[3] = 1;
                return g;
            }

            bool zero_padded (int op, int kind, out_kind mode, tensor& dest, const tensor& src1, const tensor& src2)
            {
                if (dest.size() == 0 || !admit(op, static_cast<double>(dest.size())))
                    return false;
                if (mode == out_kind::assign && rounds_in_place(dest.host(), {src1.host(), src2.host()}))
                    return false;
                const int64_t rows = sz(dest.num_samples() * dest.k() * dest.nr());
                const int64_t cols = sz(dest.nc());
                params p;
                p << kind << rows << cols << 2 << 0;
                const bool product = kind == NK_multiply;
                return route(op, &build_nary, p, [&](call& c)
                {
                    c.add_input(product ? in_kind::scaled : in_kind::values, src1.host(), corner_of(src1, dest));
                    c.add_input(product ? in_kind::scaled : in_kind::values, src2.host(), corner_of(src2, dest));
                    c.add_output(mode, dest.host(),
                        dense(sz(dest.num_samples()), sz(dest.k()), sz(dest.nr()), sz(dest.nc())));
                    if (product)
                        c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
                });
            }
        }

        bool inverse_norms (resizable_tensor& invnorms, const tensor& data, const double eps)
        {
            const int64_t rows = sz(data.num_samples());
            if (data.size() == 0 || rows == 0 || !admit(OP_inverse_norms, static_cast<double>(data.size())))
                return false;
            const int64_t cols = static_cast<int64_t>(data.size()) / rows;
            const int64_t pr = bucket(rows);
            params p;
            p << pr << cols << 0;
            if (!(invnorms.num_samples() == rows && invnorms.size() == static_cast<size_t>(rows)))
                invnorms.set_size(rows, 1);
            return route(OP_inverse_norms, &build_row_reduce, p, [&](call& c)
            {
                input& x = c.add_input(in_kind::scaled, data.host(), dense(1, 1, rows, cols, 1, 1, pr, cols));
                x.target = 6;
                c.add_scalar(0);
                // 1/sqrt(sum + eps) = 1/sqrt(cols) * 1/sqrt(mean + eps/cols)
                c.add_output(out_kind::assign, invnorms.host(),
                    flat(static_cast<size_t>(rows), static_cast<size_t>(pr))).alpha =
                    1.0f / std::sqrt(static_cast<float>(cols));
                c.finalize = [eps, cols](call& c)
                {
                    int e = c.in[0].exp;
                    while (e > 0 && eps / cols * std::ldexp(1.0, 2 * e) > 1e4) --e;
                    c.in[0].exp = e;
                    c.in[1].value = static_cast<float>(eps / cols * std::ldexp(1.0, 2 * e));
                    c.out[0].exp = e;
                    return true;
                };
            });
        }

        bool dot_prods (bool add_to, tensor& out, const tensor& lhs, const tensor& rhs)
        {
            const int64_t rows = sz(lhs.num_samples());
            if (lhs.size() == 0 || rows == 0 || lhs.size() != rhs.size() || out.size() != static_cast<size_t>(rows) ||
                !admit(OP_dot_prods, static_cast<double>(lhs.size())))
                return false;
            const int64_t cols = static_cast<int64_t>(lhs.size()) / rows;
            const int64_t pr = bucket(rows);
            params p;
            p << pr << cols << 1;
            return route(OP_dot_prods, &build_row_reduce, p, [&](call& c)
            {
                const geometry gm = dense(1, 1, rows, cols, 1, 1, pr, cols);
                c.add_input(in_kind::scaled, lhs.host(), gm);
                c.add_input(in_kind::scaled, rhs.host(), gm);
                c.add_output(add_to ? out_kind::add : out_kind::assign, out.host(),
                    flat(static_cast<size_t>(rows), static_cast<size_t>(pr))).alpha = static_cast<float>(cols);
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool dot_prods (resizable_tensor& out, const tensor& lhs, const tensor& rhs)
        {
            const int64_t rows = sz(lhs.num_samples());
            if (lhs.size() == 0 || rows == 0 || !admit(OP_dot_prods, static_cast<double>(lhs.size())))
                return false;
            if (!(out.num_samples() == rows && out.size() == static_cast<size_t>(rows)))
                out.set_size(rows, 1);
            return dot_prods(false, out, lhs, rhs);
        }

        bool scale_columns (tensor& out, const tensor& m, const tensor& v) { return scale_matrix(0, out, m, v); }
        bool scale_rows (tensor& out, const tensor& m, const tensor& v) { return scale_matrix(1, out, m, v); }

        bool scale_rows2 (float beta, tensor& out, const tensor& m1, const tensor& m2,
            const tensor& v1, const tensor& v2)
        {
            const int64_t rows = sz(m1.num_samples());
            if (m1.size() == 0 || rows == 0 || m2.size() != m1.size() || out.size() != m1.size() ||
                v1.size() != static_cast<size_t>(rows) || v2.size() != static_cast<size_t>(rows) ||
                !admit(OP_scale_rows2, static_cast<double>(m1.size())))
                return false;
            if (beta == 0 && rounds_in_place(out.host(), {m1.host(), m2.host(), v1.host(), v2.host()}))
                return false;
            const int64_t cols = static_cast<int64_t>(m1.size()) / rows;
            const int64_t pr = bucket(rows);
            params p;
            p << pr << cols;
            return route(OP_scale_rows2, &build_scale_rows2, p, [&](call& c)
            {
                const geometry gm = dense(1, 1, rows, cols, 1, 1, pr, cols);
                const geometry gv = flat(static_cast<size_t>(rows), static_cast<size_t>(pr));
                c.add_input(in_kind::values, m1.host(), gm);
                c.add_input(in_kind::values, m2.host(), gm);
                c.add_input(in_kind::values, v1.host(), gv);
                c.add_input(in_kind::values, v2.host(), gv);
                output& o = c.add_output(beta == 0 ? out_kind::assign : out_kind::blend, out.host(), gm);
                o.beta = beta;
            });
        }

        bool multiply (bool add_to, tensor& dest, const tensor& src1, const tensor& src2)
        {
            if (dest.size() == 0)
                return false;
            if (dest.size() == src1.size() && src1.size() == src2.size())
                return nary(OP_multiply, NK_multiply, add_to ? out_kind::add : out_kind::assign, dest.host(),
                    {src1.host(), src2.host()}, dest.size());
            // Broadcasting over the samples, with a sum over them when dest holds one sample.
            const int64_t md = std::max({sz(dest.num_samples()), sz(src1.num_samples()), sz(src2.num_samples())});
            const int64_t cols = static_cast<int64_t>(dest.size()) / sz(dest.num_samples());
            if (!admit(OP_multiply, static_cast<double>(md * cols)))
                return false;
            if (!add_to && rounds_in_place(dest.host(), {src1.host(), src2.host()}))
                return false;
            const bool reduce = dest.num_samples() == 1 && md > 1;
            const int64_t pm = bucket(md);
            const int64_t r1 = src1.num_samples() == 1 ? 1 : pm, r2 = src2.num_samples() == 1 ? 1 : pm;
            params p;
            p << r1 << r2 << pm << cols << (reduce ? 1 : 0);
            return route(OP_multiply, &build_broadcast_multiply, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src1.host(),
                    dense(1, 1, sz(src1.num_samples()), cols, 1, 1, r1, cols)).target =
                    product_target(md);
                c.add_input(in_kind::scaled, src2.host(),
                    dense(1, 1, sz(src2.num_samples()), cols, 1, 1, r2, cols)).target =
                    product_target(md);
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(),
                    reduce ? flat(static_cast<size_t>(cols), static_cast<size_t>(cols))
                           : dense(1, 1, md, cols, 1, 1, pm, cols));
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool scale_channels (bool add_to, tensor& dest, const tensor& src, const tensor& scales)
        {
            if (src.size() == 0 || dest.size() != src.size() ||
                scales.size() != static_cast<size_t>(src.num_samples() * src.k()) ||
                !admit(OP_scale_channels, static_cast<double>(src.size())))
                return false;
            if (!add_to && rounds_in_place(dest.host(), {src.host(), scales.host()}))
                return false;
            const int64_t n = sz(src.num_samples()), k = sz(src.k()), inner = sz(src.nr() * src.nc());
            const int64_t pn = bucket(n);
            params p;
            p << pn << k << inner << 2;
            return route(OP_scale_channels, &build_channels, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src.host(), view3(n, k, inner, pn, k));
                c.add_input(in_kind::scaled, scales.host(), view3(n, k, 1, pn, k));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(), view3(n, k, inner, pn, k));
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool multiply_conv (bool add_to, tensor& dest, const tensor& src1, const tensor& src2)
        {
            if (src1.size() == 0 || !admit(OP_multiply_conv, static_cast<double>(src1.size())))
                return false;
            if (!add_to && rounds_in_place(dest.host(), {src1.host(), src2.host()}))
                return false;
            const int64_t n = sz(src1.num_samples()), k = sz(src1.k()), inner = sz(src1.nr() * src1.nc());
            if (have_same_dimensions(dest, src1))
            {
                if (src2.size() != static_cast<size_t>(k)) return false;
                const int64_t pn = bucket(n);
                params p;
                p << pn << k << inner << 0;
                return route(OP_multiply_conv, &build_channels, p, [&](call& c)
                {
                    c.add_input(in_kind::scaled, src1.host(), view3(n, k, inner, pn, k));
                    c.add_input(in_kind::scaled, src2.host(), flat(static_cast<size_t>(k), static_cast<size_t>(k)));
                    c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(), view3(n, k, inner, pn, k));
                    c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
                });
            }
            if (!have_same_dimensions(src1, src2) || dest.size() != static_cast<size_t>(k))
                return false;
            params p;
            p << n << k << inner << 1;
            return route(OP_multiply_conv, &build_channels, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src1.host(), view3(n, k, inner, n, k)).target = product_target(n * inner);
                c.add_input(in_kind::scaled, src2.host(), view3(n, k, inner, n, k)).target = product_target(n * inner);
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(),
                    flat(static_cast<size_t>(k), static_cast<size_t>(k))).alpha = static_cast<float>(n * inner);
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool multiply_zero_padded (bool add_to, tensor& dest, const tensor& src1, const tensor& src2)
        {
            return zero_padded(OP_multiply_zero_padded, NK_multiply, add_to ? out_kind::add : out_kind::assign,
                dest, src1, src2);
        }

        bool add (tensor& dest, const tensor& src1, const tensor& src2)
        {
            return zero_padded(OP_add, NK_add, out_kind::assign, dest, src1, src2);
        }

        bool add (float beta, tensor& dest, float alpha, const tensor& src)
        {
            if (dest.size() == 0 || src.size() == 0 || (beta == 0 && alpha == 0) || is_same_object(src, dest) ||
                !admit(OP_add, static_cast<double>(dest.size())))
                return false;
            const bool shape_ok = have_same_dimensions(src, dest) ||
                (src.num_samples() == 1 && src.k() == dest.k() && src.nr() == 1 && src.nc() == 1) ||
                (src.num_samples() == 1 && src.k() == dest.k() && src.nr() == dest.nr() && src.nc() == dest.nc()) ||
                (src.num_samples() == 1 && src.k() == 1 && src.nr() == dest.nr() && src.nc() == dest.nc()) ||
                (src.num_samples() == dest.num_samples() && src.k() == 1 && src.nr() == 1 && src.nc() == 1);
            if (!shape_ok)
                return false;
            params p;
            p << sz(src.num_samples()) << sz(src.k()) << sz(src.nr()) << sz(src.nc())
              << sz(dest.num_samples()) << sz(dest.k()) << sz(dest.nr()) << sz(dest.nc());
            return route(OP_add, &build_broadcast, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src.host(),
                    dense(sz(src.num_samples()), sz(src.k()), sz(src.nr()), sz(src.nc())));
                output& o = c.add_output(beta == 0 ? out_kind::assign : out_kind::blend, dest.host(),
                    dense(sz(dest.num_samples()), sz(dest.k()), sz(dest.nr()), sz(dest.nc())));
                o.alpha = alpha;
                o.beta = beta;
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

        bool affine_transform (const rectangle& rect, tensor& dest, const tensor& src1,
            const tensor& src2, const tensor& src3, float A, float B, float C)
        {
            if (dest.size() == 0 || rect.is_empty() || dest.size() != src1.size() || dest.size() != src2.size() ||
                dest.size() != src3.size())
                return false;
            const int64_t ld = static_cast<int64_t>(dest.size()) / sz(dest.num_samples());
            const int64_t h = rect.height(), w = rect.width();
            if (!admit(OP_affine_transform, static_cast<double>(h * w)) ||
                rounds_in_place(dest.host(), {src1.host(), src2.host(), src3.host()}))
                return false;
            const int64_t ph = bucket(h);
            params p;
            p << NK_affine3 << ph << w << 3 << 4;
            const size_t offset = static_cast<size_t>(rect.top() * ld + rect.left());
            return route(OP_affine_transform, &build_nary, p, [&](call& c)
            {
                const geometry gm = matrix(h, w, ld, ph, w);
                c.add_input(in_kind::values, src1.host() + offset, gm);
                c.add_input(in_kind::values, src2.host() + offset, gm);
                c.add_input(in_kind::values, src3.host() + offset, gm);
                c.add_scalar(A);
                c.add_scalar(B);
                c.add_scalar(C);
                c.add_scalar(0);
                c.add_output(out_kind::assign, dest.host() + offset, gm);
            });
        }

        bool affine_transform (tensor& dest, const tensor& src, const tensor& A, const tensor& B)
        {
            if (src.size() == 0 || !have_same_dimensions(dest, src) || !have_same_dimensions(A, B) ||
                !admit(OP_affine_transform, static_cast<double>(src.size())) ||
                rounds_in_place(dest.host(), {src.host(), A.host(), B.host()}))
                return false;
            const int64_t rows = sz(src.num_samples()), cols = static_cast<int64_t>(src.size()) / rows;
            const bool broadcast = A.num_samples() == 1;
            if (static_cast<int64_t>(A.size()) != (broadcast ? cols : rows * cols))
                return false;
            const int64_t pr = bucket(rows);
            params p;
            p << pr << cols << (broadcast ? 1 : 0);
            return route(OP_affine_transform, &build_affine_tensors, p, [&](call& c)
            {
                const geometry gm = dense(1, 1, rows, cols, 1, 1, pr, cols);
                const geometry ga = broadcast ? flat(static_cast<size_t>(cols), static_cast<size_t>(cols)) : gm;
                c.add_input(in_kind::values, src.host(), gm);
                c.add_input(in_kind::values, A.host(), ga);
                c.add_input(in_kind::values, B.host(), ga);
                c.add_output(out_kind::assign, dest.host(), gm);
            });
        }

        bool affine_transform_conv (tensor& dest, const tensor& src, const tensor& A, const tensor& B)
        {
            if (src.size() == 0 || !have_same_dimensions(dest, src) || A.size() != static_cast<size_t>(src.k()) ||
                B.size() != A.size() || !admit(OP_affine_transform_conv, static_cast<double>(src.size())) ||
                rounds_in_place(dest.host(), {src.host(), A.host(), B.host()}))
                return false;
            const int64_t n = sz(src.num_samples()), k = sz(src.k()), inner = sz(src.nr() * src.nc());
            const int64_t pn = bucket(n);
            params p;
            p << pn << k << inner;
            return route(OP_affine_transform_conv, &build_affine_channels, p, [&](call& c)
            {
                const geometry gx = view3(n, k, inner, pn, k);
                const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                c.add_input(in_kind::values, src.host(), gx);
                c.add_input(in_kind::values, A.host(), gk);
                c.add_input(in_kind::values, B.host(), gk);
                c.add_output(out_kind::assign, dest.host(), gx);
            });
        }

        bool dot (const tensor& a, const tensor& b, tensor& result, size_t idx)
        {
            const size_t n = a.size();
            if (n == 0 || b.size() != n || idx >= result.size() || !admit(OP_dot, static_cast<double>(n)))
                return false;
            const int64_t rows = flat_rows(n);
            const size_t padded = static_cast<size_t>(rows * flat_width);
            params p;
            p << rows << flat_width << 1;
            std::vector<float> value;
            const bool ok = route(OP_dot, &build_row_reduce, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, a.host(), flat(n, padded)).target =
                    product_target(static_cast<int64_t>(n));
                c.add_input(in_kind::scaled, b.host(), flat(n, padded)).target =
                    product_target(static_cast<int64_t>(n));
                output& o = c.add_output(out_kind::staged, nullptr,
                    flat(static_cast<size_t>(rows), static_cast<size_t>(rows)));
                o.staged = &value;
                o.alpha = static_cast<float>(flat_width);
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
            if (!ok)
                return false;
            double sum = 0;
            for (float v : value) sum += v;
            result.host()[idx] += static_cast<float>(sum);
            return true;
        }

        bool assign_bias_gradient (tensor& grad, const tensor& gradient_input)
        {
            if (gradient_input.size() == 0 || grad.num_samples() != 1 ||
                grad.size() * gradient_input.num_samples() != gradient_input.size() ||
                !admit(OP_assign_bias_gradient, static_cast<double>(gradient_input.size())))
                return false;
            const int64_t n = sz(gradient_input.num_samples()), cols = static_cast<int64_t>(grad.size());
            const int64_t pn = bucket(n);
            params p;
            p << pn << cols << 1 << 3;
            return route(OP_assign_bias_gradient, &build_channels, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, gradient_input.host(), dense(1, 1, n, cols, 1, 1, pn, cols));
                c.add_output(out_kind::assign, grad.host(),
                    flat(static_cast<size_t>(cols), static_cast<size_t>(cols))).alpha =
                    static_cast<float>(pn);
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

        bool assign_conv_bias_gradient (tensor& grad, const tensor& gradient_input)
        {
            if (gradient_input.size() == 0 || grad.size() != static_cast<size_t>(gradient_input.k()) ||
                is_same_object(grad, gradient_input) ||
                !admit(OP_assign_conv_bias_gradient, static_cast<double>(gradient_input.size())))
                return false;
            const int64_t n = sz(gradient_input.num_samples()), k = sz(gradient_input.k());
            const int64_t inner = sz(gradient_input.nr() * gradient_input.nc());
            const int64_t pn = bucket(n);
            params p;
            p << pn << k << inner << 3;
            return route(OP_assign_conv_bias_gradient, &build_channels, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, gradient_input.host(), view3(n, k, inner, pn, k));
                // build_channels kind 3 reduces an [n, k, inner] view to [k]; for inner == 1
                // the view is the [n, k] matrix, whose mean over the rows is the same thing.
                c.add_output(out_kind::assign, grad.host(),
                    flat(static_cast<size_t>(k), static_cast<size_t>(k))).alpha =
                    static_cast<float>(pn * inner);
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

    // ----------------------------------------------------------------------------------------
    //                           Optimizers and normalizations
    // ----------------------------------------------------------------------------------------

        namespace
        {
            /*
                p: rows, width, kind (0: Adam, 1: Adam-atan2). Inputs m, v, params, grad and
                the scalars; outputs the new m, v and the step s.
            */
            void build_adam (graph& g, const params& p)
            {
                const node m = g.input({p[0], p[1]});
                const node v = g.input({p[0], p[1]});
                const node w = g.input({p[0], p[1]});
                const node grad = g.input({p[0], p[1]});
                const node m1 = g.scalar(), m2 = g.scalar(), a = g.scalar(), b = g.scalar();
                if (p[2] == 0)
                {
                    // a = weight_decay, b = alpha; eps = 1e-8 as on the host
                    const node gr = g.add(g.mul(a, w), grad);
                    const node mm = g.add(g.mul(m1, m), g.mul(g.sub(g.c(1), m1), gr));
                    const node vv = g.add(g.mul(m2, v), g.mul(g.sub(g.c(1), m2), g.square(gr)));
                    g.output(mm);
                    g.output(vv);
                    g.output(g.div(g.mul(g.mul(b, g.c(-1)), mm), g.add(g.sqrt(vv), g.c(1e-8))));
                }
                else
                {
                    // a = learning_rate*weight_decay, b = learning_rate/bias1, c = 1/sqrt(bias2)
                    const node cs = g.scalar();
                    const node mm = g.add(g.mul(m1, m), g.mul(g.sub(g.c(1), m1), grad));
                    const node vv = g.add(g.mul(m2, v), g.mul(g.sub(g.c(1), m2), g.square(grad)));
                    const node x = g.mul(g.sqrt(vv), cs);
                    // atan2(mm, x) for x >= 0
                    const node half_pi = g.mul(g.sign(mm), g.c(1.57079632679489661923));
                    const node angle = g.select(g.greater(x, g.c(0)), g.atan(g.div(mm, x)), half_pi);
                    g.output(mm);
                    g.output(vv);
                    g.output(g.sub(g.mul(g.mul(a, g.c(-1)), w), g.mul(b, angle)));
                }
            }

            bool adam (int kind, size_t begin, size_t end, tensor& s, tensor& m, tensor& v, const float t,
                const float learning_rate, const float weight_decay, const float momentum1,
                const float momentum2, const tensor& params_t, const tensor& params_grad)
            {
                const int op = kind == 0 ? OP_adam : OP_adam_atan2;
                if (begin >= end || end > params_t.size() || s.size() != params_t.size() || m.size() != s.size() ||
                    v.size() != s.size() || params_grad.size() != s.size() ||
                    !admit(op, static_cast<double>(end - begin)))
                    return false;
                const size_t n = end - begin;
                const int64_t rows = flat_rows(n);
                const size_t padded = static_cast<size_t>(rows * flat_width);
                params p;
                p << rows << flat_width << kind;
                return route(op, &build_adam, p, [&](call& c)
                {
                    const geometry gf = flat(n, padded);
                    c.add_input(in_kind::values, m.host() + begin, gf);
                    c.add_input(in_kind::values, v.host() + begin, gf);
                    c.add_input(in_kind::values, params_t.host() + begin, gf);
                    c.add_input(in_kind::values, params_grad.host() + begin, gf);
                    c.add_scalar(momentum1);
                    c.add_scalar(momentum2);
                    if (kind == 0)
                    {
                        const float alpha =
                            learning_rate * std::sqrt(1 - std::pow(momentum2, t)) / (1 - std::pow(momentum1, t));
                        c.add_scalar(weight_decay);
                        c.add_scalar(alpha);
                    }
                    else
                    {
                        const double bias1 = 1 - std::pow(static_cast<double>(momentum1), static_cast<double>(t));
                        const double bias2 = 1 - std::pow(static_cast<double>(momentum2), static_cast<double>(t));
                        c.add_scalar(learning_rate * weight_decay);
                        c.add_scalar(static_cast<float>(learning_rate / bias1));
                        c.add_scalar(static_cast<float>(1 / std::sqrt(bias2)));
                    }
                    c.add_output(out_kind::assign, m.host() + begin, gf);
                    c.add_output(out_kind::assign, v.host() + begin, gf);
                    c.add_output(out_kind::assign, s.host() + begin, gf);
                });
            }

            /*
                Batch normalization at inference over an [n, k, inner] view, the statistics being
                indexed by k. p: n, k, inner
            */
            // x arrives with the running means already subtracted.
            void build_bn_inference (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                const node gamma = g.input({1, p[1], 1});
                const node beta = g.input({1, p[1], 1});
                const node var = g.input({1, p[1], 1});
                const node eps = g.scalar();
                g.output(g.add(g.div(g.mul(gamma, x), g.sqrt(g.add(var, eps))), beta));
            }

            /*
                Batch normalization in training over an [n, k, inner] view, statistics over the
                axes 0 and 2. Outputs dest, the means, the variances and the inverse standard
                deviations. p: n, k, inner
            */
            void build_bn_train (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                const node gamma = g.input({1, p[1], 1});
                const node beta = g.input({1, p[1], 1});
                const node eps = g.scalar();
                const node mean = g.reduce_mean(x, {0, 2}, true);
                const node centered = g.sub(x, mean);
                const node var = g.reduce_mean(g.square(centered), {0, 2}, true);
                const node invstd = g.div(g.c(1), g.sqrt(g.add(var, eps)));
                g.output(g.add(g.mul(g.mul(centered, invstd), gamma), beta));
                g.output(mean);
                g.output(var);
                g.output(invstd);
            }

            /*
                Gradient of batch normalization over an [n, k, inner] view. Outputs the term
                added to src_grad, then gamma_grad and beta_grad as means over the axes 0 and 2.
                p: n, k, inner
            */
            void build_bn_gradient (graph& g, const params& p)
            {
                const node gi = g.input({p[0], p[1], p[2]});
                const node centered = g.input({p[0], p[1], p[2]}); // x minus the means
                const node invstd = g.input({1, p[1], 1});
                const node gamma = g.input({1, p[1], 1});
                const node xhat = g.mul(centered, invstd);
                const node dx = g.mul(gi, gamma);
                const node m1 = g.reduce_mean(g.mul(dx, xhat), {0, 2}, true);
                const node m2 = g.reduce_mean(dx, {0, 2}, true);
                const node m3 = g.reduce_mean(centered, {0, 2}, true);
                // invstd (dx - mean(dx) - xhat mean(dx xhat)) + invstd^2 mean(dx xhat) mean(x - m),
                // the host formula with its sums regrouped.
                const node src = g.add(g.mul(invstd, g.sub(g.sub(dx, m2), g.mul(xhat, m1))),
                    g.mul(g.mul(invstd, m3), g.mul(invstd, m1)));
                g.output(src);
                g.output(g.reduce_mean(g.mul(gi, xhat), {0, 2}, false));
                g.output(g.reduce_mean(gi, {0, 2}, false));
            }

            /*
                Layer normalization over the axes 1 and 2 of an [n, k, inner] view, gamma and
                beta indexed by k. Outputs dest, the means and the inverse standard deviations.
                p: n, k, inner
            */
            void build_layer_normalize (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2]});
                const node gamma = g.input({1, p[1], 1});
                const node beta = g.input({1, p[1], 1});
                const node eps = g.scalar();
                const node mean = g.reduce_mean(x, {1, 2}, true);
                const node centered = g.sub(x, mean);
                const node invstd = g.div(g.c(1), g.sqrt(g.add(g.reduce_mean(g.square(centered), {1, 2}, true), eps)));
                g.output(g.add(g.mul(g.mul(centered, invstd), gamma), beta));
                g.output(mean);
                g.output(invstd);
            }

            /*
                Gradient of layer normalization. Outputs the term added to src_grad, gamma_grad
                and beta_grad as means over the axes 0 and 2, and per sample the means of
                dx xhat, of dx and of x - mean, from which the host derives dvars and dmeans.
                p: n, k, inner
            */
            void build_layer_normalize_gradient (graph& g, const params& p)
            {
                const node gi = g.input({p[0], p[1], p[2]});
                const node centered = g.input({p[0], p[1], p[2]}); // x minus the means of its sample
                const node invstd = g.input({p[0], 1, 1});
                const node gamma = g.input({1, p[1], 1});
                const node xhat = g.mul(centered, invstd);
                const node dx = g.mul(gi, gamma);
                const node m1 = g.reduce_mean(g.mul(dx, xhat), {1, 2}, true);
                const node m2 = g.reduce_mean(dx, {1, 2}, true);
                const node m3 = g.reduce_mean(centered, {1, 2}, true);
                const node src = g.add(g.mul(invstd, g.sub(g.sub(dx, m2), g.mul(xhat, m1))),
                    g.mul(g.mul(invstd, m3), g.mul(invstd, m1)));
                g.output(src);
                g.output(g.reduce_mean(g.mul(gi, xhat), {0, 2}, false));
                g.output(g.reduce_mean(gi, {0, 2}, false));
                g.output(m1);
                g.output(m2);
                g.output(m3);
            }

            // eps * 2^2e must stay within the half range: lowers e if it would not.
            int clamp_eps_exponent (int e, double eps)
            {
                while (e > 0 && eps * std::ldexp(1.0, 2 * e) > 1e4) --e;
                return e;
            }

            /*
                Offsets of the operand of a normalization laid out by view3(n, k, inner), one per
                channel k or one per sample n, following the layout view3() chooses.
            */
            void offset_by_channel (input& x, const float* offsets, int64_t k, int64_t inner)
            {
                x.offsets = offsets;
                x.offset_by_column = inner == 1 && k > 1;   // rows are samples, columns channels
                x.offset_div = 1;
                x.offset_mod = (inner == 1) ? 1 : k;        // or one row per sample and channel
            }

            void offset_by_sample (input& x, const float* offsets, int64_t n, int64_t k, int64_t inner)
            {
                x.offsets = offsets;
                x.offset_by_column = inner == 1 && k == 1;  // a single row of samples
                x.offset_div = (inner == 1) ? 1 : k;
                x.offset_mod = std::max<int64_t>(n, 1);
            }

            // The first value of every channel in the first sample, or of every sample.
            std::vector<float> first_of_channels (const tensor& src, int64_t k, int64_t inner)
            {
                std::vector<float> v(static_cast<size_t>(k));
                const float* x = src.host();
                for (int64_t i = 0; i < k; ++i)
                    v[static_cast<size_t>(i)] = x[i * inner];
                return v;
            }

            std::vector<float> first_of_samples (const tensor& src, int64_t n)
            {
                std::vector<float> v(static_cast<size_t>(n));
                const float* x = src.host();
                const size_t stride = src.size() / static_cast<size_t>(n);
                for (int64_t i = 0; i < n; ++i)
                    v[static_cast<size_t>(i)] = x[static_cast<size_t>(i) * stride];
                return v;
            }

            bool bn_inference (int op, const double eps, resizable_tensor& dest, const tensor& src,
                const tensor& gamma, const tensor& beta, const tensor& running_means,
                const tensor& running_variances, int64_t k, int64_t inner)
            {
                const int64_t n = sz(src.num_samples());
                if (src.size() == 0 || eps <= 0 || gamma.size() != static_cast<size_t>(k) ||
                    beta.size() != gamma.size() ||
                    running_means.size() != gamma.size() || running_variances.size() != gamma.size() ||
                    !admit(op, static_cast<double>(src.size())))
                    return false;
                dest.copy_size(src);
                const int64_t pn = bucket(n);
                params p;
                p << pn << k << inner;
                return route(op, &build_bn_inference, p, [&](call& c)
                {
                    const geometry gx = view3(n, k, inner, pn, k);
                    const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                    offset_by_channel(c.add_input(in_kind::values, src.host(), gx), running_means.host(), k, inner);
                    c.add_input(in_kind::values, gamma.host(), gk);
                    c.add_input(in_kind::values, beta.host(), gk);
                    c.add_input(in_kind::values, running_variances.host(), gk);
                    c.add_scalar(static_cast<float>(eps));
                    c.add_output(out_kind::assign, dest.host(), gx);
                });
            }

            bool bn_train (int op, const double eps, resizable_tensor& dest, resizable_tensor& means,
                resizable_tensor& invstds, const double averaging_factor, resizable_tensor& running_means,
                resizable_tensor& running_variances, const tensor& src, const tensor& gamma, const tensor& beta,
                int64_t k, int64_t inner, bool conv)
            {
                const int64_t n = sz(src.num_samples());
                if (src.size() == 0 || n < 2 || eps <= 0 || gamma.size() != static_cast<size_t>(k) ||
                    beta.size() != gamma.size() ||
                    averaging_factor < 0 || averaging_factor > 1 || !admit(op, static_cast<double>(src.size())))
                    return false;
                if (averaging_factor != 1 && (running_means.size() != static_cast<size_t>(k) ||
                    running_variances.size() != static_cast<size_t>(k)))
                    return false;
                dest.copy_size(src);
                if (conv)
                {
                    means.set_size(1, k);
                    invstds.set_size(1, k);
                    running_means.set_size(1, k);
                    running_variances.set_size(1, k);
                }
                else
                {
                    means.set_size(1, src.k(), src.nr(), src.nc());
                    invstds.set_size(1, src.k(), src.nr(), src.nc());
                    running_means.set_size(1, src.k(), src.nr(), src.nc());
                    running_variances.set_size(1, src.k(), src.nr(), src.nc());
                }
                // The statistics are taken over the samples: they are not padded. Each channel is
                // offset by one of its values, which the means get back afterwards.
                params p;
                p << n << k << inner;
                const std::vector<float> shift = first_of_channels(src, k, inner);
                std::vector<float> mean, var;
                const bool ok = route(op, &build_bn_train, p, [&](call& c)
                {
                    const geometry gx = view3(n, k, inner, n, k);
                    const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                    input& x = c.add_input(in_kind::scaled, src.host(), gx);
                    x.target = 2;
                    offset_by_channel(x, shift.data(), k, inner);
                    c.add_input(in_kind::values, gamma.host(), gk);
                    c.add_input(in_kind::values, beta.host(), gk);
                    c.add_scalar(static_cast<float>(eps));
                    c.add_output(out_kind::assign, dest.host(), gx);
                    c.add_output(out_kind::staged, nullptr, gk).staged = &mean;
                    c.add_output(out_kind::staged, nullptr, gk).staged = &var;
                    c.add_output(out_kind::assign, invstds.host(), gk);
                    c.finalize = [eps](call& c)
                    {
                        const int e = clamp_eps_exponent(c.in[0].exp, eps);
                        c.in[0].exp = e;
                        c.in[3].value = static_cast<float>(eps * std::ldexp(1.0, 2 * e));
                        c.out[1].exp = -e;
                        c.out[2].exp = -2 * e;
                        c.out[3].exp = e;
                        return true;
                    };
                });
                if (!ok)
                    return false;
                const double count = conv ? static_cast<double>(n * inner) : static_cast<double>(n);
                const double unbias = count / (count - 1.0);
                float* rv = running_variances.host();
                float* rm = running_means.host();
                float* mu = means.host();
                for (int64_t i = 0; i < k; ++i)
                    mu[i] = mean[static_cast<size_t>(i)] + shift[static_cast<size_t>(i)];
                for (int64_t i = 0; i < k; ++i)
                {
                    if (averaging_factor == 1)
                    {
                        rv[i] = static_cast<float>(unbias * var[static_cast<size_t>(i)]);
                        rm[i] = mu[i];
                    }
                    else
                    {
                        const double v = var[static_cast<size_t>(i)];
                        rv[i] = static_cast<float>((1 - averaging_factor) * rv[i] + unbias * averaging_factor * v);
                        rm[i] = static_cast<float>((1 - averaging_factor) * rm[i] + averaging_factor * mu[i]);
                    }
                }
                return true;
            }

            bool bn_gradient (int op, const tensor& gradient_input, const tensor& means, const tensor& invstds,
                const tensor& src, const tensor& gamma, tensor& src_grad, tensor& gamma_grad, tensor& beta_grad,
                int64_t k, int64_t inner)
            {
                const int64_t n = sz(src.num_samples());
                if (src.size() == 0 || n < 2 || gradient_input.size() != src.size() || src_grad.size() != src.size() ||
                    means.size() != static_cast<size_t>(k) || invstds.size() != means.size() ||
                    gamma.size() != means.size() ||
                    gamma_grad.size() != means.size() || beta_grad.size() != means.size() ||
                    !admit(op, static_cast<double>(src.size())))
                    return false;
                params p;
                p << n << k << inner;
                return route(op, &build_bn_gradient, p, [&](call& c)
                {
                    const geometry gx = view3(n, k, inner, n, k);
                    const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                    c.add_input(in_kind::scaled, gradient_input.host(), gx);
                    offset_by_channel(c.add_input(in_kind::values, src.host(), gx), means.host(), k, inner);
                    c.add_input(in_kind::values, invstds.host(), gk);
                    c.add_input(in_kind::values, gamma.host(), gk);
                    c.add_output(out_kind::add, src_grad.host(), gx);
                    c.add_output(out_kind::assign, gamma_grad.host(), gk).alpha = static_cast<float>(n * inner);
                    c.add_output(out_kind::assign, beta_grad.host(), gk).alpha = static_cast<float>(n * inner);
                    c.finalize = [](call& c)
                    {
                        for (auto& o : c.out) o.exp = -c.in[0].exp;
                        return true;
                    };
                });
            }
        }

        bool compute_adam_update (size_t begin, size_t end, tensor& s, tensor& m, tensor& v,
            const float t, const float learning_rate, const float weight_decay,
            const float momentum1, const float momentum2, const tensor& params_t,
            const tensor& params_grad)
        {
            return adam(0, begin, end, s, m, v, t, learning_rate, weight_decay, momentum1, momentum2, params_t,
                params_grad);
        }

        bool compute_adam_atan2_update (size_t begin, size_t end, tensor& s, tensor& m, tensor& v,
            const float t, const float learning_rate, const float weight_decay,
            const float momentum1, const float momentum2, const tensor& params_t,
            const tensor& params_grad)
        {
            return adam(1, begin, end, s, m, v, t, learning_rate, weight_decay, momentum1, momentum2, params_t,
                params_grad);
        }

        bool batch_normalize_inference (const double eps, resizable_tensor& dest, const tensor& src,
            const tensor& gamma, const tensor& beta, const tensor& running_means,
            const tensor& running_variances)
        {
            // Statistics per element of a sample: a [n, k*nr*nc, 1] view.
            return bn_inference(OP_batch_normalize_inference, eps, dest, src, gamma, beta, running_means,
                running_variances, sz(src.k() * src.nr() * src.nc()), 1);
        }

        bool batch_normalize_conv_inference (const double eps, resizable_tensor& dest,
            const tensor& src, const tensor& gamma, const tensor& beta,
            const tensor& running_means, const tensor& running_variances)
        {
            return bn_inference(OP_batch_normalize_conv_inference, eps, dest, src, gamma, beta, running_means,
                running_variances, sz(src.k()), sz(src.nr() * src.nc()));
        }

        bool batch_normalize (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const double averaging_factor,
            resizable_tensor& running_means, resizable_tensor& running_variances,
            const tensor& src, const tensor& gamma, const tensor& beta)
        {
            return bn_train(OP_batch_normalize, eps, dest, means, invstds, averaging_factor, running_means,
                running_variances, src, gamma, beta, sz(src.k() * src.nr() * src.nc()), 1, false);
        }

        bool batch_normalize_conv (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const double averaging_factor,
            resizable_tensor& running_means, resizable_tensor& running_variances,
            const tensor& src, const tensor& gamma, const tensor& beta)
        {
            return bn_train(OP_batch_normalize_conv, eps, dest, means, invstds, averaging_factor, running_means,
                running_variances, src, gamma, beta, sz(src.k()), sz(src.nr() * src.nc()), true);
        }

        bool batch_normalize_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad)
        {
            (void)eps;
            return bn_gradient(OP_batch_normalize_gradient, gradient_input, means, invstds, src, gamma, src_grad,
                gamma_grad, beta_grad, sz(src.k() * src.nr() * src.nc()), 1);
        }

        bool batch_normalize_conv_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad)
        {
            (void)eps;
            return bn_gradient(OP_batch_normalize_conv_gradient, gradient_input, means, invstds, src, gamma, src_grad,
                gamma_grad, beta_grad, sz(src.k()), sz(src.nr() * src.nc()));
        }

        bool layer_normalize (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const tensor& src, const tensor& gamma, const tensor& beta)
        {
            const int64_t n = sz(src.num_samples()), k = sz(src.k()), inner = sz(src.nr() * src.nc());
            if (src.size() == 0 || eps <= 0 || gamma.size() != static_cast<size_t>(k) || beta.size() != gamma.size() ||
                !admit(OP_layer_normalize, static_cast<double>(src.size())))
                return false;
            dest.copy_size(src);
            means.set_size(n);
            invstds.set_size(n);
            const int64_t pn = bucket(n);
            params p;
            p << pn << k << inner;
            // Each sample is offset by one of its values, which its mean gets back afterwards.
            const std::vector<float> shift = first_of_samples(src, n);
            std::vector<float> mean;
            const bool ok = route(OP_layer_normalize, &build_layer_normalize, p, [&](call& c)
            {
                const geometry gx = view3(n, k, inner, pn, k);
                const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                const geometry gs = flat(static_cast<size_t>(n), static_cast<size_t>(pn));
                input& x = c.add_input(in_kind::scaled, src.host(), gx);
                x.target = 6;
                offset_by_sample(x, shift.data(), n, k, inner);
                c.add_input(in_kind::values, gamma.host(), gk);
                c.add_input(in_kind::values, beta.host(), gk);
                c.add_scalar(static_cast<float>(eps));
                c.add_output(out_kind::assign, dest.host(), gx);
                c.add_output(out_kind::staged, nullptr, gs).staged = &mean;
                c.add_output(out_kind::assign, invstds.host(), gs);
                c.finalize = [eps](call& c)
                {
                    const int e = clamp_eps_exponent(c.in[0].exp, eps);
                    c.in[0].exp = e;
                    c.in[3].value = static_cast<float>(eps * std::ldexp(1.0, 2 * e));
                    c.out[1].exp = -e;
                    c.out[2].exp = e;
                    return true;
                };
            });
            if (!ok)
                return false;
            float* mu = means.host();
            for (int64_t i = 0; i < n; ++i)
                mu[i] = mean[static_cast<size_t>(i)] + shift[static_cast<size_t>(i)];
            return true;
        }

        bool layer_normalize_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad,
            resizable_tensor& dmeans, resizable_tensor& dvars)
        {
            (void)eps;
            const int64_t n = sz(src.num_samples()), k = sz(src.k()), inner = sz(src.nr() * src.nc());
            if (src.size() == 0 || gradient_input.size() != src.size() || src_grad.size() != src.size() ||
                means.size() != static_cast<size_t>(n) || invstds.size() != means.size() ||
                gamma.size() != static_cast<size_t>(k) || gamma_grad.size() != gamma.size() ||
                beta_grad.size() != gamma.size() ||
                !admit(OP_layer_normalize_gradient, static_cast<double>(src.size())))
                return false;
            const int64_t pn = bucket(n);
            params p;
            p << pn << k << inner;
            std::vector<float> s1, s2, s3;
            const bool ok = route(OP_layer_normalize_gradient, &build_layer_normalize_gradient, p, [&](call& c)
            {
                const geometry gx = view3(n, k, inner, pn, k);
                const geometry gk = flat(static_cast<size_t>(k), static_cast<size_t>(k));
                const geometry gs = flat(static_cast<size_t>(n), static_cast<size_t>(pn));
                const float num = static_cast<float>(k * inner);
                c.add_input(in_kind::scaled, gradient_input.host(), gx);
                offset_by_sample(c.add_input(in_kind::values, src.host(), gx), means.host(), n, k, inner);
                c.add_input(in_kind::values, invstds.host(), gs);
                c.add_input(in_kind::values, gamma.host(), gk);
                c.add_output(out_kind::add, src_grad.host(), gx);
                c.add_output(out_kind::assign, gamma_grad.host(), gk).alpha = static_cast<float>(pn * inner);
                c.add_output(out_kind::assign, beta_grad.host(), gk).alpha = static_cast<float>(pn * inner);
                c.add_output(out_kind::staged, nullptr, gs).staged = &s1;
                c.out.back().alpha = num;
                c.add_output(out_kind::staged, nullptr, gs).staged = &s2;
                c.out.back().alpha = num;
                c.add_output(out_kind::staged, nullptr, gs).staged = &s3;
                c.out.back().alpha = num;
                c.finalize = [](call& c)
                {
                    for (size_t i = 0; i < 5; ++i) c.out[i].exp = -c.in[0].exp;
                    return true;
                };
            });
            if (!ok)
                return false;
            // dvars = -0.5 invstd^2 sum(dx xhat), dmeans = -invstd sum(dx) - 2 dvars sum(x - m)/num
            dvars.copy_size(invstds);
            dmeans.copy_size(means);
            const float* inv = invstds.host();
            float* dv = dvars.host();
            float* dm = dmeans.host();
            const float invnum = 1.0f / static_cast<float>(k * inner);
            for (int64_t i = 0; i < n; ++i)
            {
                const size_t j = static_cast<size_t>(i);
                dv[j] = -0.5f * inv[j] * inv[j] * s1[j];
                dm[j] = -inv[j] * s2[j] - 2 * dv[j] * s3[j] * invnum;
            }
            return true;
        }

    // ----------------------------------------------------------------------------------------
    //                 Data movement, embeddings, resizing and adaptive halting
    // ----------------------------------------------------------------------------------------

        namespace
        {
            // p: rows, width. Outputs the gradient to add and the mean of gi * x where x <= 0.
            void build_prelu_gradient (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1]});
                const node gi = g.input({p[0], p[1]});
                const node a = g.scalar();
                const node positive = g.greater(x, g.c(0));
                g.output(g.select(positive, gi, g.mul(a, gi)));
                g.output(g.reshape(g.reduce_mean(g.select(positive, g.c(0), g.mul(gi, x)), {0, 1}, false), {1}));
            }

            // Weights of dlib's bilinear resize along one axis, as an [out, in] matrix.
            std::vector<float> bilinear_weights (int64_t out, int64_t in)
            {
                std::vector<float> w(static_cast<size_t>(out * in), 0.0f);
                const double scale = (in - 1) / static_cast<double>(std::max<int64_t>(out - 1, 1));
                for (int64_t i = 0; i < out; ++i)
                {
                    const double x = i * scale;
                    const int64_t left = static_cast<int64_t>(std::floor(x));
                    const int64_t right = std::min(left + 1, in - 1);
                    const double frac = x - left;
                    w[static_cast<size_t>(i * in + left)] += static_cast<float>(1 - frac);
                    w[static_cast<size_t>(i * in + right)] += static_cast<float>(frac);
                }
                return w;
            }

            /*
                Bilinear resizing as two products with constant interpolation matrices, which
                makes the gradient the same products transposed.
                p: planes, src rows, src cols, dest rows, dest cols, direction (0: resize src
                to dest, 1: the adjoint, from dest back to src)
            */
            void build_resize (graph& g, const params& p)
            {
                const int64_t planes = p[0], sr = p[1], sc = p[2], dr = p[3], dc = p[4];
                const node wy = g.constant({dr, sr}, bilinear_weights(dr, sr));
                const node wx = g.constant({dc, sc}, bilinear_weights(dc, sc));
                if (p[5] == 0)
                {
                    const node x = g.input({planes, sr, sc});
                    g.output(g.matmul(wy, g.matmul(x, wx, false, true)));
                }
                else
                {
                    const node gi = g.input({planes, dr, dc});
                    g.output(g.matmul(wy, g.matmul(gi, wx), true, false));
                }
            }

            // p: n, sk, dnr, row stride, dnc, col stride, direction (0: reorg, 1: its gradient)
            void build_reorg (graph& g, const params& p)
            {
                const int64_t n = p[0], sk = p[1], dnr = p[2], rs = p[3], dnc = p[4], cs = p[5];
                if (p[6] == 0)
                {
                    const node x = g.input({n, sk, dnr * rs, dnc * cs});
                    const node t = g.transpose(g.reshape(x, {n, sk, dnr, rs, dnc, cs}), {0, 3, 5, 1, 2, 4});
                    g.output(g.reshape(t, {n, rs * cs * sk, dnr, dnc}));
                }
                else
                {
                    const node x = g.input({n, rs * cs * sk, dnr, dnc});
                    const node t = g.transpose(g.reshape(x, {n, rs, cs, sk, dnr, dnc}), {0, 3, 4, 1, 5, 2});
                    g.output(g.reshape(t, {n, sk, dnr * rs, dnc * cs}));
                }
            }

            // p: n, k, r, c. The block moves through the device unchanged.
            void build_copy (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2], p[3]});
                g.output(g.mul(x, g.c(1)));
            }

            // p: n. Matrix inverse.
            void build_inverse (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[0]});
                g.output(std::make_shared<ov::op::v14::Inverse>(x, false));
            }

            // p: n, k, r, c, kind (0: swap the last two axes, 1: split heads, 2: merge heads)
            void build_permute (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2], p[3]});
                g.output(g.transpose(x, p[4] == 0 ? std::vector<int64_t>{0, 1, 3, 2}
                    : std::vector<int64_t>{0, 2, 1, 3}));
            }

            // p: table rows, second table rows (0 if none), dim, positions
            void build_gather (graph& g, const params& p)
            {
                node table = g.input({p[0], p[2]});
                if (p[1] > 0)
                    table = g.concat({table, g.input({p[1], p[2]})}, 0);
                const node idx = g.input_i32({p[3]});
                const node weight = g.input({p[3], 1});
                g.output(g.mul(g.gather(table, idx, 0), weight));
            }

            node scatter_add (graph& g, const node& data, const node& idx_column, const node& updates, int64_t rows,
                int64_t dim)
            {
                const node idx = g.broadcast(idx_column, {rows, dim});
                return std::make_shared<ov::op::v12::ScatterElementsUpdate>(data, idx, updates, graph::ints({0}),
                    ov::op::v12::ScatterElementsUpdate::Reduction::SUM, true);
            }

            // p: vocabulary, dim, positions. table - lr * sum of freq_scale * gi per token.
            void build_embeddings_gradient (graph& g, const params& p)
            {
                const node table = g.input({p[0], p[1]});
                const node idx = g.input_i32({p[2], 1});
                const node gi = g.input({p[2], p[1]});
                const node fs = g.input({p[2], 1});
                const node lr = g.scalar();
                const node updates = g.mul(gi, g.mul(fs, g.mul(lr, g.c(-1))));
                g.output(scatter_add(g, table, idx, updates, p[2], p[1]));
            }

            /*
                p: vocabulary, puzzles, dim, positions. Outputs the token gradient and, when the
                batch names puzzles, the puzzle table after its sign descent step.
            */
            void build_puzzle_gradient (graph& g, const params& p)
            {
                const int64_t vocab = p[0], puzzles = p[1], dim = p[2], positions = p[3];
                const node gi = g.input({positions, dim});
                const node idx_t = g.input_i32({positions, 1});
                const node w_t = g.input({positions, 1});
                const node zeros_t = g.broadcast(g.c(0), {vocab, dim});
                g.output(scatter_add(g, zeros_t, idx_t, g.mul(gi, w_t), positions, dim));
                if (puzzles > 0)
                {
                    const node idx_p = g.input_i32({positions, 1});
                    const node w_p = g.input({positions, 1});
                    const node touched = g.input({puzzles, 1});
                    const node table = g.input({puzzles, dim});
                    const node keep = g.scalar();
                    const node rate = g.scalar();
                    const node zeros_p = g.broadcast(g.c(0), {puzzles, dim});
                    const node sums = scatter_add(g, zeros_p, idx_p, g.mul(gi, w_p), positions, dim);
                    const node stepped = g.sub(g.mul(table, keep), g.mul(rate, g.sign(sums)));
                    g.output(g.select(g.greater(touched, g.c(0.5)), stepped, table));
                }
            }

            // p: n, k, repeat, inner, direction (0: repeat channels, 1: accumulate them)
            void build_repeat (graph& g, const params& p)
            {
                const int64_t n = p[0], k = p[1], r = p[2], inner = p[3];
                if (p[4] == 0)
                {
                    const node x = g.input({n, k, 1, inner});
                    g.output(g.reshape(g.broadcast(x, {n, k, r, inner}), {n, k * r, inner}));
                }
                else
                {
                    const node x = g.input({n, k, r, inner});
                    g.output(g.reduce_sum(x, {2}, false));
                }
            }

            // p: batch, channels, seq, features
            void build_halt_logits (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2], p[3]});
                const node w = g.input({1, 1, 1, p[3]});
                const node b = g.scalar();
                const node mask = g.input({p[0], p[2]});
                const node z =
                    g.add(g.mul(g.reduce_mean(g.mul(x, w), {1, 3}, false), g.c(static_cast<double>(p[3]))), b);
                g.output(g.mul(z, mask));
                g.output(g.mul(g.sigmoid(z), mask));
            }

            /*
                p: batch, channels, seq, features. Outputs the terms added to the halting weights,
                to the halting bias and to the state gradient.
            */
            void build_halt_gradient (graph& g, const params& p)
            {
                const int64_t B = p[0], C = p[1], S = p[2], F = p[3];
                const node gi = g.input({B, C, S, F});
                const node x = g.input({B, C, S, F});
                const node fin = g.input({B, C, S, F});
                const node h = g.input({B, S});
                const node w = g.input({1, 1, 1, F});
                const node mask = g.input({B, S});
                const node steps = g.input({B, S});
                const node step = g.scalar();
                const node ponder = g.scalar();
                const node check = g.scalar();
                const node thresh = g.scalar();
                const node diff =
                    g.mul(g.reduce_mean(g.mul(gi, g.sub(x, fin)), {1, 3}, false), g.c(static_cast<double>(C * F)));
                const node gh = g.sub(diff, ponder);
                const node active =
                    g.logical_and(g.logical_and(g.greater_equal(mask, g.c(0.5)), g.greater(steps, step)),
                    g.not_equal(steps, g.add(step, g.c(1))));
                const node keep =
                    g.logical_and(active, g.logical_or(g.greater(check, g.c(0.5)), g.greater(g.abs(gh), thresh)));
                const node gl = g.select(keep, g.mul(gh, g.mul(h, g.sub(g.c(1), h))), g.c(0));
                const node gl4 = g.reshape(gl, {B, 1, S, 1});
                g.output(g.reduce_mean(g.mul(gl4, x), {0, 1, 2}, false));
                g.output(g.reshape(g.reduce_mean(gl, {0, 1}, false), {1}));
                g.output(g.broadcast(g.mul(gl4, g.mul(w, g.c(1.0 / static_cast<double>(C)))), {B, C, S, F}));
            }

            /*
                Position weighted state updates of the adaptive halting loop, over a
                [batch, channels, seq, features] state with per position vectors.
                p: batch, channels, seq, features, kind (0: accumulate_weighted_state,
                1: apply_act_depth_scaling, 2: act_accumulate_output_grad)
            */
            void build_halt_weighting (graph& g, const params& p)
            {
                const int64_t B = p[0], C = p[1], S = p[2], F = p[3];
                const node x = g.input({B, C, S, F});
                switch (p[4])
                {
                    case 0:
                    {
                        const node w = g.input({B, 1, S, 1});
                        const node mask = g.input({B, 1, S, 1});
                        g.output(g.mul(x, g.select(g.greater_equal(mask, g.c(0.5)), w, g.c(0))));
                        break;
                    }
                    case 1:
                    {
                        const node steps = g.input({B, 1, S, 1});
                        const node inv_max = g.scalar();
                        g.output(g.div(x, g.add(g.c(1), g.mul(g.c(0.1), g.mul(steps, inv_max)))));
                        break;
                    }
                    case 2:
                    {
                        const node w = g.input({B, 1, S, 1});
                        const node mask = g.input({B, 1, S, 1});
                        const node steps = g.input({B, 1, S, 1});
                        const node step = g.scalar();
                        const node on = g.logical_and(g.greater_equal(mask, g.c(0.5)), g.greater(steps, step));
                        g.output(g.mul(x, g.select(on, w, g.c(0))));
                        break;
                    }
                    default: throw std::logic_error("unknown halting operation");
                }
            }

            // p: rows, width. Inputs cum, rem, steps, weights, probs, mask and the scalars
            // step + 1, last step, threshold; outputs the four updated vectors.
            void build_halt_update (graph& g, const params& p)
            {
                const node cum = g.input({p[0], p[1]});
                const node rem = g.input({p[0], p[1]});
                const node steps = g.input({p[0], p[1]});
                const node weights = g.input({p[0], p[1]});
                const node h = g.input({p[0], p[1]});
                const node mask = g.input({p[0], p[1]});
                const node next = g.scalar();
                const node last = g.scalar();
                const node thresh = g.scalar();
                const node active = g.greater_equal(mask, g.c(0.5));
                const node fresh = g.add(cum, h);
                const node halt = g.logical_or(g.greater_equal(fresh, thresh), g.greater(last, g.c(0.5)));
                const node halting = g.logical_and(active, halt);
                const node running = g.logical_and(active, std::make_shared<ov::op::v1::LogicalNot>(halt));
                g.output(g.select(active, g.select(halt, thresh, fresh), cum));
                g.output(g.select(running, g.sub(rem, h), rem));
                g.output(g.select(halting, next, steps));
                g.output(g.select(active, g.select(halt, rem, h), weights));
            }

            // p: rows, width. mask = cum < threshold.
            void build_mark_active (graph& g, const params& p)
            {
                const node cum = g.input({p[0], p[1]});
                const node thresh = g.scalar();
                g.output(g.to_value(g.less(cum, thresh)));
            }

            // p: batch, channels, seq, features. dst = steps == target ? src : dst
            void build_gather_final (graph& g, const params& p)
            {
                const node src = g.input({p[0], p[1], p[2], p[3]});
                const node dst = g.input({p[0], p[1], p[2], p[3]});
                const node steps = g.input({p[0], 1, p[2], 1});
                const node target = g.scalar();
                g.output(g.select(g.equal(steps, target), src, dst));
            }

            // p: rows, cols. Every row receives the same vector.
            void build_broadcast_row (graph& g, const params& p)
            {
                const node row = g.input({1, p[1]});
                g.output(g.broadcast(row, {p[0], p[1]}));
            }

            /*
                Rotary embedding of a [bh, seq, dim] tensor with [seq, half] caches; the sign
                input is 1 for the rotation and -1 for its inverse. p: bh, seq, dim, half
            */
            void build_rotary (graph& g, const params& p)
            {
                const int64_t bh = p[0], seq = p[1], dim = p[2], half = p[3], rot = 2 * half;
                const node x = g.input({bh, seq, dim});
                const node cs = g.reshape(g.input({seq, half}), {1, seq, half});
                const node sn = g.reshape(g.input({seq, half}), {1, seq, half});
                const node sign = g.scalar();
                const node xr = (rot == dim) ? x : g.slice(x, {0}, {rot}, {2});
                const node pairs = g.reshape(xr, {bh, seq, half, 2});
                const node x0 = g.reshape(g.slice(pairs, {0}, {1}, {3}), {bh, seq, half});
                const node x1 = g.reshape(g.slice(pairs, {1}, {2}, {3}), {bh, seq, half});
                const node y0 = g.sub(g.mul(x0, cs), g.mul(sign, g.mul(x1, sn)));
                const node y1 = g.add(g.mul(sign, g.mul(x0, sn)), g.mul(x1, cs));
                node y = g.reshape(g.concat({g.reshape(y0, {bh, seq, half, 1}), g.reshape(y1, {bh, seq, half, 1})}, 3),
                    {bh, seq, rot});
                if (rot < dim)
                    y = g.concat({y, g.slice(x, {rot}, {dim}, {2})}, 2);
                g.output(y);
            }

            // p: rows, width. Outputs y - out and the mean of its squares.
            void build_mean_squared (graph& g, const params& p)
            {
                const node out = g.input({p[0], p[1]});
                const node y = g.input({p[0], p[1]});
                const node d = g.sub(y, out);
                g.output(d);
                g.output(g.reshape(g.reduce_mean(g.square(d), {0, 1}, false), {1}));
            }

            // A block of a tensor: extents and strides.
            geometry block (int64_t n, int64_t k, int64_t r, int64_t c, int64_t s0, int64_t s1, int64_t s2)
            {
                geometry g = dense(n, k, r, c);
                g.s[0] = s0; g.s[1] = s1; g.s[2] = s2; g.s[3] = 1;
                return g;
            }

            bool resize (int op, int direction, tensor& out, const tensor& in, long out_row_stride,
                long out_channel_stride, long in_row_stride, long in_channel_stride, out_kind mode)
            {
                if (out.size() == 0 || in.size() == 0 || out.num_samples() != in.num_samples() || out.k() != in.k() ||
                    is_same_object(out, in))
                    return false;
                // Only the dense layouts; strided views stay on the host.
                if (out_row_stride != out.nc() || out_channel_stride != out.nr() * out.nc() ||
                    in_row_stride != in.nc() || in_channel_stride != in.nr() * in.nc())
                    return false;
                const int64_t planes = sz(in.num_samples() * in.k());
                if (!admit(op, static_cast<double>(std::max(out.size(), in.size()))))
                    return false;
                // The source geometry of the forward resize is the smaller side of the adjoint.
                const tensor& src = direction == 0 ? in : out;
                const tensor& dst = direction == 0 ? out : in;
                params p;
                p << planes << sz(src.nr()) << sz(src.nc()) << sz(dst.nr()) << sz(dst.nc()) << direction;
                return route(op, &build_resize, p, [&](call& c)
                {
                    c.add_input(in_kind::scaled, in.host(), dense(planes, 1, sz(in.nr()), sz(in.nc())));
                    c.add_output(mode, out.host(), dense(planes, 1, sz(out.nr()), sz(out.nc())));
                    c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
                });
            }

            bool halt_weighting (int op, int kind, tensor& out, out_kind mode, const tensor& x,
                const tensor* w, const tensor* mask, const tensor* steps, float scalar,
                long B, long S, long C, long F)
            {
                const size_t total = static_cast<size_t>(B) * C * S * F;
                if (total == 0 || x.size() != total || out.size() != total || !admit(op, static_cast<double>(total)))
                    return false;
                const size_t positions = static_cast<size_t>(B) * S;
                if ((w && w->size() != positions) || (mask && mask->size() != positions) ||
                    (steps && steps->size() != positions))
                    return false;
                params p;
                p << B << C << S << F << kind;
                return route(op, &build_halt_weighting, p, [&](call& c)
                {
                    const geometry gx = dense(B, C, S, F);
                    const geometry gp = flat(positions, positions);
                    c.add_input(in_kind::scaled, x.host(), gx);
                    if (kind == 1)
                    {
                        c.add_input(in_kind::values, steps->host(), gp);
                        c.add_scalar(scalar);
                    }
                    else
                    {
                        c.add_input(in_kind::values, w->host(), gp);
                        c.add_input(in_kind::values, mask->host(), gp);
                        if (kind == 2)
                        {
                            c.add_input(in_kind::values, steps->host(), gp);
                            c.add_scalar(scalar);
                        }
                    }
                    c.add_output(mode, out.host(), gx);
                    c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
                });
            }

            // Conv geometry: p: n, c, h, w, o, kh, kw, sy, sx, py, px, oh, ow
            void build_conv (graph& g, const params& p)
            {
                const node x = g.input({p[0], p[1], p[2], p[3]});
                const node f = g.input({p[4], p[1], p[5], p[6]});
                g.output(std::make_shared<ov::op::v1::Convolution>(x, f,
                    ov::Strides{static_cast<size_t>(p[7]), static_cast<size_t>(p[8])},
                    ov::CoordinateDiff{p[9], p[10]}, ov::CoordinateDiff{p[9], p[10]},
                    ov::Strides{1, 1}, ov::op::PadType::EXPLICIT));
            }

            void build_conv_data (graph& g, const params& p)
            {
                const int64_t h = p[2], w = p[3], kh = p[5], kw = p[6], sy = p[7], sx = p[8], py = p[9], px = p[10];
                const int64_t oh = p[11], ow = p[12];
                const node gi = g.input({p[0], p[4], oh, ow});
                const node f = g.input({p[4], p[1], kh, kw});
                // The rows and columns of the input no window reaches, after the last stride.
                const int64_t extra_y = h - (sy * (oh - 1) + kh - 2 * py);
                const int64_t extra_x = w - (sx * (ow - 1) + kw - 2 * px);
                g.output(std::make_shared<ov::op::v1::ConvolutionBackpropData>(gi, f,
                    ov::Strides{static_cast<size_t>(sy), static_cast<size_t>(sx)},
                    ov::CoordinateDiff{py, px}, ov::CoordinateDiff{py, px}, ov::Strides{1, 1},
                    ov::op::PadType::EXPLICIT, ov::CoordinateDiff{extra_y, extra_x}));
            }

            void build_conv_filters (graph& g, const params& p)
            {
                const int64_t kh = p[5], kw = p[6], sy = p[7], sx = p[8], py = p[9], px = p[10];
                const node gi = g.input({p[0], p[4], p[11], p[12]});
                const node x = g.input({p[0], p[1], p[2], p[3]});
                // A convolution of the data with the gradient, the samples as the reduced
                // channels and the stride as the dilation, cropped to the filter size.
                const node xt = g.transpose(x, {1, 0, 2, 3});
                const node gt = g.transpose(gi, {1, 0, 2, 3});
                const node conv = std::make_shared<ov::op::v1::Convolution>(xt, gt, ov::Strides{1, 1},
                    ov::CoordinateDiff{py, px}, ov::CoordinateDiff{py, px},
                    ov::Strides{static_cast<size_t>(sy), static_cast<size_t>(sx)}, ov::op::PadType::EXPLICIT);
                const node crop = g.slice(conv, {0, 0}, {kh, kw}, {2, 3});
                g.output(g.transpose(crop, {1, 0, 2, 3}));
            }

            // p: planes, h, w, window h, window w, sy, sx, py, px, max
            void build_pool (graph& g, const params& p)
            {
                const node x = g.input({p[0], 1, p[1], p[2]});
                const ov::Strides strides{static_cast<size_t>(p[5]), static_cast<size_t>(p[6])};
                const ov::Shape pads{static_cast<size_t>(p[7]), static_cast<size_t>(p[8])};
                const ov::Shape kernel{static_cast<size_t>(p[3]), static_cast<size_t>(p[4])};
                if (p[9])
                    g.output(std::make_shared<ov::op::v1::MaxPool>(x, strides, pads, pads, kernel,
                        ov::op::RoundingType::FLOOR, ov::op::PadType::EXPLICIT));
                else
                    g.output(std::make_shared<ov::op::v1::AvgPool>(x, strides, pads, pads, kernel, true,
                        ov::op::RoundingType::FLOOR, ov::op::PadType::EXPLICIT));
            }

            /*
                Adjoint of average pooling over the clipped windows: a product with the
                constant [oh*ow, h*w] matrix of the window weights.
            */
            std::vector<float> avg_pool_adjoint (int64_t h, int64_t w, int64_t wh, int64_t ww, int64_t sy, int64_t sx,
                int64_t py, int64_t px, int64_t oh, int64_t ow)
            {
                std::vector<float> a(static_cast<size_t>(oh * ow * h * w), 0.0f);
                for (int64_t r = 0; r < oh; ++r)
                    for (int64_t c = 0; c < ow; ++c)
                    {
                        const int64_t top = std::max<int64_t>(r * sy - py, 0);
                        const int64_t bottom = std::min<int64_t>(r * sy - py + wh - 1, h - 1);
                        const int64_t left = std::max<int64_t>(c * sx - px, 0);
                        const int64_t right = std::min<int64_t>(c * sx - px + ww - 1, w - 1);
                        const int64_t area = (bottom - top + 1) * (right - left + 1);
                        if (area <= 0) continue;
                        for (int64_t y = top; y <= bottom; ++y)
                            for (int64_t x = left; x <= right; ++x)
                                a[static_cast<size_t>((r * ow + c) * h * w + y * w + x)] =
                                    1.0f / static_cast<float>(area);
                    }
                return a;
            }

            // p: planes, h, w, window h, window w, sy, sx, py, px, max, oh, ow
            void build_pool_gradient (graph& g, const params& p)
            {
                const int64_t planes = p[0], h = p[1], w = p[2], oh = p[10], ow = p[11];
                if (p[9])
                {
                    const node x = g.input({planes, 1, h, w});
                    const node gi = g.input({planes, 1, oh, ow});
                    auto pool = std::make_shared<ov::op::v8::MaxPool>(x,
                        ov::Strides{static_cast<size_t>(p[5]), static_cast<size_t>(p[6])}, ov::Strides{1, 1},
                        ov::Shape{static_cast<size_t>(p[7]), static_cast<size_t>(p[8])},
                        ov::Shape{static_cast<size_t>(p[7]), static_cast<size_t>(p[8])},
                        ov::Shape{static_cast<size_t>(p[3]), static_cast<size_t>(p[4])},
                        ov::op::RoundingType::FLOOR, ov::op::PadType::EXPLICIT, ov::element::i64, 2);
                    const node idx = g.reshape(pool->output(1), {planes, oh * ow});
                    const node zeros = g.broadcast(g.c(0), {planes, h * w});
                    const node scattered = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros, idx,
                        g.reshape(gi, {planes, oh * ow}), graph::ints({1}),
                        ov::op::v12::ScatterElementsUpdate::Reduction::SUM, true);
                    g.output(scattered);
                }
                else
                {
                    const node gi = g.input({planes, 1, oh * ow});
                    const node a =
                        g.constant({oh * ow, h * w},
                            avg_pool_adjoint(h, w, p[3], p[4], p[5], p[6], p[7], p[8], oh, ow));
                    g.output(g.matmul(gi, a));
                }
            }
        }

        bool prelu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input,
            const tensor& param, tensor& params_grad)
        {
            const size_t n = src.size();
            if (n == 0 || grad.size() != n || gradient_input.size() != n || is_same_object(grad, gradient_input) ||
                params_grad.size() == 0 || !admit(OP_prelu_gradient, static_cast<double>(n)))
                return false;
            const int64_t rows = flat_rows(n);
            const size_t padded = static_cast<size_t>(rows * flat_width);
            params p;
            p << rows << flat_width;
            return route(OP_prelu_gradient, &build_prelu_gradient, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(), flat(n, padded));
                c.add_input(in_kind::scaled, gradient_input.host(), flat(n, padded));
                c.add_scalar(param.host()[0]);
                c.add_output(out_kind::add, grad.host(), flat(n, padded));
                c.add_output(out_kind::assign, params_grad.host(), flat(1, 1)).alpha = static_cast<float>(padded);
                c.finalize = [](call& c)
                {
                    c.out[0].exp = c.out[1].exp = -c.in[1].exp;
                    return true;
                };
            });
        }

        bool resize_bilinear (tensor& dest, long dest_row_stride, long dest_channel_stride,
            const tensor& src, long src_row_stride, long src_channel_stride)
        {
            return resize(OP_resize_bilinear, 0, dest, src, dest_row_stride, dest_channel_stride,
                src_row_stride, src_channel_stride, out_kind::assign);
        }

        bool resize_bilinear_gradient (tensor& grad, long grad_row_stride, long grad_channel_stride,
            const tensor& gradient_input, long gradient_input_row_stride, long gradient_input_channel_stride)
        {
            return resize(OP_resize_bilinear_gradient, 1, grad, gradient_input, grad_row_stride, grad_channel_stride,
                gradient_input_row_stride, gradient_input_channel_stride, out_kind::add);
        }

        bool reorg (bool add_to, tensor& dest, const int row_stride, const int col_stride, const tensor& src)
        {
            if (src.size() == 0 || is_same_object(dest, src) || src.nr() % row_stride != 0 ||
                src.nc() % col_stride != 0 ||
                dest.size() != src.size() || !admit(OP_reorg, static_cast<double>(src.size())))
                return false;
            params p;
            p << sz(src.num_samples()) << sz(src.k()) << sz(src.nr() / row_stride) << row_stride
              << sz(src.nc() / col_stride) << col_stride << 0;
            return route(OP_reorg, &build_reorg, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(),
                    dense(sz(src.num_samples()), sz(src.k()), sz(src.nr()), sz(src.nc())));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(),
                    dense(sz(dest.num_samples()), sz(dest.k()), sz(dest.nr()), sz(dest.nc())));
            });
        }

        bool reorg_gradient (bool add_to, tensor& grad, const int row_stride, const int col_stride,
            const tensor& gradient_input)
        {
            if (grad.size() == 0 || is_same_object(grad, gradient_input) || grad.nr() % row_stride != 0 ||
                grad.nc() % col_stride != 0 || grad.size() != gradient_input.size() ||
                !admit(OP_reorg_gradient, static_cast<double>(grad.size())))
                return false;
            params p;
            p << sz(grad.num_samples()) << sz(grad.k()) << sz(grad.nr() / row_stride) << row_stride
              << sz(grad.nc() / col_stride) << col_stride << 1;
            return route(OP_reorg_gradient, &build_reorg, p, [&](call& c)
            {
                c.add_input(in_kind::values, gradient_input.host(), dense(sz(gradient_input.num_samples()),
                    sz(gradient_input.k()), sz(gradient_input.nr()), sz(gradient_input.nc())));
                c.add_output(add_to ? out_kind::add : out_kind::assign, grad.host(),
                    dense(sz(grad.num_samples()), sz(grad.k()), sz(grad.nr()), sz(grad.nc())));
            });
        }

        bool copy_tensor (bool add_to, tensor& dest, size_t dest_k_offset, const tensor& src,
            size_t src_k_offset, size_t count_k)
        {
            return copy_tensor(add_to, dest, dest_k_offset, 0, 0, src, src_k_offset, 0, 0, count_k,
                static_cast<size_t>(src.nr()), static_cast<size_t>(src.nc()));
        }

        bool copy_tensor (bool add_to, tensor& dest, size_t dk, size_t dnr, size_t dnc,
            const tensor& src, size_t sk, size_t snr, size_t snc, size_t k, size_t nr, size_t nc)
        {
            const int64_t n = sz(src.num_samples());
            const size_t total = static_cast<size_t>(n) * k * nr * nc;
            if (total == 0 || dest.num_samples() != src.num_samples() ||
                !admit(OP_copy_tensor, static_cast<double>(total)))
                return false;
            params p;
            p << n << static_cast<int64_t>(k) << static_cast<int64_t>(nr) << static_cast<int64_t>(nc);
            const size_t soff = (sk * src.nr() + snr) * src.nc() + snc;
            const size_t doff = (dk * dest.nr() + dnr) * dest.nc() + dnc;
            return route(OP_copy_tensor, &build_copy, p, [&](call& c)
            {
                const int64_t bk = static_cast<int64_t>(k), bnr = static_cast<int64_t>(nr);
                const int64_t bnc = static_cast<int64_t>(nc);
                c.add_input(in_kind::values, src.host() + soff, block(n, bk, bnr, bnc,
                    sz(src.k() * src.nr() * src.nc()), sz(src.nr() * src.nc()), sz(src.nc())));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host() + doff, block(n, bk, bnr, bnc,
                    sz(dest.k() * dest.nr() * dest.nc()), sz(dest.nr() * dest.nc()), sz(dest.nc())));
            });
        }

        bool inv (const tensor& m, resizable_tensor& out)
        {
            const int64_t n = sz(m.num_samples());
            if (n == 0 || m.size() != static_cast<size_t>(n * n) || !admit(OP_inv, static_cast<double>(n) * n * n))
                return false;
            out.set_size(n, n);
            params p;
            p << n;
            return route(OP_inv, &build_inverse, p, [&](call& c)
            {
                c.add_input(in_kind::values, m.host(), dense(1, 1, n, n));
                c.add_output(out_kind::assign, out.host(), dense(1, 1, n, n));
            });
        }

        bool transpose (bool add_to, tensor& dest, const tensor& src)
        {
            if (src.size() == 0 || dest.num_samples() != src.num_samples() || dest.k() != src.k() ||
                dest.nr() != src.nc() || dest.nc() != src.nr() || !admit(OP_transpose, static_cast<double>(src.size())))
                return false;
            params p;
            p << sz(src.num_samples()) << sz(src.k()) << sz(src.nr()) << sz(src.nc()) << 0;
            return route(OP_transpose, &build_permute, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(),
                    dense(sz(src.num_samples()), sz(src.k()), sz(src.nr()), sz(src.nc())));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dest.host(),
                    dense(sz(dest.num_samples()), sz(dest.k()), sz(dest.nr()), sz(dest.nc())));
            });
        }

        bool split_heads (bool add_to, tensor& dst, const tensor& src)
        {
            if (src.size() == 0 || src.k() != 1 || dst.num_samples() != src.num_samples() || dst.nr() != src.nr() ||
                src.nc() != dst.k() * dst.nc() || is_same_object(dst, src) ||
                !admit(OP_split_heads, static_cast<double>(src.size())))
                return false;
            const int64_t B = sz(dst.num_samples()), H = sz(dst.k()), N = sz(dst.nr()), D = sz(dst.nc());
            params p;
            p << B << N << H << D << 1;
            return route(OP_split_heads, &build_permute, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(), dense(B, N, H, D));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dst.host(), dense(B, H, N, D));
            });
        }

        bool merge_heads (bool add_to, tensor& dst, const tensor& src)
        {
            if (src.size() == 0 || dst.k() != 1 || dst.num_samples() != src.num_samples() || dst.nr() != src.nr() ||
                dst.nc() != src.k() * src.nc() || is_same_object(dst, src) ||
                !admit(OP_merge_heads, static_cast<double>(src.size())))
                return false;
            const int64_t B = sz(src.num_samples()), H = sz(src.k()), N = sz(src.nr()), D = sz(src.nc());
            params p;
            p << B << H << N << D << 2;
            return route(OP_merge_heads, &build_permute, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(), dense(B, H, N, D));
                c.add_output(add_to ? out_kind::add : out_kind::assign, dst.host(), dense(B, N, H, D));
            });
        }

        bool embeddings (resizable_tensor& dest, const tensor& src, const tensor& embs)
        {
            const int64_t V = sz(embs.num_samples()), D = sz(embs.k());
            const int64_t ns = sz(dest.num_samples()), nk = sz(dest.k()), nr = sz(dest.nr());
            const int64_t count = ns * nk * nr;
            if (count == 0 || V == 0 || dest.nc() != D || embs.nr() != 1 || embs.nc() != 1 ||
                !admit(OP_embeddings, static_cast<double>(count * D)))
                return false;
            const int64_t cp = bucket(count);
            params p;
            p << V << 0 << D << cp;
            std::vector<int32_t> idx(static_cast<size_t>(cp), 0);
            std::vector<float> weight(static_cast<size_t>(cp), 0.0f);
            const float* tokens = src.host();
            for (int64_t s = 0; s < ns; ++s)
                for (int64_t k = 0; k < nk; ++k)
                    for (int64_t r = 0; r < nr; ++r)
                    {
                        const float t = tokens[((s * src.k() + k) * src.nr() + r) * src.nc()];
                        const size_t i = static_cast<size_t>((s * nk + k) * nr + r);
                        if (t >= 0 && t < static_cast<float>(V))
                        {
                            idx[i] = static_cast<int32_t>(static_cast<unsigned long>(t));
                            weight[i] = 1.0f;
                        }
                    }
            return route(OP_embeddings, &build_gather, p, [&](call& c)
            {
                c.add_input(in_kind::values, embs.host(), dense(1, 1, V, D));
                c.add_indices(idx, static_cast<size_t>(cp));
                c.add_input(in_kind::values, weight.data(), flat(static_cast<size_t>(cp), static_cast<size_t>(cp)));
                c.add_output(out_kind::assign, dest.host(), dense(1, 1, count, D, 1, 1, cp, D));
            });
        }

        bool embeddings_gradient (const tensor& prev, const tensor& gradient_input, tensor& grads,
            const tensor& freqs, float learning_rate, bool scale)
        {
            const int64_t V = sz(grads.num_samples()), D = sz(grads.k());
            const int64_t ns = sz(gradient_input.num_samples()), nk = sz(gradient_input.k());
            const int64_t nr = sz(gradient_input.nr());
            const int64_t count = ns * nk * nr;
            if (count == 0 || V == 0 || gradient_input.nc() != D || grads.nr() != 1 || grads.nc() != 1 ||
                !admit(OP_embeddings_gradient, static_cast<double>(count * D)))
                return false;
            params p;
            p << V << D << count;
            std::vector<int32_t> idx(static_cast<size_t>(count), 0);
            std::vector<float> fs(static_cast<size_t>(count), 0.0f);
            const float* tokens = prev.host();
            const float* f = freqs.host();
            for (int64_t s = 0; s < ns; ++s)
                for (int64_t k = 0; k < nk; ++k)
                    for (int64_t r = 0; r < nr; ++r)
                    {
                        const float t = tokens[((s * prev.k() + k) * prev.nr() + r) * prev.nc()];
                        const size_t i = static_cast<size_t>((s * nk + k) * nr + r);
                        if (t >= 0 && t < static_cast<float>(V))
                        {
                            const unsigned long token = static_cast<unsigned long>(t);
                            idx[i] = static_cast<int32_t>(token);
                            const float freq = f[token];
                            fs[i] = (scale && freq != 0.0f) ? std::min(0.15f, std::max(1.0f / freq, 1.0f)) : 1.0f;
                        }
                    }
            return route(OP_embeddings_gradient, &build_embeddings_gradient, p, [&](call& c)
            {
                c.add_input(in_kind::values, grads.host(), dense(1, 1, V, D));
                c.add_indices(idx, static_cast<size_t>(count));
                c.add_input(in_kind::values, gradient_input.host(), dense(1, 1, count, D));
                c.add_input(in_kind::values, fs.data(), flat(static_cast<size_t>(count), static_cast<size_t>(count)));
                c.add_scalar(learning_rate);
                c.add_output(out_kind::assign, grads.host(), dense(1, 1, V, D));
            });
        }

        bool puzzle_embeddings (resizable_tensor& dest, const tensor& tokens, const tensor& token_table,
            const tensor& puzzle_table, float scale)
        {
            const int64_t vocab = sz(token_table.num_samples()), D = sz(token_table.k());
            const int64_t puzzles = puzzle_table.size() != 0 ? sz(puzzle_table.num_samples()) : 0;
            const int64_t positions = static_cast<int64_t>(tokens.size());
            if (positions == 0 || vocab == 0 || tokens.nc() != 1 || dest.nc() != D ||
                dest.size() != static_cast<size_t>(positions * D) ||
                !admit(OP_puzzle_embeddings, static_cast<double>(positions * D)))
                return false;
            const int64_t cp = bucket(positions);
            params p;
            p << vocab << puzzles << D << cp;
            std::vector<int32_t> idx(static_cast<size_t>(cp), 0);
            std::vector<float> weight(static_cast<size_t>(cp), 0.0f);
            const float* t = tokens.host();
            for (int64_t i = 0; i < positions; ++i)
            {
                const long tok = static_cast<long>(t[i]);
                if (tok >= 0 && tok < vocab + puzzles)
                {
                    idx[static_cast<size_t>(i)] = static_cast<int32_t>(tok);
                    weight[static_cast<size_t>(i)] = scale;
                }
            }
            return route(OP_puzzle_embeddings, &build_gather, p, [&](call& c)
            {
                c.add_input(in_kind::values, token_table.host(), dense(1, 1, vocab, D));
                if (puzzles > 0)
                    c.add_input(in_kind::values, puzzle_table.host(), dense(1, 1, puzzles, D));
                c.add_indices(idx, static_cast<size_t>(cp));
                c.add_input(in_kind::values, weight.data(), flat(static_cast<size_t>(cp), static_cast<size_t>(cp)));
                c.add_output(out_kind::assign, dest.host(), dense(1, 1, positions, D, 1, 1, cp, D));
            });
        }

        bool puzzle_embeddings_gradient (const tensor& tokens, const tensor& gradient_input,
            tensor& token_grad, tensor& puzzle_table, float scale, float puzzle_rate,
            float puzzle_decay)
        {
            const int64_t vocab = sz(token_grad.num_samples()), D = sz(token_grad.k());
            const int64_t puzzles = puzzle_table.size() != 0 ? sz(puzzle_table.num_samples()) : 0;
            const int64_t positions = static_cast<int64_t>(tokens.size());
            if (positions == 0 || vocab == 0 || tokens.nc() != 1 ||
                gradient_input.size() != static_cast<size_t>(positions * D) ||
                puzzle_rate < 0 || puzzle_decay < 0 ||
                !admit(OP_puzzle_embeddings_gradient, static_cast<double>(positions * D)))
                return false;
            std::vector<int32_t> idx_t(static_cast<size_t>(positions), 0), idx_p(static_cast<size_t>(positions), 0);
            std::vector<float> w_t(static_cast<size_t>(positions), 0.0f), w_p(static_cast<size_t>(positions), 0.0f);
            std::vector<float> touched(static_cast<size_t>(std::max<int64_t>(puzzles, 1)), 0.0f);
            bool named = false;
            const float* t = tokens.host();
            for (int64_t i = 0; i < positions; ++i)
            {
                const long tok = static_cast<long>(t[i]);
                const size_t j = static_cast<size_t>(i);
                if (tok >= 0 && tok < vocab)
                {
                    idx_t[j] = static_cast<int32_t>(tok);
                    w_t[j] = scale;
                }
                else if (tok >= vocab && tok - vocab < puzzles)
                {
                    idx_p[j] = static_cast<int32_t>(tok - vocab);
                    w_p[j] = 1.0f;
                    touched[static_cast<size_t>(tok - vocab)] = 1.0f;
                    named = true;
                }
            }
            const bool step_puzzles = puzzles > 0 && named && puzzle_rate != 0;
            params p;
            p << vocab << (step_puzzles ? puzzles : 0) << D << positions;
            return route(OP_puzzle_embeddings_gradient, &build_puzzle_gradient, p, [&](call& c)
            {
                const geometry gw = flat(static_cast<size_t>(positions), static_cast<size_t>(positions));
                c.add_input(in_kind::values, gradient_input.host(), dense(1, 1, positions, D));
                c.add_indices(idx_t, static_cast<size_t>(positions));
                c.add_input(in_kind::values, w_t.data(), gw);
                c.add_output(out_kind::assign, token_grad.host(), dense(1, 1, vocab, D));
                if (step_puzzles)
                {
                    c.add_indices(idx_p, static_cast<size_t>(positions));
                    c.add_input(in_kind::values, w_p.data(), gw);
                    c.add_input(in_kind::values, touched.data(),
                        flat(static_cast<size_t>(puzzles), static_cast<size_t>(puzzles)));
                    c.add_input(in_kind::values, puzzle_table.host(), dense(1, 1, puzzles, D));
                    c.add_scalar(1 - puzzle_rate * puzzle_decay);
                    c.add_scalar(puzzle_rate);
                    c.add_output(out_kind::assign, puzzle_table.host(), dense(1, 1, puzzles, D));
                }
            });
        }

        bool repeat_channels (tensor& dest, const tensor& src, long repeat_factor)
        {
            if (src.size() == 0 || repeat_factor < 2 || dest.size() != src.size() * repeat_factor ||
                !admit(OP_repeat_channels, static_cast<double>(dest.size())))
                return false;
            const int64_t n = sz(src.num_samples()), k = sz(src.k()), inner = sz(src.nr() * src.nc());
            params p;
            p << n << k << repeat_factor << inner << 0;
            return route(OP_repeat_channels, &build_repeat, p, [&](call& c)
            {
                c.add_input(in_kind::values, src.host(), view3(n, k, inner, n, k));
                c.add_output(out_kind::assign, dest.host(), view3(n, k * repeat_factor, inner, n, k * repeat_factor));
            });
        }

        bool accumulate_repeated_channels (tensor& dest, const tensor& src, long repeat_factor)
        {
            if (dest.size() == 0 || repeat_factor < 2 || src.size() != dest.size() * repeat_factor ||
                !admit(OP_accumulate_repeated_channels, static_cast<double>(src.size())))
                return false;
            const int64_t n = sz(dest.num_samples()), k = sz(dest.k()), inner = sz(dest.nr() * dest.nc());
            params p;
            p << n << k << repeat_factor << inner << 1;
            return route(OP_accumulate_repeated_channels, &build_repeat, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src.host(), view3(n, k * repeat_factor, inner, n, k * repeat_factor));
                c.add_output(out_kind::add, dest.host(), view3(n, k, inner, n, k));
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

        bool compute_halt_logits (tensor& logits, tensor& halt_probs, const tensor& state,
            const tensor& halt_params, const tensor& active_mask, long batch_size, long seq_len,
            long num_channels, long feature_dim)
        {
            const size_t positions = static_cast<size_t>(batch_size) * seq_len;
            const size_t total = positions * num_channels * feature_dim;
            if (total == 0 || state.size() != total || halt_params.size() != static_cast<size_t>(feature_dim + 1) ||
                logits.size() != positions || halt_probs.size() != positions || active_mask.size() != positions ||
                !admit(OP_compute_halt_logits, static_cast<double>(total)))
                return false;
            params p;
            p << batch_size << num_channels << seq_len << feature_dim;
            return route(OP_compute_halt_logits, &build_halt_logits, p, [&](call& c)
            {
                const geometry gp = flat(positions, positions);
                c.add_input(in_kind::values, state.host(), dense(batch_size, num_channels, seq_len, feature_dim));
                c.add_input(in_kind::values, halt_params.host(),
                    flat(static_cast<size_t>(feature_dim), static_cast<size_t>(feature_dim)));
                c.add_scalar(halt_params.host()[feature_dim]);
                c.add_input(in_kind::values, active_mask.host(), gp);
                c.add_output(out_kind::assign, logits.host(), gp);
                c.add_output(out_kind::assign, halt_probs.host(), gp);
            });
        }

        bool compute_halt_gradient (tensor& grad_params, tensor& grad_state,
            const tensor& gradient_input, const tensor& state, const tensor& halt_probs,
            const tensor& halt_params, const tensor& active_mask, const tensor& n_steps,
            const tensor& final_state, long current_step, long batch_size, long seq_len,
            long num_channels, long feature_dim, float ponder_penalty, bool gradient_check_mode)
        {
            const size_t positions = static_cast<size_t>(batch_size) * seq_len;
            const size_t total = positions * num_channels * feature_dim;
            if (total == 0 || state.size() != total || gradient_input.size() != total || grad_state.size() != total ||
                final_state.size() != total || grad_params.size() != static_cast<size_t>(feature_dim + 1) ||
                halt_params.size() != grad_params.size() || halt_probs.size() != positions ||
                active_mask.size() != positions || n_steps.size() != positions ||
                !admit(OP_compute_halt_gradient, static_cast<double>(total)))
                return false;
            params p;
            p << batch_size << num_channels << seq_len << feature_dim;
            const float ponder = gradient_check_mode ? 0.0f : ponder_penalty;
            return route(OP_compute_halt_gradient, &build_halt_gradient, p, [&](call& c)
            {
                const geometry gx = dense(batch_size, num_channels, seq_len, feature_dim);
                const geometry gp = flat(positions, positions);
                c.add_input(in_kind::scaled, gradient_input.host(), gx);
                c.add_input(in_kind::values, state.host(), gx);
                c.add_input(in_kind::values, final_state.host(), gx);
                c.add_input(in_kind::values, halt_probs.host(), gp);
                c.add_input(in_kind::values, halt_params.host(),
                    flat(static_cast<size_t>(feature_dim), static_cast<size_t>(feature_dim)));
                c.add_input(in_kind::values, active_mask.host(), gp);
                c.add_input(in_kind::values, n_steps.host(), gp);
                c.add_scalar(static_cast<float>(current_step));
                c.add_scalar(ponder);
                c.add_scalar(gradient_check_mode ? 1.0f : 0.0f);
                c.add_scalar(1e-10f);
                c.add_output(out_kind::add, grad_params.host(), flat(static_cast<size_t>(feature_dim),
                    static_cast<size_t>(feature_dim))).alpha = static_cast<float>(positions);
                c.add_output(out_kind::add, grad_params.host() + feature_dim, flat(1, 1)).alpha =
                    static_cast<float>(positions);
                c.add_output(out_kind::add, grad_state.host(), gx);
                c.finalize = [ponder](call& c)
                {
                    // The penalty and the threshold move with the gradient's scale.
                    const int e = c.in[0].exp;
                    c.in[8].value = ponder * exp2i(e);
                    c.in[10].value = 1e-10f * exp2i(e);
                    for (auto& o : c.out) o.exp = -e;
                    return true;
                };
            });
        }

        bool accumulate_weighted_state (tensor& output, const tensor& state, const tensor& weights,
            const tensor& active_mask, long batch_size, long seq_len, long num_channels, long feature_dim)
        {
            return halt_weighting(OP_accumulate_weighted_state, 0, output, out_kind::add, state, &weights, &active_mask,
                nullptr, 0, batch_size, seq_len, num_channels, feature_dim);
        }

        bool apply_act_depth_scaling (tensor& gradients, const tensor& n_steps, long batch_size,
            long seq_len, long num_channels, long feature_dim, long max_steps)
        {
            return halt_weighting(OP_apply_act_depth_scaling, 1, gradients, out_kind::assign, gradients,
                nullptr, nullptr, &n_steps, 1.0f / static_cast<float>(max_steps), batch_size, seq_len,
                num_channels, feature_dim);
        }

        bool act_accumulate_output_grad (tensor& grad_state, const tensor& gradient_input,
            const tensor& step_weights, const tensor& active_mask, const tensor& n_steps, long step,
            long batch_size, long seq_len, long num_channels, long feature_dim)
        {
            return halt_weighting(OP_act_accumulate_output_grad, 2, grad_state, out_kind::add, gradient_input,
                &step_weights, &active_mask, &n_steps, static_cast<float>(step), batch_size, seq_len, num_channels,
                feature_dim);
        }

        bool broadcast_row (tensor& dest, const tensor& row)
        {
            if (dest.size() == 0 || row.size() != static_cast<size_t>(dest.nc()) ||
                !admit(OP_broadcast_row, static_cast<double>(dest.size())))
                return false;
            const int64_t cols = sz(dest.nc()), rows = static_cast<int64_t>(dest.size()) / cols;
            params p;
            p << rows << cols;
            return route(OP_broadcast_row, &build_broadcast_row, p, [&](call& c)
            {
                c.add_input(in_kind::values, row.host(), flat(static_cast<size_t>(cols), static_cast<size_t>(cols)));
                c.add_output(out_kind::assign, dest.host(), dense(1, 1, rows, cols));
            });
        }

        bool act_mark_active (tensor& active_mask, tensor& active_count,
            const tensor& cumulative_halting, float halt_threshold)
        {
            const size_t n = cumulative_halting.size();
            if (n == 0 || active_mask.size() != n || active_count.size() != 1 ||
                !admit(OP_act_mark_active, static_cast<double>(n)))
                return false;
            const int64_t rows = flat_rows(n);
            const size_t padded = static_cast<size_t>(rows * flat_width);
            params p;
            p << rows << flat_width;
            const bool ok = route(OP_act_mark_active, &build_mark_active, p, [&](call& c)
            {
                c.add_input(in_kind::values, cumulative_halting.host(), flat(n, padded));
                c.add_scalar(halt_threshold);
                c.add_output(out_kind::assign, active_mask.host(), flat(n, padded));
            });
            if (!ok)
                return false;
            const float* m = active_mask.host();
            long active = 0;
            for (size_t i = 0; i < n; ++i)
                active += m[i] > 0.5f ? 1 : 0;
            active_count.host()[0] = static_cast<float>(active);
            return true;
        }

        bool act_update_halting (tensor& cumulative_halting, tensor& remainders, tensor& n_steps,
            tensor& step_weights, const tensor& halt_probs, const tensor& active_mask, long step,
            long max_steps, float halt_threshold)
        {
            const size_t n = cumulative_halting.size();
            if (n == 0 || remainders.size() != n || n_steps.size() != n || step_weights.size() != n ||
                halt_probs.size() != n || active_mask.size() != n ||
                !admit(OP_act_update_halting, static_cast<double>(n)))
                return false;
            const int64_t rows = flat_rows(n);
            const size_t padded = static_cast<size_t>(rows * flat_width);
            params p;
            p << rows << flat_width;
            return route(OP_act_update_halting, &build_halt_update, p, [&](call& c)
            {
                const geometry gf = flat(n, padded);
                c.add_input(in_kind::values, cumulative_halting.host(), gf);
                c.add_input(in_kind::values, remainders.host(), gf);
                c.add_input(in_kind::values, n_steps.host(), gf);
                c.add_input(in_kind::values, step_weights.host(), gf);
                c.add_input(in_kind::values, halt_probs.host(), gf);
                c.add_input(in_kind::values, active_mask.host(), gf);
                c.add_scalar(static_cast<float>(step + 1));
                c.add_scalar(step == max_steps - 1 ? 1.0f : 0.0f);
                c.add_scalar(halt_threshold);
                c.add_output(out_kind::assign, cumulative_halting.host(), gf);
                c.add_output(out_kind::assign, remainders.host(), gf);
                c.add_output(out_kind::assign, n_steps.host(), gf);
                c.add_output(out_kind::assign, step_weights.host(), gf);
            });
        }

        bool act_gather_final_state (tensor& final_state, const tensor& step_state,
            const tensor& n_steps, long step, long batch_size, long seq_len, long num_channels,
            long feature_dim)
        {
            const size_t positions = static_cast<size_t>(batch_size) * seq_len;
            const size_t total = positions * num_channels * feature_dim;
            if (total == 0 || final_state.size() != total || step_state.size() != total ||
                n_steps.size() != positions ||
                !admit(OP_act_gather_final_state, static_cast<double>(total)))
                return false;
            params p;
            p << batch_size << num_channels << seq_len << feature_dim;
            return route(OP_act_gather_final_state, &build_gather_final, p, [&](call& c)
            {
                const geometry gx = dense(batch_size, num_channels, seq_len, feature_dim);
                c.add_input(in_kind::values, step_state.host(), gx);
                c.add_input(in_kind::values, final_state.host(), gx);
                c.add_input(in_kind::values, n_steps.host(), flat(positions, positions));
                c.add_scalar(static_cast<float>(step + 1));
                c.add_output(out_kind::assign, final_state.host(), gx);
            });
        }

        bool apply_rotary_positional_embedding (bool is_backward, resizable_tensor& data,
            const resizable_tensor& cos_cache, const resizable_tensor& sin_cache)
        {
            const int64_t B = sz(data.num_samples()), H = sz(data.k()), T = sz(data.nr()), D = sz(data.nc());
            const int64_t half = D / 2;
            if (data.size() == 0 || half == 0 || cos_cache.nr() != T || cos_cache.nc() != half ||
                sin_cache.nr() != T || sin_cache.nc() != half ||
                !admit(OP_rotary_embedding, static_cast<double>(data.size())))
                return false;
            const int64_t pt = bucket(T);
            params p;
            p << B * H << pt << D << half;
            return route(OP_rotary_embedding, &build_rotary, p, [&](call& c)
            {
                const geometry gx = dense(1, B * H, T, D, 1, B * H, pt, D);
                const geometry gt = dense(1, 1, T, half, 1, 1, pt, half);
                c.add_input(in_kind::scaled, data.host(), gx);
                c.add_input(in_kind::values, cos_cache.host(), gt);
                c.add_input(in_kind::values, sin_cache.host(), gt);
                c.add_scalar(is_backward ? -1.0f : 1.0f);
                c.add_output(out_kind::assign, data.host(), gx);
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

        bool conv (bool add_to_output, tensor& output, const tensor& data, const tensor& filters,
            int stride_y, int stride_x, int padding_y, int padding_x)
        {
            const int64_t n = sz(data.num_samples()), ch = sz(data.k()), h = sz(data.nr()), w = sz(data.nc());
            const int64_t o = sz(filters.num_samples()), kh = sz(filters.nr()), kw = sz(filters.nc());
            if (data.size() == 0 || filters.k() != ch || stride_y <= 0 || stride_x <= 0 ||
                is_same_object(output, data) || is_same_object(output, filters))
                return false;
            const int64_t oh = 1 + (h + 2 * padding_y - kh) / stride_y, ow = 1 + (w + 2 * padding_x - kw) / stride_x;
            if (output.num_samples() != n || output.k() != o || output.nr() != oh || output.nc() != ow ||
                !admit(OP_conv, static_cast<double>(n) * o * oh * ow * ch * kh * kw))
                return false;
            params p;
            p << n << ch << h << w << o << kh << kw << stride_y << stride_x << padding_y << padding_x << oh << ow;
            const int target = product_target(ch * kh * kw);
            return route(OP_conv, &build_conv, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, data.host(), dense(n, ch, h, w)).target = target;
                input& f = c.add_input(in_kind::scaled, filters.host(), dense(o, ch, kh, kw));
                f.target = target;
                f.storage = &storage_access::of(filters);
                c.add_output(add_to_output ? out_kind::add : out_kind::assign, output.host(), dense(n, o, oh, ow));
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool conv_gradient_for_data (bool add_to_output, const tensor& gradient_input,
            const tensor& filters, tensor& data_gradient, int stride_y, int stride_x,
            int padding_y, int padding_x)
        {
            const int64_t n = sz(data_gradient.num_samples()), ch = sz(data_gradient.k());
            const int64_t h = sz(data_gradient.nr()), w = sz(data_gradient.nc());
            const int64_t o = sz(filters.num_samples()), kh = sz(filters.nr()), kw = sz(filters.nc());
            if (data_gradient.size() == 0 || filters.k() != ch || stride_y <= 0 || stride_x <= 0 ||
                is_same_object(data_gradient, filters) || is_same_object(data_gradient, gradient_input))
                return false;
            const int64_t oh = 1 + (h + 2 * padding_y - kh) / stride_y, ow = 1 + (w + 2 * padding_x - kw) / stride_x;
            if (gradient_input.num_samples() != n || gradient_input.k() != o || gradient_input.nr() != oh ||
                gradient_input.nc() != ow ||
                !admit(OP_conv_gradient_for_data, static_cast<double>(n) * o * oh * ow * ch * kh * kw))
                return false;
            params p;
            p << n << ch << h << w << o << kh << kw << stride_y << stride_x << padding_y << padding_x << oh << ow;
            const int target = product_target(o * kh * kw);
            return route(OP_conv_gradient_for_data, &build_conv_data, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, gradient_input.host(), dense(n, o, oh, ow)).target = target;
                input& f = c.add_input(in_kind::scaled, filters.host(), dense(o, ch, kh, kw));
                f.target = target;
                f.storage = &storage_access::of(filters);
                c.add_output(add_to_output ? out_kind::add : out_kind::assign, data_gradient.host(),
                    dense(n, ch, h, w));
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool conv_gradient_for_filters (bool add_to_output, const tensor& gradient_input,
            const tensor& data, tensor& filters_gradient, int stride_y, int stride_x,
            int padding_y, int padding_x)
        {
            const int64_t n = sz(data.num_samples()), ch = sz(data.k()), h = sz(data.nr()), w = sz(data.nc());
            const int64_t o = sz(filters_gradient.num_samples()), kh = sz(filters_gradient.nr());
            const int64_t kw = sz(filters_gradient.nc());
            if (data.size() == 0 || filters_gradient.k() != ch || stride_y <= 0 || stride_x <= 0 ||
                is_same_object(filters_gradient, data) || is_same_object(filters_gradient, gradient_input))
                return false;
            const int64_t oh = 1 + (h + 2 * padding_y - kh) / stride_y, ow = 1 + (w + 2 * padding_x - kw) / stride_x;
            if (gradient_input.num_samples() != n || gradient_input.k() != o || gradient_input.nr() != oh ||
                gradient_input.nc() != ow ||
                !admit(OP_conv_gradient_for_filters, static_cast<double>(n) * o * oh * ow * ch * kh * kw))
                return false;
            params p;
            p << n << ch << h << w << o << kh << kw << stride_y << stride_x << padding_y << padding_x << oh << ow;
            const int target = product_target(n * oh * ow);
            return route(OP_conv_gradient_for_filters, &build_conv_filters, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, gradient_input.host(), dense(n, o, oh, ow)).target = target;
                c.add_input(in_kind::scaled, data.host(), dense(n, ch, h, w)).target = target;
                c.add_output(add_to_output ? out_kind::add : out_kind::assign, filters_gradient.host(),
                    dense(o, ch, kh, kw));
                c.finalize = [](call& c) { c.out[0].exp = -(c.in[0].exp + c.in[1].exp); return true; };
            });
        }

        bool pooling (bool max_pooling, int window_height, int window_width, int stride_y,
            int stride_x, int padding_y, int padding_x, tensor& dest, const tensor& src)
        {
            const int64_t planes = sz(src.num_samples() * src.k()), h = sz(src.nr()), w = sz(src.nc());
            const int64_t oh = 1 + (h + 2 * padding_y - window_height) / stride_y;
            const int64_t ow = 1 + (w + 2 * padding_x - window_width) / stride_x;
            if (src.size() == 0 || dest.num_samples() != src.num_samples() || dest.k() != src.k() || dest.nr() != oh ||
                dest.nc() != ow || !admit(OP_pooling, static_cast<double>(src.size())))
                return false;
            params p;
            p << planes << h << w << window_height << window_width << stride_y << stride_x << padding_y << padding_x
              << (max_pooling ? 1 : 0);
            return route(OP_pooling, &build_pool, p, [&](call& c)
            {
                c.add_input(in_kind::scaled, src.host(), dense(planes, 1, h, w));
                c.add_output(out_kind::assign, dest.host(), dense(planes, 1, oh, ow));
                c.finalize = [](call& c) { c.out[0].exp = -c.in[0].exp; return true; };
            });
        }

        bool pooling_gradient (bool max_pooling, int window_height, int window_width,
            int stride_y, int stride_x, int padding_y, int padding_x,
            const tensor& gradient_input, const tensor& dest, const tensor& src, tensor& grad)
        {
            const int64_t planes = sz(src.num_samples() * src.k()), h = sz(src.nr()), w = sz(src.nc());
            const int64_t oh = sz(dest.nr()), ow = sz(dest.nc());
            if (src.size() == 0 || !have_same_dimensions(gradient_input, dest) || !have_same_dimensions(src, grad) ||
                dest.num_samples() * dest.k() != planes || !admit(OP_pooling_gradient, static_cast<double>(src.size())))
                return false;
            // The average adjoint is a dense matrix: kept to sizes where it stays small.
            if (!max_pooling && oh * ow * h * w > (int64_t(1) << 22))
                return false;
            params p;
            p << planes << h << w << window_height << window_width << stride_y << stride_x << padding_y << padding_x
              << (max_pooling ? 1 : 0) << oh << ow;
            return route(OP_pooling_gradient, &build_pool_gradient, p, [&](call& c)
            {
                if (max_pooling)
                    c.add_input(in_kind::scaled, src.host(), dense(planes, 1, h, w));
                c.add_input(in_kind::scaled, gradient_input.host(), dense(planes, 1, oh, ow));
                c.add_output(out_kind::add, grad.host(), dense(planes, 1, h, w));
                const size_t gi_index = max_pooling ? 1 : 0;
                c.finalize = [gi_index](call& c) { c.out[0].exp = -c.in[gi_index].exp; return true; };
            });
        }

        bool mean_squared_per_pixel (const tensor& output,
            const std::function<void(float*)>& fill_targets, tensor& grad, double& loss)
        {
            const size_t n = output.size();
            if (n == 0 || grad.size() != n || !admit(OP_mean_squared, static_cast<double>(n)))
                return false;
            std::vector<float> targets;
            const int64_t rows = flat_rows(n);
            const size_t padded = static_cast<size_t>(rows * flat_width);
            params p;
            p << rows << flat_width;
            const double scale = 1.0 / static_cast<double>(n);
            std::vector<float> mean;
            const bool ok = route(OP_mean_squared, &build_mean_squared, p, [&](call& c)
            {
                targets.resize(n);
                fill_targets(targets.data());
                c.add_input(in_kind::values, output.host(), flat(n, padded));
                c.add_input(in_kind::values, targets.data(), flat(n, padded));
                c.add_output(out_kind::assign, grad.host(), flat(n, padded)).alpha = -static_cast<float>(scale);
                auto& m = c.add_output(out_kind::staged, nullptr, flat(1, 1));
                m.staged = &mean;
                m.alpha = static_cast<float>(padded);
            });
            if (!ok)
                return false;
            loss = mean[0] * scale;
            return true;
        }

    } // namespace npu
} // namespace dlib

#endif // DLIB_USE_NPU

#endif // DLIB_DNN_NPU_DLIB_CPP_
