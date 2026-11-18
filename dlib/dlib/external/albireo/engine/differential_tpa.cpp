// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "differential_tpa.h"

#include <cmath>

#include "qkernels.h"

// MSVC names no macro for FMA, which /arch:AVX2 enables with AVX2.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define ALBIREO_DTPA_AVX2 1
#include <immintrin.h>
#endif

namespace albireo
{

    namespace
    {
        // y = c x when first, y += c x otherwise.
        inline void scaled(float* y, float c, const float* x, long n, bool first)
        {
            long i = 0;
#if ALBIREO_DTPA_AVX2
            const __m256 vc = _mm256_set1_ps(c);
            if (first)
                for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(vc, _mm256_loadu_ps(x + i)));
            else
                for (; i + 8 <= n; i += 8)
                    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(vc, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#endif
            if (first)
                for (; i < n; ++i) y[i] = c * x[i];
            else
                for (; i < n; ++i) y[i] += c * x[i];
        }
    }

    void tpa_combine(const float* a, const float* b, long n_heads, long rank, long head_dim, float* out)
    {
        const float share = 1.0f / static_cast<float>(rank);
        for (long h = 0; h < n_heads; ++h)
            for (long r = 0; r < rank; ++r)
                scaled(out + h * head_dim, a[h * rank + r] * share, b + r * head_dim, head_dim, r == 0);
    }

    void tpa_combine_rows(const float* a, long a_stride, const float* b, long b_stride, long n, long n_heads,
        long rank, long head_dim, float* out, long out_stride)
    {
        for (long t = 0; t < n; ++t)
            tpa_combine(a + t * a_stride, b + t * b_stride, n_heads, rank, head_dim, out + t * out_stride);
    }

    float differential_lambda_init(long depth)
    {
        return 0.8f - 0.6f * std::exp(-0.3f * static_cast<float>(depth));
    }

    float differential_lambda(const float* lq1, const float* lk1, const float* lq2, const float* lk2, long n,
        float lambda_init)
    {
        return std::exp(qk_dot_f32(lq1, lk1, n)) - std::exp(qk_dot_f32(lq2, lk2, n)) + lambda_init;
    }

    void head_rms_norm(float* x, long n_heads, long width, const float* weight, float eps)
    {
        for (long h = 0; h < n_heads; ++h)
        {
            float* row = x + h * width;
            const float scale = 1.0f / std::sqrt(qk_dot_f32(row, row, width) / static_cast<float>(width) + eps);
            if (weight)
                for (long i = 0; i < width; ++i) row[i] *= scale * weight[i];
            else
                qk_scale(row, scale, width);
        }
    }

    void differential_combine(const float* maps, long n_heads, long width, float lambda, const float* weight,
        float eps, float scale, float* out)
    {
        for (long h = 0; h < n_heads; ++h)
        {
            const float* first = maps + 2 * h * width;
            const float* second = first + width;
            float* row = out + h * width;
            // The head h of out ends before the second map of head h of maps starts: out may be maps.
            for (long i = 0; i < width; ++i) row[i] = first[i] - lambda * second[i];
            const float norm = scale / std::sqrt(qk_dot_f32(row, row, width) / static_cast<float>(width) + eps);
            if (weight)
                for (long i = 0; i < width; ++i) row[i] *= norm * weight[i];
            else
                qk_scale(row, norm, width);
        }
    }

}
