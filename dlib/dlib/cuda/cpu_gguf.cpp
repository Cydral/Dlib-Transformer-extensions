// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CPU_GGUF_CPP_
#define DLIB_CPU_GGUF_CPP_

#include "cpu_gguf.h"
#include "gguf_blocks.h"

#include <cmath>
#include <limits>
#include <vector>

namespace dlib
{
    namespace cpu
    {

    // ----------------------------------------------------------------------------------------

        void gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && rows > 0 && cols > 0);
            DLIB_CASSERT(static_cast<long long>(rows) <= std::numeric_limits<int>::max()
                && static_cast<long long>(cols) <= std::numeric_limits<int>::max());
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights));
            DLIB_CASSERT(src.size() % static_cast<size_t>(cols) == 0);
            const long n = static_cast<long>(src.size() / static_cast<size_t>(cols));
            DLIB_CASSERT(dest.size() == static_cast<size_t>(n) * static_cast<size_t>(rows));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(rows) * row_bytes);
            if (n == 0) return;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            const float* x = src.host();
            float* y = dest.host_write_only();
            const long groups = cols / 32;
            float v[32];
            std::vector<float> acc(static_cast<size_t>(n));
            for (long r = 0; r < rows; ++r)
            {
                const uint8_t* wr = w + static_cast<size_t>(r) * row_bytes;
                for (auto& a : acc) a = 0;
                for (long g = 0; g < groups; ++g)
                {
                    gguf_blocks::decode_group(type, wr, g, v);
                    for (long t = 0; t < n; ++t)
                    {
                        const float* xt = x + static_cast<size_t>(t) * static_cast<size_t>(cols) + g * 32;
                        float s = 0;
                        for (int i = 0; i < 32; ++i) s += v[i] * xt[i];
                        acc[static_cast<size_t>(t)] += s;
                    }
                }
                for (long t = 0; t < n; ++t)
                    y[static_cast<size_t>(t) * static_cast<size_t>(rows) + r] = acc[static_cast<size_t>(t)];
            }
        }

    // ----------------------------------------------------------------------------------------

        void gguf_dequantize (
            tensor& dest,
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && first >= 0 && count >= 0 && cols > 0);
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, weights));
            DLIB_CASSERT(dest.size() == static_cast<size_t>(count) * static_cast<size_t>(cols));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(first + count) * row_bytes);
            if (count == 0) return;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            float* d = dest.host_write_only();
            const long groups = cols / 32;
            for (long r = 0; r < count; ++r)
            {
                const uint8_t* wr = w + static_cast<size_t>(first + r) * row_bytes;
                float* out = d + static_cast<size_t>(r) * static_cast<size_t>(cols);
                for (long g = 0; g < groups; ++g) gguf_blocks::decode_group(type, wr, g, out + g * 32);
            }
        }

    // ----------------------------------------------------------------------------------------

        void rms_norm_rows (
            tensor& dest,
            const tensor& src,
            const tensor& gamma,
            float eps
        )
        {
            const size_t width = gamma.size();
            DLIB_CASSERT(width > 0 && src.size() % width == 0 && dest.size() == src.size() && eps >= 0);
            const size_t rows = src.size() / width;
            const float* x = src.host();
            const float* g = gamma.host();
            // dest may be src: each value is read before it is written.
            float* y = is_same_object(dest, src) ? dest.host() : dest.host_write_only();
            for (size_t r = 0; r < rows; ++r)
            {
                const float* xr = x + r * width;
                float* yr = y + r * width;
                float s = 0;
                for (size_t i = 0; i < width; ++i) s += xr[i] * xr[i];
                const float scale = 1.0f / std::sqrt(s / static_cast<float>(width) + eps);
                for (size_t i = 0; i < width; ++i) yr[i] = xr[i] * scale * g[i];
            }
        }

    // ----------------------------------------------------------------------------------------

        void silu_multiply (
            tensor& gate,
            const tensor& up
        )
        {
            DLIB_CASSERT(gate.size() == up.size() && !is_same_object(gate, up));
            float* g = gate.host();
            const float* u = up.host();
            for (size_t i = 0; i < gate.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        }

    // ----------------------------------------------------------------------------------------

    }
}

#endif // DLIB_CPU_GGUF_CPP_
