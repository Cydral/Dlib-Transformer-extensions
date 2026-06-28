// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF tensor dequantization to float.
//
// Converts the raw on-disk bytes of a GGUF tensor (as returned by gguf_reader) into a
// flat float buffer, in the tensor's own logical orientation (ggml dims[0] contiguous).
// Matrix transposition, RoPE head permutation and fused packing for the Dlib layers are
// the repacking stage's responsibility, not this one.
//
// Supported now: F32, F16, Q8_0. Other quantization formats throw until added.

#ifndef DLIB_GGUF_DEQUANTIZE_H_
#define DLIB_GGUF_DEQUANTIZE_H_

#include "gguf_reader.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace dlib
{
    /* IEEE half-precision (binary16) to float. Handles zero, subnormal, inf and NaN. */
    inline float gguf_half_to_float(uint16_t h)
    {
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        const uint32_t exp = (h >> 10) & 0x1Fu;
        const uint32_t mant = h & 0x3FFu;
        uint32_t bits;
        if (exp == 0)
        {
            if (mant == 0) bits = sign;                       // signed zero
            else
            {
                int e = 0;
                uint32_t m = mant;
                while ((m & 0x400u) == 0) { m <<= 1; ++e; }   // normalize the subnormal
                m &= 0x3FFu;
                bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
            }
        }
        else if (exp == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mant << 13);         // inf / NaN
        }
        else
        {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    /* Dequantize one tensor into out (resized to t.n_elements()). */
    inline void gguf_dequantize(const gguf_tensor_info& t, const std::vector<uint8_t>& raw,
        std::vector<float>& out)
    {
        const uint64_t n = t.n_elements();
        out.resize(static_cast<size_t>(n));

        switch (t.type)
        {
        case ggml_type::F32:
        {
            if (raw.size() < n * 4) throw std::runtime_error("gguf_dequantize: short F32 buffer for '" + t.name + "'");
            std::memcpy(out.data(), raw.data(), static_cast<size_t>(n) * 4);
            break;
        }
        case ggml_type::F16:
        {
            if (raw.size() < n * 2) throw std::runtime_error("gguf_dequantize: short F16 buffer for '" + t.name + "'");
            const uint8_t* p = raw.data();
            for (uint64_t i = 0; i < n; ++i)
            {
                uint16_t h;
                std::memcpy(&h, p + i * 2, 2);
                out[static_cast<size_t>(i)] = gguf_half_to_float(h);
            }
            break;
        }
        case ggml_type::Q8_0:
        {
            /* Block of 32 values: one fp16 scale d, then 32 int8 quants; 34 bytes total.
               Dequantized value = q * d. */
            constexpr uint64_t QK = 32;
            if (n % QK != 0) throw std::runtime_error("gguf_dequantize: Q8_0 element count not a multiple of 32 for '" + t.name + "'");
            const uint64_t nb = n / QK;
            if (raw.size() < nb * 34) throw std::runtime_error("gguf_dequantize: short Q8_0 buffer for '" + t.name + "'");
            const uint8_t* p = raw.data();
            for (uint64_t b = 0; b < nb; ++b)
            {
                uint16_t dh;
                std::memcpy(&dh, p, 2);
                const float d = gguf_half_to_float(dh);
                const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
                float* o = out.data() + static_cast<size_t>(b * QK);
                for (uint64_t j = 0; j < QK; ++j) o[j] = static_cast<float>(qs[j]) * d;
                p += 34;
            }
            break;
        }
        default:
            throw std::runtime_error("gguf_dequantize: ggml type not yet supported: "
                + std::to_string(static_cast<uint32_t>(t.type)));
        }
    }

    /* Convenience: read a tensor from the reader and dequantize it in one call. */
    inline void gguf_read_dequantized(gguf_reader& g, const gguf_tensor_info& t, std::vector<float>& out)
    {
        std::vector<uint8_t> raw;
        g.read_tensor_raw(t, raw);
        gguf_dequantize(t, raw, out);
    }
}

#endif // DLIB_GGUF_DEQUANTIZE_H_
