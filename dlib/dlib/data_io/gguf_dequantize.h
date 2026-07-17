// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF tensor dequantization to float.
//
// Converts the raw on-disk bytes of a GGUF tensor (as returned by gguf_reader) into a
// flat float buffer, in the tensor's own logical orientation (ggml dims[0] contiguous).
// Matrix transposition, RoPE head permutation and fused packing for the Dlib layers are
// the repacking stage's responsibility, not this one.
//
// Supported now: F32, F16, the legacy blocks (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1), the
// k-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) and IQ4_NL (4-bit with a fixed non-linear
// value table, used as the fallback when a row is not a multiple of the k-quant super-block).
// The _S/_M/_L mixes (e.g. Q4_K_M) are not tensor types: each tensor in such a file already
// carries one of the types above, so they are handled with no extra code. The grid-based
// i-quants (IQ1/IQ2/IQ3, IQ4_XS) are not covered and throw.

#ifndef DLIB_GGUF_DEQUANTIZE_H_
#define DLIB_GGUF_DEQUANTIZE_H_

#include "gguf_dequantize_abstract.h"

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
                /* value = (1 + m/1024) * 2^(-14 - e), so the float exponent field is
                   127 - 14 - e. An off-by-one here halves every subnormal, and k-quant
                   super-block scales (d ~ block_max / 1e3) do land in the subnormal
                   range for low-magnitude blocks, silently halving their weights. */
                bits = sign | (static_cast<uint32_t>(127 - 14 - e) << 23) | (m << 13);
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

    /* Read a little-endian fp16 at p and convert to float. */
    inline float gguf_rd_half(const uint8_t* p)
    {
        uint16_t h;
        std::memcpy(&h, p, 2);
        return gguf_half_to_float(h);
    }

    /* Legacy block formats, 32 values per block. The "_0" forms are symmetric (scale only),
       w = d * (q - offset); the "_1" forms are asymmetric, w = d * q + m. The low nibble of
       qs[j] holds value j, the high nibble holds value j+16; the 5-bit forms carry the extra
       bit per value in qh. */

    inline void gguf_dq_q4_0(const uint8_t* p, float* y)
    {
        const float d = gguf_rd_half(p);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j)
        {
            y[j]      = ((qs[j] & 0x0F) - 8) * d;
            y[j + 16] = ((qs[j] >>   4) - 8) * d;
        }
    }

    inline void gguf_dq_q4_1(const uint8_t* p, float* y)
    {
        const float d = gguf_rd_half(p);
        const float m = gguf_rd_half(p + 2);
        const uint8_t* qs = p + 4;
        for (int j = 0; j < 16; ++j)
        {
            y[j]      = (qs[j] & 0x0F) * d + m;
            y[j + 16] = (qs[j] >>   4) * d + m;
        }
    }

    inline void gguf_dq_q5_0(const uint8_t* p, float* y)
    {
        const float d = gguf_rd_half(p);
        uint32_t qh;
        std::memcpy(&qh, p + 2, 4);
        const uint8_t* qs = p + 6;
        for (int j = 0; j < 16; ++j)
        {
            const uint8_t h0 = static_cast<uint8_t>(((qh >> (j +  0)) << 4) & 0x10u);
            const uint8_t h1 = static_cast<uint8_t>(( qh >> (j + 12))       & 0x10u);
            y[j]      = (((qs[j] & 0x0F) | h0) - 16) * d;
            y[j + 16] = (((qs[j] >>   4) | h1) - 16) * d;
        }
    }

    inline void gguf_dq_q5_1(const uint8_t* p, float* y)
    {
        const float d = gguf_rd_half(p);
        const float m = gguf_rd_half(p + 2);
        uint32_t qh;
        std::memcpy(&qh, p + 4, 4);
        const uint8_t* qs = p + 8;
        for (int j = 0; j < 16; ++j)
        {
            const uint8_t h0 = static_cast<uint8_t>(((qh >> (j +  0)) << 4) & 0x10u);
            const uint8_t h1 = static_cast<uint8_t>(( qh >> (j + 12))       & 0x10u);
            y[j]      = ((qs[j] & 0x0F) | h0) * d + m;
            y[j + 16] = ((qs[j] >>   4) | h1) * d + m;
        }
    }

    inline void gguf_dq_q8_0(const uint8_t* p, float* y)
    {
        const float d = gguf_rd_half(p);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
        for (int j = 0; j < 32; ++j) y[j] = qs[j] * d;
    }

    inline void gguf_dq_q8_1(const uint8_t* p, float* y)
    {
        /* Intermediate dot-product format; the stored group sum is not needed here. */
        const float d = gguf_rd_half(p);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 4);
        for (int j = 0; j < 32; ++j) y[j] = qs[j] * d;
    }

    /* IQ4_NL, 32 values per block: one fp16 scale then 16 bytes of nibbles indexing a
       fixed non-linear table of 16 signed values, w = d * table[q]. */
    inline void gguf_dq_iq4_nl(const uint8_t* p, float* y)
    {
        static const int8_t kvalues[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        const float d = gguf_rd_half(p);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < 16; ++j)
        {
            y[j]      = d * kvalues[qs[j] & 0x0F];
            y[j + 16] = d * kvalues[qs[j] >>   4];
        }
    }

    /* K-quant super-block formats, 256 values per super-block split into sub-blocks. Sub-block
       scales (and mins) are quantized to 6 bits and rescaled by the fp16 super-block factors d
       (and dmin). Q2/3/4/5_K are asymmetric (scale and min); Q6_K is symmetric (scale only). */

    /* Unpack the 6-bit sub-block scale and min at index j from the 12-byte packed array. */
    inline void gguf_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m)
    {
        if (j < 4)
        {
            d = q[j] & 63;
            m = q[j + 4] & 63;
        }
        else
        {
            d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
        }
    }

    inline void gguf_dq_q2_K(const uint8_t* p, float* y)
    {
        const uint8_t* scales = p;
        const uint8_t* q = p + 16;
        const float d    = gguf_rd_half(p + 80);
        const float dmin = gguf_rd_half(p + 82);
        int is = 0;
        for (int n = 0; n < 256; n += 128)
        {
            int shift = 0;
            for (int j = 0; j < 4; ++j)
            {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0x0F), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0x0F); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }

    inline void gguf_dq_q3_K(const uint8_t* p, float* y)
    {
        const uint8_t* hm = p;
        const uint8_t* q = p + 32;
        const float d_all = gguf_rd_half(p + 108);

        /* Unpack sixteen 6-bit sub-block scales packed into 12 bytes. */
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint32_t aux[4];
        std::memcpy(aux, p + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] =  (aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] =  (aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < 256; n += 128)
        {
            int shift = 0;
            for (int j = 0; j < 4; ++j)
            {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    inline void gguf_dq_q4_K(const uint8_t* p, float* y)
    {
        const float d    = gguf_rd_half(p);
        const float dmin = gguf_rd_half(p + 2);
        const uint8_t* scales = p + 4;
        const uint8_t* qs = p + 16;
        int is = 0;
        uint8_t sc, mn;
        for (int j = 0; j < 256; j += 64)
        {
            gguf_get_scale_min_k4(is + 0, scales, sc, mn);
            const float d1 = d * sc, m1 = dmin * mn;
            gguf_get_scale_min_k4(is + 1, scales, sc, mn);
            const float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (qs[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (qs[l] >>   4) - m2;
            qs += 32; is += 2;
        }
    }

    inline void gguf_dq_q5_K(const uint8_t* p, float* y)
    {
        const float d    = gguf_rd_half(p);
        const float dmin = gguf_rd_half(p + 2);
        const uint8_t* scales = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* qs = p + 48;
        int is = 0;
        uint8_t sc, mn, u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64)
        {
            gguf_get_scale_min_k4(is + 0, scales, sc, mn);
            const float d1 = d * sc, m1 = dmin * mn;
            gguf_get_scale_min_k4(is + 1, scales, sc, mn);
            const float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l)
                *y++ = d1 * ((qs[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *y++ = d2 * ((qs[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            qs += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }

    inline void gguf_dq_q6_K(const uint8_t* p, float* y)
    {
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
        const float d = gguf_rd_half(p + 208);
        for (int n = 0; n < 256; n += 128)
        {
            for (int l = 0; l < 32; ++l)
            {
                const int is = l / 16;
                const int q1 = ((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l +  0] = d * sc[is + 0] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    }

    inline void gguf_dq_q8_K(const uint8_t* p, float* y)
    {
        /* Intermediate format; d is a 32-bit float, the per-group sums are not needed here. */
        float d;
        std::memcpy(&d, p, 4);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 4);
        for (int j = 0; j < 256; ++j) y[j] = qs[j] * d;
    }

    /* Dequantize an array of fixed-size blocks using a per-block function. */
    inline void gguf_dq_blocks(const gguf_tensor_info& t, const std::vector<uint8_t>& raw,
        std::vector<float>& out, uint64_t qk, uint64_t bytes, void (*fn)(const uint8_t*, float*))
    {
        const uint64_t n = t.n_elements();
        if (n % qk != 0)
            throw std::runtime_error("gguf_dequantize: element count not a multiple of "
                + std::to_string(qk) + " for '" + t.name + "'");
        const uint64_t nb = n / qk;
        if (raw.size() < nb * bytes)
            throw std::runtime_error("gguf_dequantize: short buffer for '" + t.name + "'");
        const uint8_t* p = raw.data();
        float* o = out.data();
        for (uint64_t b = 0; b < nb; ++b)
        {
            fn(p, o);
            p += static_cast<size_t>(bytes);
            o += static_cast<size_t>(qk);
        }
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
        case ggml_type::Q4_0: gguf_dq_blocks(t, raw, out,  32,  18, gguf_dq_q4_0); break;
        case ggml_type::Q4_1: gguf_dq_blocks(t, raw, out,  32,  20, gguf_dq_q4_1); break;
        case ggml_type::Q5_0: gguf_dq_blocks(t, raw, out,  32,  22, gguf_dq_q5_0); break;
        case ggml_type::Q5_1: gguf_dq_blocks(t, raw, out,  32,  24, gguf_dq_q5_1); break;
        case ggml_type::Q8_0: gguf_dq_blocks(t, raw, out,  32,  34, gguf_dq_q8_0); break;
        case ggml_type::Q8_1: gguf_dq_blocks(t, raw, out,  32,  36, gguf_dq_q8_1); break;
        case ggml_type::Q2_K: gguf_dq_blocks(t, raw, out, 256,  84, gguf_dq_q2_K); break;
        case ggml_type::Q3_K: gguf_dq_blocks(t, raw, out, 256, 110, gguf_dq_q3_K); break;
        case ggml_type::Q4_K: gguf_dq_blocks(t, raw, out, 256, 144, gguf_dq_q4_K); break;
        case ggml_type::Q5_K: gguf_dq_blocks(t, raw, out, 256, 176, gguf_dq_q5_K); break;
        case ggml_type::Q6_K: gguf_dq_blocks(t, raw, out, 256, 210, gguf_dq_q6_K); break;
        case ggml_type::Q8_K: gguf_dq_blocks(t, raw, out, 256, 292, gguf_dq_q8_K); break;
        case ggml_type::IQ4_NL: gguf_dq_blocks(t, raw, out, 32, 18, gguf_dq_iq4_nl); break;
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
