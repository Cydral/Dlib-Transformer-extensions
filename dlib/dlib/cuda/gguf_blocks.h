// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GGUF_BLOCKS_H_
#define DLIB_GGUF_BLOCKS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define DLIB_GGUF_HD __host__ __device__ __forceinline__
#else
#define DLIB_GGUF_HD inline
#endif

// The loops of the decoders are unrolled in the kernels, so that their values stay in registers.
#if defined(__CUDA_ARCH__)
#define DLIB_GGUF_UNROLL _Pragma("unroll")
#else
#define DLIB_GGUF_UNROLL
#endif

namespace dlib
{
    namespace gguf_blocks
    {
        /*!
            The weight formats of GGUF files that tt::gguf_matvec() and tt::gguf_dequantize()
            read in place, numbered as ggml_type (data_io/gguf_reader.h) numbers them.

            Every function below decodes a row of a weight matrix by groups of 32 consecutive
            values: a legacy format holds one group per block, a k-quant eight groups per
            super-block of 256 values, and a floating format any number of them.  The same
            code runs on the processor and, compiled by nvcc, in the kernels of the device, so
            that both decode the bytes of a file the same way.  Every read is a read of bytes:
            the blocks of most formats do not start on a multiple of four bytes.
        !*/
        enum : int
        {
            f32 = 0, f16 = 1,
            q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8,
            q4_K = 12, q5_K = 13, q6_K = 14,
            iq4_nl = 20, bf16 = 30
        };

        // The values of one group.
        enum : long { group_size = 32 };

        DLIB_GGUF_HD bool known_format(int type)
        {
            switch (type)
            {
            case f32: case f16: case bf16:
            case q4_0: case q4_1: case q5_0: case q5_1: case q8_0: case iq4_nl:
            case q4_K: case q5_K: case q6_K:
                return true;
            default:
                return false;
            }
        }

        /*!
            ensures
                - Returns the bytes of a row of cols values in the format type, or 0 when the
                  format is not one of the above or cols does not fill its blocks.  The floating
                  formats need whole groups too.
        !*/
        DLIB_GGUF_HD size_t row_bytes(int type, long cols)
        {
            if (cols <= 0 || cols % 32 != 0) return 0;
            const size_t c = static_cast<size_t>(cols);
            switch (type)
            {
            case f32: return 4 * c;
            case f16: case bf16: return 2 * c;
            case q4_0: case iq4_nl: return c / 32 * 18;
            case q4_1: return c / 32 * 20;
            case q5_0: return c / 32 * 22;
            case q5_1: return c / 32 * 24;
            case q8_0: return c / 32 * 34;
            case q4_K: return cols % 256 == 0 ? c / 256 * 144 : 0;
            case q5_K: return cols % 256 == 0 ? c / 256 * 176 : 0;
            case q6_K: return cols % 256 == 0 ? c / 256 * 210 : 0;
            default: return 0;
            }
        }

        DLIB_GGUF_HD uint32_t read_u16(const uint8_t* p)
        {
            return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
        }

        DLIB_GGUF_HD uint32_t read_u32(const uint8_t* p)
        {
            return read_u16(p) | (read_u16(p + 2) << 16);
        }

        DLIB_GGUF_HD float bits_to_float(uint32_t bits)
        {
#if defined(__CUDA_ARCH__)
            return __uint_as_float(bits);
#else
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
#endif
        }

        // The IEEE half precision value whose bits are h, subnormals included.
        DLIB_GGUF_HD float half_to_float(uint32_t h)
        {
#if defined(__CUDA_ARCH__)
            return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
#else
            const uint32_t sign = (h & 0x8000u) << 16;
            const uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    bits = sign;
                }
                else
                {
                    // A subnormal: (1 + m / 1024) * 2^(-14 - e) once normalized.
                    uint32_t e = 0;
                    while ((mant & 0x400u) == 0)
                    {
                        mant <<= 1;
                        ++e;
                    }
                    bits = sign | ((127 - 14 - e) << 23) | ((mant & 0x3FFu) << 13);
                }
            }
            else if (exp == 0x1Fu)
            {
                bits = sign | 0x7F800000u | (mant << 13);
            }
            else
            {
                bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            return bits_to_float(bits);
#endif
        }

        DLIB_GGUF_HD float read_half(const uint8_t* p)
        {
            return half_to_float(read_u16(p));
        }

        // The 6-bit scale and minimum of sub-block j of a Q4_K or Q5_K super-block.
        DLIB_GGUF_HD void scale_min_k4(int j, const uint8_t* q, int& sc, int& mn)
        {
            if (j < 4)
            {
                sc = q[j] & 63;
                mn = q[j + 4] & 63;
            }
            else
            {
                sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            }
        }

        /*!
            requires
                - row points to the bytes of a row in the format TYPE
                - g < the number of values of the row / 32
            ensures
                - v[i] == the value i of group g of the row, for i < 32.
        !*/
        template <int TYPE>
        DLIB_GGUF_HD void decode_group(const uint8_t* row, long g, float* v);

        template <>
        DLIB_GGUF_HD void decode_group<f32>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 128;
            DLIB_GGUF_UNROLL
            for (int i = 0; i < 32; ++i) v[i] = bits_to_float(read_u32(p + 4 * i));
        }

        template <>
        DLIB_GGUF_HD void decode_group<f16>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 64;
            DLIB_GGUF_UNROLL
            for (int i = 0; i < 32; ++i) v[i] = read_half(p + 2 * i);
        }

        template <>
        DLIB_GGUF_HD void decode_group<bf16>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 64;
            DLIB_GGUF_UNROLL
            for (int i = 0; i < 32; ++i) v[i] = bits_to_float(read_u16(p + 2 * i) << 16);
        }

        template <>
        DLIB_GGUF_HD void decode_group<q4_0>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 18;
            const float d = read_half(p);
            const uint8_t* qs = p + 2;
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 16; ++j)
            {
                v[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
                v[j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
            }
        }

        template <>
        DLIB_GGUF_HD void decode_group<q4_1>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 20;
            const float d = read_half(p);
            const float m = read_half(p + 2);
            const uint8_t* qs = p + 4;
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 16; ++j)
            {
                v[j] = static_cast<float>(qs[j] & 0x0F) * d + m;
                v[j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
            }
        }

        template <>
        DLIB_GGUF_HD void decode_group<q5_0>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 22;
            const float d = read_half(p);
            const uint32_t qh = read_u32(p + 2);
            const uint8_t* qs = p + 6;
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 16; ++j)
            {
                const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10u);
                const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10u);
                v[j] = static_cast<float>(((qs[j] & 0x0F) | h0) - 16) * d;
                v[j + 16] = static_cast<float>(((qs[j] >> 4) | h1) - 16) * d;
            }
        }

        template <>
        DLIB_GGUF_HD void decode_group<q5_1>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 24;
            const float d = read_half(p);
            const float m = read_half(p + 2);
            const uint32_t qh = read_u32(p + 4);
            const uint8_t* qs = p + 8;
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 16; ++j)
            {
                const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10u);
                const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10u);
                v[j] = static_cast<float>((qs[j] & 0x0F) | h0) * d + m;
                v[j + 16] = static_cast<float>((qs[j] >> 4) | h1) * d + m;
            }
        }

        template <>
        DLIB_GGUF_HD void decode_group<q8_0>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 34;
            const float d = read_half(p);
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 32; ++j) v[j] = static_cast<float>(static_cast<int8_t>(p[2 + j])) * d;
        }

        /*
            The non-linear value of IQ4_NL that nibble q selects, among -127, -104, -83, -65,
            -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89 and 113.  The values are packed four
            per word, so that a kernel keeps them in registers instead of a table in memory.
        */
        DLIB_GGUF_HD int iq4_nl_value(uint32_t q)
        {
            const uint32_t word = q < 8 ? (q < 4 ? 0xBFAD9881u : 0xF6EADDCFu) : (q < 12 ? 0x26190D01u : 0x71594535u);
            return static_cast<int>(static_cast<int8_t>(static_cast<uint8_t>(word >> (8 * (q & 3)))));
        }

        template <>
        DLIB_GGUF_HD void decode_group<iq4_nl>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + g * 18;
            const float d = read_half(p);
            const uint8_t* qs = p + 2;
            DLIB_GGUF_UNROLL
            for (int j = 0; j < 16; ++j)
            {
                v[j] = d * static_cast<float>(iq4_nl_value(qs[j] & 0x0Fu));
                v[j + 16] = d * static_cast<float>(iq4_nl_value(static_cast<uint32_t>(qs[j]) >> 4));
            }
        }

        /*
            A Q4_K super-block: d and dmin in half precision, the twelve bytes of the eight
            6-bit scales and minimums, then 128 bytes of nibbles.  Groups 2c and 2c + 1 share
            the 32 bytes c, the low nibbles for the first and the high ones for the second.
        */
        template <>
        DLIB_GGUF_HD void decode_group<q4_K>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + (g >> 3) * 144;
            const int j = static_cast<int>(g & 7);
            const float d = read_half(p);
            const float dmin = read_half(p + 2);
            int sc, mn;
            scale_min_k4(j, p + 4, sc, mn);
            const float scale = d * static_cast<float>(sc);
            const float minimum = dmin * static_cast<float>(mn);
            const uint8_t* qs = p + 16 + (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            DLIB_GGUF_UNROLL
            for (int l = 0; l < 32; ++l) v[l] = scale * static_cast<float>((qs[l] >> shift) & 0x0F) - minimum;
        }

        // A Q5_K super-block: as Q4_K, with the fifth bit of the 256 values in 32 bytes before the nibbles.
        template <>
        DLIB_GGUF_HD void decode_group<q5_K>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + (g >> 3) * 176;
            const int j = static_cast<int>(g & 7);
            const float d = read_half(p);
            const float dmin = read_half(p + 2);
            int sc, mn;
            scale_min_k4(j, p + 4, sc, mn);
            const float scale = d * static_cast<float>(sc);
            const float minimum = dmin * static_cast<float>(mn);
            const uint8_t* qh = p + 16;
            const uint8_t* qs = p + 48 + (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            const uint32_t bit = 1u << j;
            DLIB_GGUF_UNROLL
            for (int l = 0; l < 32; ++l)
            {
                const int q = ((qs[l] >> shift) & 0x0F) + ((qh[l] & bit) ? 16 : 0);
                v[l] = scale * static_cast<float>(q) - minimum;
            }
        }

        /*
            A Q6_K super-block: 128 bytes of low nibbles, 64 bytes of high bit pairs, sixteen
            8-bit scales of 16 values each, then d in half precision.  Each half of 128 values
            takes 64 bytes of nibbles, 32 bytes of bit pairs and 8 scales; its four groups of 32
            values take the low nibbles of the first and second 32 bytes, then their high
            nibbles, with the bit pairs 0, 2, 4 and 6 of the same 32 bytes.
        */
        template <>
        DLIB_GGUF_HD void decode_group<q6_K>(const uint8_t* row, long g, float* v)
        {
            const uint8_t* p = row + (g >> 3) * 210;
            const int j = static_cast<int>(g & 7);
            const int half = j >> 2;
            const int k = j & 3;
            const uint8_t* ql = p + half * 64 + (k & 1) * 32;
            const uint8_t* qh = p + 128 + half * 32;
            const uint8_t* sc = p + 192 + half * 8 + 2 * k;
            const float d = read_half(p + 208);
            const int low_shift = (k >> 1) * 4;
            const int high_shift = 2 * k;
            DLIB_GGUF_UNROLL
            for (int l = 0; l < 32; ++l)
            {
                const int q = (((ql[l] >> low_shift) & 0x0F) | (((qh[l] >> high_shift) & 3) << 4)) - 32;
                v[l] = d * static_cast<float>(static_cast<int8_t>(sc[l >> 4])) * static_cast<float>(q);
            }
        }

        /*!
            requires
                - known_format(type) == true
            ensures
                - Decodes group g of the row in v, as decode_group<type>() does: the dispatch of
                  the processor code, whose kernels instantiate the templates themselves.
        !*/
        inline void decode_group(int type, const uint8_t* row, long g, float* v)
        {
            switch (type)
            {
            case f32: decode_group<f32>(row, g, v); break;
            case f16: decode_group<f16>(row, g, v); break;
            case bf16: decode_group<bf16>(row, g, v); break;
            case q4_0: decode_group<q4_0>(row, g, v); break;
            case q4_1: decode_group<q4_1>(row, g, v); break;
            case q5_0: decode_group<q5_0>(row, g, v); break;
            case q5_1: decode_group<q5_1>(row, g, v); break;
            case q8_0: decode_group<q8_0>(row, g, v); break;
            case iq4_nl: decode_group<iq4_nl>(row, g, v); break;
            case q4_K: decode_group<q4_K>(row, g, v); break;
            case q5_K: decode_group<q5_K>(row, g, v); break;
            case q6_K: decode_group<q6_K>(row, g, v); break;
            default: break;
            }
        }
    }
}

#endif // DLIB_GGUF_BLOCKS_H_
