// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KV_PAGES_H_
#define DLIB_KV_PAGES_H_

#include "gguf_blocks.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct kv_page_layout
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The layout of the keys and values of a decoder in pages, which
                tt::paged_kv_store(), tt::paged_kv_gather() and tt::paged_attention() read and
                write.  A page holds the keys and values of block_positions consecutive positions
                of a sequence, for every layer and every key/value head: the keys first, then the
                values, each as layers x heads slabs, the slab of layer l and head h holding the
                rows of the positions of the page in order.

                A row of head_dim values is stored in half precision, or, when q8 is true, as
                head_dim signed 8-bit integers whose scales follow the rows of the slab: one scale
                in half precision per group of 32 values, a value being the scale of its group
                times its integer.  The keys of the 8-bit format are stored rotated
                (kv_pages::rotate_row()), and the queries are rotated alike before their products
                with them.
        !*/

        long layers = 0;
        long heads = 0;
        long head_dim = 0;
        long block_positions = 0;
        bool q8 = false;

        DLIB_GGUF_HD bool valid() const
        {
            return layers > 0 && heads > 0 && head_dim > 0 && block_positions > 0 && (!q8 || head_dim % 32 == 0);
        }

        // The bytes of the values of a row, of its scales (8-bit format), and of both.
        DLIB_GGUF_HD size_t data_bytes() const { return static_cast<size_t>(head_dim) * (q8 ? 1 : 2); }
        DLIB_GGUF_HD size_t scale_bytes() const { return q8 ? static_cast<size_t>(head_dim / 32) * 2 : 0; }
        DLIB_GGUF_HD size_t row_bytes() const { return data_bytes() + scale_bytes(); }

        DLIB_GGUF_HD size_t slab_bytes() const { return static_cast<size_t>(block_positions) * row_bytes(); }
        DLIB_GGUF_HD size_t page_bytes() const
        {
            return 2 * static_cast<size_t>(layers) * static_cast<size_t>(heads) * slab_bytes();
        }

        // The offset in a page of the slab of the keys (kind 0) or of the values (kind 1) of a layer and a head.
        DLIB_GGUF_HD size_t slab_offset(long kind, long layer, long head) const
        {
            return ((static_cast<size_t>(kind) * static_cast<size_t>(layers) + static_cast<size_t>(layer))
                * static_cast<size_t>(heads) + static_cast<size_t>(head)) * slab_bytes();
        }
    };

// ----------------------------------------------------------------------------------------

    namespace kv_pages
    {
        /*!
            The arithmetic of the keys and values in pages that the processor and the kernels of
            the device share, so that both store a row with the same bytes.
        !*/

        // Whether value i of a row changes its sign before the transform of rotate_row().
        DLIB_GGUF_HD bool flipped(long i)
        {
            return ((static_cast<uint32_t>(i) * 2654435761u) >> 31) != 0;
        }

        // The length of the blocks of the transform of rotate_row(): the largest power of two dividing n.
        DLIB_GGUF_HD long rotation_block(long n)
        {
            return n & -n;
        }

        // The bits of the half precision value nearest to f, ties to even, subnormals included.
        DLIB_GGUF_HD uint16_t float_to_half(float f)
        {
#if defined(__CUDA_ARCH__)
            return __half_as_ushort(__float2half_rn(f));
#else
            uint32_t x;
            std::memcpy(&x, &f, sizeof(x));
            const uint32_t sign = (x >> 16) & 0x8000u;
            uint32_t mant = x & 0x007FFFFFu;
            const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF);
            if (exp == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
            const int32_t e = exp - 127 + 15;
            if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
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
#endif
        }

        /*!
            ensures
                - Returns the scale in half precision of a group of 32 values of 8-bit format whose
                  largest magnitude is amax: amax/127 rounded to the nearest half, raised to the
                  next half when that rounding, as for a subnormal half, would bring the largest
                  value beyond 127.5 steps.
        !*/
        DLIB_GGUF_HD uint16_t q8_scale(float amax)
        {
            uint16_t stored = float_to_half(amax / 127.0f);
            if (gguf_blocks::half_to_float(stored) * 127.5f < amax) ++stored;
            return stored;
        }

        // The integer of x in a group whose scale d gives inv == 1/d, or 0 when d is 0: x*inv rounded, ties to even.
        DLIB_GGUF_HD int8_t q8_level(float x, float inv)
        {
#if defined(__CUDA_ARCH__)
            const float r = rintf(x * inv);
#else
            const float r = std::nearbyint(x * inv);
#endif
            return static_cast<int8_t>(r < -127.0f ? -127.0f : (r > 127.0f ? 127.0f : r));
        }

        /*!
            requires
                - n > 0
            ensures
                - Rotates the row x of n values: the signs of flipped(), then the normalized
                  Walsh-Hadamard transform on each block of rotation_block(n) values.  The map is
                  orthogonal, so that a key and a query rotated alike keep their dot product, and
                  it spreads over the row the few channels of the keys of some models that hold
                  values ten times larger than the others.
        !*/
        inline void rotate_row(float* x, long n)
        {
            for (long i = 0; i < n; ++i)
                if (flipped(i)) x[i] = -x[i];
            const long block = rotation_block(n);
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
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KV_PAGES_H_
