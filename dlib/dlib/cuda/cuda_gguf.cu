// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_data_ptr.h"
#include "cuda_gguf.h"
#include "gguf_blocks.h"
#include "kv_pages.h"

// CUDART_INF_F: the INFINITY of the headers of Visual Studio is a constant beyond the range of a float,
// which nvcc warns of in every kernel that uses it.
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>

// The partial sums of the split attention rely on the order of the legacy default stream (partial_store).
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "cuda_gguf.cu requires the legacy default stream: build it without --default-stream per-thread"
#endif

namespace dlib
{
    namespace cuda
    {
        namespace
        {
            // The rows a block of gguf_matvec_kernel computes, one warp each.
            const int warps_per_block = 8;

            // The rows of src a launch of gguf_matvec_kernel reads at most: the weights of a row
            // are decoded once for all of them.
            const int max_matvec_rows = 8;

            // The groups of 32 columns of src a block copies to its shared memory at a time, one
            // for each lane of a warp.
            const int matvec_chunk = 32;

            /*
                The floats of a group of 32 columns in the shared memory of gguf_matvec_kernel: 4
                floats after its values, so that the eight lanes of a quarter of a warp, which read
                16 bytes each at the same place of eight neighbouring groups, use the 32 banks once.
            */
            const int matvec_stride = 36;

            __device__ __forceinline__ float warp_sum(float s)
            {
                for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_down_sync(0xffffffffu, s, offset);
                return s;
            }

            /*
                y[t * rows + r] = sum over i of W[r][i] * x[t * cols + i], for t < n <= N and r < rows.

                A warp computes a row, and the warps of a block neighbouring rows.  The block copies
                the columns of the n rows of x to its shared memory by chunks of 32 groups of 32
                columns, with loads of 16 bytes when x starts on 16 bytes, which its rows then do
                too since cols is a multiple of 32.  Lane l of a warp decodes the group l of each
                chunk of its row once and takes its products with the n rows of x from the shared
                memory; the lanes of a warp decode neighbouring groups, so that the warp reads a run
                of contiguous blocks, and the warp adds the sums of its lanes.  A lane adds the
                products of a group in the same order for every token and every n, so that a token
                gets the same result in any launch.
            */
            template <int TYPE, int N>
            __global__ void __launch_bounds__(32 * warps_per_block) gguf_matvec_kernel(
                const uint8_t* __restrict__ w,
                size_t row_bytes,
                int rows,
                int cols,
                const float* __restrict__ x,
                int n,
                bool aligned,
                float* __restrict__ y
            )
            {
                __shared__ __align__(16) float xs[N * matvec_chunk * matvec_stride];
                const int lane = threadIdx.x;
                const int thread = threadIdx.y * 32 + lane;
                const int row = blockIdx.x * warps_per_block + threadIdx.y;
                // A warp past the matrix copies its share of x and computes nothing.
                const bool computes = row < rows;
                const uint8_t* wr = w + static_cast<size_t>(computes ? row : 0) * row_bytes;
                const int groups = cols / 32;
                float acc[N];
#pragma unroll
                for (int t = 0; t < N; ++t) acc[t] = 0;
                for (int first = 0; first < groups; first += matvec_chunk)
                {
                    const int count = groups - first < matvec_chunk ? groups - first : matvec_chunk;
                    const float* xc = x + static_cast<size_t>(first) * 32;
                    // The warps have read the previous chunk.
                    __syncthreads();
                    if (aligned)
                    {
                        const int quads = count * 8;
                        for (int i = thread; i < n * quads; i += 32 * warps_per_block)
                        {
                            const int t = i / quads;
                            const int q = i - t * quads;
                            const float4 a = reinterpret_cast<const float4*>(xc + static_cast<size_t>(t) * cols)[q];
                            *reinterpret_cast<float4*>(xs + (t * matvec_chunk + (q >> 3)) * matvec_stride
                                + (q & 7) * 4) = a;
                        }
                    }
                    else
                    {
                        const int values = count * 32;
                        for (int i = thread; i < n * values; i += 32 * warps_per_block)
                        {
                            const int t = i / values;
                            const int j = i - t * values;
                            xs[(t * matvec_chunk + (j >> 5)) * matvec_stride + (j & 31)]
                                = xc[static_cast<size_t>(t) * cols + j];
                        }
                    }
                    __syncthreads();
                    if (computes && lane < count)
                    {
                        float v[32];
                        gguf_blocks::decode_group<TYPE>(wr, first + lane, v);
#pragma unroll
                        for (int t = 0; t < N; ++t)
                        {
                            if (t < n)
                            {
                                const float4* x4 = reinterpret_cast<const float4*>(xs + (t * matvec_chunk + lane)
                                    * matvec_stride);
                                float s = 0;
#pragma unroll
                                for (int i = 0; i < 8; ++i)
                                {
                                    const float4 a = x4[i];
                                    s += v[4 * i] * a.x + v[4 * i + 1] * a.y + v[4 * i + 2] * a.z + v[4 * i + 3] * a.w;
                                }
                                acc[t] += s;
                            }
                        }
                    }
                }
                // A warp past the matrix leaves once the last barrier is behind it.
                if (!computes) return;
#pragma unroll
                for (int t = 0; t < N; ++t)
                {
                    const float s = warp_sum(acc[t]);
                    if (lane == 0 && t < n) y[static_cast<size_t>(t) * rows + row] = s;
                }
            }

            /*
                dst[r * cols + i] = W[first + r][i], for r < count and i < cols: a thread decodes
                one group of 32 values at a time, and writes them four at a time when dst starts
                on 16 bytes.
            */
            template <int TYPE>
            __global__ void gguf_dequantize_kernel(
                const uint8_t* __restrict__ w,
                size_t row_bytes,
                long first,
                long count,
                long cols,
                bool aligned,
                float* __restrict__ dst
            )
            {
                const long groups = cols / 32;
                const size_t total = static_cast<size_t>(count) * static_cast<size_t>(groups);
                const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
                for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += step)
                {
                    const long r = static_cast<long>(i / static_cast<size_t>(groups));
                    const long g = static_cast<long>(i % static_cast<size_t>(groups));
                    float v[32];
                    gguf_blocks::decode_group<TYPE>(w + static_cast<size_t>(first + r) * row_bytes, g, v);
                    float* out = dst + static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(g) * 32;
                    if (aligned)
                    {
                        float4* out4 = reinterpret_cast<float4*>(out);
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            out4[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
                    }
                    else
                    {
#pragma unroll
                        for (int j = 0; j < 32; ++j) out[j] = v[j];
                    }
                }
            }

            /*
                Calls f with std::integral_constant<int, type>, so that f instantiates its kernels
                for the format of the weights.
            */
            template <typename F>
            void with_format(int type, F&& f)
            {
                switch (type)
                {
                case gguf_blocks::f32: f(std::integral_constant<int, gguf_blocks::f32>()); break;
                case gguf_blocks::f16: f(std::integral_constant<int, gguf_blocks::f16>()); break;
                case gguf_blocks::bf16: f(std::integral_constant<int, gguf_blocks::bf16>()); break;
                case gguf_blocks::q4_0: f(std::integral_constant<int, gguf_blocks::q4_0>()); break;
                case gguf_blocks::q4_1: f(std::integral_constant<int, gguf_blocks::q4_1>()); break;
                case gguf_blocks::q5_0: f(std::integral_constant<int, gguf_blocks::q5_0>()); break;
                case gguf_blocks::q5_1: f(std::integral_constant<int, gguf_blocks::q5_1>()); break;
                case gguf_blocks::q8_0: f(std::integral_constant<int, gguf_blocks::q8_0>()); break;
                case gguf_blocks::iq4_nl: f(std::integral_constant<int, gguf_blocks::iq4_nl>()); break;
                case gguf_blocks::q4_K: f(std::integral_constant<int, gguf_blocks::q4_K>()); break;
                case gguf_blocks::q5_K: f(std::integral_constant<int, gguf_blocks::q5_K>()); break;
                case gguf_blocks::q6_K: f(std::integral_constant<int, gguf_blocks::q6_K>()); break;
                default: DLIB_CASSERT(false, "unknown GGUF weight format " << type);
                }
            }

            template <int N>
            void launch_matvec(
                int type,
                const uint8_t* w,
                size_t row_bytes,
                int rows,
                int cols,
                const float* x,
                int n,
                float* y
            )
            {
                // The rows of x start on 16 bytes when x does: cols is a multiple of 32.
                const bool aligned = reinterpret_cast<uintptr_t>(x) % 16 == 0;
                const dim3 threads(32, warps_per_block);
                const dim3 blocks((rows + warps_per_block - 1) / warps_per_block);
                with_format(type, [&](auto format) {
                    gguf_matvec_kernel<decltype(format)::value, N><<<blocks, threads>>>(w, row_bytes, rows, cols, x,
                        n, aligned, y);
                });
            }

            void launch_dequantize(
                int type,
                const uint8_t* w,
                size_t row_bytes,
                long first,
                long count,
                long cols,
                float* dst
            )
            {
                const size_t groups = static_cast<size_t>(count) * static_cast<size_t>(cols / 32);
                const bool aligned = reinterpret_cast<uintptr_t>(dst) % 16 == 0;
                const int threads = 256;
                const int blocks = static_cast<int>(std::min<size_t>((groups + threads - 1) / threads, 65535));
                with_format(type, [&](auto format) {
                    gguf_dequantize_kernel<decltype(format)::value><<<blocks, threads>>>(w, row_bytes, first, count,
                        cols, aligned, dst);
                });
            }

            /*
                dst[r][i] = src[r][i] / sqrt(mean over j of src[r][j]^2 + eps) * gamma[i]: a block
                per row, which sums the squares of its row before it scales it.  dst may be src.
            */
            __global__ void rms_norm_rows_kernel(
                const float* src,
                const float* __restrict__ gamma,
                float* dst,
                int width,
                float eps
            )
            {
                __shared__ float partial[32];
                const float* x = src + static_cast<size_t>(blockIdx.x) * width;
                float* y = dst + static_cast<size_t>(blockIdx.x) * width;
                float s = 0;
                for (int i = threadIdx.x; i < width; i += blockDim.x) s += x[i] * x[i];
                s = warp_sum(s);
                const int lane = threadIdx.x % 32;
                const int warp = threadIdx.x / 32;
                if (lane == 0) partial[warp] = s;
                __syncthreads();
                if (warp == 0)
                {
                    s = lane < static_cast<int>(blockDim.x / 32) ? partial[lane] : 0.0f;
                    s = warp_sum(s);
                    if (lane == 0) partial[0] = s;
                }
                __syncthreads();
                const float scale = 1.0f / sqrtf(partial[0] / static_cast<float>(width) + eps);
                for (int i = threadIdx.x; i < width; i += blockDim.x) y[i] = x[i] * scale * gamma[i];
            }

            // g[i] = g[i] / (1 + exp(-g[i])) * u[i], the gate of a SwiGLU block.
            __global__ void silu_multiply_kernel(float* g, const float* __restrict__ u, size_t n)
            {
                for (auto i : grid_stride_range(0, n))
                {
                    const float a = g[i];
                    g[i] = a / (1.0f + expf(-a)) * u[i];
                }
            }

            /*
                Turns the pairs of the first rope_dims values of each row of hd values by the
                angles of its token (tt::rope_rows()): a thread per pair, pair k being pair k %
                (rope_dims/2) of row k / (rope_dims/2), which belongs to token row / heads.
            */
            __global__ void rope_rows_kernel(
                float* rows,
                const float* __restrict__ angles,
                int heads,
                int hd,
                int rope_dims,
                bool neox,
                size_t pairs
            )
            {
                const size_t half = static_cast<size_t>(rope_dims / 2);
                for (auto k : grid_stride_range(0, pairs))
                {
                    const size_t i = k % half;
                    const size_t row = k / half;
                    const float* a = angles + row / static_cast<size_t>(heads) * static_cast<size_t>(rope_dims);
                    const float c = a[i], s = a[half + i];
                    float* x = rows + row * static_cast<size_t>(hd);
                    const size_t i0 = neox ? i : 2 * i;
                    const size_t i1 = neox ? i + half : 2 * i + 1;
                    const float x0 = x[i0], x1 = x[i1];
                    x[i0] = x0 * c - x1 * s;
                    x[i1] = x0 * s + x1 * c;
                }
            }

            // The sum and the maximum of s over the lanes of a warp, which every lane receives.
            __device__ __forceinline__ float warp_sum_all(float s)
            {
                for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor_sync(0xffffffffu, s, offset);
                return s;
            }

            __device__ __forceinline__ float warp_max_all(float s)
            {
                for (int offset = 16; offset > 0; offset >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, offset));
                return s;
            }

            /*
                Completes the rotation of kv_pages::rotate_row() on count rows of n values in
                shared memory, whose signs the caller changed as it filled them: the threads of
                the block share the butterflies of each stage of the transform, in the order of
                the processor, so that both give the same values.  Every thread of the block
                calls it.
            */
            __device__ void rotate_rows_shared(float* x, int count, int n)
            {
                const int block = n & -n;
                const int half = n / 2;
                const int pairs = count * half;
                for (int len = 1; len < block; len <<= 1)
                {
                    __syncthreads();
                    for (int b = threadIdx.x; b < pairs; b += blockDim.x)
                    {
                        const int within = b % half;
                        const int lb = within % (block / 2);
                        const int i = within / (block / 2) * block + lb / len * 2 * len + lb % len;
                        float* y = x + b / half * n;
                        const float a = y[i], c = y[i + len];
                        y[i] = a + c;
                        y[i + len] = a - c;
                    }
                }
                __syncthreads();
                const float norm = 1.0f / sqrtf(static_cast<float>(block));
                for (int i = threadIdx.x; i < count * n; i += blockDim.x) x[i] *= norm;
                __syncthreads();
            }

            /*
                Writes the row of the keys (blockIdx.z == 0) or of the values (1) of head blockIdx.y
                of token blockIdx.x at the slot of the token (tt::paged_kv_store()): in half
                precision, or in the 8-bit format, a warp quantizing each group of 32 values, the
                keys rotated before.  row holds head_dim floats of shared memory.
            */
            __global__ void paged_kv_store_kernel(
                uint8_t* pages,
                kv_page_layout layout,
                int layer,
                const float* __restrict__ keys,
                const float* __restrict__ values,
                const int32_t* __restrict__ slots
            )
            {
                extern __shared__ float row[];
                const int kind = blockIdx.z;
                const int hd = static_cast<int>(layout.head_dim);
                const float* src = (kind == 0 ? keys : values)
                    + (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(layout.heads) + blockIdx.y) * hd;
                const int32_t slot = slots[blockIdx.x];
                const size_t r = static_cast<size_t>(slot % layout.block_positions);
                uint8_t* slab = pages + static_cast<size_t>(slot / layout.block_positions) * layout.page_bytes()
                    + layout.slab_offset(kind, layer, blockIdx.y);
                uint8_t* data = slab + r * layout.data_bytes();
                if (!layout.q8)
                {
                    uint16_t* dst = reinterpret_cast<uint16_t*>(data);
                    for (int i = threadIdx.x; i < hd; i += blockDim.x) dst[i] = kv_pages::float_to_half(src[i]);
                    return;
                }
                for (int i = threadIdx.x; i < hd; i += blockDim.x)
                    row[i] = kind == 0 && kv_pages::flipped(i) ? -src[i] : src[i];
                if (kind == 0)
                    rotate_rows_shared(row, 1, hd);
                else
                    __syncthreads();
                uint16_t* scales = reinterpret_cast<uint16_t*>(slab
                    + static_cast<size_t>(layout.block_positions) * layout.data_bytes() + r * layout.scale_bytes());
                const int lane = threadIdx.x % 32;
                for (int g = threadIdx.x / 32; g < hd / 32; g += blockDim.x / 32)
                {
                    const float x = row[g * 32 + lane];
                    const uint16_t stored = kv_pages::q8_scale(warp_max_all(fabsf(x)));
                    const float d = gguf_blocks::half_to_float(stored);
                    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
                    reinterpret_cast<int8_t*>(data)[g * 32 + lane] = kv_pages::q8_level(x, inv);
                    if (lane == 0) scales[g] = stored;
                }
            }

            /*
                Copies the bytes of the rows of the slots, every kind, layer and head of each, in
                the order of tt::paged_kv_gather(): a thread per byte.
            */
            __global__ void paged_kv_gather_kernel(
                uint8_t* __restrict__ dest,
                const uint8_t* __restrict__ pages,
                kv_page_layout layout,
                const int32_t* __restrict__ slots,
                size_t total
            )
            {
                const size_t row_bytes = layout.row_bytes();
                const size_t data = layout.data_bytes();
                const size_t slabs = 2 * static_cast<size_t>(layout.layers) * static_cast<size_t>(layout.heads);
                const size_t B = static_cast<size_t>(layout.block_positions);
                for (auto i : grid_stride_range(0, total))
                {
                    const size_t byte = i % row_bytes;
                    const size_t row = i / row_bytes;
                    const int32_t slot = slots[row / slabs];
                    const size_t r = static_cast<size_t>(slot) % B;
                    // The slabs of a page follow the order of the rows of a slot: kind, layer, head.
                    const uint8_t* slab = pages + static_cast<size_t>(slot) / B * layout.page_bytes()
                        + row % slabs * layout.slab_bytes();
                    dest[i] = byte < data ? slab[r * data + byte]
                        : slab[B * data + r * layout.scale_bytes() + byte - data];
                }
            }

            // The warps of a block of paged_attention_kernel, and the query heads it attends at most.
            const int attention_warps = 4;
            const int attention_heads = 4;

            /*
                The values of row r of a slab of keys or values that lane holds: the values lane,
                lane + 32, lane + 64... of the row, as floats.  scales_at is the offset of the
                scales of the 8-bit format in the slab.
            */
            template <int DPL, bool Q8>
            __device__ __forceinline__ void load_row(const uint8_t* slab, int r, size_t scales_at, int lane, float* x)
            {
                if (Q8)
                {
                    const int8_t* data = reinterpret_cast<const int8_t*>(slab) + static_cast<size_t>(r) * (32 * DPL);
                    const uint16_t* scales = reinterpret_cast<const uint16_t*>(slab + scales_at)
                        + static_cast<size_t>(r) * DPL;
#pragma unroll
                    for (int i = 0; i < DPL; ++i)
                        x[i] = gguf_blocks::half_to_float(scales[i]) * static_cast<float>(data[i * 32 + lane]);
                }
                else
                {
                    const uint16_t* data = reinterpret_cast<const uint16_t*>(slab)
                        + static_cast<size_t>(r) * (32 * DPL);
#pragma unroll
                    for (int i = 0; i < DPL; ++i) x[i] = gguf_blocks::half_to_float(data[i * 32 + lane]);
                }
            }

            /*
                The attention of up to attention_heads query heads that share a key/value head,
                for one token, over a range of its positions (tt::paged_attention()): token
                blockIdx.z, head group blockIdx.y, part blockIdx.x of the positions [0, p] of the
                token.  The queries, scaled and rotated as the keys of the 8-bit format are, go
                through shared memory into registers, lane l of a warp holding the values l, l +
                32, l + 64... of each row, the head dimension being 32*DPL.  The warps take the
                positions in turn, each keeping the running maximum, sum and weighted values of
                its positions (the online softmax), which the block merges at the end: into the
                output, or, for a split along the positions, into partial, unnormalized, with
                their maximum and sum, for paged_attention_merge_kernel.
            */
            template <int DPL, bool Q8>
            __global__ void __launch_bounds__(32 * attention_warps) paged_attention_kernel(
                const uint8_t* __restrict__ pages,
                kv_page_layout layout,
                int layer,
                const float* __restrict__ queries,
                int q_heads,
                float scale,
                const int32_t* __restrict__ tokens,
                const int32_t* __restrict__ tables,
                int parts,
                float* __restrict__ out,
                float* __restrict__ partial
            )
            {
                const int HD = 32 * DPL;
                const int G = attention_heads;
                __shared__ float q_shared[G * HD];
                __shared__ float m_shared[attention_warps][G];
                __shared__ float l_shared[attention_warps][G];
                __shared__ float o_shared[attention_warps][G][HD];

                const int rep = q_heads / static_cast<int>(layout.heads);
                const int groups = (rep + G - 1) / G;
                const int kvh = blockIdx.y / groups;
                const int h0 = kvh * rep + blockIdx.y % groups * G;
                const int heads = kvh * rep + rep - h0 < G ? kvh * rep + rep - h0 : G;
                const int t = blockIdx.z;
                const int position = tokens[2 * t];
                const int first = tokens[2 * t + 1];
                // ceil((position + 1) / parts) positions per part.
                const int per = (position + parts) / parts;
                const int begin = blockIdx.x * per;
                const int end = position + 1 < begin + per ? position + 1 : begin + per;

                for (int i = threadIdx.x; i < heads * HD; i += blockDim.x)
                {
                    const int e = i % HD;
                    float x = queries[(static_cast<size_t>(t) * q_heads + h0 + i / HD) * HD + e] * scale;
                    if (Q8 && kv_pages::flipped(e)) x = -x;
                    q_shared[i] = x;
                }
                if (Q8)
                    rotate_rows_shared(q_shared, heads, HD);
                else
                    __syncthreads();

                const int lane = threadIdx.x % 32;
                const int warp = threadIdx.x / 32;
                float q[G][DPL], o[G][DPL], m[G], l[G];
#pragma unroll
                for (int g = 0; g < G; ++g)
                {
                    m[g] = -CUDART_INF_F;
                    l[g] = 0;
#pragma unroll
                    for (int i = 0; i < DPL; ++i)
                    {
                        q[g][i] = g < heads ? q_shared[g * HD + i * 32 + lane] : 0.0f;
                        o[g][i] = 0;
                    }
                }

                const int B = static_cast<int>(layout.block_positions);
                const size_t page_bytes = layout.page_bytes();
                const size_t key_slab = layout.slab_offset(0, layer, kvh);
                const size_t value_slab = layout.slab_offset(1, layer, kvh);
                const size_t scales_at = static_cast<size_t>(B) * layout.data_bytes();
                for (int p = begin + warp; p < end; p += attention_warps)
                {
                    const uint8_t* page = pages + static_cast<size_t>(tables[first + p / B]) * page_bytes;
                    const int r = p % B;
                    float k[DPL], v[DPL];
                    load_row<DPL, Q8>(page + key_slab, r, scales_at, lane, k);
                    load_row<DPL, Q8>(page + value_slab, r, scales_at, lane, v);
#pragma unroll
                    for (int g = 0; g < G; ++g)
                    {
                        if (g >= heads) break;
                        float d = 0;
#pragma unroll
                        for (int i = 0; i < DPL; ++i) d += q[g][i] * k[i];
                        const float s = warp_sum_all(d);
                        const float mx = fmaxf(m[g], s);
                        const float shrink = expf(m[g] - mx);
                        const float e = expf(s - mx);
                        l[g] = l[g] * shrink + e;
#pragma unroll
                        for (int i = 0; i < DPL; ++i) o[g][i] = o[g][i] * shrink + e * v[i];
                        m[g] = mx;
                    }
                }

#pragma unroll
                for (int g = 0; g < G; ++g)
                {
                    if (lane == 0)
                    {
                        m_shared[warp][g] = m[g];
                        l_shared[warp][g] = l[g];
                    }
#pragma unroll
                    for (int i = 0; i < DPL; ++i) o_shared[warp][g][i * 32 + lane] = o[g][i];
                }
                __syncthreads();
                for (int i = threadIdx.x; i < heads * HD; i += blockDim.x)
                {
                    const int g = i / HD;
                    const int e = i % HD;
                    float mx = -CUDART_INF_F;
                    for (int w = 0; w < attention_warps; ++w) mx = fmaxf(mx, m_shared[w][g]);
                    float sum = 0, acc = 0;
                    // A part without positions has no maximum: its sums stay 0.
                    if (mx > -CUDART_INF_F)
                        for (int w = 0; w < attention_warps; ++w)
                        {
                            const float f = expf(m_shared[w][g] - mx);
                            sum += l_shared[w][g] * f;
                            acc += o_shared[w][g][e] * f;
                        }
                    const size_t row = static_cast<size_t>(t) * q_heads + h0 + g;
                    if (parts == 1)
                    {
                        out[row * HD + e] = acc / sum;
                    }
                    else
                    {
                        float* dst = partial + (row * parts + blockIdx.x) * (HD + 2);
                        dst[e] = acc;
                        if (e == 0)
                        {
                            dst[HD] = mx;
                            dst[HD + 1] = sum;
                        }
                    }
                }
            }

            /*
                Merges the parts of the positions of a row of the output, block blockIdx.x being
                the row of a query head of a token: each part weighs by the exponential of its
                maximum less the largest one, the parts without positions left out.
            */
            __global__ void paged_attention_merge_kernel(
                const float* __restrict__ partial,
                int parts,
                int hd,
                float* __restrict__ out
            )
            {
                const size_t row = blockIdx.x;
                const float* first = partial + row * static_cast<size_t>(parts) * (hd + 2);
                float mx = -CUDART_INF_F;
                for (int q = 0; q < parts; ++q)
                    if (first[q * (hd + 2) + hd + 1] > 0.0f) mx = fmaxf(mx, first[q * (hd + 2) + hd]);
                for (int e = threadIdx.x; e < hd; e += blockDim.x)
                {
                    float sum = 0, acc = 0;
                    for (int q = 0; q < parts; ++q)
                    {
                        const float* range = first + q * (hd + 2);
                        if (range[hd + 1] <= 0.0f) continue;
                        const float f = expf(range[hd] - mx);
                        sum += range[hd + 1] * f;
                        acc += range[e] * f;
                    }
                    out[row * hd + e] = acc / sum;
                }
            }

            /*
                The partial sums of the split attention: one buffer per card for the process, kept
                from one call to the next and grown by half at least, so that the steps of a
                generation neither allocate nor free memory of the device, whose release waits for
                the device.  The buffers are never released, the driver taking their memory back
                with the process: the storage of a thread is released on Windows under the lock of
                the loader, where the calls of CUDA fail.  A caller holds the lock from the launch
                of the attention to the launch of the merge, both on the default stream, so that
                the kernels of another thread run before or after them.
            */
            struct partial_store
            {
                std::mutex m;
                std::vector<cuda_data_void_ptr> buffers;
            };

            partial_store& partial_buffers()
            {
                static partial_store* store = new partial_store();
                return *store;
            }

            float* partial_sums(const std::unique_lock<std::mutex>& lock, size_t bytes)
            {
                DLIB_CASSERT(lock.owns_lock() && lock.mutex() == &partial_buffers().m);
                int device = 0;
                CHECK_CUDA(cudaGetDevice(&device));
                std::vector<cuda_data_void_ptr>& buffers = partial_buffers().buffers;
                if (buffers.size() <= static_cast<size_t>(device)) buffers.resize(static_cast<size_t>(device) + 1);
                cuda_data_void_ptr& held = buffers[static_cast<size_t>(device)];
                if (held.size() < bytes) held = cuda_data_void_ptr(std::max(bytes, held.size() + held.size() / 2));
                return static_cast<float*>(held.data());
            }

            // The multiprocessors of the current card, which the split of the attention fills.
            int multiprocessors()
            {
                static thread_local int device = -1;
                static thread_local int count = 0;
                int current = 0;
                CHECK_CUDA(cudaGetDevice(&current));
                if (current != device)
                {
                    CHECK_CUDA(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, current));
                    device = current;
                }
                return std::max(1, count);
            }

            template <int DPL>
            void launch_attention(
                bool q8,
                dim3 blocks,
                const uint8_t* pages,
                const kv_page_layout& layout,
                int layer,
                const float* queries,
                int q_heads,
                float scale,
                const int32_t* tokens,
                const int32_t* tables,
                int parts,
                float* out,
                float* partial
            )
            {
                if (q8)
                    paged_attention_kernel<DPL, true><<<blocks, 32 * attention_warps>>>(pages, layout, layer, queries,
                        q_heads, scale, tokens, tables, parts, out, partial);
                else
                    paged_attention_kernel<DPL, false><<<blocks, 32 * attention_warps>>>(pages, layout, layer, queries,
                        q_heads, scale, tokens, tables, parts, out, partial);
            }
        }

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
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.device());
            const float* x = src.device();
            float* y = dest.device_write_only();
            for (long t0 = 0; t0 < n; t0 += max_matvec_rows)
            {
                const int k = static_cast<int>(std::min<long>(max_matvec_rows, n - t0));
                const float* xk = x + static_cast<size_t>(t0) * static_cast<size_t>(cols);
                float* yk = y + static_cast<size_t>(t0) * static_cast<size_t>(rows);
                const int r = static_cast<int>(rows);
                const int c = static_cast<int>(cols);
                if (k == 1) launch_matvec<1>(type, w, row_bytes, r, c, xk, k, yk);
                else if (k == 2) launch_matvec<2>(type, w, row_bytes, r, c, xk, k, yk);
                else if (k <= 4) launch_matvec<4>(type, w, row_bytes, r, c, xk, k, yk);
                else launch_matvec<8>(type, w, row_bytes, r, c, xk, k, yk);
                CHECK_CUDA(cudaGetLastError());
            }
        }

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
            launch_dequantize(type, reinterpret_cast<const uint8_t*>(weights.device()), row_bytes, first, count, cols,
                dest.device_write_only());
            CHECK_CUDA(cudaGetLastError());
        }

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
            if (rows == 0) return;
            // dest may be src: its memory is then read before it is written, by the same thread.
            float* y = is_same_object(dest, src) ? dest.device() : dest.device_write_only();
            const float* x = src.device();
            rms_norm_rows_kernel<<<static_cast<unsigned int>(rows), 256>>>(x, gamma.device(), y,
                static_cast<int>(width), eps);
            CHECK_CUDA(cudaGetLastError());
        }

        void silu_multiply (
            tensor& gate,
            const tensor& up
        )
        {
            DLIB_CASSERT(gate.size() == up.size() && !is_same_object(gate, up));
            if (gate.size() == 0) return;
            // Every layer of a step calls it: no occupancy query of launch_kernel().
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>((gate.size() + 255) / 256, 4096));
            silu_multiply_kernel<<<blocks, 256>>>(gate.device(), up.device(), gate.size());
            CHECK_CUDA(cudaGetLastError());
        }

        void rope_rows (
            tensor& rows,
            const tensor& angles,
            long heads,
            long head_dim,
            long rope_dims,
            bool neox
        )
        {
            DLIB_CASSERT(heads > 0 && head_dim > 0 && rope_dims > 0 && rope_dims <= head_dim && rope_dims % 2 == 0);
            DLIB_CASSERT(angles.size() % static_cast<size_t>(rope_dims) == 0);
            const size_t n = angles.size() / static_cast<size_t>(rope_dims);
            DLIB_CASSERT(rows.size() == n * static_cast<size_t>(heads) * static_cast<size_t>(head_dim));
            DLIB_CASSERT(!is_same_object(rows, angles));
            const size_t pairs = n * static_cast<size_t>(heads) * static_cast<size_t>(rope_dims / 2);
            if (pairs == 0) return;
            // Every layer of a step calls it twice: no occupancy query of launch_kernel().
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>((pairs + 255) / 256, 4096));
            rope_rows_kernel<<<blocks, 256>>>(rows.device(), angles.device(), static_cast<int>(heads),
                static_cast<int>(head_dim), static_cast<int>(rope_dims), neox, pairs);
            CHECK_CUDA(cudaGetLastError());
        }

        void paged_kv_store (
            tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& keys,
            const tensor& values,
            const tensor& slots
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            DLIB_CASSERT(layout.head_dim <= 4096 && layout.heads <= 65535);
            const size_t n = slots.size();
            const size_t width = static_cast<size_t>(layout.heads) * static_cast<size_t>(layout.head_dim);
            DLIB_CASSERT(keys.size() == n * width && values.size() == n * width);
            DLIB_CASSERT(!is_same_object(pages, keys) && !is_same_object(pages, values)
                && !is_same_object(pages, slots));
            DLIB_CASSERT(n <= 65535);
            if (n == 0) return;
            const dim3 blocks(static_cast<unsigned>(n), static_cast<unsigned>(layout.heads), 2);
            paged_kv_store_kernel<<<blocks, 128, sizeof(float) * static_cast<size_t>(layout.head_dim)>>>(
                reinterpret_cast<uint8_t*>(pages.device()), layout, static_cast<int>(layer), keys.device(),
                values.device(), reinterpret_cast<const int32_t*>(slots.device()));
            CHECK_CUDA(cudaGetLastError());
        }

        void paged_kv_gather (
            tensor& dest,
            const tensor& pages,
            const kv_page_layout& layout,
            const tensor& slots
        )
        {
            DLIB_CASSERT(layout.valid());
            const size_t n = slots.size();
            const size_t total = n * 2 * static_cast<size_t>(layout.layers) * static_cast<size_t>(layout.heads)
                * layout.row_bytes();
            DLIB_CASSERT(dest.size() * sizeof(float) >= total);
            DLIB_CASSERT(!is_same_object(dest, pages) && !is_same_object(dest, slots));
            if (total == 0) return;
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>((total + 255) / 256, 4096));
            paged_kv_gather_kernel<<<blocks, 256>>>(reinterpret_cast<uint8_t*>(dest.device()),
                reinterpret_cast<const uint8_t*>(pages.device()), layout,
                reinterpret_cast<const int32_t*>(slots.device()), total);
            CHECK_CUDA(cudaGetLastError());
        }

        void paged_attention (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tokens,
            const tensor& tables,
            long span
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            const long hd = layout.head_dim;
            DLIB_CASSERT(hd == 32 || hd == 64 || hd == 96 || hd == 128 || hd == 256);
            DLIB_CASSERT(q_heads > 0 && q_heads % layout.heads == 0 && tokens.size() % 2 == 0);
            const size_t n = tokens.size() / 2;
            const size_t width = static_cast<size_t>(q_heads) * static_cast<size_t>(hd);
            DLIB_CASSERT(queries.size() == n * width && out.size() == n * width);
            DLIB_CASSERT(!is_same_object(out, queries) && !is_same_object(out, pages) && !is_same_object(out, tokens)
                && !is_same_object(out, tables));
            DLIB_CASSERT(n <= 65535 && span > 0 && span <= std::numeric_limits<int>::max());
            if (n == 0) return;
            const long rep = q_heads / layout.heads;
            const long groups = (rep + attention_heads - 1) / attention_heads;
            DLIB_CASSERT(layout.heads * groups <= 65535);
            /*
                The blocks of the tokens and heads alone may leave most of the card idle, as in the
                generation of a single answer: the positions are then split into parts of 128
                positions at least, so that the blocks fill the multiprocessors four times.
            */
            const long units = static_cast<long>(n) * layout.heads * groups;
            const long target = 4L * multiprocessors();
            long parts = units >= target ? 1 : (target + units - 1) / units;
            parts = std::max(1L, std::min(parts, (span + 127) / 128));
            std::unique_lock<std::mutex> lock;
            float* partial = nullptr;
            if (parts > 1)
            {
                lock = std::unique_lock<std::mutex>(partial_buffers().m);
                partial = partial_sums(lock, sizeof(float) * n * static_cast<size_t>(q_heads)
                    * static_cast<size_t>(parts) * static_cast<size_t>(hd + 2));
            }
            const dim3 blocks(static_cast<unsigned>(parts), static_cast<unsigned>(layout.heads * groups),
                static_cast<unsigned>(n));
            const uint8_t* p = reinterpret_cast<const uint8_t*>(pages.device());
            const int32_t* tk = reinterpret_cast<const int32_t*>(tokens.device());
            const int32_t* tb = reinterpret_cast<const int32_t*>(tables.device());
            float* o = out.device_write_only();
            const int qh = static_cast<int>(q_heads);
            const int li = static_cast<int>(layer);
            const int np = static_cast<int>(parts);
            switch (hd)
            {
            case 32: launch_attention<1>(layout.q8, blocks, p, layout, li, queries.device(), qh, scale, tk, tb, np, o,
                partial); break;
            case 64: launch_attention<2>(layout.q8, blocks, p, layout, li, queries.device(), qh, scale, tk, tb, np, o,
                partial); break;
            case 96: launch_attention<3>(layout.q8, blocks, p, layout, li, queries.device(), qh, scale, tk, tb, np, o,
                partial); break;
            case 128: launch_attention<4>(layout.q8, blocks, p, layout, li, queries.device(), qh, scale, tk, tb, np, o,
                partial); break;
            default: launch_attention<8>(layout.q8, blocks, p, layout, li, queries.device(), qh, scale, tk, tb, np, o,
                partial); break;
            }
            CHECK_CUDA(cudaGetLastError());
            if (parts > 1)
            {
                paged_attention_merge_kernel<<<static_cast<unsigned>(n * static_cast<size_t>(q_heads)), 128>>>(partial,
                    np, static_cast<int>(hd), o);
                CHECK_CUDA(cudaGetLastError());
            }
        }

    // ----------------------------------------------------------------------------------------

    }
}
