// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_gguf.h"
#include "gguf_blocks.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dlib
{
    namespace cuda
    {
        namespace
        {
            // The rows a block of gguf_matvec_kernel computes, one warp each.
            const int warps_per_block = 4;

            // The rows of src a launch of gguf_matvec_kernel reads at most: the weights of a row
            // are decoded once for all of them.
            const int max_matvec_rows = 8;

            __device__ __forceinline__ float warp_sum(float s)
            {
                for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_down_sync(0xffffffffu, s, offset);
                return s;
            }

            /*
                y[t * rows + r] = sum over i of W[r][i] * x[t * cols + i], for t < n <= N and r < rows.

                A warp computes a row: lane l decodes the groups l, l + 32, l + 64... of the row,
                takes their products with the n rows of x, and the warp adds the sums of its lanes.
                The lanes of a warp decode neighbouring groups, so that the warp reads a run of
                contiguous blocks; every value of x a lane reads is read by the lanes of the other
                rows of the launch too, from the caches.  When x starts on 16 bytes, which its rows
                then do too since cols is a multiple of 32, a lane reads its values four at a time.
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
                const int row = blockIdx.x * warps_per_block + threadIdx.y;
                // A warp leaves as a whole: its row is past the matrix for all its lanes.
                if (row >= rows) return;
                const int lane = threadIdx.x;
                const uint8_t* wr = w + static_cast<size_t>(row) * row_bytes;
                const int groups = cols / 32;
                float acc[N];
#pragma unroll
                for (int t = 0; t < N; ++t) acc[t] = 0;
                for (int g = lane; g < groups; g += 32)
                {
                    float v[32];
                    gguf_blocks::decode_group<TYPE>(wr, g, v);
                    const float* xg = x + static_cast<size_t>(g) * 32;
#pragma unroll
                    for (int t = 0; t < N; ++t)
                    {
                        if (t < n)
                        {
                            const float* xt = xg + static_cast<size_t>(t) * cols;
                            float s = 0;
                            if (aligned)
                            {
                                const float4* x4 = reinterpret_cast<const float4*>(xt);
#pragma unroll
                                for (int i = 0; i < 8; ++i)
                                {
                                    const float4 a = x4[i];
                                    s += v[4 * i] * a.x + v[4 * i + 1] * a.y + v[4 * i + 2] * a.z + v[4 * i + 3] * a.w;
                                }
                            }
                            else
                            {
#pragma unroll
                                for (int i = 0; i < 32; ++i) s += v[i] * xt[i];
                            }
                            acc[t] += s;
                        }
                    }
                }
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
            launch_kernel(silu_multiply_kernel, max_jobs(gate.size()), gate.device(), up.device(), gate.size());
        }

    // ----------------------------------------------------------------------------------------

    }
}
