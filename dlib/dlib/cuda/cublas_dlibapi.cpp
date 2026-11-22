// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_CPP_
#define DLIB_DNN_CuBLAS_CPP_

#ifdef DLIB_USE_CUDA

#include "cublas_dlibapi.h"
#include "cuda_utils.h"
#include "cuda_gguf.h"

#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

static const char* cublas_get_error_string(cublasStatus_t s)
{
    switch(s)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED: 
            return "CUDA Runtime API initialization failed.";
        case CUBLAS_STATUS_ALLOC_FAILED: 
            return "CUDA Resources could not be allocated.";
        default:
            return "A call to cuBLAS failed";
    }
}

// Check the return value of a call to the cuBLAS runtime for an error condition.
#define CHECK_CUBLAS(call)                                                      \
do{                                                                              \
    const cublasStatus_t error = call;                                         \
    if (error != CUBLAS_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cublas_get_error_string(error);\
        throw dlib::cublas_error(sout.str());                            \
    }                                                                          \
}while(false)

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class cublas_context
        {
        public:
            // not copyable
            cublas_context(const cublas_context&) = delete;
            cublas_context& operator=(const cublas_context&) = delete;

            cublas_context()
            {
                handles.resize(16);
            }
            ~cublas_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cublasDestroy(h);
                }
            }

            cublasHandle_t get_handle (
            )  
            { 
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)handles.size())
                    handles.resize(new_device_id+16);

                // If we don't have a handle already for this device then make one
                if (!handles[new_device_id])
                    CHECK_CUBLAS(cublasCreate(&handles[new_device_id]));

                // Finally, return the handle for the current device
                return handles[new_device_id];
            }

        private:

            std::vector<cublasHandle_t> handles;
        };

        static cublasHandle_t context()
        {
            thread_local cublas_context c;
            return c.get_handle();
        }

    // -----------------------------------------------------------------------------------

        void gemm (
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs,
            operation_mode mode
        )
        {
            if (mode == operation_mode::CHANNEL_WISE)
            {
                // Recall that BLAS uses column major order so to deal with that we flip the
                // order of the lhs and rhs arguments.
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                const int dest_nr = dest.num_samples();
                const int dest_nc = dest.size() / dest_nr;
                const int lhs_nr = lhs.num_samples();
                const int lhs_nc = lhs.size() / lhs_nr;
                const int rhs_nr = rhs.num_samples();
                const int rhs_nc = rhs.size() / rhs_nr;
                if (trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nr &&
                        lhs_nr == rhs_nc)
                }
                else if (!trans_lhs && trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nr &&
                        lhs_nc == rhs_nc)
                }
                else if (trans_lhs && !trans_rhs)
                {
                    DLIB_ASSERT(dest_nr == lhs_nc &&
                        dest_nc == rhs_nc &&
                        lhs_nr == rhs_nr)
                }
                else
                {
                    DLIB_ASSERT(dest_nr == lhs_nr &&
                        dest_nc == rhs_nc &&
                        lhs_nc == rhs_nr)
                }

                const int k = trans_rhs ? rhs_nc : rhs_nr;
                CHECK_CUBLAS(cublasSgemm(context(),
                    transb,
                    transa,
                    dest_nc, dest_nr, k,
                    &alpha,
                    rhs.device(), rhs_nc,
                    lhs.device(), lhs_nc,
                    &beta,
                    dest.device(), dest_nc));
            }
            else if (mode == operation_mode::PLANE_WISE)
            {
                const auto transa = trans_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
                const auto transb = trans_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;

                long num_samples = std::min({ lhs.num_samples(), rhs.num_samples(), dest.num_samples() });
                long num_channels = std::min({ lhs.k(), rhs.k(), dest.k() });

                /* Disambiguate the two matrix layouts against the batch geometry of the
                   operands: when one of them is unambiguously plane-batched and carries
                   the same (num_samples, k), the tensor belongs to that batch and its
                   1x1 planes are planes, not the cells of a matrix. */
                auto plane_batched = [](const tensor& t) {
                    return t.num_samples() * t.k() > 1 && t.nr() * t.nc() > 1;
                };
                auto shares_batch = [&](const tensor& t) {
                    auto same = [&](const tensor& o) {
                        return plane_batched(o)
                            && o.num_samples() == t.num_samples() && o.k() == t.k();
                        };
                    return same(lhs) || same(rhs) || same(dest);
                };
                auto is_matrix = [&](const tensor& t) {
                    if (t.num_samples() * t.k() == 1) return t.nr() * t.nc() > 1;
                    return t.nr() * t.nc() == 1 && !shares_batch(t);
                };
                const bool lhs_is_matrix = is_matrix(lhs), rhs_is_matrix = is_matrix(rhs), dest_is_matrix = is_matrix(dest);

                if (lhs_is_matrix && rhs_is_matrix && dest_is_matrix) num_samples = num_channels = 1;

                size_t lhs_rows = lhs.nr();
                size_t lhs_cols = lhs.nc();
                if (lhs_is_matrix && (lhs.num_samples() > 1 || lhs.k() > 1)) {
                    lhs_rows = lhs.num_samples();
                    lhs_cols = lhs.k();
                }
                size_t rhs_rows = rhs.nr();
                size_t rhs_cols = rhs.nc();
                if (rhs_is_matrix && (rhs.num_samples() > 1 || rhs.k() > 1)) {
                    rhs_rows = rhs.num_samples();
                    rhs_cols = rhs.k();
                }
                size_t dest_rows = dest.nr();
                size_t dest_cols = dest.nc();
                if (dest_is_matrix && (dest.num_samples() > 1 || dest.k() > 1)) {
                    dest_rows = dest.num_samples();
                    dest_cols = dest.k();
                }

                const long long lhs_plane_size = (long long)(lhs_rows * lhs_cols);
                const long long rhs_plane_size = (long long)(rhs_rows * rhs_cols);
                const long long dest_plane_size = (long long)(dest_rows * dest_cols);
                const long long planes = (long long)num_samples * num_channels;
                const int k = trans_rhs ? rhs_cols : rhs_rows;

                /* The pointers are resolved once for the whole product. Each device() call
                   is an access the memory extension records, so resolving them per plane
                   would record the same three blocks thousands of times. */
                const float* lhs_data = lhs.device();
                const float* rhs_data = rhs.device();
                float* dest_data = dest.device();

                if (dest_is_matrix && planes > 1)
                {
                    /* Every plane writes the same matrix and reads what the previous one left
                       there through beta, so the planes keep their order, one call each. */
                    for (long long p = 0; p < planes; ++p)
                    {
                        const float* lhs_slice = lhs_is_matrix ? lhs_data : lhs_data + p * lhs_plane_size;
                        const float* rhs_slice = rhs_is_matrix ? rhs_data : rhs_data + p * rhs_plane_size;
                        CHECK_CUBLAS(cublasSgemm(
                            context(), transb, transa, dest_cols, dest_rows, k,
                            &alpha, rhs_slice, rhs_cols, lhs_slice, lhs_cols,
                            &beta, dest_data, dest_cols
                        ));
                    }
                }
                else
                {
                    /* The planes are independent products at a fixed stride, which a strided
                       batched call computes in one launch where a loop would issue one per
                       plane, the number of samples times the number of heads in attention. A
                       matrix shared by every plane is passed with a stride of zero. Batches
                       above 65535 planes go in several calls, since some batched kernels index
                       the batch on the third grid dimension, which stops there. */
                    const long long lhs_stride = lhs_is_matrix ? 0 : lhs_plane_size;
                    const long long rhs_stride = rhs_is_matrix ? 0 : rhs_plane_size;
                    const long long max_batch = 65535;
                    for (long long first = 0; first < planes; first += max_batch)
                    {
                        const int count = (int)std::min(max_batch, planes - first);
                        CHECK_CUBLAS(cublasSgemmStridedBatched(
                            context(), transb, transa, dest_cols, dest_rows, k,
                            &alpha,
                            rhs_data + first * rhs_stride, rhs_cols, rhs_stride,
                            lhs_data + first * lhs_stride, lhs_cols, lhs_stride,
                            &beta,
                            dest_data + first * dest_plane_size, dest_cols, dest_plane_size,
                            count
                        ));
                    }
                }
            }
        }

    // ------------------------------------------------------------------------------------

        // The major number of the compute capability of the current card.
        static int capability_major()
        {
            static thread_local int device = -1;
            static thread_local int major = 0;
            int current = 0;
            CHECK_CUDA(cudaGetDevice(&current));
            if (current != device)
            {
                CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current));
                device = current;
            }
            return major;
        }

        bool gguf_gemm_computes (
            gguf_precision precision
        )
        {
            if (precision == gguf_precision::single) return true;
            // The tensor cores came with the capability 7.0, their TF32 arithmetic with 8.0.
            return capability_major() >= (precision == gguf_precision::tf32 ? 8 : 7);
        }

        void gguf_gemm (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes,
            tensor& scratch,
            gguf_precision precision,
            tensor& saturated
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && rows > 0 && cols > 0);
            DLIB_CASSERT(static_cast<long long>(rows) <= std::numeric_limits<int>::max()
                && static_cast<long long>(cols) <= std::numeric_limits<int>::max());
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights) && !is_same_object(dest, scratch)
                && !is_same_object(src, scratch) && !is_same_object(weights, scratch));
            DLIB_CASSERT(src.size() % static_cast<size_t>(cols) == 0);
            const size_t n = src.size() / static_cast<size_t>(cols);
            DLIB_CASSERT(n <= static_cast<size_t>(std::numeric_limits<int>::max()));
            DLIB_CASSERT(dest.size() == n * static_cast<size_t>(rows));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(rows) * row_bytes);
            DLIB_CASSERT(gguf_gemm_computes(precision));
            if (n == 0) return;
            const bool half = precision == gguf_precision::half;
            DLIB_CASSERT(!half || (saturated.size() > 0 && !is_same_object(saturated, dest)
                && !is_same_object(saturated, src) && !is_same_object(saturated, scratch)));
            // The activations in half precision come first in scratch, the tiles after them on 16 bytes.
            const size_t x_bytes = half ? (n * static_cast<size_t>(cols) * sizeof(uint16_t) + 15) / 16 * 16 : 0;
            const size_t row = static_cast<size_t>(cols) * (half ? sizeof(uint16_t) : sizeof(float));
            const size_t bytes = scratch.size() * sizeof(float);
            DLIB_CASSERT(bytes >= x_bytes + row, "scratch holds " << bytes << " bytes, less than the " << x_bytes + row
                << " bytes of the activations and of one row of the weights");
            const long tile = static_cast<long>(std::min<size_t>(static_cast<size_t>(rows), (bytes - x_bytes) / row));
            uint8_t* work = reinterpret_cast<uint8_t*>(scratch.device_write_only());
            void* expanded = work + x_bytes;
            const float* x = src.device();
            float* y = dest.device_write_only();
            float* flag = half ? saturated.device() : nullptr;
            const float one = 1, zero = 0;
            const int tokens = static_cast<int>(n);
            const int width = static_cast<int>(cols);
            if (half) floats_to_halves(x, reinterpret_cast<uint16_t*>(work), n * static_cast<size_t>(cols), flag);
            /*
                Each tile gives the columns [first, first + count) of the rows of dest: in the
                column-major order of BLAS, the tile is a matrix of cols x count that the product
                transposes, src one of cols x n, and their product lands in the matrix of rows x n
                that dest is, count rows from first on.
            */
            for (long first = 0; first < rows; first += tile)
            {
                const int count = static_cast<int>(std::min(tile, rows - first));
                if (half)
                {
                    gguf_expand_half_tile(weights, type, first, count, cols, row_bytes,
                        reinterpret_cast<uint16_t*>(expanded), flag);
                    CHECK_CUBLAS(cublasGemmEx(context(), CUBLAS_OP_T, CUBLAS_OP_N, count, tokens, width, &one,
                        expanded, CUDA_R_16F, width, work, CUDA_R_16F, width, &zero, y + first, CUDA_R_32F,
                        static_cast<int>(rows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    continue;
                }
                gguf_expand_tile(weights, type, first, count, cols, row_bytes, reinterpret_cast<float*>(expanded));
                if (precision == gguf_precision::tf32)
                    CHECK_CUBLAS(cublasGemmEx(context(), CUBLAS_OP_T, CUBLAS_OP_N, count, tokens, width, &one,
                        expanded, CUDA_R_32F, width, x, CUDA_R_32F, width, &zero, y + first, CUDA_R_32F,
                        static_cast<int>(rows), CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
                else
                    CHECK_CUBLAS(cublasSgemm(context(), CUBLAS_OP_T, CUBLAS_OP_N, count, tokens, width, &one,
                        reinterpret_cast<const float*>(expanded), width, x, width, &zero, y + first,
                        static_cast<int>(rows)));
            }
        }

    // ------------------------------------------------------------------------------------

        void paged_attention_rows (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tables,
            long first_entry,
            long first_position,
            tensor& scratch
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            DLIB_CASSERT(q_heads > 0 && q_heads % layout.heads == 0 && first_entry >= 0 && first_position >= 0);
            const long hd = layout.head_dim;
            const size_t width = static_cast<size_t>(q_heads) * static_cast<size_t>(hd);
            DLIB_CASSERT(queries.size() % width == 0 && out.size() == queries.size());
            DLIB_CASSERT(!is_same_object(out, queries) && !is_same_object(out, pages) && !is_same_object(out, tables)
                && !is_same_object(out, scratch) && !is_same_object(queries, scratch)
                && !is_same_object(pages, scratch) && !is_same_object(tables, scratch));
            const long n = static_cast<long>(queries.size() / width);
            if (n == 0) return;
            const long long span = static_cast<long long>(first_position) + n;
            DLIB_CASSERT(span <= std::numeric_limits<int>::max() && hd <= std::numeric_limits<int>::max());
            DLIB_CASSERT(static_cast<size_t>(first_entry + (span - 1) / layout.block_positions) < tables.size());
            const long rep = q_heads / layout.heads;
            // The keys and the values of the span, then the queries, the outputs and the scores of a run of tokens.
            const size_t kv = static_cast<size_t>(layout.heads) * static_cast<size_t>(span) * static_cast<size_t>(hd);
            const size_t per_token = 2 * width + static_cast<size_t>(q_heads) * static_cast<size_t>(span);
            DLIB_CASSERT(scratch.size() >= 2 * kv + per_token, "scratch holds " << scratch.size()
                << " floats, less than the " << 2 * kv + per_token << " of the keys and values and of one token");
            const long run = static_cast<long>(std::min<size_t>(static_cast<size_t>(n),
                std::min<size_t>((scratch.size() - 2 * kv) / per_token,
                    static_cast<size_t>(std::numeric_limits<int>::max() / rep))));
            float* keys = scratch.device_write_only();
            float* values = keys + kv;
            float* heads_q = values + kv;
            float* heads_out = heads_q + static_cast<size_t>(run) * width;
            float* scores = heads_out + static_cast<size_t>(run) * width;
            paged_kv_expand(pages, layout, layer, reinterpret_cast<const int32_t*>(tables.device()) + first_entry,
                static_cast<long>(span), keys);
            const float* q = queries.device();
            float* o = out.device_write_only();
            const float one = 1, zero = 0;
            const int positions = static_cast<int>(span);
            const int dim = static_cast<int>(hd);
            const int kv_heads = static_cast<int>(layout.heads);
            const long long head_stride = span * hd;
            for (long first = 0; first < n; first += run)
            {
                const long count = std::min(run, n - first);
                // The columns of a key/value head: the tokens of each of its query heads.
                const int columns = static_cast<int>(rep * count);
                attention_queries(q + static_cast<size_t>(first) * width, q_heads, count, hd, layout.q8, heads_q);
                /*
                    The scores of a key/value head, a matrix of positions x columns in the
                    column-major order of BLAS, are its keys (dim x positions, transposed) times its
                    queries (dim x columns); the heads follow one another in the three arrays.
                */
                CHECK_CUBLAS(cublasSgemmStridedBatched(context(), CUBLAS_OP_T, CUBLAS_OP_N, positions, columns, dim,
                    &scale, keys, dim, head_stride, heads_q, dim, static_cast<long long>(columns) * dim, &zero,
                    scores, positions, static_cast<long long>(columns) * span, kv_heads));
                attention_softmax(scores, static_cast<long>(span), q_heads * count, count, first_position + first + 1);
                // The outputs (dim x columns) are the values (dim x positions) times the weights.
                CHECK_CUBLAS(cublasSgemmStridedBatched(context(), CUBLAS_OP_N, CUBLAS_OP_N, dim, columns, positions,
                    &one, values, dim, head_stride, scores, positions, static_cast<long long>(columns) * span, &zero,
                    heads_out, dim, static_cast<long long>(columns) * dim, kv_heads));
                rows_from_heads(heads_out, q_heads, count, hd, o + static_cast<size_t>(first) * width);
            }
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_



