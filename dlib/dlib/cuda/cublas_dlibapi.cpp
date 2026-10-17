// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_CPP_
#define DLIB_DNN_CuBLAS_CPP_

#ifdef DLIB_USE_CUDA

#include "cublas_dlibapi.h"
#include "cuda_utils.h"

#include <cublas_v2.h>
#include <algorithm>
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

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_CPP_



