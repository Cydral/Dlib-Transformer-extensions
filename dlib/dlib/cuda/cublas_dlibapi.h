// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuBLAS_H_
#define DLIB_DNN_CuBLAS_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"
#include "cuda_errors.h"
#include "kv_pages.h"

namespace dlib
{    
    namespace cuda 
    {        

    // -----------------------------------------------------------------------------------

        void gemm (
            float beta,
            tensor& dest,
            float alpha,
            const tensor& lhs,
            bool trans_lhs,
            const tensor& rhs,
            bool trans_rhs,
            operation_mode mode = operation_mode::CHANNEL_WISE
        );
    /*!
        requires
            - The dimensions of lhs and rhs must be compatible for matrix multiplication.
                The specific requirements depend on the mode:

                For CHANNEL_WISE mode (default):
                    - Let L == trans_lhs ? trans(mat(lhs)) : mat(lhs)
                    - Let R == trans_rhs ? trans(mat(rhs)) : mat(rhs)
                    - Let D == mat(dest)
                    - D.nr() == L.nr() && D.nc() == R.nc()
                        (i.e. dest must be preallocated and have the correct output dimensions)
                    - L.nc() == R.nr()

                For PLANE_WISE mode:
                    - lhs.num_samples() == rhs.num_samples() && lhs.k() == rhs.k()
                    - If !trans_lhs && !trans_rhs:
                        lhs.nc() == rhs.nr()
                        dest.nr() == lhs.nr() && dest.nc() == rhs.nc()
                    - If trans_lhs && !trans_rhs:
                        lhs.nr() == rhs.nr()
                        dest.nr() == lhs.nc() && dest.nc() == rhs.nc()
                    - If !trans_lhs && trans_rhs:
                        lhs.nc() == rhs.nc()
                        dest.nr() == lhs.nr() && dest.nc() == rhs.nr()
                    - If trans_lhs && trans_rhs:
                        lhs.nr() == rhs.nc()
                        dest.nr() == lhs.nc() && dest.nc() == rhs.nr()

        ensures
            - Performs matrix multiplication based on the specified mode:

                For CHANNEL_WISE mode:
                    - performs: dest = alpha*L*R + beta*mat(dest)
                        where L, R, and D are as defined above.

                For PLANE_WISE mode:
                    - Performs matrix multiplication for each corresponding 2D plane (nr x nc)
                        in lhs and rhs across all samples and channels.
                    - The operation is equivalent to performing the following for each sample
                        and channel:
                        dest[s][k] = alpha * (lhs[s][k] * rhs[s][k]) + beta * dest[s][k]
                        where [s][k] represents the 2D plane for sample s and channel k.
    !*/

    // ------------------------------------------------------------------------------------

        /*
            The products on weights in their GGUF formats and the attention of consecutive
            tokens that cuBLAS computes, with the kernels of cuda_gguf.h.  tensor_tools.h
            documents them.
        */

        bool gguf_gemm_computes (
            gguf_precision precision
        );

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
        );

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
        );

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuBLAS_H_


