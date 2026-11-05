// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_GGUF_H_
#define DLIB_CUDA_GGUF_H_

#include "tensor.h"

#ifdef DLIB_USE_CUDA

namespace dlib
{
    namespace cuda
    {
        /*
            The kernels of the operations of tensor_tools.h on weight matrices kept in their
            GGUF formats (gguf_blocks.h), and the row operations of the forward pass of a
            decoder.  tensor_tools.h documents them.
        */

        void gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        );

        void gguf_dequantize (
            tensor& dest,
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes
        );

        void rms_norm_rows (
            tensor& dest,
            const tensor& src,
            const tensor& gamma,
            float eps
        );

        void silu_multiply (
            tensor& gate,
            const tensor& up
        );
    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_CUDA_GGUF_H_
