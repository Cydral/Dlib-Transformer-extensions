// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_AVX2_DLIB_H_
#define DLIB_DNN_AVX2_DLIB_H_

/*
    This file contains the functions of the tensor tools written for the processors with AVX2,
    FMA and F16C.  tensor_tools.cpp calls them, on such a processor, in place of those of
    cpu_dlib.h: use the tt:: functions, which choose at run time.
*/

#include "tensor.h"
#include "operation_mode.h"

namespace dlib
{
    namespace avx2
    {

    // -----------------------------------------------------------------------------------

        bool is_available (
        );
        /*!
            ensures
                - returns true when the functions of this file compute: the processor has AVX2,
                  FMA and F16C, the system saves their registers, the build compiled the
                  kernels (avx2_kernels.h), and DLIB_DISABLE_CPU_FEATURES names none of these
                  instructions (simd/cpu_features.h).  The answer never changes during a run.
        !*/

    // -----------------------------------------------------------------------------------

        bool tanh (
            tensor& dest,
            const tensor& src
        );

        bool sigmoid (
            tensor& dest,
            const tensor& src
        );

        bool silu (
            tensor& dest,
            const tensor& src
        );

        bool gelu (
            tensor& dest,
            const tensor& src
        );
        /*!
            requires
                - The requirements of the functions of the same names of tensor_tools.h.
            ensures
                - returns false and leaves dest alone when is_available() is false.
                - Otherwise computes what the function of the same name of cpu_dlib.h
                  computes, eight values at a time, and returns true.  The results differ
                  from those of cpu_dlib.h by the rounding of the exponentials, a few units
                  of the last place, and for gelu() by an approximation of the error function
                  within 1.5e-7.
        !*/

    // -----------------------------------------------------------------------------------

        bool softmax (
            tensor& dest,
            const tensor& src,
            operation_mode mode
        );

        bool softmax_all (
            tensor& dest,
            const tensor& src
        );
        /*!
            requires
                - The requirements of the functions of the same names of tensor_tools.h.
            ensures
                - returns false and leaves dest alone when is_available() is false, or when
                  the values of a softmax do not follow one another in memory: the mode
                  CHANNEL_WISE over tensors with several rows or columns.
                - Otherwise computes what the function of the same name of cpu_dlib.h
                  computes and returns true.
        !*/

    // -----------------------------------------------------------------------------------

        bool gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        );
        /*!
            requires
                - The requirements of tt::gguf_matvec().
            ensures
                - returns false and leaves dest alone when is_available() is false.
                - Otherwise computes what cpu::gguf_matvec() computes and returns true.  Each
                  group of 32 weights is decoded once for all the vectors of src and
                  multiplied with fused operations; the rows of a large product are shared
                  among the threads of dlib (parallel_for).  The sums run in another order
                  than those of cpu::gguf_matvec(), whose results they follow within the
                  rounding of single precision.
        !*/

    // -----------------------------------------------------------------------------------

    }
}

#ifdef NO_MAKEFILE
#include "avx2_dlib.cpp"
#endif

#endif // DLIB_DNN_AVX2_DLIB_H_
