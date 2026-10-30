// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_OFFLOAD_H_
#define ALBIREO_ENGINE_OFFLOAD_H_

#include <functional>
#include <string>

#include "qkernels.h"

namespace albireo
{

    class matmul_offload
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A device that computes some of the products of a cpu_runtime with its weight
                matrices: a graphics processor through CUDA, or the NPU and the integrated
                graphics processor of an Intel processor through OpenVINO.

                The runtime keeps the rest of the forward pass on the processor (the
                normalizations, the rotary positions, the attention over the paged keys and
                values) and hands every product to this object together with the way to
                compute it on the processor.  The object then decides where the product runs,
                from the time each place took on products of the same shape: the device wins
                the products over many tokens, which are bound by arithmetic, and the
                processor often keeps those of a single token, which are bound by the reading
                of the weights and which its quantized kernels serve well.

            THREAD SAFETY
                One thread uses an object at a time: the thread that runs the forward passes
                of its model.
        !*/
    public:
        virtual ~matmul_offload() {}

        /*!
            requires
                - host computes the product on the processor, as qk_matmul() does.
            ensures
                - Computes Y[t][o] = sum_i W[o][i] * X[t][i] for t < n_tokens and o < W.rows,
                  with the layout of qk_matmul(), either on the device or by calling host().
        !*/
        virtual void matmul(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
            float* Y, long y_stride, const std::function<void()>& host) = 0;

        /*!
            ensures
                - Tells whether the model has requests in progress or waiting.  When the
                  device lacks room, copies of the models without requests leave it, the
                  least recently used model first, for those of a model that has some; the
                  copies of a model with requests stay until its requests are served.
        !*/
        virtual void set_busy(bool busy) = 0;

        // A short description, such as "NVIDIA GeForce RTX 4070 through CUDA".
        virtual std::string description() const = 0;
    };

}

#endif // ALBIREO_ENGINE_OFFLOAD_H_
