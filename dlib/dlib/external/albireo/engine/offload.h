// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_OFFLOAD_H_
#define ALBIREO_ENGINE_OFFLOAD_H_

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "qkernels.h"

namespace albireo
{

    /*!
        The weights of a dense decoder whose layers a layer_pass computes: blocks normalized
        before the attention and before the feed-forward block by RMS norms, with the four
        projections of the attention and a SwiGLU feed-forward block.  The norms and the
        matrices belong to the runtime, which keeps them while the layer_pass lives.
    !*/
    struct dense_layer
    {
        const float* attn_norm = nullptr;   // d_model values
        const float* ffn_norm = nullptr;    // d_model values
        weight_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
    };

    struct dense_model
    {
        long d_model = 0;
        long q_dim = 0;
        long kv_dim = 0;
        long d_ffn = 0;
        float rms_eps = 0;
        // The factor of the outputs of the blocks added to the hidden states.
        float residual_scale = 1;
        std::vector<dense_layer> layers;
    };

    /*!
        The failure of a device during a layer_pass.  memory tells whether the device lacked
        memory, which the end of another program may give back, rather than failed.
    !*/
    class device_failure : public std::runtime_error
    {
    public:
        device_failure(const std::string& what, bool lack_of_memory)
            : std::runtime_error(what), memory(lack_of_memory) {}

        bool memory;
    };

    class layer_pass
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The layers of a dense decoder computed on a device, around the attention that
                the processor runtime keeps.  The hidden states of the tokens of a pass stay on
                the device from the first layer to the last: each layer sends the processor the
                queries, keys and values of the tokens and receives the outputs of their
                attention, one exchange per layer instead of one per product.  The device
                computes the normalizations of the blocks, the products, the SwiGLU gates and
                the residual sums; the processor the biases, the normalizations of queries and
                keys, the rotary positions, the keys and values of the sequences and the
                attention.

                A pass calls begin(), then attention_inputs() and attention_output() for each
                layer in turn, then end().  The functions that compute throw device_failure
                when the device fails, the pass then being lost: the runtime computes it again
                with its products.

            THREAD SAFETY
                One thread uses an object at a time: the thread that runs the forward passes
                of its model.
        !*/
    public:
        virtual ~layer_pass() {}

        /*!
            requires
                - x holds n rows of d_model floats, the hidden states of the tokens of the pass
                  before the first layer.
            ensures
                - Starts a pass and returns true, or returns false when the device does not
                  take the pass now: the copies of the weights lack room, or the pass is too
                  small for the device.  The runtime then computes the pass with its products,
                  of which the device may still take some.
        !*/
        virtual bool begin(const float* x, long n) = 0;

        /*!
            requires
                - begin() returned true, and every layer before layer had its
                  attention_output().
            ensures
                - Writes the queries, keys and values of the n tokens of the pass for layer:
                  n rows of q_dim floats into q, n rows of kv_dim floats into k and into v.
        !*/
        virtual void attention_inputs(long layer, float* q, float* k, float* v) = 0;

        /*!
            requires
                - attention_inputs() was called for layer, and ctx holds the n rows of q_dim
                  floats the attention of the layer gave.
            ensures
                - Adds the projection of ctx, then the output of the feed-forward block of
                  layer, to the hidden states.
        !*/
        virtual void attention_output(long layer, const float* ctx) = 0;

        /*!
            requires
                - every layer had its attention_output().
            ensures
                - Writes the hidden states after the last layer of the given rows of the pass
                  into x, row r at x + r * d_model, and ends the pass.
        !*/
        virtual void end(const std::vector<long>& rows, float* x) = 0;

        // The tokens a pass needs at least for begin() to take it.
        virtual long min_tokens() const = 0;

        /*!
            ensures
                - Writes in the log of the device the outcome of the comparison that the
                  runtime makes between passes of this object and passes of the processor
                  before it uses the object (ok tells whether they agreed), or why the runtime
                  stops using the object (ok false).
        !*/
        virtual void report(bool ok, const std::string& detail) = 0;
    };

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

                A device that computes whole layers also gives a layer_pass
                (make_layer_pass()), which takes the layers of a dense model, normalizations
                included, and keeps the hidden states of a pass on the device; the products of
                the other passes, and the final products of the logits, still come here.

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

        /*!
            ensures
                - Returns an object that computes the layers of model on the device, or null
                  when the device takes products only.  The object uses this one and the
                  memory of the weights of model, and is destroyed before them.
        !*/
        virtual std::unique_ptr<layer_pass> make_layer_pass(const dense_model& model)
        {
            (void)model;
            return std::unique_ptr<layer_pass>();
        }
    };

}

#endif // ALBIREO_ENGINE_OFFLOAD_H_
