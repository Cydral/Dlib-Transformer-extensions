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

        The attention of a layer may add biases to its queries, keys and values and normalize
        the rows of its queries and keys (null pointers when the model does neither); its
        queries and keys then turn by the rotary positions of their tokens, the first rope_dims
        values of each head in pairs, (i, i + rope_dims/2) when neox is true and (2i, 2i + 1)
        otherwise.  attention tells whether the model leaves nothing else to its attention, so
        that a device may compute it (layer_pass::begin_attention()).
    !*/
    struct dense_layer
    {
        const float* attn_norm = nullptr;   // d_model values
        const float* ffn_norm = nullptr;    // d_model values
        weight_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
        const float* bq = nullptr;          // q_dim values
        const float* bk = nullptr;          // kv_dim values
        const float* bv = nullptr;          // kv_dim values
        const float* q_norm = nullptr;      // head_dim values
        const float* k_norm = nullptr;      // head_dim values
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

        long n_heads = 0;
        long n_kv_heads = 0;
        long head_dim = 0;
        long rope_dims = 0;
        bool neox = true;
        // The factor of the products of the queries and the keys.
        float attention_scale = 1;
        bool attention = false;
    };

    /*!
        The tokens of a pass whose attention a device computes (layer_pass::begin_attention()),
        their keys and values going into the blocks of a kv_pool.  Token t of the pass sits at
        positions[t] of its sequence, where its keys and values go, and attends the positions
        [0, positions[t]] of that sequence, whose blocks are blocks[firsts[t] + j] for j <=
        positions[t] / pool->block_positions().  Its rotary angles are angles[t * rope_dims +
        i], rope_dims/2 cosines then rope_dims/2 sines.  Each segment of the pass describes the
        blocks of one sequence: its run of blocks starts at first, and it held length positions
        before the pass, which writes count more.
    !*/
    class kv_pool;

    struct attention_plan
    {
        struct segment
        {
            long first = 0;
            long length = 0;
            long count = 0;
        };

        kv_pool* pool = nullptr;
        std::vector<long> positions;
        std::vector<long> firsts;
        std::vector<int> blocks;
        std::vector<float> angles;
        std::vector<segment> segments;
    };

    /*!
        The failure of a device during a layer_pass.  memory tells whether the device lacked
        memory, which the end of another program may give back, rather than failed.  changed
        tells that the device gave up an arithmetic that could not hold a value of the pass and
        computes the next passes with another one: the pass is lost, and the device stays in use.
    !*/
    class device_failure : public std::runtime_error
    {
    public:
        device_failure(const std::string& what, bool lack_of_memory, bool arithmetic_changed = false)
            : std::runtime_error(what), memory(lack_of_memory), changed(arithmetic_changed) {}

        bool memory;
        bool changed;
    };

    /*!
        The tokens of a pass up to which a device attends each token with a kernel of its own, as
        the steps of a generation need (layer_pass::begin_attention()); it attends the tokens of a
        longer pass together, with matrix products, when it can.
    !*/
    const long few_attention_tokens = 8;

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

                A device that keeps copies of the blocks of the keys and values may also attend
                the tokens of a pass itself: after begin(), begin_attention() takes the pass,
                whose layers then run whole on the device through attend(), and the pass
                exchanges nothing with the processor before end(), which returns the keys and
                values of its tokens with the hidden states.  The passes of few_attention_tokens
                tokens at most, those of a generation, are attended token by token (attends());
                the device may attend the longer ones, the chunks of a prompt, with matrix
                products over all their tokens (attends_many()).

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

        /*
            The tokens from which a pass computes its products another way, from tiles of the
            weights expanded rather than from their blocks in place, which the runtime checks
            apart; 0 for a pass that computes them one way.
        */
        virtual long tiled_tokens() const { return 0; }

        /*
            Whether the runtime times the passes of this object against the processor and keeps
            for the device those it computes faster; false when the configuration sends every
            pass to the device.
        */
        virtual bool timed() const { return true; }

        /*!
            ensures
                - Writes in the log of the device the outcome of the comparison that the
                  runtime makes between passes of this object and passes of the processor
                  before it uses the object (ok tells whether they agreed), or why the runtime
                  stops using the object (ok false).
        !*/
        virtual void report(bool ok, const std::string& detail) = 0;

        /*
            Whether the device can attend the tokens of a pass itself (begin_attention()), for the
            model and as allow_attention() leaves it.
        */
        virtual bool attends() const { return false; }

        // Lets the device attend the tokens of the passes, or keeps their attention on the processor.
        virtual void allow_attention(bool on) { (void)on; }

        /*
            Whether the device can attend the passes of more than few_attention_tokens tokens,
            for the model and as allow_attention() and allow_long_attention() leave it.
        */
        virtual bool attends_many() const { return false; }

        // Keeps the attention of the passes of more than few_attention_tokens tokens on the processor, or not.
        virtual void allow_long_attention(bool on) { (void)on; }

        /*!
            ensures
                - Called when a pass of the device differed from the processor.  Returns true
                  when the device computed its products with an arithmetic of lower precision
                  that it gives up for the passes to come, which the runtime then compares
                  again; false when it has no other way to compute.  A call gives up one
                  arithmetic, the one of lowest precision among those the pass still uses.
        !*/
        virtual bool retreat() { return false; }

        /*!
            requires
                - begin() returned true, and no layer of the pass was computed yet.
                - plan describes the n tokens of the pass (attention_plan), their keys and values
                  at the positions before theirs being in the blocks of plan.pool.
            ensures
                - Returns true when the device computes the attention of the pass itself: the
                  device keeps copies of the blocks of the pool, which it updates from the pool
                  before the pass, and the layers then run through attend(); end() writes the
                  keys and values of the tokens of the pass into their blocks of the pool.
                - Returns false when the device does not attend the pass, for lack of room for
                  the blocks, because attends() is false, or, for a pass of more than
                  few_attention_tokens tokens, because attends_many() is false or the pass holds
                  the tokens of several sequences: the layers then run through
                  attention_inputs() and attention_output().
        !*/
        virtual bool begin_attention(const attention_plan& plan)
        {
            (void)plan;
            return false;
        }

        /*!
            requires
                - begin_attention() returned true for this pass, and every layer before layer had
                  its attend().
            ensures
                - Computes layer for the tokens of the pass on the device, their attention
                  included: the normalizations, the products, the biases, the normalizations of
                  the queries and keys, the rotary positions, the keys and values written into
                  the copies of their blocks, the attention, the feed-forward block and the
                  residual sums.
        !*/
        virtual void attend(long layer) { (void)layer; }

        /*!
            ensures
                - The copies of the device hold no row of block of pool from row on: the
                  sequence of the block was cut there, and the processor writes these rows
                  again.
        !*/
        virtual void truncated(const kv_pool& pool, int block, long row)
        {
            (void)pool;
            (void)block;
            (void)row;
        }
    };

    /*!
        The weights of a hybrid decoder whose layers a hybrid_pass computes (cpu_runtime.h): a
        Mamba-3 layer or a differential attention in each block, then a mixture of experts in
        a latent space beside a shared expert, the hidden states turning on their sphere
        towards the output of each.  The matrices and the constants belong to the runtime,
        which keeps them while the hybrid_pass lives.

        The matrices of the experts of a layer lie one after the other in two matrices: the
        rows of expert e are the rows [e*r, (e + 1)*r) of experts_up, r == experts_up.rows /
        n_experts, and likewise in experts_down.
    !*/
    struct hybrid_layer
    {
        // A Mamba-3 layer, or a differential attention.
        bool mamba = false;

        // Mamba-3: the projections of the hidden states, then the constants of the heads.
        weight_matrix ssm_in, ssm_dyn;
        const float* b_norm = nullptr;      // ssm_state values
        const float* c_norm = nullptr;      // ssm_state values
        const float* b_bias = nullptr;      // ssm_heads x ssm_rank x ssm_state values
        const float* c_bias = nullptr;      // ssm_heads x ssm_rank x ssm_state values
        const float* dt_bias = nullptr;     // ssm_heads values
        const float* skip = nullptr;        // ssm_heads values
        const float* mimo_x = nullptr;      // ssm_heads x ssm_rank x ssm_head_dim values
        const float* mimo_z = nullptr;      // ssm_heads x ssm_rank x ssm_head_dim values
        const float* mimo_o = nullptr;      // ssm_heads x ssm_rank x ssm_head_dim values

        // An attention: the products that read the hidden states.
        weight_matrix wq, wa_k, wb_k, wa_v, wb_v, shift_k, shift_v, noise, gate;

        // The projection of the output of the mixer of either kind.
        weight_matrix wo;

        // The experts.
        weight_matrix router, latent_down, latent_up, experts_up, experts_down, shared_up, shared_down;
        const float* expert_bias = nullptr; // n_experts values

        // The scale and the bias of the two geodesic updates of the block.
        float mixer_scale = 1, mixer_bias = 0, ffn_scale = 1, ffn_bias = 0;

        // Where the state of a Mamba-3 layer lies in the states of a sequence, in floats.
        size_t state_offset = 0;
    };

    struct hybrid_model
    {
        long d_model = 0;
        float rms_eps = 0;
        long ssm_heads = 0;
        long ssm_head_dim = 0;
        long ssm_state = 0;
        long ssm_rank = 0;
        long ssm_angles = 0;
        long n_experts = 0;
        long n_experts_used = 0;
        bool expert_weights_norm = true;
        float expert_weights_scale = 1;
        // The floats of the states of the layers of a sequence.
        size_t state_floats = 0;
        // The matrix of the logits.
        weight_matrix output;
        std::vector<hybrid_layer> layers;
    };

    class hybrid_pass
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The layers of a hybrid decoder computed on a device.  The hidden states of the
                tokens of a pass stay on the device from the first layer to the last, with the
                products, the Mamba-3 layers, the experts and the geodesic updates; the
                processor keeps the attention proper of the attention layers, for which the
                device sends it the products that read the hidden states and receives the
                gated outputs, one exchange per attention layer.

                The device also keeps the states of the Mamba-3 layers of the sequences it
                computes, in the layout of the runtime (hybrid_layer::state_offset), which a
                generated token would otherwise carry across the bus both ways.  The states of
                a sequence then live on the device, and the runtime brings them back when it
                needs them (fetch()): to save them at the end of a chunk of a prompt, to
                compute a pass itself, or before its model rests.  Each pass keeps the states
                its sequences had before it, so that a pass the device fails goes back to the
                processor (recover()).

                A pass calls begin(), then for each layer in turn mixer(), or
                attention_inputs() and attention_output(), followed by experts(), then end().
                The functions that compute throw device_failure when the device fails, the
                pass then being lost.

            THREAD SAFETY
                One thread uses an object at a time: the thread that runs the forward passes
                of its model.
        !*/
    public:
        // The consecutive tokens of one sequence in a pass.
        struct segment
        {
            int sequence = 0;
            long start = 0;
            long count = 0;
            // The states of the layers of the sequence in the memory of the host.
            float* state = nullptr;
            // Whether these hold the states of the Mamba-3 layers as they stand before the pass.
            bool current = false;
        };

        virtual ~hybrid_pass() {}

        /*!
            requires
                - x holds n rows of d_model floats, the hidden states of the tokens of the pass
                  before the first layer, and segments describes these tokens, each sequence
                  once.
                - The device holds the states of every sequence whose segment is not current
                  (holds()).
            ensures
                - Starts a pass and returns true, the states of the sequences that are current
                  on the host crossing to the device when it does not hold them; or returns
                  false when the device does not take the pass now, for lack of room or
                  because the pass is too small, the states it holds staying as they were.
        !*/
        virtual bool begin(const float* x, long n, const std::vector<segment>& segments) = 0;

        /*!
            requires
                - begin() returned true, layer is a Mamba-3 layer and the layers before it were
                  computed.
            ensures
                - Computes the mixer of layer for the tokens of the pass, the tokens of each
                  sequence in order, and turns the hidden states towards its output.
        !*/
        virtual void mixer(long layer) = 0;

        /*!
            requires
                - begin() returned true, layer is an attention layer and the layers before it
                  were computed.
            ensures
                - Writes the products of layer that read the hidden states of the n tokens: the
                  queries (n rows of wq.rows floats); the factors of the keys then those of the
                  values, each as n rows of wa_k.rows floats into factors_a and n rows of
                  wb_k.rows floats into factors_b; for each token, the shifts of its keys then
                  those of its values (n rows of 2 * shift_k.rows floats); the shares of the
                  noise (n rows of noise.rows floats); the gate (n rows of gate.rows floats).
        !*/
        virtual void attention_inputs(long layer, float* q, float* factors_a, float* factors_b, float* shift,
            float* noise, float* gate) = 0;

        /*!
            requires
                - attention_inputs() was called for layer, and gated holds the n rows of
                  gate.rows floats the attention of the layer gave, its gate applied.
            ensures
                - Projects gated and turns the hidden states towards the projection.
        !*/
        virtual void attention_output(long layer, const float* gated) = 0;

        /*!
            requires
                - the mixer or the attention of layer was computed.
            ensures
                - Computes the mixture of experts of layer and turns the hidden states towards
                  its output.
        !*/
        virtual void experts(long layer) = 0;

        /*!
            requires
                - every layer was computed.
            ensures
                - Writes the hidden states after the last layer of the given rows of the pass
                  into x, row r at x + r * d_model, and ends the pass.
                - When logits is not null and the device holds the matrix of the logits, also
                  writes the logits of these rows, one row of output.rows floats after the
                  other, and returns true; returns false otherwise, the runtime then computing
                  them from x.
        !*/
        virtual bool end(const std::vector<long>& rows, float* x, float* logits) = 0;

        // Whether the device holds the states of the Mamba-3 layers of a sequence.
        virtual bool holds(int sequence) const = 0;

        /*!
            requires
                - holds(sequence), and no pass is under way.
            ensures
                - Writes the states of the Mamba-3 layers of the sequence, as its last pass left
                  them, into state, each layer at its place; the device keeps them.
            throws
                - device_failure when the device fails: the states of the sequence are lost.
        !*/
        virtual void fetch(int sequence, float* state) = 0;

        /*!
            requires
                - a function of the last pass threw device_failure, and sequence belongs to
                  that pass.
            ensures
                - Returns true once state holds the states of the Mamba-3 layers of the sequence
                  as they stood before the pass: untouched when they were current on the host
                  at begin(), written from the copy the device kept otherwise.  The device then
                  holds no state of the sequence.
                - Returns false when the device cannot give that copy: the states are lost.
        !*/
        virtual bool recover(int sequence, float* state) = 0;

        // The device drops the states of a sequence: the runtime changed or released them.
        virtual void forget(int sequence) = 0;

        // The tokens a pass needs at least for begin() to take it.
        virtual long min_tokens() const = 0;

        // As layer_pass::tiled_tokens().
        virtual long tiled_tokens() const { return 0; }

        // As layer_pass::timed().
        virtual bool timed() const { return true; }

        // As layer_pass::retreat().
        virtual bool retreat() { return false; }

        // As layer_pass::report().
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
                the other passes, and the final products of the logits, still come here.  For a
                hybrid decoder, it gives a hybrid_pass (make_hybrid_pass()), which also keeps
                the states of the Mamba-3 layers of the sequences on the device.

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
            requires
                - items holds count products of fewer than qk_float_product_tokens tokens, none
                  of which writes what another one reads or writes, and host computes them all
                  on the processor, as qk_matmul_group() does.
            ensures
                - Returns true after calling host when the object leaves every product of the
                  group to the processor without measuring any: the products then share one
                  parallel section of the processor instead of one each.
                - Returns false, nothing being computed, when the device computes or measures
                  one of the products: the runtime then hands them over one by one (matmul()).
        !*/
        virtual bool matmul_group(const qk_product* items, long count, const std::function<void()>& host)
        {
            (void)items;
            (void)count;
            (void)host;
            return false;
        }

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

        /*
            The bytes of the weights of the model, as its file holds them, whose products of few
            tokens, as those of a generated token, the device computes from its own copies, so that the
            processor no longer reads their pages; 0 for a device that keeps no copies or leaves these
            products to the processor.  Read on the thread of the forward passes.
        */
        virtual size_t mapped_bytes() const { return 0; }

        /*
            The bytes of the copies of the device that the system charges to the commit of the
            process, though they take no memory of the host while the device holds them: Windows
            charges every allocation of a card, in case it moves the allocation to the memory of the
            host.  0 for a device whose copies the system does not charge, or charges as memory of the
            host.
        */
        virtual size_t charged_bytes() const { return 0; }

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

        /*!
            ensures
                - Returns an object that computes the layers of a hybrid decoder on the device,
                  or null when the device takes products only or cannot compute this model.
                  The object uses this one and the memory of the weights of model, and is
                  destroyed before them.
        !*/
        virtual std::unique_ptr<hybrid_pass> make_hybrid_pass(const hybrid_model& model)
        {
            (void)model;
            return std::unique_ptr<hybrid_pass>();
        }
    };

}

#endif // ALBIREO_ENGINE_OFFLOAD_H_
