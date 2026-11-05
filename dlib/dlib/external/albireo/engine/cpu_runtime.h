// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_CPU_RUNTIME_H_
#define ALBIREO_ENGINE_CPU_RUNTIME_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache.h"
#include "model_file.h"
#include "offload.h"
#include "qkernels.h"
#include "runtime.h"
#include "thread_pool.h"

namespace albireo
{

    class cpu_runtime : public model_runtime
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The forward pass of a decoder-only model on the processor.

                The weights stay in the memory mapping of the GGUF file, in their
                quantized form, and every product goes through the kernels of
                qkernels.h.  Activations are single-precision floats; keys and values are
                stored in half precision in a kv_pool, one paged sequence per request
                slot.

                The numerical path follows the reference implementations of the
                supported families: pre-normalization residual blocks with RMS norms,
                grouped-query attention with rotary positions (consecutive pairs or
                halves, with linear, YaRN, llama 3 or LongRoPE frequency adjustments),
                optional QKV biases and per-head QK normalization, SwiGLU feed-forward
                blocks or mixtures of SwiGLU experts, and the global scale factors of the
                families that declare them.  The fused matrices of phi3 (attn_qkv, and
                ffn_up with the gate) are used as row ranges, one product per range.

                Attention runs over the paged keys of each sequence directly, by tiles: a
                task takes the heads that share one key/value head for up to tile_rows
                rows of consecutive tokens of one sequence, and reads the cache block after
                block with a running maximum and sum (the softmax of FlashAttention), so
                that every key and value read from memory serves all the rows of the tile.
                When the tiles are fewer than the threads, as in the generation of a single
                answer, each head of the group makes a task of its own.

                With a matmul_offload, the products with the weight matrices go through it,
                and it decides, product by product, whether a device computes them.  When the
                offload computes whole layers (layer_pass), the runtime of a dense model hands
                it the layers of its passes, once the logits of a pass of the device over
                probe_tokens tokens agreed with those of the processor alone: the hidden
                states then stay on the device, and the processor keeps the attention.  The
                batches with DeepStack slices keep the products, as does every pass after a
                failure of the device.

                A batch that gives the rotary positions of its tokens (forward_batch::
                positions) rotates each pair of a head by the position component its
                section names (model_config::rope_sections), and the rows it gives as
                embeddings take them in place of the embedding table, their DeepStack
                slices being added to their hidden states after the first layers.  A batch
                without them runs the path of the text models unchanged.
        !*/
    public:
        cpu_runtime(std::shared_ptr<model_file> file, const model_config& config, thread_pool& pool,
            size_t kv_budget_bytes, long kv_block_positions = 256);

        const model_config& config() const override { return cfg_; }

        int create_sequence() override;
        void release_sequence(int seq) override;
        long sequence_length(int seq) const override;
        void truncate_sequence(int seq, long length) override;
        bool reserve_sequence(int seq, long length) override;
        bool copy_sequence(int src, int dst, long length) override;

        void forward(const forward_batch& batch, std::vector<float>& logits) override;

        size_t kv_bytes_used() const override { return kv_.used_blocks() * kv_.block_bytes(); }
        size_t kv_bytes_budget() const override { return kv_.max_blocks() * kv_.block_bytes(); }
        size_t kv_bytes_per_token() const override
        {
            return 2 * static_cast<size_t>(cfg_.n_layers) * cfg_.kv_dim() * sizeof(uint16_t);
        }
        void trim_memory() override { kv_.trim(); }
        void set_busy(bool busy) override
        {
            if (offload_) offload_->set_busy(busy);
        }
        size_t weight_bytes() const override { return static_cast<size_t>(file_->size()); }

        std::string description() const override;

        // Hands the products, or the layers, of the following passes to a device.
        void set_offload(std::unique_ptr<matmul_offload> offload);

        /*!
            requires
                - no forward pass has run, or the rotary divisors stay the same.
            ensures
                - Chooses the rotary divisors of a model with LongRoPE for sequences of up to
                  positions tokens (select_rope_factors()).
        !*/
        void set_context(long positions);

        const std::vector<float>& hidden() const override { return hidden_out_; }

    private:
        struct layer
        {
            std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv;
            weight_matrix wq, wk, wv, wo;
            weight_matrix w_gate, w_up, w_down;
            weight_matrix router;
            std::vector<weight_matrix> e_gate, e_up, e_down;
        };

        // The rows of queries of an attention tile, at most: the scores of a tile over one
        // block of the cache stay within a few tens of kilobytes.
        static const long tile_rows = 64;

        // The tokens of the passes that compare the layers of the device with the processor.
        static const long probe_tokens = 8;

        // Failures for lack of device memory in a row after which the layers stay on the processor.
        static const int max_layer_pass_memory_failures = 4;

        // The passes that follow a check of the layer pass put off for lack of room, before the next one.
        static const int layer_pass_check_interval = 64;

        // Consecutive tokens of one segment of the batch, which attend together.
        struct attention_tile
        {
            int sequence;
            long start;     // first token in the batch
            long count;
        };

        // The buffers of one thread of the attention.
        struct attention_work
        {
            aligned_floats q, s, o, max, sum, scratch;
        };

        void rms_norm_rows(const float* x, const float* gamma, float* y, long rows, long width, float eps);
        void ensure_rope(long positions);
        void apply_rope(float* rows, long n_rows, long n_heads, const std::vector<long>& positions);
        // The rotation of the tokens of a batch that gives their positions: the cosines and
        // sines of every pair of each token, gathered once per pass (prepare_given_positions),
        // then applied to the queries and keys of every layer.
        void prepare_given_positions(const forward_batch& batch, long T);
        void apply_given_rope(float* rows, long n_rows, long n_heads);
        // The rows of the batch that come from an encoder: their embeddings, and the slice of
        // a DeepStack layer added after it.
        void check_given_embeddings(const forward_batch& batch, long T) const;
        void add_given_embeddings(const forward_batch& batch, long slice);
        void attention(long layer, const forward_batch& batch, const std::vector<long>& positions);
        void moe_ffn(const layer& L, long n_tokens);
        void embed(const std::vector<int>& tokens);
        /*
            The forward pass, with the layers on the device through layer_pass_ when on_device is
            true; false when the device did not take the pass, which then did nothing.
        */
        bool run(const forward_batch& batch, std::vector<float>& logits, bool on_device);
        bool takes_layer_pass(const forward_batch& batch) const;
        void check_layer_pass();
        void drop_layer_pass(const std::string& why);
        // Y = X W^T through the offload when there is one, with the kernels of qkernels.h
        // otherwise.
        void product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            qk_path path = qk_path::automatic);

        std::shared_ptr<model_file> file_;
        model_config cfg_;
        thread_pool& pool_;
        std::unique_ptr<matmul_offload> offload_;
        // Destroyed before offload_, which it uses.
        std::unique_ptr<layer_pass> layer_pass_;
        bool layer_pass_checked_ = false;
        int layer_pass_memory_failures_ = 0;
        int layer_pass_wait_ = 0;
        // The products of the probe passes of the processor do not go to the offload.
        bool host_only_ = false;
        kv_pool kv_;
        std::vector<std::unique_ptr<kv_sequence>> sequences_;
        std::vector<int> free_sequences_;

        std::vector<layer> layers_;
        weight_matrix token_embd_, output_;
        std::vector<float> output_norm_;

        // Rotary tables: cos and sin for every position computed so far.
        std::vector<float> rope_cos_, rope_sin_;
        long rope_positions_ = 0;
        // The position component of each rotary pair (0 time, 1 row, 2 column), and the
        // cosines and sines of the tokens of a batch that gives their positions.
        std::vector<int> pair_component_;
        std::vector<float> given_cos_, given_sin_;

        // Work buffers, reused from one pass to the next.
        aligned_floats x_, xn_, q_, k_, v_, ctx_, tmp_, gate_, up_;
        aligned_floats router_logits_, expert_x_, expert_gate_, expert_up_, expert_y_;
        std::vector<float> hidden_out_, router_probs_;
        std::vector<long> router_order_;
        std::vector<std::vector<std::pair<long, float>>> routed_;
        std::vector<qk_scratch> scratch_;
        std::vector<attention_work> attention_work_;
        std::vector<attention_tile> tiles_;
        std::vector<long> positions_;
    };

}

#endif // ALBIREO_ENGINE_CPU_RUNTIME_H_
