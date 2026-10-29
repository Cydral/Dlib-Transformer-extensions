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
                halves, with linear, YaRN or llama 3 frequency adjustments), optional QKV
                biases and per-head QK normalization, SwiGLU feed-forward blocks or
                mixtures of SwiGLU experts, and the global scale factors of the families
                that declare them.

                Attention runs over the paged keys of each sequence directly, one task per
                token and head, with a numerically stable softmax.

                With a matmul_offload, the products with the weight matrices go through it,
                and it decides, product by product, whether a device computes them.
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
        size_t weight_bytes() const override { return static_cast<size_t>(file_->size()); }

        std::string description() const override;

        // Hands the products of the following passes to a device.
        void set_offload(std::unique_ptr<matmul_offload> offload) { offload_ = std::move(offload); }

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

        void rms_norm_rows(const float* x, const float* gamma, float* y, long rows, long width, float eps);
        void ensure_rope(long positions);
        void apply_rope(float* rows, long n_rows, long n_heads, const std::vector<long>& positions);
        void attention(long layer, const forward_batch& batch, const std::vector<long>& positions);
        void moe_ffn(const layer& L, long n_tokens);
        void embed(const std::vector<int>& tokens);
        // Y = X W^T through the offload when there is one, with the kernels of qkernels.h
        // otherwise.
        void product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            qk_path path = qk_path::automatic);

        std::shared_ptr<model_file> file_;
        model_config cfg_;
        thread_pool& pool_;
        std::unique_ptr<matmul_offload> offload_;
        kv_pool kv_;
        std::vector<std::unique_ptr<kv_sequence>> sequences_;
        std::vector<int> free_sequences_;

        std::vector<layer> layers_;
        weight_matrix token_embd_, output_;
        std::vector<float> output_norm_;

        // Rotary tables: cos and sin for every position computed so far.
        std::vector<float> rope_cos_, rope_sin_;
        long rope_positions_ = 0;

        // Work buffers, reused from one pass to the next.
        std::vector<float> x_, xn_, q_, k_, v_, ctx_, tmp_, gate_, up_;
        std::vector<float> router_logits_, hidden_out_;
        std::vector<qk_scratch> scratch_;
        std::vector<std::vector<float>> scores_;
        std::vector<long> positions_;
    };

}

#endif // ALBIREO_ENGINE_CPU_RUNTIME_H_
