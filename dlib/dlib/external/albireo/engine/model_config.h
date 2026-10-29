// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_MODEL_CONFIG_H_
#define ALBIREO_ENGINE_MODEL_CONFIG_H_

#include <string>
#include <vector>

#include "model_file.h"

namespace albireo
{

    enum class rope_style
    {
        norm,   // consecutive pairs (x[2i], x[2i+1]), the llama convention of GGUF files
        neox    // halves (x[i], x[i + d/2]), the convention of qwen, gemma and phi
    };

    struct model_config
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The geometry and the numeric conventions of a decoder-only model, read
                from the metadata of its GGUF file.  Everything the forward pass depends
                on is resolved here once, so that the runtimes branch on fields rather
                than on architecture names.

            SUPPORTED ARCHITECTURES
                llama, mistral and mistral3 (dense, GQA, SwiGLU, tied or separate
                embeddings, llama 3 frequency factors, YaRN and the long context query
                scaling of Ministral 3), qwen2 (with QKV biases), qwen3 (with QK
                normalization), qwen3moe, mixtral and granitemoe (mixtures of experts),
                granite (with its embedding, residual, attention and logit scales).
        !*/

        std::string arch;               // general.architecture
        std::string name;               // general.name, cleaned
        long n_layers = 0;
        long n_heads = 0;
        long n_kv_heads = 0;
        long d_model = 0;
        long d_ffn = 0;
        long head_dim = 0;              // keys and queries
        long head_dim_v = 0;            // values
        long vocab = 0;
        long ctx_train = 0;             // context length the model was trained with
        double rms_eps = 1e-5;

        // Rotary embedding.
        rope_style rope = rope_style::norm;
        long rope_dims = 0;             // leading dimensions of each head that rotate
        double rope_base = 10000.0;
        double rope_freq_scale = 1.0;   // 1 / linear scaling factor
        std::vector<float> rope_freq_factors;   // llama 3 per-frequency divisors
        bool yarn = false;
        double yarn_orig_ctx = 0;
        double yarn_beta_fast = 32.0;
        double yarn_beta_slow = 1.0;
        double yarn_ext_factor = 1.0;
        double yarn_attn_factor = 1.0;  // magnitude correction applied to cos and sin

        // Long context query scaling: q *= 1 + attn_temp_scale * log(1 + floor(pos / attn_temp_length)).
        double attn_temp_scale = 0.0;
        double attn_temp_length = 8192.0;

        // Optional parts.
        bool qkv_bias = false;
        bool qk_norm = false;
        bool tied_embeddings = false;

        // Global scale factors (granite); neutral values otherwise.
        double embedding_scale = 1.0;
        double residual_scale = 1.0;
        double attention_scale = 0.0;   // 0: 1/sqrt(head_dim)
        double logit_scale = 1.0;       // logits are divided by it

        // Mixture of experts.
        long n_experts = 0;
        long n_experts_used = 0;
        long d_ffn_expert = 0;
        bool expert_weights_norm = true;   // renormalize the selected routing weights

        // Tokenizer and chat format, as declared.
        std::string chat_template;
        std::string tokenizer_pre;

        bool is_moe() const { return n_experts > 0; }
        long kv_dim() const { return n_kv_heads * head_dim; }
        long q_dim() const { return n_heads * head_dim; }
        long v_dim() const { return n_kv_heads * head_dim_v; }
    };

    // Reads the configuration of a model, or throws with the reason it cannot be served.
    model_config read_model_config(const model_file& f);

    // One line describing a configuration, for the logs.
    std::string describe(const model_config& c);

}

#endif // ALBIREO_ENGINE_MODEL_CONFIG_H_
