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
                granite (with its embedding, residual, attention and logit scales),
                qwen3vl and qwen3vlmoe (the decoders of Qwen3-VL: qwen3 and qwen3moe with
                multimodal rotary positions and DeepStack inputs, see rope_sections and
                n_deepstack), phi3 (Phi-3 and Phi-4-mini: one matrix for the queries, keys
                and values, one for the gate and up projections, rotary positions on the
                leading dimensions of the heads, LongRoPE; the sliding window the files
                declare is not applied, as in llama.cpp).

            EMBEDDING MODELS
                A decoder that declares <arch>.pooling_type computes vectors instead of
                text: its last step returns the normalized hidden state rather than the
                logits.  The causal decoders take the state of their last token, the one
                that has read the whole text (pooling type 3).
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

        /*
            LongRoPE (phi3): per-frequency divisors for a context within rope_orig_ctx
            positions and for a longer one.  rope_freq_factors receives the set that
            select_rope_factors() chooses for the context of a sequence.
        */
        std::vector<float> rope_factors_short, rope_factors_long;
        long rope_orig_ctx = 0;
        double rope_attn_factor = 1.0;  // <arch>.rope.scaling.attn_factor, applied to cos and sin

        // Fused projections (phi3): attn_qkv holds the rows of the queries, then those of the
        // keys and of the values; ffn_up holds the rows of the gate, then those of the up
        // projection.
        bool fused_qkv = false;
        bool fused_gate_up = false;

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

        // Pooling of an embedding model: 0 none (a language model), 1 mean, 2 first token,
        // 3 last token, 4 rank.
        int pooling = 0;

        /*
            Multimodal rotary positions (qwen3vl): a token has three position components,
            time, row and column, and rotary pair j of a head takes its angle from the row
            when j % 3 == 1 and j < 3 * rope_sections[1], from the column when j % 3 == 2
            and j < 3 * rope_sections[2], and from the time otherwise.  A text token has the
            same value in its three components, which gives the rotation of a model with
            one position per token.  Empty for the other models.
        */
        std::vector<int> rope_sections;

        // DeepStack (qwen3vl): an image token carries d_model * (1 + n_deepstack) input
        // floats; the first d_model replace its embedding, and slice k is added to its
        // hidden state after layer k.
        long n_deepstack = 0;

        bool is_moe() const { return n_experts > 0; }
        bool multimodal_rope() const { return !rope_sections.empty(); }
        long input_width() const { return d_model * (1 + n_deepstack); }
        bool is_embedding() const { return pooling != 0; }
        long kv_dim() const { return n_kv_heads * head_dim; }
        long q_dim() const { return n_heads * head_dim; }
        long v_dim() const { return n_kv_heads * head_dim_v; }
    };

    // Reads the configuration of a model, or throws with the reason it cannot be served.
    model_config read_model_config(const model_file& f);

    /*!
        ensures
            - When c holds LongRoPE divisors, makes c.rope_freq_factors the long ones if a
              sequence may hold more than c.rope_orig_ctx positions, and the short ones
              otherwise, as llama.cpp chooses them from the context of a sequence.
              read_model_config() chooses them for the context the model was trained with.
    !*/
    void select_rope_factors(model_config& c, long context);

    // One line describing a configuration, for the logs.
    std::string describe(const model_config& c);

}

#endif // ALBIREO_ENGINE_MODEL_CONFIG_H_
