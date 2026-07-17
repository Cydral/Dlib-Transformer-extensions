// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DECODER_TRANSFORMER_CONFIG_H_
#define DLIB_DECODER_TRANSFORMER_CONFIG_H_

#include "decoder_transformer_config_abstract.h"

#include "transformer_config.h"
#include <string>
#include <sstream>

namespace dlib
{
    /* High-level configuration of a Llama-family decoder-only transformer (token
       embeddings, pre-norm GQA + SwiGLU blocks, final RMSNorm and bias-free output
       head). Serves both as the reconstruction target of the GGUF import and as a
       directly trainable structure. See decoder_transformer_config_abstract.h for
       the full contract, requirements and exported types. */
    template <
        long vocab_size,
        long num_layers,
        long num_heads,
        long num_kv_heads,
        long embedding_dim,
        long ffn_num,
        long ffn_den,
        long head_dim = embedding_dim / num_heads,
        bool use_qk_norm = false
    >
    struct decoder_transformer_config
    {
        static constexpr long VOCAB_SIZE    = vocab_size;
        static constexpr long NUM_LAYERS    = num_layers;
        static constexpr long NUM_HEADS     = num_heads;
        static constexpr long NUM_KV_HEADS  = num_kv_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long HEAD_DIM      = head_dim;
        static constexpr bool USE_QK_NORM   = use_qk_norm;
        static constexpr long FFN_NUM       = ffn_num;
        static constexpr long FFN_DEN       = ffn_den;
        static constexpr long FFN_HIDDEN    = embedding_dim * ffn_num / ffn_den;

        /* The decoder stack: num_layers pre-norm GQA + SwiGLU blocks over the token
           embeddings, with ACT disabled, the exact feed-forward ratio, and bias-free
           feed-forward projections (Llama carries no bias). */
        using subnet = gqa_transformer_unified::transformer_stack<
            NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>,
            /*UseAct=*/false, FFN_NUM, FFN_DEN, /*FFN linear=*/linear_no_bias,
            /*head_dim=*/HEAD_DIM, /*use_qk_norm=*/USE_QK_NORM>;

        /* The full network: final RMSNorm + bias-free output projection + per-token loss
           head. The is_training flag is accepted for API symmetry with the other
           configurations; the attention layer selects its training or inference behavior at
           run time, so the network type itself is identical in both cases. */
        template <bool is_training>
        using network_type = loss_cross_entropy_per_token<linear_no_bias<VOCAB_SIZE, rms_norm<subnet>>>;

        struct model_info
        {
            static std::string describe()
            {
                std::ostringstream o;
                o << "decoder_transformer_config\n"
                  << "  vocab        : " << VOCAB_SIZE << "\n"
                  << "  layers       : " << NUM_LAYERS << "\n"
                  << "  heads        : " << NUM_HEADS << " (kv " << NUM_KV_HEADS
                  << ", head_dim " << HEAD_DIM << ")\n"
                  << "  d_model      : " << EMBEDDING_DIM << "\n"
                  << "  qk_norm      : " << (USE_QK_NORM ? "on" : "off") << "\n"
                  << "  ffn_hidden   : " << FFN_HIDDEN
                  << " (= d_model * " << FFN_NUM << " / " << FFN_DEN << ")";
                return o.str();
            }
        };
    };
}

#endif // DLIB_DECODER_TRANSFORMER_CONFIG_H_
