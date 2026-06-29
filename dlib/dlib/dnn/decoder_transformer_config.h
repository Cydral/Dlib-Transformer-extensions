// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DECODER_TRANSFORMER_CONFIG_Hh_
#define DLIB_DECODER_TRANSFORMER_CONFIG_Hh_

#include "transformer_config.h"
#include <string>
#include <sstream>

namespace dlib
{
    /*!
        decoder_transformer_config

        WHAT THIS OBJECT REPRESENTS
            This is a high-level configuration describing a Llama-family decoder-only
            transformer, used as the reconstruction target when importing an open-weight
            model from the GGUF container (Llama 1/2/3, Mistral, Qwen-dense and similar).

            The architecture it expresses is fixed and matches that family exactly:
              - token embeddings,
              - num_layers pre-norm blocks, each one
                    RMSNorm -> RoPE grouped-query attention -> residual add
                    RMSNorm -> SwiGLU feed-forward            -> residual add,
              - a final RMSNorm and an output projection (the classification head).
            Adaptive computation time is disabled and there is no mixture of experts. The
            feed-forward and the output projection are bias-free (LINEAR_NO_BIAS), matching
            the Llama family, which carries no bias on any projection.

            The feed-forward hidden size is given as the exact rational ffn_num/ffn_den of
            the model dimension, so that arbitrary intermediate sizes (for example Llama-2-7B's
            11008 = 4096 * 43 / 16) are represented without rounding.

            It is a thin, named composition over the library's existing layers, chosen so the
            forward pass, the RoPE convention, the grouped-query expansion and the KV cache are
            exactly those already validated elsewhere in the library; only the weights differ.

        TEMPLATE PARAMETERS
            - vocab_size     : size of the token vocabulary.
            - num_layers     : number of transformer blocks.
            - num_heads      : number of query heads.
            - num_kv_heads   : number of key/value heads (num_heads for plain MHA).
            - embedding_dim  : model dimension d_model.
            - ffn_num/ffn_den: feed-forward hidden size = embedding_dim * ffn_num / ffn_den.

        REQUIREMENTS ON THE TEMPLATE PARAMETERS
            - all values are > 0,
            - num_heads is a multiple of num_kv_heads,
            - embedding_dim is a multiple of num_heads (head_dim = embedding_dim / num_heads),
            - embedding_dim * ffn_num is a multiple of ffn_den.
    !*/
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

#endif // DLIB_DECODER_TRANSFORMER_CONFIG_Hh_
