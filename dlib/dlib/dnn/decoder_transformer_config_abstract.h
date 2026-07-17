// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DECODER_TRANSFORMER_CONFIG_ABSTRACT_H_
#ifdef DLIB_DECODER_TRANSFORMER_CONFIG_ABSTRACT_H_

#include <string>
#include "transformer_abstract.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

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
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - All values are > 0.
                - num_heads is a multiple of num_kv_heads.
                - head_dim is even (RoPE constraint). It defaults to
                  embedding_dim / num_heads (requiring embedding_dim to be a
                  multiple of num_heads in that case) but may be given
                  explicitly for families that decouple it from the model
                  dimension (e.g. Qwen3: head_dim 128, d_model 1024, 16 heads).
                - embedding_dim * ffn_num is a multiple of ffn_den.

            WHAT THIS OBJECT REPRESENTS
                A high-level configuration describing a Llama-family decoder-only
                transformer, used both as the reconstruction target when importing
                an open-weight model from the GGUF container (Llama 1/2/3, Mistral,
                Qwen-dense and similar) and as a directly trainable structure.

                The architecture it expresses is fixed and matches that family
                exactly:
                  - token embeddings,
                  - num_layers pre-norm blocks, each one
                        RMSNorm -> RoPE grouped-query attention -> residual add
                        RMSNorm -> SwiGLU feed-forward          -> residual add,
                  - a final RMSNorm and a bias-free output projection (the
                    classification head) with a per-token cross-entropy loss.
                Adaptive computation time is disabled and there is no mixture of
                experts. The feed-forward and the output projection use
                linear_no_bias, matching the Llama family, which carries no bias
                on any of those projections. The attention layer's constant-layout
                QKV bias slots remain available for the families that need them
                (e.g. Qwen2); they are inert unless enabled at run time.

                The feed-forward hidden size is given as the exact rational
                ffn_num/ffn_den of the model dimension, so that arbitrary
                intermediate sizes (for example Llama-2-7B's
                11008 = 4096 * 43 / 16) are represented without rounding.

                It is a thin, named composition over the library's existing
                layers (gqa_transformer_unified::transformer_stack), chosen so
                the forward pass, the RoPE convention, the grouped-query
                expansion and the KV cache are exactly those already validated
                elsewhere in the library; only the weights differ.

                Two run-time settings complete the structure, because they are
                values rather than shapes: the RMSNorm epsilon and the rotary
                frequency base. The GGUF import path applies them automatically
                from the container metadata; standalone use applies them through
                the layer setters or the corresponding visitors before the first
                forward pass.

            EXPORTED CONSTANTS
                - VOCAB_SIZE    == vocab_size
                - NUM_LAYERS    == num_layers
                - NUM_HEADS     == num_heads
                - NUM_KV_HEADS  == num_kv_heads
                - EMBEDDING_DIM == embedding_dim
                - HEAD_DIM      == head_dim
                - USE_QK_NORM   == use_qk_norm
                - FFN_NUM       == ffn_num
                - FFN_DEN       == ffn_den
                - FFN_HIDDEN    == embedding_dim * ffn_num / ffn_den

            EXPORTED TYPES
                - subnet
                    The decoder stack over the token embeddings: an
                    input<matrix<int, 0, 1>> of token ids, the embedding table,
                    and NUM_LAYERS fused-attention pre-norm blocks.
                - network_type<is_training>
                    The full network: subnet, final RMSNorm, bias-free output
                    projection to VOCAB_SIZE, and per-token cross-entropy loss.
                    The is_training flag is accepted for API symmetry with the
                    other configurations of the library; the attention layer
                    selects its training or inference behavior at run time
                    through network_context, so the network type itself is
                    identical for both values.
        !*/

        struct model_info
        {
            static std::string describe();
            /*!
                ensures
                    - Returns a multi-line human-readable summary of the
                      configuration (vocabulary, layers, head geometry, model
                      dimension, QK-Norm state and feed-forward size).
            !*/
        };
    };

    // ----------------------------------------------------------------------------------------

}

#endif // DLIB_DECODER_TRANSFORMER_CONFIG_ABSTRACT_H_
