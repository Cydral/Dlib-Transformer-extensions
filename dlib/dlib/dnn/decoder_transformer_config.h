// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DECODER_TRANSFORMER_CONFIG_H_
#define DLIB_DECODER_TRANSFORMER_CONFIG_H_

#include "decoder_transformer_config_abstract.h"
#include "modality_fusion.h"
#include "vision_transformer.h"

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
        bool use_qk_norm = false,
        template <typename> class MODALITY = no_modality
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
        /* MODALITY sits between the embeddings and the first block, which is the only
           place a modality other than text can enter: below it there are identifiers and
           no vectors, above it the stack no longer knows where a vector came from. The
           default, no_modality, is a plain alias, so a text-only model gains no layer at
           all, keeps its network type and reads its existing archives unchanged. */
        using subnet = gqa_transformer_unified::transformer_stack<
            NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS,
            MODALITY<embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>,
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

// ----------------------------------------------------------------------------------------

    /* A decoder and a vision tower as one parameterizable model.

       This exists so that the composition is named once, here, rather than assembled by
       hand wherever a multimodal model is declared. It also makes one class of mistake
       impossible to express: the projector width is not a parameter, it is embedding_dim,
       since a tower whose vectors do not have the width of the decoder they feed is not a
       model but two files that happen to sit in the same directory.

       The two halves remain usable apart. vision_transformer_config is a backbone in its
       own right, with its own pooling and heads for classification, metric learning or a
       self-supervised objective, and decoder_transformer_config left to its default policy
       is the text-only model it always was, with the same network type and the same
       archives. This struct is their composition, not their home. */
    template <
        // Decoder
        long vocab_size,
        long num_layers,
        long num_heads,
        long num_kv_heads,
        long embedding_dim,
        long ffn_num,
        long ffn_den,
        // Vision tower
        long image_size,
        long patch_size,
        long vision_width,
        long vision_layers,
        long vision_heads,
        long vision_ffn,
        long shuffle_factor,
        // Options
        long head_dim = embedding_dim / num_heads,
        bool use_qk_norm = false
    >
    struct multimodal_transformer_config
    {
        static constexpr long VOCAB_SIZE     = vocab_size;
        static constexpr long NUM_LAYERS     = num_layers;
        static constexpr long NUM_HEADS      = num_heads;
        static constexpr long NUM_KV_HEADS   = num_kv_heads;
        static constexpr long EMBEDDING_DIM  = embedding_dim;
        static constexpr long HEAD_DIM       = head_dim;
        static constexpr long FFN_HIDDEN     = embedding_dim * ffn_num / ffn_den;

        using vision_tower = vision_transformer_config<image_size, patch_size, vision_width,
            vision_layers, vision_heads, vision_ffn, shuffle_factor, embedding_dim>;

        // Positions a prompt must reserve for one image.
        static constexpr long VISUAL_TOKENS = vision_tower::NUM_TOKENS;

        /* The policy handed to the decoder: one layer above the embeddings, holding the
           tower and writing its vectors over the reserved positions. */
        template <typename SUBNET>
        using vision_modality = visual_fusion<typename vision_tower::network_type,
            VISUAL_TOKENS, EMBEDDING_DIM, SUBNET>;

        using text = decoder_transformer_config<vocab_size, num_layers, num_heads,
            num_kv_heads, embedding_dim, ffn_num, ffn_den, head_dim, use_qk_norm,
            vision_modality>;

        using subnet = typename text::subnet;

        template <bool is_training>
        using network_type = typename text::template network_type<is_training>;

        struct model_info
        {
            static std::string describe()
            {
                std::ostringstream o;
                o << "multimodal_transformer_config\n"
                  << "  decoder      : " << NUM_LAYERS << " layers, " << NUM_HEADS
                  << " heads, " << NUM_KV_HEADS << " kv heads, width " << EMBEDDING_DIM
                  << ", ffn " << FFN_HIDDEN << "\n"
                  << "  vocabulary   : " << VOCAB_SIZE << "\n"
                  << "  vision       : " << vision_layers << " layers, " << vision_heads
                  << " heads, width " << vision_width << ", ffn " << vision_ffn << "\n"
                  << "  image        : " << image_size << " (patches of " << patch_size
                  << "), " << VISUAL_TOKENS << " positions per image";
                return o.str();
            }
        };
    };

}

#endif // DLIB_DECODER_TRANSFORMER_CONFIG_H_
