// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_TRANSFORMER_CONFIG_ABSTRACT_H_
#ifdef DLIB_DNN_TRANSFORMER_CONFIG_ABSTRACT_H_

#include <string>
#include "transformer_abstract.h"

namespace dlib
{
    /*!
        OVERVIEW
            This header gathers the high-level model configurations of the
            transformer family: each one is a template struct that fixes the
            architectural dimensions at compile time, validates them through
            static assertions, and exposes a network_type<is_training> alias
            producing the complete Dlib network. The configurations differ by
            the transformer implementation they compose (canonical multi-head,
            fused, grouped-query, hierarchical, mixture of experts); the
            decoder-only configuration used by the GGUF import lives separately
            in decoder_transformer_config.h.

            Every configuration follows the same conventions:
              - constants exported in upper case (VOCAB_SIZE, NUM_LAYERS, ...),
              - a nested validation struct (or top-level static assertions)
                enforcing the parameter requirements at compile time,
              - a network_type<is_training> alias; when the two variants differ,
                the training one carries the dropout policy and the inference
                one replaces it with multiply (identity scaling),
              - a model_info::describe() static function returning a
                human-readable summary.
    !*/

    // ----------------------------------------------------------------------------------------

    template <long num_logits, typename SUBNET>
    using classification_head = loss_cross_entropy_per_token<linear<num_logits, rms_norm<SUBNET>>>;
    /*!
        ensures
            - The standard next-token prediction head: final RMSNorm, linear
              projection to num_logits, and per-token cross-entropy loss over
              every sequence position.
    !*/

    template <long num_logits, typename SUBNET>
    using fused_classification_head = loss_multiclass_log<fc<num_logits, rms_norm<SUBNET>>>;
    /*!
        ensures
            - Single-position classification head used by the fused
              configuration: final RMSNorm, fully-connected projection to
              num_logits, and multiclass log loss (one prediction per sample
              rather than per token).
    !*/

    // ----------------------------------------------------------------------------------------

    enum class attention_impl
    {
        chained,
        unified
    };
    /*!
        WHAT THIS ENUM REPRESENTS
            Selector for the attention implementation backing a transformer
            stack:
              - chained : historical reference implementation built from a
                          chain of elementary Dlib layers (linear, reshape,
                          RoPE, repeat_heads, multm_prev, tril, softmax, ...).
              - unified : fused gqa_attention_ layer consolidating the same
                          operations into a single Dlib layer, with embedded
                          RoPE and causal mask, and the KV cache enabling
                          prefill / incremental inference.
            Both produce the same forward computation; unified is required for
            cache-accelerated generation and grouped-query geometries beyond
            what the chained path expresses.
    !*/

    // ----------------------------------------------------------------------------------------

    template <
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct fused_transformer_config
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - vocab_size > 0, num_layers > 0, num_heads > 0
                - embedding_dim is a multiple of num_heads

            WHAT THIS OBJECT REPRESENTS
                Configuration of the fused single-position transformer: a
                fused_transformer::transformer_stack over positional
                embeddings, followed by a squeeze-and-excite style reduction
                (fc/bn_fc chain) and a fused_classification_head. Intended for
                sequence-level classification rather than per-token language
                modeling.

            EXPORTED CONSTANTS
                - VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM

            EXPORTED TYPES
                - network_type<is_training> : the full network; the training
                  variant uses dropout_policy, the inference variant replaces
                  it with multiply.

            MEMBERS
                - model_info::describe() : human-readable summary.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template <
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct canonical_transformer_config
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - vocab_size > 0, num_layers > 0, num_heads > 0
                - embedding_dim is a multiple of num_heads

            WHAT THIS OBJECT REPRESENTS
                Configuration of the canonical per-token language model: a
                canonical_transformer::transformer_stack (standard multi-head
                attention from separate Q/K/V projections) over token
                embeddings, closed by classification_head. This is the most
                modular reference configuration, suitable for prototyping and
                inspection.

            EXPORTED CONSTANTS
                - VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM

            EXPORTED TYPES
                - network_type<is_training> : training variant with
                  dropout_policy, inference variant with multiply.

            MEMBERS
                - model_info::describe() : human-readable summary.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template <
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long num_kv_heads = 2,
        long embedding_dim = 228,
        attention_impl impl = attention_impl::chained
    >
    struct gqa_transformer_config
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - vocab_size > 0, num_layers > 0
                - num_heads > 0, num_kv_heads > 0
                - num_heads is a multiple of num_kv_heads
                - embedding_dim is a multiple of num_heads
                - embedding_dim / num_heads is even (RoPE constraint)

            WHAT THIS OBJECT REPRESENTS
                Configuration of the grouped-query per-token language model.
                The impl parameter selects the attention implementation
                (chained reference or unified fused layer); both produce
                identical network semantics, and the unified path additionally
                supports KV-cached generation.

                The training and inference network types are identical (the
                fused attention selects its behavior at run time through
                network_context), which allows direct construction of an
                inference network from a trained one without a serialization
                round-trip.

            EXPORTED CONSTANTS
                - VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS,
                  EMBEDDING_DIM, IMPL

            EXPORTED TYPES
                - network_type<is_training> : the full network (identical for
                  both values of is_training).

            MEMBERS
                - model_info::describe() : human-readable summary including the
                  selected attention implementation.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template <
        long vocab_size = 16,
        long num_h_layers = 4,
        long num_l_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        long hrm_N = 4,
        long hrm_T = 4,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10,
        attention_impl impl = attention_impl::chained,
        long num_kv_heads = 2,
        bool use_act = false
    >
    struct hrm_transformer_config
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - vocab_size > 0, num_h_layers > 0, num_l_layers > 0
                - num_heads > 0 and embedding_dim is a multiple of num_heads
                - hrm_N > 0 and hrm_T > 0
                - When impl == attention_impl::unified:
                    * num_kv_heads > 0 and num_heads is a multiple of it,
                    * embedding_dim / num_heads is even (RoPE constraint).

            WHAT THIS OBJECT REPRESENTS
                Configuration of the Hierarchical Reasoning Model: two
                transformer sub-stacks (a high-level H module of num_h_layers
                and a low-level L module of num_l_layers) composed by the hrm
                layer, which iterates them for hrm_N high-level cycles of
                hrm_T low-level steps each (hrm_N * hrm_T reasoning iterations
                in total), over token embeddings and under a
                classification_head.

                The impl parameter selects the sub-stack implementation: the
                chained value maps to the canonical multi-head stack (the
                num_kv_heads and use_act parameters are then unused), the
                unified value maps to the fused GQA stack. use_act defaults to
                false because HRM already recurs at the module level; enabling
                ACT would nest two recurrence mechanisms.

            EXPORTED CONSTANTS
                - VOCAB_SIZE, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS,
                  NUM_KV_HEADS, USE_ACT, EMBEDDING_DIM, HRM_N, HRM_T, IMPL

            EXPORTED TYPES
                - train_h_net_type / train_l_net_type,
                  infer_h_net_type / infer_l_net_type : the H and L sub-stacks
                  with the training or inference dropout policy.
                - network_type<is_training> : the full HRM network.

            MEMBERS
                - model_info::describe() : human-readable summary including the
                  GQA geometry when the unified implementation is selected.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template <
        long vocab_size_ = 2000,
        long num_layers_ = 4,
        long num_heads_ = 6,
        long num_kv_heads_ = 2,
        long embedding_dim_ = 228,
        long num_experts_ = 4,
        long top_k_ = 0,
        template <typename> class dropout_policy = dropout_10
    >
    struct gqa_moe_transformer_config
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - vocab_size_ > 0, num_layers_ > 0
                - num_heads_ > 0, num_kv_heads_ > 0
                - num_heads_ is a multiple of num_kv_heads_
                - embedding_dim_ is a multiple of num_heads_
                - embedding_dim_ / num_heads_ is even (RoPE constraint)
                - num_experts_ > 0
                - top_k_ == 0 selects automatic routing width (20% of the
                  experts, at least 1); a positive value is clamped to
                  num_experts_.

            WHAT THIS OBJECT REPRESENTS
                Configuration of the grouped-query mixture-of-experts language
                model. Each block is a pre-norm residual pair: fused GQA
                attention, then a moe feed-forward sublayer routing across
                num_experts_ SwiGLU experts (hidden size d_model * 8/3) through
                an internally managed gate network (sequence-pooled projection
                to routing logits). The gate lives inside the moe_ layer, not
                in the main topology, so the block remains a clean residual
                structure.

                The training and inference network types differ only by the
                routing mode tag passed to the moe layer (training_mode_tag /
                inference_mode_tag), which selects dense-vs-sparse expert
                evaluation.

            EXPORTED CONSTANTS
                - VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS,
                  EMBEDDING_DIM, HEAD_DIM, NUM_EXPERTS, TOP_K

            EXPORTED TYPES
                - gate_net_type, expert_net_type : the internally managed gate
                  and expert sub-network types.
                - transformer_block<MODE, SUBNET>, transformer_stack<n, MODE,
                  SUBNET> : the block and stack builders parameterized by the
                  routing mode tag.
                - network_type<is_training> : the full network.

            MEMBERS
                - model_info::describe() : human-readable summary including the
                  active expert count.
        !*/
    };

    // ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_TRANSFORMER_CONFIG_ABSTRACT_H_
