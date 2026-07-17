// Copyright (C) 2026  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_TRANSFORMER_CONFIG_H_
#define DLIB_DNN_TRANSFORMER_CONFIG_H_

#include "transformer_config_abstract.h"

#include "input.h"
#include "layers.h"
#include "loss.h"

namespace dlib
{
    // Forward declarations
    struct training_mode_tag;
    struct inference_mode_tag;

    // ----------------------------------------------------------------------------------------

    // Classification head for next-token prediction
    template <long num_logits, typename SUBNET>
    using classification_head = loss_cross_entropy_per_token<linear<num_logits, rms_norm<SUBNET>>>;

    template <long num_logits, typename SUBNET>
    using fused_classification_head = loss_multiclass_log<fc<num_logits, rms_norm<SUBNET>>>;

    // ----------------------------------------------------------------------------------------

    /* High-level transformer model configurations: each struct fixes the model
       dimensions at compile time, validates them statically, and exposes a
       network_type<is_training> alias building the complete network. See
       transformer_config_abstract.h for the full contracts. */

    template<
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct fused_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;

        // Compile-time validation of model configuration
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        template <long vocab, template <typename> class ACT, typename SUBNET>
        using se = fc<vocab/4, ACT<bn_fc<fc<vocab/2, SUBNET>>>>;

        // Network definition selector based on training mode (with dropout for training, without for inference)
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            fused_classification_head<VOCAB_SIZE, se<VOCAB_SIZE, dropout_policy,
            fused_transformer::transformer_stack<NUM_LAYERS, activation_func, dropout_policy, EMBEDDING_DIM, NUM_HEADS,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>,
            fused_classification_head<VOCAB_SIZE, se<VOCAB_SIZE, dropout_policy,
            fused_transformer::transformer_stack<NUM_LAYERS, activation_func, multiply, EMBEDDING_DIM, NUM_HEADS,
            positional_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Transformer model configuration (fused version):\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM;
                return ss.str();
            }
        };
    };

    template<
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct canonical_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;

        // Compile-time validation of model configuration
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        // Network definition selector based on training mode (with dropout for training, without for inference)
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            canonical_transformer::transformer_stack<NUM_LAYERS, activation_func, dropout_policy, EMBEDDING_DIM, NUM_HEADS,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head<VOCAB_SIZE,
            canonical_transformer::transformer_stack<NUM_LAYERS, activation_func, multiply, EMBEDDING_DIM, NUM_HEADS,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Transformer model configuration:\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM;
                return ss.str();
            }
        };
    };

    // Selector for the attention implementation backing the transformer stack.
    //   - chained: legacy implementation built from a chain of ~16 elementary
    //     Dlib layers (linear, reshape, RoPE, repeat_heads, multm_prev, tril,
    //     softmax, etc.). This is the historical reference.
    //   - unified: fused gqa_attention_ layer that consolidates the same
    //     operations into a single Dlib layer with embedded RoPE/tril and a
    //     KV cache enabling prefill / incremental inference.
    enum class attention_impl
    {
        chained,
        unified
    };

    namespace impl
    {
        // Dispatcher: selects the appropriate transformer_stack alias based on
        // the requested attention_impl. Specialized per enum value below
        template <attention_impl Impl, long num_layers, long d_model,
            long num_heads, long num_kv_heads, typename SUBNET>
        struct gqa_stack_selector;

        template <long num_layers, long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        struct gqa_stack_selector<attention_impl::chained, num_layers, d_model, num_heads, num_kv_heads, SUBNET>
        {
            using type = gqa_transformer::transformer_stack<
                num_layers, d_model, num_heads, num_kv_heads, SUBNET>;
            static const char* name() { return "chained"; }
        };

        template <long num_layers, long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        struct gqa_stack_selector<attention_impl::unified, num_layers, d_model, num_heads, num_kv_heads, SUBNET>
        {
            using type = gqa_transformer_unified::transformer_stack<
                num_layers, d_model, num_heads, num_kv_heads, SUBNET>;
            static const char* name() { return "unified"; }
        };

        template <attention_impl Impl, long num_layers,
            long d_model, long num_heads, long num_kv_heads,
            template <typename> class ACT, template <typename> class DO,
            typename SUBNET, bool UseAct = false>
        struct hrm_stack_selector;

        template <long num_layers, long d_model, long num_heads, long num_kv_heads,
            template <typename> class ACT, template <typename> class DO, typename SUBNET, bool UseAct>
        struct hrm_stack_selector<attention_impl::chained, num_layers,
            d_model, num_heads, num_kv_heads, ACT, DO, SUBNET, UseAct>
        {
            using type = canonical_transformer::transformer_stack<
                num_layers, ACT, DO, d_model, num_heads, SUBNET> ;
            static const char* name() { return "canonical (multi-head)"; }
        };

        template <long num_layers, long d_model, long num_heads, long num_kv_heads,
            template <typename> class ACT, template <typename> class DO, typename SUBNET, bool UseAct>
        struct hrm_stack_selector<attention_impl::unified, num_layers,
            d_model, num_heads, num_kv_heads, ACT, DO, SUBNET, UseAct>
        {
            using type = gqa_transformer_unified::transformer_stack<
                num_layers, d_model, num_heads, num_kv_heads, SUBNET, UseAct>;
            static const char* name() { return "unified (gqa_attention_)"; }
        };
    }

    template<
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long num_kv_heads = 2,
        long embedding_dim = 228,
        attention_impl impl = attention_impl::chained
    >
    struct gqa_transformer_config
    {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long NUM_KV_HEADS = num_kv_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr attention_impl IMPL = impl;

        // Compile-time validation of model configuration
        struct validation
        {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(NUM_KV_HEADS > 0, "Number of KV heads must be positive");
            static_assert(NUM_HEADS% NUM_KV_HEADS == 0,
                "num_heads must be divisible by num_kv_heads");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0,
                "Embedding dimension must be divisible by number of heads");
            static_assert((EMBEDDING_DIM / NUM_HEADS) % 2 == 0,
                "head_dim (embedding_dim / num_heads) must be even for RoPE");
        };

        // Network definition. The selected attention implementation determines
        // whether the stack is built from chained Dlib primitives or from the
        // fused gqa_attention_ layer
        template <bool is_training>
        using network_type =
            classification_head<VOCAB_SIZE,
            typename impl::gqa_stack_selector<IMPL, NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>::type>;

        struct model_info
        {
            static std::string describe()
            {
                std::stringstream ss;
                ss << "Transformer model configuration:\n"
                    << "- vocabulary size  : " << VOCAB_SIZE << "\n"
                    << "- layers           : " << NUM_LAYERS << "\n"
                    << "- attention heads  : " << NUM_HEADS << "\n"
                    << "- KV heads (GQA)   : " << NUM_KV_HEADS << "\n"
                    << "- embedding dim    : " << EMBEDDING_DIM << "\n"
                    << "- head dim         : " << (EMBEDDING_DIM / NUM_HEADS) << "\n"
                    << "- attention impl   : " << impl::gqa_stack_selector<
                        IMPL, NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, void>::name();
                return ss.str();
            }
        };
    };

    template<
        /* Placeholder default sized for ARC-style symbol vocabularies; every real
           instantiation passes its vocabulary size explicitly. A library header must
           not depend on constants defined by example programs. */
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
        bool use_act = false        // HRM recurs at the module level; ACT off by default
    >
    struct hrm_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_H_LAYERS = num_h_layers;
        static constexpr long NUM_L_LAYERS = num_l_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long NUM_KV_HEADS = num_kv_heads;
        static constexpr bool USE_ACT = use_act;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long HRM_N = hrm_N;
        static constexpr long HRM_T = hrm_T;
        static constexpr attention_impl IMPL = impl;

        // Compile-time validation
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_H_LAYERS > 0, "Number of H layers must be positive");
            static_assert(NUM_L_LAYERS > 0, "Number of L layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0,
                "Embedding dimension must be divisible by number of heads");
            static_assert(HRM_N > 0, "HRM N cycles must be positive");
            static_assert(HRM_T > 0, "HRM T steps must be positive");
            // Extra checks active only when using unified attention
            static_assert(IMPL != attention_impl::unified || NUM_KV_HEADS > 0,
                "num_kv_heads must be positive when using unified attention");
            static_assert(IMPL != attention_impl::unified || NUM_HEADS % NUM_KV_HEADS == 0,
                "num_heads must be divisible by num_kv_heads when using unified attention");
            static_assert(IMPL != attention_impl::unified || (EMBEDDING_DIM / NUM_HEADS) % 2 == 0,
                "head_dim must be even (RoPE constraint) when using unified attention");
        };

        // H and L sub-network types dispatched by the attention implementation
        using train_h_net_type = typename impl::hrm_stack_selector<IMPL, NUM_H_LAYERS,
            EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, activation_func, dropout_policy, input_tensor, USE_ACT>::type;
        using train_l_net_type = typename impl::hrm_stack_selector<IMPL, NUM_L_LAYERS,
            EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, activation_func, dropout_policy, input_tensor, USE_ACT>::type;

        using infer_h_net_type = typename impl::hrm_stack_selector<IMPL, NUM_H_LAYERS,
            EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, activation_func, multiply, input_tensor, USE_ACT>::type;
        using infer_l_net_type = typename impl::hrm_stack_selector<IMPL, NUM_L_LAYERS,
            EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, activation_func, multiply, input_tensor, USE_ACT>::type;

        // Network definition selector based on training mode
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            hrm<train_h_net_type, train_l_net_type, HRM_N, HRM_T,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head<VOCAB_SIZE,
            hrm<infer_h_net_type, infer_l_net_type, HRM_N, HRM_T,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "HRM network configuration:\n"
                    << "- Vocabulary       : " << VOCAB_SIZE << " tokens\n"
                    << "- H module         : " << NUM_H_LAYERS << " transformer layers\n"
                    << "- L module         : " << NUM_L_LAYERS << " transformer layers\n"
                    << "- Attention heads  : " << NUM_HEADS << "\n";
                if (IMPL == attention_impl::unified) {
                    const long repeat_factor = NUM_HEADS / NUM_KV_HEADS;
                    ss << "- KV heads (GQA)   : " << NUM_KV_HEADS
                        << " (repeat: " << repeat_factor << "x)\n"
                        << "- Head dim         : " << (EMBEDDING_DIM / NUM_HEADS) << "\n";
                }
                ss << "- Embedding dim    : " << EMBEDDING_DIM << "\n"
                    << "- Attention impl   : " << impl::hrm_stack_selector<IMPL, 1,
                    EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS,
                    activation_func, dropout_policy, void>::name() << "\n"
                    << "- HRM cycles       : N=" << HRM_N << " (high-level), T=" << HRM_T << " (low-level)\n"
                    << "- Total reasoning  : " << (HRM_N * HRM_T) << " iterations";
                return ss.str();
            }
        };
    };

    template<
        long vocab_size_ = 2000,
        long num_layers_ = 4,
        long num_heads_ = 6,
        long num_kv_heads_ = 2,
        long embedding_dim_ = 228,
        long num_experts_ = 4,
        long top_k_ = 0, // 0 = auto (20% of experts, minimum 1)
        template <typename> class dropout_policy = dropout_10
    >
    struct gqa_moe_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size_;
        static constexpr long NUM_LAYERS = num_layers_;
        static constexpr long NUM_HEADS = num_heads_;
        static constexpr long NUM_KV_HEADS = num_kv_heads_;
        static constexpr long EMBEDDING_DIM = embedding_dim_;
        static constexpr long HEAD_DIM = embedding_dim_ / num_heads_;
        static constexpr long NUM_EXPERTS = num_experts_;
        static constexpr long TOP_K = top_k_;

        // Compile-time validation of model configuration
        static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
        static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
        static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
        static_assert(NUM_KV_HEADS > 0, "Number of KV heads must be positive");
        static_assert(NUM_HEADS% NUM_KV_HEADS == 0, "num_heads must be divisible by num_kv_heads");
        static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dim must be divisible by num_heads");
        static_assert(HEAD_DIM % 2 == 0, "head_dim (embedding_dim / num_heads) must be even for RoPE");
        static_assert(NUM_EXPERTS > 0, "Number of experts must be positive");

        // Gating network for MoE expert routing (managed internally by the moe_ layer)
        // Takes the full input tensor via avg_pool_everything to produce a
        // sequence-length-invariant representation, then projects to routing logits.
        using gate_net_type = fc<NUM_EXPERTS, leaky_relu<fc<NUM_EXPERTS * 8,
            avg_pool_everything<tag10<input_tensor>>>>>;

        // Expert network: SwiGLU with inner hidden dimension = d_model * 8/3
        // Output shape matches input — required by the MoE routing accumulation
        using expert_net_type = swiglu<EMBEDDING_DIM, 8, 3, input_tensor>;

        // Single GQA transformer block with MoE feed-forward sublayer.
        // The attention sublayer uses the fused gqa_attention_ layer, which embeds
        // RoPE on Q and K, GQA repeat, scaled dot-product attention, causal mask
        // and output projection in a single Dlib layer, with optional KV cache
        // for fast incremental inference.
        // The gate network is managed internally by moe_ (not in the main topology),
        // so the block is a clean residual: attention + residual, then MoE FFN + residual.
        template <typename MODE, typename SUBNET>
        using transformer_block =
            add_prev5<moe<expert_net_type, gate_net_type, NUM_EXPERTS, TOP_K, MODE, rms_norm<tag5<
            add_prev1<gqa_attention<EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            rms_norm<tag1<SUBNET>>>>>>>>;

        template<long remaining_layers, typename MODE, typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<MODE,
                typename transformer_stack_impl<remaining_layers - 1, MODE, SUBNET>::type>;
        };

        template<typename MODE, typename SUBNET>
        struct transformer_stack_impl<0, MODE, SUBNET, void>
        {
            using type = tag10<SUBNET>;
        };

        template<long num_layers, typename MODE, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, MODE, SUBNET>::type;

        // Network definition selector based on training mode
        template <bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            transformer_stack<NUM_LAYERS, training_mode_tag,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM,
            input<matrix<int, 0, 1>>>>>,
            classification_head<VOCAB_SIZE,
            transformer_stack<NUM_LAYERS, inference_mode_tag,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM,
            input<matrix<int, 0, 1>>>>>>;

        struct model_info {
            static std::string describe() {
                long repeat_factor = NUM_HEADS / NUM_KV_HEADS;
                long active_experts = (TOP_K == 0)
                    ? std::max(1L, static_cast<long>(std::floor(NUM_EXPERTS * 0.2f)))
                    : std::min(TOP_K, NUM_EXPERTS);
                std::stringstream ss;
                ss << "GQA-MoE transformer configuration:\n"
                    << "  vocabulary size   : " << VOCAB_SIZE << "\n"
                    << "  layers            : " << NUM_LAYERS << "\n"
                    << "  Q attention heads : " << NUM_HEADS << "\n"
                    << "  KV attention heads: " << NUM_KV_HEADS
                    << "  (GQA repeat: " << repeat_factor << "x)\n"
                    << "  embedding dim     : " << EMBEDDING_DIM << "\n"
                    << "  head dim          : " << HEAD_DIM << "\n"
                    << "  attention impl    : unified (gqa_attention_)\n"
                    << "  experts per layer : " << NUM_EXPERTS
                    << "  (active top-k: " << active_experts << ")\n";
                return ss.str();
            }
        };
    };
}

#endif // DLIB_DNN_TRANSFORMER_CONFIG_H_