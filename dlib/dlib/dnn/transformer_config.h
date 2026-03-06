// Copyright (C) 2026  Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_TRANSFORMER_CONFIG_H_
#define DLIB_DNN_TRANSFORMER_CONFIG_H_

#include "../dlib/data_io.h"
#include "layers.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

    // Classification head for next-token prediction
    template <long num_logits, typename SUBNET>
    using classification_head = loss_cross_entropy_per_logit<linear<num_logits, rms_norm<SUBNET>>>;

    // ----------------------------------------------------------------------------------------

    /**
     * @brief Transformer model configuration templates
     *
     * Provides a flexible and type-safe configuration mechanism for transformer models
     * with compile-time parameter validation and network generation.
     *
     */

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

    template<
        long vocab_size = 2000,
        long num_layers = 4,
        long num_heads = 6,
        long num_kv_heads = 2,
        long embedding_dim = 228
    >
    struct gqa_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long NUM_KV_HEADS = num_kv_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;

        // Compile-time validation of model configuration
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(NUM_KV_HEADS > 0, "Number of KV heads must be positive");
            static_assert(NUM_HEADS% NUM_KV_HEADS == 0, "num_heads must be divisible by num_kv_heads");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

		// Network definition for GQA transformer (no dropout for simplicity)
        using network_type =
            classification_head<VOCAB_SIZE,
            gqa_transformer::transformer_stack<NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Transformer model configuration:\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- KV heads (GQA): " << NUM_KV_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM;
                return ss.str();
            }
        };
    };

    template<
        long vocab_size = ARC_VOCAB_SIZE_TOTAL,
        long num_h_layers = 4,
        long num_l_layers = 4,
        long num_heads = 6,
        long embedding_dim = 228,
        long hrm_N = 4,
        long hrm_T = 4,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct hrm_transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_H_LAYERS = num_h_layers;
        static constexpr long NUM_L_LAYERS = num_l_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long HRM_N = hrm_N;
        static constexpr long HRM_T = hrm_T;

        // Compile-time validation of model configuration
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_H_LAYERS > 0, "Number of H layers must be positive");
            static_assert(NUM_L_LAYERS > 0, "Number of L layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
            static_assert(HRM_N > 0, "HRM N cycles must be positive");
            static_assert(HRM_T > 0, "HRM T steps must be positive");
        };

        // Network component definitions for training (with dropout)
        using train_h_net_type = canonical_transformer::transformer_stack<NUM_H_LAYERS, activation_func, dropout_policy,
            EMBEDDING_DIM, NUM_HEADS, input_tensor>;
        using train_l_net_type = canonical_transformer::transformer_stack<NUM_L_LAYERS, activation_func, dropout_policy,
            EMBEDDING_DIM, NUM_HEADS, input_tensor>;

        // Network component definitions for inference (without dropout)
        using infer_h_net_type = canonical_transformer::transformer_stack<NUM_H_LAYERS, activation_func, multiply,
            EMBEDDING_DIM, NUM_HEADS, input_tensor>;
        using infer_l_net_type = canonical_transformer::transformer_stack<NUM_L_LAYERS, activation_func, multiply,
            EMBEDDING_DIM, NUM_HEADS, input_tensor>;

		// Network definition selector based on training mode (with dropout for training, without for inference)
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            tag10<hrm<train_h_net_type, train_l_net_type, HRM_N, HRM_T,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>,
            classification_head<VOCAB_SIZE,
            tag10<hrm<infer_h_net_type, infer_l_net_type, HRM_N, HRM_T,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "HRM network configuration:\n"
                    << "- Vocabulary: " << VOCAB_SIZE << " tokens\n"
                    << "- H module: " << NUM_H_LAYERS << " transformer layers\n"
                    << "- L module: " << NUM_L_LAYERS << " transformer layers\n"
                    << "- Attention heads: " << NUM_HEADS << "\n"
                    << "- Embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- HRM cycles: N=" << HRM_N << " (high-level), T=" << HRM_T << " (low-level)\n"
                    << "- Total reasoning steps: " << (HRM_N * HRM_T) << " iterations";
                return ss.str();
            }
        };
    };

}

#endif // DLIB_DNN_TRANSFORMER_CONFIG_H_