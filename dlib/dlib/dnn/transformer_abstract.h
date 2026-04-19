// Copyright (C) 2026  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_TRANSFORMER_ABSTRACT_H_
#ifdef DLIB_DNN_TRANSFORMER_ABSTRACT_H_

#include "layers_abstract.h"

/*!
    The transformer.h file contains specialized layers and building blocks designed
    specifically for transformer architectures and attention mechanisms.

    Two architectural variants are provided:

    1. CANONICAL TRANSFORMER (namespace canonical_transformer):
       - Separate Q, K, V projections using linear_no_bias
       - Explicit reshape operations
       - More modular, easier to understand
       - Suitable for fine-grained control

    2. FUSED TRANSFORMER (namespace fused_transformer):
       - Combined QKV projection
       - Extraction-based separation
       - Optimized for performance and memory efficiency
!*/

namespace dlib
{
    // ----------------------------------------------------------------------------------------

    template <long repeat_factor_>
    class repeat_heads_
    {
        /*!
            REQUIREMENTS ON TEMPLATE PARAMETERS
                - repeat_factor_ >= 1

            WHAT THIS OBJECT REPRESENTS
                This layer replicates each channel (k dimension) of the input tensor
                repeat_factor_ times along the k axis. It is used in Grouped Query
                Attention (GQA) to broadcast Key/Value heads to match the number of
                Query heads before computing scaled dot-product attention.

                Given an input tensor of shape (batch, num_kv_heads, seq_len, head_dim),
                the output has shape (batch, num_kv_heads * repeat_factor, seq_len, head_dim)
                where each KV head is duplicated repeat_factor times consecutively.

                Forward mapping (per sample):
                    output(n, c * repeat_factor + r, :, :) = input(n, c, :, :)
                    for c in [0, num_kv_heads), r in [0, repeat_factor)

                Backward mapping (gradient accumulation, per sample):
                    grad_input(n, c, :, :) += sum_{r=0}^{repeat_factor-1}
                        grad_output(n, c * repeat_factor + r, :, :)

                When repeat_factor == 1 (standard multi-head attention without GQA),
                the layer reduces to a simple tensor copy with no overhead.
        !*/

    public:

        explicit repeat_heads_();
        /*!
            ensures
                - #repeat_factor == repeat_factor_ (from template parameter)
                - No trainable parameters (get_layer_params() returns empty tensor)
        !*/

        template <typename SUBNET>
        void setup(const SUBNET& sub);
        /*!
            ensures
                - No-op: this layer has no learnable parameters to initialize
        !*/

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output);
        /*!
            requires
                - sub.get_output() is a valid tensor with shape
                  (batch_size, num_kv_heads, seq_len, head_dim)
            ensures
                - #output.num_samples() == sub.get_output().num_samples()
                - #output.k() == sub.get_output().k() * repeat_factor
                - #output.nr() == sub.get_output().nr()
                - #output.nc() == sub.get_output().nc()
                - Each channel c of the input is replicated repeat_factor times
                  at consecutive positions c * repeat_factor .. c * repeat_factor + repeat_factor - 1
                - Implemented via tt::repeat_channels()
        !*/

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor& params_grad
        );
        /*!
            requires
                - gradient_input has shape (batch_size, num_kv_heads * repeat_factor,
                  seq_len, head_dim)
                - sub.get_gradient_input() has shape (batch_size, num_kv_heads,
                  seq_len, head_dim)
            ensures
                - Accumulates gradients from repeated head channels back into the
                  original KV head channels in sub.get_gradient_input()
                - For each destination channel c, sums the gradients from the
                  repeat_factor source channels c * repeat_factor .. c * repeat_factor + repeat_factor - 1
                - This is an accumulation (+=), not a replacement
                - Implemented via tt::accumulate_repeated_channels()
        !*/

        dpoint map_input_to_output(const dpoint& p) const;
        dpoint map_output_to_input(const dpoint& p) const;
        /*!
            ensures
                - Both return p unchanged (identity spatial mapping)
        !*/

        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            ensures
                - Returns an empty tensor (this layer has no trainable parameters)
        !*/

        friend void serialize(const repeat_heads_& item, std::ostream& out);
        /*!
            ensures
                - Serializes with version tag "repeat_heads_"
                - Saves: repeat_factor
        !*/

        friend void deserialize(repeat_heads_& item, std::istream& in);
        /*!
            ensures
                - Restores repeat_factor from serialized state
            throws
                - serialization_error if version tag does not match "repeat_heads_"
        !*/

        friend std::ostream& operator<<(
            std::ostream& out, const repeat_heads_& item);
        /*!
            ensures
                - Prints "repeat_heads (repeat_factor=N)"
        !*/

        friend void to_xml(const repeat_heads_& item, std::ostream& out);
        /*!
            ensures
                - Writes: <repeat_heads repeat_factor='N'/>
        !*/
    };

    template <long rep_fact, typename SUBNET>
    using repeat_heads = add_layer<repeat_heads_<rep_fact>, SUBNET>;

    // ---------------------------------------------------------------------------------------

    class network_context
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thread-safe singleton providing a shared execution context accessible
                by any layer during forward and backward passes.

                It solves the problem that deeply nested layers (e.g. a transition
                network inside an ACT layer) cannot directly access information about
                the enclosing network or the current training state. The caller sets
                the context once before each forward pass; any layer reads it as needed.

                The context has an explicit active/inactive state. It is considered
                inactive until at least one setter has been called. Any layer can
                query is_active() before reading values, allowing graceful fallback
                to default behavior when the context has not been initialized by the
                caller.

                The context distinguishes two operating modes:
                  - Training:   learning_rate > 0, set by the training loop
                  - Inference:  learning_rate == 0.0 (default)

            THREAD SAFETY
                All public methods are thread-safe through internal mutex protection.
                The mutex is non-recursive; methods that must operate under an
                already-held lock are provided as private _nolock_ variants.

            LIFECYCLE
                Inactive by default. Becomes active on first setter call.
                reset() returns it to the inactive state and clears all fields.

            TYPICAL USAGE
                // --- Training loop (before each train_one_step) ---
                network_context::set_learning_rate(trainer.get_learning_rate());
                network_context::set_padding(input_tensor, pad_token);

                // --- Inference (before net forward pass) ---
                network_context::set_learning_rate(0.0);
                network_context::set_padding_uniform(0, 1);
                // or simply do not set padding if no padding is present

                // --- Inside any layer's forward() ---
                if (network_context::is_active()) {
                    if (network_context::is_training()) { ... }
                    long pad = network_context::get_padding_length(sample_idx);
                    double lr  = network_context::get_learning_rate();
                }

                // --- Full teardown after pipeline completes ---
                network_context::reset();
        !*/

    public:

        // Lifecycle

        static bool is_active();
        /*!
            ensures
                - Returns true iff at least one setter has been called since the
                  last reset() (or since program start).
                - Layers should check this before reading any context value, and
                  fall back to their default behavior when false.
        !*/

        static void reset();
        /*!
            ensures
                - Returns the singleton to its initial inactive state.
                - #is_active()          == false
                - #is_training()        == false
                - #get_learning_rate()  == 0.0
                - #is_padding_set()     == false
        !*/

        // Learning rate / training mode

        static void set_learning_rate(double lr);
        /*!
            ensures
                - Stores the current learning rate.
                - #get_learning_rate() == lr
                - #is_training()       == (lr > 0.0)
                - #is_active()         == true
                - A value of 0.0 signals inference mode.
        !*/

        static double get_learning_rate();
        /*!
            ensures
                - Returns the stored learning rate, or 0.0 if never set.
        !*/

        static bool is_training();
        /*!
            ensures
                - Returns true iff get_learning_rate() > 0.0.
        !*/

        // Padding context

        static void set_padding(const tensor& input_tokens, long padding_token);
        /*!
            requires
                - input_tokens has shape (batch_size, k, seq_len, nc)
            ensures
                - Computes and stores per-sample leading-padding lengths by scanning
                  input_tokens for leading tokens equal to padding_token.
                - If padding_token < 0, clears padding state only (is_active unchanged).
                - #is_padding_set() == true  (unless padding_token < 0)
                - #is_active()      == true
        !*/

        static void set_padding_from_lengths(const std::vector<long>& lengths);
        /*!
            ensures
                - Stores the provided per-sample padding lengths directly.
                - #is_padding_set() == true
                - #is_active()      == true
        !*/

        static void set_padding_uniform(long padding_length, long batch_size);
        /*!
            ensures
                - Sets the same padding length for all samples.
                - #get_padding_length(i) == padding_length for i in [0, batch_size)
                - #is_padding_set() == true
                - #is_active()      == true
        !*/

        static void clear_padding();
        /*!
            ensures
                - Clears padding state only. is_active() and learning rate are unchanged.
                - #is_padding_set() == false
        !*/

        static long get_padding_length(long sample_idx);
        /*!
            ensures
                - Returns the padding length for sample_idx, or 0 if !is_padding_set()
                  or sample_idx is out of range.
        !*/

        static std::vector<long> get_all_padding_lengths();
        /*!
            ensures
                - Returns a copy of all stored padding lengths.
                - Returns an empty vector if !is_padding_set().
        !*/

        static bool is_padding_set();
        /*!
            ensures
                - Returns true iff padding lengths have been stored and not yet cleared.
        !*/
    };

    // ------------------------------------------------------------------------------------

    template <typename T>
    long count_leading_padding(const matrix<T, 0, 1>& seq, T padding_token);
    /*!
        ensures
            - Returns the number of leading elements in seq equal to padding_token.
            - Returns 0 if seq is empty or seq(0) != padding_token.
    !*/

    // ------------------------------------------------------------------------------------

    template <long d_k_>
    class scale_weights_ : public multiply_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This layer scales inputs by 1/sqrt(d_k), which is the standard scaling
                factor used in transformer attention mechanisms.

                This scaling prevents the dot products in attention from growing too large,
                which would push the softmax function into regions with small gradients.

                The scaling factor is: 1/sqrt(d_k) where d_k is the key/query dimension.

            TEMPLATE PARAMETERS
                - d_k: The dimension of keys/queries in the attention mechanism
        !*/
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    template <long num_embeddings, long embedding_length, typename SUBNET>
    using token_embeddings = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Converts discrete token IDs to continuous embedding vectors with positional
            encoding.

        ARCHITECTURE FLOW
            1. Token embedding lookup: maps token IDs to dense vectors
            2. Positional encoding: adds learnable position information

        TEMPLATE PARAMETERS
            - num_embeddings: vocabulary size (number of unique tokens)
            - embedding_length: embedding dimension (typically d_model)

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, 1) - matrix of token IDs (long integers)
            Output: (batch_size, 1, seq_len, embedding_length) - embedding vectors

        TYPICAL USAGE
            using my_model =
                loss_multiclass_log<fc<vocab_size,
                transformer_stack<6, gelu, dropout_10, seq_len, d_model, num_heads,
                token_embeddings<vocab_size, d_model,
                input<matrix<int, 0, 1>>>>>>;

        NOTES
            - Input tokens must be integers in range [0, num_embeddings)
            - embedding_length should match d_model for transformer architectures
    !*/

    namespace canonical_transformer
    {
        /*!
            WHAT THIS REPRESENTS
                Standard transformer implementation with separate Q, K, V projections.

                This architecture uses three independent linear transformations followed
                by reshape operations to create the multi-head attention structure.

                Advantages:
                - Conceptually clearer and more modular
                - Easier to debug and understand
                - Each projection can be independently modified or analyzed

                Use cases:
                - When fine-grained control over each projection is needed
                - Prototyping new attention mechanisms
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Query projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Key projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Value projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                This template implements a complete multi-head self-attention mechanism with
                causal masking, rotary positional embeddings (RoPE), and post-attention
                normalization.

                The attention mechanism computes:
                    Attention(Q, K, V) = softmax((Q*K^T) / sqrt(d_k)) * V

                Where Q, K, V are the Query, Key, and Value projections respectively.

            ARCHITECTURE FLOW
                1. RMS normalization
                2. Input is split into Query, Key, and Value projections
                3. RoPE is applied to Query and Key for positional encoding
                4. Scaled dot-product attention: Q*K^T / sqrt(d_head)
                5. Causal masking (tril_mask) prevents attending to future positions
                6. Softmax normalization across the sequence dimension
                7. Attention weights multiply Values: softmax(scores)*V
                8. Reshape and project back to d_model dimension
                9. Residual connection with input                

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., silu, gelu, relu)
                - DO: dropout policy template (e.g., dropout_10, multiply for inference)
                - seq_len: maximum sequence length (context window size)
                - d_model: model dimension (must be divisible by num_heads)
                - num_heads: number of parallel attention heads

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Uses causal masking (tril_mask) for autoregressive generation
                - RoPE is applied to both Query and Key for relative position encoding
                - The d_head per head is d_model / num_heads
                - Attention scores are scaled by 1/sqrt(d_head) for stability
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using std_ffn = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Standard position-wise feed-forward network used in transformer blocks.
                Implements a two-layer MLP with one intermediate activation and dropout
                regularization.

            ARCHITECTURE FLOW
                1. Linear expansion: d_model => 4*d_model
                2. Activation function (ACT)
                3. Linear projection: 4*d_model => d_model
                4. Dropout (DO) for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., gelu, silu, relu)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Expansion factor is fixed at 4x (standard transformer practice)
                - Single dropout applied after final projection
                - No normalization inside FFN (handled by transformer_block)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                SwiGLU (Swish-Gated Linear Unit) feed-forward network, an alternative to
                standard FFN with improved performance on language modeling tasks.

            REFERENCE
                Noam Shazeer, "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

            ARCHITECTURE FLOW
                1. Split into two branches from input:
                   - Gate branch: W1 projection => ACT activation
                   - Linear branch: V projection
                2. Element-wise multiplication of branches (Hadamard product)
                3. Final projection: W2 => d_model
                4. Dropout for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (typically silu for true SwiGLU)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Uses (8*d_model)/3 for hidden dimension (equivalent parameters to 4x expansion)
                - More expressive than standard FFN due to gating mechanism
                - Single dropout applied after final projection
                - ACT is typically silu (Swish) for standard SwiGLU
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                A complete transformer decoder block combining multi-head self-attention and
                feed-forward network with residual connections and RMS normalization.

            ARCHITECTURE FLOW
                Input => MultiHeadAttention => FFN => Output
                Each sub-layer uses the pattern: RMSNorm(input + SubLayer(input))

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., silu, gelu, relu)
                - DO: dropout policy template (e.g., dropout_10, multiply for inference)
                - seq_len: maximum sequence length (context window size)
                - d_model: model dimension (must be divisible by num_heads)
                - num_heads: number of parallel attention heads

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Decoder-only architecture with causal masking
                - Uses RMS normalization for improved training stability
                - Cannot be used directly with repeat<> due to multiple template parameters
                  (use transformer_stack<> instead for stacking multiple blocks)
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Stacks multiple transformer blocks using compile-time recursion.

            TEMPLATE PARAMETERS
                - num_layers: number of transformer blocks to stack (model depth)
                - ACT: activation function template
                - DO: dropout policy template
                - seq_len: maximum sequence length
                - d_model: model dimension
                - num_heads: number of attention heads

            TYPICAL USAGE
                Create a 6-layer transformer:

                using my_model =
					loss_multiclass_log<fc<vocab_size, rms_norm<
                    transformer_stack<6, silu, dropout_10, 512, 256, 8,
                    token_embeddings<vocab_size, 256,
                    input<matrix<int, 0, 1>>>>>>>;

            NOTES
                - Each layer has independent trainable parameters
                - Equivalent to manually nesting num_layers transformer_block definitions
        !*/

    } // namespace std_transformer

    namespace fused_transformer
    {

        /*!
            WHAT THIS REPRESENTS
                Optimized transformer implementation with fused QKV projections,
                sometimes referred to as "kernel-fused" attention in the literature

                This architecture uses a single fc_no_bias layer to compute all Q, K, V
                projections simultaneously (dimension: d_model => 3*d_model), then uses
                extract layers to separate them. This approach leverages Dlib's fc_ layer
                optimizations and reduces memory access patterns.

                Advantages:
                - Single matrix multiplication instead of three
                - Reduced memory bandwidth requirements
                - Better GPU utilization through larger operations

                Performance considerations:
                - Typically 10-30% faster than standard implementation
                - Lower memory footprint during forward/backward passes
                - Better cache utilization
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Query projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, d_model/num_heads, 1)
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Key projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, 1, d_model/num_heads)
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Value projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, d_model/num_heads, 1)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS OBJECT REPRESENTS
                Optimized multi-head self-attention using fused QKV projection.
                Functionally equivalent to canonical version but with better performance.

            ARCHITECTURE FLOW
                1. RMS normalization
                2. Single fused projection: d_model => 3*d_model for Q, K, V
                3. Extract Q, K, V from combined output
                4. Compute attention with causal masking
                5. Concatenate heads and project
                6. Residual connection and normalization

            TEMPLATE PARAMETERS
                - ACT: activation function (for compatibility)
                - DO: dropout policy
                - d_model: model dimension
                - num_heads: number of attention heads
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using std_ffn = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Fused implementation of standard feed-forward network using fc layers with
                automatic dimension flattening for better BLAS/GEMM utilization.

            ARCHITECTURE FLOW
                1. fc layer: d_model => 4*d_model (with dimension flattening)
                2. Activation function (ACT)
                3. fc layer: 4*d_model => d_model
                4. Dropout (DO)
                5. extract operation to restore proper tensor dimensions

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., gelu, silu, relu)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Fused implementation of SwiGLU using fc layers with automatic dimension
                flattening for better BLAS/GEMM utilization.

            REFERENCE
                Noam Shazeer, "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

            ARCHITECTURE FLOW
                1. fc projections with dimension flattening
                2. Split into gate and linear branches
                3. Element-wise multiplication
                4. Final fc projection with extraction
                5. Dropout for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (typically silu)
                - DO: dropout policy template
                - d_model: model dimension

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            Same interface as canonical_transformer::transformer_block but with
            optimized implementation using fused operations.
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            Same interface as canonical_transformer::transformer_stack but with
            optimized implementation using fused operations.
        !*/

    } // namespace fused_transformer

    // ----------------------------------------------------------------------------------------

    template<
        typename H_NET,
        typename L_NET,
        int N,
        int T
    >
    class hrm_
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - H_NET must be a valid dlib network type that can process tensors
                  through its forward() and back_propagate_error() methods.
                  Must use tag10<input_tensor> as base layer (not raw input_tensor).
                - L_NET must be a valid dlib network type with the same constraints.
                - N > 0 (number of high-level cycles)
                - T > 0 (number of low-level steps per high-level cycle)

            WHAT THIS OBJECT REPRESENTS
                This object implements a Hierarchical Reasoning Model (HRM) layer, a
                dual-recurrent architecture that separates computation into two temporal
                scales for multi-level sequence processing.

                The model consists of two interdependent recurrent modules managed as
                internal sub-networks with their own AdamW solvers:
                    - High-level module (H_NET): executes N slow cycles for abstract
                      planning and global reasoning
                    - Low-level module (L_NET): executes T fast iterations per H-cycle
                      for detailed, rapid computations

                During forward propagation, the network performs N x T total recurrent
                steps with hierarchical convergence. For each of the N high-level
                cycles, the low-level module performs T iterations, converging locally
                before the high-level module updates. All iterations except the final
                L-step and final H-step are executed without gradient tracking.

                Mathematical formulation:
                    z_H = z_h_init (broadcast to batch)
                    z_L = z_l_init (broadcast to batch)

                    For each high-level cycle n in [0, N):
                        For each low-level step t in [0, T):
                            z_L = f_L(z_L + z_H + x)
                        z_H = f_H(z_H + z_L)

                    Output = z_H^N

                where:
                    - x is the input tensor from the subnet
                    - z_H and z_L are hidden states initialized from learned vectors
                    - f_H and f_L are the recurrent transformations (H_NET and L_NET)
                    - The + operator denotes element-wise addition (via tt::add)

                The backward pass uses a one-step gradient approximation, computing
                gradients only through the final H-module and final L-module updates.
                This provides O(1) memory complexity instead of O(N x T) required by
                full Backpropagation Through Time (BPTT).

                Gradient path (1-step approximation):
                    dL/d(output) = gradient_input
                      -> backprop through H: h_net.back_propagate_error(last_h_input, ...)
                      -> backprop through L: l_net.back_propagate_error(last_l_input, ...)
                      -> accumulate to sub.get_gradient_input()

                Parameter management:
                    H_NET and L_NET are internal sub-networks invisible to the main
                    trainer's solver. Their parameters are updated via dedicated AdamW
                    solvers inside update_subnet_parameters(), called at the end of
                    backward(). The learning rate is synchronized from network_context
                    when active. The layer itself has no direct trainable parameters
                    (get_layer_params() returns an empty tensor); use
                    internal_parameters() to obtain the total parameter count of both
                    H_NET and L_NET combined.

                References:
                    - Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734
                    - Bai et al., "Deep Equilibrium Models", NeurIPS 2019
        !*/

    public:

        using h_net_type = H_NET;
        using l_net_type = L_NET;

        explicit hrm_();
        /*!
            ensures
                - #hidden_dim == 0
                - #learning_rate_multiplier == 1.0
                - #current_learning_rate_ == 0.001
                - #solvers_initialized_ == false
                - Internal networks (h_net, l_net) are default-constructed
        !*/

        hrm_(const hrm_& other);
        /*!
            ensures
                - Performs deep copy of h_net, l_net, z_h_init, z_l_init,
                  hidden_dim, learning_rate_multiplier, current_learning_rate_
                - Does NOT copy solvers (solvers_initialized_ = false)
                - Does NOT copy forward/backward computation tensors
        !*/

        hrm_& operator=(const hrm_& other);
        /*!
            ensures
                - Same deep copy semantics as copy constructor
        !*/

        template <typename SUBNET>
        void setup(const SUBNET& sub);
        /*!
            requires
                - sub.get_output() is a valid tensor
            ensures
                - #hidden_dim == sub.get_output().nc()
                - Initializes z_h_init and z_l_init as 1x1x1xhidden_dim tensors
                  with truncated normal distribution (std=1, truncated at +/-2)
                - Uses an atomic seed counter for reproducible but distinct
                  initialization across multiple hrm_ instances
        !*/

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output);
        /*!
            requires
                - setup(sub) has been called
                - sub.get_output() is a valid tensor with nc() == hidden_dim
            ensures
                - Initializes z_H and z_L by broadcasting z_h_init and z_l_init
                  across all samples, channels, and sequence positions
                - Executes N*T-1 recurrent iterations without gradient tracking:
                    * L-step: l_input = z_L + z_H + x; z_L = l_net.forward(l_input)
                    * H-step (at end of each cycle except last): h_input = z_H + z_L;
                      z_H = h_net.forward(h_input)
                - Executes final L-step and final H-step with gradient tracking:
                    * Saves last_l_input = z_L + z_H + x for backward
                    * z_L = l_net.forward(last_l_input)
                    * Saves last_h_input = z_H + z_L for backward
                    * z_H = h_net.forward(last_h_input)
                - #output has same dimensions as sub.get_output()
                - #output contains the final high-level state z_H^N
        !*/

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor& params_grad
        );
        /*!
            requires
                - forward(sub, output) was previously called
                - gradient_input has same dimensions as the forward() output
            ensures
                - Backpropagates through final H-module:
                    h_net.back_propagate_error(last_h_input, gradient_input)
                - Backpropagates through final L-module:
                    l_net.back_propagate_error(last_l_input, h_net.get_final_data_gradient())
                - Accumulates l_net.get_final_data_gradient() into
                  sub.get_gradient_input() via tt::add
                - If in training mode (network_context::is_training() or context inactive):
                    calls update_subnet_parameters() to update H and L network
                    parameters via their internal AdamW solvers
                - Memory complexity: O(1) instead of O(N*T) for full BPTT
        !*/

        void set_learning_rate(double lr);
        /*!
            ensures
                - #current_learning_rate_ == lr
                - Propagates lr to both h_net and l_net via set_all_learning_rates()
        !*/

        double get_learning_rate() const;
        /*!
            ensures
                - Returns the current learning rate used by internal solvers
        !*/

        void set_learning_rate_multiplier(double val);
        /*!
            ensures
                - #learning_rate_multiplier == val
                - Propagates val to both h_net and l_net via
                  set_all_learning_rate_multipliers()
                - If val == 0.0, update_subnet_parameters() becomes a no-op
        !*/

        double get_learning_rate_multiplier() const;
        /*!
            ensures
                - Returns the learning rate multiplier applied to effective_lr
        !*/

        void configure_solvers(double weight_decay, double beta1, double beta2);
        /*!
            ensures
                - Stores solver hyperparameters for next solver initialization
                - #solvers_initialized_ == false (forces re-creation of solvers
                  on next backward pass)
        !*/

        size_t internal_parameters() const;
        /*!
            ensures
                - Returns count_parameters(h_net) + count_parameters(l_net)
                - This is the total number of trainable parameters managed
                  internally by this layer
                - Note: get_layer_params().size() returns 0 because the hrm_
                  layer has no direct parameters; all parameters belong to the
                  internal H and L sub-networks
        !*/

        void clean();
        /*!
            ensures
                - Calls clean() on h_net and l_net if they support it
                - Prepares the networks for inference or serialization
        !*/

        const h_net_type& get_h_net() const;
        h_net_type& get_h_net();
        /*!
            ensures
                - Returns a reference to the high-level (H) network module
        !*/

        const l_net_type& get_l_net() const;
        l_net_type& get_l_net();
        /*!
            ensures
                - Returns a reference to the low-level (L) network module
        !*/

        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            ensures
                - Returns an empty tensor (size 0)
                - The hrm_ layer has no direct trainable parameters; all
                  parameters are contained within H_NET and L_NET and are
                  accessed via internal_parameters() or get_h_net()/get_l_net()
        !*/

        friend void serialize(const hrm_& item, std::ostream& out);
        /*!
            ensures
                - Serializes the complete state including:
                  h_net, l_net, z_h_init, z_l_init, hidden_dim,
                  learning_rate_multiplier, current_learning_rate_,
                  h_solvers_, l_solvers_, solvers_initialized_
                - Version tag: "hrm_"
        !*/

        friend void deserialize(hrm_& item, std::istream& in);
        /*!
            ensures
                - Restores all serialized state
            throws
                - serialization_error if version tag does not match "hrm_"
        !*/

        friend std::ostream& operator<<(std::ostream& out, const hrm_& item);
        /*!
            ensures
                - Prints "hrm (N=<N>, T=<T>) learning_rate_mult=<val>"
        !*/

        friend void to_xml(const hrm_& item, std::ostream& out);
        /*!
            ensures
                - Writes XML representation including N, T,
                  learning_rate_multiplier, and nested XML for h_net and l_net
        !*/
    };

    template<typename H_NET, typename L_NET, int N, int T, typename SUBNET>
    using hrm = add_layer<hrm_<H_NET, L_NET, N, T>, SUBNET>;

    // ----------------------------------------------------------------------------------------

// Tags for compile-time mode selection in MoE layers
    struct training_mode_tag {};
    struct inference_mode_tag {};

    template<
        typename EXPERT_NET,
        typename GATE_NET,
        long n_experts_,
        long top_e,
        typename MODE
    >
    class moe_
    {
        /*!
            REQUIREMENTS ON TEMPLATE PARAMETERS
                - EXPERT_NET must be a valid dlib network type that can process tensors
                  through its forward() and back_propagate_error() methods.
                  Must use tag10<input_tensor> as base layer (not raw input_tensor).
                - GATE_NET must be a valid dlib network type that produces a tensor
                  with shape (batch_size, n_experts_, 1, 1) or (batch_size, 1, 1, n_experts_)
                  containing raw routing logits. Must also use tag10<input_tensor> as
                  base layer.
                - n_experts_ > 0 (number of expert sub-networks)
                - top_e >= 0 (use 0 for automatic selection of 20% of experts, minimum 1)
                - MODE must be either training_mode_tag or inference_mode_tag

            WHAT THIS OBJECT REPRESENTS
                This layer implements a Mixture of Experts (MoE) architecture with
                per-sample dynamic routing, inspired by Switch Transformer and ST-MoE.
                Each input sample in a batch is independently routed to the top-k
                experts based on a learned internal gating function. This enables
                increased model capacity without proportional computational cost.

                Both the gate network and the expert networks are managed as internal
                sub-networks with their own AdamW solvers, following the same pattern
                as hrm_. They are invisible to the main trainer's solver; their
                parameters are updated inside backward() via update_subnet_parameters().
                The layer itself has no direct trainable parameters (get_layer_params()
                returns an empty tensor); use internal_parameters() for the total count
                or active_parameters() for the inference-time count (top-k experts + gate).

                IMPORTANT: Both GATE_NET and EXPERT_NET definitions must use
                tag10<input_tensor> as their base layer instead of raw input_tensor.
                This is because input<tensor> requires sample_expansion_factor
                initialization via to_tensor(), which is never called for internally
                managed sub-networks. Wrapping in tag10<> yields an add_tag_layer
                that bypasses this requirement.

                Example definitions:
                    using gate = fc<N, leaky_relu<fc<N*8,
                        avg_pool_everything<tag10<input_tensor>>>>>;
                    using expert = swiglu<d_model, 8, 3, tag10<input_tensor>>;

            ROUTING MECHANISM
                For each sample in the batch:
                1. The internal gate network produces raw logits from the input
                2. Logits are clamped to [-3, +3] to prevent softmax saturation
                3. Gaussian exploration noise is added (training only, noise_scale=0.5)
                   AFTER clamping, so noise can freely break ties
                4. Numerically stable softmax converts logits to probabilities
                5. Top-k experts are selected by descending probability
                6. A capacity factor limits per-expert load; if the preferred expert
                   is full, the sample overflows to the next-best available expert
                7. Selected expert weights are renormalized to sum to 1
                8. Expert outputs are combined: output = sum(w_i * expert_i(input))

            GATE GRADIENT
                The gate receives three gradient signals in backward():

                Part A - Task gradient (differentiable routing weight):
                    For each sample, computes score_e = dot(expert_output, task_gradient)
                    normalized by 1/sqrt(sample_size) and clamped to [-1, +1].
                    Converted to logit gradient via softmax Jacobian:
                        dL/dz_j = P_j * (score_j - sum_e(P_e * score_e))

                Part B - Load balance loss (Switch Transformer):
                    L_aux = alpha * N * sum_e(f_e * P_bar_e)
                    where f_e = routing fraction, P_bar_e = average gate probability.
                    Gradient through softmax Jacobian with per-batch normalization.

                Part C - Router z-loss (ST-MoE):
                    L_z = c_z * mean(logsumexp(logits)^2)
                    Penalizes large logits to prevent numerical instability.
                    Gradient: 2 * c_z * logsumexp * P_j / batch_size

                All gate gradients are clamped to [-1, +1] before back_propagate_error.
                Gate data gradient is NOT propagated to sub.get_gradient_input();
                only expert data gradients flow back to the main network.

            LOAD BALANCE WEIGHT DECAY
                The load_balance_weight decays exponentially each training step:
                    lb = max(lb_floor, lb * lb_decay)
                Starting at 0.1 (strong balancing to prevent expert collapse), it
                decays toward lb_floor=0.01 at rate lb_decay=0.9999, allowing
                progressive expert specialization as training proceeds.

            CAPACITY FACTOR
                Each expert can process at most:
                    capacity = ceil(batch_size * top_k * capacity_factor / n_experts)
                samples per forward pass. When an expert is full, overflow tokens are
                routed to the next-best expert with available capacity. Default
                capacity_factor=1.5 ensures all samples are processed while limiting
                load imbalance.
        !*/

    public:

        using expert_net_type = EXPERT_NET;
        using gate_net_type = GATE_NET;

        explicit moe_();
        /*!
            ensures
                - #n_experts == n_experts_ (from template parameter)
                - #noise_scale == 0.5
                - #capacity_factor == 1.5
                - #usage_update_rate == 0.05
                - #load_balance_weight == 0.1
                - #load_balance_floor_ == 0.01
                - #load_balance_decay_ == 0.9999
                - #z_loss_weight == 0.001
                - #learning_rate_multiplier == 1.0
                - #current_learning_rate_ == 1e-4
                - #solvers_initialized_ == false
                - #cached_batch_size_ == 0
                - If top_e == 0: #top_k == max(1, floor(n_experts * 0.2))
                - If top_e > 0: #top_k == min(top_e, n_experts)
                - Expert vector is empty (created during setup())
        !*/

        moe_(const moe_& other);
        /*!
            ensures
                - Performs deep copy of gate_net, all experts, expert_usage,
                  and all configuration fields
                - Does NOT copy solvers (solvers_initialized_ = false)
                - Does NOT copy forward/backward cache (cached_batch_size_ = 0)
        !*/

        moe_& operator=(const moe_& other);
        /*!
            ensures
                - Same deep copy semantics as copy constructor
                - Clears expert_solvers_ and gate_solvers_
        !*/

        template <typename SUBNET>
        void setup(const SUBNET& sub);
        /*!
            ensures
                - If experts vector is empty:
                    * Creates n_experts_ instances of EXPERT_NET
                    * Initializes expert_usage vector with zeros
                    * Sets solvers_initialized_ = false
                - Called automatically by Dlib during first forward pass
        !*/

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output);
        /*!
            requires
                - setup(sub) has been called
                - sub.get_output() is a valid tensor
            ensures
                - Caches input for expert and gate backward
                - Runs gate_net.forward() to obtain routing logits
                - For each sample in the batch:
                    * Clamps raw logits to [-3, +3]
                    * Adds Gaussian noise if MODE == training_mode_tag
                    * Applies numerically stable softmax
                    * Selects top-k experts with capacity overflow handling
                    * Renormalizes selected weights to sum to 1
                - Gathers samples per expert, runs batched expert forward
                - Scatters weighted expert outputs: output[n] += w * expert(x[n])
                - #output has same dimensions as sub.get_output()
                - If MODE == training_mode_tag:
                    * Caches assignments, gate probs, logsumexp, expert outputs
                    * Computes load_balance_loss_
                    * Updates expert_usage via EMA
        !*/

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub,
            tensor& params_grad);
        /*!
            requires
                - forward(sub, output) was previously called
                - gradient_input has same dimensions as the forward() output
            ensures
                - Phase 1: batched backward through each active expert
                    * Gathers gradient weighted by routing weight w
                    * Calls back_propagate_error on each expert's cached input
                    * Scatters expert data gradient to sub.get_gradient_input()
                    * Computes task scores: dot(expert_out, task_grad) / sqrt(dim)
                - Phase 2 (if training and learning_rate_multiplier > 0):
                    * Computes gate gradient from task + load-balance + z-loss
                    * Clamps gate gradient to [-1, +1]
                    * Calls gate_net.back_propagate_error (parameter update only,
                      gate data gradient NOT propagated to main network)
                - Phase 3 (if training):
                    * Calls update_subnet_parameters() to update active experts
                      and gate via internal AdamW solvers
                    * Decays load_balance_weight toward load_balance_floor_
        !*/

        void clean();
        /*!
            ensures
                - Calls clean() on each expert and on gate_net if supported
        !*/

        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            ensures
                - Returns an empty tensor (size 0)
                - The moe_ layer has no direct trainable parameters; all
                  parameters belong to the internal expert and gate sub-networks
                  and are accessed via internal_parameters() / active_parameters()
        !*/

        void set_learning_rate(double lr);
        /*!
            ensures
                - #current_learning_rate_ == lr
                - Propagates lr to all experts and gate_net via
                  set_all_learning_rates()
        !*/

        double get_learning_rate() const;
        /*!
            ensures
                - Returns the current learning rate used by internal solvers
        !*/

        void set_learning_rate_multiplier(double val);
        /*!
            ensures
                - #learning_rate_multiplier == val
                - Propagates val to all experts and gate_net via
                  set_all_learning_rate_multipliers()
                - If val == 0.0, update_subnet_parameters() becomes a no-op
        !*/

        double get_learning_rate_multiplier() const;
        /*!
            ensures
                - Returns the multiplier applied to effective learning rate
        !*/

        void configure_solvers(
            double weight_decay, double beta1, double beta2);
        /*!
            ensures
                - Stores solver hyperparameters for next initialization
                - #solvers_initialized_ == false
        !*/

        size_t internal_parameters() const;
        /*!
            ensures
                - Returns count_parameters(gate_net) +
                  count_parameters(experts[0]) * n_experts
                - This is the total training-time parameter footprint
                - Returns 0 if experts have not been created yet
        !*/

        size_t active_parameters() const;
        /*!
            ensures
                - Returns count_parameters(gate_net) +
                  count_parameters(experts[0]) * top_k
                - This is the inference-time parameter footprint
                  (only top-k experts are activated per sample)
                - Returns 0 if experts have not been created yet
        !*/

        EXPERT_NET& get_expert(size_t idx);
        const EXPERT_NET& get_expert(size_t idx) const;
        /*!
            requires
                - idx < num_experts()
            ensures
                - Returns reference to the expert network at index idx
        !*/

        GATE_NET& get_gate();
        const GATE_NET& get_gate() const;
        /*!
            ensures
                - Returns reference to the internal gate network
        !*/

        long num_experts() const;
        /*!
            ensures
                - Returns n_experts (the total number of expert sub-networks)
        !*/

        long num_active_experts() const;
        /*!
            ensures
                - Returns top_k (the number of experts activated per sample)
        !*/

        bool is_training_mode() const;
        /*!
            ensures
                - Returns true if MODE == training_mode_tag
        !*/

        const std::vector<float>& get_expert_usage() const;
        /*!
            ensures
                - Returns the EMA of expert usage fractions
                - Vector size == num_experts()
                - Ideal balanced value: 1.0 / num_experts() for each expert
                - Updated only in training mode when usage_update_rate > 0
        !*/

        float get_load_balance_loss() const;
        /*!
            ensures
                - Returns the auxiliary load balancing loss from the last
                  forward pass
                - Loss = load_balance_weight * n_experts * sum_e(f_e * P_bar_e)
                - Returns 0.0 if not in training mode
        !*/

        float get_load_balance_weight() const;
        /*!
            ensures
                - Returns the current load balance weight (decays during training)
        !*/

        friend void serialize(const moe_& item, std::ostream& out);
        /*!
            ensures
                - Serializes the complete state with version tag "moe_"
                - Saves: n_experts, top_k, noise_scale, capacity_factor,
                  usage_update_rate, load_balance_weight, load_balance_floor_,
                  load_balance_decay_, z_loss_weight, learning_rate_multiplier,
                  current_learning_rate_, gate_net, experts, expert_usage,
                  expert_solvers_, gate_solvers_, solvers_initialized_
        !*/

        friend void deserialize(moe_& item, std::istream& in);
        /*!
            ensures
                - Restores all serialized state
                - Clears forward/backward cache
            throws
                - serialization_error if version tag does not match "moe_"
        !*/

        friend std::ostream& operator<<(
            std::ostream& out, const moe_& item);
        /*!
            ensures
                - Prints: "moe (experts=N, top_k=K, mode=train/infer, noise=X,
                  cf=C, lb=W/F, z=Z) learning_rate_mult=M"
        !*/

        friend void to_xml(const moe_& item, std::ostream& out);
        /*!
            ensures
                - Writes XML representation including all configuration fields,
                  nested XML for gate_net and each expert, and expert_usage vector
        !*/
    };

    template<
        typename EXPERT_NET,
        typename GATE_NET,
        long n_experts_,
        long top_e,
        typename MODE,
        typename SUBNET
    >
    using moe = add_layer<
        moe_<EXPERT_NET, GATE_NET, n_experts_, top_e, MODE>, SUBNET>;
}

#endif // DLIB_DNN_TRANSFORMER_H_