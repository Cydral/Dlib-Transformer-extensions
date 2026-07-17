// Copyright (C) 2026  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_TRANSFORMER_ABSTRACT_H_
#ifdef DLIB_DNN_TRANSFORMER_ABSTRACT_H_

#include "layers_abstract.h"

/*!
    The transformer.h file contains specialized layers and building blocks designed
    for transformer architectures and attention mechanisms.

    Four architectural variants are provided:

    1. CANONICAL TRANSFORMER (namespace canonical_transformer):
       - Separate Q, K, V projections using linear_no_bias
       - Explicit reshape operations
       - Standard multi-head attention (no GQA)
       - More modular and easier to understand
       - Suitable for fine-grained control and prototyping

    2. FUSED TRANSFORMER (namespace fused_transformer):
       - Combined QKV projection through fc_no_bias
       - Extract-based separation
       - Optimized for performance and memory efficiency
       - Tensors carry the embedding dimension on k() (fc-style layout)

    3. GQA TRANSFORMER, CHAINED (namespace gqa_transformer):
       - Grouped Query Attention: fewer K/V heads than Q heads
       - Built from ~16 elementary Dlib layers (reshape, RoPE, repeat_heads,
         multm_prev, tril_mask, softmax, etc.)
       - Historical reference implementation
       - Re-aliased by `using namespace gqa_transformer;` so its symbols are
         available unqualified as the default GQA implementation

    4. GQA TRANSFORMER, UNIFIED (namespace gqa_transformer_unified):
       - Same topology as the chained version (same residual structure and
         RMSNorms), but the attention sublayer is the fused gqa_attention_
         layer (single Dlib layer with embedded RoPE, GQA repeat, scaled
         dot-product, causal mask, output projection, and KV cache)
       - Supports prefill / incremental inference modes via network_context
       - Drop-in replacement for gqa_transformer in any topology

    Auxiliary classes:
       - network_context  : thread-safe singleton sharing learning rate,
                            padding lengths, inference mode and KV cache
                            configuration across all layers.
       - hrm_             : Hierarchical Reasoning Model (Wang et al., 2025).
       - moe_             : Mixture of Experts with internal gate network
                            and per-sample top-k routing.
!*/

namespace dlib
{

    // ----------------------------------------------------------------------------------------

    /* network_context, the thread-safe singleton sharing the learning rate,
       padding lengths, inference mode, KV cache configuration and parameter
       residency across all layers, now lives in its own header; see
       network_context.h and network_context_abstract.h. */

    // ----------------------------------------------------------------------------------------

    template <typename T>
    long count_leading_padding(const matrix<T, 0, 1>& seq, T padding_token);
    /*!
        ensures
            - Returns the number of leading elements in seq equal to padding_token.
            - Returns 0 if seq is empty or seq(0) != padding_token.
    !*/

    // ----------------------------------------------------------------------------------------

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
    using positional_embeddings = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Converts discrete token IDs to continuous embedding vectors with
            additive positional encoding.

        ARCHITECTURE FLOW
            1. Token embedding lookup: maps token IDs to dense vectors of size
               embedding_length.
            2. Positional encoding: adds learnable position information.

        TEMPLATE PARAMETERS
            - num_embeddings:   vocabulary size (number of unique tokens).
            - embedding_length: embedding dimension (typically d_model).

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, 1)              - token IDs as longs.
            Output: (batch_size, 1, seq_len, embedding_length) - embedding vectors.

        TYPICAL USAGE
            using my_model =
                classification_head<vocab_size,
                transformer_stack<6, 256, 8, 2,
                positional_embeddings<vocab_size, 256,
                input<matrix<int, 0, 1>>>>>;

        NOTES
            - Token IDs must be integers in range [0, num_embeddings).
            - embedding_length should match d_model in the rest of the network.
    !*/

    // ----------------------------------------------------------------------------------------

    class param_offload_store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Host-side residency store for a layer's parameter tensor, used by the
                inference-only "host resident" mode selected through
                network_context::set_parameter_residency(). On first use under that
                mode, capture() moves the parameter tensor to a host float buffer and
                releases the resident tensor; every subsequent forward calls
                materialize(), which copies the weights into a slot of the shared
                rotating buffer pool owned by network_context.

                The returned buffer stays valid until the pool hands the same slot
                out again, i.e. across one full layer forward when every layer
                materializes at most once per pass. Training never engages this
                mechanism: the optimizer always reads the resident tensor.
        !*/
    public:

        bool active() const;
        /*!
            ensures
                - Returns true if and only if a parameter tensor has been captured
                  to host memory.
        !*/

        void capture(resizable_tensor& params);
        /*!
            ensures
                - Copies the contents and geometry of params to host memory.
                - #params.size() == 0 (the resident tensor is released).
                - #active() == true
        !*/

        const tensor& materialize() const;
        /*!
            requires
                - active() == true
            ensures
                - Returns a tensor with the captured geometry and weights, backed by
                  a slot of the shared buffer pool. The reference is only guaranteed
                  valid until the pool recycles the slot; bind it once per forward.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template <long EMBEDDING_DIM_, long NUM_HEADS_, long NUM_KV_HEADS_, long HEAD_DIM_,
        bool USE_QK_NORM_ = false>
    class gqa_attention_
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - EMBEDDING_DIM_  > 0
                - NUM_HEADS_      > 0
                - NUM_KV_HEADS_   > 0 and NUM_HEADS_ % NUM_KV_HEADS_ == 0
                - HEAD_DIM_       > 0 and even (RoPE rotates pairs of components).
                                  Typically EMBEDDING_DIM_ / NUM_HEADS_, but it may
                                  be decoupled from the model dimension (e.g. Qwen3
                                  uses head_dim 128 with d_model 1024 and 16 heads);
                                  the query/output projection width is then
                                  NUM_HEADS_ * HEAD_DIM_ instead of EMBEDDING_DIM_.
                - USE_QK_NORM_    when true, a per-head RMSNorm (QK-Norm) is applied
                                  to Q and K after the head split and before RoPE,
                                  each scaled by its own learned gamma of size
                                  HEAD_DIM_. Default false adds no parameter and
                                  changes no computation.

            WHAT THIS OBJECT REPRESENTS
                Fused Grouped-Query Attention layer with built-in RoPE, causal
                mask, and KV cache. Consolidates into a single Dlib layer what
                the gqa_transformer chained namespace builds from ~16 separate
                elementary layers.

                The layer performs, in order:
                  1. Linear projections W_Q, W_K, W_V (with Q pre-scaled by
                     1/sqrt(HEAD_DIM)), plus the b_q/b_k/b_v biases when the
                     QKV-bias mode is enabled (see set_qkv_bias_enabled()).
                  2. Transposition into the canonical multi-head 4D layout
                     [B, NUM_HEADS,    N, HEAD_DIM] for Q
                     [B, NUM_KV_HEADS, N, HEAD_DIM] for K and V.
                  3. When USE_QK_NORM is true, per-head RMSNorm of Q and K,
                     each scaled by its learned gamma (gamma_q, gamma_k).
                  4. Rotary Positional Embedding on Q and K.
                  5. GQA repeat of K/V from NUM_KV_HEADS to NUM_HEADS.
                  6. Scaled dot-product scores = Q @ K^T (Q already pre-scaled).
                  7. Causal mask via embedded tril_<0, neg_infinity_tag>.
                  8. Softmax over the last dimension.
                  9. ctx = attn @ V.
                 10. Merge heads back to [B, 1, N, Q_PROJ_DIM].
                 11. Final linear projection W_O back to EMBEDDING_DIM.

                Three inference modes are dispatched at runtime through
                network_context::get_inference_mode() (read at the start of
                each forward call):

                  - training / full : standard forward (full path), no cache
                    interaction. backward() is well-defined only after a
                    forward in one of these two modes.

                  - prefill : same compute as full, plus the side-effect of
                    populating K_cache and V_cache with the K and V of the
                    effective (non-padded) prompt positions. The cache stores
                    K *before* RoPE is applied; RoPE is reapplied at every
                    subsequent attention computation. This makes sliding-window
                    shifts trivially consistent and keeps RoPE positions inside
                    [0, max_seq_len_train).

                  - incremental : single-token forward (x.nr() == 1).
                    Projects the new Q/K/V; if the cache is full, evicts a
                    block of the oldest evictable positions (preserving the
                    first network_context::get_kv_cache_keep_length() sink
                    positions and dropping half of the remaining range) to
                    free room; appends the new K_pre_rope and V to the
                    cache; reads back the K/V window; applies RoPE to the
                    K window with positions [0, L-1] and to Q with position
                    L-1; performs the GQA repeat, attention and W_O
                    projection. Currently restricted to batch size 1.

                Numerical equivalence: in incremental mode, the output at
                output[0,0,0,:] matches (within floating-point rounding) what
                forward_full would produce at output[0,0,N-1,:] when invoked
                with the equivalent length-N prompt, provided positions are
                consistent.

                The parameter tensor stores, contiguously in a single
                resizable_tensor with alias_tensor views over each part:
                    [ W_Q | W_K | W_V | W_O (| gamma_q | gamma_k) | b_q | b_k | b_v ]
                W_Q maps EMBEDDING_DIM to Q_PROJ_DIM, W_K and W_V map
                EMBEDDING_DIM to KV_PROJ_DIM, and W_O maps Q_PROJ_DIM back to
                EMBEDDING_DIM. The gamma slots exist only when USE_QK_NORM is
                true. The bias slots are always present (constant layout across
                families) and stay zero-initialized and inert unless the
                QKV-bias mode is enabled. When host-resident inference is
                selected via network_context::set_parameter_residency(), the
                tensor is captured by an internal param_offload_store and
                materialized just in time at each forward. The KV cache state
                is run-time only and is NOT serialized.

            EXPORTED CONSTANTS
                - EMBEDDING_DIM   == EMBEDDING_DIM_
                - NUM_HEADS       == NUM_HEADS_
                - NUM_KV_HEADS    == NUM_KV_HEADS_
                - HEAD_DIM        == HEAD_DIM_
                - USE_QK_NORM     == USE_QK_NORM_
                - REPEAT_FACTOR   == NUM_HEADS / NUM_KV_HEADS
                - Q_PROJ_DIM      == NUM_HEADS * HEAD_DIM
                - KV_PROJ_DIM     == NUM_KV_HEADS * HEAD_DIM
        !*/

    public:

        gqa_attention_();
        /*!
            ensures
                - #get_learning_rate_multiplier() == 1
                - #get_weight_decay_multiplier()  == 1
                - The KV cache is empty (filled_len == 0, capacity == 0).
        !*/

        double get_learning_rate_multiplier() const;
        double get_weight_decay_multiplier() const;
        void set_learning_rate_multiplier(double v);
        void set_weight_decay_multiplier(double v);

        void set_yarn_params(float alpha, float beta,
            long original_len = 0, bool enabled = true);
        /*!
            ensures
                - Forwards the YaRN parameters to the two embedded RoPE helpers
                  (one for Q, one for K). See rotary_positional_embedding_
                  for the meaning of alpha/beta/original_len/enabled.
        !*/

        const yarn_config& get_yarn_config() const;
        /*!
            ensures
                - Returns the YaRN configuration currently held by the embedded
                  RoPE helpers (Q and K share the same configuration).
        !*/

        void set_rope_theta_base(float base);
        float get_rope_theta_base() const;
        /*!
            ensures
                - Sets / returns the rotary frequency base (theta) shared by the
                  two embedded RoPE helpers.
                - Must be set before the first forward pass touching a given
                  sequence length: the trigonometric caches are built on first
                  use for that length and are not rebuilt on a later base change.
        !*/

        float get_rope_cos_probe(long pos, long dim);
        /*!
            requires
                - A forward pass (or RoPE probe) has already populated the cosine
                  cache, and pos/dim index inside it.
            ensures
                - Returns the cached cos value at position pos, component dim, of
                  the Q RoPE helper. Diagnostic accessor used to verify which
                  rotary state a network is actually running with.
        !*/

        void set_qk_norm_eps(double eps);
        double get_qk_norm_eps() const;
        /*!
            ensures
                - Sets / returns the epsilon used by the per-head QK RMSNorm.
                - Only meaningful when USE_QK_NORM is true; ignored otherwise.
        !*/

        void set_qkv_bias_enabled(bool enabled);
        bool qkv_bias_enabled() const;
        /*!
            ensures
                - Enables / disables the addition of the b_q/b_k/b_v bias slots
                  after the Q/K/V projections (used by families such as Qwen2
                  that carry QKV projection biases).
                - The slots exist in the parameter layout either way; disabled,
                  they stay zero and cost nothing numerically. The flag is
                  serialized with the layer.
        !*/

        void reset_kv_cache();
        /*!
            ensures
                - Marks the KV cache as logically empty (cache_filled_len_ == 0).
                  The underlying storage is preserved and will be reused on the
                  next prefill / incremental forward.
        !*/

        long get_kv_cache_filled_len() const;
        /*!
            ensures
                - Returns the number of valid positions currently held in the
                  KV cache (0 if the cache has never been written or has been
                  reset).
        !*/

        long get_kv_cache_capacity() const;
        /*!
            ensures
                - Returns the allocated capacity of the KV cache. 0 if the cache
                  has not been allocated yet.
        !*/

        template <typename SUBNET>
        void setup(const SUBNET& sub);
        /*!
            requires
                - sub.get_output().k()  == 1
                - sub.get_output().nc() == EMBEDDING_DIM
            ensures
                - Allocates the packed parameter tensor and randomly initializes
                  the four weight matrices W_Q, W_K, W_V, W_O (each with Glorot
                  scaling derived from its own row+column count); when
                  USE_QK_NORM is true the gamma_q/gamma_k slots are initialized
                  to 1; the b_q/b_k/b_v bias slots are zeroed (inert until
                  enabled).
                - Probes the embedded RoPE helpers with q_probe and k_probe of
                  the input's sequence length so their trigonometric caches are
                  pre-allocated.
        !*/

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output);
        /*!
            requires
                - sub.get_output().k()  == 1
                - sub.get_output().nc() == EMBEDDING_DIM
            ensures
                - Reads the active inference mode from network_context and
                  observes the cache-clear flag (resetting the cache if set).
                - If the active mode is incremental and the input is a single
                  position with a non-empty cache, runs the incremental path.
                  Otherwise runs the full path; if the active mode is prefill,
                  the full path additionally writes K_pre_rope and V to the
                  cache for the effective (non-padded) positions.
                - #output is the attention output with the same outer shape
                  as the input ([B, 1, N, EMBEDDING_DIM]); in incremental mode
                  it is [1, 1, 1, EMBEDDING_DIM].
        !*/

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        /*!
            requires
                - The most recent forward() call ran the full path
                  (training, full, or prefill mode), not the incremental path.
                - gradient_input has the same shape as that forward()'s output.
            ensures
                - Accumulates the gradient with respect to the layer parameters
                  into params_grad (in the same packed layout as get_layer_params()).
                - Accumulates the gradient with respect to the input into
                  sub.get_gradient_input().
        !*/

        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            ensures
                - Returns the packed parameter tensor, laid out as
                  [W_Q | W_K | W_V | W_O (| gamma_q | gamma_k) | b_q | b_k | b_v].
                  Use the alias views internally maintained by the layer for
                  partial access.
        !*/

        friend void serialize(const gqa_attention_& item, std::ostream& out);
        friend void deserialize(gqa_attention_& item, std::istream& in);
        /*!
            ensures
                - Serializes / deserializes the parameter tensor, the learning
                  rate and weight decay multipliers, the QKV-bias flag, and the
                  embedded RoPE and tril helpers. When USE_QK_NORM is true the
                  QK-Norm epsilon is also part of the stream (and only then, so
                  every non-QK-Norm model keeps the exact same serialized
                  format as before the feature existed).
                - Does NOT serialize the KV cache state; on deserialize the
                  cache is reset (empty, no allocation).
                - Version tag: "gqa_attention_".
            throws
                - serialization_error if the version tag does not match.
        !*/

        friend std::ostream& operator<<(std::ostream& out, const gqa_attention_& item);
        friend void to_xml(const gqa_attention_& item, std::ostream& out);
    };

    template <long EMBEDDING_DIM, long NUM_HEADS, long NUM_KV_HEADS, long HEAD_DIM, typename SUBNET,
        bool USE_QK_NORM = false>
    using gqa_attention = add_layer<
        gqa_attention_<EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, USE_QK_NORM>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    namespace gqa_transformer
    {
        /*!
            WHAT THIS NAMESPACE REPRESENTS
                Grouped-Query Attention (GQA) transformer assembled from a chain
                of elementary Dlib layers. This is the historical reference
                implementation: K and V projections produce NUM_KV_HEADS heads
                (with NUM_KV_HEADS <= NUM_HEADS), then repeat_heads expands them
                to NUM_HEADS so the attention computation can proceed as in a
                standard multi-head attention.

                RoPE is applied on Q and K after their projections, scaled
                dot-product attention is performed with a causal mask, and the
                concatenated heads are projected back to d_model.

                The transformer_block places the attention sublayer and an ACT
                steps sublayer (with SwiGLU transition) inside two pre-norm
                residual paths (rms_norm on each input).

            NOTE: this namespace is re-aliased at the end of transformer.h via
            `using namespace gqa_transformer;` so that its symbols are also
            available unqualified as the default GQA implementation.
        !*/

        template <long d_model, long num_heads, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Linear_no_bias projection to d_model followed by reshape into
                  (num_heads, -1, d_model / num_heads).
        !*/

        template <long num_kv_heads, long head_dim, typename SUBNET>
        using key = some_template_expression;
        /*!
            ensures
                - Linear_no_bias projection to num_kv_heads * head_dim, then
                  reshape into (num_kv_heads, -1, head_dim).
        !*/

        template <long num_kv_heads, long head_dim, typename SUBNET>
        using value = some_template_expression;
        /*!
            ensures
                - Linear_no_bias projection to num_kv_heads * head_dim, then
                  reshape into (num_kv_heads, -1, head_dim).
        !*/

        template <long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        using multihead_attention_gqa = some_template_expression;
        /*!
            requires
                - num_heads    % num_kv_heads == 0
                - d_model      % num_heads    == 0
            ensures
                - Builds the full GQA attention sub-graph as a chain of Dlib
                  layers: Q/K/V projections (with K and V using only num_kv_heads
                  heads), RoPE on Q and K, repeat_heads to match the Q head
                  count, scale_weights, multm_prev for the scores, tril_mask,
                  softmaxm, multm_prev for the context, reshape, and final
                  linear_no_bias projection to d_model.
        !*/

        template <long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            ensures
                - Pre-norm transformer block: rms_norm -> attention -> residual,
                  then rms_norm -> act_steps<swiglu, 4> -> residual.
        !*/

        template<long num_layers, long d_model, long num_heads, long num_kv_heads,
            typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            ensures
                - Stacks num_layers transformer_block units, with a tag10<>
                  sentinel at the bottom of the stack for external addressing.
        !*/

    } // namespace gqa_transformer

    // ----------------------------------------------------------------------------------------

    namespace gqa_transformer_unified
    {
        /*!
            WHAT THIS NAMESPACE REPRESENTS
                Same transformer topology as gqa_transformer, but the attention
                sublayer is the single fused gqa_attention_ layer rather than a
                chain of ~16 elementary layers. The residual connections, the
                RMSNorms and the act_steps<swiglu> feed-forward are unchanged.

                Beyond the structural compaction, the fused attention also
                supports prefill / incremental inference modes with a built-in
                KV cache: a generation loop can call set_inference_mode(prefill)
                once on the prompt, then set_inference_mode(incremental) for
                each new token, with the layer handling cache lookup, RoPE
                application, GQA repeat and sliding window internally.

                External code that referenced gqa_transformer::transformer_stack
                or its tag10<> sentinel can switch to this namespace without
                further changes to the surrounding topology.
        !*/

        template<long d_model, long num_heads, long num_kv_heads, typename SUBNET,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using multihead_attention_gqa = some_template_expression;
        /*!
            ensures
                - Alias for gqa_attention<d_model, num_heads, num_kv_heads,
                  head_dim, SUBNET, use_qk_norm>. The trailing defaults keep
                  every pre-existing instantiation unchanged.
        !*/

        template<bool UseAct, long d_model, long hidden_num, long hidden_den,
            long max_steps, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear>
        using ffn_sublayer = some_template_expression;
        /*!
            ensures
                - The feed-forward sublayer of a block. With UseAct == true the
                  SwiGLU transition is wrapped in an Adaptive Computation Time
                  layer (act_steps, per-position latent recurrence, at most
                  max_steps steps); with UseAct == false it is a plain SwiGLU.
                  Networks that already recur at a higher level (e.g. HRM, which
                  loops the whole stack) build with UseAct == false to avoid
                  nesting two recurrence mechanisms.
                - The SwiGLU hidden size is d_model * hidden_num / hidden_den,
                  and LINEAR selects the projection layer of its three linears
                  (linear, or linear_no_bias for the bias-free Llama family).
        !*/

        template<bool UseAct, long d_model, long num_heads, long num_kv_heads,
            long hidden_num, long hidden_den, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using transformer_block = some_template_expression;
        /*!
            ensures
                - Same pre-norm residual topology as
                  gqa_transformer::transformer_block but with the fused attention
                  sublayer and the ffn_sublayer feed-forward described above:
                  rms_norm -> gqa_attention -> residual, then
                  rms_norm -> ffn_sublayer -> residual.
        !*/

        template<long num_layers, long d_model, long num_heads, long num_kv_heads,
            typename SUBNET, bool UseAct = true,
            long hidden_num = 8, long hidden_den = 3,
            template <unsigned long, typename> class LINEAR = linear,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using transformer_stack = some_template_expression;
        /*!
            ensures
                - Stacks num_layers fused-attention transformer_block units with
                  a tag10<> sentinel at the bottom of the stack.
                - The trailing parameters default to the historical behavior
                  (ACT-wrapped feed-forward, 8/3 hidden ratio, biased linears,
                  standard head_dim, no QK-Norm) so existing dense
                  configurations that omit them are unchanged. Decoder imports
                  (decoder_transformer_config) pass UseAct == false, the exact
                  model ratio, linear_no_bias, and the model's head_dim /
                  qk_norm.
        !*/

    } // namespace gqa_transformer_unified

    // ----------------------------------------------------------------------------------------

    namespace canonical_transformer
    {
        /*!
            WHAT THIS NAMESPACE REPRESENTS
                Standard multi-head attention transformer (no GQA), implemented
                with separate Q, K, V projections via linear_no_bias followed by
                explicit reshape operations. This is the most modular variant.

                Advantages:
                - Conceptually clearer and easier to debug.
                - Each projection can be independently inspected or modified.

                Use cases:
                - Prototyping new attention mechanisms.
                - Fine-grained control over each projection.

                Tensors keep the embedding dimension on nc() throughout the
                pipeline (linear-style layout).
        !*/

        template <long d_model, long num_heads, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - linear_no_bias projection to d_model followed by reshape into
                  (num_heads, -1, d_model / num_heads).
        !*/

        template <long d_model, long num_heads, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - linear_no_bias projection to d_model followed by reshape into
                  (num_heads, -1, d_model / num_heads).
        !*/

        template <long d_model, long num_heads, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - linear_no_bias projection to d_model followed by reshape into
                  (num_heads, -1, d_model / num_heads).
        !*/

        template <template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Complete multi-head self-attention sub-graph with causal masking
                and Rotary Positional Embedding on Q and K.

                Computes Attention(Q, K, V) = softmax((Q*K^T) / sqrt(d_head)) * V
                with a causal mask, then linearly projects the concatenated heads
                back to d_model.

            ARCHITECTURE FLOW
                1. Q, K, V projections (each into num_heads heads of size
                   d_model / num_heads).
                2. RoPE applied to Q and K.
                3. Scaled dot-product scores: Q @ K^T / sqrt(d_head).
                4. Causal mask via tril_mask.
                5. softmaxm over the last dimension.
                6. Multiplication by V.
                7. Reshape to (1, -1, d_model) and linear_no_bias projection
                   to d_model.
                8. Dropout policy DO applied to the result.

            TEMPLATE PARAMETERS
                - DO        : dropout policy (e.g. dropout_10 in training,
                              multiply in inference).
                - d_model   : model dimension (divisible by num_heads).
                - num_heads : number of attention heads.
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Pre-norm transformer block: rms_norm -> multihead_attention
                -> residual, then rms_norm -> ffn<ACT, d_model, 4> -> residual.

                The feed-forward expansion factor is 4 (standard practice).

            TEMPLATE PARAMETERS
                - ACT       : activation function template (e.g. gelu, silu).
                - DO        : dropout policy template.
                - d_model   : model dimension.
                - num_heads : number of attention heads.

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            ensures
                - Stacks num_layers transformer_block units, with a tag10<>
                  sentinel at the bottom of the stack.

            TYPICAL USAGE
                using my_model =
                    classification_head<vocab_size,
                    canonical_transformer::transformer_stack<6, gelu, dropout_10, 256, 8,
                    embeddings<vocab_size, 256,
                    input<matrix<int, 0, 1>>>>>;
        !*/

    } // namespace canonical_transformer

    // ----------------------------------------------------------------------------------------

    namespace fused_transformer
    {
        /*!
            WHAT THIS NAMESPACE REPRESENTS
                Performance-optimized transformer variant: Q, K and V are
                computed by a single fc_no_bias projection of width 3*d_model,
                then separated through extract operations. The same fc_no_bias
                pattern is used in the feed-forward sublayer.

                Advantages:
                - One large GEMM instead of three smaller ones for QKV.
                - Better GPU utilization through larger matrix operations.
                - Reduced memory bandwidth requirements.

                Tensors in this namespace carry the embedding dimension on k()
                (fc-style layout), which is why the embedded rms_norm auto-
                detects the embedding axis to remain consistent.
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - extract<0, num_heads, d_model/num_heads, 1, SUBNET> applied to
                  the fused QKV projection: pulls out the Q block.
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - extract<d_model, num_heads, 1, d_model/num_heads, SUBNET>:
                  pulls out the K block from the fused QKV projection in a
                  transposed shape ready for the Q @ K^T multiplication.
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - extract<2*d_model, num_heads, d_model/num_heads, 1, SUBNET>:
                  pulls out the V block from the fused QKV projection.
        !*/

        template <template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Fused-QKV multi-head self-attention. Functionally equivalent to
                canonical_transformer::multihead_attention but built around a
                single fc_no_bias<d_model*3> projection.

            ARCHITECTURE FLOW
                1. fc_no_bias to d_model*3 (fused QKV).
                2. extract Q, K, V.
                3. Scaled dot-product scores Q @ K^T (no explicit RoPE in this
                   variant; positional information comes through learnable
                   positional_embeddings upstream).
                4. tril_mask + softmaxm.
                5. Multiplication by V, reshape to (1, -1, d_model).
                6. fc_no_bias output projection followed by extract and dropout
                   policy DO.
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using fused_ffn = some_template_expression;
        /*!
            ensures
                - Feed-forward built from fc<d_model*4> -> ACT -> fc<d_model>
                  -> DO, with an extract<0, 1, 1, d_model> at the front so the
                  fc layers see the expected layout.
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            ensures
                - Pre-norm transformer block, same residual topology as the
                  canonical variant, with the fused attention and fused_ffn
                  sublayers.
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            ensures
                - Stacks num_layers fused-attention transformer_block units with
                  a tag10<> sentinel at the bottom of the stack.
        !*/

    } // namespace fused_transformer

    // The chained GQA namespace is re-exported as the default GQA
    // implementation at the end of transformer.h:
    //   using namespace gqa_transformer;
    // so that gqa_transformer::transformer_stack and friends are also
    // available unqualified.

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
                - Phase 2: backward through gate net (with task gradient,
                      load balance gradient and z-loss gradient combined;
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

#endif // DLIB_DNN_TRANSFORMER_ABSTRACT_H_