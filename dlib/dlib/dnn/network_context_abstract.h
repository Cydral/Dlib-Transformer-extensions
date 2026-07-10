// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_NETWORK_CONTEXT_ABSTRACT_H_
#ifdef DLIB_DNN_NETWORK_CONTEXT_ABSTRACT_H_

#include "../cuda/tensor_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class network_context
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thread-safe singleton providing a shared execution context accessible
                by any layer during forward and backward passes.

                It solves the problem that deeply nested layers (e.g. a transition
                network inside an ACT layer, or a KV-cache-aware attention layer)
                cannot directly access information about the enclosing network or
                the current execution state. The caller sets the context once before
                each forward pass; any layer reads it as needed.

                The context has an explicit active/inactive state. It is considered
                inactive until at least one setter has been called. Any layer can
                query is_active() before reading values, allowing graceful fallback
                to default behavior when the context has not been initialized by the
                caller.

                The context distinguishes the following operating modes:
                  - Training       : learning_rate > 0, set by the training loop
                  - Inference full : default inference, no KV cache interaction
                  - Inference prefill : initial forward on the prompt that also
                                        populates the per-layer KV caches
                  - Inference incremental : subsequent single-token forwards that
                                            read and extend the KV caches

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

                // --- Inference (full mode, no cache) ---
                network_context::reset();
                network_context::set_inference_mode(
                    network_context::inference_mode::full);

                // --- Inference (prefill + incremental, with KV cache) ---
                network_context::reset();
                network_context::set_kv_cache_capacity(max_seq_len);
                network_context::set_inference_mode(
                    network_context::inference_mode::prefill);
                // ... forward on the prompt ...
                network_context::set_inference_mode(
                    network_context::inference_mode::incremental);
                // ... forward on single new tokens, in a loop ...

                // --- Full teardown after pipeline completes ---
                network_context::reset();
        !*/

    public:

        enum class inference_mode
        {
            // No inference-time state management. Used during training and for
            // any forward call that should not interact with the KV cache.
            training,

            // Standard inference forward without cache. Equivalent to training
            // mode for the attention layers but signals "we are not training".
            full,

            // Prefill mode: forward pass on the initial prompt; attention layers
            // populate their KV caches in addition to producing the regular
            // output. This is the first step of an autoregressive generation.
            prefill,

            // Incremental mode: forward pass on a single new token; attention
            // layers read the existing KV cache, append the new token's K/V, and
            // produce the output for this single position. When the cache is
            // full, attention layers evict a block of the oldest evictable
            // positions, preserving the configured attention sinks (see
            // set_kv_cache_keep_length).
            incremental
        };

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
                - Returns the singleton to its initial inactive state and clears
                  all fields:
                    * #is_active()             == false
                    * #is_training()           == false
                    * #get_learning_rate()     == 0.0
                    * #is_padding_set()        == false
                    * #get_inference_mode()    == inference_mode::training
                    * #get_kv_cache_capacity() == 0
                    * #get_kv_cache_keep_length() == 0
                    * #get_parameter_residency() == parameter_residency::off
                    * The internal "clear KV cache" flag is cleared.
                    * Optimizer hyperparameters revert to their defaults
                      (weight_decay=0.004, beta1=0.9, beta2=0.999).
        !*/

        // Learning rate and training mode

        static void set_learning_rate(double lr);
        /*!
            ensures
                - Stores the current learning rate.
                - #get_learning_rate() == lr
                - #is_training()       == (lr > 0.0)
                - #is_active()         == true
                - A value of 0.0 signals inference mode (no parameter updates).
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

        // Optimizer hyperparameters (read by AdamW-style internal solvers in
        // sub-network layers such as hrm_ and moe_)

        static void set_optimizer_params(double weight_decay, double beta1, double beta2);
        /*!
            ensures
                - Stores the optimizer hyperparameters used by internally managed
                  sub-networks (hrm_, moe_, ...).
                - #get_optimizer_weight_decay() == weight_decay
                - #get_optimizer_beta1()        == beta1
                - #get_optimizer_beta2()        == beta2
                - #is_active()                  == true
        !*/

        static double get_optimizer_weight_decay();
        static double get_optimizer_beta1();
        static double get_optimizer_beta2();
        /*!
            ensures
                - Return the stored optimizer hyperparameters (defaults:
                  weight_decay=0.004, beta1=0.9, beta2=0.999).
        !*/

        // Inference mode and KV cache configuration

        static void set_inference_mode(inference_mode m);
        /*!
            ensures
                - Stores the active inference mode. KV-cache-aware attention
                  layers read this at the start of every forward call to decide
                  whether to use cached K/V tensors.
                - #get_inference_mode() == m
                - #is_active()          == true
        !*/

        static inference_mode get_inference_mode();
        /*!
            ensures
                - Returns the current inference mode (default
                  inference_mode::training, which corresponds to the legacy
                  no-cache behavior).
        !*/

        static void set_kv_cache_capacity(long capacity);
        /*!
            requires
                - capacity >= 0
            ensures
                - Stores the maximum number of positions each per-layer KV cache
                  may hold. When the cache reaches this capacity in incremental
                  mode, attention layers evict a block of the oldest evictable
                  positions in one operation: the first get_kv_cache_keep_length()
                  positions are preserved (attention sinks), then half of the
                  remaining range is dropped and the tail is compacted down, so
                  the compaction cost is amortized over many generated tokens.
                - A value of 0 means "no preset capacity"; the cache then grows
                  on demand and the generation loop becomes responsible for
                  managing its size (e.g. by re-prefilling).
                - #get_kv_cache_capacity() == capacity
                - #is_active()             == true
        !*/

        static long get_kv_cache_capacity();
        /*!
            ensures
                - Returns the configured KV cache capacity, or 0 if never set.
        !*/

        static void set_kv_cache_keep_length(long keep);
        /*!
            requires
                - keep >= 0
            ensures
                - Stores the number of initial cache positions that are never
                  evicted when a full KV cache is compacted in incremental mode
                  ("attention sinks": typically the BOS token and the system
                  block of the prompt). Small decoder models concentrate a large
                  share of attention mass on the first positions; evicting them
                  collapses generation into repetitive output, so the eviction
                  drops a block of the oldest positions after the sinks instead.
                - The value is clamped by the attention layers so that at least
                  two evictable positions remain.
                - A value of 0 gives a plain sliding window.
                - #get_kv_cache_keep_length() == keep
                - #is_active()                == true
        !*/

        static long get_kv_cache_keep_length();
        /*!
            ensures
                - Returns the configured keep length, or 0 if never set.
        !*/

        static void request_kv_cache_clear();
        /*!
            ensures
                - Sets an internal one-shot flag instructing every KV-cache-aware
                  layer to discard its cached state on the next forward pass.
                - The flag must be explicitly cleared via clear_kv_cache_request()
                  by the generation loop once it has been consumed by all layers
                  in a forward pass.
                - #is_active() == true
        !*/

        static bool consume_kv_cache_clear_request();
        /*!
            ensures
                - Returns the current value of the cache-clear flag.
                - Does NOT clear the flag (so that every attention layer in the
                  same forward pass observes the same value).
        !*/

        static void clear_kv_cache_request();
        /*!
            ensures
                - Clears the cache-clear flag. The generation loop calls this
                  after a forward pass that has consumed the request.
        !*/

        enum class parameter_residency
        {
            off,     // every layer keeps its parameters resident on the device
            host_f32 // offload-capable layers keep their parameters in host memory
        };
        /*!
            WHAT THIS ENUM REPRESENTS
                Residency of the network parameters during inference. With host_f32,
                the parameter tensors of the layers that support offloading are moved
                to host memory after their first inference forward, and every forward
                materializes the weights of the running layer just in time into the
                shared buffers of acquire_param_buffer(). Device memory for those
                parameters then drops to the pool size (twice the largest offloaded
                layer) whatever the number of layers, traded against one host-to-device
                transfer per layer and per forward.
        !*/

        static void set_parameter_residency(parameter_residency r);
        /*!
            ensures
                - Stores the parameter residency applied by offload-capable layers
                  during inference. Training mode ignores the setting: the training
                  forward always reads the resident tensors, so optimizer semantics
                  are untouched.
                - Enable the residency after the weights are loaded and before the
                  first inference forward: layers capture their parameters on first
                  use, and a capture performed on unloaded parameters would preserve
                  meaningless values.
                - #get_parameter_residency() == r
                - #is_active()               == true
        !*/

        static parameter_residency get_parameter_residency();
        /*!
            ensures
                - Returns the configured residency, or parameter_residency::off if
                  never set.
        !*/

        static resizable_tensor& acquire_param_buffer();
        /*!
            ensures
                - Returns one buffer of the shared parameter materialization pool,
                  rotating across calls. Two buffers back the pool, so a later
                  increment can prefetch the next layer asynchronously while the
                  current one computes; the synchronous path simply alternates.
                - The returned buffer stays valid until the pool hands the same slot
                  out again, i.e. across one full layer forward when every layer
                  materializes at most once per pass.
                - A single generation thread is assumed, like the rest of the
                  inference state.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_NETWORK_CONTEXT_ABSTRACT_H_
