// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Global inference/training context shared by the DNN layers.
//
// network_context lives in its own header because it is referenced from both
// layers.h (tril padding, adaptive computation time) and transformer.h (KV cache,
// parameter residency): a single declaration point breaks the include cycle and
// satisfies the strict two-phase name lookup of conforming compilers, where names
// used inside template bodies must be declared before the template definition.
#ifndef DLIB_DNN_NETWORK_CONTEXT_H_
#define DLIB_DNN_NETWORK_CONTEXT_H_

#include "network_context_abstract.h"
#include <mutex>
#include <vector>
#include "../cuda/tensor.h"

namespace dlib
{
    class network_context
    {
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
            // produce the output for this single position.
            incremental
        };

        static bool is_active()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_is_active_();
        }

        static void reset()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            reset_nolock_();
        }

        static void set_learning_rate(double lr)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_learning_rate_() = lr;
            get_is_active_() = true;
        }

        static double get_learning_rate()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_learning_rate_();
        }

        static bool is_training()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_learning_rate_() > 0.0;
        }

        static void set_padding(const tensor& input_tokens, long padding_token)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            if (padding_token < 0) {
                clear_padding_nolock_();
                return;
            }
            const long batch_size = input_tokens.num_samples();
            const long seq_len = input_tokens.nr();
            const long k = input_tokens.k();
            const long nc = input_tokens.nc();
            const float* data = input_tokens.host();
            get_padding_lengths_().resize(batch_size);
            for (long s = 0; s < batch_size; ++s)
            {
                long count = 0;
                for (long t = 0; t < seq_len; ++t)
                {
                    const long idx = (s * k * seq_len + t) * nc;
                    const long token = static_cast<long>(data[idx]);
                    if (token == padding_token) count++;
                    else break;
                }
                get_padding_lengths_()[s] = count;
            }
            get_is_padding_set_() = true;
            get_is_active_() = true;
        }

        static void set_padding_from_lengths(const std::vector<long>& lengths)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_padding_lengths_() = lengths;
            get_is_padding_set_() = true;
            get_is_active_() = true;
        }

        static void set_padding_uniform(long padding_length, long batch_size)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_padding_lengths_().assign(batch_size, padding_length);
            get_is_padding_set_() = true;
            get_is_active_() = true;
        }

        static void clear_padding()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            clear_padding_nolock_();
        }

        static long get_padding_length(long sample_idx)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            if (!get_is_padding_set_() || sample_idx < 0 ||
                sample_idx >= static_cast<long>(get_padding_lengths_().size()))
                return 0;
            return get_padding_lengths_()[sample_idx];
        }

        static std::vector<long> get_all_padding_lengths()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_padding_lengths_();
        }

        static bool is_padding_set()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_is_padding_set_();
        }

        static void set_optimizer_params(double weight_decay, double beta1, double beta2)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_optimizer_weight_decay_() = weight_decay;
            get_optimizer_beta1_() = beta1;
            get_optimizer_beta2_() = beta2;
            get_is_active_() = true;
        }

        static double get_optimizer_weight_decay()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_optimizer_weight_decay_();
        }

        static double get_optimizer_beta1()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_optimizer_beta1_();
        }

        static double get_optimizer_beta2()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_optimizer_beta2_();
        }

        // Sets the active inference mode. Attention layers read this to decide
        // how to interact with their KV caches. The default value is
        // inference_mode::training, which corresponds to the legacy behavior.
        static void set_inference_mode(inference_mode m)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_inference_mode_() = m;
            get_is_active_() = true;
        }

        static inference_mode get_inference_mode()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_inference_mode_();
        }

        // Sets the maximum number of positions that each per-layer KV cache
        // can hold. When the cache reaches this capacity, the generation loop
        // is expected to trigger a sliding-window reset (re-prefill). A value
        // of 0 means "no preset capacity"; the cache will then grow on demand
        // and the generation loop is responsible for managing its size.
        static void set_kv_cache_capacity(long capacity)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            DLIB_CASSERT(capacity >= 0, "KV cache capacity must be non-negative");
            get_kv_cache_capacity_() = capacity;
            get_is_active_() = true;
        }

        static long get_kv_cache_capacity()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_kv_cache_capacity_();
        }

        // Number of initial cache positions that are never evicted when the KV cache is
        // full ("attention sinks": typically the BOS token and the system block). Small
        // decoder models concentrate a large share of attention mass on the first
        // positions; evicting them collapses generation into repetitive output, so the
        // sliding mechanism keeps them and evicts a block of the oldest evictable
        // positions instead. A value of 0 gives a plain sliding window.
        static void set_kv_cache_keep_length(long keep)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            DLIB_CASSERT(keep >= 0, "KV cache keep length must be non-negative");
            get_kv_cache_keep_length_() = keep;
            get_is_active_() = true;
        }

        static long get_kv_cache_keep_length()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_kv_cache_keep_length_();
        }

        // Residency of the network parameters during inference. With `off` every layer
        // keeps its parameters resident on the device. With `host_f32` the parameter
        // tensors of the layers that support offloading are moved to host memory after
        // their first inference forward, and each forward materializes the weights of
        // the running layer just in time into a small pool of shared device buffers.
        // Device memory for those parameters then drops to the pool size (twice the
        // largest offloaded layer) whatever the number of layers, traded against one
        // host-to-device transfer per layer and per forward. Inference-only: training
        // mode ignores the setting. Enable it after the weights are loaded and before
        // the first inference forward.
        enum class parameter_residency { off, host_f32 };

        static void set_parameter_residency(parameter_residency r)
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_parameter_residency_() = r;
            get_is_active_() = true;
        }

        static parameter_residency get_parameter_residency()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            return get_parameter_residency_();
        }

        // Shared pool of parameter materialization buffers, rotated across calls. Two
        // buffers so a later increment can prefetch the next layer asynchronously while
        // the current one computes; the synchronous path simply alternates. Single
        // generation thread assumed, like the rest of the inference state.
        static resizable_tensor& acquire_param_buffer()
        {
            static resizable_tensor buffers[2];
            static int next = 0;
            resizable_tensor& b = buffers[static_cast<size_t>(next)];
            next = (next + 1) % 2;
            return b;
        }

        // Signals to all KV-cache-aware layers that any cached state should be
        // discarded on the next forward pass. The flag persists until the
        // generation loop explicitly calls clear_kv_cache_request(), so that
        // every attention layer of the sweep observes the same request. Layers
        // can also manage their cache lifetime independently.
        static void request_kv_cache_clear()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_kv_cache_clear_request_() = true;
            get_is_active_() = true;
        }

        // Layers call this at the start of forward to check whether they should
        // discard their cache. Returns the current value of the flag without
        // clearing it, so every layer of the same forward pass sees the same
        // request; the generation loop clears it afterwards.
        static bool consume_kv_cache_clear_request()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            const bool flag = get_kv_cache_clear_request_();
            // The flag is intentionally NOT cleared here. The generation loop
            // is responsible for clearing it explicitly via clear_kv_cache_request().
            // This way, all attention layers in the same forward pass observe
            // the same value.
            return flag;
        }

        // Explicitly clears the cache-clear request flag. The generation loop
        // calls this after a forward pass that has consumed the request.
        static void clear_kv_cache_request()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_kv_cache_clear_request_() = false;
        }

    private:

        static void clear_padding_nolock_()
        {
            get_padding_lengths_().clear();
            get_is_padding_set_() = false;
        }

        static void reset_nolock_()
        {
            get_is_active_() = false;
            get_learning_rate_() = 0.0;
            get_optimizer_weight_decay_() = 0.004;
            get_optimizer_beta1_() = 0.9;
            get_optimizer_beta2_() = 0.999;
            get_inference_mode_() = inference_mode::training;
            get_kv_cache_capacity_() = 0;
            get_kv_cache_keep_length_() = 0;
            get_parameter_residency_() = parameter_residency::off;
            get_kv_cache_clear_request_() = false;
            clear_padding_nolock_();
        }

        static std::mutex& get_mutex_() {
            static std::mutex m;
            return m;
        }
        static bool& get_is_active_() {
            static bool v = false;
            return v;
        }
        static double& get_learning_rate_() {
            static double v = 0.0;
            return v;
        }
        static std::vector<long>& get_padding_lengths_() {
            static std::vector<long> v;
            return v;
        }
        static bool& get_is_padding_set_() {
            static bool v = false;
            return v;
        }
        static double& get_optimizer_weight_decay_() {
            static double v = 0.004;
            return v;
        }
        static double& get_optimizer_beta1_() {
            static double v = 0.9;
            return v;
        }
        static double& get_optimizer_beta2_() {
            static double v = 0.999;
            return v;
        }

        static inference_mode& get_inference_mode_() {
            static inference_mode v = inference_mode::training;
            return v;
        }
        static long& get_kv_cache_capacity_() {
            static long v = 0;
            return v;
        }
        static long& get_kv_cache_keep_length_() {
            static long v = 0;
            return v;
        }
        static parameter_residency& get_parameter_residency_() {
            static parameter_residency v = parameter_residency::off;
            return v;
        }
        static bool& get_kv_cache_clear_request_() {
            static bool v = false;
            return v;
        }
    };
}

#endif // DLIB_DNN_NETWORK_CONTEXT_H_
