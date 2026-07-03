// Copyright (C) 2026  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_TRANSFORMER_H_
#define DLIB_DNN_TRANSFORMER_H_

#include "transformer_abstract.h"
#include "layers.h"

namespace dlib
{
    class adamw; // Forward declaration

    // ----------------------------------------------------------------------------------------

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

        // Signals to all KV-cache-aware layers that any cached state should be
        // discarded on the next forward pass. The flag auto-resets after being
        // observed (each layer that consumes it during forward will set it back
        // to false on its own end). Concretely: this is a one-shot "please
        // clear caches" marker. Layers can also manage their cache lifetime
        // independently.
        static void request_kv_cache_clear()
        {
            std::lock_guard<std::mutex> lock(get_mutex_());
            get_kv_cache_clear_request_() = true;
            get_is_active_() = true;
        }

        // Layers call this at the start of forward to check whether they should
        // discard their cache. Returns true once and resets the flag so that
        // subsequent layers/calls in the same forward pass also see the
        // request, but it does not persist beyond the current forward sweep.
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
        static bool& get_kv_cache_clear_request_() {
            static bool v = false;
            return v;
        }
    };

    // ----------------------------------------------------------------------------------------

    template <typename T>
    long count_leading_padding(const matrix<T, 0, 1>& seq, T padding_token)
    {
        long count = 0;
        for (long i = 0; i < seq.size(); ++i) {
            if (seq(i) == padding_token) count++;
            else break;
        }
        return count;
    }

    // ----------------------------------------------------------------------------------------

    // Returns the effective (non-padded) length of sample `sample_idx` given a
    // total sequence length `total_seq_len`. Reads the active padding lengths
    // from network_context. If no padding is set for this sample, returns
    // total_seq_len.
    // This is the canonical way for layers to know how many "real" positions
    // they should consider when interacting with the KV cache or any other
    // position-dependent state.
    inline long effective_length(long sample_idx, long total_seq_len)
    {
        const long pad_len = network_context::get_padding_length(sample_idx);
        return std::max<long>(0, total_seq_len - pad_len);
    }

    // ----------------------------------------------------------------------------------------

    template <long d_k_>
    class scale_weights_ : public multiply_
    {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    template <long num_embeddings, long embedding_length, typename SUBNET>
    using positional_embeddings = positional_encodings<
        embeddings<num_embeddings, embedding_length, SUBNET>>;

    // ----------------------------------------------------------------------------------------

    template <long EMBEDDING_DIM_, long NUM_HEADS_, long NUM_KV_HEADS_, long HEAD_DIM_,
        bool USE_QK_NORM_ = false>
    class gqa_attention_
    {
    public:
        static constexpr long EMBEDDING_DIM = EMBEDDING_DIM_;
        static constexpr long NUM_HEADS = NUM_HEADS_;
        static constexpr long NUM_KV_HEADS = NUM_KV_HEADS_;
        static constexpr long HEAD_DIM = HEAD_DIM_;
        // When true, a per-head RMSNorm (QK-Norm) is applied to Q and K after the head split
        // and before RoPE, each scaled by its own gamma of size HEAD_DIM (used by Qwen3).
        // Default false leaves every other model unchanged: no extra parameters, same compute.
        static constexpr bool USE_QK_NORM = USE_QK_NORM_;
        static constexpr long REPEAT_FACTOR = NUM_HEADS / NUM_KV_HEADS;
        static constexpr long KV_PROJ_DIM = NUM_KV_HEADS * HEAD_DIM;
        // Query/output projection width. Decoupled from EMBEDDING_DIM: with a non-standard
        // head_dim (e.g. Qwen3, where head_dim != EMBEDDING_DIM / NUM_HEADS) this differs from
        // EMBEDDING_DIM. When head_dim == EMBEDDING_DIM / NUM_HEADS it equals EMBEDDING_DIM, so
        // existing models are unaffected.
        static constexpr long Q_PROJ_DIM = NUM_HEADS * HEAD_DIM;

        static_assert(EMBEDDING_DIM > 0, "EMBEDDING_DIM must be positive");
        static_assert(NUM_HEADS > 0, "NUM_HEADS must be positive");
        static_assert(NUM_KV_HEADS > 0 && (NUM_HEADS % NUM_KV_HEADS) == 0,
            "NUM_HEADS must be a multiple of NUM_KV_HEADS");
        static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
        static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even (RoPE constraint)");

        gqa_attention_()
            : learning_rate_multiplier_(1),
            weight_decay_multiplier_(1)
        {
        }

        double get_learning_rate_multiplier() const { return learning_rate_multiplier_; }
        double get_weight_decay_multiplier() const { return weight_decay_multiplier_; }
        void set_learning_rate_multiplier(double v) { learning_rate_multiplier_ = v; }
        void set_weight_decay_multiplier(double v) { weight_decay_multiplier_ = v; }

        // YaRN forwarders to the embedded RoPE helpers
        void set_yarn_params(float alpha, float beta,
            long original_len = 0, bool enabled = true)
        {
            rope_q_helper_.get().set_yarn_params(alpha, beta, original_len, enabled);
            rope_k_helper_.get().set_yarn_params(alpha, beta, original_len, enabled);
        }
        const yarn_config& get_yarn_config() const
        {
            return rope_q_helper_.get().get_yarn_config();
        }

        // QK-Norm epsilon (only meaningful when USE_QK_NORM is true). Set it to the model's
        // RMSNorm epsilon (e.g. 1e-6 for Qwen3) before loading weights.
        void set_qk_norm_eps(double eps) { qk_norm_eps_ = eps; }
        double get_qk_norm_eps() const { return qk_norm_eps_; }

        // RoPE frequency base (theta). Set it to the model's value (e.g. 1000000 for Qwen3,
        // 500000 for Llama 3) before the first forward pass, since the trig caches are built
        // then and are not recomputed on a later base change.
        void set_rope_theta_base(float base)
        {
            rope_q_helper_.get().set_theta_base(base);
            rope_k_helper_.get().set_theta_base(base);
        }

        // KV cache state accessors
        void reset_kv_cache() { cache_filled_len_ = 0; }
        long get_kv_cache_filled_len() const { return cache_filled_len_; }
        long get_kv_cache_capacity() const { return cache_capacity_; }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            const tensor& x = sub.get_output();
            DLIB_CASSERT(x.k() == 1, "gqa_attention expects k()==1 input");
            DLIB_CASSERT(x.nc() == EMBEDDING_DIM,
                "gqa_attention input nc()==" << x.nc()
                << " does not match EMBEDDING_DIM=" << EMBEDDING_DIM);

            const long total_params =
                EMBEDDING_DIM * Q_PROJ_DIM
                + EMBEDDING_DIM * KV_PROJ_DIM
                + EMBEDDING_DIM * KV_PROJ_DIM
                + Q_PROJ_DIM * EMBEDDING_DIM
                + (USE_QK_NORM ? 2 * HEAD_DIM : 0);

            params.set_size(1, total_params);

            dlib::rand rnd(std::rand());
            randomize_block(rnd, EMBEDDING_DIM, Q_PROJ_DIM, params, 0);
            randomize_block(rnd, EMBEDDING_DIM, KV_PROJ_DIM, params, EMBEDDING_DIM * Q_PROJ_DIM);
            randomize_block(rnd, EMBEDDING_DIM, KV_PROJ_DIM, params, EMBEDDING_DIM * Q_PROJ_DIM
                + EMBEDDING_DIM * KV_PROJ_DIM);
            randomize_block(rnd, Q_PROJ_DIM, EMBEDDING_DIM, params, EMBEDDING_DIM * Q_PROJ_DIM
                + 2 * EMBEDDING_DIM * KV_PROJ_DIM);

            wq_alias = alias_tensor(EMBEDDING_DIM, Q_PROJ_DIM);
            wk_alias = alias_tensor(EMBEDDING_DIM, KV_PROJ_DIM);
            wv_alias = alias_tensor(EMBEDDING_DIM, KV_PROJ_DIM);
            wo_alias = alias_tensor(Q_PROJ_DIM, EMBEDDING_DIM);

            wq_offset = 0;
            wk_offset = EMBEDDING_DIM * Q_PROJ_DIM;
            wv_offset = wk_offset + EMBEDDING_DIM * KV_PROJ_DIM;
            wo_offset = wv_offset + EMBEDDING_DIM * KV_PROJ_DIM;

            if (USE_QK_NORM)
            {
                setup_qk_norm_aliases();
                gamma_q_alias(params, gamma_q_offset) = 1;
                gamma_k_alias(params, gamma_k_offset) = 1;
            }

            if (x.nr() > 0)
            {
                resizable_tensor q_probe(x.num_samples(), NUM_HEADS, x.nr(), HEAD_DIM);
                resizable_tensor k_probe(x.num_samples(), NUM_KV_HEADS, x.nr(), HEAD_DIM);
                rope_q_helper_.setup(q_probe);
                rope_k_helper_.setup(k_probe);
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& x = sub.get_output();
            DLIB_CASSERT(x.k() == 1 && x.nc() == EMBEDDING_DIM);

            const auto inference_mode_now = network_context::get_inference_mode();
            if (network_context::consume_kv_cache_clear_request())
                reset_kv_cache();

            if (inference_mode_now == network_context::inference_mode::incremental
                && x.nr() == 1 && cache_filled_len_ > 0)
            {
                forward_incremental(x, output);
                return;
            }

            forward_full(x, output, inference_mode_now);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            const long B = saved_x_shape_B;
            const long N = saved_x_shape_N;
            DLIB_CASSERT(gradient_input.num_samples() == B);
            DLIB_CASSERT(gradient_input.nr() == N);

            const tensor& x = sub.get_output();
            tensor& dx = sub.get_gradient_input();

            auto wq = wq_alias(params, wq_offset);
            auto wk = wk_alias(params, wk_offset);
            auto wv = wv_alias(params, wv_offset);
            auto wo = wo_alias(params, wo_offset);

            auto dwq = wq_alias(params_grad, wq_offset);
            auto dwk = wk_alias(params_grad, wk_offset);
            auto dwv = wv_alias(params_grad, wv_offset);
            auto dwo = wo_alias(params_grad, wo_offset);

            const float qk_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

            // Step 10 backward
            resizable_tensor dctx_flat(B, 1, N, Q_PROJ_DIM);
            {
                alias_tensor go_2d(B * N, EMBEDDING_DIM);
                auto go_view = go_2d(const_cast<tensor&>(gradient_input), 0);

                alias_tensor dctx_2d(B * N, Q_PROJ_DIM);
                auto dctx_view = dctx_2d(dctx_flat, 0);

                tt::gemm(0.0f, dctx_view, 1.0f, go_view, false, wo, true);

                alias_tensor cf_2d(B * N, Q_PROJ_DIM);
                auto cf_view = cf_2d(saved_ctx_flat, 0);
                tt::gemm(0.0f, dwo, 1.0f, cf_view, true, go_view, false);
            }

            // Step 9 backward
            resizable_tensor dctx_4d(B, NUM_HEADS, N, HEAD_DIM);
            tt::split_heads(false, dctx_4d, dctx_flat);

            // Step 8 backward
            resizable_tensor dattn(B, NUM_HEADS, N, N);
            resizable_tensor dV_repeated(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, dattn, 1.0f, dctx_4d, false, saved_v_repeated, true, operation_mode::PLANE_WISE);
            tt::gemm(0.0f, dV_repeated, 1.0f, saved_attn, true, dctx_4d, false, operation_mode::PLANE_WISE);

            // Step 7 backward
            resizable_tensor dscores_masked(B, NUM_HEADS, N, N);
            tt::softmax_gradient(dscores_masked, saved_attn, dattn, operation_mode::PLANE_WISE);

            // Step 6 backward
            resizable_tensor dscores;
            tril_helper_.backward(dscores_masked, saved_scores, dscores);

            // Step 5 backward
            resizable_tensor dQ(B, NUM_HEADS, N, HEAD_DIM);
            resizable_tensor dK_repeated(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, dQ, qk_scale, dscores, false, saved_k_repeated, false, operation_mode::PLANE_WISE);
            tt::gemm(0.0f, dK_repeated, 1.0f, dscores, true, saved_q_post_rope, false, operation_mode::PLANE_WISE);

            // Step 4 backward (GQA repeat sum)
            resizable_tensor dK_pre_repeat(B, NUM_KV_HEADS, N, HEAD_DIM);
            resizable_tensor dV_pre_repeat(B, NUM_KV_HEADS, N, HEAD_DIM);
            dK_pre_repeat = 0;
            dV_pre_repeat = 0;
            for (long kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
            {
                for (long r = 0; r < REPEAT_FACTOR; ++r)
                {
                    const size_t src_h = static_cast<size_t>(kv_h * REPEAT_FACTOR + r);
                    tt::copy_tensor(true, dK_pre_repeat, static_cast<size_t>(kv_h), dK_repeated, src_h, 1);
                    tt::copy_tensor(true, dV_pre_repeat, static_cast<size_t>(kv_h), dV_repeated, src_h, 1);
                }
            }

            // Step 3 backward: RoPE, then (optional) QK-Norm. When QK-Norm is active, the tensor
            // RoPE saw in forward was the *normed* Q/K, so RoPE backward yields the gradient w.r.t.
            // the normed tensor, which then flows back through the per-head RMSNorm to the pre-norm
            // Q/K while accumulating the gamma gradients.
            resizable_tensor dQ_pre_rope;
            resizable_tensor dK_pre_rope;
            if (USE_QK_NORM)
            {
                auto gq = gamma_q_alias(params, gamma_q_offset);
                auto gk = gamma_k_alias(params, gamma_k_offset);
                auto dgq = gamma_q_alias(params_grad, gamma_q_offset);
                auto dgk = gamma_k_alias(params_grad, gamma_k_offset);

                resizable_tensor dQ_norm, dK_norm, dscale_tmp;
                rope_q_helper_.backward(dQ, saved_q_normed, dQ_norm);
                rope_k_helper_.backward(dK_pre_repeat, saved_k_normed, dK_norm);

                // rms_normalize_gradient accumulates into src_grad, so zero it first.
                dQ_pre_rope.copy_size(saved_q_pre_rope); dQ_pre_rope = 0;
                dK_pre_rope.copy_size(saved_k_pre_rope); dK_pre_rope = 0;

                tt::rms_normalize_gradient(dQ_norm, qk_scale_q, saved_q_pre_rope, gq,
                    dQ_pre_rope, dgq, dscale_tmp, true);
                tt::rms_normalize_gradient(dK_norm, qk_scale_k, saved_k_pre_rope, gk,
                    dK_pre_rope, dgk, dscale_tmp, true);
            }
            else
            {
                rope_q_helper_.backward(dQ, saved_q_pre_rope, dQ_pre_rope);
                rope_k_helper_.backward(dK_pre_repeat, saved_k_pre_rope, dK_pre_rope);
            }

            // Step 2 backward (merge heads)
            resizable_tensor dQ_flat(B, 1, N, Q_PROJ_DIM);
            resizable_tensor dK_flat(B, 1, N, KV_PROJ_DIM);
            resizable_tensor dV_flat(B, 1, N, KV_PROJ_DIM);
            tt::merge_heads(false, dQ_flat, dQ_pre_rope);
            tt::merge_heads(false, dK_flat, dK_pre_rope);
            tt::merge_heads(false, dV_flat, dV_pre_repeat);

            // Step 1 backward (linear projections)
            {
                alias_tensor x_2d(B * N, EMBEDDING_DIM);
                auto x_view = x_2d(const_cast<tensor&>(x), 0);

                alias_tensor dx_2d(B * N, EMBEDDING_DIM);
                auto dx_view = dx_2d(dx, 0);

                alias_tensor dQ_2d(B * N, Q_PROJ_DIM);
                auto dQ_view = dQ_2d(dQ_flat, 0);
                alias_tensor dK_2d(B * N, KV_PROJ_DIM);
                auto dK_view = dK_2d(dK_flat, 0);
                alias_tensor dV_2d(B * N, KV_PROJ_DIM);
                auto dV_view = dV_2d(dV_flat, 0);

                tt::gemm(1.0f, dx_view, 1.0f, dQ_view, false, wq, true);
                tt::gemm(1.0f, dx_view, 1.0f, dK_view, false, wk, true);
                tt::gemm(1.0f, dx_view, 1.0f, dV_view, false, wv, true);

                tt::gemm(0.0f, dwq, 1.0f, x_view, true, dQ_view, false);
                tt::gemm(0.0f, dwk, 1.0f, x_view, true, dK_view, false);
                tt::gemm(0.0f, dwv, 1.0f, x_view, true, dV_view, false);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const gqa_attention_& item, std::ostream& out)
        {
            serialize("gqa_attention_", out);
            serialize(item.params, out);
            serialize(item.learning_rate_multiplier_, out);
            serialize(item.weight_decay_multiplier_, out);
            serialize(item.rope_q_helper_, out);
            serialize(item.rope_k_helper_, out);
            serialize(item.tril_helper_, out);
            // qk_norm_eps_ is part of the stream only for QK-Norm layers, so every other
            // model keeps the exact same serialized format.
            if (USE_QK_NORM)
                serialize(item.qk_norm_eps_, out);
        }

        friend void deserialize(gqa_attention_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "gqa_attention_")
                throw serialization_error(
                    "Unexpected version found while deserializing dlib::gqa_attention_. ");
            deserialize(item.params, in);
            deserialize(item.learning_rate_multiplier_, in);
            deserialize(item.weight_decay_multiplier_, in);
            deserialize(item.rope_q_helper_, in);
            deserialize(item.rope_k_helper_, in);
            deserialize(item.tril_helper_, in);
            if (USE_QK_NORM)
                deserialize(item.qk_norm_eps_, in);
            item.rebuild_aliases();

            // KV cache state is runtime-only.
            item.K_cache.clear();
            item.V_cache.clear();
            item.cache_filled_len_ = 0;
            item.cache_capacity_ = 0;
            item.cache_batch_size_ = 0;
        }

        friend std::ostream& operator<<(std::ostream& out, const gqa_attention_& item)
        {
            out << "gqa_attention"
                << " (emb=" << EMBEDDING_DIM
                << ", heads=" << NUM_HEADS
                << ", kv_heads=" << NUM_KV_HEADS
                << ", head_dim=" << HEAD_DIM
                << ", qk_norm=" << (USE_QK_NORM ? "on" : "off") << ")"
                << " learning_rate_mult=" << item.learning_rate_multiplier_
                << " weight_decay_mult=" << item.weight_decay_multiplier_;
            return out;
        }

        friend void to_xml(const gqa_attention_& item, std::ostream& out)
        {
            out << "<gqa_attention"
                << " emb='" << EMBEDDING_DIM << "'"
                << " heads='" << NUM_HEADS << "'"
                << " kv_heads='" << NUM_KV_HEADS << "'"
                << " head_dim='" << HEAD_DIM << "'"
                << " qk_norm='" << (USE_QK_NORM ? "true" : "false") << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier_ << "'"
                << " weight_decay_mult='" << item.weight_decay_multiplier_ << "'/>\n";
        }

    private:
        // Parameter storage (W_Q | W_K | W_V | W_O [| gamma_q | gamma_k]) packed contiguously.
        // The two QK-Norm gammas are present only when USE_QK_NORM is true.
        resizable_tensor params;
        alias_tensor wq_alias, wk_alias, wv_alias, wo_alias;
        size_t wq_offset = 0, wk_offset = 0, wv_offset = 0, wo_offset = 0;
        alias_tensor gamma_q_alias, gamma_k_alias;
        size_t gamma_q_offset = 0, gamma_k_offset = 0;
        double qk_norm_eps_ = DEFAULT_RMS_NORM_EPS;

        // Forward state saved for backward (training only)
        resizable_tensor saved_q_pre_rope;
        resizable_tensor saved_k_pre_rope;
        resizable_tensor saved_q_post_rope;
        resizable_tensor saved_k_post_rope;
        resizable_tensor saved_v_4d;
        resizable_tensor saved_k_repeated;
        resizable_tensor saved_v_repeated;
        resizable_tensor saved_scores;
        resizable_tensor saved_attn;
        resizable_tensor saved_ctx_flat;
        // QK-Norm forward state (training): normed Q/K fed to RoPE, and the per-row inverse RMS.
        resizable_tensor saved_q_normed;
        resizable_tensor saved_k_normed;
        resizable_tensor qk_scale_q;
        resizable_tensor qk_scale_k;

        long saved_x_shape_B = 0;
        long saved_x_shape_N = 0;

        double learning_rate_multiplier_;
        double weight_decay_multiplier_;

        // Embedded parameter-free layers
        layer_helper<rotary_positional_embedding_>  rope_q_helper_;
        layer_helper<rotary_positional_embedding_>  rope_k_helper_;
        layer_helper<tril_<0, neg_infinity_tag>>    tril_helper_;

        // KV cache: stores K *before* RoPE and V (no transformation).
        // RoPE is re-applied at every attention computation. Layout:
        // [B, NUM_KV_HEADS, capacity, HEAD_DIM]. cache_filled_len_ is the
        // number of valid positions in [0, cache_filled_len_). When the
        // cache is full and a new position is incoming, the oldest position
        // is dropped via a left shift, keeping cache_filled_len_ at capacity.
        resizable_tensor K_cache;
        resizable_tensor V_cache;
        long cache_filled_len_ = 0;
        long cache_capacity_ = 0;
        long cache_batch_size_ = 0;

        // Incremental scratch buffers (reused across calls)
        resizable_tensor inc_q;
        resizable_tensor inc_k;
        resizable_tensor inc_v;
        resizable_tensor inc_k_window;
        resizable_tensor inc_v_window;
        resizable_tensor inc_k_window_rep;
        resizable_tensor inc_v_window_rep;
        resizable_tensor inc_k_window_rope;
        resizable_tensor inc_q_rope;
        resizable_tensor inc_scores;
        resizable_tensor inc_attn;
        resizable_tensor inc_ctx;
        // QK-Norm incremental scratch (inference): normed Q and normed K window, plus scales.
        resizable_tensor inc_q_normed;
        resizable_tensor inc_k_window_normed;
        resizable_tensor inc_scale_q;
        resizable_tensor inc_scale_k;

        void rebuild_aliases()
        {
            wq_alias = alias_tensor(EMBEDDING_DIM, Q_PROJ_DIM);
            wk_alias = alias_tensor(EMBEDDING_DIM, KV_PROJ_DIM);
            wv_alias = alias_tensor(EMBEDDING_DIM, KV_PROJ_DIM);
            wo_alias = alias_tensor(Q_PROJ_DIM, EMBEDDING_DIM);
            wq_offset = 0;
            wk_offset = EMBEDDING_DIM * Q_PROJ_DIM;
            wv_offset = wk_offset + EMBEDDING_DIM * KV_PROJ_DIM;
            wo_offset = wv_offset + EMBEDDING_DIM * KV_PROJ_DIM;
            if (USE_QK_NORM)
                setup_qk_norm_aliases();
        }

        // Place the two QK-Norm gammas right after W_O in the packed parameter blob.
        void setup_qk_norm_aliases()
        {
            gamma_q_offset = wo_offset + static_cast<size_t>(Q_PROJ_DIM) * EMBEDDING_DIM;
            gamma_k_offset = gamma_q_offset + static_cast<size_t>(HEAD_DIM);
            gamma_q_alias = alias_tensor(1, 1, 1, HEAD_DIM);
            gamma_k_alias = alias_tensor(1, 1, 1, HEAD_DIM);
        }

        void randomize_block(
            dlib::rand& rnd,
            long rows, long cols,
            tensor& dest, size_t offset)
        {
            resizable_tensor tmp(rows, cols);
            randomize_parameters(tmp, rows + cols, rnd);

            alias_tensor dest_view(rows, cols);
            auto view = dest_view(dest, offset);
            tt::copy_tensor(false, view, 0, tmp, 0, tmp.k());
        }

        // Allocate the K/V caches if needed. The capacity is the maximum
        // number of positions the cache will ever hold; the sliding window
        // mechanism keeps the actual filled length at most equal to this.
        void ensure_cache_allocated(long batch_size, long min_capacity)
        {
            const long requested_cap = network_context::get_kv_cache_capacity();
            const long target_cap = std::max(min_capacity, requested_cap);

            if (cache_capacity_ >= target_cap && cache_batch_size_ == batch_size
                && K_cache.size() > 0 && V_cache.size() > 0) return;

            K_cache.set_size(batch_size, NUM_KV_HEADS, target_cap, HEAD_DIM);
            V_cache.set_size(batch_size, NUM_KV_HEADS, target_cap, HEAD_DIM);
            K_cache = 0;
            V_cache = 0;

            cache_capacity_ = target_cap;
            cache_batch_size_ = batch_size;
            cache_filled_len_ = 0;
        }

        // Shift the K/V caches one position to the left, dropping the oldest
        // entry. After this call cache_filled_len_ is decreased by 1 and a
        // free slot is available at the end of the valid range.
        void shift_cache_left()
        {
            DLIB_CASSERT(cache_filled_len_ > 0);

            const long new_len = cache_filled_len_ - 1;

            // Use a temporary buffer to perform the shift since copy_tensor
            // does not support overlapping source/destination ranges safely.
            resizable_tensor tmp_k(cache_batch_size_, NUM_KV_HEADS, new_len, HEAD_DIM);
            resizable_tensor tmp_v(cache_batch_size_, NUM_KV_HEADS, new_len, HEAD_DIM);

            tt::copy_tensor(false, tmp_k,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                K_cache,
                /*sk=*/0, /*snr=*/1, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(new_len),
                /*nc=*/HEAD_DIM);
            tt::copy_tensor(false, tmp_v,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                V_cache,
                /*sk=*/0, /*snr=*/1, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(new_len),
                /*nc=*/HEAD_DIM);

            tt::copy_tensor(false, K_cache,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                tmp_k,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(new_len),
                /*nc=*/HEAD_DIM);
            tt::copy_tensor(false, V_cache,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                tmp_v,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(new_len),
                /*nc=*/HEAD_DIM);

            cache_filled_len_ = new_len;
        }

        // Full forward path. Used in training/full mode (no cache write) and
        // in prefill mode (writes K_pre_rope and V into the cache for the
        // effective non-padded positions of the prompt).
        void forward_full(const tensor& x, resizable_tensor& output,
            network_context::inference_mode mode)
        {
            const long B = x.num_samples();
            const long N = x.nr();

            auto wq = wq_alias(params, wq_offset);
            auto wk = wk_alias(params, wk_offset);
            auto wv = wv_alias(params, wv_offset);
            auto wo = wo_alias(params, wo_offset);

            const float qk_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

            // Step 1: Q (pre-scaled), K, V projections
            resizable_tensor q_flat(B, 1, N, Q_PROJ_DIM);
            resizable_tensor k_flat(B, 1, N, KV_PROJ_DIM);
            resizable_tensor v_flat(B, 1, N, KV_PROJ_DIM);
            {
                alias_tensor x_2d(B * N, EMBEDDING_DIM);
                auto x_view = x_2d(const_cast<tensor&>(x), 0);

                alias_tensor q_2d(B * N, Q_PROJ_DIM);
                auto q_view = q_2d(q_flat, 0);
                tt::gemm(0.0f, q_view, qk_scale, x_view, false, wq, false);

                alias_tensor k_2d(B * N, KV_PROJ_DIM);
                auto k_view = k_2d(k_flat, 0);
                tt::gemm(0.0f, k_view, 1.0f, x_view, false, wk, false);

                alias_tensor v_2d(B * N, KV_PROJ_DIM);
                auto v_view = v_2d(v_flat, 0);
                tt::gemm(0.0f, v_view, 1.0f, x_view, false, wv, false);
            }

            // Step 2: split heads
            saved_q_pre_rope.set_size(B, NUM_HEADS, N, HEAD_DIM);
            saved_k_pre_rope.set_size(B, NUM_KV_HEADS, N, HEAD_DIM);
            saved_v_4d.set_size(B, NUM_KV_HEADS, N, HEAD_DIM);

            tt::split_heads(false, saved_q_pre_rope, q_flat);
            tt::split_heads(false, saved_k_pre_rope, k_flat);
            tt::split_heads(false, saved_v_4d, v_flat);

            // Step 3: optional QK-Norm (per-head RMSNorm on Q and K), then RoPE. The norm is
            // applied after the head split and before RoPE; the prefill cache below still stores
            // K before RoPE and before norm, since the norm is a deterministic per-token map
            // re-applied wherever K is used (locally here, and on the window in incremental mode).
            if (USE_QK_NORM)
            {
                auto gq = gamma_q_alias(params, gamma_q_offset);
                auto gk = gamma_k_alias(params, gamma_k_offset);
                tt::rms_normalize(qk_norm_eps_, saved_q_normed, qk_scale_q, saved_q_pre_rope, gq, true);
                tt::rms_normalize(qk_norm_eps_, saved_k_normed, qk_scale_k, saved_k_pre_rope, gk, true);
                rope_q_helper_.forward(saved_q_normed, saved_q_post_rope);
                rope_k_helper_.forward(saved_k_normed, saved_k_post_rope);
            }
            else
            {
                rope_q_helper_.forward(saved_q_pre_rope, saved_q_post_rope);
                rope_k_helper_.forward(saved_k_pre_rope, saved_k_post_rope);
            }

            // Step 3.5: prefill-mode cache write (K pre-RoPE and V).
            // Only effective non-padded positions are kept; padding is
            // stripped by adjusting the source row offset.
            if (mode == network_context::inference_mode::prefill)
            {
                const long pad_len = network_context::get_padding_length(0);
                const long eff_len = std::max<long>(0, N - pad_len);

                if (eff_len > 0)
                {
                    ensure_cache_allocated(B, eff_len);

                    tt::copy_tensor(false, K_cache,
                        /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                        saved_k_pre_rope,
                        /*sk=*/0, /*snr=*/static_cast<size_t>(pad_len), /*snc=*/0,
                        /*k=*/NUM_KV_HEADS,
                        /*nr=*/static_cast<size_t>(eff_len),
                        /*nc=*/HEAD_DIM);

                    tt::copy_tensor(false, V_cache,
                        /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                        saved_v_4d,
                        /*sk=*/0, /*snr=*/static_cast<size_t>(pad_len), /*snc=*/0,
                        /*k=*/NUM_KV_HEADS,
                        /*nr=*/static_cast<size_t>(eff_len),
                        /*nc=*/HEAD_DIM);

                    cache_filled_len_ = eff_len;
                }
                else
                {
                    cache_filled_len_ = 0;
                }
            }

            // Step 4: GQA repeat
            saved_k_repeated.set_size(B, NUM_HEADS, N, HEAD_DIM);
            saved_v_repeated.set_size(B, NUM_HEADS, N, HEAD_DIM);
            for (long kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
            {
                for (long r = 0; r < REPEAT_FACTOR; ++r)
                {
                    const size_t dst_h = static_cast<size_t>(kv_h * REPEAT_FACTOR + r);
                    tt::copy_tensor(false, saved_k_repeated, dst_h,
                        saved_k_post_rope, static_cast<size_t>(kv_h), 1);
                    tt::copy_tensor(false, saved_v_repeated, dst_h,
                        saved_v_4d, static_cast<size_t>(kv_h), 1);
                }
            }

            // Step 5: Q @ K^T
            saved_scores.set_size(B, NUM_HEADS, N, N);
            tt::gemm(0.0f, saved_scores, 1.0f,
                saved_q_post_rope, false,
                saved_k_repeated, true,
                operation_mode::PLANE_WISE);

            // Step 6: causal mask
            resizable_tensor scores_masked;
            scores_masked.copy_size(saved_scores);
            tril_helper_.forward(saved_scores, scores_masked);

            // Step 7: softmax
            saved_attn.copy_size(scores_masked);
            tt::softmax(saved_attn, scores_masked, operation_mode::PLANE_WISE);

            // Step 8: attn @ V
            resizable_tensor ctx_4d(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, ctx_4d, 1.0f,
                saved_attn, false,
                saved_v_repeated, false,
                operation_mode::PLANE_WISE);

            // Step 9: merge heads
            saved_ctx_flat.set_size(B, 1, N, Q_PROJ_DIM);
            tt::merge_heads(false, saved_ctx_flat, ctx_4d);

            // Step 10: W_O projection
            output.set_size(B, 1, N, EMBEDDING_DIM);
            {
                alias_tensor ctx_2d(B * N, Q_PROJ_DIM);
                auto ctx_view = ctx_2d(saved_ctx_flat, 0);

                alias_tensor o_2d(B * N, EMBEDDING_DIM);
                auto o_view = o_2d(output, 0);

                tt::gemm(0.0f, o_view, 1.0f, ctx_view, false, wo, false);
            }

            saved_x_shape_B = B;
            saved_x_shape_N = N;
        }

        // Incremental forward path.
        // Input  : x of shape [1, 1, 1, EMBEDDING_DIM]
        // Output : [1, 1, 1, EMBEDDING_DIM]
        //
        // Workflow:
        //   1. Project the new token into Q/K/V (no RoPE yet).
        //   2. If the cache is full, shift it left to make room.
        //   3. Append the new K_pre_rope and V to the cache.
        //   4. Build a K window of length L = cache_filled_len_ from the cache.
        //   5. Apply RoPE to the entire K window with positions [0, L-1],
        //      and to Q with position L-1.
        //   6. GQA repeat, attention, and W_O projection.
        void forward_incremental(const tensor& x, resizable_tensor& output)
        {
            const long B = x.num_samples();
            DLIB_CASSERT(B == 1, "Incremental mode supports B=1 only");
            DLIB_CASSERT(x.nr() == 1, "Incremental mode expects exactly one new position");
            DLIB_CASSERT(cache_capacity_ > 0, "KV cache must be allocated before incremental");

            auto wq = wq_alias(params, wq_offset);
            auto wk = wk_alias(params, wk_offset);
            auto wv = wv_alias(params, wv_offset);
            auto wo = wo_alias(params, wo_offset);

            const float qk_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

            // Step 1: project the new token into Q, K, V (no RoPE)
            inc_q.set_size(1, NUM_HEADS, 1, HEAD_DIM);
            inc_k.set_size(1, NUM_KV_HEADS, 1, HEAD_DIM);
            inc_v.set_size(1, NUM_KV_HEADS, 1, HEAD_DIM);
            {
                alias_tensor x_2d(1, EMBEDDING_DIM);
                auto x_view = x_2d(const_cast<tensor&>(x), 0);

                alias_tensor q_2d(1, Q_PROJ_DIM);
                auto q_view = q_2d(inc_q, 0);
                tt::gemm(0.0f, q_view, qk_scale, x_view, false, wq, false);

                alias_tensor k_2d(1, KV_PROJ_DIM);
                auto k_view = k_2d(inc_k, 0);
                tt::gemm(0.0f, k_view, 1.0f, x_view, false, wk, false);

                alias_tensor v_2d(1, KV_PROJ_DIM);
                auto v_view = v_2d(inc_v, 0);
                tt::gemm(0.0f, v_view, 1.0f, x_view, false, wv, false);
            }

            // Step 2: make room in the cache if full (sliding window)
            if (cache_filled_len_ >= cache_capacity_)
                shift_cache_left();

            // Step 3: append the new K_pre_rope and V to the cache
            const long insert_pos = cache_filled_len_;
            tt::copy_tensor(false, K_cache,
                /*dk=*/0, /*dnr=*/static_cast<size_t>(insert_pos), /*dnc=*/0,
                inc_k,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS, /*nr=*/1, /*nc=*/HEAD_DIM);
            tt::copy_tensor(false, V_cache,
                /*dk=*/0, /*dnr=*/static_cast<size_t>(insert_pos), /*dnc=*/0,
                inc_v,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS, /*nr=*/1, /*nc=*/HEAD_DIM);
            cache_filled_len_ = insert_pos + 1;

            const long L = cache_filled_len_;

            // Step 4: extract the K and V windows of length L
            inc_k_window.set_size(1, NUM_KV_HEADS, L, HEAD_DIM);
            inc_v_window.set_size(1, NUM_KV_HEADS, L, HEAD_DIM);
            tt::copy_tensor(false, inc_k_window,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                K_cache,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(L),
                /*nc=*/HEAD_DIM);
            tt::copy_tensor(false, inc_v_window,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                V_cache,
                /*sk=*/0, /*snr=*/0, /*snc=*/0,
                /*k=*/NUM_KV_HEADS,
                /*nr=*/static_cast<size_t>(L),
                /*nc=*/HEAD_DIM);

            // Step 5a: optional QK-Norm on the K window, then RoPE (positions [0, L-1])
            inc_k_window_rope.copy_size(inc_k_window);
            if (USE_QK_NORM)
            {
                auto gk = gamma_k_alias(params, gamma_k_offset);
                tt::rms_normalize(qk_norm_eps_, inc_k_window_normed, inc_scale_k, inc_k_window, gk, true);
                rope_k_helper_.forward(inc_k_window_normed, inc_k_window_rope);
            }
            else
            {
                rope_k_helper_.forward(inc_k_window, inc_k_window_rope);
            }

            // Step 5b: apply RoPE on Q with position L-1, by extracting cos/sin
            // slices from the Q helper's caches and rotating in place
            const long half_d = HEAD_DIM / 2;
            rope_q_helper_.get().ensure_caches_at_least(L, HEAD_DIM);
            const auto& q_cos = rope_q_helper_.get().get_cos_cache();
            const auto& q_sin = rope_q_helper_.get().get_sin_cache();

            resizable_tensor q_cos_slice(1, 1, 1, half_d);
            resizable_tensor q_sin_slice(1, 1, 1, half_d);
            tt::copy_tensor(false, q_cos_slice,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                q_cos,
                /*sk=*/0, /*snr=*/static_cast<size_t>(L - 1), /*snc=*/0,
                /*k=*/1, /*nr=*/1, /*nc=*/static_cast<size_t>(half_d));
            tt::copy_tensor(false, q_sin_slice,
                /*dk=*/0, /*dnr=*/0, /*dnc=*/0,
                q_sin,
                /*sk=*/0, /*snr=*/static_cast<size_t>(L - 1), /*snc=*/0,
                /*k=*/1, /*nr=*/1, /*nc=*/static_cast<size_t>(half_d));

            inc_q_rope.set_size(1, NUM_HEADS, 1, HEAD_DIM);
            if (USE_QK_NORM)
            {
                auto gq = gamma_q_alias(params, gamma_q_offset);
                tt::rms_normalize(qk_norm_eps_, inc_q_normed, inc_scale_q, inc_q, gq, true);
                tt::copy_tensor(false, inc_q_rope, 0, inc_q_normed, 0, NUM_HEADS);
            }
            else
            {
                tt::copy_tensor(false, inc_q_rope, 0, inc_q, 0, NUM_HEADS);
            }
            tt::apply_rotary_positional_embedding(false, inc_q_rope, q_cos_slice, q_sin_slice);

            // Step 6: GQA repeat from NUM_KV_HEADS to NUM_HEADS
            inc_k_window_rep.set_size(1, NUM_HEADS, L, HEAD_DIM);
            inc_v_window_rep.set_size(1, NUM_HEADS, L, HEAD_DIM);
            for (long kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
            {
                for (long r = 0; r < REPEAT_FACTOR; ++r)
                {
                    const size_t dst_h = static_cast<size_t>(kv_h * REPEAT_FACTOR + r);
                    tt::copy_tensor(false, inc_k_window_rep, dst_h,
                        inc_k_window_rope, static_cast<size_t>(kv_h), 1);
                    tt::copy_tensor(false, inc_v_window_rep, dst_h,
                        inc_v_window, static_cast<size_t>(kv_h), 1);
                }
            }

            // Step 7: scores = Q @ K_window^T (no causal mask needed)
            inc_scores.set_size(1, NUM_HEADS, 1, L);
            tt::gemm(0.0f, inc_scores, 1.0f, inc_q_rope, false, inc_k_window_rep, true,
                operation_mode::PLANE_WISE);

            // Step 8: softmax
            inc_attn.set_size(1, NUM_HEADS, 1, L);
            tt::softmax(inc_attn, inc_scores, operation_mode::PLANE_WISE);

            // Step 9: ctx = attn @ V_window
            inc_ctx.set_size(1, NUM_HEADS, 1, HEAD_DIM);
            tt::gemm(0.0f, inc_ctx, 1.0f, inc_attn, false, inc_v_window_rep, false,
                operation_mode::PLANE_WISE);

            // Step 10: merge heads + W_O projection
            resizable_tensor ctx_flat(1, 1, 1, Q_PROJ_DIM);
            tt::merge_heads(false, ctx_flat, inc_ctx);

            output.set_size(1, 1, 1, EMBEDDING_DIM);
            {
                alias_tensor ctx_2d(1, Q_PROJ_DIM);
                auto ctx_view = ctx_2d(ctx_flat, 0);

                alias_tensor o_2d(1, EMBEDDING_DIM);
                auto o_view = o_2d(output, 0);

                tt::gemm(0.0f, o_view, 1.0f, ctx_view, false, wo, false);
            }

            saved_x_shape_B = 1;
            saved_x_shape_N = 1;
        }
    };

    template <long EMBEDDING_DIM, long NUM_HEADS, long NUM_KV_HEADS, long HEAD_DIM, typename SUBNET,
        bool USE_QK_NORM = false>
    using gqa_attention = add_layer<
        gqa_attention_<EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, USE_QK_NORM>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    namespace gqa_transformer
    {
        // Query projection: projects to d_model then reshapes to (num_heads, seq_len, head_dim)
        template <long d_model, long num_heads, typename SUBNET>
        using query = reshape_to<num_heads, -1, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        // Key projection for GQA: uses fewer heads (num_kv_heads) than queries
        template <long num_kv_heads, long head_dim, typename SUBNET>
        using key = reshape_to<num_kv_heads, -1, head_dim,
            linear_no_bias<num_kv_heads* head_dim, SUBNET>>;

        // Value projection for GQA: same head count as keys
        template <long num_kv_heads, long head_dim, typename SUBNET>
        using value = reshape_to<num_kv_heads, -1, head_dim,
            linear_no_bias<num_kv_heads* head_dim, SUBNET>>;

        // Grouped Query Attention: K/V heads are repeated to match Q head count
        // Reduces memory bandwidth while maintaining model quality
        template <long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        using multihead_attention_gqa =
            linear_no_bias<d_model, reshape_to<1, -1, d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            // Q: apply RoPE after projection
            rope<query<d_model, num_heads, skip2<
            tag4<transpose<
            // K: project, apply RoPE, then repeat heads to match Q
            repeat_heads<num_heads / num_kv_heads,
            rope<key<num_kv_heads, d_model / num_heads, skip2<
            // V: project and repeat heads (no positional encoding)
            tag3<repeat_heads<num_heads / num_kv_heads,
            value<num_kv_heads, d_model / num_heads,
            tag2<SUBNET>>>>>>>>>>>>>>>>>>>>;

        // Transformer block with pre-norm architecture
        // Attention sublayer followed by SwiGLU feed-forward, each with residual connections
        template <long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        using transformer_block =
            add_prev5<act_steps<swiglu<d_model, 8, 3, input_tensor>, 4, rms_norm<tag5<
            add_prev1<multihead_attention_gqa<d_model, num_heads, num_kv_heads, rms_norm<tag1<
            SUBNET>>>>>>>>;

        // Recursive template to stack N transformer blocks
        template<long remaining_layers, long d_model, long num_heads, long num_kv_heads,
            typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<d_model, num_heads, num_kv_heads,
                typename transformer_stack_impl<remaining_layers - 1, d_model, num_heads, num_kv_heads, SUBNET>::type>;
        };

        template<long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        struct transformer_stack_impl<0, d_model, num_heads, num_kv_heads, SUBNET, void>
        {
            using type = tag10<SUBNET>;
        };

        template<long num_layers, long d_model, long num_heads, long num_kv_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, d_model, num_heads, num_kv_heads, SUBNET>::type;

    } // namespace gqa_transformer

    // ----------------------------------------------------------------------------------------

    // Multi-head Grouped Query Attention, fused into a single layer
    namespace gqa_transformer_unified
    {        
        template<long d_model, long num_heads, long num_kv_heads, typename SUBNET,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using multihead_attention_gqa = gqa_attention<d_model, num_heads, num_kv_heads,
            head_dim, SUBNET, use_qk_norm>;

        // Transformer block with pre-norm architecture, identical in topology to
        // the chained version: attention sublayer with residual connection,
        // followed by SwiGLU feed-forward sublayer with residual connection.
        //
        // The only structural difference vs. gqa_transformer::transformer_block
        // is that the attention sublayer is now a single layer (gqa_attention_)
        // rather than a chain of 16 layers. The residual connections (add_prev1
        // and add_prev5), the RMS normalizations, and the SwiGLU feed-forward
        // are unchanged.
 
        // Feed-forward sublayer selector. With UseAct=true the SwiGLU transition is wrapped
        // in an Adaptive Computation Time layer (per-position latent recurrence); with
        // UseAct=false it is a plain SwiGLU. Networks that already recur at a higher level
        // (e.g. HRM, which loops the whole stack) build with UseAct=false to avoid nesting
        // two recurrence mechanisms.
        template<bool UseAct, long d_model, long hidden_num, long hidden_den, long max_steps, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear>
        struct ffn_selector
        {
            using type = act_steps<swiglu<d_model, hidden_num, hidden_den, input_tensor, LINEAR>, max_steps, SUBNET>;
        };
        template<long d_model, long hidden_num, long hidden_den, long max_steps, typename SUBNET,
            template <unsigned long, typename> class LINEAR>
        struct ffn_selector<false, d_model, hidden_num, hidden_den, max_steps, SUBNET, LINEAR>
        {
            using type = swiglu<d_model, hidden_num, hidden_den, SUBNET, LINEAR>;
        };
        template<bool UseAct, long d_model, long hidden_num, long hidden_den, long max_steps, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear>
        using ffn_sublayer = typename ffn_selector<UseAct, d_model, hidden_num, hidden_den, max_steps, SUBNET, LINEAR>::type;

        // Transformer block with pre-norm architecture. Attention sublayer is the fused
        // gqa_attention_ layer; feed-forward sublayer is SwiGLU, optionally ACT-wrapped.
        template<bool UseAct, long d_model, long num_heads, long num_kv_heads,
            long hidden_num, long hidden_den, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using transformer_block =
            add_prev5<ffn_sublayer<UseAct, d_model, hidden_num, hidden_den, 4, rms_norm<tag5<
            add_prev1<multihead_attention_gqa<d_model, num_heads, num_kv_heads, rms_norm<tag1<
            SUBNET>>, head_dim, use_qk_norm>>>>, LINEAR >> ;

        template<long remaining_layers, bool UseAct, long d_model, long num_heads, long num_kv_heads,
            long hidden_num, long hidden_den, typename SUBNET,
            template <unsigned long, typename> class LINEAR = linear,
            long head_dim = d_model / num_heads, bool use_qk_norm = false, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<UseAct, d_model, num_heads, num_kv_heads, hidden_num, hidden_den,
                typename transformer_stack_impl<remaining_layers - 1, UseAct, d_model, num_heads,
                num_kv_heads, hidden_num, hidden_den, SUBNET, LINEAR, head_dim, use_qk_norm>::type,
                LINEAR, head_dim, use_qk_norm>;
        };
        template<bool UseAct, long d_model, long num_heads, long num_kv_heads,
            long hidden_num, long hidden_den, typename SUBNET,
            template <unsigned long, typename> class LINEAR, long head_dim, bool use_qk_norm>
        struct transformer_stack_impl<0, UseAct, d_model, num_heads, num_kv_heads,
            hidden_num, hidden_den, SUBNET, LINEAR, head_dim, use_qk_norm, void>
        {
            using type = tag10<SUBNET>;
        };

        // UseAct trails with a default of true so existing dense configurations that omit it
        // keep their ACT feed-forward unchanged.
        template<long num_layers, long d_model, long num_heads, long num_kv_heads, typename SUBNET,
            bool UseAct = true, long hidden_num = 8, long hidden_den = 3,
            template <unsigned long, typename> class LINEAR = linear,
            long head_dim = d_model / num_heads, bool use_qk_norm = false>
        using transformer_stack = typename transformer_stack_impl<num_layers, UseAct, d_model,
            num_heads, num_kv_heads, hidden_num, hidden_den, SUBNET, LINEAR, head_dim, use_qk_norm>::type;

    } // namespace gqa_transformer_unified

    // ----------------------------------------------------------------------------------------

    // CANONICAL TRANSFORMER ARCHITECTURE
    namespace canonical_transformer
    {

        template <long d_model, long num_heads, typename SUBNET>
        using query = reshape_to<num_heads, -1, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <long d_model, long num_heads, typename SUBNET>
        using key = reshape_to<num_heads, -1, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <long d_model, long num_heads, typename SUBNET>
        using value = reshape_to<num_heads, -1, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention =
            DO<linear_no_bias<d_model, reshape_to<1, -1, d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            rope<query<d_model, num_heads, skip2<
            tag4<transpose<
            rope<key<d_model, num_heads, skip2<
            tag3<value<d_model, num_heads,
            tag2<SUBNET>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            add_prev5<ffn<ACT, d_model, 4, rms_norm<tag5<
            add_prev1<multihead_attention<DO, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>>>>;

        template<long remaining_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads,
            typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<ACT, DO, d_model, num_heads,
                typename transformer_stack_impl<remaining_layers - 1, ACT, DO, d_model, num_heads, SUBNET>::type>;
        };

        template<template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        struct transformer_stack_impl<0, ACT, DO, d_model, num_heads, SUBNET, void>
        {
            using type = tag10<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, d_model, num_heads, SUBNET>::type;

    } // namespace canonical_transformer

    // FUSED TRANSFORMER ARCHITECTURE
    namespace fused_transformer
    {

        template <long num_heads, long d_model, typename SUBNET>
        using query = extract<0, num_heads, d_model / num_heads, 1, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using key = extract<d_model, num_heads, 1, d_model / num_heads, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using value = extract<(d_model * 2), num_heads, d_model / num_heads, 1, SUBNET>;

        template <template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention =
            DO<extract<0, 1, 1, d_model, fc_no_bias<d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            query<num_heads, d_model, skip2<
            tag4<key<num_heads, d_model, skip2<
            tag3<value<num_heads, d_model,
            tag2<fc_no_bias<d_model * 3,
            SUBNET>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using fused_ffn = extract<0, 1, 1, d_model,
            DO<fc<d_model, ACT<fc<d_model * 4, SUBNET>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            add_prev5<fused_ffn<ACT, DO, d_model, rms_norm<tag5<
            add_prev1<multihead_attention<DO, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>>>>;

        template<long remaining_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<ACT, DO, d_model, num_heads,
                typename transformer_stack_impl<remaining_layers - 1, ACT, DO, d_model, num_heads, SUBNET>::type>;
        };

        template<template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        struct transformer_stack_impl<0, ACT, DO, d_model, num_heads, SUBNET, void>
        {
            using type = tag10<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, d_model, num_heads, SUBNET>::type;

    } // namespace fused_transformer

    // Default to canonical transformer implementation
    using namespace gqa_transformer;

    // ----------------------------------------------------------------------------------------

    /* HIERARCHICAL REASONING MODEL(HRM)
     *
     * Implements the dual-recurrent architecture from:
     *   Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734
     *
     * Architecture:
     *   - H-module (high-level): slow cycles for abstract planning (N updates)
     *   - L-module (low-level): fast iterations for detailed computation (T per H-cycle)
     *   - Total recurrent depth: N * T steps
     *   - 1-step gradient approximation for O(1) memory training
     *
     * Gradient path (1-step approximation):
     *   loss -> output -> final H-module -> final L-module -> input embedding
     */
     
    template<
        typename H_NET,
        typename L_NET,
        int N,
        int T
    >
    class hrm_
    {
        static_assert(N > 0, "N (high-level cycles) must be positive");
        static_assert(T > 0, "T (low-level timesteps per cycle) must be positive");

    public:
        using h_net_type = H_NET;
        using l_net_type = L_NET;

        explicit hrm_() :
            hidden_dim(0),
            learning_rate_multiplier(1.0),
            current_learning_rate_(0.001),
            solvers_initialized_(false)
        {
        }

        hrm_(const hrm_& other) :
            h_net(other.h_net),
            l_net(other.l_net),
            z_h_init(other.z_h_init),
            z_l_init(other.z_l_init),
            hidden_dim(other.hidden_dim),
            learning_rate_multiplier(other.learning_rate_multiplier),
            current_learning_rate_(other.current_learning_rate_),
            solvers_initialized_(false)
        {
        }

        hrm_& operator=(const hrm_& other)
        {
            if (this != &other) {
                h_net = other.h_net;
                l_net = other.l_net;
                z_h_init = other.z_h_init;
                z_l_init = other.z_l_init;
                hidden_dim = other.hidden_dim;
                learning_rate_multiplier = other.learning_rate_multiplier;
                current_learning_rate_ = other.current_learning_rate_;
                solvers_initialized_ = false;
            }
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            const tensor& input = sub.get_output();
            hidden_dim = input.nc();
            init_hidden_states();
        }

        /* FORWARD: Hierarchical recurrence with 1 - step gradient approximation
         *
         * From the paper (Section 2, Figure 4):
         *   z_H, z_L = z_init
         *   for _i in range(N*T - 1):  # no gradient
         *     z_L = f_L(z_L + z_H + x)
         *     if (_i+1) % T == 0:
         *       z_H = f_H(z_H + z_L)
         *   # 1-step grad (final L then final H)
         *   z_L = f_L(z_L + z_H + x)
         *   z_H = f_H(z_H + z_L)
         *   output = z_H
         */
        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& x = sub.get_output();
            const long batch_size = x.num_samples();
            const long k = x.k();
            const long seq_len = x.nr();

            // Initialize z_H and z_L from their init vectors (broadcast to batch)
            z_h_current.copy_size(x);
            z_l_current.copy_size(x);
            {
                auto* z_h_ptr = z_h_current.host();
                auto* z_l_ptr = z_l_current.host();
                const auto* h_init_ptr = z_h_init.host();
                const auto* l_init_ptr = z_l_init.host();

                for (long n = 0; n < batch_size; ++n) {
                    for (long kk = 0; kk < k; ++kk) {
                        for (long r = 0; r < seq_len; ++r) {
                            for (long c = 0; c < hidden_dim; ++c) {
                                const long idx = ((n * k + kk) * seq_len + r) * hidden_dim + c;
                                z_h_ptr[idx] = h_init_ptr[c];
                                z_l_ptr[idx] = l_init_ptr[c];
                            }
                        }
                    }
                }
            }

            // Recurrent iterations without gradient tracking (all except final L+H)
            for (int n = 0; n < N; ++n)
            {
                for (int t = 0; t < T; ++t)
                {
                    // Skip the very last L-step (it will be done with gradient below)
                    if (n == N - 1 && t == T - 1) continue;

                    // L-module input: z_L + z_H + x
                    l_input.copy_size(x);
                    tt::copy_tensor(false, l_input, 0, z_l_current, 0, z_l_current.k());
                    tt::add(1.0f, l_input, 1.0f, z_h_current);
                    tt::add(1.0f, l_input, 1.0f, x);

                    l_net.forward(l_input);
                    const tensor& l_out = l_net.get_output();
                    tt::copy_tensor(false, z_l_current, 0, l_out, 0, l_out.k());
                }

                // H-module update at end of each cycle (except the last one)
                if (n == N - 1) continue;

                // H-module input: z_H + z_L
                h_input.copy_size(x);
                tt::copy_tensor(false, h_input, 0, z_h_current, 0, z_h_current.k());
                tt::add(1.0f, h_input, 1.0f, z_l_current);

                h_net.forward(h_input);
                const tensor& h_out = h_net.get_output();
                tt::copy_tensor(false, z_h_current, 0, h_out, 0, h_out.k());
            }

            // Final L-Module update (WITH gradient tracking)
            // Save input for backward: last_l_input = z_L + z_H + x
            last_l_input.copy_size(x);
            tt::copy_tensor(false, last_l_input, 0, z_l_current, 0, z_l_current.k());
            tt::add(1.0f, last_l_input, 1.0f, z_h_current);
            tt::add(1.0f, last_l_input, 1.0f, x);

            l_net.forward(last_l_input);
            const tensor& l_final = l_net.get_output();

            // Final H-Module update with gradient tracking
            // Save input for backward: last_h_input = z_H + z_L_final
            last_h_input.copy_size(x);
            tt::copy_tensor(false, last_h_input, 0, z_h_current, 0, z_h_current.k());
            tt::add(1.0f, last_h_input, 1.0f, l_final);

            h_net.forward(last_h_input);
            const tensor& h_final = h_net.get_output();

            // Output is the final H state
            output.copy_size(h_final);
            tt::copy_tensor(false, output, 0, h_final, 0, h_final.k());
        }

        /* BACKWARD: 1 - step gradient approximation
         *
         * Gradient path:
         *   dL/d(output) = gradient_input
         *     -> backprop through H: dL/d(last_h_input) = h_net.get_final_data_gradient()
         *     -> backprop through L: dL/d(last_l_input) = l_net.get_final_data_gradient()
         *     -> propagate to input x (component of last_l_input)
         */
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            // Step 1: backprop through final H-Module
            // Requires: h_net.forward(last_h_input) was the most recent forward call
            h_net.back_propagate_error(last_h_input, gradient_input);
            const tensor& grad_from_h = h_net.get_final_data_gradient();

            // Step 2: backprop through final L-Module
            // grad_from_h flows to z_L_final (component of last_h_input = z_H + z_L_final)
            // Requires: l_net.forward(last_l_input) was called just before h_net.forward()
            l_net.back_propagate_error(last_l_input, grad_from_h);
            const tensor& grad_from_l = l_net.get_final_data_gradient();

            // Step 3: propagate gradient to input x
            // last_l_input = z_L + z_H + x, so dL/dx = dL/d(last_l_input)
            // (z_L and z_H are treated as constants in the 1-step approximation)
            tensor& prev_grad = sub.get_gradient_input();
            tt::add(1.0f, prev_grad, 1.0f, grad_from_l);

            // Step 4: update sub-network parameters using internal solvers
            const bool in_training = !network_context::is_active() ||                
                network_context::is_training();
            if (in_training) update_subnet_parameters();
        }

        void set_learning_rate(double lr)
        {
            current_learning_rate_ = lr;
            set_all_learning_rates(h_net, lr);
            set_all_learning_rates(l_net, lr);
        }
        double get_learning_rate() const { return current_learning_rate_; }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;
            set_all_learning_rate_multipliers(h_net, val);
            set_all_learning_rate_multipliers(l_net, val);
        }
        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }

        void configure_solvers(double weight_decay, double beta1, double beta2)
        {
            solver_weight_decay_ = weight_decay;
            solver_beta1_ = beta1;
            solver_beta2_ = beta2;
            solvers_initialized_ = false;
        }

        size_t internal_parameters() const
        {
            return count_parameters(h_net) + count_parameters(l_net);
        }

        size_t active_parameters() const
        {
            return internal_parameters();
        }

        void clean()
        {
            clean_subnet(h_net);
            clean_subnet(l_net);
        }

        const h_net_type& get_h_net() const { return h_net; }
        const l_net_type& get_l_net() const { return l_net; }
        h_net_type& get_h_net() { return h_net; }
        l_net_type& get_l_net() { return l_net; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const hrm_& item, std::ostream& out)
        {
            serialize("hrm_", out);
            serialize(item.h_net, out);
            serialize(item.l_net, out);
            serialize(item.z_h_init, out);
            serialize(item.z_l_init, out);
            serialize(item.hidden_dim, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.current_learning_rate_, out);
            serialize(item.h_solvers_, out);
            serialize(item.l_solvers_, out);
            serialize(item.solvers_initialized_, out);
        }

        friend void deserialize(hrm_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "hrm_")
                throw serialization_error("Unexpected version '" + version +
                    "' while deserializing hrm_");

            deserialize(item.h_net, in);
            deserialize(item.l_net, in);
            deserialize(item.z_h_init, in);
            deserialize(item.z_l_init, in);
            deserialize(item.hidden_dim, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.current_learning_rate_, in);
            deserialize(item.h_solvers_, in);
            deserialize(item.l_solvers_, in);
            deserialize(item.solvers_initialized_, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const hrm_& item)
        {
            out << "hrm (N=" << N << ", T=" << T << ")"
                << " learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const hrm_& item, std::ostream& out)
        {
            out << "<hrm"
                << " N='" << N << "'"
                << " T='" << T << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier << "'"
                << ">\n";
            out << "  <h_module>\n";
            to_xml(item.h_net, out);
            out << "  </h_module>\n";
            out << "  <l_module>\n";
            to_xml(item.l_net, out);
            out << "  </l_module>\n";
            out << "</hrm>\n";
        }

    private:

        // Update both H and L network parameters using internal AdamW solvers
        void update_subnet_parameters()
        {
            if (network_context::is_active())
                current_learning_rate_ = network_context::get_learning_rate();
            if (learning_rate_multiplier == 0.0) return;

            const double effective_lr = current_learning_rate_ * learning_rate_multiplier;

            if (!solvers_initialized_) {
                const double wd = network_context::is_active()
                    ? network_context::get_optimizer_weight_decay() : solver_weight_decay_;
                const double b1 = network_context::is_active()
                    ? network_context::get_optimizer_beta1() : solver_beta1_;
                const double b2 = network_context::is_active()
                    ? network_context::get_optimizer_beta2() : solver_beta2_;
                h_solvers_.assign(h_net.num_computational_layers, adamw(wd, b1, b2));
                l_solvers_.assign(l_net.num_computational_layers, adamw(wd, b1, b2));
                solvers_initialized_ = true;
            }

            h_net.update_parameters(h_solvers_, effective_lr);
            l_net.update_parameters(l_solvers_, effective_lr);
        }

        void init_hidden_states()
        {
            z_h_init.set_size(1, 1, 1, hidden_dim);
            z_l_init.set_size(1, 1, 1, hidden_dim);

            auto* h_ptr = z_h_init.host();
            auto* l_ptr = z_l_init.host();
            for (long c = 0; c < hidden_dim; ++c) {
                h_ptr[c] = 0.0f;
                l_ptr[c] = 0.0f;
            }
        }

        template<typename NET>
        auto clean_subnet(NET& net) -> decltype(net.clean(), void()) { net.clean(); }
        template<typename NET>
        void clean_subnet(...) {}

        // Internal recurrent modules
        h_net_type h_net;
        l_net_type l_net;

        // Initial hidden states (truncated normal, kept fixed during training)
        resizable_tensor z_h_init;
        resizable_tensor z_l_init;

        long hidden_dim;
        double learning_rate_multiplier;
        double current_learning_rate_;

        // Solver configuration
        double solver_weight_decay_ = 0.004;
        double solver_beta1_ = 0.9;
        double solver_beta2_ = 0.999;

        // Internal solvers for sub-network parameter updates
        std::vector<adamw> h_solvers_;
        std::vector<adamw> l_solvers_;
        bool solvers_initialized_;

        // Temporary computation tensors (forward pass)
        resizable_tensor z_h_current;
        resizable_tensor z_l_current;
        resizable_tensor h_input;
        resizable_tensor l_input;

        // Saved inputs for backward pass (final step only, O(1) memory)
        resizable_tensor last_h_input;
        resizable_tensor last_l_input;

        resizable_tensor params;  // Unused
    };

    template<typename H_NET, typename L_NET, int N, int T, typename SUBNET>
    using hrm = add_layer<hrm_<H_NET, L_NET, N, T>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    /* MIXTURE OF EXPERTS(MoE) LAYER WITH INTERNAL GATE
     *
     * Implements sparse conditional computation via per-sample expert routing.
     * Each input sample is independently routed to top-k experts based on a learned
     * gating function, enabling increased model capacity without proportional FLOPs.
     *
     * Architecture (Switch Transformer / ST-MoE inspired):
     *   - Internal gate network produces routing logits from pooled input
     *   - Softmax with exploration noise selects top-k experts per sample
     *   - Capacity factor limits per-expert load, overflow goes to next-best expert
     *   - Expert outputs are weighted by renormalized gate probabilities
     *
     * Gate gradient combines three signals:
     *   - Task gradient: dL/dP via differentiable softmax weight (output = w * expert(x))
     *   - Load balance loss: alpha * N * sum(f_e * P_bar_e), Switch Transformer formulation
     *   - Router z-loss: penalizes large logits for numerical stability (ST-MoE)
     *
     * The load_balance_weight decays exponentially from its initial value toward
     * load_balance_floor over training steps. This provides strong balancing pressure
     * early on (preventing expert collapse), then relaxes to allow specialization.
     */

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
        static_assert(n_experts_ > 0, "Number of experts must be positive");

    public:
        using expert_net_type = EXPERT_NET;
        using gate_net_type = GATE_NET;

        explicit moe_() :
            n_experts(n_experts_),
            noise_scale(0.5f),
            top_k(1),
            capacity_factor(1.5f),
            usage_update_rate(0.05f),
            load_balance_weight(0.1f),
            load_balance_floor_(0.01f),
            load_balance_decay_(0.9999f),
            z_loss_weight(0.001f),
            learning_rate_multiplier(1.0),
            current_learning_rate_(1e-4),
            solvers_initialized_(false),
            cached_batch_size_(0),
            load_balance_loss_(0.0f)
        {
            if (top_e == 0)
                top_k = std::max(1L, static_cast<long>(std::floor(n_experts * 0.2f)));
            else
                top_k = std::min(static_cast<long>(top_e), n_experts);
        }

        moe_(const moe_& other) :
            gate_net(other.gate_net),
            n_experts(other.n_experts),
            noise_scale(other.noise_scale),
            top_k(other.top_k),
            capacity_factor(other.capacity_factor),
            usage_update_rate(other.usage_update_rate),
            load_balance_weight(other.load_balance_weight),
            load_balance_floor_(other.load_balance_floor_),
            load_balance_decay_(other.load_balance_decay_),
            z_loss_weight(other.z_loss_weight),
            learning_rate_multiplier(other.learning_rate_multiplier),
            current_learning_rate_(other.current_learning_rate_),
            expert_usage(other.expert_usage),
            solvers_initialized_(false),
            cached_batch_size_(0),
            load_balance_loss_(0.0f)
        {
            experts.reserve(other.experts.size());
            for (const auto& expert : other.experts) experts.push_back(expert);
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                gate_net = other.gate_net;
                n_experts = other.n_experts;
                noise_scale = other.noise_scale;
                top_k = other.top_k;
                capacity_factor = other.capacity_factor;
                usage_update_rate = other.usage_update_rate;
                load_balance_weight = other.load_balance_weight;
                load_balance_floor_ = other.load_balance_floor_;
                load_balance_decay_ = other.load_balance_decay_;
                z_loss_weight = other.z_loss_weight;
                learning_rate_multiplier = other.learning_rate_multiplier;
                current_learning_rate_ = other.current_learning_rate_;
                expert_usage = other.expert_usage;
                solvers_initialized_ = false;
                expert_solvers_.clear();
                gate_solvers_.clear();
                cached_batch_size_ = 0;
                load_balance_loss_ = 0.0f;
                experts.clear();
                experts.reserve(other.experts.size());
                for (const auto& expert : other.experts) experts.push_back(expert);
            }
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            if (experts.empty()) {
                expert_usage.resize(n_experts, 0.0f);
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i) experts.emplace_back(EXPERT_NET{});
                solvers_initialized_ = false;
            }
        }

        // Forward: gate routing, batched expert processing, weighted scatter
        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const long num_samples = expert_input.num_samples();
            const long k = expert_input.k();
            const long nr = expert_input.nr();
            const long nc = expert_input.nc();
            const long sample_size = k * nr * nc;
            const bool is_training =
                std::is_same<MODE, training_mode_tag>::value;

            // Cache input for gate and expert backward
            cached_input_.copy_size(expert_input);
            tt::copy_tensor(false, cached_input_, 0, expert_input, 0, k);

            // Gate forward: raw logits
            gate_net.forward(cached_input_);
            const tensor& gate_output = gate_net.get_output();
            const float* gate_data = gate_output.host();

            output.copy_size(expert_input);
            output = 0;

            // Clamp raw logits before noise to bound the gate's learned
            // range, then add exploration noise WITHOUT re-clamping so the
            // noise can freely break ties and prevent routing collapse.
            const float logit_clamp = 3.0f;
            const long expert_capacity = std::max(1L,
                static_cast<long>(std::ceil( num_samples * top_k * capacity_factor / n_experts)));

            std::vector<std::vector<std::pair<long, float>>> expert_assignments(n_experts);
            std::vector<float> routing_fraction(n_experts, 0.0f);
            std::vector<float> gate_prob_sum(n_experts, 0.0f);

            if (is_training) {
                cached_batch_size_ = num_samples;
                cached_gate_probs_.resize(num_samples);
                cached_logsumexp_.resize(num_samples);
                cached_expert_inputs_.resize(n_experts);
                cached_expert_outputs_.resize(n_experts);
            }
            cached_assignments_.resize(n_experts);
            for (long e = 0; e < n_experts; ++e) cached_assignments_[e].clear();

            // Per-sample routing
            for (long n = 0; n < num_samples; ++n) {
                const float* sample_logits = gate_data + n * n_experts;

                std::vector<float> logits(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    logits[e] = std::max(-logit_clamp,
                        std::min(logit_clamp, sample_logits[e]));
                    if (is_training && noise_scale > 0) {
                        static thread_local dlib::rand rnd(std::time(0));
                        logits[e] += noise_scale * rnd.get_random_gaussian();
                    }
                }

                // Numerically stable softmax
                float max_logit = *std::max_element(
                    logits.begin(), logits.end());
                float sum_exp = 0.0f;
                std::vector<float> probs(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    probs[e] = std::exp(logits[e] - max_logit);
                    sum_exp += probs[e];
                }
                for (long e = 0; e < n_experts; ++e) {
                    probs[e] /= sum_exp;
                    gate_prob_sum[e] += probs[e];
                }

                if (is_training) {
                    cached_gate_probs_[n] = probs;
                    cached_logsumexp_[n] = std::log(sum_exp) + max_logit;
                }

                // Top-k selection with capacity overflow
                std::vector<long> sorted_indices(n_experts);
                std::iota(sorted_indices.begin(),
                    sorted_indices.end(), 0);
                std::partial_sort(sorted_indices.begin(),
                    sorted_indices.begin() + top_k, sorted_indices.end(),
                        [&probs](long a, long b) { return probs[a] > probs[b]; });

                long selected_count = 0;
                float selected_prob_sum = 0.0f;
                std::vector<long> chosen;
                std::vector<float> chosen_probs;

                for (long rank = 0; rank < n_experts && selected_count < top_k; ++rank)
                {
                    long e = sorted_indices[rank];
                    if (static_cast<long>(expert_assignments[e].size()) < expert_capacity)
                    {
                        chosen.push_back(e);
                        chosen_probs.push_back(probs[e]);
                        selected_prob_sum += probs[e];
                        selected_count++;
                    }
                }

                // Renormalize selected weights to sum to 1
                for (long i = 0; i < static_cast<long>(chosen.size()); ++i)
                {
                    float w = (selected_prob_sum > 0)
                        ? chosen_probs[i] / selected_prob_sum : 1.0f;
                    expert_assignments[chosen[i]].emplace_back(n, w);
                    routing_fraction[chosen[i]] += 1.0f;
                }
            }

            for (long e = 0; e < n_experts; ++e)
                cached_assignments_[e] = expert_assignments[e];

            // Batched expert processing
            alias_tensor sample_alias(1, k, nr, nc);

            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = expert_assignments[e];
                if (assignments.empty()) continue;
                const long batch_count = static_cast<long>(assignments.size());

                // Gather samples assigned to this expert
                resizable_tensor batched_input;
                batched_input.set_size(batch_count, k, nr, nc);
                for (long b = 0; b < batch_count; ++b) {
                    const float* src = expert_input.host() + assignments[b].first * sample_size;
                    float* dst = batched_input.host() + b * sample_size;
                    std::copy(src, src + sample_size, dst);
                }

                if (is_training)
                    cached_expert_inputs_[e] = batched_input;

                experts[e].forward(batched_input);
                const tensor& expert_out = experts[e].get_output();

                // Cache output for task gradient in backward
                if (is_training) {
                    cached_expert_outputs_[e].copy_size(expert_out);
                    tt::copy_tensor(false, cached_expert_outputs_[e], 0, expert_out, 0, expert_out.k());
                }

                // Scatter: output[sample] += w * expert(x)
                for (long b = 0; b < batch_count; ++b) {
                    float w = assignments[b].second;
                    auto out_s = sample_alias(output, assignments[b].first * sample_size);
                    auto exp_s = sample_alias(expert_out, b * sample_size);
                    tt::add(1, out_s, w, exp_s);
                }
            }

            // Usage statistics and load balance loss
            if (is_training) {
                for (long e = 0; e < n_experts; ++e)
                    routing_fraction[e] /= (num_samples * top_k);
                cached_routing_fraction_ = routing_fraction;

                load_balance_loss_ = 0.0f;
                for (long e = 0; e < n_experts; ++e) {
                    float avg_prob = gate_prob_sum[e] / num_samples;
                    load_balance_loss_ += routing_fraction[e] * avg_prob;
                }
                load_balance_loss_ *= n_experts * load_balance_weight;

                if (usage_update_rate > 0) {
                    for (long e = 0; e < n_experts; ++e) {
                        float frac = static_cast<float>(expert_assignments[e].size()) / (num_samples * top_k);
                        expert_usage[e] = (1.0f - usage_update_rate) * expert_usage[e] + usage_update_rate * frac;
                    }
                }
            }
        }

        // Backward: expert backprop, gate gradient, parameter update
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tensor& input_grad = sub.get_gradient_input();
            const long num_samples = cached_batch_size_;
            const long k = gradient_input.k();
            const long nr = gradient_input.nr();
            const long nc = gradient_input.nc();
            const long sample_size = k * nr * nc;
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(sample_size));

            alias_tensor sample_alias(1, k, nr, nc);

            // Task scores for gate gradient
            std::vector<std::vector<float>> task_scores;
            if (is_training && learning_rate_multiplier > 0) {
                task_scores.resize(num_samples);
                for (long n = 0; n < num_samples; ++n)
                    task_scores[n].assign(n_experts, 0.0f);
            }

            // Phase 1: batched backward through each active expert
            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = cached_assignments_[e];
                if (assignments.empty()) continue;
                const long batch_count = static_cast<long>(assignments.size());

                // Gather weighted gradient
                resizable_tensor batched_grad;
                batched_grad.set_size(batch_count, k, nr, nc);
                for (long b = 0; b < batch_count; ++b) {
                    float w = assignments[b].second;
                    const float* src = gradient_input.host() + assignments[b].first * sample_size;
                    float* dst = batched_grad.host() + b * sample_size;
                    for (long j = 0; j < sample_size; ++j) dst[j] = src[j] * w;
                }

                // Expert backward
                experts[e].back_propagate_error(cached_expert_inputs_[e], batched_grad);
                const tensor& expert_data_grad = experts[e].get_final_data_gradient();

                // Scatter data gradient to main network
                for (long b = 0; b < batch_count; ++b) {
                    auto in_g = sample_alias(input_grad, assignments[b].first * sample_size);
                    auto ex_g = sample_alias(expert_data_grad, b * sample_size);
                    tt::add(1, in_g, 1, ex_g);
                }

                // Task score: dot(expert_out, task_grad) / sqrt(dim)
                if (is_training && learning_rate_multiplier > 0) {
                    const tensor& expert_out =
                        cached_expert_outputs_[e];
                    for (long b = 0; b < batch_count; ++b) {
                        const float* grad_ptr = gradient_input.host() + assignments[b].first * sample_size;
                        const float* out_ptr = expert_out.host() + b * sample_size;
                        float score = 0.0f;
                        for (long j = 0; j < sample_size; ++j)
                            score += grad_ptr[j] * out_ptr[j];
                        score *= inv_sqrt_dim;
                        score = std::max(-1.0f, std::min(1.0f, score));
                        task_scores[assignments[b].first][e] = score;
                    }
                }
            }

            // Phase 2: gate gradient and parameter updates
            const bool do_update = is_training && (!network_context::is_active() || network_context::is_training());

            if (do_update && learning_rate_multiplier > 0) {
                const tensor& gate_output = gate_net.get_output();
                resizable_tensor gate_grad;
                gate_grad.copy_size(gate_output);
                gate_grad = 0;
                float* glg = gate_grad.host();
                const float inv_batch = 1.0f / static_cast<float>(num_samples);

                // Part A: task gradient through softmax Jacobian
                for (long n = 0; n < num_samples; ++n) {
                    const auto& probs = cached_gate_probs_[n];
                    const auto& scores = task_scores[n];

                    float weighted_score_sum = 0.0f;
                    for (long e = 0; e < n_experts; ++e)
                        weighted_score_sum += probs[e] * scores[e];

                    for (long e = 0; e < n_experts; ++e)
                        glg[n * n_experts + e] += probs[e] * (scores[e] - weighted_score_sum);
                }

                // Part B: load balance gradient
                if (load_balance_weight > 0) {
                    for (long n = 0; n < num_samples; ++n) {
                        const auto& probs = cached_gate_probs_[n];
                        float sum_wp = 0.0f;
                        for (long e = 0; e < n_experts; ++e) {
                            float w_e = load_balance_weight * n_experts * cached_routing_fraction_[e];
                            sum_wp += w_e * probs[e];
                        }
                        for (long e = 0; e < n_experts; ++e) {
                            float w_e = load_balance_weight * n_experts * cached_routing_fraction_[e];
                            glg[n * n_experts + e] += probs[e] * (w_e - sum_wp) * inv_batch;
                        }
                    }
                }

                // Part C: router z-loss gradient (ST-MoE)
                if (z_loss_weight > 0) {
                    for (long n = 0; n < num_samples; ++n) {
                        const auto& probs = cached_gate_probs_[n];
                        float lse = cached_logsumexp_[n];
                        for (long e = 0; e < n_experts; ++e)
                            glg[n * n_experts + e] += 2.0f * z_loss_weight * lse * probs[e] * inv_batch;
                    }
                }

                // Safety clamp
                const float grad_clamp = 1.0f;
                long total_elems = num_samples * n_experts;
                for (long i = 0; i < total_elems; ++i)
                    glg[i] = std::max(-grad_clamp, std::min(grad_clamp, glg[i]));

                // Gate backward (data gradient NOT propagated)
                gate_net.back_propagate_error(cached_input_, gate_grad);
            }

            // Phase 3: update parameters + decay lb weight
            if (do_update) update_subnet_parameters();
        }

        void clean()
        {
            for (auto& expert : experts) clean_subnet(expert);
            clean_subnet(gate_net);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        void set_learning_rate(double lr)
        {
            current_learning_rate_ = lr;
            for (auto& expert : experts)
                set_all_learning_rates(expert, lr);
            set_all_learning_rates(gate_net, lr);
        }
        double get_learning_rate() const
        {
            return current_learning_rate_;
        }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;
            for (auto& expert : experts)
                set_all_learning_rate_multipliers(expert, val);
            set_all_learning_rate_multipliers(gate_net, val);
        }
        double get_learning_rate_multiplier() const
        {
            return learning_rate_multiplier;
        }

        void configure_solvers(
            double weight_decay, double beta1, double beta2)
        {
            solver_weight_decay_ = weight_decay;
            solver_beta1_ = beta1;
            solver_beta2_ = beta2;
            solvers_initialized_ = false;
        }

        // Total parameters across all experts + gate (training footprint)
        size_t internal_parameters() const
        {
            size_t total = count_parameters(gate_net);
            // Find any expert whose parameters have been allocated. After a single forward
            // pass with top_k < n_experts, only the routed experts have allocated params,
            // so experts[0] is not always representative. All experts share the same
            // architecture, so any initialized one gives the correct per-expert count.
            if (!experts.empty())
            {
                size_t per_expert = 0;
                for (const auto& e : experts)
                {
                    const size_t n = count_parameters(e);
                    if (n > 0) { per_expert = n; break; }
                }
                total += per_expert * static_cast<size_t>(n_experts);
            }
            return total;
        }

        // Parameters active during a single inference pass (top-k experts + gate)
        size_t active_parameters() const
        {
            size_t total = count_parameters(gate_net);
            if (!experts.empty())
            {
                size_t per_expert = 0;
                for (const auto& e : experts)
                {
                    const size_t n = count_parameters(e);
                    if (n > 0) { per_expert = n; break; }
                }
                total += per_expert * static_cast<size_t>(top_k);
            }
            return total;
        }

        EXPERT_NET& get_expert(size_t idx) {
            DLIB_CASSERT(idx < experts.size());
            return experts[idx];
        }
        const EXPERT_NET& get_expert(size_t idx) const {
            DLIB_CASSERT(idx < experts.size());
            return experts[idx];
        }

        GATE_NET& get_gate() { return gate_net; }
        const GATE_NET& get_gate() const { return gate_net; }

        long num_experts() const { return n_experts; }
        long num_active_experts() const { return top_k; }
        bool is_training_mode() const
        {
            return std::is_same<MODE, training_mode_tag>::value;
        }
        const std::vector<float>& get_expert_usage() const
        {
            return expert_usage;
        }
        float get_load_balance_loss() const
        {
            return load_balance_loss_;
        }
        float get_load_balance_weight() const
        {
            return load_balance_weight;
        }

        // Serialization
        friend void serialize(const moe_& item, std::ostream& out)
        {
            serialize("moe_", out);
            serialize(item.n_experts, out);
            serialize(item.top_k, out);
            serialize(item.noise_scale, out);
            serialize(item.capacity_factor, out);
            serialize(item.usage_update_rate, out);
            serialize(item.load_balance_weight, out);
            serialize(item.load_balance_floor_, out);
            serialize(item.load_balance_decay_, out);
            serialize(item.z_loss_weight, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.current_learning_rate_, out);
            serialize(item.gate_net, out);
            serialize(item.experts, out);
            serialize(item.expert_usage, out);
            serialize(item.expert_solvers_, out);
            serialize(item.gate_solvers_, out);
            serialize(item.solvers_initialized_, out);
        }

        friend void deserialize(moe_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "moe_")
                throw serialization_error(
                    "Unexpected version '" + version
                    + "' while deserializing moe_.");
            deserialize(item.n_experts, in);
            deserialize(item.top_k, in);
            deserialize(item.noise_scale, in);
            deserialize(item.capacity_factor, in);
            deserialize(item.usage_update_rate, in);
            deserialize(item.load_balance_weight, in);
            deserialize(item.load_balance_floor_, in);
            deserialize(item.load_balance_decay_, in);
            deserialize(item.z_loss_weight, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.current_learning_rate_, in);
            deserialize(item.gate_net, in);
            deserialize(item.experts, in);
            deserialize(item.expert_usage, in);
            deserialize(item.expert_solvers_, in);
            deserialize(item.gate_solvers_, in);
            deserialize(item.solvers_initialized_, in);
            item.cached_batch_size_ = 0;
            item.cached_expert_inputs_.clear();
            item.cached_expert_outputs_.clear();
        }

        friend std::ostream& operator<<(
            std::ostream& out, const moe_& item)
        {
            const bool training =
                std::is_same<MODE, training_mode_tag>::value;
            out << "moe\t ("
                << "experts=" << item.n_experts
                << ", top_k=" << item.top_k
                << ", mode=" << (training ? "train" : "infer")
                << ", noise=" << item.noise_scale
                << ", cf=" << item.capacity_factor
                << ", lb=" << item.load_balance_weight
                << "/" << item.load_balance_floor_
                << ", z=" << item.z_loss_weight
                << ") learning_rate_mult="
                << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const moe_& item, std::ostream& out)
        {
            const bool training =
                std::is_same<MODE, training_mode_tag>::value;
            out << "<moe"
                << " num_experts='" << item.n_experts << "'"
                << " top_k='" << item.top_k << "'"
                << " noise_scale='" << item.noise_scale << "'"
                << " capacity_factor='"
                << item.capacity_factor << "'"
                << " load_balance_weight='"
                << item.load_balance_weight << "'"
                << " load_balance_floor='"
                << item.load_balance_floor_ << "'"
                << " load_balance_decay='"
                << item.load_balance_decay_ << "'"
                << " z_loss_weight='"
                << item.z_loss_weight << "'"
                << " learning_rate_mult='"
                << item.learning_rate_multiplier << "'"
                << " mode='"
                << (training ? "training" : "inference") << "'"
                << ">\n";
            out << "  <gate>\n";
            to_xml(item.gate_net, out);
            out << "  </gate>\n";
            for (size_t i = 0; i < item.experts.size(); ++i) {
                out << "  <expert index='" << i << "'>\n";
                to_xml(item.experts[i], out);
                out << "  </expert>\n";
            }
            out << "  <expert_usage>";
            for (size_t i = 0; i < item.expert_usage.size(); ++i) {
                if (i > 0) out << " ";
                out << item.expert_usage[i];
            }
            out << "</expert_usage>\n";
            out << "</moe>\n";
        }

    private:

        template<typename NET>
        auto clean_subnet(NET& net)
            -> decltype(net.clean(), void()) {
            net.clean();
        }
        template<typename NET>
        void clean_subnet(...) {}

        void update_subnet_parameters()
        {
            if (network_context::is_active())
                current_learning_rate_ = network_context::get_learning_rate();
            if (learning_rate_multiplier == 0.0) return;

            const double effective_lr = current_learning_rate_ * learning_rate_multiplier;

            if (!solvers_initialized_) {
                const double wd = network_context::is_active()
                    ? network_context::get_optimizer_weight_decay() : solver_weight_decay_;
                const double b1 = network_context::is_active()
                    ? network_context::get_optimizer_beta1() : solver_beta1_;
                const double b2 = network_context::is_active()
                    ? network_context::get_optimizer_beta2() : solver_beta2_;
                expert_solvers_.resize(n_experts);
                for (long e = 0; e < n_experts; ++e)
                    expert_solvers_[e].assign(experts[e].num_computational_layers, adamw(wd, b1, b2));
                gate_solvers_.assign(gate_net.num_computational_layers, adamw(wd, b1, b2));
                solvers_initialized_ = true;
            }

            // Update only experts active in this batch
            for (long e = 0; e < n_experts; ++e)
                if (!cached_assignments_[e].empty())
                    experts[e].update_parameters(expert_solvers_[e], effective_lr);

            gate_net.update_parameters(gate_solvers_, effective_lr);

            // Exponential decay of load balance weight toward floor
            // (strong balancing early, then relax for specialization)
            if (load_balance_weight > load_balance_floor_)
                load_balance_weight = std::max(load_balance_floor_,
                    load_balance_weight * load_balance_decay_);
        }

        // Internal sub-networks
        GATE_NET gate_net;
        std::vector<EXPERT_NET> experts;

        // Configuration
        long n_experts;
        float noise_scale;
        long top_k;
        float capacity_factor;
        float usage_update_rate;
        float load_balance_weight;
        float load_balance_floor_;
        float load_balance_decay_;
        float z_loss_weight;
        double learning_rate_multiplier;
        double current_learning_rate_;

        // Expert usage tracking (EMA)
        std::vector<float> expert_usage;

        // Cached tensors for backward
        resizable_tensor cached_input_;
        std::vector<resizable_tensor> cached_expert_inputs_;
        std::vector<resizable_tensor> cached_expert_outputs_;

        // Internal AdamW solvers
        std::vector<std::vector<adamw>> expert_solvers_;
        std::vector<adamw> gate_solvers_;
        bool solvers_initialized_;
        double solver_weight_decay_ = 0.004;
        double solver_beta1_ = 0.9;
        double solver_beta2_ = 0.999;

        // Forward/backward routing state
        std::vector<std::vector<std::pair<long, float>>>
            cached_assignments_;
        std::vector<std::vector<float>> cached_gate_probs_;
        std::vector<float> cached_logsumexp_;
        std::vector<float> cached_routing_fraction_;
        long cached_batch_size_;
        float load_balance_loss_;

        resizable_tensor params;    // Unused
    };

    template<
        typename EXPERT_NET,
        typename GATE_NET,
        long n_experts_,
        long top_e,
        typename MODE,
        typename SUBNET
    >
    using moe = add_layer<moe_<EXPERT_NET, GATE_NET, n_experts_, top_e, MODE>, SUBNET>;
}

#endif // DLIB_DNN_TRANSFORMER_H_