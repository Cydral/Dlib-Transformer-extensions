// Copyright (C) 2026  Cydral Technology (cydraltechnology@gmail.com)
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

    template <long repeat_factor_>
    class repeat_heads_
    {
    public:
        static_assert(repeat_factor_ >= 1, "repeat_factor must be at least 1");

        explicit repeat_heads_() : repeat_factor(repeat_factor_)
        {
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            // No learnable parameters
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& input = sub.get_output();
            const long batch_size = input.num_samples();
            const long num_kv_heads = input.k();
            const long seq_len = input.nr();
            const long head_dim = input.nc();

            output.set_size(batch_size, num_kv_heads * repeat_factor, seq_len, head_dim);

            if (repeat_factor == 1)
            {
                tt::copy_tensor(false, output, 0, input, 0, num_kv_heads);
                return;
            }

            const float* in_data = input.host();
            float* out_data = output.host();
            const long kv_head_stride = seq_len * head_dim;

            for (long b = 0; b < batch_size; ++b)
            {
                for (long kv_h = 0; kv_h < num_kv_heads; ++kv_h)
                {
                    const float* src = in_data + (b * num_kv_heads + kv_h) * kv_head_stride;
                    for (long r = 0; r < repeat_factor; ++r)
                    {
                        const long out_head = kv_h * repeat_factor + r;
                        float* dst = out_data + (b * num_kv_heads * repeat_factor + out_head) * kv_head_stride;
                        std::copy(src, src + kv_head_stride, dst);
                    }
                }
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tensor& data_grad = sub.get_gradient_input();

            if (repeat_factor == 1)
            {
                tt::copy_tensor(true, data_grad, 0, gradient_input, 0, gradient_input.k());
                return;
            }

            const long batch_size = gradient_input.num_samples();
            const long num_heads = gradient_input.k();
            const long num_kv_heads = num_heads / repeat_factor;
            const long seq_len = gradient_input.nr();
            const long head_dim = gradient_input.nc();
            const long kv_head_stride = seq_len * head_dim;

            const float* grad_in = gradient_input.host();
            float* grad_out = data_grad.host();

            // Accumulate gradients from repeated heads back to original KV heads
            for (long b = 0; b < batch_size; ++b)
            {
                for (long kv_h = 0; kv_h < num_kv_heads; ++kv_h)
                {
                    float* dst = grad_out + (b * num_kv_heads + kv_h) * kv_head_stride;
                    for (long r = 0; r < repeat_factor; ++r)
                    {
                        const long in_head = kv_h * repeat_factor + r;
                        const float* src = grad_in + (b * num_heads + in_head) * kv_head_stride;
                        for (long i = 0; i < kv_head_stride; ++i)
                        {
                            dst[i] += src[i];
                        }
                    }
                }
            }
        }

        inline dpoint map_input_to_output(const dpoint& p) const { return p; }
        inline dpoint map_output_to_input(const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const repeat_heads_& item, std::ostream& out)
        {
            serialize("repeat_heads_", out);
            serialize(item.repeat_factor, out);
        }

        friend void deserialize(repeat_heads_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "repeat_heads_")
                throw serialization_error("Unexpected version '" + version + "' found while deserializing dlib::repeat_heads_.");
            deserialize(item.repeat_factor, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const repeat_heads_& item)
        {
            out << "repeat_heads\t ("
                << "repeat_factor=" << item.repeat_factor
                << ")";
            return out;
        }

        friend void to_xml(const repeat_heads_& item, std::ostream& out)
        {
            out << "<repeat_heads"
                << " repeat_factor='" << item.repeat_factor << "'"
                << "/>\n";
        }

    private:
        long repeat_factor;
        resizable_tensor params;  // unused
    };

    template <long RF, typename SUBNET>
    using repeat_heads = add_layer<repeat_heads_<RF>, SUBNET>;

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

        template <long d_model, typename SUBNET>
        using fused_swiglu = extract<0, 1, 1, d_model,
            fc<d_model, mult_prev7<fc<(d_model * 8) / 3, skip6<
            tag7<silu<fc<(d_model * 8) / 3, tag6<SUBNET>>>>>>>>>;

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

    // HIERARCHICAL REASONING MODEL (HRM)
    //
    // Implements the dual-recurrent architecture from:
    //   Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734
    //
    // Architecture:
    //   - H-module (high-level): slow cycles for abstract planning (N updates)
    //   - L-module (low-level): fast iterations for detailed computation (T per H-cycle)
    //   - Total recurrent depth: N * T steps
    //   - 1-step gradient approximation for O(1) memory training
    //
    // Gradient path (1-step approximation):
    //   loss -> output -> final H-module -> final L-module -> input embedding    
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

        // FORWARD: Hierarchical recurrence with 1-step gradient approximation
        //
        // From the paper (Section 2, Figure 4):
        //   z_H, z_L = z_init
        //   for _i in range(N*T - 1):        # no gradient
        //     z_L = f_L(z_L + z_H + x)
        //     if (_i+1) % T == 0:
        //       z_H = f_H(z_H + z_L)
        //   # 1-step grad (final L then final H)
        //   z_L = f_L(z_L + z_H + x)
        //   z_H = f_H(z_H + z_L)
        //   output = z_H
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

        // BACKWARD: 1-step gradient approximation
        //
        // Gradient path:
        //   dL/d(output) = gradient_input
        //     -> backprop through H: dL/d(last_h_input) = h_net.get_final_data_gradient()
        //     -> backprop through L: dL/d(last_l_input) = l_net.get_final_data_gradient()
        //     -> propagate to input x (component of last_l_input)
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
            // FIX: get_final_data_gradient() is the gradient w.r.t. last_l_input
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

            static std::atomic<unsigned long> seed_counter{ 0 };
            dlib::rand rnd(seed_counter++);

            auto* h_ptr = z_h_init.host();
            auto* l_ptr = z_l_init.host();

            for (long c = 0; c < hidden_dim; ++c) {
                float h_val, l_val;
                do { h_val = rnd.get_random_gaussian(); } while (std::abs(h_val) > 2.0f);
                do { l_val = rnd.get_random_gaussian(); } while (std::abs(l_val) > 2.0f);
                h_ptr[c] = h_val;
                l_ptr[c] = l_val;
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

        resizable_tensor params;  // Empty: no direct trainable parameters
    };

    template<typename H_NET, typename L_NET, int N, int T, typename SUBNET>
    using hrm = add_layer<hrm_<H_NET, L_NET, N, T>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    // Gate network: produces raw logits for expert selection
    template <long num_experts, template <typename> class DO, typename SUBNET>
    using gate = fc<num_experts, DO<leaky_relu<fc<num_experts * 8, avg_pool_everything<SUBNET>>>>>;

    struct training_mode_tag {};
    struct inference_mode_tag {};

    template<
        typename EXPERT_NET,                    // Expert network architecture
        long top_e,                             // Number of experts to activate (0 = auto: 20%)
        typename MODE,                          // Tag-based mode selection (training/inference)
        template<typename> class TAG,           // Tag for gate input location
        typename SUBNET                         // Input subnet type
    >
    class moe_
    {
    public:
        /*!
            Mixture of Experts layer with sample-wise expert routing.

            Key features:
            - Each sample independently selects top-k experts via gating network
            - Gate produces logits, optional noise added before softmax (training only)
            - Forward/backward consistency via cached expert selections
            - Tracks expert usage statistics for monitoring
            - Internal AdamW solvers update expert parameters after each backward pass

            Hyperparameters:
            - noise_scale: Gaussian noise std applied to gate logits (exploration)
            - usage_update_rate: EMA smoothing for usage statistics
        !*/
        explicit moe_() :
            n_experts(0),
            noise_scale(0.1f),
            top_k(top_e),
            usage_update_rate(0.05f),
            load_balance_weight(0.01f),
            learning_rate_multiplier(1.0),
            current_learning_rate_(1e-6),
            solvers_initialized_(false),
            cached_batch_size_(0),
            load_balance_loss_(0.0f)
        {
        }

        moe_(const moe_& other) :
            n_experts(other.n_experts),
            noise_scale(other.noise_scale),
            top_k(other.top_k),
            usage_update_rate(other.usage_update_rate),
            load_balance_weight(other.load_balance_weight),
            learning_rate_multiplier(other.learning_rate_multiplier),
            current_learning_rate_(other.current_learning_rate_),
            expert_usage(other.expert_usage),
            solvers_initialized_(false),
            cached_batch_size_(0),
            load_balance_loss_(0.0f)
        {
            // Deep copy of expert networks; solvers are runtime state, reset on copy
            experts.reserve(other.experts.size());
            for (const auto& expert : other.experts)
                experts.push_back(expert);
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                n_experts = other.n_experts;
                noise_scale = other.noise_scale;
                top_k = other.top_k;
                usage_update_rate = other.usage_update_rate;
                load_balance_weight = other.load_balance_weight;
                learning_rate_multiplier = other.learning_rate_multiplier;
                current_learning_rate_ = other.current_learning_rate_;
                expert_usage = other.expert_usage;

                // Solvers are runtime state: reset on copy
                solvers_initialized_ = false;
                expert_solvers_.clear();
                cached_batch_size_ = 0;
                load_balance_loss_ = 0.0f;

                // Deep copy of expert networks
                experts.clear();
                experts.reserve(other.experts.size());
                for (const auto& expert : other.experts)
                    experts.push_back(expert);
            }
            return *this;
        }

        /*!
            SETUP
                Initializes expert networks based on gate output dimensions.
                - Number of experts automatically determined from gate output channels
                - If top_e == 0 (auto mode), activates 20% of experts (minimum 1)
        !*/
        template <typename SUBNET_TYPE>
        void setup(const SUBNET_TYPE& sub) {
            const tensor& gate_output = layer<TAG>(sub).get_output();
            long new_n_experts = gate_output.k();

            // Initialize experts if needed
            if (new_n_experts != n_experts) {
                n_experts = new_n_experts;
                expert_usage.resize(n_experts, 0.0f);

                // Create expert network instances
                experts.clear();
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i)
                    experts.emplace_back(EXPERT_NET{});

                // Determine top-k activation count
                if (top_e == 0) {
                    // Auto mode: activate 20% of experts (minimum 1)
                    top_k = std::max(1L, static_cast<long>(std::floor(n_experts * 0.2f)));
                }
                else {
                    top_k = std::min(top_e, n_experts);
                }

                // Solvers must be re-initialized whenever the expert count changes
                solvers_initialized_ = false;
                expert_solvers_.clear();
            }
        }

        // FORWARD PASS (batched expert processing)
        template <typename SUBNET_TYPE>
        void forward(const SUBNET_TYPE& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const tensor& gate_logits = layer<TAG>(sub).get_output();

            DLIB_CASSERT(gate_logits.k() == n_experts &&
                gate_logits.nr() == 1 && gate_logits.nc() == 1,
                "\nExpected gate output shape [batch_size, " << n_experts << ", 1, 1]"
                << "\nReceived shape [" << gate_logits.num_samples() << ", "
                << gate_logits.k() << ", " << gate_logits.nr() << ", "
                << gate_logits.nc() << "]");

            const long num_samples = gate_logits.num_samples();
            const long k = expert_input.k();
            const long nr = expert_input.nr();
            const long nc = expert_input.nc();
            const long sample_size = k * nr * nc;
            const float* logits_data = gate_logits.host();

            // Initialize output tensor
            output.copy_size(expert_input);
            output = 0;

            // Prepare forward pass cache for backward consistency
            if (std::is_same<MODE, training_mode_tag>::value) {
                cached_batch_size_ = num_samples;
                selected_expert_indices_.resize(num_samples);
                selected_expert_weights_.resize(num_samples);
                cached_gate_probs_.resize(num_samples);
            }

            // Track expert usage for monitoring
            std::vector<float> batch_expert_usage(n_experts, 0.0f);
            std::vector<float> routing_fraction(n_experts, 0.0f);
            std::vector<float> gate_prob_sum(n_experts, 0.0f);

            // Phase 1: Compute routing for all samples
            // For each expert, collect which samples are routed to it and with what weight
            // expert_assignments[e] = vector of (sample_index, weight)
            std::vector<std::vector<std::pair<long, float>>> expert_assignments(n_experts);

            for (long n = 0; n < num_samples; ++n) {
                const float* sample_logits = logits_data + n * n_experts;

                // Apply optional Gaussian noise to logits before softmax
                std::vector<float> noisy_logits(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    noisy_logits[e] = sample_logits[e];

                    if (std::is_same<MODE, training_mode_tag>::value && noise_scale > 0) {
                        static thread_local dlib::rand rnd(std::time(0));
                        noisy_logits[e] += noise_scale * rnd.get_random_gaussian();
                    }
                }

                // Softmax: numerically stable implementation
                float max_logit = *std::max_element(noisy_logits.begin(), noisy_logits.end());

                std::vector<float> exp_logits(n_experts);
                float sum_exp = 0.0f;
                for (long e = 0; e < n_experts; ++e) {
                    exp_logits[e] = std::exp(noisy_logits[e] - max_logit);
                    sum_exp += exp_logits[e];
                }

                std::vector<float> probs(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    probs[e] = exp_logits[e] / sum_exp;
                    gate_prob_sum[e] += probs[e];
                }
                if (std::is_same<MODE, training_mode_tag>::value) {
                    cached_gate_probs_[n] = probs;
                }

                // Select top-k experts by probability
                std::vector<std::pair<float, size_t>> expert_scores;
                expert_scores.reserve(n_experts);
                for (long e = 0; e < n_experts; ++e)
                    expert_scores.emplace_back(probs[e], e);

                std::partial_sort(expert_scores.begin(),
                    expert_scores.begin() + top_k,
                    expert_scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // Renormalize top-k weights to sum to 1
                float sum_weights = 0.0f;
                for (long i = 0; i < top_k; ++i)
                    sum_weights += expert_scores[i].first;

                // Handle degenerate case (should be extremely rare with softmax)
                if (sum_weights < 1e-8f) {
                    sum_weights = static_cast<float>(top_k);
                    for (long i = 0; i < top_k; ++i)
                        expert_scores[i].first = 1.0f;
                }

                for (long i = 0; i < top_k; ++i)
                    expert_scores[i].first /= sum_weights;

                // Cache selection and build per-expert assignments
                if (std::is_same<MODE, training_mode_tag>::value) {
                    selected_expert_indices_[n].resize(top_k);
                    selected_expert_weights_[n].resize(top_k);
                }

                for (long i = 0; i < top_k; ++i) {
                    const size_t expert_idx = expert_scores[i].second;
                    const float weight = expert_scores[i].first;

                    expert_assignments[expert_idx].emplace_back(n, weight);

                    if (std::is_same<MODE, training_mode_tag>::value) {
                        selected_expert_indices_[n][i] = expert_idx;
                        selected_expert_weights_[n][i] = weight;
                        routing_fraction[expert_idx] += 1.0f;
                    }

                    batch_expert_usage[expert_idx] += weight;
                }
            }

            // Phase 2: Batched forward through each expert
            alias_tensor sample_alias(1, k, nr, nc);

            // Resize cache to match number of experts
            if (std::is_same<MODE, training_mode_tag>::value)
                cached_expert_inputs_.resize(n_experts);

            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = expert_assignments[e];
                if (assignments.empty()) continue;

                const long batch_count = static_cast<long>(assignments.size());

                // Build batched input tensor for this expert
                resizable_tensor batched_input;
                batched_input.set_size(batch_count, k, nr, nc);

                for (long b = 0; b < batch_count; ++b) {
                    const long src_sample = assignments[b].first;
                    const long src_offset = src_sample * sample_size;
                    const long dst_offset = b * sample_size;

                    // Copy sample data into batched tensor
                    const float* src = expert_input.host() + src_offset;
                    float* dst = batched_input.host() + dst_offset;
                    std::copy(src, src + sample_size, dst);
                }

                // Cache input for backward (training only)
                if (std::is_same<MODE, training_mode_tag>::value)
                    cached_expert_inputs_[e] = batched_input;

                // Forward the entire batch through this expert
                experts[e].forward(batched_input);
                const tensor& expert_out = experts[e].get_output();

                // Scatter weighted outputs back to the main output tensor
                for (long b = 0; b < batch_count; ++b) {
                    const long dst_sample = assignments[b].first;
                    const float weight = assignments[b].second;
                    const long dst_offset = dst_sample * sample_size;

                    auto sample_output = sample_alias(output, dst_offset);
                    auto expert_sample_out = sample_alias(expert_out, b * sample_size);

                    tt::add(1, sample_output, weight, expert_sample_out);
                }
            }

            // Update exponential moving average of expert usage (for monitoring)
            if (std::is_same<MODE, training_mode_tag>::value) {
                for (long e = 0; e < n_experts; ++e) {
                    routing_fraction[e] /= num_samples;
                    gate_prob_sum[e] /= num_samples;
                }

                load_balance_loss_ = 0.0f;
                for (long e = 0; e < n_experts; ++e) {
                    load_balance_loss_ += routing_fraction[e] * gate_prob_sum[e];
                }
                load_balance_loss_ *= n_experts * load_balance_weight;

                cached_routing_fraction_ = routing_fraction;
                cached_gate_prob_avg_ = gate_prob_sum;

                if (usage_update_rate > 0) {
                    for (long e = 0; e < n_experts; ++e) {
                        float avg_usage = batch_expert_usage[e] / num_samples;
                        expert_usage[e] = (1.0f - usage_update_rate) * expert_usage[e] +
                            usage_update_rate * avg_usage;
                    }
                }
            }
        }

        // BACKWARD PASS (batched expert processing)
        template <typename SUBNET_TYPE>
        void backward(const tensor& gradient_input, SUBNET_TYPE& sub, tensor& /*params_grad*/)
        {
            tensor& expert_input_grad = sub.get_gradient_input();

            const tensor& expert_input = sub.get_output();
            const long num_samples = cached_batch_size_;
            const long k = gradient_input.k();
            const long nr = gradient_input.nr();
            const long nc = gradient_input.nc();
            const long sample_size = k * nr * nc;

            DLIB_CASSERT(num_samples == (long)selected_expert_indices_.size(),
                "Forward pass cache missing or invalid in backward pass");

            // Phase 1: Rebuild per-expert assignments from cached routing
            // expert_assignments[e] = vector of (sample_index, weight)
            std::vector<std::vector<std::pair<long, float>>> expert_assignments(n_experts);

            for (long n = 0; n < num_samples; ++n) {
                const auto& expert_indices = selected_expert_indices_[n];
                const auto& expert_weights = selected_expert_weights_[n];

                for (size_t i = 0; i < expert_indices.size(); ++i) {
                    expert_assignments[expert_indices[i]].emplace_back(
                        n, expert_weights[i]);
                }
            }

            // Phase 2: Batched backward through each expert
            alias_tensor sample_alias(1, k, nr, nc);

            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = expert_assignments[e];
                if (assignments.empty()) continue;

                const long batch_count = static_cast<long>(assignments.size());

                // Use cached input — DO NOT re-forward, it corrupts the expert's internal state
                const resizable_tensor& batched_input = cached_expert_inputs_[e];

                // Build batched weighted gradient
                resizable_tensor batched_grad;
                batched_grad.set_size(batch_count, k, nr, nc);

                for (long b = 0; b < batch_count; ++b) {
                    const long src_sample = assignments[b].first;
                    const float weight = assignments[b].second;

                    const float* src = gradient_input.host() + src_sample * sample_size;
                    float* dst = batched_grad.host() + b * sample_size;

                    for (long j = 0; j < sample_size; ++j)
                        dst[j] = src[j] * weight;
                }

                // Single backward pass per expert:
                // Parameter gradients are correctly accumulated across all samples
                // in the batch inside Dlib's backward implementation
                experts[e].back_propagate_error(batched_input, batched_grad);
                const tensor& expert_data_grad = experts[e].get_final_data_gradient();

                // Scatter input gradients back to the main gradient tensor
                for (long b = 0; b < batch_count; ++b) {
                    const long dst_sample = assignments[b].first;
                    auto sample_grad = sample_alias(expert_input_grad, dst_sample * sample_size);

                    auto expert_sample_grad = sample_alias(expert_data_grad, b * sample_size);

                    tt::add(1, sample_grad, 1, expert_sample_grad);
                }
            }

            // Phase 3: Load-balance auxiliary gradient
            if (std::is_same<MODE, training_mode_tag>::value && load_balance_weight > 0
                && learning_rate_multiplier > 0) {
                tensor& gate_grad = layer<TAG>(sub).get_gradient_input();
                float* gate_grad_data = gate_grad.host();

                // Gradient of load balancing loss w.r.t. gate logits.
                //
                // Loss:  L = alpha * N_e * sum_e ( F_e * P_e_avg )
                //   where F_e     = fraction of samples routing to expert e
                //         P_e_avg = (1/N_s) * sum_n P_{ne}
                //
                // Gradient w.r.t. logit z_{nj}:
                //   dL/dz_{nj} = (alpha * N_e / N_s) * P_{nj} * (F_j - sum_e F_e * P_{ne})
                //
                const float inv_ns = 1.0f / static_cast<float>(num_samples);

                for (long n = 0; n < num_samples; ++n) {
                    const auto& gate_probs = cached_gate_probs_[n];

                    float sum_weighted_probs = 0.0f;
                    for (long e = 0; e < n_experts; ++e) {
                        float w_e = cached_routing_fraction_[e] * n_experts *
                            load_balance_weight * inv_ns;
                        sum_weighted_probs += w_e * gate_probs[e];
                    }

                    for (long e = 0; e < n_experts; ++e) {
                        float w_e = cached_routing_fraction_[e] * n_experts *
                            load_balance_weight * inv_ns;
                        gate_grad_data[n * n_experts + e] +=
                            gate_probs[e] * (w_e - sum_weighted_probs);
                    }
                }
            }

            // Phase 4: Update expert parameters
            const bool in_training = !network_context::is_active() ||
                network_context::is_training();
            if (std::is_same<MODE, training_mode_tag>::value && in_training)
                update_subnet_parameters();
        }

        void clean()
        {
            for (auto& expert : experts)
                clean_subnet(expert);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        void set_learning_rate(double lr)
        {
            current_learning_rate_ = lr;
            for (auto& expert : experts)
                set_all_learning_rates(expert, lr);
        }
        double get_learning_rate() const { return current_learning_rate_; }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;
            for (auto& expert : experts)
                set_all_learning_rate_multipliers(expert, val);
        }
        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }

        void configure_solvers(double weight_decay, double beta1, double beta2)
        {
            solver_weight_decay_ = weight_decay;
            solver_beta1_ = beta1;
            solver_beta2_ = beta2;
            solvers_initialized_ = false;
        }

        // Returns total parameter count across all expert networks
        size_t internal_parameters() const
        {
            if (experts.empty()) return 0;
            return count_parameters(experts[0]) * static_cast<size_t>(n_experts);
        }

        // Returns parameter count for the top-k experts active during inference
        size_t active_parameters() const
        {
            if (experts.empty()) return 0;
            return count_parameters(experts[0]) * static_cast<size_t>(top_k);
        }

        // Direct access to expert networks (for inspection/debugging)
        EXPERT_NET& get_expert(size_t idx) {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        const EXPERT_NET& get_expert(size_t idx) const {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        // Accessors
        long num_experts() const { return n_experts; }
        long num_active_experts() const { return top_k; }
        bool is_training_mode() const { return std::is_same<MODE, training_mode_tag>::value; }
        const std::vector<float>& get_expert_usage() const { return expert_usage; }
        float get_load_balance_loss() const { return load_balance_loss_; }

        friend void serialize(const moe_& item, std::ostream& out)
        {
            serialize("moe_", out);
            serialize(item.n_experts, out);
            serialize(item.top_k, out);
            serialize(item.noise_scale, out);
            serialize(item.usage_update_rate, out);
            serialize(item.load_balance_weight, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.current_learning_rate_, out);
            serialize(item.experts, out);
            serialize(item.expert_usage, out);
            serialize(item.expert_solvers_, out);
            serialize(item.solvers_initialized_, out);
        }

        friend void deserialize(moe_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "moe_") {                
                deserialize(item.n_experts, in);
                deserialize(item.top_k, in);
                deserialize(item.noise_scale, in);
                deserialize(item.usage_update_rate, in);
                deserialize(item.load_balance_weight, in);
                deserialize(item.learning_rate_multiplier, in);
                deserialize(item.current_learning_rate_, in);
                deserialize(item.experts, in);
                deserialize(item.expert_usage, in);
                deserialize(item.expert_solvers_, in);
                deserialize(item.solvers_initialized_, in);
            }
            else {
                throw serialization_error("Incorrect version '" + version + "' found while deserializing moe_.");
            }
            item.cached_batch_size_ = 0;
            item.cached_expert_inputs_.clear();
        }

        friend std::ostream& operator<<(std::ostream& out, const moe_& item)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "moe\t ("
                << "experts=" << item.n_experts
                << ", top_k=" << item.top_k
                << ", mode=" << (is_training ? "train" : "infer")
                << ", noise=" << item.noise_scale
                << ", lb=" << item.load_balance_weight
                << ")";
            out << " learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const moe_& item, std::ostream& out)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "<moe"
                << " num_experts='" << item.n_experts << "'"
                << " top_k='" << item.top_k << "'"
                << " noise_scale='" << item.noise_scale << "'"
                << " usage_update_rate='" << item.usage_update_rate << "'"
                << " load_balance_weight='" << item.load_balance_weight << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier << "'"
                << " mode='" << (is_training ? "training" : "inference") << "'"
                << ">\n";
            for (size_t i = 0; i < item.experts.size(); ++i)
            {
                out << "<expert index='" << i << "'>\n";
                to_xml(item.experts[i], out);
                out << "</expert>\n";
            }
            out << "<expert_usage>";
            for (size_t i = 0; i < item.expert_usage.size(); ++i)
            {
                if (i > 0) out << " ";
                out << item.expert_usage[i];
            }
            out << "</expert_usage>\n";
            out << "</moe>\n";
        }

    private:
        template<typename NET>
        auto clean_subnet(NET& net) -> decltype(net.clean(), void())
        {
            net.clean();
        }

        template<typename NET>
        void clean_subnet(...)
        {
            // No-op if network doesn't have clean() method
        }

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
                solvers_initialized_ = true;
            }

            // Only update experts activated in this forward pass (others have no valid gradients)
            std::set<size_t> used_experts;
            for (const auto& indices : selected_expert_indices_)
                for (size_t idx : indices)
                    used_experts.insert(idx);

            for (size_t e : used_experts)
                experts[e].update_parameters(expert_solvers_[e], effective_lr);
        }

        // Configuration
        long n_experts;                     // Number of expert networks
        float noise_scale;                  // Gaussian noise std for exploration
        long top_k;                         // Number of experts to activate per sample
        float usage_update_rate;            // EMA smoothing rate for usage tracking
        float load_balance_weight;          // Auxiliary loss coefficient for expert load balancing
        double learning_rate_multiplier;
        double current_learning_rate_;

        // Expert networks
        std::vector<EXPERT_NET> experts;
        std::vector<float> expert_usage;    // Usage statistics (for monitoring)
        std::vector<resizable_tensor> cached_expert_inputs_;

        // Internal solvers for expert parameter updates
        std::vector<std::vector<adamw>> expert_solvers_;  // [expert_idx][layer_idx]
        bool solvers_initialized_;
        double solver_weight_decay_ = 0.004;
        double solver_beta1_ = 0.9;
        double solver_beta2_ = 0.999;

        // Forward/backward cache (training mode only)
        std::vector<std::vector<size_t>> selected_expert_indices_;  // [sample][top_k]
        std::vector<std::vector<float>> selected_expert_weights_;   // [sample][top_k]
        std::vector<std::vector<float>> cached_gate_probs_;
        std::vector<float> cached_routing_fraction_;
        std::vector<float> cached_gate_prob_avg_;
        long cached_batch_size_;
        float load_balance_loss_;

        resizable_tensor params; // Unused — required for get_layer_params() interface
    };

    template<
        typename EXPERT_NET,
        long top_e,
        typename MODE,
        template<typename> class TAG,
        typename SUBNET
    >
    using moe = add_layer<moe_<EXPERT_NET, top_e, MODE, TAG, SUBNET>, SUBNET>;

    // This is a drop-in replacement for standard transformer feed-forward layers
    template <typename SUBNET> using moe_tag_input = add_tag_layer<1600 + 0, SUBNET>;
    template <typename SUBNET> using moe_tag_gate = add_tag_layer<1600 + 1, SUBNET>;
    template <typename SUBNET> using add_moe_tag_input = add_prev<moe_tag_input, SUBNET>;
    using add_moe_tag_input_ = add_prev_<moe_tag_input>;
    template <typename SUBNET> using skip_moe_tag_input = add_skip_layer<moe_tag_input, SUBNET>;

    template<
        typename EXPERT_NET,
        long num_experts,
        long top_e,
        typename MODE,
        template <typename> class DO,
        typename SUBNET
    >
    using moe_ffn = add_moe_tag_input<
        moe<EXPERT_NET, top_e, MODE, moe_tag_gate, rms_norm<skip_moe_tag_input<
        moe_tag_gate<gate<num_experts, DO, moe_tag_input<SUBNET>>>>>>>;
}

#endif // DLIB_DNN_TRANSFORMER_H_