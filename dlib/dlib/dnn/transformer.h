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

// ========================================================================================
    // MIXTURE OF EXPERTS (MoE) LAYER WITH INTERNAL GATE
    //
    // IMPORTANT: GATE_NET and EXPERT_NET definitions must use tag10<input_tensor>
    // (not raw input_tensor) as their base layer. This is because input_tensor is
    // defined as input<tensor> which is an input-layer type. When add_layer wraps
    // an input-layer type directly, its forward() requires sample_expansion_factor
    // to be initialized via to_tensor() — which is never called for internal
    // sub-networks managed manually. Wrapping in tag10<> makes the base an
    // add_tag_layer, which is a non-loss-layer type, and the general add_layer
    // specialization does NOT check sample_expansion_factor. This is the same
    // pattern used by canonical_transformer::transformer_stack (base case = tag10<SUBNET>).
    //
    // Example gate definition:
    //   using gate = fc<N, leaky_relu<fc<N*8, avg_pool_everything<tag10<input_tensor>>>>>;
    //
    // Example expert definition (swiglu already uses this pattern internally):
    //   using expert = swiglu<d_model, 8, 3, tag10<input_tensor>>;
    //
    // BACKWARD GRADIENT FLOW:
    //   The gate is an internal routing mechanism. Only expert data gradients flow
    //   back to the main network's input gradient. The gate backward is called solely
    //   to compute the gate's own parameter gradients for its internal solver update.
    //   Gate data gradient is NOT propagated to sub.get_gradient_input().
    //
    //   Main network ← expert data gradient only
    //   Gate params  ← routing gradient + load balance gradient (internal update)
    // ========================================================================================

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
            noise_scale(0.1f),
            top_k(1),
            usage_update_rate(0.05f),
            load_balance_weight(0.01f),
            learning_rate_multiplier(1.0),
            current_learning_rate_(1e-6),
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
            usage_update_rate(other.usage_update_rate),
            load_balance_weight(other.load_balance_weight),
            learning_rate_multiplier(other.learning_rate_multiplier),
            current_learning_rate_(other.current_learning_rate_),
            expert_usage(other.expert_usage),
            solvers_initialized_(false),
            cached_batch_size_(0),
            load_balance_loss_(0.0f)
        {
            experts.reserve(other.experts.size());
            for (const auto& expert : other.experts)
                experts.push_back(expert);
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                gate_net = other.gate_net;
                n_experts = other.n_experts;
                noise_scale = other.noise_scale;
                top_k = other.top_k;
                usage_update_rate = other.usage_update_rate;
                load_balance_weight = other.load_balance_weight;
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
                for (const auto& expert : other.experts)
                    experts.push_back(expert);
            }
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            if (experts.empty()) {
                expert_usage.resize(n_experts, 0.0f);
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i)
                    experts.emplace_back(EXPERT_NET{});
                solvers_initialized_ = false;
            }
        }

        // -----------------------------------------------------------------------
        // FORWARD
        // -----------------------------------------------------------------------
        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const long num_samples = expert_input.num_samples();
            const long k = expert_input.k();
            const long nr = expert_input.nr();
            const long nc = expert_input.nc();
            const long sample_size = k * nr * nc;
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;

            // Cache input for expert and gate backward
            cached_input_.copy_size(expert_input);
            tt::copy_tensor(false, cached_input_, 0, expert_input, 0, k);

            // Step 1: Gate forward → routing logits
            gate_net.forward(cached_input_);
            const tensor& gate_output = gate_net.get_output();

            // Gate output shape: (batch, n_experts, 1, 1) from fc-based gate
            //                 or (batch, 1, 1, n_experts) from fused_swiglu-based gate
            // Detect layout and extract logits accordingly
            const long gate_k = gate_output.k();
            const long gate_nc = gate_output.nc();
            const float* gate_data = gate_output.host();

            DLIB_CASSERT(
                (gate_k == n_experts && gate_output.nr() == 1 && gate_nc == 1) ||
                (gate_k == 1 && gate_output.nr() == 1 && gate_nc == n_experts),
                "\nExpected gate output (batch," << n_experts << ",1,1) or (batch,1,1," << n_experts << ")"
                << "\nReceived (" << gate_output.num_samples() << "," << gate_k
                << "," << gate_output.nr() << "," << gate_nc << ")");

            // Logit stride: distance between consecutive expert logits for one sample
            // For (batch, N, 1, 1): stride = 1, offset between experts along k
            // For (batch, 1, 1, N): stride = 1, offset between experts along nc
            // In both cases the N values are contiguous in memory for each sample
            const long logits_per_sample = n_experts;

            // Initialize output to zero
            output.copy_size(expert_input);
            output = 0;

            // Prepare caches (training only)
            if (is_training) {
                cached_batch_size_ = num_samples;
                selected_expert_indices_.resize(num_samples);
                selected_expert_weights_.resize(num_samples);
                cached_gate_probs_.resize(num_samples);
            }

            // Track expert usage
            std::vector<float> batch_expert_usage(n_experts, 0.0f);
            std::vector<float> routing_fraction(n_experts, 0.0f);
            std::vector<float> gate_prob_sum(n_experts, 0.0f);

            // Step 2: Per-sample routing
            std::vector<std::vector<std::pair<long, float>>> expert_assignments(n_experts);

            for (long n = 0; n < num_samples; ++n) {
                const float* sample_logits = gate_data + n * logits_per_sample;

                // Add Gaussian noise (training only)
                std::vector<float> noisy_logits(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    noisy_logits[e] = sample_logits[e];
                    if (is_training && noise_scale > 0) {
                        static thread_local dlib::rand rnd(std::time(0));
                        noisy_logits[e] += noise_scale * rnd.get_random_gaussian();
                    }
                }

                // Numerically stable softmax
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
                if (is_training)
                    cached_gate_probs_[n] = probs;

                // Select top-k experts
                std::vector<size_t> sorted_indices(n_experts);
                std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
                std::partial_sort(sorted_indices.begin(),
                    sorted_indices.begin() + top_k,
                    sorted_indices.end(),
                    [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

                // Renormalize selected weights to sum to 1
                float selected_sum = 0.0f;
                for (long i = 0; i < top_k; ++i)
                    selected_sum += probs[sorted_indices[i]];

                std::vector<size_t> chosen_experts(top_k);
                std::vector<float> chosen_weights(top_k);
                for (long i = 0; i < top_k; ++i) {
                    chosen_experts[i] = sorted_indices[i];
                    chosen_weights[i] = probs[sorted_indices[i]] / selected_sum;
                    routing_fraction[sorted_indices[i]] += 1.0f;
                    batch_expert_usage[sorted_indices[i]] += 1.0f;
                }

                if (is_training) {
                    selected_expert_indices_[n] = chosen_experts;
                    selected_expert_weights_[n] = chosen_weights;
                }

                for (long i = 0; i < top_k; ++i)
                    expert_assignments[chosen_experts[i]].emplace_back(n, chosen_weights[i]);
            }

            // Step 3: Batched expert processing
            alias_tensor sample_alias(1, k, nr, nc);

            if (is_training) {
                cached_expert_inputs_.resize(n_experts);
                cached_expert_outputs_.resize(n_experts);
            }

            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = expert_assignments[e];
                if (assignments.empty()) continue;

                const long batch_count = static_cast<long>(assignments.size());

                // Gather samples routed to this expert
                resizable_tensor batched_input;
                batched_input.set_size(batch_count, k, nr, nc);
                for (long b = 0; b < batch_count; ++b) {
                    const long src_sample = assignments[b].first;
                    const float* src = expert_input.host() + src_sample * sample_size;
                    float* dst = batched_input.host() + b * sample_size;
                    std::copy(src, src + sample_size, dst);
                }

                // Cache input for backward (no re-forward in backward)
                if (is_training)
                    cached_expert_inputs_[e] = batched_input;

                // Forward through expert
                experts[e].forward(batched_input);
                const tensor& expert_out = experts[e].get_output();

                // Cache output for routing gradient
                if (is_training) {
                    cached_expert_outputs_[e].copy_size(expert_out);
                    tt::copy_tensor(false, cached_expert_outputs_[e], 0, expert_out, 0, expert_out.k());
                }

                // Scatter weighted outputs
                for (long b = 0; b < batch_count; ++b) {
                    const long dst_sample = assignments[b].first;
                    const float weight = assignments[b].second;
                    auto out_s = sample_alias(output, dst_sample * sample_size);
                    auto exp_s = sample_alias(expert_out, b * sample_size);
                    tt::add(1, out_s, weight, exp_s);
                }
            }

            // Step 4: Usage statistics and load balance loss (training only)
            if (is_training) {
                for (long e = 0; e < n_experts; ++e) {
                    routing_fraction[e] /= num_samples;
                    gate_prob_sum[e] /= num_samples;
                }

                load_balance_loss_ = 0.0f;
                for (long e = 0; e < n_experts; ++e)
                    load_balance_loss_ += routing_fraction[e] * gate_prob_sum[e];
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

        // -----------------------------------------------------------------------
        // BACKWARD
        //
        // Phase 1: Batched backward through experts → data grad to main network
        // Phase 2: Compute gate logit gradient (routing + load balance)
        // Phase 3: Gate backward for gate parameter update ONLY
        //          (gate data grad is NOT added to sub.get_gradient_input())
        // Phase 4: Update expert + gate parameters via internal solvers
        //
        // The gate is an internal routing mechanism. Only expert data gradients
        // flow back to the main network. The gate backward computes parameter
        // gradients for the gate's own solver update, nothing more.
        // -----------------------------------------------------------------------
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

            DLIB_CASSERT(num_samples == (long)selected_expert_indices_.size(),
                "Forward pass cache missing or invalid in backward pass");

            // Rebuild per-expert assignments
            std::vector<std::vector<std::pair<long, float>>> expert_assignments(n_experts);
            for (long n = 0; n < num_samples; ++n) {
                const auto& indices = selected_expert_indices_[n];
                const auto& weights = selected_expert_weights_[n];
                for (size_t i = 0; i < indices.size(); ++i)
                    expert_assignments[indices[i]].emplace_back(n, weights[i]);
            }

            // Phase 1: Batched backward through experts
            alias_tensor sample_alias(1, k, nr, nc);

            // Per-sample per-expert importance scores for gate gradient
            std::vector<std::vector<float>> expert_scores;
            if (is_training && learning_rate_multiplier > 0) {
                expert_scores.resize(num_samples);
                for (long n = 0; n < num_samples; ++n)
                    expert_scores[n].assign(n_experts, 0.0f);
            }

            for (long e = 0; e < n_experts; ++e) {
                const auto& assignments = expert_assignments[e];
                if (assignments.empty()) continue;

                const long batch_count = static_cast<long>(assignments.size());
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

                // Expert backward (parameter + data gradients)
                experts[e].back_propagate_error(batched_input, batched_grad);
                const tensor& expert_data_grad = experts[e].get_final_data_gradient();

                // Scatter expert data gradient back to main network input
                for (long b = 0; b < batch_count; ++b) {
                    const long dst_sample = assignments[b].first;
                    auto in_g = sample_alias(input_grad, dst_sample * sample_size);
                    auto ex_g = sample_alias(expert_data_grad, b * sample_size);
                    tt::add(1, in_g, 1, ex_g);
                }

                // Compute routing importance scores
                if (is_training && learning_rate_multiplier > 0) {
                    const tensor& expert_out = cached_expert_outputs_[e];
                    for (long b = 0; b < batch_count; ++b) {
                        const long sample_idx = assignments[b].first;
                        const float* grad_ptr = gradient_input.host() + sample_idx * sample_size;
                        const float* out_ptr = expert_out.host() + b * sample_size;
                        float score = 0.0f;
                        for (long j = 0; j < sample_size; ++j)
                            score += grad_ptr[j] * out_ptr[j];
                        expert_scores[sample_idx][e] = score;
                    }
                }
            }

            // Phase 2 + 3: Gate gradient computation and gate backward
            // Gate backward is for gate parameter update ONLY.
            // Gate data gradient is NOT added to input_grad.
            if (is_training && learning_rate_multiplier > 0) {

                // Build gate logit gradient (same shape as gate output)
                const tensor& gate_output = gate_net.get_output();
                resizable_tensor gate_logit_grad;
                gate_logit_grad.copy_size(gate_output);
                gate_logit_grad = 0;
                float* glg = gate_logit_grad.host();

                // Detect gate output layout
                const long logits_per_sample = n_experts;

                // Part A: Routing gradient — dL/dz_j = P_j * (score_j - E[score])
                for (long n = 0; n < num_samples; ++n) {
                    const auto& probs = cached_gate_probs_[n];
                    const auto& scores = expert_scores[n];

                    float weighted_score_sum = 0.0f;
                    for (long e = 0; e < n_experts; ++e)
                        weighted_score_sum += probs[e] * scores[e];

                    for (long e = 0; e < n_experts; ++e)
                        glg[n * logits_per_sample + e] +=
                            probs[e] * (scores[e] - weighted_score_sum);
                }

                // Part B: Load balance gradient
                if (load_balance_weight > 0) {
                    for (long n = 0; n < num_samples; ++n) {
                        const auto& probs = cached_gate_probs_[n];

                        float sum_wp = 0.0f;
                        for (long e = 0; e < n_experts; ++e) {
                            float w_e = load_balance_weight * n_experts *
                                cached_routing_fraction_[e];
                            sum_wp += w_e * probs[e];
                        }

                        for (long e = 0; e < n_experts; ++e) {
                            float w_e = load_balance_weight * n_experts *
                                cached_routing_fraction_[e];
                            glg[n * logits_per_sample + e] +=
                                probs[e] * (w_e - sum_wp) / static_cast<float>(num_samples);
                        }
                    }
                }

                // Gate backward: computes gate parameter gradients for internal update
                // gate_net.forward(cached_input_) was called during forward
                gate_net.back_propagate_error(cached_input_, gate_logit_grad);

                // NOTE: gate_net.get_final_data_gradient() is deliberately NOT added
                // to input_grad. The gate is an internal routing mechanism; only expert
                // data gradients flow back to the main network.
            }

            // Phase 4: Update expert
            update_subnet_parameters();
        }

        // -----------------------------------------------------------------------
        // Public interface
        // -----------------------------------------------------------------------

        void clean()
        {
            for (auto& expert : experts)
                clean_subnet(expert);
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
        double get_learning_rate() const { return current_learning_rate_; }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;
            for (auto& expert : experts)
                set_all_learning_rate_multipliers(expert, val);
            set_all_learning_rate_multipliers(gate_net, val);
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
            size_t total = count_parameters(gate_net);
            if (!experts.empty())
                total += count_parameters(experts[0]) * static_cast<size_t>(n_experts);
            return total;
        }

        size_t active_parameters() const
        {
            size_t total = count_parameters(gate_net);
            if (!experts.empty())
                total += count_parameters(experts[0]) * static_cast<size_t>(top_k);
            return total;
        }

        EXPERT_NET& get_expert(size_t idx) {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }
        const EXPERT_NET& get_expert(size_t idx) const {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        GATE_NET& get_gate() { return gate_net; }
        const GATE_NET& get_gate() const { return gate_net; }

        long num_experts() const { return n_experts; }
        long num_active_experts() const { return top_k; }
        bool is_training_mode() const { return std::is_same<MODE, training_mode_tag>::value; }
        const std::vector<float>& get_expert_usage() const { return expert_usage; }
        float get_load_balance_loss() const { return load_balance_loss_; }

        // -----------------------------------------------------------------------
        // Serialization
        // -----------------------------------------------------------------------

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
            if (version == "moe_") {
                deserialize(item.n_experts, in);
                deserialize(item.top_k, in);
                deserialize(item.noise_scale, in);
                deserialize(item.usage_update_rate, in);
                deserialize(item.load_balance_weight, in);
                deserialize(item.learning_rate_multiplier, in);
                deserialize(item.current_learning_rate_, in);
                deserialize(item.gate_net, in);
                deserialize(item.experts, in);
                deserialize(item.expert_usage, in);
                deserialize(item.expert_solvers_, in);
                deserialize(item.gate_solvers_, in);
                deserialize(item.solvers_initialized_, in);
            }
            else {
                throw serialization_error("Incorrect version '" + version +
                    "' found while deserializing moe_.");
            }
            item.cached_batch_size_ = 0;
            item.cached_expert_inputs_.clear();
            item.cached_expert_outputs_.clear();
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
        auto clean_subnet(NET& net) -> decltype(net.clean(), void()) { net.clean(); }
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

            // Update only experts activated in this forward pass
            std::set<size_t> used_experts;
            for (const auto& indices : selected_expert_indices_)
                for (size_t idx : indices)
                    used_experts.insert(idx);

            for (size_t e : used_experts)
                experts[e].update_parameters(expert_solvers_[e], effective_lr);

            // Update gate (always has valid gradients)
            gate_net.update_parameters(gate_solvers_, effective_lr);
        }

        // Internal sub-networks
        GATE_NET gate_net;
        std::vector<EXPERT_NET> experts;

        // Configuration
        long n_experts;
        float noise_scale;
        long top_k;
        float usage_update_rate;
        float load_balance_weight;
        double learning_rate_multiplier;
        double current_learning_rate_;

        std::vector<float> expert_usage;

        // Cached tensors for backward
        resizable_tensor cached_input_;
        std::vector<resizable_tensor> cached_expert_inputs_;
        std::vector<resizable_tensor> cached_expert_outputs_;

        // Internal solvers
        std::vector<std::vector<adamw>> expert_solvers_;
        std::vector<adamw> gate_solvers_;
        bool solvers_initialized_;
        double solver_weight_decay_ = 0.004;
        double solver_beta1_ = 0.9;
        double solver_beta2_ = 0.999;

        // Forward/backward routing cache (training only)
        std::vector<std::vector<size_t>> selected_expert_indices_;
        std::vector<std::vector<float>> selected_expert_weights_;
        std::vector<std::vector<float>> cached_gate_probs_;
        std::vector<float> cached_routing_fraction_;
        std::vector<float> cached_gate_prob_avg_;
        long cached_batch_size_;
        float load_balance_loss_;

        resizable_tensor params;
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