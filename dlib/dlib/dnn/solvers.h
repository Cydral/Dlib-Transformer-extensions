// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_SOLVERS_H_
#define DLIB_DNn_SOLVERS_H_

#include "solvers_abstract.h"
#include "../cuda/tensor.h"
#include <iostream>
#include "layers.h"

namespace dlib
{
    class sgd
    {
    public:

        explicit sgd(
            float weight_decay_,
            float momentum_ = 0.9
        ) 
        { 
            weight_decay = weight_decay_;
            momentum = momentum_;
        }

        sgd(
        ) : sgd(0.0005f, 0.9f)
        { 
        }

        float get_momentum (
        ) const { return momentum; }

        float get_weight_decay (
        ) const { return weight_decay; }

        template <typename layer_type> 
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
        )
        {
            const tensor& params = l.get_layer_params();

            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                v.copy_size(params_grad);
                v = 0;
            }

            const double lr = learning_rate*get_learning_rate_multiplier(l);
            const double wd = weight_decay*get_weight_decay_multiplier(l);
            
            //perform: v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);
            tt::affine_transform(v, v, params, params_grad, momentum, -wd*lr, -lr);

            return v;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N,FC_HAS_BIAS>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.get_num_outputs());
            return v;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return v;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return v;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()/2);
            return v;
        }

        friend void serialize(const sgd& item, std::ostream& out)
        {
            serialize("sgd2", out);
            serialize(item.v, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum, out);
        }

        friend void deserialize(sgd& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "sgd2")
                throw serialization_error("Unexpected version found while deserializing dlib::sgd.");
            deserialize(item.v, in);
            deserialize(item.weight_decay, in);
            deserialize(item.momentum, in);
        }

        friend std::ostream& operator<< (std::ostream& out, const sgd& item)
        {
            out << "sgd: weight_decay="<<item.get_weight_decay() << ", momentum="<<item.get_momentum(); 
            return out;
        }

    private:

        template <typename layer_type> 
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();

            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                v.copy_size(params_grad);
                v = 0;
            }

            double lr = learning_rate*get_learning_rate_multiplier(l);
            double wd = weight_decay*get_weight_decay_multiplier(l);
            
            //perform: v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                tt::affine_transform(v, v, params, params_grad, momentum, -wd*lr, -lr);
            }
            else
            {

                tt::affine_transform_range(0, bias_offset, v, v, params, params_grad, momentum, -wd*lr, -lr);

                // now update the biases but apply their multipliers
                lr *= l.get_bias_learning_rate_multiplier();
                wd *= l.get_bias_weight_decay_multiplier();
                tt::affine_transform_range(bias_offset, v.size(), v, v, params, params_grad, momentum, -wd*lr, -lr);
            }
        }

        resizable_tensor v;
        float weight_decay;
        float momentum;

    };

// ----------------------------------------------------------------------------------------

    class adam 
    {
    public:

        adam(
            float weight_decay_,
            float momentum1_, 
            float momentum2_
        ) 
        { 
            weight_decay = weight_decay_;
            momentum1 = momentum1_;
            momentum2 = momentum2_;
            t = 0;
        }

        adam(
        ) : adam(0.0005f, 0.9f, 0.999f)
        {}

        float get_momentum1 (
        ) const { return momentum1; }

        float get_momentum2 (
        ) const { return momentum2; }

        float get_weight_decay (
        ) const { return weight_decay; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
        )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }

            ++t;

            
            tt::compute_adam_update(0, params.size(), s, m, v, t,
                learning_rate*get_learning_rate_multiplier(l),
                weight_decay*get_weight_decay_multiplier(l), 
                momentum1, momentum2, params, params_grad);

            return s;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N,FC_HAS_BIAS>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.get_num_outputs());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return s;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()/2);
            return s;
        }


        friend void serialize(const adam& item, std::ostream& out)
        {
            serialize("adam2", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adam& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adam2")
                throw serialization_error("Unexpected version found while deserializing dlib::adam.");
            deserialize(item.m, in);
            deserialize(item.v, in);
            deserialize(item.s, in);
            deserialize(item.weight_decay, in);
            deserialize(item.momentum1, in);
            deserialize(item.momentum2, in);
            deserialize(item.t, in);
        }

        friend std::ostream& operator<< (std::ostream& out, const adam& item)
        {
            out << "adam: weight_decay="<<item.get_weight_decay() << ", momentum1="<<item.get_momentum1() << ", momentum2="<<item.get_momentum2(); 
            return out;
        }

    private:

        template <typename layer_type> 
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }


            ++t;

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                tt::compute_adam_update(0, params.size(), s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l),
                    weight_decay*get_weight_decay_multiplier(l), 
                    momentum1, momentum2, params, params_grad);
            }
            else
            {
                tt::compute_adam_update(0, bias_offset, s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l),
                    weight_decay*get_weight_decay_multiplier(l), 
                    momentum1, momentum2, params, params_grad);

                tt::compute_adam_update(bias_offset, params.size(), s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l)*l.get_bias_learning_rate_multiplier(),
                    weight_decay*get_weight_decay_multiplier(l)*l.get_bias_weight_decay_multiplier(), 
                    momentum1, momentum2, params, params_grad);
            }
        }
        resizable_tensor m;
        resizable_tensor v;
        resizable_tensor s;
        float weight_decay;
        float momentum1;
        float momentum2;
        float t;
    };

// ----------------------------------------------------------------------------------------

    /*!
        AdamW optimizer with decoupled weight decay regularization.

        This optimizer implements the AdamW algorithm from "Decoupled Weight Decay
        Regularization" (Loshchilov & Hutter, ICLR 2019). Unlike standard Adam,
        AdamW decouples the weight decay from the gradient-based optimization step,
        leading to better generalization and easier hyperparameter tuning.

        THEORETICAL FOUNDATION:
            Standard Adam with L2 regularization computes:
                theta_t = theta_{t-1} - alpha * m_hat_t / sqrt(v_hat_t + epsilon)
                where gradients include the L2 regularization term

            AdamW decouples weight decay and computes:
                m_t = beta1 * m_{t-1} + (1-beta1) * gradient_L
                v_t = beta2 * v_{t-1} + (1-beta2) * (gradient_L)^2
                theta_t = theta_{t-1} - alpha * (m_hat_t/sqrt(v_hat_t) + lambda*theta_{t-1})

            This formulation makes the optimal weight decay factor independent of
            the learning rate, improving generalization especially for long training runs.

        IMPLEMENTATION STRATEGY:
            1. Compute standard Adam update with weight_decay = 0 (decoupled)
            2. Explicitly apply weight decay: update = update - lr * wd * params
            3. The update is then added to parameters by the trainer

        KEY DIFFERENCES FROM ADAM:
            - Weight decay is applied directly to parameters (multiplicative)
            - Weight decay does not interact with adaptive learning rates
            - Better hyperparameter independence (learning rate vs weight decay)
            - Superior generalization on image classification and NLP tasks

        CONSTRUCTOR PARAMETERS:
            - weight_decay: Decoupled weight decay coefficient (default: 0.01)
                           Typical range: 0.0001 to 0.1
                           Higher values = stronger regularization
            - momentum1 (beta1): Exponential decay rate for first moment (default: 0.9)
                                Controls the momentum of gradient moving average
            - momentum2 (beta2): Exponential decay rate for second moment (default: 0.999)
                                Controls the momentum of squared gradient moving average

        REFERENCES:
            - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
              ICLR 2019. https://arxiv.org/abs/1711.05101
            - Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
              ICLR 2015. https://arxiv.org/abs/1412.6980

        NOTE: AdamW is the standard optimizer for modern transformer models including
              GPT, BERT, LLaMA, Mistral, Qwen, DeepSeek, and other large language models.
              It consistently outperforms standard Adam with L2 regularization.
    !*/
    class adamw
    {
    public:

        explicit adamw(
            float weight_decay_ = 0.01f,
            float momentum1_ = 0.9f,
            float momentum2_ = 0.999f
        )
        {
            weight_decay = weight_decay_;
            momentum1 = momentum1_;
            momentum2 = momentum2_;
            t = 0;
        }

        float get_momentum1() const { return momentum1; }
        float get_momentum2() const { return momentum2; }
        float get_weight_decay() const { return weight_decay; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
            )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);

            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
                s = 0;
            }

            ++t;

            // Step 1: compute standard Adam update with decoupled weight decay (wd = 0)
            // This populates 's' with the adaptive gradient step: -alpha * m_hat_t / sqrt(v_hat_t)
            // By passing weight_decay = 0, we decouple the regularization from the adaptive update
            tt::compute_adam_update(0, params.size(), s, m, v, t,
                learning_rate * get_learning_rate_multiplier(l),
                0, // Critical: weight_decay = 0 for decoupled regularization
                momentum1, momentum2, params, params_grad);

            // Step 2: apply decoupled weight decay explicitly
            // Formula: s = s - alpha * lambda * theta_{t-1}
            // This implements the AdamW update: theta_t = theta_{t-1} - alpha * (m_hat_t/sqrt(v_hat_t) + lambda * theta_{t-1})
            const double lr = learning_rate * get_learning_rate_multiplier(l);
            const double wd = weight_decay * get_weight_decay_multiplier(l);

            if (wd != 0)
            {
                // Compute: s = s + params * (-lr * wd)
                tt::affine_transform(s, s, params, 1.0, -lr * wd);
            }

            return s;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N, FC_HAS_BIAS>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.get_num_outputs());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
        >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
        >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
            return s;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() / 2);
            return s;
        }

        friend void serialize(const adamw& item, std::ostream& out)
        {
            serialize("adamw", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adamw& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adamw")
                throw serialization_error("Unexpected version found while deserializing dlib::adamw.");
            item.deserialize_state(in);
        }

        friend std::ostream& operator<< (std::ostream& out, const adamw& item)
        {
            out << "adamw: weight_decay=" << item.get_weight_decay()
                << ", momentum1=" << item.get_momentum1()
                << ", momentum2=" << item.get_momentum2();
            return out;
        }

    private:

        /* What follows the tag in the serialized form. A subnet_solver reads its tag itself
           to learn which algorithm the stream holds, then hands over the rest. */
        friend class subnet_solver;
        void deserialize_state(std::istream& in)
        {
            deserialize(m, in);
            deserialize(v, in);
            deserialize(s, in);
            deserialize(weight_decay, in);
            deserialize(momentum1, in);
            deserialize(momentum2, in);
            deserialize(t, in);
        }

        /*!
            Updates parameters that may have different learning rate and weight decay
            multipliers for weights vs biases (e.g., fully connected and convolutional layers).

            BIAS HANDLING:
                Most layers separate weights and biases:
                - Weights: indices [0, bias_offset)
                - Biases: indices [bias_offset, end)

                Different multipliers may apply to each section:
                - bias_learning_rate_multiplier (typically 1.0 or 2.0)
                - bias_weight_decay_multiplier (typically 0.0 - no decay on biases)

            PARAMETERS:
                - learning_rate: base learning rate from trainer
                - l: layer containing parameters and multiplier settings
                - params_grad: gradient tensor
                - bias_offset: index where biases start in the parameter tensor
        !*/
        template <typename layer_type>
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);

            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }

            /* The step counter drives Adam's bias correction, whose factor is
               sqrt(1 - momentum2^t) / (1 - momentum1^t). At t == 0 that reads 0/0, so a
               parameter blob reached through this overload came out entirely NaN on its
               first update while the generic operator(), which does increment, was fine.
               Only fc_ and con_ route here, which is why a stack built from linear_ never
               showed it.  */
            ++t;

            // Step 1: compute adaptive gradient update with decoupled weight decay
            if (l.get_bias_learning_rate_multiplier() == 1)
            {
                // Simple case: uniform learning rate for all parameters
                tt::compute_adam_update(0, params.size(), s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l),
                    0, // Decoupled: weight_decay = 0 in Adam computation
                    momentum1, momentum2, params, params_grad);
            }
            else
            {
                // Complex case: different learning rates for weights and biases

                // Process weights: indices [0, bias_offset)
                tt::compute_adam_update(0, bias_offset, s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l),
                    0, // Decoupled weight decay
                    momentum1, momentum2, params, params_grad);

                // Process biases: indices [bias_offset, end)
                // Apply bias learning rate multiplier
                tt::compute_adam_update(bias_offset, params.size(), s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l) * l.get_bias_learning_rate_multiplier(),
                    0, // Decoupled weight decay
                    momentum1, momentum2, params, params_grad);
            }

            // Step 2: apply decoupled weight decay
            // Formula: s = s - lr * wd * params
            // This is applied separately to weights and biases because they may have
            // different weight decay multipliers
            double lr = learning_rate * get_learning_rate_multiplier(l);
            double wd = weight_decay * get_weight_decay_multiplier(l);

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                // Simple case: uniform weight decay for all parameters
                if (wd != 0)
                    tt::affine_transform(s, s, params, 1.0, -lr * wd);
            }
            else
            {
                // Complex case: different weight decay for weights vs biases

                // Apply weight decay to weights: indices [0, bias_offset)
                // Computation: s[i] = 1.0 * s[i] + (-lr * wd) * params[i] + 0.0 * params[i]
                // The third source (params) is not used since C = 0.0
                if (wd != 0)
                {
                    tt::affine_transform_range(0, bias_offset,
                        s,          // dest
                        s,          // src1 (A coefficient)
                        params,     // src2 (B coefficient) 
                        params,     // src3 (C coefficient = 0, so this is unused)
                        1.0,        // A: keep current update
                        -lr * wd,   // B: subtract weight decay term
                        0.0);       // C: ignore third source
                }

                // Apply weight decay to biases: indices [bias_offset, end)
                // Note: typically bias_weight_decay_multiplier = 0 (no regularization on biases)
                // This is a common practice in deep learning to prevent biases from becoming too small
                lr *= l.get_bias_learning_rate_multiplier();
                wd *= l.get_bias_weight_decay_multiplier();

                if (wd != 0)
                {
                    tt::affine_transform_range(bias_offset, v.size(),
                        s,
                        s,
                        params,
                        params,
                        1.0,
                        -lr * wd,
                        0.0);
                }
            }
        }

        resizable_tensor m;  // First moment estimate (exponential moving average of gradients)
        resizable_tensor v;  // Second moment estimate (exponential moving average of squared gradients)
        resizable_tensor s;  // Parameter update computed by the optimizer
        float weight_decay;  // Weight decay coefficient (lambda in the paper)
        float momentum1;     // Beta1: decay rate for first moment
        float momentum2;     // Beta2: decay rate for second moment
        float t;             // Time step counter for bias correction
    };

// ----------------------------------------------------------------------------------------

    /*!
        Adam-atan2, the epsilon-free variant of Adam with decoupled weight decay.

        Adam divides the first moment by the square root of the second plus an epsilon.
        The epsilon sets a scale below which the update stops being adaptive, and its
        right value depends on the size of the gradients, which varies with the layer, the
        loss and the stage of training. Adam-atan2 replaces the quotient by an arctangent:
            m_t = beta1 * m_{t-1} + (1-beta1) * g
            v_t = beta2 * v_{t-1} + (1-beta2) * g^2
            theta_t = theta_{t-1} - lr * wd * theta_{t-1}
                      - lr / (1-beta1^t) * atan2(m_t, sqrt(v_t / (1-beta2^t)))
        For a small quotient the arctangent is close to the quotient, so the update is
        close to Adam's. For a large one it saturates at pi/2, so no step exceeds pi/2
        times the step size, and scaling the gradient by any positive factor leaves the
        update unchanged.

        The bias correction of the first moment scales the step and stays out of the
        arctangent, as in the implementation the hierarchical reasoning model is trained
        with, so that a model trained here and one trained there follow the same rule.

        REFERENCES:
            - Everett et al. (2024). "Scaling Exponents Across Parameterizations and
              Optimizers". ICML 2024. https://arxiv.org/abs/2407.05872
            - Wang et al. (2025). "Hierarchical Reasoning Model".
              https://arxiv.org/abs/2506.21734
    !*/
    class adam_atan2
    {
    public:

        explicit adam_atan2(
            float weight_decay_ = 0.01f,
            float momentum1_ = 0.9f,
            float momentum2_ = 0.999f
        ) :
            weight_decay(weight_decay_),
            momentum1(momentum1_),
            momentum2(momentum2_),
            t(0)
        {
        }

        float get_momentum1() const { return momentum1; }
        float get_momentum2() const { return momentum2; }
        float get_weight_decay() const { return weight_decay; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
            )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            allocate(params_grad);
            ++t;

            tt::compute_adam_atan2_update(0, params.size(), s, m, v, t,
                learning_rate*get_learning_rate_multiplier(l),
                weight_decay*get_weight_decay_multiplier(l),
                momentum1, momentum2, params, params_grad);
            return s;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N, FC_HAS_BIAS>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.get_num_outputs());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
        >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
        >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
            return s;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() / 2);
            return s;
        }

        friend void serialize(const adam_atan2& item, std::ostream& out)
        {
            serialize("adam_atan2", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adam_atan2& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adam_atan2")
                throw serialization_error("Unexpected version found while deserializing dlib::adam_atan2.");
            item.deserialize_state(in);
        }

        friend std::ostream& operator<< (std::ostream& out, const adam_atan2& item)
        {
            out << "adam_atan2: weight_decay=" << item.get_weight_decay()
                << ", momentum1=" << item.get_momentum1()
                << ", momentum2=" << item.get_momentum2();
            return out;
        }

    private:

        friend class subnet_solver;
        void deserialize_state(std::istream& in)
        {
            deserialize(m, in);
            deserialize(v, in);
            deserialize(s, in);
            deserialize(weight_decay, in);
            deserialize(momentum1, in);
            deserialize(momentum2, in);
            deserialize(t, in);
        }

        void allocate(const tensor& params_grad)
        {
            if (v.size() != 0)
                return;
            m.copy_size(params_grad);
            m = 0;
            v.copy_size(params_grad);
            v = 0;
            s.copy_size(params_grad);
            s = 0;
        }

        /*
            The weights and the biases of fc_, con_, cont_ and bn_ may carry multipliers of
            their own. The decay is decoupled and computed in the same pass as the step, so
            each part is one call with its own rate and its own decay. The step counter is
            shared, since both parts belong to the same update.
        */
        template <typename layer_type>
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            allocate(params_grad);
            ++t;

            const float lr = learning_rate*get_learning_rate_multiplier(l);
            const float wd = weight_decay*get_weight_decay_multiplier(l);

            tt::compute_adam_atan2_update(0, bias_offset, s, m, v, t,
                lr, wd, momentum1, momentum2, params, params_grad);

            tt::compute_adam_atan2_update(bias_offset, params.size(), s, m, v, t,
                lr*l.get_bias_learning_rate_multiplier(),
                wd*l.get_bias_weight_decay_multiplier(),
                momentum1, momentum2, params, params_grad);
        }

        resizable_tensor m;  // First moment estimate
        resizable_tensor v;  // Second moment estimate
        resizable_tensor s;  // Parameter update computed by the optimizer
        float weight_decay;  // Decoupled weight decay coefficient
        float momentum1;     // Beta1: decay rate for first moment
        float momentum2;     // Beta2: decay rate for second moment
        float t;             // Time step counter for bias correction
    };

// ----------------------------------------------------------------------------------------

    /*!
        The solver a layer uses for a sub-network whose parameters it updates itself.

        hrm_, moe_ and the modality fusion layer each own networks that the trainer never
        sees, and update them inside their backward pass. The trainer's solver cannot reach
        them, so its algorithm and settings reach them through network_context, and this
        object applies whichever of AdamW and Adam-atan2 it names.

        It serializes exactly as the solver it applies. A vector of these that applies
        AdamW writes what a vector of adamw writes, and reads it back, so a model saved
        before the choice existed loads unchanged.
    !*/
    class subnet_solver
    {
    public:

        subnet_solver(
        ) : type(network_context::optimizer_type::adamw)
        {
        }

        subnet_solver(
            network_context::optimizer_type type_,
            float weight_decay,
            float momentum1,
            float momentum2
        ) :
            type(type_),
            adamw_solver(weight_decay, momentum1, momentum2),
            atan2_solver(weight_decay, momentum1, momentum2)
        {
        }

        /*!
            The algorithm network_context names when it is active, AdamW otherwise.
        !*/
        static network_context::optimizer_type requested_type()
        {
            return network_context::is_active()
                ? network_context::get_optimizer_type()
                : network_context::optimizer_type::adamw;
        }

        network_context::optimizer_type get_type() const { return type; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
            )
        {
            if (type == network_context::optimizer_type::adam_atan2)
                return atan2_solver(learning_rate, l, params_grad);
            return adamw_solver(learning_rate, l, params_grad);
        }

        friend void serialize(const subnet_solver& item, std::ostream& out)
        {
            if (item.type == network_context::optimizer_type::adam_atan2)
                serialize(item.atan2_solver, out);
            else
                serialize(item.adamw_solver, out);
        }

        friend void deserialize(subnet_solver& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            item.deserialize_state(version, in);
        }

        friend std::ostream& operator<< (std::ostream& out, const subnet_solver& item)
        {
            if (item.type == network_context::optimizer_type::adam_atan2)
                out << item.atan2_solver;
            else
                out << item.adamw_solver;
            return out;
        }

    private:
        /* Reads the state that follows the tag into the solver the tag names. The friendship
           adamw and adam_atan2 grant to this class covers its member functions on every
           compiler, while MSVC denies it to the friend functions defined in the class. */
        void deserialize_state(const std::string& version, std::istream& in)
        {
            if (version == "adamw")
            {
                type = network_context::optimizer_type::adamw;
                adamw_solver.deserialize_state(in);
            }
            else if (version == "adam_atan2")
            {
                type = network_context::optimizer_type::adam_atan2;
                atan2_solver.deserialize_state(in);
            }
            else
            {
                throw serialization_error("Unexpected version '" + version +
                    "' found while deserializing dlib::subnet_solver.");
            }
        }

        network_context::optimizer_type type;
        adamw adamw_solver;
        adam_atan2 atan2_solver;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_H_

