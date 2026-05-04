// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "../cuda/tensor.h"
#include "utilities_abstract.h"
#include "../geometry.h"
#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        for (auto& val : params)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        label_type label{};
        float weight = 1.f;
    };

// ----------------------------------------------------------------------------------------

    inline double log1pexp(double x)
    {
        using std::exp;
        if (x <= -37)
            return exp(x);
        else if (-37 < x && x <= 18)
            return log1p(exp(x));
        else if (18 < x && x <= 33.3)
            return x + exp(-x);
        else
            return x;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    T safe_log(T input, T epsilon = 1e-10)
    {
        // Prevent trying to calculate the logarithm of a very small number (let alone zero)
        return std::log(std::max(input, epsilon));
    }

// ----------------------------------------------------------------------------------------

    inline size_t tensor_index(
        const tensor& t,
        const long sample,
        const long k,
        const long r,
        const long c
    )
    {
        return ((sample * t.k() + k) * t.nr() + r) * t.nc() + c;
    }

// ----------------------------------------------------------------------------------------

    template <typename LayerType>
    class layer_helper
    {
    public:
        layer_helper() = default;

        LayerType& get() { return layer_; }
        const LayerType& get() const { return layer_; }

        // Run the wrapped layer's setup with `input_proxy` standing in for
        // sub.get_output(). Useful when the host layer wants to pre-allocate
        // caches that depend on input shape
        void setup(const tensor& input_proxy)
        {
            forward_subnet adapter{ input_proxy };
            layer_.setup(adapter);
        }

        // Forward: applies the wrapped layer to `input` and writes to `output`
        void forward(const tensor& input, resizable_tensor& output)
        {
            forward_subnet adapter{ input };
            layer_.forward(adapter, output);
        }

        // Backward: given the gradient w.r.t. the layer's output and the
        // original input that was passed to forward, computes the gradient
        // w.r.t. that input and writes it into `input_grad`
        //
        // The destination `input_grad` is initialised to zero internally so
        // that wrapped layers using add-to semantics in their backward
        // (most Dlib layers) produce the expected stand-alone gradient
        void backward(
            const tensor& gradient_input,
            const tensor& original_input,
            resizable_tensor& input_grad)
        {
            input_grad.copy_size(original_input);
            input_grad = 0;
            backward_subnet adapter{ original_input, input_grad };
            resizable_tensor unused_params_grad;
            layer_.backward(gradient_input, adapter, unused_params_grad);
        }

        friend void serialize(const layer_helper& item, std::ostream& out)
        {
            serialize("layer_helper_v1", out);
            serialize(item.layer_, out);
        }

        friend void deserialize(layer_helper& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "layer_helper_v1")
                throw serialization_error(
                    "Unexpected version while deserializing layer_helper: " + version);
            deserialize(item.layer_, in);
        }

    private:
        LayerType layer_;

        // Adapter exposing only get_output() (used by setup and forward)
        struct forward_subnet
        {
            const tensor& output_ref;
            const tensor& get_output() const { return output_ref; }
        };

        // Adapter exposing get_output() and get_gradient_input() for backward
        struct backward_subnet
        {
            const tensor& output_ref;
            tensor& gradient_input_ref;
            const tensor& get_output() const { return output_ref; }
            tensor& get_gradient_input() { return gradient_input_ref; }
        };
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



