// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_UTILITIES_ABSTRACT_H_
#ifdef DLIB_DNn_UTILITIES_ABSTRACT_H_

#include "core_abstract.h"
#include "../geometry/vector_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    double log1pexp(
        double x
    );
    /*!
        ensures
            - returns log(1+exp(x))
              (except computes it using a numerically accurate method)
    !*/

// ----------------------------------------------------------------------------------------

    void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    );
    /*!
        ensures
            - This function assigns random values into params based on the given random
              number generator.  In particular, it uses the parameter initialization method
              of formula 16 from the paper "Understanding the difficulty of training deep
              feedforward neural networks" by Xavier Glorot and Yoshua Bengio.
            - It is assumed that the total number of inputs and outputs from the layer is
              num_inputs_and_outputs.  That is, you should set num_inputs_and_outputs to
              the sum of the dimensionalities of the vectors going into and out of the
              layer that uses params as its parameters.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename LayerType>
    class layer_helper
    {
        /*!
            REQUIREMENTS ON LayerType
                - LayerType is default-constructible.
                - LayerType implements the layer interface used by add_layer<>:
                  template <typename SUBNET> void setup(const SUBNET& sub);
                  template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
                  template <typename SUBNET> void backward(const tensor& gradient_input,
                                                           SUBNET& sub,
                                                           tensor& params_grad);
                - In setup() and forward(), LayerType reads only sub.get_output() from the
                  SUBNET argument; it does not require access to deeper layers via layer<>().
                - In backward(), LayerType writes to sub.get_gradient_input(); the
                  params_grad argument is ignored (LayerType is parameter-free or its
                  parameters are managed internally without participating in the host
                  layer's parameter optimization).

            WHAT THIS OBJECT REPRESENTS
                This object is a generic adapter that lets you reuse any parameter-free
                Dlib layer as a sub-component inside another layer's forward/backward
                implementation, without having to hand-roll a SUBNET-like wrapper for
                each call site.

                The motivating use case is a fused composite layer (e.g. a
                grouped-query attention block consolidating linear projections,
                rotary positional embedding, causal masking, softmax, and matmul
                into a single Dlib layer). Such a layer typically wants to delegate
                some of its sub-computations to existing parameter-free Dlib layers
                like rotary_positional_embedding_ or tril_ rather than reimplement
                their internal state and edge cases.

                Examples of compatible LayerTypes:
                    - tril_<...>                       (causal masking)
                    - rotary_positional_embedding_     (RoPE with optional YaRN)

                Examples of NOT compatible LayerTypes (have trainable parameters
                that would need to participate in the host layer's parameter
                optimization):
                    - linear_, fc_, con_, rms_norm_, layer_norm_

                The wrapped layer's internal state (caches, masks, position-
                dependent precomputations, etc.) persists across forward/backward
                calls on the same layer_helper instance. The wrapped layer is
                serialized and deserialized along with the helper so that
                save/load round-trips preserve any precomputed state.

            THREAD SAFETY
                This object is not thread-safe. The wrapped layer maintains internal
                state that is mutated by setup/forward/backward. Concurrent calls on
                the same instance from different threads are not supported.
        !*/

    public:

        layer_helper(
        );
        /*!
            ensures
                - The wrapped layer is default-constructed.
        !*/

        LayerType& get(
        );
        /*!
            ensures
                - returns a non-const reference to the wrapped layer instance, so
                  callers can configure it (e.g. set hyperparameters) before the
                  first forward call.
        !*/

        const LayerType& get(
        ) const;
        /*!
            ensures
                - returns a const reference to the wrapped layer instance.
        !*/

        void setup(
            const tensor& input_proxy
        );
        /*!
            ensures
                - Invokes the wrapped layer's setup() method, with input_proxy
                  standing in for sub.get_output() of a normal SUBNET.
                - Useful when the host layer wants to pre-allocate caches or
                  internal state that depend on the expected input shape, before
                  the first forward call.
                - input_proxy does not need to contain meaningful data; only its
                  shape is read by typical layer setup() implementations.
        !*/

        void forward(
            const tensor& input,
            resizable_tensor& output
        );
        /*!
            ensures
                - Applies the wrapped layer's forward computation to input and
                  writes the result to output.
                - output is resized as needed by the wrapped layer's forward()
                  implementation.
                - The wrapped layer's internal state may be updated (e.g. caches
                  rebuilt if the input shape changed since the previous call).
        !*/

        void backward(
            const tensor& gradient_input,
            const tensor& original_input,
            resizable_tensor& input_grad
        );
        /*!
            requires
                - gradient_input has the shape of the output produced by the
                  most recent forward() call (i.e. dL/d(output)).
                - original_input is the same tensor (or numerically equivalent)
                  that was passed to the most recent forward() call.
            ensures
                - Computes the gradient of the loss with respect to the input of
                  the wrapped layer and writes it into input_grad.
                - input_grad is resized to match original_input and zero-
                  initialized before the wrapped layer's backward() is called.
                  This ensures that wrapped layers using add-to semantics for
                  their input gradient (the Dlib convention) produce a clean
                  stand-alone gradient rather than accumulating into stale data.
                - The wrapped layer is treated as parameter-free: any params_grad
                  that the wrapped layer's backward() tries to populate is
                  discarded.
        !*/

        friend void serialize(
            const layer_helper& item,
            std::ostream& out
        );
        /*!
            ensures
                - Serializes item to out by delegating to the wrapped layer's
                  own serialize() function. A version tag is prepended so future
                  changes to the helper format can be detected at load time.
        !*/

        friend void deserialize(
            layer_helper& item,
            std::istream& in
        );
        /*!
            ensures
                - Deserializes item from in.
            throws
                - serialization_error if the data on the stream was not produced
                  by a compatible serialize() call (in particular, if the
                  version tag does not match).
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_ABSTRACT_H_ 


