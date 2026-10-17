// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_SOLVERS_ABSTRACT_H_
#ifdef DLIB_DNn_SOLVERS_ABSTRACT_H_

#include "../cuda/tensor_abstract.h"
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class EXAMPLE_SOLVER 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A solver defines the parameter update rule for a single layer in a deep
                neural network.  It takes a parameter gradient vector and the layer's
                parameters and tells you how the parameters should be updated.
                Importantly, each solver instance is used with only one layer in a network.
                This allows us to define solvers that have per layer state, for example, a
                solver may keep a momentum term and apply it to its update rule.

                Note that there is no dlib::EXAMPLE_SOLVER type.  It is shown here purely
                to document the interface a solver object must implement.
        !*/

    public:

        EXAMPLE_SOLVER(
        );

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
        )
        /*!
            requires
                - l.get_layer_params().size() != 0
                - have_same_dimensions(l.get_layer_params(), params_grad) == true.
                - When this function is invoked on a particular solver instance, it is
                  always supplied with the same layer instance, l.  That is, the solver is
                  allowed to remember things from one invocation to another and to assume
                  that it is being serially applied to optimize the same layer's
                  parameters. 
            ensures
                - Returns a step vector V that is intended to be used to update the
                  parameters by adding V to l.get_layer_params().
                - This function will use the given "learning rate" to compute V.  How the
                  learning rate is used is solver dependent.  But in general the learning
                  rate should be used to select the step size, i.e. to somehow determine
                  the magnitude of V.
        !*/
    };

    void serialize(const EXAMPLE_SOLVER& item, std::ostream& out);
    void deserialize(EXAMPLE_SOLVER& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    std::ostream& operator<< (std::ostream& out, const EXAMPLE_SOLVER& item);
    /*!
        Prints the solver's name and parameters to out.
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class sgd
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above.  It is a
                basic stochastic gradient descent solver which uses momentum and weight
                decay.  In particular, it computes the update vector V according to:
                    V = momentum*V - weight_decay*learning_rate*l.get_layer_params() - learning_rate*params_grad;
                Here V is a momentum term that is remembered by the solver from one
                invocation of operator() to the next.  


                Note that the actual learning rate and weight decay used by the solver are
                multiplied by the per layer multipliers.  That is, the solver will call
                get_learning_rate_multiplier(l) and get_weight_decay_multiplier(l) and
                multiply these values with the nominal learning rate and weight decay,
                respectively, to determine the values it will use during each step.  It is
                also overloaded to allow additional learning rate multipliers to be applied
                to fc_ and con_ bias parameters.
        !*/
    public:

        sgd(
        ); 
        /*!
            ensures
                - #get_weight_decay()  == 0.0005 
                - #get_momentum()      == 0.9 
        !*/

        explicit sgd(
            float weight_decay,
            float momentum = 0.9
        ); 
        /*!
            requires
                - weight_decay >= 0
                - momentum >= 0
            ensures
                - #get_weight_decay()  == weight_decay 
                - #get_momentum()      == momentum 
        !*/

        float get_weight_decay () const;
        float get_momentum () const; 
    };

    void serialize(const sgd& item, std::ostream& out);
    void deserialize(sgd& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    std::ostream& operator<< (std::ostream& out, const sgd& item);
    /*!
        Prints the solver's name and parameters to out.
    !*/

// ----------------------------------------------------------------------------------------

    class adam
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above.  In
                particular, it implements the ADAM parameter update method described in the
                paper:
                    Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic
                    optimization." International Conference on Learning Representation. 2015.


                Note that the actual learning rate and weight decay used by the solver are
                multiplied by the per layer multipliers.  That is, the solver will call
                get_learning_rate_multiplier(l) and get_weight_decay_multiplier(l) and
                multiply these values with the nominal learning rate and weight decay,
                respectively, to determine the values it will use during each step.  It is
                also overloaded to allow additional learning rate multipliers to be applied
                to fc_ and con_ bias parameters.
        !*/

    public:

        adam(
        ); 
        /*!
            ensures
                - #get_weight_decay()  == 0.0005 
                - #get_momentum1()     == 0.9 
                - #get_momentum2()     == 0.999 
        !*/

        adam(
            float weight_decay,
            float momentum1, 
            float momentum2 
        ); 
        /*!
            requires
                - weight_decay >= 0
                - 0 <= momentum1 < 1
                - 0 <= momentum2 < 1
            ensures
                - #get_weight_decay()  == weight_decay 
                - #get_momentum1()     == momentum1
                - #get_momentum2()     == momentum2
        !*/

        float get_weight_decay () const;
        float get_momentum1 () const; 
        float get_momentum2 () const; 
    };

    void serialize(const adam& item, std::ostream& out);
    void deserialize(adam& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    std::ostream& operator<< (std::ostream& out, const adam& item);
    /*!
        Prints the solver's name and parameters to out.
    !*/

// ----------------------------------------------------------------------------------------

    class adamw
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above. In
                particular, it implements the AdamW parameter update method with decoupled
                weight decay regularization as described in the paper:
                    Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay
                    regularization." International Conference on Learning Representations. 2019.

                The key difference from standard Adam is that weight decay is decoupled from
                the gradient-based optimization step. This leads to better generalization
                performance and makes the optimal weight decay factor more independent of the
                learning rate setting. AdamW has become the standard optimizer for training
                large language models and transformer architectures.

                The update is computed as:
                    m_t = momentum1*m_{t-1} + (1-momentum1)*params_grad
                    v_t = momentum2*v_{t-1} + (1-momentum2)*(params_grad^2)
                    V = -learning_rate * (m_hat_t/sqrt(v_hat_t) + weight_decay*l.get_layer_params())
                where m_hat_t and v_hat_t are bias-corrected moment estimates.

                Note that the actual learning rate and weight decay used by the solver are
                multiplied by the per layer multipliers. That is, the solver will call
                get_learning_rate_multiplier(l) and get_weight_decay_multiplier(l) and
                multiply these values with the nominal learning rate and weight decay,
                respectively, to determine the values it will use during each step. It is
                also overloaded to allow additional learning rate multipliers to be applied
                to fc_ and con_ bias parameters.
        !*/

    public:

        adamw(
        );
        /*!
            ensures
                - #get_weight_decay()  == 0.01
                - #get_momentum1()     == 0.9
                - #get_momentum2()     == 0.999
        !*/

        explicit adamw(
            float weight_decay,
            float momentum1 = 0.9,
            float momentum2 = 0.999
        );
        /*!
            requires
                - weight_decay >= 0
                - 0 <= momentum1 < 1
                - 0 <= momentum2 < 1
            ensures
                - #get_weight_decay()  == weight_decay
                - #get_momentum1()     == momentum1
                - #get_momentum2()     == momentum2
        !*/

        float get_weight_decay() const;
        float get_momentum1() const;
        float get_momentum2() const;
    };

    void serialize(const adamw& item, std::ostream& out);
    void deserialize(adamw& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    std::ostream& operator<< (std::ostream& out, const adamw& item);
    /*!
        Prints the solver's name and parameters to out.
    !*/

// ----------------------------------------------------------------------------------------

    class adam_atan2
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above. In
                particular, it implements the Adam-atan2 parameter update method described
                in the paper:
                    Everett, Katie, et al. "Scaling Exponents Across Parameterizations and
                    Optimizers." International Conference on Machine Learning. 2024.

                Adam divides the first moment by the square root of the second plus an
                epsilon, whose right value depends on the scale of the gradients. Adam-atan2
                takes the arctangent of the two instead. It needs no epsilon, it bounds
                every step by pi/2 times the step size, and it leaves the update unchanged
                when the gradient is scaled by any positive factor. Where the quotient is
                small the arctangent is close to it, and so is the update to Adam's.

                The update is computed as:
                    m_t = momentum1*m_{t-1} + (1-momentum1)*params_grad
                    v_t = momentum2*v_{t-1} + (1-momentum2)*(params_grad^2)
                    V = -learning_rate*weight_decay*l.get_layer_params()
                        - learning_rate/(1-momentum1^t) * atan2(m_t, sqrt(v_t/(1-momentum2^t)))
                The weight decay is decoupled, as in adamw. The bias correction of the first
                moment scales the step and stays out of the arctangent, which is how the
                Hierarchical Reasoning Model (Wang et al., 2025) is trained.

                Note that the actual learning rate and weight decay used by the solver are
                multiplied by the per layer multipliers. That is, the solver will call
                get_learning_rate_multiplier(l) and get_weight_decay_multiplier(l) and
                multiply these values with the nominal learning rate and weight decay,
                respectively, to determine the values it will use during each step. It is
                also overloaded to allow additional learning rate and weight decay
                multipliers to be applied to the bias parameters of fc_, con_, cont_ and
                bn_.
        !*/

    public:

        adam_atan2(
        );
        /*!
            ensures
                - #get_weight_decay()  == 0.01
                - #get_momentum1()     == 0.9
                - #get_momentum2()     == 0.999
        !*/

        explicit adam_atan2(
            float weight_decay,
            float momentum1 = 0.9,
            float momentum2 = 0.999
        );
        /*!
            requires
                - weight_decay >= 0
                - 0 <= momentum1 < 1
                - 0 <= momentum2 < 1
            ensures
                - #get_weight_decay()  == weight_decay
                - #get_momentum1()     == momentum1
                - #get_momentum2()     == momentum2
        !*/

        float get_weight_decay() const;
        float get_momentum1() const;
        float get_momentum2() const;
    };

    void serialize(const adam_atan2& item, std::ostream& out);
    void deserialize(adam_atan2& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    std::ostream& operator<< (std::ostream& out, const adam_atan2& item);
    /*!
        Prints the solver's name and parameters to out.
    !*/

// ----------------------------------------------------------------------------------------

    class subnet_solver
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above. It is the
                solver a layer uses for a sub-network whose parameters it updates itself,
                as hrm_, moe_ and the modality fusion layer do inside their backward pass.

                The trainer's solver never reaches those sub-networks, so the trainer's
                algorithm and settings reach them through network_context, and this object
                applies the algorithm it was constructed with, adamw or adam_atan2.

                It serializes exactly as the solver it applies. A vector of subnet_solver
                objects applying AdamW writes what a vector of adamw objects writes, and
                deserializes what such a vector wrote, so a model saved before the choice
                of algorithm existed loads unchanged.
        !*/

    public:

        subnet_solver(
        );
        /*!
            ensures
                - #get_type() == network_context::optimizer_type::adamw, with the
                  default settings of adamw.
        !*/

        subnet_solver(
            network_context::optimizer_type type,
            float weight_decay,
            float momentum1,
            float momentum2
        );
        /*!
            requires
                - weight_decay >= 0
                - 0 <= momentum1 < 1
                - 0 <= momentum2 < 1
            ensures
                - #get_type() == type
                - The solver applies adamw(weight_decay, momentum1, momentum2) or
                  adam_atan2(weight_decay, momentum1, momentum2), as type says.
        !*/

        static network_context::optimizer_type requested_type(
        );
        /*!
            ensures
                - returns network_context::get_optimizer_type() if
                  network_context::is_active(), and network_context::optimizer_type::adamw
                  otherwise. This is the algorithm a layer owning sub-networks builds its
                  solvers with.
        !*/

        network_context::optimizer_type get_type(
        ) const;
        /*!
            ensures
                - returns the algorithm this solver applies.
        !*/
    };

    void serialize(const subnet_solver& item, std::ostream& out);
    void deserialize(subnet_solver& item, std::istream& in);
    /*!
        provides serialization support. The serialized form is that of the adamw or
        adam_atan2 object applied, and deserialize() reads either.
    !*/

    std::ostream& operator<< (std::ostream& out, const subnet_solver& item);
    /*!
        Prints the name and parameters of the solver applied to out.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_ABSTRACT_H_

