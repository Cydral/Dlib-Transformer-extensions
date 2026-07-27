// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_MODALITY_FUSION_ABSTRACT_H_
#ifdef DLIB_DNN_MODALITY_FUSION_ABSTRACT_H_

#include "core_abstract.h"
#include "layers_abstract.h"
#include "network_context_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename SUBNET>
    using no_modality = SUBNET;
    /*!
        WHAT THIS OBJECT REPRESENTS
            The pass-through policy. A network that names this one gains no layer at all,
            not even a transparent one, so its archives are unchanged and its forward pass
            costs exactly what it did before. This is what a text-only model uses.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename ENCODER_NET,
        long SLOT,
        long TOKENS_PER_PAYLOAD,
        long WIDTH
        >
    class modality_fusion_
    {
        /*!
            REQUIREMENTS ON THE TEMPLATE ARGUMENTS
                - TOKENS_PER_PAYLOAD > 0 and WIDTH > 0
                - ENCODER_NET is a network whose input layer accepts a tensor and whose
                  output is [rows, WIDTH, 1, 1], emitting TOKENS_PER_PAYLOAD rows per
                  payload. vision_transformer_config::network_type is such a network.

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It brings a modality that has no identifiers
                into a token stream.

                Placed just above the embeddings, it expects and produces a stream shaped
                [sequences, 1, positions, WIDTH]. On each forward pass it takes whatever
                network_context holds for its slot, runs the encoder it owns over those
                payloads, and writes the resulting vectors over the positions each payload
                reserved. With an empty slot it copies its input to its output, which is
                the common case, including every single-token step of a generation that
                followed an image.

                THE DIVISION OF LABOUR

                The payload travels through network_context, the parameters live here. The
                context is a channel, not a store: it carries pixels or samples for one
                forward pass and forgets them, while the encoder that turns them into
                vectors is a subnetwork held by this layer, serialized with it, and updated
                by it. That is what lets a gradient descend from the language loss into the
                encoder, and it is the whole reason for this layer to exist rather than for
                the embedding lookup to be patched.

                THE BACKWARD PASS

                A position that was written over does not depend on the embedding table:
                its vector came from the encoder, so the derivative with respect to the row
                the lookup produced is zero there. This layer therefore hands the gradient
                down with those rows cleared, and routes them into the encoder instead.
                Passing the gradient down unchanged would teach the table from vectors it
                never emitted, and nothing downstream would report it.

                The encoder is driven by hand, as hrm_ and moe_ drive their own
                subnetworks: back_propagate_error on the batch built during the forward
                pass, then update_parameters through AdamW solvers fed from
                network_context. get_layer_params() stays empty, the encoder's weights
                being its own, which is what keeps an enclosing network's weight import
                untouched by the presence of this layer.

                What that also means is that visitors walking the enclosing network do not
                see the encoder. get_encoder() is provided for a loader, a visitor or a
                parameter count that needs to reach it.
        !*/

    public:

        modality_fusion_(
        );
        /*!
            ensures
                - #get_learning_rate_multiplier() == 1
                - The encoder is default constructed.
        !*/

        double get_learning_rate_multiplier() const;
        void set_learning_rate_multiplier(double val);
        double get_weight_decay_multiplier() const;
        void set_weight_decay_multiplier(double val);
        void set_learning_rate(double lr);
        double get_learning_rate() const;
        /*!
            These behave as described in the EXAMPLE_COMPUTATIONAL_LAYER_ interface, and
            propagate to the encoder. Setting the learning rate multiplier to 0 freezes the
            encoder, which is what the first stage of a two-stage multimodal training does.
        !*/

        void configure_solvers(double weight_decay, double beta1, double beta2);
        /*!
            ensures
                - Sets the hyperparameters of the internal AdamW solvers, used when
                  network_context is not active. The solvers are rebuilt on the next
                  update.
        !*/

        ENCODER_NET& get_encoder();
        const ENCODER_NET& get_encoder() const;
        /*!
            ensures
                - Returns the encoder this layer owns, so that a loader can fill it, a
                  visitor can walk it, or a caller can count what it holds.
        !*/

        size_t internal_parameters() const;
        /*!
            ensures
                - Returns count_parameters(get_encoder()). These are not reported by
                  get_layer_params() and are therefore invisible to the enclosing network's
                  own parameter walk.
        !*/

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            These functions are implemented as described in the
            EXAMPLE_COMPUTATIONAL_LAYER_ interface.  Note that get_layer_params() always
            returns an empty tensor.

            setup() requires that sub.get_output().nc() == WIDTH.

            forward() requires, for each payload held by slot SLOT:
                - all payloads of the slot share their shape
                - positions.size() == TOKENS_PER_PAYLOAD
                - sequence is within the batch and every position within the stream
                - the encoder emits exactly TOKENS_PER_PAYLOAD rows per payload
            and it empties the slot, so a payload is never presented twice.

            backward() updates the encoder when the pass is a training one, which is what
            network_context::is_training() reports, or when the context is inactive.
        !*/
    };

    template <typename ENCODER_NET, long SLOT, long TOKENS_PER_PAYLOAD, long WIDTH>
    void serialize(const modality_fusion_<ENCODER_NET, SLOT, TOKENS_PER_PAYLOAD, WIDTH>& item, std::ostream& out);
    template <typename ENCODER_NET, long SLOT, long TOKENS_PER_PAYLOAD, long WIDTH>
    void deserialize(modality_fusion_<ENCODER_NET, SLOT, TOKENS_PER_PAYLOAD, WIDTH>& item, std::istream& in);
    /*!
        provides serialization support

        The encoder is carried whole, so one archive holds the entire multimodal model. The
        solver moments are not: they belong to a run, and a network just read from disk has
        no run behind it.
    !*/

    template <typename ENCODER_NET, long SLOT, long TOKENS_PER_PAYLOAD, long WIDTH, typename SUBNET>
    using modality_fusion =
        add_layer<modality_fusion_<ENCODER_NET, SLOT, TOKENS_PER_PAYLOAD, WIDTH>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <typename ENCODER_NET, long TOKENS_PER_IMAGE, long WIDTH, typename SUBNET>
    using visual_fusion = modality_fusion<ENCODER_NET, modality_slot::vision,
        TOKENS_PER_IMAGE, WIDTH, SUBNET>;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Vision is the modality this library serves today, so it gets a name of its own.
            ENCODER_NET is any network emitting [rows, WIDTH, 1, 1], which is what
            vision_transformer_config::network_type produces.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_MODALITY_FUSION_ABSTRACT_H_
