// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Fusion of a non-text modality into a token stream.
//
// The standard input of the networks here is the token stream, and it stays that way. A
// modality that has no identifiers reaches the stream through this layer instead: an
// encoder produces vectors, the prompt reserves their positions with a placeholder token,
// and this layer writes them over those positions on the way up from the embeddings.
//
// Two things are worth stating plainly about the division of labour.
//
// The payload travels through network_context, the parameters live in this layer. The
// context is a channel, not a store: it carries pixels or samples for one forward pass and
// forgets them, while the encoder that turns them into vectors is a subnetwork held here,
// serialized here, and updated here. This is what lets a gradient descend from the
// language loss into the encoder, which is the whole reason for the layer to exist rather
// than for the embeddings to be patched.
//
// A layer whose slot is empty copies its input to its output. That is the common case,
// including every single-token step of a generation that followed an image, and it costs
// one comparison plus one copy.
//
// The design follows hrm_ and moe_, which already hold subnetworks: the encoder is driven
// by hand through back_propagate_error and get_final_data_gradient, and updated through
// its own AdamW solvers fed from network_context.

#ifndef DLIB_DNN_MODALITY_FUSION_H_
#define DLIB_DNN_MODALITY_FUSION_H_

#include "modality_fusion_abstract.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "core.h"
#include "layers.h"
#include "network_context.h"
#include "solvers.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /* Pass-through policy. A text-only network names this one and gains no layer at all,
       not even a transparent one, so its archives are unchanged and its forward pass costs
       exactly what it did before. */
    template <typename SUBNET>
    using no_modality = SUBNET;

// ----------------------------------------------------------------------------------------

    template <
        typename ENCODER_NET,
        long SLOT_,
        long TOKENS_PER_PAYLOAD_,
        long WIDTH_
    >
    class modality_fusion_
    {
        static_assert(TOKENS_PER_PAYLOAD_ > 0, "TOKENS_PER_PAYLOAD must be positive");
        static_assert(WIDTH_ > 0, "WIDTH must be positive");

    public:

        static constexpr long SLOT              = SLOT_;
        static constexpr long TOKENS_PER_PAYLOAD = TOKENS_PER_PAYLOAD_;
        static constexpr long WIDTH             = WIDTH_;

        modality_fusion_() :
            learning_rate_multiplier_(1),
            current_learning_rate_(1e-4),
            solver_weight_decay_(0.004),
            solver_beta1_(0.9),
            solver_beta2_(0.999),
            solvers_initialized_(false)
        {
        }

        double get_learning_rate_multiplier() const { return learning_rate_multiplier_; }
        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier_ = val;
            set_all_learning_rate_multipliers(encoder_, val);
            solvers_initialized_ = false;
        }
        double get_weight_decay_multiplier() const { return 1.0; }
        void set_weight_decay_multiplier(double) {}

        void set_learning_rate(double lr)
        {
            current_learning_rate_ = lr;
            set_all_learning_rates(encoder_, lr);
        }
        double get_learning_rate() const { return current_learning_rate_; }

        void configure_solvers(double weight_decay, double beta1, double beta2)
        {
            solver_weight_decay_ = weight_decay;
            solver_beta1_ = beta1;
            solver_beta2_ = beta2;
            solvers_initialized_ = false;
        }

        /* The encoder is reachable so that a loader can fill it, a visitor can walk it, and
           a caller can count what it holds. Nothing else here exposes it: it is not part of
           the parameter list this layer reports, which is what keeps the enclosing
           network's own weight import untouched. */
        ENCODER_NET& get_encoder() { return encoder_; }
        const ENCODER_NET& get_encoder() const { return encoder_; }

        size_t internal_parameters() const { return count_parameters(encoder_); }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            DLIB_CASSERT(sub.get_output().nc() == WIDTH,
                "modality_fusion expects a stream of width " << WIDTH
                << ", got " << sub.get_output().nc());
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& x = sub.get_output();
            output.copy_size(x);
            memcpy(output, x);

            injected_.clear();
            std::vector<modality_input> inputs =
                network_context::take_modality_inputs(SLOT);
            if (inputs.empty()) return;

            /* All payloads of a slot share their shape, an encoder having one input
               geometry, so they are encoded as a single batch: one forward through the
               encoder per forward through the network, which is also what lets the backward
               pass reuse the state this one leaves behind. */
            const size_t per_payload = inputs[0].payload.size();
            DLIB_CASSERT(per_payload > 0, "a modality payload cannot be empty");
            batched_input_.set_size(
                static_cast<long>(inputs.size()) * inputs[0].payload.num_samples(),
                inputs[0].payload.k(), inputs[0].payload.nr(), inputs[0].payload.nc());
            {
                float* dst = batched_input_.host();
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    DLIB_CASSERT(inputs[i].payload.size() == per_payload,
                        "the payloads of one slot must share their shape");
                    std::memcpy(dst + i * per_payload, inputs[i].payload.host(),
                        per_payload * sizeof(float));
                }
            }

            const tensor& rows = encoder_.forward(batched_input_);
            DLIB_CASSERT(rows.k() == WIDTH && rows.nr() == 1 && rows.nc() == 1,
                "the encoder must emit [rows, " << WIDTH << ", 1, 1], got k=" << rows.k()
                << " nr=" << rows.nr() << " nc=" << rows.nc());
            DLIB_CASSERT(rows.num_samples()
                    == static_cast<long>(inputs.size()) * TOKENS_PER_PAYLOAD,
                "the encoder emitted " << rows.num_samples() << " rows where "
                << inputs.size() * TOKENS_PER_PAYLOAD << " were expected");

            const float* src = rows.host();
            float* dst = output.host();
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                const modality_input& in = inputs[i];
                DLIB_CASSERT(static_cast<long>(in.positions.size()) == TOKENS_PER_PAYLOAD,
                    "a payload reserved " << in.positions.size() << " positions where "
                    << TOKENS_PER_PAYLOAD << " were expected");
                DLIB_CASSERT(in.sequence >= 0 && in.sequence < output.num_samples(),
                    "a payload names a sequence outside the batch");

                placement p;
                p.sequence = in.sequence;
                p.positions = in.positions;
                for (long t = 0; t < TOKENS_PER_PAYLOAD; ++t)
                {
                    const long pos = in.positions[static_cast<size_t>(t)];
                    DLIB_CASSERT(pos >= 0 && pos < output.nr(),
                        "a reserved position falls outside the stream");
                    std::memcpy(dst + tensor_index(output, in.sequence, 0, pos, 0),
                        src + (i * TOKENS_PER_PAYLOAD + t) * WIDTH,
                        WIDTH * sizeof(float));
                }
                injected_.push_back(std::move(p));
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tensor& dx = sub.get_gradient_input();

            if (injected_.empty())
            {
                tt::add(1.0f, dx, 1.0f, gradient_input);
                return;
            }

            /* A position that was written over does not depend on the embedding table: its
               vector came from the encoder, so the derivative with respect to the row the
               lookup produced is zero there. Passing the gradient down unchanged would
               teach the table from vectors it never emitted, and nothing downstream would
               report it. */
            passthrough_.copy_size(gradient_input);
            memcpy(passthrough_, gradient_input);

            row_grads_.set_size(
                static_cast<long>(injected_.size()) * TOKENS_PER_PAYLOAD, WIDTH, 1, 1);
            float* rows = row_grads_.host();
            float* through = passthrough_.host();
            const float* g = gradient_input.host();

            for (size_t i = 0; i < injected_.size(); ++i)
            {
                const placement& p = injected_[i];
                for (long t = 0; t < TOKENS_PER_PAYLOAD; ++t)
                {
                    const long pos = p.positions[static_cast<size_t>(t)];
                    const size_t at = tensor_index(gradient_input, p.sequence, 0, pos, 0);
                    std::memcpy(rows + (i * TOKENS_PER_PAYLOAD + t) * WIDTH, g + at,
                        WIDTH * sizeof(float));
                    std::memset(through + at, 0, WIDTH * sizeof(float));
                }
            }

            tt::add(1.0f, dx, 1.0f, passthrough_);

            /* The encoder is driven by hand, on the batch this layer built during the
               forward pass rather than on a fresh one: re-running it here would both cost a
               second forward and, on any encoder holding state, compute the gradient of
               something else. */
            encoder_.back_propagate_error(batched_input_, row_grads_);

            const bool in_training = !network_context::is_active()
                || network_context::is_training();
            if (in_training) update_encoder_parameters();
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const modality_fusion_& item, std::ostream& out)
        {
            serialize("modality_fusion_", out);
            serialize(item.encoder_, out);
            serialize(item.learning_rate_multiplier_, out);
            serialize(item.current_learning_rate_, out);
            serialize(item.solver_weight_decay_, out);
            serialize(item.solver_beta1_, out);
            serialize(item.solver_beta2_, out);
        }
        friend void deserialize(modality_fusion_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "modality_fusion_")
                throw serialization_error("Unexpected version found while deserializing dlib::modality_fusion_.");
            deserialize(item.encoder_, in);
            deserialize(item.learning_rate_multiplier_, in);
            deserialize(item.current_learning_rate_, in);
            deserialize(item.solver_weight_decay_, in);
            deserialize(item.solver_beta1_, in);
            deserialize(item.solver_beta2_, in);
            /* Solvers are not carried over: their moments belong to a run, and a network
               that has just been read from disk has no run behind it. */
            item.solvers_.clear();
            item.solvers_initialized_ = false;
            item.injected_.clear();
        }
        friend std::ostream& operator<<(std::ostream& out, const modality_fusion_& item)
        {
            out << "modality_fusion (slot=" << SLOT << ", tokens=" << TOKENS_PER_PAYLOAD
                << ", width=" << WIDTH << ", encoder_params=" << item.internal_parameters()
                << ") learning_rate_mult=" << item.learning_rate_multiplier_;
            return out;
        }
        friend void to_xml(const modality_fusion_& item, std::ostream& out)
        {
            out << "<modality_fusion slot='" << SLOT << "' tokens='" << TOKENS_PER_PAYLOAD
                << "' width='" << WIDTH << "'>\n";
            to_xml(item.encoder_, out);
            out << "</modality_fusion>\n";
        }

    private:

        void update_encoder_parameters()
        {
            if (network_context::is_active())
                current_learning_rate_ = network_context::get_learning_rate();
            if (learning_rate_multiplier_ == 0.0) return;

            const double effective_lr = current_learning_rate_ * learning_rate_multiplier_;

            if (!solvers_initialized_)
            {
                const double wd = network_context::is_active()
                    ? network_context::get_optimizer_weight_decay() : solver_weight_decay_;
                const double b1 = network_context::is_active()
                    ? network_context::get_optimizer_beta1() : solver_beta1_;
                const double b2 = network_context::is_active()
                    ? network_context::get_optimizer_beta2() : solver_beta2_;
                solvers_.assign(ENCODER_NET::num_computational_layers, adamw(wd, b1, b2));
                solvers_initialized_ = true;
            }

            encoder_.update_parameters(solvers_, effective_lr);
        }

        struct placement
        {
            long sequence = 0;
            std::vector<long> positions;
        };

        resizable_tensor params; // unused: the encoder's weights are its own
        ENCODER_NET encoder_;

        double learning_rate_multiplier_;
        double current_learning_rate_;
        double solver_weight_decay_, solver_beta1_, solver_beta2_;
        std::vector<adamw> solvers_;
        bool solvers_initialized_;

        std::vector<placement> injected_;
        resizable_tensor batched_input_, row_grads_, passthrough_;
    };

    template <typename ENCODER_NET, long SLOT, long TOKENS_PER_PAYLOAD, long WIDTH,
        typename SUBNET>
    using modality_fusion =
        add_layer<modality_fusion_<ENCODER_NET, SLOT, TOKENS_PER_PAYLOAD, WIDTH>, SUBNET>;

// ----------------------------------------------------------------------------------------

    /* Vision is the modality this library serves today, so it gets a name of its own. The
       encoder is any network emitting [rows, WIDTH, 1, 1], which is what
       vision_transformer_config::network_type produces. */
    template <typename ENCODER_NET, long TOKENS_PER_IMAGE, long WIDTH, typename SUBNET>
    using visual_fusion = modality_fusion<ENCODER_NET, modality_slot::vision,
        TOKENS_PER_IMAGE, WIDTH, SUBNET>;
}

#endif // DLIB_DNN_MODALITY_FUSION_H_
