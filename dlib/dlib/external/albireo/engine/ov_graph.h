// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_OV_GRAPH_H_
#define ALBIREO_ENGINE_OV_GRAPH_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model_file.h"
#include "runtime.h"

namespace albireo
{

    /*!
        The whole decoder of a model as one OpenVINO graph.

        This header holds no OpenVINO type: engine/ov_graph.cpp alone includes the headers of
        OpenVINO, which need C++17, and is compiled apart from the rest of the engine.

        GRAPH
            One graph computes a pass of one sequence: the RMS normalizations, the products
            with the weights, the rotary positions, the attention over the keys and values of
            the sequence and the feed-forward blocks of every layer, then the logits of the
            rows the pass asks for.  The keys and values live in the variables of the graph,
            one pair per layer, so that each sequence has an inference request of its own
            whose variables hold its cache: the plugin of the processor then fuses each
            attention with the concatenation of its cache into one node, and keeps the cache
            in its own format.

            The host computes what does not depend on the weights of the layers: the
            embeddings of the tokens, which it reads from the file as the processor runtime
            does, and the cosines and sines of their rotary positions, whatever the frequency
            adjustments and the multimodal positions of the model.  The products of Q, K and V,
            and those of the gate and up projections, run as one product each when their
            matrices share a format.

        WEIGHTS
            The weights become constants of the graph in a form the plugins multiply without
            expanding them first: 4-bit or 8-bit integers by groups of 32 values with a scale
            in half precision.  Q4_0, Q5_0 and Q8_0 keep their exact values.  The formats with
            minimums keep their 4-bit values and their minimums: the values and the scales form
            one product, and the minimums a second, smaller one, with the sums of the inputs
            over each group of 32.  Q4_1 stays exact; the scale of a group of Q4_K, the product
            of the scale of its block and of its 6-bit integer, is rounded to half precision, a
            relative error below 2^-11.  The other formats (Q5_K, Q6_K, Q5_1, Q2_K, Q3_K,
            IQ4_NL) are rounded again to 8-bit integers by groups of 32, which adds to their
            own rounding an error of about a quarter of it for Q6_K, an eighth for Q5_K and
            Q5_1, and less for the formats of fewer bits.  Matrices in half precision stay so,
            and the others in single precision.

        SUPPORTED MODELS
            The dense decoders of model_config (llama, mistral, mistral3, qwen2, qwen3,
            qwen3vl, granite, phi3), language and embedding models.  The mixtures of experts
            stay with the processor runtime.
    !*/

    struct ov_graph_settings
    {
        // The device of OpenVINO that computes the graph: "CPU", "GPU", "GPU.1".
        std::string device = "CPU";
        // Keys and values in half precision; otherwise in 8-bit integers on the processor.
        bool kv_half = true;
        // Threads of the plugin of the processor, 0 for its own choice.
        int threads = 0;
        /*
            The causal mask the graphs of the other devices take as an input, on the processor
            too, whose plugin otherwise fuses a causal attention without it: the tests check that
            graph on the processor.
        */
        bool explicit_mask = false;
        /*
            A folder where the graph is written once built and read back with its weights mapped
            from their file, which the runtime deletes when it no longer needs it (at once on
            Linux, when the model leaves on Windows), or when the graph fails.  The plugins copy the
            weights into their own layout, and the pages of the file, which the system can drop,
            then hold the constants the graph keeps rather than the memory of the process.  A graph
            that cannot be written there fails.  Empty: the constants stay in the memory of the
            process, which then holds the weights twice.
        */
        std::string weights_dir;
    };

    class ov_graph_runtime : public model_runtime
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A model_runtime whose passes run the graph of the model on a device of
                OpenVINO.  A pass that holds several sequences runs one inference per
                sequence.  Truncating or copying a sequence reads its variables and writes
                the part kept back, a cost that grows with the positions kept, and may throw.

                The plugin of the processor grows the buffers of the cache of a sequence to
                twice its positions when a pass fills them: the budget of the keys and values
                and the guard of the memory count these buffers, the old ones beside the new
                ones while the plugin copies them, those of the other devices counted the same
                way, and kv_bytes_per_token() gives two positions of these buffers, so that a
                context the budget holds can always be read to its end.
        !*/
    public:
        // The check of the memory of the system before the keys and values grow (kv_pool::set_guard).
        virtual void set_kv_guard(std::function<bool(size_t, size_t)> guard) = 0;

        /*!
            requires
                - no pass has run, or the rotary divisors stay the same.
            ensures
                - Chooses the rotary divisors of a model with LongRoPE for sequences of up to
                  positions tokens (select_rope_factors()).
        !*/
        virtual void set_context(long positions) = 0;

        // The seconds the conversion of the weights and the compilation of the graph took.
        virtual double build_seconds() const = 0;
    };

    /*!
        ensures
            - Returns why the graph cannot compute the model of config and file, or an empty
              string when it can: its architecture, its tensors, and the formats of the
              matrices of its layers and of its output.
    !*/
    std::string ov_graph_unsupported(const model_config& config, const model_file& file);

    /*!
        ensures
            - Converts the weights of the model, builds its graph and compiles it for
              settings.device, the keys and values of its sequences taking up to
              kv_budget_bytes.
            - Throws std::runtime_error with the reason when OpenVINO finds no such device,
              before the weights are converted, and when it cannot compile or run the graph
              on the device.
    !*/
    std::unique_ptr<ov_graph_runtime> make_ov_graph_runtime(std::shared_ptr<model_file> file,
        const model_config& config, const ov_graph_settings& settings, size_t kv_budget_bytes);

    // The devices OpenVINO finds ("CPU", "GPU", "NPU", ...), and the version of its runtime.
    std::vector<std::string> ov_graph_devices();
    std::string ov_graph_version();

    /*!
        ensures
            - Returns true when the plugin of the processor of OpenVINO loads, which runs the
              graphs on the processor, and gives in name the name of the processor it reports.
              Only that plugin is loaded.
    !*/
    bool ov_graph_processor(std::string& name);

}

#endif // ALBIREO_ENGINE_OV_GRAPH_H_
