// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF -> Dlib weight import for the vision tower.
//
// Counterpart of gguf_weight_loader.h for the second file a multimodal model ships. It
// fills a network built from vision_transformer_config with the contents of an mmproj
// container, and it exists for the same reason the decoder loader does: the file layout
// and the layer layout agree nowhere by accident, and the mapping between them is worth
// stating once, in one place, rather than rediscovering it per model.
//
// Three things about that mapping are not obvious.
//
// The feed-forward names are inverted relative to the decoder's convention. In this
// container ffn_down carries the expansion and ffn_up the contraction, which their bias
// sizes confirm without ambiguity. Reading them the other way round produces a tower that
// runs, returns plausible numbers, and describes the wrong image.
//
// Linear weights are stored as out rows of in values, and every Dlib layer here multiplies
// x[rows, in] by W[in, out], so each of them is transposed on the way in. The convolution
// is the exception: its filters are already in the layout con expects, so they go in as
// they come out of the file.
//
// The normalization epsilon travels with the container rather than with the code. A tower
// trained at 1e-6 and run at Dlib's 1e-5 default drifts quietly, so the value is pushed
// into every layer_norm before the weights are copied.

#ifndef DLIB_GGUF_VISION_LOADER_H_
#define DLIB_GGUF_VISION_LOADER_H_

#include "gguf_vision_loader_abstract.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_reader.h"
#include "gguf_dequantize.h"
#include "gguf_vision_spec.h"
#include "../dnn.h"

namespace dlib
{
    struct gguf_vision_load_options
    {
        bool verbose = true;
    };

    namespace gguf_vision_load_impl
    {
        inline std::vector<float> fetch(gguf_reader& g, const std::string& name)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error(
                "import_gguf_vision_weights: missing tensor '" + name + "'");
            std::vector<float> out;
            gguf_read_dequantized(g, *t, out);
            return out;
        }

        inline bool present(gguf_reader& g, const std::string& name)
        {
            return g.find_tensor(name) != nullptr;
        }

        /* Straight copy into a layer parameter tensor, zeroing whatever the layer holds
           beyond the source. That trailing part is always a bias slot the container does
           not carry, which must then read as absent rather than as leftover memory. */
        inline void copy_into(tensor& dst, const std::vector<float>& src,
            const std::string& what)
        {
            if (src.size() > dst.size())
                throw std::runtime_error("import_gguf_vision_weights: source larger than "
                    "layer for " + what + " (layer " + std::to_string(dst.size())
                    + " vs source " + std::to_string(src.size()) + ")");
            float* h = dst.host();
            std::memcpy(h, src.data(), src.size() * sizeof(float));
            if (src.size() < dst.size())
                std::memset(h + src.size(), 0, (dst.size() - src.size()) * sizeof(float));
        }

        /* Transpose a [out_dim, in_dim] container matrix into a [in_dim, out_dim] buffer. */
        inline void transpose_into(const std::vector<float>& src, long out_dim, long in_dim,
            float* dst)
        {
            for (long o = 0; o < out_dim; ++o)
            {
                const float* row = src.data() + static_cast<size_t>(o) * in_dim;
                for (long i = 0; i < in_dim; ++i)
                    dst[static_cast<size_t>(i) * out_dim + o] = row[i];
            }
        }

        /* Fills a Dlib fully connected layer, whose parameter blob is the [in, out] weight
           matrix followed by a single bias row. Passing an empty bias name leaves that row
           at zero, which is how a container without one is represented. */
        inline void load_fc(gguf_reader& g, tensor& params, long in_dim, long out_dim,
            const std::string& weight_name, const std::string& bias_name)
        {
            const std::vector<float> w = fetch(g, weight_name);
            if (w.size() != static_cast<size_t>(in_dim) * out_dim)
                throw std::runtime_error("import_gguf_vision_weights: shape mismatch for "
                    + weight_name);
            const size_t expected = static_cast<size_t>(in_dim) * out_dim + out_dim;
            if (params.size() != expected)
                throw std::runtime_error("import_gguf_vision_weights: layer holds "
                    + std::to_string(params.size()) + " values where " + weight_name
                    + " calls for " + std::to_string(expected));

            float* h = params.host();
            transpose_into(w, out_dim, in_dim, h);
            float* bias_row = h + static_cast<size_t>(in_dim) * out_dim;
            if (!bias_name.empty() && present(g, bias_name))
            {
                const std::vector<float> b = fetch(g, bias_name);
                if (b.size() != static_cast<size_t>(out_dim))
                    throw std::runtime_error("import_gguf_vision_weights: shape mismatch for "
                        + bias_name);
                std::memcpy(bias_row, b.data(), b.size() * sizeof(float));
            }
            else
            {
                std::memset(bias_row, 0, static_cast<size_t>(out_dim) * sizeof(float));
            }
        }

        /* Fills a layer_norm, whose blob is gamma followed by beta. */
        inline void load_layer_norm(gguf_reader& g, tensor& params, long width,
            const std::string& prefix)
        {
            const std::vector<float> gamma = fetch(g, prefix + ".weight");
            const std::vector<float> beta = fetch(g, prefix + ".bias");
            if (gamma.size() != static_cast<size_t>(width) || beta.size() != gamma.size())
                throw std::runtime_error("import_gguf_vision_weights: shape mismatch for "
                    + prefix);
            if (params.size() != gamma.size() + beta.size())
                throw std::runtime_error("import_gguf_vision_weights: layer_norm holds "
                    + std::to_string(params.size()) + " values for " + prefix);
            float* h = params.host();
            std::memcpy(h, gamma.data(), gamma.size() * sizeof(float));
            std::memcpy(h + gamma.size(), beta.data(), beta.size() * sizeof(float));
        }

        /* Fills the fused attention layer: four transposed [width, width] matrices in the
           order q, k, v, out, then the four biases in the same order. The offsets come from
           the layer rather than from a second copy of the layout kept here. */
        template <typename attention_layer>
        void load_attention(gguf_reader& g, tensor& params, long width,
            const std::string& prefix)
        {
            using l = attention_layer;
            if (params.size() != l::parameter_count())
                throw std::runtime_error("import_gguf_vision_weights: attention layer holds "
                    + std::to_string(params.size()) + " values for " + prefix);
            float* h = params.host();
            const char* const weights[] = { "attn_q", "attn_k", "attn_v", "attn_out" };
            const size_t w_offsets[] = { l::wq_offset(), l::wk_offset(), l::wv_offset(),
                l::wo_offset() };
            const size_t b_offsets[] = { l::bq_offset(), l::bk_offset(), l::bv_offset(),
                l::bo_offset() };

            for (int i = 0; i < 4; ++i)
            {
                const std::string name = prefix + "." + weights[i];
                const std::vector<float> w = fetch(g, name + ".weight");
                if (w.size() != static_cast<size_t>(width) * width)
                    throw std::runtime_error("import_gguf_vision_weights: shape mismatch for "
                        + name + ".weight");
                transpose_into(w, width, width, h + w_offsets[i]);

                float* bias = h + b_offsets[i];
                if (present(g, name + ".bias"))
                {
                    const std::vector<float> b = fetch(g, name + ".bias");
                    if (b.size() != static_cast<size_t>(width))
                        throw std::runtime_error("import_gguf_vision_weights: shape mismatch "
                            "for " + name + ".bias");
                    std::memcpy(bias, b.data(), b.size() * sizeof(float));
                }
                else
                {
                    std::memset(bias, 0, static_cast<size_t>(width) * sizeof(float));
                }
            }
        }

        /* Pushes the container's epsilon into every layer_norm of the network. The overload
           on the exact layer type does the work; the template catches everything else. */
        struct layer_norm_eps_setter
        {
            double eps;
            void operator()(layer_norm_& l) const { l.set_eps(eps); }
            template <typename T> void operator()(T&) const {}
        };
    }

// ----------------------------------------------------------------------------------------

    /* Imports an mmproj container into a network built from vision_transformer_config.

       The parameter tensors are collected in network visit order, which runs from the
       output side inward, and consumed in that same order: projector, final normalization,
       then each block from the last to the first, then the position table and the patch
       embedding. A single counter walks that list, so a network whose shape does not match
       the container is caught at the first tensor whose size disagrees rather than halfway
       through with plausible garbage in place. */
    template <typename net_type, typename config>
    void import_gguf_vision_weights(net_type& net, gguf_reader& g, const vision_spec& spec,
        const config&, const gguf_vision_load_options& opt = gguf_vision_load_options())
    {
        using namespace gguf_vision_load_impl;

        if (spec.d_model != config::WIDTH || spec.n_layers != config::NUM_LAYERS
            || spec.n_heads != config::NUM_HEADS || spec.d_ffn != config::FFN_HIDDEN
            || spec.image_size != config::IMAGE_SIZE || spec.patch_size != config::PATCH_SIZE
            || spec.scale_factor != config::SHUFFLE_FACTOR
            || spec.projection_dim != config::PROJECTION_DIM)
            throw std::runtime_error("import_gguf_vision_weights: the container geometry "
                "does not match the compiled tower");

        /* The epsilon must be in place before any forward runs. */
        visit_computational_layers(net, layer_norm_eps_setter{ spec.layer_norm_eps });

        /* Allocate every parameter tensor with one forward pass on a blank image. */
        {
            resizable_tensor dummy(1, 3, config::IMAGE_SIZE, config::IMAGE_SIZE);
            dummy = 0;
            net.forward(dummy);
        }

        std::vector<tensor*> params;
        visit_computational_layers(net, [&params](auto& layer) {
            tensor& p = layer.get_layer_params();
            if (p.size() > 0) params.push_back(&p);
        });

        size_t k = 0;
        auto next = [&](const std::string& what) -> tensor& {
            if (k >= params.size())
                throw std::runtime_error("import_gguf_vision_weights: ran out of layers at "
                    + what);
            return *params[k++];
        };

        const long d = config::WIDTH, ff = config::FFN_HIDDEN;
        using attention_type = vision_attention_<config::WIDTH, config::NUM_HEADS,
            config::NUM_PATCHES>;

        // Projector: [projection_dim, folded_width] in the file, [folded_width, proj] here.
        load_fc(g, next("projector"), config::FOLDED_WIDTH, config::PROJECTION_DIM,
            "mm.model.fc.weight", "mm.model.fc.bias");

        load_layer_norm(g, next("post_ln"), d, "v.post_ln");

        for (long b = config::NUM_LAYERS - 1; b >= 0; --b)
        {
            const std::string p = "v.blk." + std::to_string(b);

            /* Named backwards in this container: ffn_down expands and ffn_up contracts,
               as their bias sizes show. */
            load_fc(g, next("ffn_contract"), ff, d, p + ".ffn_up.weight", p + ".ffn_up.bias");
            load_fc(g, next("ffn_expand"), d, ff, p + ".ffn_down.weight", p + ".ffn_down.bias");
            load_layer_norm(g, next("ln2"), d, p + ".ln2");
            load_attention<attention_type>(g, next("attention"), d, p);
            load_layer_norm(g, next("ln1"), d, p + ".ln1");
        }

        // Position table: one vector per patch, already in the layout this layer wants.
        copy_into(next("positions"), fetch(g, "v.position_embd.weight"), "positions");

        /* Patch embedding. The filters are stored as the convolution wants them, so the
           copy is straight; the bias follows in the same blob and is zeroed when absent. */
        {
            tensor& conv = next("patch_embd");
            const std::vector<float> w = fetch(g, "v.patch_embd.weight");
            float* h = conv.host();
            if (w.size() > conv.size())
                throw std::runtime_error("import_gguf_vision_weights: patch embedding larger "
                    "than the convolution");
            std::memcpy(h, w.data(), w.size() * sizeof(float));
            const size_t rest = conv.size() - w.size();
            if (rest > 0)
            {
                if (present(g, "v.patch_embd.bias"))
                {
                    const std::vector<float> b = fetch(g, "v.patch_embd.bias");
                    if (b.size() != rest)
                        throw std::runtime_error("import_gguf_vision_weights: patch embedding "
                            "bias does not fill the convolution");
                    std::memcpy(h + w.size(), b.data(), b.size() * sizeof(float));
                }
                else
                {
                    std::memset(h + w.size(), 0, rest * sizeof(float));
                }
            }
        }

        if (k != params.size())
            throw std::runtime_error("import_gguf_vision_weights: "
                + std::to_string(params.size() - k) + " parameter tensors were left unfilled");

        if (opt.verbose)
        {
            long long total = 0;
            for (tensor* p : params) total += static_cast<long long>(p->size());
            std::cout << "Vision parameters  : " << total << " in " << params.size()
                      << " tensors\n";
        }
    }
}

#endif // DLIB_GGUF_VISION_LOADER_H_
