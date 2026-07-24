// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Vision tower geometry read from a multimodal projector container.
//
// A multimodal GGUF model comes in two files. The first is an ordinary decoder, which the
// existing machinery already reads. The second, conventionally named mmproj-*.gguf, holds
// a vision encoder and the projector that brings its output into the decoder's embedding
// space; it declares "clip" as its architecture and carries its own metadata namespace.
// This header does for that second file what gguf_model_spec.h does for the first: it
// reads the geometry, checks that what the container declares and what it contains agree,
// and reports both in a form that can be compared against a reference implementation.
//
// The geometry closes on itself, which is what makes the check worth running. A 512-pixel
// image cut into 16-pixel patches gives 32 by 32 positions; a pixel shuffle of factor 4
// folds those into 8 by 8 positions carrying sixteen times the channels; and the projector
// matrix must then have exactly that many inputs. Any container where those three numbers
// do not agree is a container this pipeline cannot serve, and it is better to say so at
// load time than to produce an image description that is quietly wrong.

#ifndef DLIB_GGUF_VISION_SPEC_H_
#define DLIB_GGUF_VISION_SPEC_H_

#include "gguf_vision_spec_abstract.h"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "gguf_reader.h"

namespace dlib
{
    /* Families differ in how they reduce the patch grid before the projector. The scheme
       is named rather than inferred, because two families can share every dimension and
       still order the folded channels differently, which a projector trained on one
       ordering will not survive. */
    enum class vision_projector_kind
    {
        unknown,
        idefics3,   // pixel shuffle by scale_factor, then one linear layer
        mlp,        // one or two linear layers over the patch grid, no reduction
        ldp         // depthwise reduction of the MobileVLM family
    };

    inline std::string describe(vision_projector_kind k)
    {
        switch (k)
        {
        case vision_projector_kind::idefics3: return "idefics3";
        case vision_projector_kind::mlp:      return "mlp";
        case vision_projector_kind::ldp:      return "ldp";
        default:                              return "unknown";
        }
    }

    struct vision_spec
    {
        std::string model_name;
        std::string arch_name;                 // "clip" for every container of this shape

        long image_size = 0;                   // side of the square the encoder expects
        long patch_size = 0;
        long d_model = 0;                      // width of the vision transformer
        long n_layers = 0;
        long n_heads = 0;
        long d_ffn = 0;
        double layer_norm_eps = 1e-6;
        bool use_gelu = true;

        vision_projector_kind projector = vision_projector_kind::unknown;
        long scale_factor = 1;                 // pixel shuffle factor, 1 meaning none
        long projection_dim = 0;               // width of the decoder this feeds

        std::vector<float> image_mean;         // per-channel normalization of the input
        std::vector<float> image_std;

        long head_dim() const { return n_heads ? d_model / n_heads : 0; }
        long grid_side() const { return patch_size ? image_size / patch_size : 0; }
        long num_patches() const { return grid_side() * grid_side(); }

        // Positions and width the decoder actually receives, after the reduction.
        long tokens_per_image() const
        {
            const long side = grid_side() / (scale_factor > 0 ? scale_factor : 1);
            return side * side;
        }
        long folded_width() const { return d_model * scale_factor * scale_factor; }
    };

    inline vision_spec detect_vision(const gguf_reader& g)
    {
        vision_spec s;
        s.arch_name = g.get_str("general.architecture");
        s.model_name = g.get_str("general.name");
        if (s.arch_name != "clip")
            throw std::runtime_error("detect_vision: '" + s.arch_name
                + "' is not a vision projector container");

        s.image_size = static_cast<long>(g.get_int("clip.vision.image_size"));
        s.patch_size = static_cast<long>(g.get_int("clip.vision.patch_size"));
        s.d_model = static_cast<long>(g.get_int("clip.vision.embedding_length"));
        s.n_layers = static_cast<long>(g.get_int("clip.vision.block_count"));
        s.n_heads = static_cast<long>(g.get_int("clip.vision.attention.head_count"));
        s.d_ffn = static_cast<long>(g.get_int("clip.vision.feed_forward_length"));
        s.layer_norm_eps = g.get_double("clip.vision.attention.layer_norm_epsilon", 1e-6);
        s.use_gelu = g.get_int("clip.use_gelu", 1) != 0;
        s.projection_dim = static_cast<long>(g.get_int("clip.vision.projection_dim"));
        s.scale_factor = static_cast<long>(g.get_int("clip.vision.projector.scale_factor", 1));
        if (s.scale_factor <= 0) s.scale_factor = 1;

        const std::string kind = g.get_str("clip.projector_type");
        if (kind == "idefics3")      s.projector = vision_projector_kind::idefics3;
        else if (kind == "mlp" || kind == "mlp_norm") s.projector = vision_projector_kind::mlp;
        else if (kind == "ldp" || kind == "ldpv2")    s.projector = vision_projector_kind::ldp;

        /* Normalization of the pixels, which the container carries because it is part of
           the encoder rather than of the image loader. Defaulting to the CLIP constants
           would hide a container that declares different ones. */
        if (g.has("clip.vision.image_mean"))
            for (double v : g.at("clip.vision.image_mean").arr_float)
                s.image_mean.push_back(static_cast<float>(v));
        if (g.has("clip.vision.image_std"))
            for (double v : g.at("clip.vision.image_std").arr_float)
                s.image_std.push_back(static_cast<float>(v));

        return s;
    }

    struct vision_compat_result
    {
        std::vector<std::string> notes;
        std::vector<std::string> blockers;
        bool usable() const { return blockers.empty(); }
    };

    /* Checks the declared geometry against itself and against the tensors present. The
       three numbers that must agree are the patch grid, the reduction factor and the width
       of the projector matrix; a container where they do not is one this pipeline cannot
       serve, and saying so at load time costs nothing next to an image description that is
       quietly wrong. */
    inline vision_compat_result check_vision_compatibility(const vision_spec& s,
        const gguf_reader& g)
    {
        vision_compat_result r;

        if (s.projector != vision_projector_kind::idefics3)
            r.blockers.push_back("projector kind '" + describe(s.projector)
                + "' is not implemented; only idefics3 is");
        if (s.patch_size <= 0 || s.image_size % s.patch_size != 0)
            r.blockers.push_back("the image size is not a whole number of patches");
        if (s.scale_factor > 1 && s.grid_side() % s.scale_factor != 0)
            r.blockers.push_back("the patch grid is not divisible by the pixel shuffle factor");
        if (s.n_heads <= 0 || s.d_model % s.n_heads != 0)
            r.blockers.push_back("the vision width is not a whole number of heads");

        auto require = [&](const std::string& name) -> const gguf_tensor_info* {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) r.blockers.push_back("missing tensor " + name);
            return t;
        };

        require("v.patch_embd.weight");
        require("v.position_embd.weight");
        require("v.post_ln.weight");
        for (long i = 0; i < s.n_layers; ++i)
        {
            const std::string p = "v.blk." + std::to_string(i) + ".";
            for (const char* suffix : { "ln1.weight", "ln2.weight", "attn_q.weight",
                "attn_k.weight", "attn_v.weight", "attn_out.weight",
                "ffn_up.weight", "ffn_down.weight" })
                require(p + suffix);
        }

        /* The projector matrix is where the arithmetic closes: its input width has to be
           the folded width, and its output the width of the decoder that will receive the
           result. */
        if (const gguf_tensor_info* fc = require("mm.model.fc.weight"))
        {
            const uint64_t expected = static_cast<uint64_t>(s.folded_width()) * s.projection_dim;
            if (fc->n_elements() != expected)
            {
                std::ostringstream o;
                o << "the projector holds " << fc->n_elements() << " values where the geometry "
                  << "calls for " << s.folded_width() << " x " << s.projection_dim
                  << "; the declared pixel shuffle factor and the projector disagree";
                r.blockers.push_back(o.str());
            }
        }

        if (const gguf_tensor_info* pos = g.find_tensor("v.position_embd.weight"))
        {
            const uint64_t expected = static_cast<uint64_t>(s.num_patches()) * s.d_model;
            if (pos->n_elements() != expected)
                r.blockers.push_back("the position table does not cover the patch grid");
        }

        if (s.image_mean.size() != 3 || s.image_std.size() != 3)
            r.notes.push_back("the container declares no pixel normalization; "
                "the CLIP constants will be assumed");
        if (!s.use_gelu)
            r.notes.push_back("the container asks for an activation other than GELU");

        return r;
    }

    inline std::string describe(const vision_spec& s)
    {
        std::ostringstream o;
        o << "Vision tower       : " << s.model_name << "\n"
          << "Image size         : " << s.image_size << " (patches of " << s.patch_size
          << ", grid " << s.grid_side() << "x" << s.grid_side() << ")\n"
          << "Embedding dim      : " << s.d_model << " (head dim " << s.head_dim() << ")\n"
          << "Layers             : " << s.n_layers << "\n"
          << "Attention heads    : " << s.n_heads << "\n"
          << "FFN hidden         : " << s.d_ffn << "\n"
          << "LayerNorm epsilon  : " << s.layer_norm_eps << "\n"
          << "Activation         : " << (s.use_gelu ? "GELU" : "other") << "\n"
          << "Projector          : " << describe(s.projector)
          << " (pixel shuffle " << s.scale_factor << ")\n"
          << "Visual tokens      : " << s.tokens_per_image() << " per image, "
          << s.folded_width() << " -> " << s.projection_dim << "\n";
        if (s.image_mean.size() == 3 && s.image_std.size() == 3)
            o << "Pixel normalization: mean " << s.image_mean[0] << "/" << s.image_mean[1]
              << "/" << s.image_mean[2] << ", std " << s.image_std[0] << "/"
              << s.image_std[1] << "/" << s.image_std[2] << "\n";
        return o.str();
    }
}

#endif // DLIB_GGUF_VISION_SPEC_H_
