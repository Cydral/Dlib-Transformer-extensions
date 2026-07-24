// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GGUF_VISION_SPEC_ABSTRACT_H_
#ifdef DLIB_GGUF_VISION_SPEC_ABSTRACT_H_

#include "gguf_reader_abstract.h"

namespace dlib
{
    /*!
        WHAT THIS FILE REPRESENTS
            The geometry of a vision tower, read from the projector container that a
            multimodal model ships alongside its decoder.

            Such a model comes in two files. The first is an ordinary decoder, which
            gguf_model_spec.h already reads. The second, conventionally named
            mmproj-*.gguf, holds a vision encoder and the projector that brings its output
            into the decoder's embedding space; it declares "clip" as its architecture and
            carries its own metadata namespace. This header does for that file what
            gguf_model_spec.h does for the first.

        WHY THE CHECK IS WORTH RUNNING
            The geometry closes on itself. A 512-pixel image cut into 16-pixel patches
            gives a 32 by 32 grid; a pixel shuffle of factor 4 folds it into 8 by 8
            positions carrying sixteen times the channels; and the projector matrix must
            then have exactly that many inputs. A container where those three numbers
            disagree is one this pipeline cannot serve, and saying so at load time costs
            nothing next to an image description that is quietly wrong.
    !*/

// ----------------------------------------------------------------------------------------

    enum class vision_projector_kind
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                How a family reduces the patch grid before the projector.

                The scheme is named rather than inferred, because two families can share
                every dimension and still order the folded channels differently, which a
                projector trained on one ordering will not survive.

            VALUES
                unknown  - the container declares a scheme this header does not know
                idefics3 - pixel shuffle by scale_factor, then one linear layer
                mlp      - one or two linear layers over the patch grid, no reduction
                ldp      - the depthwise reduction of the MobileVLM family
        !*/
    };

    std::string describe(
        vision_projector_kind k
    );
    /*!
        ensures
            - Returns the name of the scheme, or "unknown".
    !*/

// ----------------------------------------------------------------------------------------

    struct vision_spec
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Everything the container declares about its vision tower and projector.

            FIELDS
                model_name, arch_name   - identity; arch_name is "clip" for this shape
                image_size, patch_size  - side of the square the encoder expects, and of
                                          the patches it is cut into
                d_model, n_layers,
                n_heads, d_ffn          - geometry of the vision transformer
                layer_norm_eps, use_gelu- values rather than shapes
                projector, scale_factor - reduction scheme and its factor, 1 meaning none
                projection_dim          - width of the decoder this feeds
                image_mean, image_std   - per-channel normalization of the input, which
                                          belongs to the encoder rather than to the loader

            METHODS
                head_dim()         - d_model / n_heads
                grid_side()        - image_size / patch_size
                num_patches()      - grid_side squared
                tokens_per_image() - positions the decoder receives, after the reduction
                folded_width()     - width of one of those positions before the projector
        !*/
    };

    vision_spec detect_vision(
        const gguf_reader& g
    );
    /*!
        ensures
            - Returns the geometry the container declares.
        throws
            - std::runtime_error when the container is not a vision projector, that is
              when its architecture is not "clip".
    !*/

// ----------------------------------------------------------------------------------------

    struct vision_compat_result
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What a container costs to serve, split between what a caller should know
                and what stops it outright.

            FIELDS
                notes    - departures from the assumed defaults, worth reporting
                blockers - reasons the container cannot be served as it stands

            METHODS
                usable() - blockers.empty()
        !*/
    };

    vision_compat_result check_vision_compatibility(
        const vision_spec& s,
        const gguf_reader& g
    );
    /*!
        ensures
            - Checks the declared geometry against itself and against the tensors present:
              the patch grid divides the image, the pixel shuffle divides the grid, the
              heads divide the width, every layer carries its weights, the position table
              covers the grid, and the projector matrix has the folded width for input and
              the decoder width for output.
            - Returns those findings, blockers first.
            - The projector check is the one that matters most: it is where the patch grid,
              the reduction factor and the trained weights have to agree, and the only
              place where a mistaken pixel shuffle factor becomes visible before inference.
    !*/

    std::string describe(
        const vision_spec& s
    );
    /*!
        ensures
            - Returns a human-readable report of the geometry, in the layout the decoder
              report uses, so the two halves of a multimodal model read alike.
    !*/
}

#endif // DLIB_GGUF_VISION_SPEC_ABSTRACT_H_
