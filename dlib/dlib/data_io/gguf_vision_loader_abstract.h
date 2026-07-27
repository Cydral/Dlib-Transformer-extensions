// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GGUF_VISION_LOADER_ABSTRACT_H_
#ifdef DLIB_GGUF_VISION_LOADER_ABSTRACT_H_

#include "gguf_reader_abstract.h"
#include "gguf_vision_spec_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct gguf_vision_load_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Options of the vision weight import.

                - verbose: print a one line summary of what was loaded.
        !*/

        bool verbose = true;
    };

// ----------------------------------------------------------------------------------------

    template <typename net_type, typename config>
    void import_gguf_vision_weights(
        net_type& net,
        gguf_reader& g,
        const vision_spec& spec,
        const config&,
        const gguf_vision_load_options& opt = gguf_vision_load_options()
    );
    /*!
        requires
            - config is a vision_transformer_config instantiation and net_type is its
              network_type.
            - g is open on a multimodal projector container, conventionally named
              mmproj-*.gguf, and spec was obtained from detect_vision(g).
        ensures
            - Every parameter of net is filled from the container: the patch embedding
              convolution, the position table, each encoder block, the final normalization
              and the projector.
            - The normalization epsilon of the container is pushed into every layer_norm of
              net before anything else, since a tower trained at 1e-6 and run at Dlib's
              1e-5 default drifts quietly rather than failing.
            - net has been run once on a blank image, which is what allocates its parameter
              tensors, so it is ready to use on return.
        throws
            - std::runtime_error if the geometry of spec disagrees with config, if a tensor
              the tower needs is absent from the container, if any tensor has a shape the
              corresponding layer cannot hold, or if the walk ends with parameters left
              unfilled. Each of these is reported before any partially filled network can
              be mistaken for a working one.

        THE MAPPING BETWEEN THE FILE AND THE LAYERS

            Three points of this mapping are not obvious and are the reason this function
            exists rather than a loop over tensor names.

            The feed-forward names are inverted relative to the decoder's convention: in
            this container ffn_down carries the expansion and ffn_up the contraction, which
            their bias sizes confirm without ambiguity. Read the other way round they
            produce a tower that runs and describes the wrong image.

            Linear weights are stored as out rows of in values, while every layer here
            multiplies x[rows, in] by W[in, out], so each is transposed on the way in. The
            convolution is the exception: its filters are already in the layout con
            expects and go in unchanged.

            A bias the container does not carry is written as zero rather than left as it
            was, so that an absent bias reads as absent and not as whatever the layer was
            initialized with.

        VALIDATION

            The result is verifiable against runtime_vision_encoder, which is the
            shape-dynamic implementation of the same encoder: loaded from the same
            container and run on the same prepared image, the two produce identical
            tensors. See examples/slm_vision_tower_ex.cpp, which performs exactly that
            comparison.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GGUF_VISION_LOADER_ABSTRACT_H_
