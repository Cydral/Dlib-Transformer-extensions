// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_RUNTIME_VISION_ENCODER_ABSTRACT_H_
#ifdef DLIB_DNN_RUNTIME_VISION_ENCODER_ABSTRACT_H_

#include "../data_io/gguf_vision_spec_abstract.h"
#include "../cuda/tensor_abstract.h"

namespace dlib
{
    class runtime_vision_encoder
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A vision tower and its projector, resolved at load time from the container
                a multimodal model ships alongside its decoder. It compiles once and serves
                any geometry gguf_vision_spec.h accepts.

                That matters more here than on the decoder side. A factory ingesting
                arbitrary open-weight models meets a new patch size or a new grid far more
                often than a new decoder shape, and none of them should cost the hour and a
                half a template instantiation takes.

            THE PIPELINE
                    convolution, kernel equal to stride   a grid of patch vectors
                    + learned positions
                    N pre-norm blocks                     LayerNorm, attention, GELU
                    LayerNorm
                    pixel shuffle                         fewer positions, more channels
                    one linear layer                      the decoder's embedding width

            THREE THINGS WORTH KNOWING
                The attention carries no mask. A vision tower reads an image, where no
                position comes before another, so every patch attends to every patch. That
                is why this encoder is written against tensor operations rather than
                against the decoder's attention layer, which is causal by construction.

                The patch embedding is a convolution whose kernel equals its stride, and
                the container stores its filters in the layout tensor_conv expects: the
                weight goes in as it comes out of the file.

                The feed-forward tensor names are inverted relative to the decoder's
                convention. In this container ffn_down carries the expansion and ffn_up the
                contraction, which their bias sizes confirm. Reading them the other way
                round produces an encoder that runs, returns plausible numbers, and
                describes the wrong image.

            THREAD SAFETY
                One instance holds the scratch buffers of one forward and is not thread
                safe.

            TYPICAL USAGE
                gguf_reader gv("mmproj-model.gguf");
                const vision_spec vs = detect_vision(gv);
                if (!check_vision_compatibility(vs, gv).usable()) return;

                runtime_vision_encoder enc;
                enc.load(gv, vs);

                matrix<rgb_pixel> img;
                load_image(img, "photo.jpg");
                resizable_tensor prepared;
                enc.prepare_image(img, prepared);
                const tensor& visual = enc.encode(prepared);
                // visual is [vs.tokens_per_image(), vs.projection_dim()]
        !*/

    public:

        const vision_spec& spec() const;
        /*!
            ensures
                - Returns the geometry this encoder was loaded with.
        !*/

        bool loaded() const;
        /*!
            ensures
                - Returns whether load() has run.
        !*/

        void load(
            gguf_reader& g,
            const vision_spec& s
        );
        /*!
            requires
                - s was returned by detect_vision(g)
                - check_vision_compatibility(s, g).usable() == true
            ensures
                - Reads every weight of the tower into resident float32.
                - The container is small and, unlike the decoder, competes with no
                  generation loop for memory, so nothing is kept quantized at rest.
            throws
                - std::runtime_error on a missing tensor or a shape that contradicts the
                  geometry.
        !*/

        template <typename image_type>
        void prepare_image(
            const image_type& img,
            resizable_tensor& out
        ) const;
        /*!
            requires
                - load() has run
                - image_type is an image object as defined in dlib/image_processing
            ensures
                - #out is a [1, 3, image_size, image_size] tensor holding img resized to
                  the square the tower expects and normalized channel by channel.
                - The normalization constants come from the container rather than from a
                  table of known families, because they belong to the encoder: a tower
                  trained on data centered at 0.5 and one centered on the ImageNet
                  statistics see different pictures for the same file.
        !*/

        const tensor& encode(
            const tensor& image
        );
        /*!
            requires
                - loaded() == true
                - image has the dimensions prepare_image() produces
            ensures
                - Returns the visual embeddings of the image as a
                  [spec().tokens_per_image(), spec().projection_dim] matrix, ready to be
                  written over the placeholder positions of a token stream.
                - The result lives in the encoder and stays valid until the next call.
        !*/
    };
}

#endif // DLIB_DNN_RUNTIME_VISION_ENCODER_ABSTRACT_H_
