// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_VISION_TRANSFORMER_ABSTRACT_H_
#ifdef DLIB_DNN_VISION_TRANSFORMER_ABSTRACT_H_

#include "core_abstract.h"
#include "layers_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
/*!A tensor layout used throughout this header

    Every layer below carries one patch per sample, features on k, that is tensors of
    shape [images*patches, width, 1, 1]. This is not an arbitrary convention: it is the
    layout in which the existing layers are already correct.

        - layer_norm normalizes a sample over k*nr*nc with a gamma and a beta indexed by
          k, which is exactly per-patch LayerNorm over the features.
        - fc consumes k*nr*nc and emits [samples, out, 1, 1], which is exactly a
          position-wise linear projection.
        - a bias of shape [1, width, 1, 1] broadcasts over samples through tt::add.

    The sequence layout the decoder uses, [batch, 1, positions, width], would have made
    all three wrong, which is why the two stacks do not share one.

    Attention is the single operation that mixes patches, so it is the single fused layer
    here; it reinterprets its input as [images, 1, patches, width] at no cost, both views
    sharing their storage.
!*/

// ----------------------------------------------------------------------------------------

    class patch_sequence_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It turns the grid a convolution produces
                into the patch sequence the rest of a vision tower works in.

                An input of shape [images, channels, side, side] becomes an output of shape
                [images*side*side, channels, 1, 1]. The two hold the same values in a
                different order, the grid being channel-major and the sequence
                position-major, so the operation is the transpose of the
                [channels, positions] matrix of each image.

                This layer has no parameters.
        !*/

    public:

        patch_sequence_(
        );
        /*!
            ensures
                - This object is properly initialized.
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

            forward() requires that sub.get_output() have a positive spatial extent.
        !*/
    };

    void serialize(const patch_sequence_& item, std::ostream& out);
    void deserialize(patch_sequence_& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    template <typename SUBNET>
    using patch_sequence = add_layer<patch_sequence_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long NUM_PATCHES,
        long WIDTH
        >
    class patch_positions_
    {
        /*!
            REQUIREMENTS ON NUM_PATCHES AND WIDTH
                Both must be > 0.

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It adds a learned position vector to each
                patch of its input, which must be shaped
                [images*NUM_PATCHES, WIDTH, 1, 1].

                The table is a parameter rather than a fixed encoding: the containers this
                stack imports carry a trained one, and a tower trained here should learn
                its own. Several images in a batch share the table, so the row added to a
                sample is chosen by its position within its image and not by its index in
                the batch.

                The parameter tensor is the [NUM_PATCHES, WIDTH] table itself. Its weight
                decay multiplier is 0 by default, a position table having no reason to be
                pulled towards zero.
        !*/

    public:

        patch_positions_(
        );
        /*!
            ensures
                - #get_learning_rate_multiplier() == 1
                - #get_weight_decay_multiplier() == 0
        !*/

        double get_learning_rate_multiplier() const;
        double get_weight_decay_multiplier() const;
        void set_learning_rate_multiplier(double val);
        void set_weight_decay_multiplier(double val);
        /*!
            These are implemented as described in the EXAMPLE_COMPUTATIONAL_LAYER_
            interface.
        !*/

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            These functions are implemented as described in the
            EXAMPLE_COMPUTATIONAL_LAYER_ interface.

            setup() requires that sub.get_output().k() == WIDTH.
            forward() requires that sub.get_output().num_samples() be a whole number of
            images, that is a multiple of NUM_PATCHES.
        !*/
    };

    template <long NUM_PATCHES, long WIDTH>
    void serialize(const patch_positions_<NUM_PATCHES, WIDTH>& item, std::ostream& out);
    template <long NUM_PATCHES, long WIDTH>
    void deserialize(patch_positions_<NUM_PATCHES, WIDTH>& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    template <long NUM_PATCHES, long WIDTH, typename SUBNET>
    using patch_positions = add_layer<patch_positions_<NUM_PATCHES, WIDTH>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long FACTOR,
        long GRID_SIDE
        >
    class patch_shuffle_
    {
        /*!
            REQUIREMENTS ON FACTOR AND GRID_SIDE
                Both must be > 0 and GRID_SIDE must be a multiple of FACTOR.

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It folds a FACTOR by FACTOR neighbourhood of
                patches into the channels, which the idefics3 family calls a pixel shuffle
                and which is what divides the position count before a projector.

                An input of shape [images*GRID_SIDE*GRID_SIDE, channels, 1, 1] becomes an
                output of shape
                [images*(GRID_SIDE/FACTOR)^2, channels*FACTOR^2, 1, 1].

                The channel order the fold produces is the one a trained projector was
                fitted against, so it is not a free choice: read the other way round it
                yields an encoder that runs, returns plausible numbers, and describes the
                wrong image.

                This layer has no parameters.
        !*/

    public:

        static constexpr long NUM_PATCHES = GRID_SIDE * GRID_SIDE;
        static constexpr long OUT_SIDE    = GRID_SIDE / FACTOR;
        static constexpr long OUT_PATCHES = OUT_SIDE * OUT_SIDE;

        patch_shuffle_(
        );
        /*!
            ensures
                - This object is properly initialized.
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

            forward() requires that sub.get_output().num_samples() be a multiple of
            NUM_PATCHES.
        !*/
    };

    template <long FACTOR, long GRID_SIDE>
    void serialize(const patch_shuffle_<FACTOR, GRID_SIDE>& item, std::ostream& out);
    template <long FACTOR, long GRID_SIDE>
    void deserialize(patch_shuffle_<FACTOR, GRID_SIDE>& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    template <long FACTOR, long GRID_SIDE, typename SUBNET>
    using patch_shuffle = add_layer<patch_shuffle_<FACTOR, GRID_SIDE>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long TOKENS
        >
    class patch_pool_
    {
        /*!
            REQUIREMENTS ON TOKENS
                TOKENS > 0

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It averages the tokens of each image:
                an input of shape [images*TOKENS, width, 1, 1] becomes an output of shape
                [images, width, 1, 1], one vector per picture.

                This is the brick a vision-only use needs and that no existing layer
                provides. In the layout of this tower a token is a sample, so pooling over
                tokens is pooling over samples, which the pooling layers cannot express:
                they work inside a sample. With this layer on top, the tower feeds a
                classification head, a metric loss for face recognition, or a
                self-supervised objective, exactly like any other Dlib backbone.

                This layer has no parameters.
        !*/

    public:

        patch_pool_(
        );

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            These functions are implemented as described in the
            EXAMPLE_COMPUTATIONAL_LAYER_ interface.  Note that get_layer_params() always
            returns an empty tensor.

            forward() requires an input shaped [samples, width, 1, 1] whose sample count is
            a multiple of TOKENS.
        !*/
    };

    template <long TOKENS>
    void serialize(const patch_pool_<TOKENS>& item, std::ostream& out);
    template <long TOKENS>
    void deserialize(patch_pool_<TOKENS>& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    template <long TOKENS, typename SUBNET>
    using patch_pool = add_layer<patch_pool_<TOKENS>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long WIDTH,
        long NUM_HEADS,
        long NUM_PATCHES
        >
    class vision_attention_
    {
        /*!
            REQUIREMENTS ON THE TEMPLATE ARGUMENTS
                All must be > 0 and WIDTH must be a multiple of NUM_HEADS.

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_COMPUTATIONAL_LAYER_ interface
                defined in layers_abstract.h.  It is bidirectional multi-head attention
                over the patches of an image, with a bias on each of its four projections.
                Its input and its output are both shaped
                [images*NUM_PATCHES, WIDTH, 1, 1].

                It carries no mask. A vision tower reads an image, where no position comes
                before another, so every patch attends to every patch. It carries no rotary
                encoding either, position being supplied by patch_positions_, and no
                key/value cache, an image being encoded once.

                NUM_PATCHES is a template argument because the layer would otherwise have
                no way to tell one image from the next in a batch: samples are patches
                here, and attention must not run across the boundary between two pictures.

                The parameter blob holds four [WIDTH, WIDTH] matrices in the order q, k, v,
                out, followed by four biases of WIDTH values in the same order. The offsets
                are exposed as constexpr functions so that a loader need not restate the
                layout.
        !*/

    public:

        static constexpr long HEAD_DIM = WIDTH / NUM_HEADS;

        vision_attention_(
        );
        /*!
            ensures
                - #get_learning_rate_multiplier() == 1
                - #get_weight_decay_multiplier() == 1
        !*/

        double get_learning_rate_multiplier() const;
        double get_weight_decay_multiplier() const;
        void set_learning_rate_multiplier(double val);
        void set_weight_decay_multiplier(double val);
        /*!
            These are implemented as described in the EXAMPLE_COMPUTATIONAL_LAYER_
            interface.
        !*/

        static constexpr size_t weight_count();
        static constexpr size_t parameter_count();
        static constexpr size_t wq_offset();
        static constexpr size_t wk_offset();
        static constexpr size_t wv_offset();
        static constexpr size_t wo_offset();
        static constexpr size_t bq_offset();
        static constexpr size_t bk_offset();
        static constexpr size_t bv_offset();
        static constexpr size_t bo_offset();
        /*!
            ensures
                - weight_count() == WIDTH*WIDTH, the size of one projection matrix.
                - parameter_count() == 4*weight_count() + 4*WIDTH, which is the size of
                  get_layer_params() once setup() has run.
                - the *_offset() functions give the index, within get_layer_params(), at
                  which each matrix and each bias begins. Each matrix is stored as WIDTH
                  rows of WIDTH values, an input index selecting the row, which is the
                  layout tt::gemm multiplies against without a transpose.
        !*/

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            These functions are implemented as described in the
            EXAMPLE_COMPUTATIONAL_LAYER_ interface.

            setup() requires an input shaped [samples, WIDTH, 1, 1].
            forward() requires that sub.get_output().num_samples() be a multiple of
            NUM_PATCHES.
        !*/
    };

    template <long WIDTH, long NUM_HEADS, long NUM_PATCHES>
    void serialize(const vision_attention_<WIDTH, NUM_HEADS, NUM_PATCHES>& item, std::ostream& out);
    template <long WIDTH, long NUM_HEADS, long NUM_PATCHES>
    void deserialize(vision_attention_<WIDTH, NUM_HEADS, NUM_PATCHES>& item, std::istream& in);
    /*!
        provides serialization support
    !*/

    template <long WIDTH, long NUM_HEADS, long NUM_PATCHES, typename SUBNET>
    using vision_attention = add_layer<vision_attention_<WIDTH, NUM_HEADS, NUM_PATCHES>, SUBNET>;

// ----------------------------------------------------------------------------------------

    namespace vision_transformer
    {
        template <long WIDTH, long NUM_HEADS, long NUM_PATCHES, long FFN_HIDDEN, typename SUBNET>
        using vision_block = /*!see the implementation header!*/;
        /*!
            WHAT THIS OBJECT REPRESENTS
                One pre-norm encoder block: LayerNorm then bidirectional attention with a
                residual, LayerNorm then a GELU feed-forward with a residual.

                The topology is that of the decoder's block. What differs is the
                normalization, which subtracts a mean and carries a bias here, the absence
                of any mask, and the feed-forward, which is a plain expansion rather than a
                gated one.
        !*/

        template <long NUM_LAYERS, long WIDTH, long NUM_HEADS, long NUM_PATCHES,
            long FFN_HIDDEN, typename SUBNET>
        using vision_stack = /*!see the implementation header!*/;
        /*!
            ensures
                - Stacks NUM_LAYERS vision_block on top of SUBNET.
        !*/
    }

// ----------------------------------------------------------------------------------------

    template <
        long image_size,
        long patch_size,
        long width,
        long num_layers,
        long num_heads,
        long ffn_hidden,
        long shuffle_factor,
        long projection_dim
        >
    struct vision_transformer_config
    {
        /*!
            REQUIREMENTS ON THE TEMPLATE ARGUMENTS
                - image_size % patch_size == 0
                - the resulting grid side must be divisible by shuffle_factor

            WHAT THIS OBJECT REPRESENTS
                This object gathers the geometry of a vision tower and exposes the Dlib
                network type that implements it. It is the static counterpart of
                runtime_vision_encoder: the same encoder expressed as a network type, so
                that its weights live in an archive, its gradients flow, and it can be
                trained or adapted like any other network here.

                network_type expects a normalized image tensor of shape
                [images, 3, image_size, image_size] and produces
                [images*NUM_TOKENS, projection_dim, 1, 1], one row per visual position.

                The pipeline is: a convolution whose kernel equals its stride, which makes
                the grid exactly the patches; the grid turned into a patch sequence; the
                learned position table; num_layers encoder blocks; a final normalization;
                the pixel shuffle; and the projector.

                Its input layer is input_tensor behind a tag, so the tower can be driven
                directly with a prepared image tensor and can also serve as the subnetwork
                of an enclosing layer.

                The pixel normalization that turns an image into that input tensor is a
                property of the trained model rather than of this type: see
                runtime_vision_encoder::prepare_image, which reads it from the container.
        !*/

        static constexpr long IMAGE_SIZE     = image_size;
        static constexpr long PATCH_SIZE     = patch_size;
        static constexpr long WIDTH          = width;
        static constexpr long NUM_LAYERS     = num_layers;
        static constexpr long NUM_HEADS      = num_heads;
        static constexpr long FFN_HIDDEN     = ffn_hidden;
        static constexpr long SHUFFLE_FACTOR = shuffle_factor;
        static constexpr long PROJECTION_DIM = projection_dim;

        static constexpr long GRID_SIDE    = image_size / patch_size;
        static constexpr long NUM_PATCHES  = GRID_SIDE * GRID_SIDE;
        static constexpr long OUT_SIDE     = GRID_SIDE / shuffle_factor;
        static constexpr long NUM_TOKENS   = OUT_SIDE * OUT_SIDE;
        static constexpr long FOLDED_WIDTH = width * shuffle_factor * shuffle_factor;

        using stack        = /*!see the implementation header!*/;
        using network_type = /*!see the implementation header!*/;

        using image_embedding = /*!see the implementation header!*/;
        template <long num_classes>
        using classifier = /*!see the implementation header!*/;
        /*!
            WHAT THESE OBJECTS REPRESENT
                Vision-only use. The tower is a backbone in its own right: pooled over the
                tokens of each image it yields one vector per picture, which any Dlib head
                can consume. image_embedding is the tower plus that pooling, producing
                [images, projection_dim, 1, 1]; classifier adds a linear head over it.

                Nothing here involves a decoder, so the same tower serves image
                classification, metric learning for face recognition, or a self-supervised
                objective, and a tower trained that way can afterwards be handed to a
                fusion layer as an already trained encoder.
        !*/

        struct model_info
        {
            static std::string describe();
            /*!
                ensures
                    - Returns a human readable summary of the geometry above.
            !*/
        };
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_VISION_TRANSFORMER_ABSTRACT_H_
