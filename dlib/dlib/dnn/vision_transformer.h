// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Vision transformer building blocks for the Dlib layer stack.
//
// Static counterpart of runtime_vision_encoder: the same encoder expressed as a Dlib
// network type, so that its weights live in the archive, its gradients flow, and it can
// be trained or adapted like any other network here.
//
// The layout is the one decision everything else follows from. Throughout this tower one
// patch is one sample, features on k, tensors of shape [patches, width, 1, 1]. The reason
// is that the existing layers are then correct as they stand: layer_norm normalizes a
// sample over k*nr*nc with a gamma and a beta indexed by k, which is exactly per-patch
// LayerNorm over the features; fc consumes k*nr*nc and emits [patches, out, 1, 1], which
// is exactly a position-wise linear; and a bias of shape [1, width, 1, 1] broadcasts over
// samples through tt::add, so no rank-one trick is needed. The sequence layout used by
// the decoder, [batch, 1, positions, width], would have made all three wrong.
//
// Attention is the one operation that mixes patches, so it is the one that has to leave
// this layout, and it is the only fused layer here. It reinterprets its input as
// [images, 1, patches, width] at no cost, since the two views share their storage, splits
// the heads, and comes back. It carries no mask: a vision tower reads an image, where no
// position comes before another, so every patch attends to every patch.
//
// The three remaining layers are shape work. patch_sequence turns the grid a convolution
// produces into one sample per patch; patch_positions adds the learned table; patch_shuffle
// folds a spatial neighbourhood into the channels, which is what the idefics3 family calls
// a pixel shuffle and what divides the position count before the projector.

#ifndef DLIB_DNN_VISION_TRANSFORMER_H_
#define DLIB_DNN_VISION_TRANSFORMER_H_

#include "vision_transformer_abstract.h"

#include <cmath>
#include <sstream>
#include <string>

#include "core.h"
#include "layers.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /* Grid to patch sequence. Input [images, channels, side, side] as a convolution leaves
       it, output [images * side * side, channels, 1, 1]. The two hold the same values in a
       different order: the grid is channel-major, the sequence is position-major, so the
       operation is the transpose of the [channels, positions] matrix of each image. */
    class patch_sequence_
    {
    public:

        patch_sequence_() {}

        template <typename SUBNET> void setup(const SUBNET& /*sub*/) {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& src = sub.get_output();
            const long B = src.num_samples(), C = src.k(), N = src.nr() * src.nc();
            DLIB_CASSERT(N > 0, "patch_sequence expects a spatial input");

            output.set_size(B * N, C, 1, 1);
            alias_tensor as_grid(1, 1, C, N), as_seq(1, 1, N, C);
            for (long b = 0; b < B; ++b)
            {
                auto g = as_grid(const_cast<tensor&>(src), static_cast<size_t>(b) * C * N);
                auto s = as_seq(output, static_cast<size_t>(b) * C * N);
                tt::transpose(false, s, g);
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tensor& grad = sub.get_gradient_input();
            const long B = grad.num_samples(), C = grad.k(), N = grad.nr() * grad.nc();

            alias_tensor as_grid(1, 1, C, N), as_seq(1, 1, N, C);
            for (long b = 0; b < B; ++b)
            {
                auto g = as_grid(grad, static_cast<size_t>(b) * C * N);
                auto s = as_seq(const_cast<tensor&>(gradient_input),
                    static_cast<size_t>(b) * C * N);
                tt::transpose(true, g, s);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const patch_sequence_&, std::ostream& out)
        {
            serialize("patch_sequence_", out);
        }
        friend void deserialize(patch_sequence_&, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "patch_sequence_")
                throw serialization_error("Unexpected version found while deserializing dlib::patch_sequence_.");
        }
        friend std::ostream& operator<<(std::ostream& out, const patch_sequence_&)
        {
            out << "patch_sequence";
            return out;
        }
        friend void to_xml(const patch_sequence_&, std::ostream& out)
        {
            out << "<patch_sequence/>\n";
        }

    private:
        resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using patch_sequence = add_layer<patch_sequence_, SUBNET>;

// ----------------------------------------------------------------------------------------

    /* Learned position table, one vector per patch, added to the patch embeddings. The
       table is a parameter of the layer rather than a fixed encoding: the containers this
       tower reads carry a trained one, and a tower trained here should learn it too.

       Several images in a batch share the table, so the row index is the position within
       the image rather than the sample index. */
    template <long NUM_PATCHES_, long WIDTH_>
    class patch_positions_
    {
        static_assert(NUM_PATCHES_ > 0, "NUM_PATCHES must be positive");
        static_assert(WIDTH_ > 0, "WIDTH must be positive");

    public:

        static constexpr long NUM_PATCHES = NUM_PATCHES_;
        static constexpr long WIDTH = WIDTH_;

        patch_positions_() : learning_rate_multiplier_(1), weight_decay_multiplier_(0) {}

        double get_learning_rate_multiplier() const { return learning_rate_multiplier_; }
        double get_weight_decay_multiplier() const { return weight_decay_multiplier_; }
        void set_learning_rate_multiplier(double v) { learning_rate_multiplier_ = v; }
        void set_weight_decay_multiplier(double v) { weight_decay_multiplier_ = v; }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            DLIB_CASSERT(sub.get_output().k() == WIDTH,
                "patch_positions input width " << sub.get_output().k()
                << " does not match WIDTH=" << WIDTH);
            params.set_size(NUM_PATCHES, WIDTH);
            /* Filled element by element rather than through tt::tensor_rand, which
               requires an even count: a table of an odd number of values is perfectly
               legitimate here and would otherwise assert. */
            dlib::rand rnd(std::rand());
            float* p = params.host();
            for (size_t i = 0; i < params.size(); ++i)
                p[i] = 0.02f * static_cast<float>(rnd.get_random_gaussian());
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& src = sub.get_output();
            DLIB_CASSERT(src.num_samples() % NUM_PATCHES == 0,
                "patch_positions received " << src.num_samples()
                << " samples, which is not a whole number of images of "
                << NUM_PATCHES << " patches");

            output.copy_size(src);
            memcpy(output, src);

            const long B = src.num_samples() / NUM_PATCHES;
            alias_tensor block(NUM_PATCHES, WIDTH);
            for (long b = 0; b < B; ++b)
            {
                auto view = block(output, static_cast<size_t>(b) * NUM_PATCHES * WIDTH);
                tt::add(1.0f, view, 1.0f, params);
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            tt::add(1.0f, sub.get_gradient_input(), 1.0f, gradient_input);

            const long B = gradient_input.num_samples() / NUM_PATCHES;
            alias_tensor block(NUM_PATCHES, WIDTH);
            params_grad = 0;
            for (long b = 0; b < B; ++b)
            {
                auto view = block(const_cast<tensor&>(gradient_input),
                    static_cast<size_t>(b) * NUM_PATCHES * WIDTH);
                tt::add(1.0f, params_grad, 1.0f, view);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const patch_positions_& item, std::ostream& out)
        {
            serialize("patch_positions_", out);
            serialize(item.params, out);
            serialize(item.learning_rate_multiplier_, out);
            serialize(item.weight_decay_multiplier_, out);
        }
        friend void deserialize(patch_positions_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "patch_positions_")
                throw serialization_error("Unexpected version found while deserializing dlib::patch_positions_.");
            deserialize(item.params, in);
            deserialize(item.learning_rate_multiplier_, in);
            deserialize(item.weight_decay_multiplier_, in);
        }
        friend std::ostream& operator<<(std::ostream& out, const patch_positions_& item)
        {
            out << "patch_positions (num_patches=" << NUM_PATCHES << ", width=" << WIDTH
                << ") learning_rate_mult=" << item.learning_rate_multiplier_;
            return out;
        }
        friend void to_xml(const patch_positions_& item, std::ostream& out)
        {
            out << "<patch_positions num_patches='" << NUM_PATCHES << "' width='" << WIDTH
                << "'>\n" << mat(item.params) << "</patch_positions>\n";
        }

    private:
        resizable_tensor params;
        double learning_rate_multiplier_;
        double weight_decay_multiplier_;
    };

    template <long NUM_PATCHES, long WIDTH, typename SUBNET>
    using patch_positions = add_layer<patch_positions_<NUM_PATCHES, WIDTH>, SUBNET>;

// ----------------------------------------------------------------------------------------

    /* Pixel shuffle. Folds a FACTOR x FACTOR neighbourhood of patches into the channels,
       dividing the position count by FACTOR squared and multiplying the width by as much.
       The sequence goes back to a grid, tt::reorg does the fold, and the result becomes a
       sequence again, shorter and wider.

       The channel order the fold produces is what the projector was trained against, so it
       is not a free choice: reading it the other way round gives an encoder that runs and
       describes the wrong image. */
    template <long FACTOR_, long GRID_SIDE_>
    class patch_shuffle_
    {
        static_assert(FACTOR_ > 0, "FACTOR must be positive");
        static_assert(GRID_SIDE_ > 0, "GRID_SIDE must be positive");
        static_assert(GRID_SIDE_ % FACTOR_ == 0,
            "the patch grid must be divisible by the shuffle factor");

    public:

        static constexpr long FACTOR = FACTOR_;
        static constexpr long GRID_SIDE = GRID_SIDE_;
        static constexpr long NUM_PATCHES = GRID_SIDE_ * GRID_SIDE_;
        static constexpr long OUT_SIDE = GRID_SIDE_ / FACTOR_;
        static constexpr long OUT_PATCHES = OUT_SIDE * OUT_SIDE;

        patch_shuffle_() {}

        template <typename SUBNET> void setup(const SUBNET& /*sub*/) {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& src = sub.get_output();
            const long C = src.k();
            DLIB_CASSERT(src.num_samples() % NUM_PATCHES == 0,
                "patch_shuffle received " << src.num_samples()
                << " samples, which is not a whole number of grids of " << NUM_PATCHES);

            const long B = src.num_samples() / NUM_PATCHES;
            const long OUT_C = C * FACTOR * FACTOR;
            output.set_size(B * OUT_PATCHES, OUT_C, 1, 1);

            grid_.set_size(1, C, GRID_SIDE, GRID_SIDE);
            folded_.set_size(1, OUT_C, OUT_SIDE, OUT_SIDE);
            alias_tensor as_seq(1, 1, NUM_PATCHES, C), as_grid(1, 1, C, NUM_PATCHES);
            alias_tensor as_fold(1, 1, OUT_C, OUT_PATCHES), as_out(1, 1, OUT_PATCHES, OUT_C);

            for (long b = 0; b < B; ++b)
            {
                auto s = as_seq(const_cast<tensor&>(src),
                    static_cast<size_t>(b) * NUM_PATCHES * C);
                auto g = as_grid(grid_, 0);
                tt::transpose(false, g, s);

                tt::reorg(false, folded_, static_cast<int>(FACTOR), static_cast<int>(FACTOR), grid_);

                auto f = as_fold(folded_, 0);
                auto o = as_out(output, static_cast<size_t>(b) * OUT_PATCHES * OUT_C);
                tt::transpose(false, o, f);
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tensor& grad = sub.get_gradient_input();
            const long C = grad.k();
            const long B = grad.num_samples() / NUM_PATCHES;
            const long OUT_C = C * FACTOR * FACTOR;

            dfolded_.set_size(1, OUT_C, OUT_SIDE, OUT_SIDE);
            dgrid_.set_size(1, C, GRID_SIDE, GRID_SIDE);
            alias_tensor as_out(1, 1, OUT_PATCHES, OUT_C), as_fold(1, 1, OUT_C, OUT_PATCHES);
            alias_tensor as_grid(1, 1, C, NUM_PATCHES), as_seq(1, 1, NUM_PATCHES, C);

            for (long b = 0; b < B; ++b)
            {
                auto o = as_out(const_cast<tensor&>(gradient_input),
                    static_cast<size_t>(b) * OUT_PATCHES * OUT_C);
                auto f = as_fold(dfolded_, 0);
                tt::transpose(false, f, o);

                tt::reorg_gradient(false, dgrid_, static_cast<int>(FACTOR),
                    static_cast<int>(FACTOR), dfolded_);

                auto g = as_grid(dgrid_, 0);
                auto s = as_seq(grad, static_cast<size_t>(b) * NUM_PATCHES * C);
                tt::transpose(true, s, g);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const patch_shuffle_&, std::ostream& out)
        {
            serialize("patch_shuffle_", out);
        }
        friend void deserialize(patch_shuffle_&, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "patch_shuffle_")
                throw serialization_error("Unexpected version found while deserializing dlib::patch_shuffle_.");
        }
        friend std::ostream& operator<<(std::ostream& out, const patch_shuffle_&)
        {
            out << "patch_shuffle (factor=" << FACTOR << ", grid=" << GRID_SIDE << ")";
            return out;
        }
        friend void to_xml(const patch_shuffle_&, std::ostream& out)
        {
            out << "<patch_shuffle factor='" << FACTOR << "' grid='" << GRID_SIDE << "'/>\n";
        }

    private:
        resizable_tensor params; // unused
        resizable_tensor grid_, folded_, dgrid_, dfolded_;
    };

    template <long FACTOR, long GRID_SIDE, typename SUBNET>
    using patch_shuffle = add_layer<patch_shuffle_<FACTOR, GRID_SIDE>, SUBNET>;

// ----------------------------------------------------------------------------------------

    /* Bidirectional multi-head attention over the patches of an image, with biases on the
       four projections. Input and output are [images * NUM_PATCHES, WIDTH, 1, 1].

       Fused rather than chained for the reason given at the top of this file: the head
       split needs [images, heads, patches, head_dim], which is not the layout the rest of
       the tower works in. The reinterpretation between the two costs nothing, both views
       sharing their storage, but it has to happen somewhere and a single layer is the
       cheapest place to confine it.

       NUM_PATCHES is a template parameter because the layer would otherwise have no way to
       tell one image from the next in a batch: samples are patches here, and attention must
       not run across the boundary between two pictures. */
    template <long WIDTH_, long NUM_HEADS_, long NUM_PATCHES_>
    class vision_attention_
    {
        static_assert(WIDTH_ > 0, "WIDTH must be positive");
        static_assert(NUM_HEADS_ > 0, "NUM_HEADS must be positive");
        static_assert(WIDTH_ % NUM_HEADS_ == 0, "WIDTH must be a whole number of heads");
        static_assert(NUM_PATCHES_ > 0, "NUM_PATCHES must be positive");

    public:

        static constexpr long WIDTH = WIDTH_;
        static constexpr long NUM_HEADS = NUM_HEADS_;
        static constexpr long NUM_PATCHES = NUM_PATCHES_;
        static constexpr long HEAD_DIM = WIDTH_ / NUM_HEADS_;

        vision_attention_() : learning_rate_multiplier_(1), weight_decay_multiplier_(1) {}

        double get_learning_rate_multiplier() const { return learning_rate_multiplier_; }
        double get_weight_decay_multiplier() const { return weight_decay_multiplier_; }
        void set_learning_rate_multiplier(double v) { learning_rate_multiplier_ = v; }
        void set_weight_decay_multiplier(double v) { weight_decay_multiplier_ = v; }

        /* Offsets of the packed parameter blob, in the order weights then biases. The
           loader writes through these so that neither it nor this class has to agree on a
           layout twice. */
        static constexpr size_t weight_count() { return static_cast<size_t>(WIDTH) * WIDTH; }
        static constexpr size_t wq_offset() { return 0; }
        static constexpr size_t wk_offset() { return weight_count(); }
        static constexpr size_t wv_offset() { return 2 * weight_count(); }
        static constexpr size_t wo_offset() { return 3 * weight_count(); }
        static constexpr size_t bq_offset() { return 4 * weight_count(); }
        static constexpr size_t bk_offset() { return bq_offset() + WIDTH; }
        static constexpr size_t bv_offset() { return bk_offset() + WIDTH; }
        static constexpr size_t bo_offset() { return bv_offset() + WIDTH; }
        static constexpr size_t parameter_count() { return 4 * weight_count() + 4 * WIDTH; }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            const tensor& x = sub.get_output();
            DLIB_CASSERT(x.k() == WIDTH && x.nr() == 1 && x.nc() == 1,
                "vision_attention expects [patches, WIDTH, 1, 1] input, got k=" << x.k()
                << " nr=" << x.nr() << " nc=" << x.nc());

            params.set_size(1, static_cast<long>(4 * weight_count() + 4 * WIDTH));

            dlib::rand rnd(std::rand());
            const float sigma = static_cast<float>(std::sqrt(2.0 / (WIDTH + WIDTH)));
            float* p = params.host();
            for (size_t i = 0; i < 4 * weight_count(); ++i)
                p[i] = sigma * static_cast<float>(rnd.get_random_gaussian());
            for (size_t i = 4 * weight_count(); i < params.size(); ++i) p[i] = 0.0f;

            w_alias = alias_tensor(WIDTH, WIDTH);
            b_alias = alias_tensor(1, WIDTH);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& x = sub.get_output();
            const long BN = x.num_samples();
            DLIB_CASSERT(BN % NUM_PATCHES == 0,
                "vision_attention received " << BN << " samples, which is not a whole "
                "number of images of " << NUM_PATCHES << " patches");
            const long B = BN / NUM_PATCHES, N = NUM_PATCHES;

            auto wq = w_alias(params, wq_offset());
            auto wk = w_alias(params, wk_offset());
            auto wv = w_alias(params, wv_offset());
            auto wo = w_alias(params, wo_offset());
            auto bq = b_alias(params, bq_offset());
            auto bk = b_alias(params, bk_offset());
            auto bv = b_alias(params, bv_offset());
            auto bo = b_alias(params, bo_offset());

            /* The projections read [patches, WIDTH] as a matrix directly: in the default
               gemm mode a tensor is its num_samples by k*nr*nc matrix, which is what this
               layout already is. The biases broadcast over samples. */
            project(x, wq, bq, q_, BN);
            project(x, wk, bk, k_, BN);
            project(x, wv, bv, v_, BN);

            alias_tensor as_seq(B, 1, N, WIDTH);
            q4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            k4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            v4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            tt::split_heads(false, q4_, as_seq(q_, 0));
            tt::split_heads(false, k4_, as_seq(k_, 0));
            tt::split_heads(false, v4_, as_seq(v_, 0));

            const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
            scores_.set_size(B, NUM_HEADS, N, N);
            tt::gemm(0.0f, scores_, scale, q4_, false, k4_, true, operation_mode::PLANE_WISE);
            attn_.copy_size(scores_);
            tt::softmax(attn_, scores_, operation_mode::PLANE_WISE);

            ctx4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, ctx4_, 1.0f, attn_, false, v4_, false, operation_mode::PLANE_WISE);

            ctx_.set_size(BN, WIDTH, 1, 1);
            auto ctx_view = as_seq(ctx_, 0);
            tt::merge_heads(false, ctx_view, ctx4_);

            output.set_size(BN, WIDTH, 1, 1);
            tt::gemm(0.0f, output, 1.0f, ctx_, false, wo, false);
            tt::add(1.0f, output, 1.0f, bo);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            const tensor& x = sub.get_output();
            const long BN = x.num_samples();
            const long B = BN / NUM_PATCHES, N = NUM_PATCHES;

            auto wq = w_alias(params, wq_offset());
            auto wk = w_alias(params, wk_offset());
            auto wv = w_alias(params, wv_offset());
            auto wo = w_alias(params, wo_offset());

            params_grad = 0;
            auto dwq = w_alias(params_grad, wq_offset());
            auto dwk = w_alias(params_grad, wk_offset());
            auto dwv = w_alias(params_grad, wv_offset());
            auto dwo = w_alias(params_grad, wo_offset());
            auto dbq = b_alias(params_grad, bq_offset());
            auto dbk = b_alias(params_grad, bk_offset());
            auto dbv = b_alias(params_grad, bv_offset());
            auto dbo = b_alias(params_grad, bo_offset());

            // Output projection.
            tt::gemm(0.0f, dwo, 1.0f, ctx_, true, gradient_input, false);
            tt::assign_conv_bias_gradient(dbo, gradient_input);
            dctx_.set_size(BN, WIDTH, 1, 1);
            tt::gemm(0.0f, dctx_, 1.0f, gradient_input, false, wo, true);

            alias_tensor as_seq(B, 1, N, WIDTH);
            dctx4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            tt::split_heads(false, dctx4_, as_seq(dctx_, 0));

            // Context product.
            dattn_.set_size(B, NUM_HEADS, N, N);
            dv4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, dattn_, 1.0f, dctx4_, false, v4_, true, operation_mode::PLANE_WISE);
            tt::gemm(0.0f, dv4_, 1.0f, attn_, true, dctx4_, false, operation_mode::PLANE_WISE);

            /* Softmax and the scaled score product. tt::softmax_gradient adds into its
               destination whenever that destination is a different object from its input,
               so the buffer has to be cleared first: left dirty, it would carry the
               previous backward pass and double every Q and K gradient. */
            dscores_.copy_size(dattn_);
            dscores_ = 0;
            tt::softmax_gradient(dscores_, attn_, dattn_, operation_mode::PLANE_WISE);

            const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
            dq4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            dk4_.set_size(B, NUM_HEADS, N, HEAD_DIM);
            tt::gemm(0.0f, dq4_, scale, dscores_, false, k4_, false, operation_mode::PLANE_WISE);
            tt::gemm(0.0f, dk4_, scale, dscores_, true, q4_, false, operation_mode::PLANE_WISE);

            dq_.set_size(BN, WIDTH, 1, 1);
            dk_.set_size(BN, WIDTH, 1, 1);
            dv_.set_size(BN, WIDTH, 1, 1);
            auto dq_view = as_seq(dq_, 0);
            auto dk_view = as_seq(dk_, 0);
            auto dv_view = as_seq(dv_, 0);
            tt::merge_heads(false, dq_view, dq4_);
            tt::merge_heads(false, dk_view, dk4_);
            tt::merge_heads(false, dv_view, dv4_);

            /* Input projections. The three contributions accumulate into the data gradient,
               which is why every gemm below carries a beta of one. */
            tensor& dx = sub.get_gradient_input();
            tt::gemm(0.0f, dwq, 1.0f, x, true, dq_, false);
            tt::assign_conv_bias_gradient(dbq, dq_);
            tt::gemm(1.0f, dx, 1.0f, dq_, false, wq, true);

            tt::gemm(0.0f, dwk, 1.0f, x, true, dk_, false);
            tt::assign_conv_bias_gradient(dbk, dk_);
            tt::gemm(1.0f, dx, 1.0f, dk_, false, wk, true);

            tt::gemm(0.0f, dwv, 1.0f, x, true, dv_, false);
            tt::assign_conv_bias_gradient(dbv, dv_);
            tt::gemm(1.0f, dx, 1.0f, dv_, false, wv, true);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const vision_attention_& item, std::ostream& out)
        {
            serialize("vision_attention_", out);
            serialize(item.params, out);
            serialize(item.learning_rate_multiplier_, out);
            serialize(item.weight_decay_multiplier_, out);
        }
        friend void deserialize(vision_attention_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "vision_attention_")
                throw serialization_error("Unexpected version found while deserializing dlib::vision_attention_.");
            deserialize(item.params, in);
            deserialize(item.learning_rate_multiplier_, in);
            deserialize(item.weight_decay_multiplier_, in);
            item.w_alias = alias_tensor(WIDTH, WIDTH);
            item.b_alias = alias_tensor(1, WIDTH);
        }
        friend std::ostream& operator<<(std::ostream& out, const vision_attention_& item)
        {
            out << "vision_attention (width=" << WIDTH << ", heads=" << NUM_HEADS
                << ", patches=" << NUM_PATCHES << ") learning_rate_mult="
                << item.learning_rate_multiplier_;
            return out;
        }
        friend void to_xml(const vision_attention_& item, std::ostream& out)
        {
            out << "<vision_attention width='" << WIDTH << "' heads='" << NUM_HEADS
                << "' patches='" << NUM_PATCHES << "'>\n"
                << mat(item.params) << "</vision_attention>\n";
        }

    private:

        void project(const tensor& x, const alias_tensor_instance& w,
            const alias_tensor_instance& b, resizable_tensor& y, long rows)
        {
            y.set_size(rows, WIDTH, 1, 1);
            tt::gemm(0.0f, y, 1.0f, x, false, w, false);
            tt::add(1.0f, y, 1.0f, b);
        }

        resizable_tensor params;
        alias_tensor w_alias, b_alias;
        double learning_rate_multiplier_;
        double weight_decay_multiplier_;

        resizable_tensor q_, k_, v_, q4_, k4_, v4_, scores_, attn_, ctx4_, ctx_;
        resizable_tensor dctx_, dctx4_, dattn_, dscores_, dq4_, dk4_, dv4_, dq_, dk_, dv_;
    };

    template <long WIDTH, long NUM_HEADS, long NUM_PATCHES, typename SUBNET>
    using vision_attention = add_layer<vision_attention_<WIDTH, NUM_HEADS, NUM_PATCHES>, SUBNET>;

// ----------------------------------------------------------------------------------------

    namespace vision_transformer
    {
        /* Pre-norm block: LayerNorm then bidirectional attention with a residual, LayerNorm
           then a GELU feed-forward with a residual. The topology is that of the decoder's
           block; what differs is the normalization, which subtracts a mean and carries a
           bias here, the absence of any mask, and the feed-forward, which is a plain
           expansion rather than a gated one. */
        template <long WIDTH, long NUM_HEADS, long NUM_PATCHES, long FFN_HIDDEN, typename SUBNET>
        using vision_block =
            add_prev5<fc<WIDTH, gelu<fc<FFN_HIDDEN, layer_norm<tag5<
            add_prev1<vision_attention<WIDTH, NUM_HEADS, NUM_PATCHES, layer_norm<tag1<
            SUBNET>>>>>>>>>>;

        template <long REMAINING, long WIDTH, long NUM_HEADS, long NUM_PATCHES,
            long FFN_HIDDEN, typename SUBNET, typename enabled = void>
        struct vision_stack_impl
        {
            using type = vision_block<WIDTH, NUM_HEADS, NUM_PATCHES, FFN_HIDDEN,
                typename vision_stack_impl<REMAINING - 1, WIDTH, NUM_HEADS, NUM_PATCHES,
                    FFN_HIDDEN, SUBNET>::type>;
        };

        template <long WIDTH, long NUM_HEADS, long NUM_PATCHES, long FFN_HIDDEN, typename SUBNET>
        struct vision_stack_impl<0, WIDTH, NUM_HEADS, NUM_PATCHES, FFN_HIDDEN, SUBNET, void>
        {
            using type = SUBNET;
        };

        template <long NUM_LAYERS, long WIDTH, long NUM_HEADS, long NUM_PATCHES,
            long FFN_HIDDEN, typename SUBNET>
        using vision_stack = typename vision_stack_impl<NUM_LAYERS, WIDTH, NUM_HEADS,
            NUM_PATCHES, FFN_HIDDEN, SUBNET>::type;

    } // namespace vision_transformer

// ----------------------------------------------------------------------------------------

    /* Whole tower, from a normalized image to the vectors the decoder receives.

       The patch embedding is a convolution whose kernel equals its stride, which is what
       makes the grid exactly the patches; the containers this reads store its filters in
       the layout con expects, so nothing is repacked on the way in. */
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

        static_assert(image_size % patch_size == 0,
            "the image size must be a whole number of patches");
        static_assert(GRID_SIDE % shuffle_factor == 0,
            "the patch grid must be divisible by the shuffle factor");

        using stack = vision_transformer::vision_stack<NUM_LAYERS, WIDTH, NUM_HEADS,
            NUM_PATCHES, FFN_HIDDEN,
            patch_positions<NUM_PATCHES, WIDTH,
            patch_sequence<
            con<WIDTH, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
            tag10<input_tensor>>>>>;

        /* The projector carries a bias in some containers and not in others. The slot is
           always present and left at zero when the file has none, which costs a vector of
           floats and spares a second network type. */
        using network_type = fc<PROJECTION_DIM,
            patch_shuffle<SHUFFLE_FACTOR, GRID_SIDE,
            layer_norm<stack>>>;

        struct model_info
        {
            static std::string describe()
            {
                std::ostringstream o;
                o << "vision_transformer_config\n"
                  << "  image        : " << IMAGE_SIZE << " (patches of " << PATCH_SIZE
                  << ", grid " << GRID_SIDE << "x" << GRID_SIDE << ")\n"
                  << "  width        : " << WIDTH << " (head dim "
                  << (NUM_HEADS ? WIDTH / NUM_HEADS : 0) << ")\n"
                  << "  layers       : " << NUM_LAYERS << "\n"
                  << "  heads        : " << NUM_HEADS << "\n"
                  << "  ffn_hidden   : " << FFN_HIDDEN << "\n"
                  << "  shuffle      : " << SHUFFLE_FACTOR << " (" << NUM_PATCHES
                  << " patches -> " << NUM_TOKENS << " tokens of " << FOLDED_WIDTH << ")\n"
                  << "  projection   : " << PROJECTION_DIM;
                return o.str();
            }
        };
    };
}

#endif // DLIB_DNN_VISION_TRANSFORMER_H_
