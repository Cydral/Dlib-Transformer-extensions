// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Shape-dynamic vision encoder for multimodal GGUF models.
//
// Counterpart of runtime_transformer for the second file a multimodal model ships. It
// compiles once and serves any vision tower whose geometry gguf_vision_spec.h accepts,
// which matters more here than on the decoder side: a factory ingesting arbitrary
// open-weight models meets a new patch size or a new grid far more often than a new
// decoder shape, and none of them should cost a recompilation.
//
// The pipeline it runs, for an image already resized and normalized:
//
//     conv 16x16 stride 16   -> a grid of patch vectors
//     + learned positions
//     12 pre-norm blocks     -> LayerNorm, bidirectional attention, LayerNorm, GELU
//     LayerNorm
//     pixel shuffle          -> fewer positions, more channels
//     one linear layer       -> the decoder's embedding width
//
// Three things about it are not obvious from the description.
//
// The attention carries no mask. A vision tower reads an image, where no position comes
// before another, so every patch attends to every patch. That is the whole reason this
// encoder is written against tensor operations rather than against the decoder's attention
// layer, which is causal by construction.
//
// The patch embedding is a convolution whose kernel equals its stride, and the container
// stores its filters in exactly the layout tensor_conv expects. There is nothing to repack
// and nothing to write: the weight goes in as it comes out of the file.
//
// The feed-forward tensor names are inverted relative to the decoder's convention. In this
// container ffn_down carries the expansion and ffn_up the contraction, which their bias
// sizes confirm without ambiguity. Reading them the other way round produces an encoder
// that runs, returns plausible numbers, and describes the wrong image.

#ifndef DLIB_DNN_RUNTIME_VISION_ENCODER_H_
#define DLIB_DNN_RUNTIME_VISION_ENCODER_H_

#include "runtime_vision_encoder_abstract.h"

#include <cmath>
#include <string>
#include <vector>

#include "../dnn.h"
#include "../data_io/gguf_vision_spec.h"
#include "../data_io/gguf_dequantize.h"

namespace dlib
{
    /* Pixels prepared for a vision tower: resized to the square the encoder expects and
       normalized with the statistics the container declares. This depends on the geometry
       alone and not on a single weight, so it is a free function rather than a member: the
       shape-dynamic encoder and a compiled tower must see exactly the same pixels for the
       same file, and the surest way to guarantee that is for them to run one piece of
       code. */
    template <typename image_type>
    void prepare_vision_image(const image_type& img, const vision_spec& spec,
        resizable_tensor& out)
    {
        DLIB_CASSERT(spec.image_size > 0, "the vision spec has no geometry");
        const long side = spec.image_size;

        /* The source goes in as the image object it is. Wrapping it in mat() would
           make it a matrix expression, which generic_image does not accept: an image
           is recognized by its traits, not by being indexable. */
        matrix<rgb_pixel> resized(side, side);
        resize_image(img, resized, interpolate_bilinear());

        const float mr = spec.image_mean.size() == 3 ? spec.image_mean[0] : 0.5f;
        const float mg = spec.image_mean.size() == 3 ? spec.image_mean[1] : 0.5f;
        const float mb = spec.image_mean.size() == 3 ? spec.image_mean[2] : 0.5f;
        const float sr = spec.image_std.size() == 3 ? spec.image_std[0] : 0.5f;
        const float sg = spec.image_std.size() == 3 ? spec.image_std[1] : 0.5f;
        const float sb = spec.image_std.size() == 3 ? spec.image_std[2] : 0.5f;

        out.set_size(1, 3, side, side);
        float* p = out.host_write_only();
        const long plane = side * side;
        for (long r = 0; r < side; ++r)
            for (long c = 0; c < side; ++c)
            {
                const rgb_pixel& px = resized(r, c);
                const long i = r * side + c;
                p[0 * plane + i] = (px.red   / 255.0f - mr) / sr;
                p[1 * plane + i] = (px.green / 255.0f - mg) / sg;
                p[2 * plane + i] = (px.blue  / 255.0f - mb) / sb;
            }
        }

    class runtime_vision_encoder
    {
    public:

        const vision_spec& spec() const { return spec_; }
        bool loaded() const { return spec_.n_layers > 0 && !layers_.empty(); }

        /* Reads every weight of the tower into resident float32. The container is small,
           a hundred megabytes or so, and unlike the decoder there is no generation loop
           whose memory it would compete with, so the quantized-at-rest machinery of
           runtime_transformer would buy nothing here. */
        void load(gguf_reader& g, const vision_spec& s)
        {
            spec_ = s;
            const long d = s.d_model, ff = s.d_ffn;

            load_raw(g, "v.patch_embd.weight", patch_w_);
            patch_w_.set_size(d, 3, s.patch_size, s.patch_size);
            load_into(g, "v.patch_embd.weight", patch_w_);
            load_vector(g, "v.patch_embd.bias", patch_b_, d);

            /* The position table is one vector per patch, so it is already in the layout
               the sequence uses and needs no transposition. */
            pos_.set_size(s.num_patches(), d);
            load_into(g, "v.position_embd.weight", pos_);

            load_vector(g, "v.post_ln.weight", post_ln_w_, d);
            load_vector(g, "v.post_ln.bias", post_ln_b_, d);

            layers_.clear();
            layers_.resize(static_cast<size_t>(s.n_layers));
            for (long i = 0; i < s.n_layers; ++i)
            {
                vision_layer& L = layers_[static_cast<size_t>(i)];
                const std::string p = "v.blk." + std::to_string(i) + ".";

                load_vector(g, p + "ln1.weight", L.ln1_w, d);
                load_vector(g, p + "ln1.bias", L.ln1_b, d);
                load_vector(g, p + "ln2.weight", L.ln2_w, d);
                load_vector(g, p + "ln2.bias", L.ln2_b, d);

                load_linear(g, p + "attn_q.weight", d, d, L.wq);
                load_linear(g, p + "attn_k.weight", d, d, L.wk);
                load_linear(g, p + "attn_v.weight", d, d, L.wv);
                load_linear(g, p + "attn_out.weight", d, d, L.wo);
                load_vector(g, p + "attn_q.bias", L.bq, d);
                load_vector(g, p + "attn_k.bias", L.bk, d);
                load_vector(g, p + "attn_v.bias", L.bv, d);
                load_vector(g, p + "attn_out.bias", L.bo, d);

                /* Named backwards in this container: ffn_down expands and ffn_up
                   contracts, as their bias sizes show. */
                load_linear(g, p + "ffn_down.weight", d, ff, L.w_expand);
                load_vector(g, p + "ffn_down.bias", L.b_expand, ff);
                load_linear(g, p + "ffn_up.weight", ff, d, L.w_contract);
                load_vector(g, p + "ffn_up.bias", L.b_contract, d);
            }

            load_linear(g, "mm.model.fc.weight", s.folded_width(), s.projection_dim, fc_);
            has_fc_bias_ = g.find_tensor("mm.model.fc.bias") != nullptr;
            if (has_fc_bias_) load_vector(g, "mm.model.fc.bias", fc_b_, s.projection_dim);
        }

        /* Resizes an image to the square the tower expects and normalizes it into the
           layout the convolution reads: one sample, three planes, height by width.

           The normalization constants come from the container rather than from a table of
           known families, because they are part of the encoder: a tower trained on data
           centered at 0.5 and one centered on the ImageNet statistics see different
           pictures for the same file. */
        template <typename image_type>
        void prepare_image(const image_type& img, resizable_tensor& out) const
        {
            prepare_vision_image(img, spec_, out);
        }

        /* Visual embeddings of one prepared image, as a [tokens, projection_dim] matrix
           ready to be written over the placeholder positions of a token stream. The result
           lives in the encoder and stays valid until the next call. */
        const tensor& encode(const tensor& image)
        {
            DLIB_CASSERT(loaded(), "the encoder holds no weights");
            DLIB_CASSERT(image.num_samples() == 1 && image.k() == 3
                && image.nr() == spec_.image_size && image.nc() == spec_.image_size,
                "the image does not match the geometry the tower expects");

            const long d = spec_.d_model, side = spec_.grid_side();
            const long N = spec_.num_patches(), H = spec_.n_heads, hd = spec_.head_dim();

            // Patch embedding. Kernel equals stride, so the grid is exactly the patches.
            conv_.setup(image, patch_w_, spec_.patch_size, spec_.patch_size, 0, 0);
            conv_(false, grid_, image, patch_w_);
            tt::add(1.0f, grid_, 1.0f, patch_b_);

            /* The convolution lays the grid out plane by plane; the transformer reads it
               position by position. One transpose of the [d, N] matrix turns one into the
               other, and the same operation runs backwards before the pixel shuffle. */
            to_sequence(grid_, seq_, d, N);
            tt::add(1.0f, seq_, 1.0f, pos_);

            for (const vision_layer& L : layers_)
            {
                tt::layer_normalize(spec_.layer_norm_eps, normed_, means_, invstds_,
                    seq_, L.ln1_w, L.ln1_b);

                project(normed_, L.wq, L.bq, q_, N, d);
                project(normed_, L.wk, L.bk, k_, N, d);
                project(normed_, L.wv, L.bv, v_, N, d);

                /* split_heads reads a [batch, 1, positions, features] tensor, not the
                   [positions, features] matrix the projections produce. The two are views
                   of the same storage, so the reinterpretation costs nothing, but they are
                   not interchangeable: a matrix view carries its features in k, where the
                   head split expects them in nc. */
                q4_.set_size(1, H, N, hd); tt::split_heads(false, q4_, as_sequence(q_, N, d));
                k4_.set_size(1, H, N, hd); tt::split_heads(false, k4_, as_sequence(k_, N, d));
                v4_.set_size(1, H, N, hd); tt::split_heads(false, v4_, as_sequence(v_, N, d));

                /* No mask. Every patch sees every patch, which is the one place this
                   encoder parts ways with the decoder's attention. */
                const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
                scores_.set_size(1, H, N, N);
                tt::gemm(0.0f, scores_, scale, q4_, false, k4_, true,
                    operation_mode::PLANE_WISE);
                attn_.copy_size(scores_);
                tt::softmax(attn_, scores_, operation_mode::PLANE_WISE);

                ctx4_.set_size(1, H, N, hd);
                tt::gemm(0.0f, ctx4_, 1.0f, attn_, false, v4_, false,
                    operation_mode::PLANE_WISE);
                ctx_.set_size(1, 1, N, d);
                tt::merge_heads(false, ctx_, ctx4_);

                project(reshape(ctx_, N, d), L.wo, L.bo, attn_out_, N, d);
                tt::add(1.0f, seq_, 1.0f, attn_out_);

                tt::layer_normalize(spec_.layer_norm_eps, normed_, means_, invstds_,
                    seq_, L.ln2_w, L.ln2_b);
                project(normed_, L.w_expand, L.b_expand, hidden_, N, spec_.d_ffn);
                if (spec_.use_gelu)
                {
                    activated_.copy_size(hidden_);
                    tt::gelu(activated_, hidden_);
                    project(activated_, L.w_contract, L.b_contract, ffn_out_, N, d);
                }
                else
                {
                    project(hidden_, L.w_contract, L.b_contract, ffn_out_, N, d);
                }
                tt::add(1.0f, seq_, 1.0f, ffn_out_);
            }

            tt::layer_normalize(spec_.layer_norm_eps, normed_, means_, invstds_,
                seq_, post_ln_w_, post_ln_b_);

            /* Pixel shuffle. The sequence goes back to a grid, the reduction folds the
               spatial neighbourhood into the channels, and the result becomes a sequence
               again, shorter and wider. */
            const long f = spec_.scale_factor;
            if (f > 1)
            {
                to_grid(normed_, grid_, d, side);
                folded_.set_size(1, spec_.folded_width(), side / f, side / f);
                tt::reorg(false, folded_, static_cast<int>(f), static_cast<int>(f), grid_);
                to_sequence(folded_, reduced_, spec_.folded_width(), spec_.tokens_per_image());
            }
            else
            {
                reduced_.copy_size(normed_);
                memcpy(reduced_, normed_);
            }

            out_.set_size(spec_.tokens_per_image(), spec_.projection_dim);
            tt::gemm(0.0f, out_, 1.0f, reshape(reduced_, spec_.tokens_per_image(),
                spec_.folded_width()), false, fc_, false);
            if (has_fc_bias_) tt::add(1.0f, out_, 1.0f, fc_b_);
            return out_;
        }

    private:

        struct vision_layer
        {
            resizable_tensor ln1_w, ln1_b, ln2_w, ln2_b;
            resizable_tensor wq, wk, wv, wo, bq, bk, bv, bo;
            resizable_tensor w_expand, b_expand, w_contract, b_contract;
        };

        // y = x . W + b, on a [rows, in] matrix view.
        void project(const tensor& x, const tensor& w, const tensor& b,
            resizable_tensor& y, long rows, long out_dim)
        {
            y.set_size(rows, out_dim);
            alias_tensor xm(rows, static_cast<long>(x.size() / rows));
            auto xv = xm(const_cast<tensor&>(x), 0);
            tt::gemm(0.0f, y, 1.0f, xv, false, w, false);
            tt::add(1.0f, y, 1.0f, b);
        }

        /* Two views of the same buffer, which the operations here do not agree on. A gemm
           wants a matrix, that is rows in num_samples and columns in k; the head split and
           merge want a sequence, that is positions in nr and features in nc. Naming them
           apart is the cheapest way to stop confusing the two. */
        alias_tensor_instance reshape(tensor& t, long rows, long cols) const
        {
            DLIB_CASSERT(t.size() == static_cast<size_t>(rows) * cols);
            return alias_tensor(rows, cols)(t, 0);
        }
        alias_tensor_instance reshape(resizable_tensor& t, long rows, long cols) const
        {
            return reshape(static_cast<tensor&>(t), rows, cols);
        }

        alias_tensor_instance as_sequence(tensor& t, long positions, long features) const
        {
            DLIB_CASSERT(t.size() == static_cast<size_t>(positions) * features);
            return alias_tensor(1, 1, positions, features)(t, 0);
        }
        alias_tensor_instance as_sequence(resizable_tensor& t, long positions, long features) const
        {
            return as_sequence(static_cast<tensor&>(t), positions, features);
        }

        /* Grid to sequence and back. The grid holds one plane per channel, the sequence
           one row per position, so the two are transposes of the same [channels,
           positions] matrix. */
        void to_sequence(const tensor& grid, resizable_tensor& seq, long channels, long n)
        {
            seq.set_size(n, channels);
            alias_tensor as_matrix(1, 1, channels, n);
            auto gm = as_matrix(const_cast<tensor&>(grid), 0);
            alias_tensor as_seq(1, 1, n, channels);
            auto sm = as_seq(seq, 0);
            tt::transpose(false, sm, gm);
        }

        void to_grid(const tensor& seq, resizable_tensor& grid, long channels, long side)
        {
            const long n = side * side;
            grid.set_size(1, channels, side, side);
            alias_tensor as_seq(1, 1, n, channels);
            auto sm = as_seq(const_cast<tensor&>(seq), 0);
            alias_tensor as_matrix(1, 1, channels, n);
            auto gm = as_matrix(grid, 0);
            tt::transpose(false, gm, sm);
        }

        void load_raw(gguf_reader& g, const std::string& name, resizable_tensor&) const
        {
            if (!g.find_tensor(name))
                throw std::runtime_error("runtime_vision_encoder: missing tensor " + name);
        }

        // Straight copy, for tensors whose file layout is the one the engine wants.
        void load_into(gguf_reader& g, const std::string& name, resizable_tensor& dst)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_vision_encoder: missing tensor " + name);
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            if (src.size() != dst.size())
                throw std::runtime_error("runtime_vision_encoder: shape mismatch for " + name);
            std::memcpy(dst.host_write_only(), src.data(), src.size() * sizeof(float));
        }

        void load_vector(gguf_reader& g, const std::string& name, resizable_tensor& dst, long n)
        {
            dst.set_size(1, n);
            load_into(g, name, dst);
        }

        /* Linear weights are stored as out rows of in values, and the engine multiplies
           x[rows, in] by W[in, out], so the load transposes. Same convention as the
           decoder, and the reason every matrix here is declared by its true in and out
           rather than by the dimensions the file lists. */
        void load_linear(gguf_reader& g, const std::string& name, long in_dim, long out_dim,
            resizable_tensor& dst)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_vision_encoder: missing tensor " + name);
            if (t->n_elements() != static_cast<uint64_t>(in_dim) * out_dim)
                throw std::runtime_error("runtime_vision_encoder: shape mismatch for " + name);
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            dst.set_size(in_dim, out_dim);
            float* d = dst.host_write_only();
            for (long o = 0; o < out_dim; ++o)
                for (long i = 0; i < in_dim; ++i)
                    d[i * out_dim + o] = src[static_cast<size_t>(o) * in_dim + i];
        }

        vision_spec spec_;
        std::vector<vision_layer> layers_;

        resizable_tensor patch_w_, patch_b_, pos_, post_ln_w_, post_ln_b_, fc_, fc_b_;
        bool has_fc_bias_ = false;

        tt::tensor_conv conv_;
        resizable_tensor grid_, seq_, normed_, means_, invstds_;
        resizable_tensor q_, k_, v_, q4_, k4_, v4_, scores_, attn_, ctx4_, ctx_, attn_out_;
        resizable_tensor hidden_, activated_, ffn_out_, folded_, reduced_, out_;
    };
}

#endif // DLIB_DNN_RUNTIME_VISION_ENCODER_H_
