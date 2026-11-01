// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_VISION_ENCODER_H_
#define ALBIREO_ENGINE_VISION_ENCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "image_input.h"
#include "model_file.h"
#include "offload.h"
#include "qkernels.h"
#include "thread_pool.h"

namespace albireo
{

    class vision_encoder
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The image encoder of a model that reads images, read from a projector GGUF
                file (architecture "clip"), or from the vision tensors and the clip metadata
                of the model file itself.  Two families are read.

                Qwen3-VL (clip.projector_type "qwen3vl_merger"): a vision transformer whose
                patches are merged by squares of merge_size() x merge_size() into the input
                tokens of the language model.  The image, resized to multiples of factor()
                pixels, is cut into patches of patch_size() pixels, taken square by square.
                The patch embedding is the convolution of the reference with a still image,
                which repeats the image in time: the sum of its two temporal kernels.  The
                learned position table is interpolated bilinearly, corners aligned, to the
                grid of the patches, and each head of the attention rotates the first half
                of its rotary pairs by the row of the patch and the second half by its
                column.  After the blocks that the file names, a DeepStack merger normalizes
                each square of patches, then applies two linear layers around an exact GELU;
                the final merger normalizes each patch, merges the squares and applies the
                same layers.  The output of a merged token is the output of the final merger
                followed by those of the DeepStack mergers, output_width() floats, which a
                Qwen3-VL decoder reads as its input embedding and the slices it adds after
                its first layers.

                LLaVA-NeXT (clip.projector_type "mlp" with clip.vision.image_grid_pinpoints,
                EuroVLM): a SigLIP vision transformer reads views of image_size() pixels
                square, each one a token per patch.  The views are the whole image resized
                to a square, then the tiles of the image resized, its aspect ratio kept, into
                the resolution of the pinpoints that loses the least of it, black bands
                filling the rest.  Each view goes through the blocks up to the feature layer
                of the file, then through the projector, two linear layers around an exact
                GELU.  The tokens of the image are those of the whole image, then the grid of
                the tiles, row by row, without the rows or the columns of the black bands,
                each row followed by the newline embedding of the file, as the reference
                implementation packs them.

                Pixtral (clip.projector_type "pixtral", Ministral 3, Devstral Small 2 and the
                other Mistral models that read images): the image, resized within the longest
                edge of the file (image_size()), its aspect ratio kept, to multiples of
                factor() pixels, is cut into patches of patch_size() pixels.  Each head of
                the attention rotates its pairs of consecutive dimensions, the first half of
                them by the row of the patch and the second half by its column, at the even
                and the odd frequencies of the head.  The blocks normalize by their root mean
                square and their feed-forward layer is a SiLU gate.  The projector normalizes
                each patch, merges the squares of merge_size() x merge_size() patches through
                a linear layer, then applies two linear layers around an exact GELU.  The
                tokens of the image are its merged tokens row by row, each row but the last
                followed by the [IMG_BREAK] embedding of the file, which a Pixtral encoder
                must hold; the decoder reads the token [IMG_END] after them.

                The blocks of Qwen3-VL and LLaVA-NeXT normalize before the attention and
                before the feed-forward layer, with a mean and a bias; the attention reads
                every patch of the view; the feed-forward layer activates with the tanh form
                of GELU.  Matrices stored in bfloat16 are converted to half precision when
                the encoder loads, which keeps their values: the half format holds more
                mantissa bits.

                The products go through the quantized kernels of qkernels.h on the threads of
                the engine (Q8_0, F16 and F32 weights, and the other types the kernels read),
                and the attention runs by tiles of queries against blocks of keys
                (qk_encoder_attention).  With a matmul_offload (set_offload()), the products
                go through it, which computes on its device the shapes the device runs
                faster; the offload counts as busy from start() to release_work() only, so
                that its copies of the weights may leave the device for the model between two
                images.

                An encoding runs a step at a time (advance()): the scheduler of the engine
                interleaves the blocks of an image with the passes of the other requests.

            THREAD SAFETY
                One thread drives an encoder: its work buffers serve the job it advances.
                The constructor runs on the loading thread alone and opens no section of
                the pool, whose threads may be computing the passes of other models.
                plan(), most_tokens() and the other const functions may be called from any
                thread.
        !*/
    public:
        enum class family { qwen3vl, llava_next, pixtral };

        vision_encoder(std::shared_ptr<model_file> file, thread_pool& pool);

        // Whether a file holds the tensors of an image encoder.
        static bool present(const model_file& f);

        family kind() const { return kind_; }
        long patch_size() const { return patch_; }
        long merge_size() const { return merge_; }
        // Pixels per side of a merged token, the factor the sides of an image are multiples of
        // (Qwen3-VL, Pixtral).
        long factor() const { return patch_ * merge_; }
        // Pixels per side of a view (LLaVA-NeXT), the longest edge of an image (Pixtral).
        long image_size() const { return image_size_; }
        long output_dim() const { return out_dim_; }
        long deepstack_count() const { return static_cast<long>(deepstack_.size()); }
        long output_width() const { return out_dim_ * (1 + deepstack_count()); }
        size_t weight_bytes() const { return weight_bytes_; }
        std::string description() const;

        struct image_plan
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    How the encoder reads an image, from its size alone: the size its pixels
                    are decoded at, the tokens it takes in the prompt and, for Qwen3-VL, the
                    grid of these tokens, which gives their rotary positions; for LLaVA-NeXT,
                    the resolution of its tiles.
            !*/
            long width = 0, height = 0;
            long tokens = 0;
            long grid_w = 0, grid_h = 0;
            long tiles_w = 0, tiles_h = 0;      // pixels the tiles cover (LLaVA-NeXT)
        };

        /*!
            ensures
                - Returns how an image of width x height pixels is read with the detail
                  "low", "auto" (also "") or "high": for Qwen3-VL, the size resized to
                  multiples of factor() within 64 and 256, 576 or 1024 merged tokens; for
                  LLaVA-NeXT, the pinpoints of 1, 2 or 4 tiles at most; for Pixtral, the
                  size within the longest edge, rounded up to multiples of factor(), then
                  reduced, when it holds more, to 256, 576 or 1024 merged tokens.
                - Throws std::invalid_argument for another detail or an unusable size.
        !*/
        image_plan plan(long width, long height, const std::string& detail) const;

        // The most tokens an image takes with a detail, "min" giving the fewest an image can take.
        long most_tokens(const std::string& detail) const;

        class job
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    The encoding of one image in progress: its patches between the blocks,
                    then its output.
            !*/
        public:
            long tokens() const { return tokens_; }
            long grid_width() const { return grid_w_; }     // merged tokens per row
            long grid_height() const { return grid_h_; }
            bool done() const { return done_; }
            // tokens() rows of output_width() floats, once done.
            const std::vector<float>& output() const { return out_; }
            std::vector<float>& output() { return out_; }
            double seconds() const { return seconds_; }

        private:
            friend class vision_encoder;
            long pw_ = 0, ph_ = 0, n_ = 0;      // patches per row and column, patches
            long grid_w_ = 0, grid_h_ = 0, tokens_ = 0;
            long next_ = 0;                     // the next block; the final merger after the last one
            bool done_ = false;
            aligned_floats x_;                  // n_ rows of the width of the encoder
            std::vector<float> cos_, sin_;      // n_ rows of head_dim / 2 values
            std::vector<float> out_;
            double seconds_ = 0;
            // LLaVA-NeXT: the views, the one being encoded, the features of the views and the
            // part of the grid of the tiles kept (rows and columns).
            std::vector<rgb_image> views_;
            size_t view_ = 0;
            long tile_rows_ = 0, tile_cols_ = 0;
            long keep_y0_ = 0, keep_y1_ = 0, keep_x0_ = 0, keep_x1_ = 0;
            std::vector<float> features_;
        };

        /*!
            requires
                - For Qwen3-VL and Pixtral, image.width and image.height are positive
                  multiples of factor(); for LLaVA-NeXT, image is the image plan() was given
                  the size of, decoded at plan.width x plan.height.
            ensures
                - Returns an encoding of the image with its first patch embedding computed.
        !*/
        std::unique_ptr<job> start(const rgb_image& image, const image_plan& plan);

        /*!
            ensures
                - Runs the next steps of j, a block each (with its DeepStack merger) and the
                  final merger last, until seconds have passed or the encoding is complete;
                  at least one step runs.
                - Returns j.done().
        !*/
        bool advance(job& j, double seconds);

        // Encodes an image at once: out receives its tokens, output_width() floats each.
        void encode(const rgb_image& image, const image_plan& plan, std::vector<float>& out);

        // Returns the memory of the work buffers to the system.
        void release_work();

        // Hands the products of the following encodings to a device.
        void set_offload(std::unique_ptr<matmul_offload> offload) { offload_ = std::move(offload); }

    private:
        struct block
        {
            // The biases are empty when the file has none (Pixtral).
            std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, out_b, up_b, down_b;
            weight_matrix qkv, q, k, v, out, gate, up, down;
            bool fused = true;          // queries, keys and values in one matrix
            bool gated = false;         // a SiLU gate in the feed-forward layer (Pixtral)
        };

        struct merger
        {
            std::vector<float> norm_w, norm_b, fc1_b, fc2_b;
            weight_matrix fc1, fc2;
            long after = -1;            // the block it reads the output of (DeepStack)
        };

        // The layout of the tokens of an image read by LLaVA-NeXT.
        struct tile_layout
        {
            long res_w = 0, res_h = 0;  // the pinpoint, in pixels
            long rows = 0, cols = 0;    // tiles
            long y0 = 0, y1 = 0, x0 = 0, x1 = 0;
            long tokens = 0;
        };
        tile_layout layout(long width, long height, long res_w, long res_h) const;
        tile_layout best_layout(long width, long height, long max_tiles) const;

        void embed_patches(job& j, const rgb_image& image);
        // A layer norm with the mean and the bias of b, or a root mean square norm when b is empty.
        void layer_norm_rows(const float* x, const std::vector<float>& w, const std::vector<float>& b, float* y,
            long rows, long width);
        void add_bias_rows(float* y, long stride, const std::vector<float>& bias, long rows);
        void run_block(job& j, const block& b);
        void run_merger(job& j, const merger& m, bool merged_norm, long column);
        void project_view(job& j);
        void pack_views(job& j);
        void project_pixtral(job& j);
        void product(const weight_matrix& w, const float* x, long n, long x_stride, float* y, long y_stride,
            bool accumulate = false);
        // A matrix of the file, converted to half precision when it is stored in bfloat16.
        weight_matrix readable(const weight_matrix& m);

        std::shared_ptr<model_file> file_;
        thread_pool& pool_;
        family kind_ = family::qwen3vl;
        long hidden_ = 0, heads_ = 0, head_dim_ = 0, ffn_ = 0, patch_ = 0, merge_ = 0, out_dim_ = 0, side_ = 0;
        long image_size_ = 0;           // LLaVA-NeXT: pixels per side of a view
        long feature_blocks_ = 0;       // the blocks a view goes through
        float eps_ = 1e-6f;
        float mean_[3] = { 0.5f, 0.5f, 0.5f };
        float std_[3] = { 0.5f, 0.5f, 0.5f };
        std::string weight_type_;
        size_t weight_bytes_ = 0;

        std::vector<float> patch_w_, patch_b_, pos_;
        weight_matrix patch_matrix_;
        std::vector<float> inv_freq_;
        std::vector<block> blocks_;
        std::vector<merger> deepstack_;
        merger final_;
        // LLaVA-NeXT: the pinpoints (width, height) and the newline embedding.
        std::vector<std::pair<long, long>> pinpoints_;
        std::vector<float> newline_;
        // Pixtral: the norm after the patch embedding and before the projector, the merger of
        // the squares of patches (its columns in the order of the patches of a square), the two
        // layers of the projector and the [IMG_BREAK] embedding.
        std::vector<float> pre_ln_w_, proj_norm_w_, proj1_b_, proj2_b_, img_break_;
        weight_matrix patch_merger_, proj1_, proj2_;
        // The matrices converted when the encoder loads.
        std::vector<std::vector<uint16_t>> owned_;

        // Work buffers; sum_ receives the products of the device that the encoder adds to its rows.
        aligned_floats xn_, qkv_, q_, kt_, v_, ctx_, h_, u_, m_, sum_;
        std::vector<aligned_floats> attention_work_;
        qk_scratch scratch_;
        std::unique_ptr<matmul_offload> offload_;
    };

}

#endif // ALBIREO_ENGINE_VISION_ENCODER_H_
