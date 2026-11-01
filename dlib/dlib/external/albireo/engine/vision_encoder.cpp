// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "vision_encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace albireo
{

    namespace
    {
        // Rows of patches normalized and multiplied at a time: the products of a block take
        // a few megabytes of work memory whatever the size of the image.
        const long chunk_rows = 512;

        std::string type_name(dlib::ggml_type t)
        {
            switch (t)
            {
            case dlib::ggml_type::F32: return "F32";
            case dlib::ggml_type::F16: return "F16";
            case dlib::ggml_type::BF16: return "BF16";
            case dlib::ggml_type::Q8_0: return "Q8_0";
            case dlib::ggml_type::Q4_0: return "Q4_0";
            case dlib::ggml_type::Q4_1: return "Q4_1";
            case dlib::ggml_type::Q5_0: return "Q5_0";
            case dlib::ggml_type::Q5_1: return "Q5_1";
            case dlib::ggml_type::Q4_K: return "Q4_K";
            case dlib::ggml_type::Q5_K: return "Q5_K";
            case dlib::ggml_type::Q6_K: return "Q6_K";
            default: return "quantized";
            }
        }

        long metadata_int(const dlib::gguf_reader& g, const std::string& key, long lo, long hi)
        {
            if (!g.has(key)) throw std::runtime_error("the image encoder lacks the metadata " + key);
            const int64_t v = g.get_int(key, 0);
            if (v < lo || v > hi)
                throw std::runtime_error("the image encoder holds an invalid " + key + " (" + std::to_string(v) + ")");
            return static_cast<long>(v);
        }

        double seconds_since(std::chrono::steady_clock::time_point t)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        }
    }

    bool vision_encoder::present(const model_file& f)
    {
        return f.has("v.patch_embd.weight") && f.reader().has("clip.vision.embedding_length");
    }

    vision_encoder::vision_encoder(std::shared_ptr<model_file> file, thread_pool& pool)
        : file_(std::move(file)), pool_(pool)
    {
        const model_file& f = *file_;
        const dlib::gguf_reader& g = f.reader();
        const std::string type = g.get_str("clip.projector_type", "");
        if (type == "qwen3vl_merger" || type == "qwen3vl")
            kind_ = family::qwen3vl;
        else if (type == "mlp" && g.has("clip.vision.image_grid_pinpoints"))
            kind_ = family::llava_next;
        else if (type == "pixtral")
            kind_ = family::pixtral;
        else
            throw std::runtime_error(type.empty() ? std::string("the file holds no image encoder (clip.projector_type)")
                : "the image encoder of type '" + type + "' is not supported by this engine, which reads those of "
                  "Qwen3-VL (qwen3vl_merger), of LLaVA-NeXT (mlp with pinpoints) and of Pixtral (pixtral)");
        if (!present(f)) throw std::runtime_error("the file holds no vision tensors (v.patch_embd.weight)");
        const bool llava = kind_ == family::llava_next;
        const bool pixtral = kind_ == family::pixtral;

        hidden_ = metadata_int(g, "clip.vision.embedding_length", 16, 1 << 14);
        heads_ = metadata_int(g, "clip.vision.attention.head_count", 1, 256);
        ffn_ = metadata_int(g, "clip.vision.feed_forward_length", 1, 1 << 17);
        const long layers = metadata_int(g, "clip.vision.block_count", 1, 256);
        patch_ = metadata_int(g, "clip.vision.patch_size", 1, 64);
        merge_ = llava ? 1 : g.has("clip.vision.spatial_merge_size")
            ? metadata_int(g, "clip.vision.spatial_merge_size", 1, 8) : pixtral ? 1 : 2;
        // The output width of Pixtral is that of its last layer: its projection_dim names another one.
        if (kind_ == family::qwen3vl) out_dim_ = metadata_int(g, "clip.vision.projection_dim", 1, 1 << 17);
        const double eps = g.get_double("clip.vision.attention.layer_norm_epsilon", 1e-6);
        if (!(eps >= 0.0 && eps < 1.0)) throw std::runtime_error("the image encoder holds an invalid epsilon");
        eps_ = static_cast<float>(eps);
        if (hidden_ % heads_ != 0)
            throw std::runtime_error("the width of the image encoder is not a multiple of its heads");
        head_dim_ = hidden_ / heads_;
        // The two axes take a quarter of the dimensions each, rotated by halves.
        if (!llava && head_dim_ % 4 != 0) throw std::runtime_error("unsupported head geometry of the image encoder");
        /*
            The blocks a patch goes through: all of them, or for LLaVA-NeXT those up to the
            feature layer of the file (the default of the reference, the one before the last,
            when it names none).
        */
        feature_blocks_ = layers;
        if (llava)
        {
            feature_blocks_ = layers - 1;
            if (g.has("clip.vision.feature_layer"))
            {
                const std::vector<int64_t>& fl = g.at("clip.vision.feature_layer").arr_int;
                if (fl.size() != 1 || fl[0] < 1 || fl[0] > layers)
                    throw std::runtime_error("the image encoder reads its features from several layers, which this "
                        "engine does not do");
                feature_blocks_ = static_cast<long>(fl[0]);
            }
            if (feature_blocks_ < 1) throw std::runtime_error("the image encoder has no block");
            image_size_ = metadata_int(g, "clip.vision.image_size", patch_, 4096);
            if (!g.has("clip.vision.image_grid_pinpoints"))
                throw std::runtime_error("the image encoder lacks its pinpoints");
            const std::vector<int64_t>& pins = g.at("clip.vision.image_grid_pinpoints").arr_int;
            if (pins.empty() || pins.size() % 2 != 0)
                throw std::runtime_error("the pinpoints of the image encoder are not pairs");
            // Pairs of height and width, as the configuration of the reference lists them.
            for (size_t i = 0; i < pins.size(); i += 2)
            {
                const long h = static_cast<long>(pins[i]), w = static_cast<long>(pins[i + 1]);
                if (h < image_size_ || w < image_size_ || h % image_size_ != 0 || w % image_size_ != 0 || h > 16384
                    || w > 16384)
                    throw std::runtime_error("a pinpoint of the image encoder is not a grid of whole views");
                pinpoints_.push_back(std::make_pair(w, h));
            }
        }
        // The longest edge of an image read by Pixtral.
        if (pixtral) image_size_ = metadata_int(g, "clip.vision.image_size", factor(), 16384);
        for (int c = 0; c < 3; ++c)
        {
            if (g.has("clip.vision.image_mean") && g.at("clip.vision.image_mean").arr_float.size() == 3)
                mean_[c] = static_cast<float>(g.at("clip.vision.image_mean").arr_float[static_cast<size_t>(c)]);
            if (g.has("clip.vision.image_std") && g.at("clip.vision.image_std").arr_float.size() == 3)
                std_[c] = static_cast<float>(g.at("clip.vision.image_std").arr_float[static_cast<size_t>(c)]);
            if (!(std_[c] > 0.0f) || !std::isfinite(mean_[c]))
                throw std::runtime_error("the image encoder holds an invalid normalization of the pixels");
        }

        // The patch embedding: the two temporal kernels of the convolution, summed, as one
        // matrix of hidden rows of 3 x patch x patch values (channel, row, column).
        const long K = 3 * patch_ * patch_;
        patch_w_.assign(static_cast<size_t>(hidden_ * K), 0.0f);
        std::vector<float> row(static_cast<size_t>(K));
        for (const char* name : { "v.patch_embd.weight", "v.patch_embd.weight.1" })
        {
            const tensor_view* t = f.find(name);
            if (!t) continue;
            if (t->elements() != static_cast<uint64_t>(hidden_ * K) || t->dims.empty()
                || static_cast<long>(t->dims[0]) != patch_ || (t->type != dlib::ggml_type::F32
                && t->type != dlib::ggml_type::F16 && t->type != dlib::ggml_type::BF16))
                throw std::runtime_error(std::string("tensor ") + name + " does not hold the patch embedding");
            const size_t row_bytes = qk_row_bytes(t->type, K);
            for (long o = 0; o < hidden_; ++o)
            {
                qk_dequantize_row(t->type, t->data + static_cast<size_t>(o) * row_bytes, row.data(), K);
                float* dst = patch_w_.data() + o * K;
                for (long k = 0; k < K; ++k) dst[k] += row[static_cast<size_t>(k)];
            }
        }
        patch_matrix_.type = dlib::ggml_type::F32;
        patch_matrix_.data = reinterpret_cast<const uint8_t*>(patch_w_.data());
        patch_matrix_.rows = hidden_;
        patch_matrix_.cols = K;
        patch_matrix_.row_bytes = static_cast<size_t>(K) * sizeof(float);
        // Pixtral convolves without a bias.
        if (pixtral) f.optional_floats("v.patch_embd.bias", static_cast<size_t>(hidden_), patch_b_);
        else patch_b_ = f.floats("v.patch_embd.bias", static_cast<size_t>(hidden_));

        // The learned positions, a square grid (Pixtral has none: its heads rotate by the rows
        // and the columns alone).
        if (!pixtral)
        {
            const tensor_view* pos = f.find("v.position_embd.weight");
            if (!pos || pos->dims.size() < 2) throw std::runtime_error("missing tensor v.position_embd.weight");
            const long cells = static_cast<long>(pos->elements() / static_cast<uint64_t>(hidden_));
            side_ = static_cast<long>(std::lround(std::sqrt(static_cast<double>(cells))));
            if (side_ < 1 || side_ * side_ != cells)
                throw std::runtime_error("the position table is not a square grid");
            if (llava && side_ != image_size_ / patch_)
                throw std::runtime_error("the position table of the image encoder does not cover its views");
            const weight_matrix pm = f.matrix("v.position_embd.weight", cells, hidden_);
            pos_.resize(static_cast<size_t>(cells * hidden_));
            for (long c = 0; c < cells; ++c)
                qk_dequantize_row(pm.type, pm.data + static_cast<size_t>(c) * pm.row_bytes, pos_.data() + c * hidden_,
                    hidden_);
        }

        if (kind_ == family::qwen3vl)
        {
            // Frequencies of the rotary pairs: each axis takes head_dim / 4 of them.
            const long quarter = head_dim_ / 4;
            for (long i = 0; i < quarter; ++i)
                inv_freq_.push_back(static_cast<float>(1.0 / std::pow(10000.0, 2.0 * static_cast<double>(i)
                    / static_cast<double>(head_dim_ / 2))));
        }
        else if (pixtral)
        {
            /*
                Pair p of a head turns at theta^(-e / head_dim): the pairs of the first half,
                rotated by the row of the patch, take the even exponents e = 4p, those of the
                second half, rotated by its column, the odd ones e = 4(p - head_dim / 4) + 2.
            */
            const double theta = g.get_double("clip.rope.freq_base", 10000.0);
            if (!(theta > 1.0 && theta < 1e12))
                throw std::runtime_error("the image encoder holds an invalid rotary base");
            const long half = head_dim_ / 2, quarter = head_dim_ / 4;
            for (long p = 0; p < half; ++p)
            {
                const double e = p < quarter ? 4.0 * static_cast<double>(p)
                    : 4.0 * static_cast<double>(p - quarter) + 2.0;
                inv_freq_.push_back(static_cast<float>(std::pow(theta, -e / static_cast<double>(head_dim_))));
            }
        }

        std::vector<bool> deep(static_cast<size_t>(layers), false);
        if (g.has("clip.vision.is_deepstack_layers"))
        {
            const std::vector<int64_t>& flags = g.at("clip.vision.is_deepstack_layers").arr_int;
            if (flags.size() != static_cast<size_t>(layers))
                throw std::runtime_error("clip.vision.is_deepstack_layers does not name every block");
            for (size_t i = 0; i < flags.size(); ++i) deep[i] = flags[i] != 0;
        }

        std::map<std::string, uint64_t> by_type;
        auto count = [&](const weight_matrix& m) {
            by_type[type_name(m.type)] += m.row_bytes * static_cast<uint64_t>(m.rows);
        };
        const long D = hidden_, M4 = hidden_ * merge_ * merge_;
        for (long i = 0; i < feature_blocks_; ++i)
        {
            const std::string p = "v.blk." + std::to_string(i) + ".";
            block b;
            // Pixtral normalizes by the root mean square: its norms have no bias.
            b.ln1_w = f.floats(p + "ln1.weight", static_cast<size_t>(D));
            if (!pixtral) b.ln1_b = f.floats(p + "ln1.bias", static_cast<size_t>(D));
            b.ln2_w = f.floats(p + "ln2.weight", static_cast<size_t>(D));
            if (!pixtral) b.ln2_b = f.floats(p + "ln2.bias", static_cast<size_t>(D));
            if (f.has(p + "attn_qkv.weight"))
            {
                b.qkv = readable(f.matrix(p + "attn_qkv.weight", 3 * D, D));
                b.qkv_b = f.floats(p + "attn_qkv.bias", static_cast<size_t>(3 * D));
                count(b.qkv);
            }
            else
            {
                // The queries, keys and values apart (SigLIP, Pixtral), their biases end to end.
                b.fused = false;
                b.q = readable(f.matrix(p + "attn_q.weight", D, D));
                b.k = readable(f.matrix(p + "attn_k.weight", D, D));
                b.v = readable(f.matrix(p + "attn_v.weight", D, D));
                bool any = false;
                std::vector<float> biases(static_cast<size_t>(3 * D), 0.0f);
                long at = 0;
                for (const char* part : { "attn_q.bias", "attn_k.bias", "attn_v.bias" })
                {
                    std::vector<float> bias;
                    if (pixtral) any = f.optional_floats(p + part, static_cast<size_t>(D), bias) || any;
                    else
                    {
                        bias = f.floats(p + part, static_cast<size_t>(D));
                        any = true;
                    }
                    if (!bias.empty()) std::copy(bias.begin(), bias.end(), biases.begin() + at);
                    at += D;
                }
                if (any) b.qkv_b.swap(biases);
                count(b.q);
                count(b.k);
                count(b.v);
            }
            b.out = readable(f.matrix(p + "attn_out.weight", D, D));
            if (pixtral) f.optional_floats(p + "attn_out.bias", static_cast<size_t>(D), b.out_b);
            else b.out_b = f.floats(p + "attn_out.bias", static_cast<size_t>(D));
            if (pixtral)
            {
                // A SiLU gate: silu(gate x) * (up x), then the layer down.
                b.gated = true;
                b.gate = readable(f.matrix(p + "ffn_gate.weight", ffn_, D));
                b.up = readable(f.matrix(p + "ffn_up.weight", ffn_, D));
                b.down = readable(f.matrix(p + "ffn_down.weight", D, ffn_));
                f.optional_floats(p + "ffn_up.bias", static_cast<size_t>(ffn_), b.up_b);
                f.optional_floats(p + "ffn_down.bias", static_cast<size_t>(D), b.down_b);
                if (f.has(p + "ffn_gate.bias"))
                    throw std::runtime_error("the gate of the image encoder has a bias, which this engine does not "
                        "read");
                count(b.gate);
            }
            else
            {
                // The files of the first conversion tools name the two layers of the feed-forward
                // part the other way round: the layer that reads the width of the encoder comes first.
                const tensor_view* up = f.find(p + "ffn_up.weight");
                const bool swapped = up && !up->dims.empty() && static_cast<long>(up->dims[0]) != D;
                const std::string first = swapped ? "ffn_down" : "ffn_up", second = swapped ? "ffn_up" : "ffn_down";
                b.up = readable(f.matrix(p + first + ".weight", ffn_, D));
                b.up_b = f.floats(p + first + ".bias", static_cast<size_t>(ffn_));
                b.down = readable(f.matrix(p + second + ".weight", D, ffn_));
                b.down_b = f.floats(p + second + ".bias", static_cast<size_t>(D));
            }
            count(b.out);
            count(b.up);
            count(b.down);
            blocks_.push_back(std::move(b));
            if (deep[static_cast<size_t>(i)])
            {
                const std::string q = "v.deepstack." + std::to_string(i) + ".";
                merger m;
                m.after = i;
                m.norm_w = f.floats(q + "norm.weight", static_cast<size_t>(M4));
                m.norm_b = f.floats(q + "norm.bias", static_cast<size_t>(M4));
                m.fc1 = readable(f.matrix(q + "fc1.weight", M4, M4));
                m.fc1_b = f.floats(q + "fc1.bias", static_cast<size_t>(M4));
                m.fc2 = readable(f.matrix(q + "fc2.weight", out_dim_, M4));
                m.fc2_b = f.floats(q + "fc2.bias", static_cast<size_t>(out_dim_));
                count(m.fc1);
                count(m.fc2);
                deepstack_.push_back(std::move(m));
            }
        }
        if (llava)
        {
            // The projector: two linear layers, the width of their middle and of their output
            // read from their tensors.
            const tensor_view* fc1 = f.find("mm.0.weight");
            const tensor_view* fc2 = f.find("mm.2.weight");
            if (!fc1 || !fc2 || fc1->dims.size() < 2 || fc2->dims.size() < 2)
                throw std::runtime_error("the image encoder lacks the two layers of its projector (mm.0, mm.2)");
            const long middle = static_cast<long>(fc1->dims[1]);
            out_dim_ = static_cast<long>(fc2->dims[1]);
            if (middle < 1 || middle > (1 << 17) || out_dim_ < 1 || out_dim_ > (1 << 17))
                throw std::runtime_error("the projector of the image encoder has an invalid width");
            final_.fc1 = readable(f.matrix("mm.0.weight", middle, D));
            final_.fc1_b = f.floats("mm.0.bias", static_cast<size_t>(middle));
            final_.fc2 = readable(f.matrix("mm.2.weight", out_dim_, middle));
            final_.fc2_b = f.floats("mm.2.bias", static_cast<size_t>(out_dim_));
            if (f.has("model.image_newline")) newline_ = f.floats("model.image_newline", static_cast<size_t>(out_dim_));
            count(final_.fc1);
            count(final_.fc2);
        }
        else if (pixtral)
        {
            // The norm after the patch embedding, then the projector of the merged tokens.
            f.optional_floats("v.pre_ln.weight", static_cast<size_t>(D), pre_ln_w_);
            const tensor_view* l1 = f.find("mm.1.weight");
            const tensor_view* l2 = f.find("mm.2.weight");
            if (!l1 || !l2 || l1->dims.size() < 2 || l2->dims.size() < 2)
                throw std::runtime_error("the image encoder lacks the two layers of its projector (mm.1, mm.2)");
            const long middle = static_cast<long>(l1->dims[1]);
            out_dim_ = static_cast<long>(l2->dims[1]);
            if (middle < 1 || middle > (1 << 17) || out_dim_ < 1 || out_dim_ > (1 << 17))
                throw std::runtime_error("the projector of the image encoder has an invalid width");
            proj1_ = readable(f.matrix("mm.1.weight", middle, D));
            proj2_ = readable(f.matrix("mm.2.weight", out_dim_, middle));
            f.optional_floats("mm.1.bias", static_cast<size_t>(middle), proj1_b_);
            f.optional_floats("mm.2.bias", static_cast<size_t>(out_dim_), proj2_b_);
            count(proj1_);
            count(proj2_);
            if (merge_ > 1)
            {
                /*
                    The merger reads the patches of a square, merge x merge of them, which the
                    encoding keeps one after the other; the reference unfolds them channel by
                    channel (column c * merge^2 + k for the channel c of the patch k of the
                    square), so its columns are put in the order of the patches.
                */
                proj_norm_w_ = f.floats("mm.input_norm.weight", static_cast<size_t>(D));
                const long m2 = merge_ * merge_;
                const weight_matrix pm = f.matrix("mm.patch_merger.weight", D, M4);
                owned_.emplace_back(static_cast<size_t>(D * M4));
                uint16_t* half = owned_.back().data();
                // On the loading thread: the threads of the pool may be computing the passes of other models.
                std::vector<float> row(static_cast<size_t>(M4)), order(static_cast<size_t>(M4));
                for (long o = 0; o < D; ++o)
                {
                    qk_dequantize_row(pm.type, pm.data + static_cast<size_t>(o) * pm.row_bytes, row.data(), M4);
                    for (long c = 0; c < D; ++c)
                        for (long k = 0; k < m2; ++k)
                            order[static_cast<size_t>(k * D + c)] = row[static_cast<size_t>(c * m2 + k)];
                    qk_float_to_half_row(order.data(), half + o * M4, M4);
                }
                patch_merger_.type = dlib::ggml_type::F16;
                patch_merger_.data = reinterpret_cast<const uint8_t*>(half);
                patch_merger_.rows = D;
                patch_merger_.cols = M4;
                patch_merger_.row_bytes = static_cast<size_t>(M4) * sizeof(uint16_t);
                count(patch_merger_);
            }
            // The decoder reads an [IMG_BREAK] after each row of tokens but the last.
            if (!f.has("v.token_embd.img_break"))
                throw std::runtime_error("the image encoder lacks the [IMG_BREAK] embedding (v.token_embd.img_break)");
            img_break_ = f.floats("v.token_embd.img_break", static_cast<size_t>(out_dim_));
        }
        else
        {
            final_.norm_w = f.floats("v.post_ln.weight", static_cast<size_t>(D));
            final_.norm_b = f.floats("v.post_ln.bias", static_cast<size_t>(D));
            final_.fc1 = readable(f.matrix("mm.0.weight", M4, M4));
            final_.fc1_b = f.floats("mm.0.bias", static_cast<size_t>(M4));
            final_.fc2 = readable(f.matrix("mm.2.weight", out_dim_, M4));
            final_.fc2_b = f.floats("mm.2.bias", static_cast<size_t>(out_dim_));
            count(final_.fc1);
            count(final_.fc2);
        }
        uint64_t most = 0;
        for (const auto& kv : by_type)
        {
            weight_bytes_ += static_cast<size_t>(kv.second);
            if (kv.second > most) { most = kv.second; weight_type_ = kv.first; }
        }
        weight_bytes_ += (patch_w_.size() + pos_.size()) * sizeof(float);

        attention_work_.resize(static_cast<size_t>(pool_.size()));
        for (auto& w : attention_work_) w.resize(static_cast<size_t>(qk_encoder_attention_work(head_dim_)));
        scratch_ = qk_scratch();
    }

    std::string vision_encoder::description() const
    {
        std::ostringstream o;
        if (kind_ == family::llava_next)
        {
            o << "image encoder LLaVA-NeXT (" << blocks_.size() << " blocks, width " << hidden_ << ", " << heads_
              << " heads, patch " << patch_ << ", views of " << image_size_ << " pixels, " << pinpoints_.size()
              << " pinpoints, output " << out_dim_ << (newline_.empty() ? "" : ", newline") << ", " << weight_type_
              << ")";
            return o.str();
        }
        if (kind_ == family::pixtral)
        {
            o << "image encoder Pixtral (" << blocks_.size() << " blocks, width " << hidden_ << ", " << heads_
              << " heads, patch " << patch_ << ", merge " << merge_ << "x" << merge_ << ", longest edge "
              << image_size_ << ", output " << out_dim_ << ", [IMG_BREAK], " << weight_type_ << ")";
            return o.str();
        }
        o << "image encoder (" << blocks_.size() << " blocks, width " << hidden_ << ", " << heads_ << " heads, patch "
          << patch_ << ", merge " << merge_ << "x" << merge_ << ", positions " << side_ << "x" << side_;
        if (!deepstack_.empty())
        {
            o << ", deepstack after blocks";
            for (size_t i = 0; i < deepstack_.size(); ++i) o << (i ? ", " : " ") << deepstack_[i].after;
        }
        o << ", output " << out_dim_ << " x " << 1 + deepstack_.size() << ", " << weight_type_ << ")";
        return o.str();
    }

    void vision_encoder::product(const weight_matrix& w, const float* x, long n, long x_stride, float* y, long y_stride,
        bool accumulate)
    {
        qk_matmul(w, x, n, x_stride, y, y_stride, pool_, scratch_, accumulate, qk_path::automatic);
    }

    weight_matrix vision_encoder::readable(const weight_matrix& m)
    {
        if (m.type != dlib::ggml_type::BF16 || !m.valid()) return m;
        owned_.emplace_back(static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols));
        uint16_t* half = owned_.back().data();
        // On the loading thread: the threads of the pool may be computing the passes of other models.
        std::vector<float> row(static_cast<size_t>(m.cols));
        for (long r = 0; r < m.rows; ++r)
        {
            qk_dequantize_row(m.type, m.data + static_cast<size_t>(r) * m.row_bytes, row.data(), m.cols);
            qk_float_to_half_row(row.data(), half + static_cast<size_t>(r) * static_cast<size_t>(m.cols), m.cols);
        }
        weight_matrix c = m;
        c.type = dlib::ggml_type::F16;
        c.data = reinterpret_cast<const uint8_t*>(half);
        c.row_bytes = static_cast<size_t>(m.cols) * sizeof(uint16_t);
        return c;
    }

    void vision_encoder::layer_norm_rows(const float* x, const std::vector<float>& w, const std::vector<float>& b,
        float* y, long rows, long width)
    {
        if (b.empty())
        {
            // The root mean square norm of Pixtral; x and y may be the same rows.
            pool_.parallel_for(rows, [&](long r, int) {
                const float* xr = x + r * width;
                float* yr = y + r * width;
                const float s = 1.0f / std::sqrt(qk_dot_f32(xr, xr, width) / static_cast<float>(width) + eps_);
                for (long i = 0; i < width; ++i) yr[i] = xr[i] * s * w[static_cast<size_t>(i)];
            });
            return;
        }
        pool_.parallel_for(rows, [&](long r, int) {
            qk_layer_norm(x + r * width, w.data(), b.data(), y + r * width, width, eps_);
        });
    }

    void vision_encoder::add_bias_rows(float* y, long stride, const std::vector<float>& bias, long rows)
    {
        const long n = static_cast<long>(bias.size());
        if (n == 0) return;
        pool_.parallel_for(rows, [&](long r, int) { qk_add(y + r * stride, bias.data(), n); });
    }

    namespace
    {
        // Python's round(x, 7): the products of the reference round there before they are truncated.
        double round7(double x)
        {
            return std::nearbyint(x * 1e7) / 1e7;
        }

        // The tiles a detail allows a LLaVA-NeXT image at most.
        long max_tiles(const std::string& detail)
        {
            if (detail == "low") return 1;
            if (detail == "high") return 4;
            if (detail.empty() || detail == "auto") return 2;
            throw std::invalid_argument("the detail of an image is \"low\", \"auto\" or \"high\"");
        }
    }

    vision_encoder::tile_layout vision_encoder::layout(long width, long height, long res_w, long res_h) const
    {
        /*
            The grid of the features of the tiles loses the rows or the columns of the black
            bands of the resized image, as unpad_image() of the reference computes them from the
            size of the image; each row kept takes the newline embedding after it.
        */
        tile_layout t;
        t.res_w = res_w;
        t.res_h = res_h;
        t.rows = res_h / image_size_;
        t.cols = res_w / image_size_;
        const long ch = t.rows * side_, cw = t.cols * side_;
        t.y1 = ch;
        t.x1 = cw;
        const double original = static_cast<double>(width) / static_cast<double>(height);
        const double current = static_cast<double>(cw) / static_cast<double>(ch);
        if (original > current)
        {
            const double scale = static_cast<double>(cw) / static_cast<double>(width);
            const long kept = static_cast<long>(round7(static_cast<double>(height) * scale));
            const long pad = std::max<long>(0, (ch - kept) / 2);
            t.y0 = pad;
            t.y1 = ch - pad;
        }
        else
        {
            const double scale = static_cast<double>(ch) / static_cast<double>(height);
            const long kept = static_cast<long>(round7(static_cast<double>(width) * scale));
            const long pad = std::max<long>(0, (cw - kept) / 2);
            t.x0 = pad;
            t.x1 = cw - pad;
        }
        t.tokens = side_ * side_ + (t.y1 - t.y0) * (t.x1 - t.x0 + (newline_.empty() ? 0 : 1));
        return t;
    }

    vision_encoder::tile_layout vision_encoder::best_layout(long width, long height, long tiles) const
    {
        // select_best_resolution() of the reference: the pinpoint that keeps the most pixels of
        // the image, then the one that wastes the fewest.
        long best_w = 0, best_h = 0;
        long long most = 0, waste = std::numeric_limits<long long>::max();
        const double W = static_cast<double>(width), H = static_cast<double>(height);
        for (const auto& pin : pinpoints_)
        {
            const long w = pin.first, h = pin.second;
            if ((w / image_size_) * (h / image_size_) > tiles) continue;
            const double scale = std::min(static_cast<double>(w) / W, static_cast<double>(h) / H);
            const long long dw = static_cast<long long>(W * scale), dh = static_cast<long long>(H * scale);
            const long long effective = std::min(dw * dh, static_cast<long long>(width) * height);
            const long long wasted = static_cast<long long>(w) * h - effective;
            if (effective > most || (effective == most && wasted < waste))
            {
                most = effective;
                waste = wasted;
                best_w = w;
                best_h = h;
            }
        }
        if (best_w == 0) throw std::invalid_argument("the image encoder has no pinpoint for this detail");
        return layout(width, height, best_w, best_h);
    }

    vision_encoder::image_plan vision_encoder::plan(long width, long height, const std::string& detail) const
    {
        if (width <= 0 || height <= 0) throw std::invalid_argument("the image has no pixels");
        image_plan p;
        if (kind_ == family::qwen3vl)
        {
            long most = 576;
            if (detail == "low") most = 256;
            else if (detail == "high") most = 1024;
            else if (!detail.empty() && detail != "auto")
                throw std::invalid_argument("the detail of an image is \"low\", \"auto\" or \"high\"");
            const long f = factor();
            fit_image_size(width, height, f, 64 * f * f, most * f * f, p.width, p.height);
            p.grid_w = p.width / f;
            p.grid_h = p.height / f;
            p.tokens = p.grid_w * p.grid_h;
            return p;
        }
        if (kind_ == family::pixtral)
        {
            long most = 576;
            if (detail == "low") most = 256;
            else if (detail == "high") most = 1024;
            else if (!detail.empty() && detail != "auto")
                throw std::invalid_argument("the detail of an image is \"low\", \"auto\" or \"high\"");
            /*
                The reference brings the longest edge within image_size() pixels, the sides
                floored, then rounds each side up to a multiple of factor() pixels; a grid of
                more merged tokens than the detail allows shrinks by the same ratio on both
                sides, floored.
            */
            const long f = factor();
            double w = static_cast<double>(width), h = static_cast<double>(height);
            const double ratio = std::max(w, h) / static_cast<double>(image_size_);
            if (ratio > 1.0)
            {
                w = std::max(1.0, std::floor(w / ratio));
                h = std::max(1.0, std::floor(h / ratio));
            }
            long gw = (static_cast<long>(w) - 1) / f + 1, gh = (static_cast<long>(h) - 1) / f + 1;
            if (gw * gh > most)
            {
                const double s = std::sqrt(static_cast<double>(most) / static_cast<double>(gw * gh));
                gw = std::max<long>(1, static_cast<long>(std::floor(static_cast<double>(gw) * s)));
                gh = std::max<long>(1, static_cast<long>(std::floor(static_cast<double>(gh) * s)));
            }
            p.width = gw * f;
            p.height = gh * f;
            p.grid_w = gw;
            p.grid_h = gh;
            p.tokens = gw * gh + gh - 1;
            return p;
        }
        const long tiles = max_tiles(detail);
        if (static_cast<double>(std::max(width, height)) / static_cast<double>(std::min(width, height)) > 200.0)
            throw std::invalid_argument("the sides of the image differ by a ratio over 200");
        // The image is decoded at its size, within 4096 pixels per side.
        const long side = 4096;
        p.width = width;
        p.height = height;
        if (std::max(width, height) > side)
        {
            const double s = static_cast<double>(side) / static_cast<double>(std::max(width, height));
            p.width = std::max<long>(1, static_cast<long>(std::lround(static_cast<double>(width) * s)));
            p.height = std::max<long>(1, static_cast<long>(std::lround(static_cast<double>(height) * s)));
        }
        const tile_layout t = best_layout(p.width, p.height, tiles);
        p.tokens = t.tokens;
        p.grid_w = t.tokens;
        p.grid_h = 1;
        p.tiles_w = t.res_w;
        p.tiles_h = t.res_h;
        return p;
    }

    long vision_encoder::most_tokens(const std::string& detail) const
    {
        if (kind_ == family::qwen3vl)
            return detail == "min" ? 64 : detail == "low" ? 256 : detail == "high" ? 1024 : 576;
        if (kind_ == family::pixtral)
        {
            if (detail == "min") return 1;
            const long most = detail == "low" ? 256 : detail == "high" ? 1024 : 576;
            // The merged tokens, and an [IMG_BREAK] after each row but the last.
            const long rows = std::min(most, image_size_ / factor());
            return most + rows - 1;
        }
        if (detail == "min") return side_ * side_;
        const long tiles = max_tiles(detail);
        long most = 0;
        for (const auto& pin : pinpoints_)
        {
            const long rows = pin.second / image_size_, cols = pin.first / image_size_;
            if (rows * cols > tiles) continue;
            most = std::max(most, side_ * side_ + rows * side_ * (cols * side_ + (newline_.empty() ? 0 : 1)));
        }
        return most;
    }

    void vision_encoder::embed_patches(job& j, const rgb_image& image)
    {
        j.pw_ = image.width / patch_;
        j.ph_ = image.height / patch_;
        j.n_ = j.pw_ * j.ph_;
        const long n = j.n_, D = hidden_, K = 3 * patch_ * patch_, half = head_dim_ / 2, quarter = head_dim_ / 4;
        const long m2 = merge_ * merge_;
        const long grid_w = j.pw_ / merge_;
        const bool rotary = kind_ != family::llava_next;

        // The row and column of patch i, the patches taken square by square of merge x merge.
        auto place = [&](long i, long& row, long& col) {
            const long square = i / m2, within = i % m2;
            row = square / grid_w * merge_ + within / merge_;
            col = square % grid_w * merge_ + within % merge_;
        };

        // The pixels of each patch, normalized, in the order of the kernel.
        xn_.resize(static_cast<size_t>(n * K));
        float scale[3], shift[3];
        for (int c = 0; c < 3; ++c)
        {
            scale[c] = 1.0f / (255.0f * std_[c]);
            shift[c] = mean_[c] / std_[c];
        }
        pool_.parallel_for(n, [&](long i, int) {
            long row, col;
            place(i, row, col);
            float* dst = xn_.data() + i * K;
            for (long c = 0; c < 3; ++c)
                for (long y = 0; y < patch_; ++y)
                {
                    const uint8_t* src = image.pixels.data()
                        + (static_cast<size_t>(row * patch_ + y) * image.width + col * patch_) * 3 + c;
                    float* d = dst + (c * patch_ + y) * patch_;
                    for (long x = 0; x < patch_; ++x) d[x] = static_cast<float>(src[3 * x]) * scale[c] - shift[c];
                }
        });
        j.x_.resize(static_cast<size_t>(n * D));
        product(patch_matrix_, xn_.data(), n, K, j.x_.data(), D);

        // The bias, the interpolated positions and the angles of the rotary pairs.
        if (rotary)
        {
            j.cos_.resize(static_cast<size_t>(n * half));
            j.sin_.resize(static_cast<size_t>(n * half));
        }
        const float ry = j.ph_ > 1 ? static_cast<float>(side_ - 1) / static_cast<float>(j.ph_ - 1) : 0.0f;
        const float rx = j.pw_ > 1 ? static_cast<float>(side_ - 1) / static_cast<float>(j.pw_ - 1) : 0.0f;
        pool_.parallel_for(n, [&](long i, int) {
            long row, col;
            place(i, row, col);
            float* x = j.x_.data() + i * D;
            if (!patch_b_.empty()) qk_add(x, patch_b_.data(), D);
            if (!rotary)
            {
                // A view covers the table of the positions exactly.
                qk_add(x, pos_.data() + (row * side_ + col) * D, D);
                return;
            }
            if (kind_ == family::pixtral)
            {
                // The pairs of the first half turn with the row, those of the second half with the column.
                float* c = j.cos_.data() + i * half;
                float* s = j.sin_.data() + i * half;
                for (long k = 0; k < half; ++k)
                {
                    const double a = static_cast<double>(k < quarter ? row : col) * inv_freq_[static_cast<size_t>(k)];
                    c[k] = static_cast<float>(std::cos(a));
                    s[k] = static_cast<float>(std::sin(a));
                }
                return;
            }
            const float fy = static_cast<float>(row) * ry, fx = static_cast<float>(col) * rx;
            const long y0 = static_cast<long>(std::floor(fy)), x0 = static_cast<long>(std::floor(fx));
            const long y1 = std::min(y0 + 1, side_ - 1), x1 = std::min(x0 + 1, side_ - 1);
            const float wy = fy - static_cast<float>(y0), wx = fx - static_cast<float>(x0);
            const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx, w10 = wy * (1.0f - wx), w11 = wy * wx;
            const float* p00 = pos_.data() + (y0 * side_ + x0) * D;
            const float* p01 = pos_.data() + (y0 * side_ + x1) * D;
            const float* p10 = pos_.data() + (y1 * side_ + x0) * D;
            const float* p11 = pos_.data() + (y1 * side_ + x1) * D;
            for (long d = 0; d < D; ++d) x[d] += w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
            float* c = j.cos_.data() + i * half;
            float* s = j.sin_.data() + i * half;
            for (long k = 0; k < quarter; ++k)
            {
                const float ay = static_cast<float>(row) * inv_freq_[static_cast<size_t>(k)];
                const float ax = static_cast<float>(col) * inv_freq_[static_cast<size_t>(k)];
                c[k] = static_cast<float>(std::cos(static_cast<double>(ay)));
                s[k] = static_cast<float>(std::sin(static_cast<double>(ay)));
                c[quarter + k] = static_cast<float>(std::cos(static_cast<double>(ax)));
                s[quarter + k] = static_cast<float>(std::sin(static_cast<double>(ax)));
            }
        });
        // Pixtral normalizes the patches before its first block.
        if (kind_ == family::pixtral && !pre_ln_w_.empty())
            layer_norm_rows(j.x_.data(), pre_ln_w_, std::vector<float>(), j.x_.data(), n, D);
    }

    std::unique_ptr<vision_encoder::job> vision_encoder::start(const rgb_image& image, const image_plan& plan)
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<job> j(new job());
        if (image.width <= 0 || image.height <= 0
            || image.pixels.size() != static_cast<size_t>(image.width * image.height * 3))
            throw std::invalid_argument("vision_encoder: the image has no pixels");
        if (kind_ != family::llava_next)
        {
            const long fct = factor();
            if (image.width % fct != 0 || image.height % fct != 0)
                throw std::invalid_argument("vision_encoder: the sides of the image are not multiples of "
                    + std::to_string(fct) + " pixels");
            j->grid_w_ = image.width / patch_ / merge_;
            j->grid_h_ = image.height / patch_ / merge_;
            j->tokens_ = j->grid_w_ * j->grid_h_;
            if (kind_ == family::pixtral) j->tokens_ += j->grid_h_ - 1;
            embed_patches(*j, image);
            j->out_.assign(static_cast<size_t>(j->tokens_ * output_width()), 0.0f);
            j->seconds_ += seconds_since(t0);
            return j;
        }

        if (image.width != plan.width || image.height != plan.height || plan.tiles_w < image_size_
            || plan.tiles_h < image_size_)
            throw std::invalid_argument("vision_encoder: the image does not match its plan");
        const tile_layout t = layout(image.width, image.height, plan.tiles_w, plan.tiles_h);
        if (t.tokens != plan.tokens) throw std::invalid_argument("vision_encoder: the tiles of the image changed");
        const long S = image_size_;
        // The whole image, resized to a view.
        j->views_.push_back(resize_bilinear(image, S, S));
        /*
            The image resized into the pinpoint, its aspect ratio kept (get_patch_output_size()
            of the reference rounds the other side up), centered on black, then cut into views.
        */
        const double sw = static_cast<double>(t.res_w) / image.width, sh = static_cast<double>(t.res_h) / image.height;
        long fw = t.res_w, fh = t.res_h;
        if (sw < sh) fh = std::min<long>(static_cast<long>(std::ceil(image.height * sw)), t.res_h);
        else fw = std::min<long>(static_cast<long>(std::ceil(image.width * sh)), t.res_w);
        fw = std::max<long>(1, fw);
        fh = std::max<long>(1, fh);
        const rgb_image fitted = resize_bilinear(image, fw, fh);
        const long left = (t.res_w - fw) / 2, top = (t.res_h - fh) / 2;
        for (long r = 0; r < t.rows; ++r)
            for (long c = 0; c < t.cols; ++c)
            {
                rgb_image v;
                v.width = S;
                v.height = S;
                v.pixels.assign(static_cast<size_t>(S * S * 3), 0);
                for (long y = 0; y < S; ++y)
                {
                    const long fy = r * S + y - top;
                    if (fy < 0 || fy >= fh) continue;
                    const long x0 = std::max<long>(0, left - c * S), x1 = std::min<long>(S, left + fw - c * S);
                    if (x0 >= x1) continue;
                    const uint8_t* src = fitted.pixels.data()
                        + (static_cast<size_t>(fy) * fw + (c * S + x0 - left)) * 3;
                    std::memcpy(v.pixels.data() + (static_cast<size_t>(y) * S + x0) * 3, src,
                        static_cast<size_t>(x1 - x0) * 3);
                }
                j->views_.push_back(std::move(v));
            }
        j->tile_rows_ = t.rows;
        j->tile_cols_ = t.cols;
        j->keep_y0_ = t.y0;
        j->keep_y1_ = t.y1;
        j->keep_x0_ = t.x0;
        j->keep_x1_ = t.x1;
        j->tokens_ = t.tokens;
        j->grid_w_ = t.tokens;
        j->grid_h_ = 1;
        embed_patches(*j, j->views_[0]);
        j->features_.assign(j->views_.size() * static_cast<size_t>(side_ * side_ * out_dim_), 0.0f);
        j->out_.assign(static_cast<size_t>(j->tokens_ * out_dim_), 0.0f);
        j->seconds_ += seconds_since(t0);
        return j;
    }

    void vision_encoder::run_block(job& j, const block& b)
    {
        const long n = j.n_, D = hidden_, H = heads_, hd = head_dim_, half = hd / 2;
        const long C = std::min(chunk_rows, n);
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const bool rotary = kind_ != family::llava_next;
        // Pixtral turns pairs of consecutive dimensions, Qwen3-VL the dimensions half a head apart.
        const bool pairs = kind_ == family::pixtral;
        xn_.resize(static_cast<size_t>(C * D));
        qkv_.resize(static_cast<size_t>(C * 3 * D));
        q_.resize(static_cast<size_t>(n * D));
        kt_.resize(static_cast<size_t>(n * D));
        v_.resize(static_cast<size_t>(n * D));
        ctx_.resize(static_cast<size_t>(n * D));

        /*
            The queries, scaled, and the values of head h lie in rows of hd floats from h * n * hd;
            its keys lie transposed, hd rows of n floats from h * hd * n, so that the scores of
            a query against consecutive keys are vector products.
        */
        for (long c0 = 0; c0 < n; c0 += C)
        {
            const long rows = std::min(C, n - c0);
            layer_norm_rows(j.x_.data() + c0 * D, b.ln1_w, b.ln1_b, xn_.data(), rows, D);
            if (b.fused)
            {
                product(b.qkv, xn_.data(), rows, D, qkv_.data(), 3 * D);
            }
            else
            {
                product(b.q, xn_.data(), rows, D, qkv_.data(), 3 * D);
                product(b.k, xn_.data(), rows, D, qkv_.data() + D, 3 * D);
                product(b.v, xn_.data(), rows, D, qkv_.data() + 2 * D, 3 * D);
            }
            auto one = [&](long r) {
                float* t = qkv_.data() + r * 3 * D;
                if (!b.qkv_b.empty()) qk_add(t, b.qkv_b.data(), 3 * D);
                const long p = c0 + r;
                for (long h = 0; h < H; ++h)
                {
                    float* q = t + h * hd;
                    float* k = t + D + h * hd;
                    if (rotary && pairs)
                    {
                        const float* c = j.cos_.data() + p * half;
                        const float* s = j.sin_.data() + p * half;
                        for (long i = 0; i < half; ++i)
                        {
                            const float q0 = q[2 * i], q1 = q[2 * i + 1];
                            q[2 * i] = q0 * c[i] - q1 * s[i];
                            q[2 * i + 1] = q1 * c[i] + q0 * s[i];
                            const float k0 = k[2 * i], k1 = k[2 * i + 1];
                            k[2 * i] = k0 * c[i] - k1 * s[i];
                            k[2 * i + 1] = k1 * c[i] + k0 * s[i];
                        }
                    }
                    else if (rotary)
                    {
                        const float* c = j.cos_.data() + p * half;
                        const float* s = j.sin_.data() + p * half;
                        for (long i = 0; i < half; ++i)
                        {
                            const float q0 = q[i], q1 = q[i + half];
                            q[i] = q0 * c[i] - q1 * s[i];
                            q[i + half] = q1 * c[i] + q0 * s[i];
                            const float k0 = k[i], k1 = k[i + half];
                            k[i] = k0 * c[i] - k1 * s[i];
                            k[i + half] = k1 * c[i] + k0 * s[i];
                        }
                    }
                    float* qd = q_.data() + (h * n + p) * hd;
                    for (long e = 0; e < hd; ++e) qd[e] = q[e] * scale;
                    float* kd = kt_.data() + h * hd * n + p;
                    for (long e = 0; e < hd; ++e) kd[e * n] = k[e];
                    std::memcpy(v_.data() + (h * n + p) * hd, t + 2 * D + h * hd,
                        sizeof(float) * static_cast<size_t>(hd));
                }
            };
            /*
                Sixteen consecutive patches per task: the keys of a task fill the 64-byte lines
                of the transposed rows together, instead of each line receiving the writes of
                the tasks of several threads.
            */
            const long group = 16;
            pool_.parallel_for((rows + group - 1) / group, [&](long g, int) {
                const long r1 = std::min(rows, (g + 1) * group);
                for (long r = g * group; r < r1; ++r) one(r);
            });
        }

        // Every patch attends to every patch, by tiles of queries of one head.
        const long tile = qk_encoder_tile_rows;
        const long tiles = (n + tile - 1) / tile;
        pool_.parallel_for(H * tiles, [&](long task, int thread) {
            const long h = task / tiles;
            const long t0 = task % tiles * tile;
            const long rows = std::min(tile, n - t0);
            qk_encoder_attention(q_.data() + (h * n + t0) * hd, rows, kt_.data() + h * hd * n, n,
                v_.data() + h * n * hd, hd, n, hd, ctx_.data() + t0 * D + h * hd, D,
                attention_work_[static_cast<size_t>(thread)].data());
        });
        product(b.out, ctx_.data(), n, D, j.x_.data(), D, true);
        add_bias_rows(j.x_.data(), D, b.out_b, n);

        // The feed-forward layer, a chunk of patches at a time.
        h_.resize(static_cast<size_t>(C * ffn_));
        if (b.gated) u_.resize(static_cast<size_t>(C * ffn_));
        for (long c0 = 0; c0 < n; c0 += C)
        {
            const long rows = std::min(C, n - c0);
            float* x = j.x_.data() + c0 * D;
            layer_norm_rows(x, b.ln2_w, b.ln2_b, xn_.data(), rows, D);
            if (b.gated)
            {
                product(b.gate, xn_.data(), rows, D, h_.data(), ffn_);
                product(b.up, xn_.data(), rows, D, u_.data(), ffn_);
                pool_.parallel_for(rows, [&](long r, int) {
                    float* u = u_.data() + r * ffn_;
                    if (!b.up_b.empty()) qk_add(u, b.up_b.data(), ffn_);
                    qk_silu_mul(h_.data() + r * ffn_, u, ffn_);
                });
            }
            else
            {
                product(b.up, xn_.data(), rows, D, h_.data(), ffn_);
                pool_.parallel_for(rows, [&](long r, int) {
                    float* t = h_.data() + r * ffn_;
                    qk_add(t, b.up_b.data(), ffn_);
                    qk_gelu_tanh(t, ffn_);
                });
            }
            product(b.down, h_.data(), rows, ffn_, x, D, true);
            add_bias_rows(x, D, b.down_b, rows);
        }
    }

    void vision_encoder::run_merger(job& j, const merger& m, bool merged_norm, long column)
    {
        // A merged token is merge x merge consecutive patches, whose rows follow one another.
        const long T = j.tokens_, D = hidden_, M4 = hidden_ * merge_ * merge_, W = output_width();
        m_.resize(static_cast<size_t>(T * M4));
        if (merged_norm) layer_norm_rows(j.x_.data(), m.norm_w, m.norm_b, m_.data(), T, M4);
        else layer_norm_rows(j.x_.data(), m.norm_w, m.norm_b, m_.data(), j.n_, D);
        h_.resize(static_cast<size_t>(T * M4));
        product(m.fc1, m_.data(), T, M4, h_.data(), M4);
        pool_.parallel_for(T, [&](long t, int) {
            float* r = h_.data() + t * M4;
            qk_add(r, m.fc1_b.data(), M4);
            qk_gelu_erf(r, M4);
        });
        float* out = j.out_.data() + column;
        product(m.fc2, h_.data(), T, M4, out, W);
        add_bias_rows(out, W, m.fc2_b, T);
    }

    void vision_encoder::project_view(job& j)
    {
        // The projector of LLaVA-NeXT on every patch of the view: a linear layer, an exact GELU,
        // a linear layer.
        const long n = j.n_, D = hidden_, X = final_.fc1.rows, O = out_dim_;
        h_.resize(static_cast<size_t>(n * X));
        product(final_.fc1, j.x_.data(), n, D, h_.data(), X);
        pool_.parallel_for(n, [&](long t, int) {
            float* r = h_.data() + t * X;
            qk_add(r, final_.fc1_b.data(), X);
            qk_gelu_erf(r, X);
        });
        float* out = j.features_.data() + j.view_ * static_cast<size_t>(n * O);
        product(final_.fc2, h_.data(), n, X, out, O);
        add_bias_rows(out, O, final_.fc2_b, n);
    }

    void vision_encoder::pack_views(job& j)
    {
        // The whole image, then the grid of the tiles kept, row by row, each row followed by the newline.
        const long S = side_, n = S * S, O = out_dim_;
        const size_t row_bytes = sizeof(float) * static_cast<size_t>(O);
        float* out = j.out_.data();
        std::memcpy(out, j.features_.data(), row_bytes * static_cast<size_t>(n));
        out += n * O;
        for (long gy = j.keep_y0_; gy < j.keep_y1_; ++gy)
        {
            for (long gx = j.keep_x0_; gx < j.keep_x1_; ++gx)
            {
                const long tile = 1 + (gy / S) * j.tile_cols_ + gx / S;
                std::memcpy(out, j.features_.data() + (tile * n + (gy % S) * S + gx % S) * O, row_bytes);
                out += O;
            }
            if (!newline_.empty())
            {
                std::memcpy(out, newline_.data(), row_bytes);
                out += O;
            }
        }
    }

    void vision_encoder::project_pixtral(job& j)
    {
        /*
            Each patch normalized, the squares merged (their patches follow one another), then
            a linear layer, an exact GELU and a linear layer; the rows of the merged grid go to
            the output each followed, but the last, by the [IMG_BREAK] embedding.
        */
        const long n = j.n_, D = hidden_, m2 = merge_ * merge_, T = n / m2, X = proj1_.rows, O = out_dim_;
        const float* merged = j.x_.data();
        if (patch_merger_.valid())
        {
            m_.resize(static_cast<size_t>(n * D));
            layer_norm_rows(j.x_.data(), proj_norm_w_, std::vector<float>(), m_.data(), n, D);
            xn_.resize(static_cast<size_t>(T * D));
            product(patch_merger_, m_.data(), T, m2 * D, xn_.data(), D);
            merged = xn_.data();
        }
        h_.resize(static_cast<size_t>(T * X));
        product(proj1_, merged, T, D, h_.data(), X);
        pool_.parallel_for(T, [&](long t, int) {
            float* r = h_.data() + t * X;
            if (!proj1_b_.empty()) qk_add(r, proj1_b_.data(), X);
            qk_gelu_erf(r, X);
        });
        ctx_.resize(static_cast<size_t>(T * O));
        product(proj2_, h_.data(), T, X, ctx_.data(), O);
        add_bias_rows(ctx_.data(), O, proj2_b_, T);
        const long gw = j.grid_w_, gh = j.grid_h_;
        float* out = j.out_.data();
        for (long y = 0; y < gh; ++y)
        {
            std::memcpy(out, ctx_.data() + y * gw * O, sizeof(float) * static_cast<size_t>(gw * O));
            out += gw * O;
            if (y + 1 < gh)
            {
                std::memcpy(out, img_break_.data(), sizeof(float) * static_cast<size_t>(O));
                out += O;
            }
        }
    }

    bool vision_encoder::advance(job& j, double seconds)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const long layers = static_cast<long>(blocks_.size());
        while (!j.done_)
        {
            if (j.next_ < layers)
            {
                run_block(j, blocks_[static_cast<size_t>(j.next_)]);
                for (size_t k = 0; k < deepstack_.size(); ++k)
                    if (deepstack_[k].after == j.next_)
                        run_merger(j, deepstack_[k], true, out_dim_ * static_cast<long>(1 + k));
                ++j.next_;
            }
            else if (kind_ == family::llava_next)
            {
                project_view(j);
                ++j.view_;
                if (j.view_ < j.views_.size())
                {
                    embed_patches(j, j.views_[j.view_]);
                    j.next_ = 0;
                }
                else
                {
                    pack_views(j);
                    j.done_ = true;
                    aligned_floats().swap(j.x_);
                    std::vector<float>().swap(j.features_);
                    std::vector<rgb_image>().swap(j.views_);
                }
            }
            else if (kind_ == family::pixtral)
            {
                project_pixtral(j);
                j.done_ = true;
                aligned_floats().swap(j.x_);
                std::vector<float>().swap(j.cos_);
                std::vector<float>().swap(j.sin_);
            }
            else
            {
                run_merger(j, final_, false, 0);
                j.done_ = true;
                aligned_floats().swap(j.x_);
                std::vector<float>().swap(j.cos_);
                std::vector<float>().swap(j.sin_);
            }
            if (seconds_since(t0) >= seconds) break;
        }
        j.seconds_ += seconds_since(t0);
        return j.done_;
    }

    void vision_encoder::encode(const rgb_image& image, const image_plan& plan, std::vector<float>& out)
    {
        std::unique_ptr<job> j = start(image, plan);
        while (!advance(*j, std::numeric_limits<double>::infinity()))
        {
        }
        out.swap(j->output());
        release_work();
    }

    void vision_encoder::release_work()
    {
        aligned_floats().swap(xn_);
        aligned_floats().swap(qkv_);
        aligned_floats().swap(q_);
        aligned_floats().swap(kt_);
        aligned_floats().swap(v_);
        aligned_floats().swap(ctx_);
        aligned_floats().swap(h_);
        aligned_floats().swap(u_);
        aligned_floats().swap(m_);
    }

}
