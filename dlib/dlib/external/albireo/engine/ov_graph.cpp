// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "ov_graph.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <openvino/core/graph_util.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/version.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/less_eq.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>

#include "qkernels.h"

namespace albireo
{

    namespace
    {
        typedef ov::Output<ov::Node> port;

        // The values of a group of the integer formats of the graph, which a scale shares.
        const long group_size = 32;

        port i64_values(const std::vector<int64_t>& v)
        {
            return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
        }

        port i64_scalar(int64_t v)
        {
            return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {v});
        }

        port f32_scalar(float v)
        {
            return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {v});
        }

        port f32_values(const std::vector<float>& v)
        {
            return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{v.size()}, v);
        }

        uint8_t* bytes_of(ov::Tensor& t)
        {
            return static_cast<uint8_t*>(t.data());
        }

        /*
            The weights of a product in the graph: w, rows x cols in single precision, multiplies
            the inputs as MatMul with transpose_b does, and for the formats with minimums, the
            product of the negated minimums, rows x cols / 32, with the sums of the inputs over each
            group of 32 values is added to it.  A subtraction would be turned into a product by -1
            that the plugin of the processor folds into the weights, which then lose their 8-bit
            form.
        */
        struct weight_nodes
        {
            port w;
            port minimums;
            bool with_minimums = false;
            long rows = 0;
            long cols = 0;
        };

        // A tensor whose rows join a product, in the order a permutation gives (none: their own).
        struct weight_part
        {
            const tensor_view* t = nullptr;
            std::vector<long> order;
        };

        /*
            The order of the rows of the queries or keys of a file with rotary pairs of consecutive
            dimensions (rope_style::norm) that gives each head the halves of rope_style::neox: the
            dimensions 2i and 2i + 1 of the rotated part become i and i + rope_dims / 2.  The
            attention, the dot product of a query with a key, does not change when both take the
            same order, and the rotation of the halves then gives the rotation of the pairs.
        */
        std::vector<long> halves_order(long heads, long head_dim, long rope_dims, long first = 0)
        {
            std::vector<long> order;
            order.reserve(static_cast<size_t>(heads * head_dim));
            for (long h = 0; h < heads; ++h)
            {
                const long base = first + h * head_dim;
                for (long i = 0; i < rope_dims / 2; ++i) order.push_back(base + 2 * i);
                for (long i = 0; i < rope_dims / 2; ++i) order.push_back(base + 2 * i + 1);
                for (long i = rope_dims; i < head_dim; ++i) order.push_back(base + i);
            }
            return order;
        }

        // The scale of the 6-bit pair j of the 12 bytes of scales of a block of Q4_K or Q5_K.
        void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m)
        {
            if (j < 4)
            {
                d = q[j] & 63;
                m = q[j + 4] & 63;
            }
            else
            {
                d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
                m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
            }
        }

        uint16_t read_half(const uint8_t* p)
        {
            uint16_t h;
            std::memcpy(&h, p, 2);
            return h;
        }

        class weight_converter
        {
            /*
                Turns the tensors of a GGUF file into the constants of the graph (see ov_graph.h,
                WEIGHTS), and counts their bytes.
            */
        public:
            size_t bytes() const { return bytes_; }

            // Whether the parts can form one product: one format and one width.
            static bool joinable(const std::vector<weight_part>& parts)
            {
                for (const weight_part& p : parts)
                    if (!p.t || p.t->type != parts[0].t->type || p.t->dims[0] != parts[0].t->dims[0]) return false;
                return true;
            }

            weight_nodes convert(const std::vector<weight_part>& parts)
            {
                if (parts.empty() || !joinable(parts))
                    throw std::logic_error("ov_graph: the parts of a product differ in format or width");
                const dlib::ggml_type type = parts[0].t->type;
                const long cols = static_cast<long>(parts[0].t->dims[0]);
                std::vector<const uint8_t*> rows;
                for (const weight_part& p : parts)
                {
                    if (p.t->dims.size() != 2) throw std::runtime_error("ov_graph: " + p.t->name + " is not a matrix");
                    const size_t row_bytes = qk_row_bytes(type, cols);
                    const long n = static_cast<long>(p.t->dims[1]);
                    if (static_cast<uint64_t>(n) * row_bytes > p.t->nbytes)
                        throw std::runtime_error("ov_graph: " + p.t->name + " holds fewer bytes than its shape needs");
                    if (!p.order.empty() && static_cast<long>(p.order.size()) != n)
                        throw std::logic_error("ov_graph: an order of rows of the wrong size");
                    for (long r = 0; r < n; ++r)
                    {
                        const long src = p.order.empty() ? r : p.order[static_cast<size_t>(r)];
                        rows.push_back(p.t->data + static_cast<size_t>(src) * row_bytes);
                    }
                }
                const long n = static_cast<long>(rows.size());
                weight_nodes w;
                w.rows = n;
                w.cols = cols;
                switch (type)
                {
                case dlib::ggml_type::Q4_0: q4_0(rows, w); break;
                case dlib::ggml_type::Q4_1: q4_1(rows, w); break;
                case dlib::ggml_type::Q4_K: q4_k(rows, w); break;
                case dlib::ggml_type::Q5_0: q5_0(rows, w); break;
                case dlib::ggml_type::Q8_0: q8_0(rows, w); break;
                case dlib::ggml_type::F16: case dlib::ggml_type::BF16: half(rows, type, w); break;
                case dlib::ggml_type::F32: single(rows, w); break;
                default:
                    if (!qk_supported(type) || cols % group_size != 0)
                        throw std::runtime_error("ov_graph: the format of " + parts[0].t->name + " is not supported");
                    rounded(rows, type, w);
                    break;
                }
                return w;
            }

        private:
            // values [rows, cols / 32, 32] of type t, scales [rows, cols / 32] in half precision.
            port grouped(ov::Tensor& values, ov::Tensor& scales, long rows, long cols)
            {
                bytes_ += values.get_byte_size() + scales.get_byte_size();
                port x = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(values),
                    ov::element::f16);
                x = std::make_shared<ov::op::v1::Multiply>(x, std::make_shared<ov::op::v0::Constant>(scales));
                x = std::make_shared<ov::op::v1::Reshape>(x, i64_values({rows, cols}), false);
                return std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
            }

            port minimums(ov::Tensor& m)
            {
                bytes_ += m.get_byte_size();
                return std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(m),
                    ov::element::f32);
            }

            static ov::Shape values_shape(long rows, long cols)
            {
                return ov::Shape{static_cast<size_t>(rows), static_cast<size_t>(cols / group_size),
                    static_cast<size_t>(group_size)};
            }

            static ov::Shape scales_shape(long rows, long cols)
            {
                return ov::Shape{static_cast<size_t>(rows), static_cast<size_t>(cols / group_size), 1};
            }

            // Writes the 4-bit value v at index i of a packed buffer, the low half of a byte first.
            static void put4(uint8_t* p, size_t i, unsigned v)
            {
                if (i % 2 == 0) p[i / 2] = static_cast<uint8_t>(v & 0xF);
                else p[i / 2] = static_cast<uint8_t>(p[i / 2] | ((v & 0xF) << 4));
            }

            void q4_0(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, blocks = k / 32;
                ov::Tensor values(ov::element::i4, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                uint8_t* v = bytes_of(values);
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                for (long r = 0; r < n; ++r)
                    for (long b = 0; b < blocks; ++b)
                    {
                        const uint8_t* blk = rows[static_cast<size_t>(r)] + b * 18;
                        s[static_cast<size_t>(r) * blocks + b] = read_half(blk);
                        const size_t base = static_cast<size_t>(r) * k + b * 32;
                        for (int e = 0; e < 32; ++e)
                        {
                            const unsigned q = e < 16 ? (blk[2 + e] & 0xF) : (blk[2 + e - 16] >> 4);
                            put4(v, base + static_cast<size_t>(e), static_cast<unsigned>(static_cast<int>(q) - 8));
                        }
                    }
                w.w = grouped(values, scales, n, k);
            }

            void q4_1(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, blocks = k / 32;
                ov::Tensor values(ov::element::u4, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                ov::Tensor mins(ov::element::f16, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(blocks)});
                uint8_t* v = bytes_of(values);
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                uint16_t* m = static_cast<uint16_t*>(mins.data());
                for (long r = 0; r < n; ++r)
                    for (long b = 0; b < blocks; ++b)
                    {
                        const uint8_t* blk = rows[static_cast<size_t>(r)] + b * 20;
                        s[static_cast<size_t>(r) * blocks + b] = read_half(blk);
                        // w = d q + m: the negated minimum is m.
                        m[static_cast<size_t>(r) * blocks + b] = read_half(blk + 2);
                        const size_t base = static_cast<size_t>(r) * k + b * 32;
                        for (int e = 0; e < 32; ++e)
                            put4(v, base + static_cast<size_t>(e),
                                e < 16 ? (blk[4 + e] & 0xF) : (blk[4 + e - 16] >> 4));
                    }
                w.w = grouped(values, scales, n, k);
                w.minimums = minimums(mins);
                w.with_minimums = true;
            }

            void q4_k(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, supers = k / 256, groups = k / 32;
                ov::Tensor values(ov::element::u4, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                /*
                    The negated minimums -dmin m of the groups, the 6-bit integers m of the 8 groups of
                    a block with the scale -dmin of the block: their product keeps 8-bit integers, and
                    its groups of 8 values keep the sums of the inputs from the rounding to 8 bits that
                    the plugin of the processor gives the inputs of its products by groups of 32.
                */
                ov::Tensor mins(ov::element::u8, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(supers), 8});
                ov::Tensor min_scales(ov::element::f16,
                    ov::Shape{static_cast<size_t>(n), static_cast<size_t>(supers), 1});
                uint8_t* v = bytes_of(values);
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                uint8_t* m = bytes_of(mins);
                uint16_t* ms = static_cast<uint16_t*>(min_scales.data());
                for (long r = 0; r < n; ++r)
                    for (long b = 0; b < supers; ++b)
                    {
                        const uint8_t* blk = rows[static_cast<size_t>(r)] + b * 144;
                        const float d = qk_half_to_float(read_half(blk));
                        // The negated scale of the minimums.
                        ms[static_cast<size_t>(r) * supers + b] =
                            static_cast<uint16_t>(read_half(blk + 2) ^ 0x8000);
                        const uint8_t* sc = blk + 4;
                        const uint8_t* qs = blk + 16;
                        for (int g = 0; g < 8; ++g)
                        {
                            uint8_t gs, gm;
                            scale_min_k4(g, sc, gs, gm);
                            // w = d sc q - dmin m: the values q times the scale d sc, less the minimum dmin m.
                            s[static_cast<size_t>(r) * groups + b * 8 + g] = qk_float_to_half(d * gs);
                            m[static_cast<size_t>(r) * groups + b * 8 + g] = gm;
                            const uint8_t* q = qs + (g / 2) * 32;
                            const size_t base = static_cast<size_t>(r) * k + b * 256 + g * 32;
                            for (int l = 0; l < 32; ++l)
                                put4(v, base + static_cast<size_t>(l), g % 2 == 0 ? (q[l] & 0xF) : (q[l] >> 4));
                        }
                    }
                w.w = grouped(values, scales, n, k);
                bytes_ += mins.get_byte_size() + min_scales.get_byte_size();
                port x = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(mins),
                    ov::element::f16);
                x = std::make_shared<ov::op::v1::Multiply>(x, std::make_shared<ov::op::v0::Constant>(min_scales));
                x = std::make_shared<ov::op::v1::Reshape>(x, i64_values({n, groups}), false);
                w.minimums = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
                w.with_minimums = true;
            }

            void q5_0(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, blocks = k / 32;
                ov::Tensor values(ov::element::i8, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                int8_t* v = static_cast<int8_t*>(values.data());
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                for (long r = 0; r < n; ++r)
                    for (long b = 0; b < blocks; ++b)
                    {
                        const uint8_t* blk = rows[static_cast<size_t>(r)] + b * 22;
                        s[static_cast<size_t>(r) * blocks + b] = read_half(blk);
                        uint32_t qh;
                        std::memcpy(&qh, blk + 2, 4);
                        const uint8_t* qs = blk + 6;
                        int8_t* out = v + static_cast<size_t>(r) * k + b * 32;
                        for (int j = 0; j < 16; ++j)
                        {
                            const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10);
                            const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10);
                            out[j] = static_cast<int8_t>(((qs[j] & 0x0F) | h0) - 16);
                            out[j + 16] = static_cast<int8_t>(((qs[j] >> 4) | h1) - 16);
                        }
                    }
                w.w = grouped(values, scales, n, k);
            }

            void q8_0(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, blocks = k / 32;
                ov::Tensor values(ov::element::i8, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                int8_t* v = static_cast<int8_t*>(values.data());
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                for (long r = 0; r < n; ++r)
                    for (long b = 0; b < blocks; ++b)
                    {
                        const uint8_t* blk = rows[static_cast<size_t>(r)] + b * 34;
                        s[static_cast<size_t>(r) * blocks + b] = read_half(blk);
                        std::memcpy(v + static_cast<size_t>(r) * k + b * 32, blk + 2, 32);
                    }
                w.w = grouped(values, scales, n, k);
            }

            // The other formats, rounded again to 8-bit integers by groups of 32 values.
            void rounded(const std::vector<const uint8_t*>& rows, dlib::ggml_type type, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols, blocks = k / 32;
                ov::Tensor values(ov::element::i8, values_shape(n, k));
                ov::Tensor scales(ov::element::f16, scales_shape(n, k));
                int8_t* v = static_cast<int8_t*>(values.data());
                uint16_t* s = static_cast<uint16_t*>(scales.data());
                std::vector<float> row(static_cast<size_t>(k));
                for (long r = 0; r < n; ++r)
                {
                    qk_dequantize_row(type, rows[static_cast<size_t>(r)], row.data(), k);
                    for (long b = 0; b < blocks; ++b)
                    {
                        const float* x = row.data() + b * 32;
                        float amax = 0;
                        for (int e = 0; e < 32; ++e) amax = std::max(amax, std::fabs(x[e]));
                        // The scale the graph reads, in half precision, rounds the values.
                        const uint16_t h = qk_float_to_half(amax / 127.0f);
                        const float scale = qk_half_to_float(h);
                        const float inv = scale > 0 ? 1.0f / scale : 0.0f;
                        s[static_cast<size_t>(r) * blocks + b] = h;
                        int8_t* out = v + static_cast<size_t>(r) * k + b * 32;
                        for (int e = 0; e < 32; ++e)
                        {
                            const float q = std::nearbyint(x[e] * inv);
                            out[e] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, q)));
                        }
                    }
                }
                w.w = grouped(values, scales, n, k);
            }

            void half(const std::vector<const uint8_t*>& rows, dlib::ggml_type type, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols;
                ov::Tensor values(ov::element::f16, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(k)});
                uint16_t* v = static_cast<uint16_t*>(values.data());
                std::vector<float> row(static_cast<size_t>(k));
                for (long r = 0; r < n; ++r)
                {
                    if (type == dlib::ggml_type::F16)
                    {
                        std::memcpy(v + static_cast<size_t>(r) * k, rows[static_cast<size_t>(r)],
                            static_cast<size_t>(k) * 2);
                        continue;
                    }
                    qk_dequantize_row(type, rows[static_cast<size_t>(r)], row.data(), k);
                    qk_float_to_half_row(row.data(), v + static_cast<size_t>(r) * k, k);
                }
                bytes_ += values.get_byte_size();
                w.w = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(values),
                    ov::element::f32);
            }

            void single(const std::vector<const uint8_t*>& rows, weight_nodes& w)
            {
                const long n = w.rows, k = w.cols;
                ov::Tensor values(ov::element::f32, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(k)});
                float* v = static_cast<float*>(values.data());
                for (long r = 0; r < n; ++r)
                    std::memcpy(v + static_cast<size_t>(r) * k, rows[static_cast<size_t>(r)],
                        static_cast<size_t>(k) * 4);
                bytes_ += values.get_byte_size();
                w.w = std::make_shared<ov::op::v0::Constant>(values);
            }

            size_t bytes_ = 0;
        };

        // Names of the inputs and outputs of the graph.
        const char* const input_embeds = "inputs_embeds";
        const char* const input_cos = "rope_cos";
        const char* const input_sin = "rope_sin";
        const char* const input_beam = "beam_idx";
        const char* const input_rows = "logit_index";
        const char* const input_deepstack = "deepstack";
        const char* const input_query_scale = "query_scale";
        const char* const input_mask = "attention_mask";
        const char* const output_logits = "logits";
        const char* const output_hidden = "hidden";

        class graph_builder
        {
            /*
                Builds the graph of a model (see ov_graph.h, GRAPH).  The keys and values of
                layer i are the variables "past_key.i" and "past_value.i", [1, kv heads,
                positions, head dimension], in kv_type.
            */
        public:
            graph_builder(const model_file& f, const model_config& c, bool explicit_mask, ov::element::Type kv_type,
                bool logits)
                : file_(f), cfg_(c), explicit_mask_(explicit_mask), kv_type_(kv_type), logits_(logits)
            {
            }

            size_t constant_bytes() const { return weights_.bytes(); }

            std::shared_ptr<ov::Model> build()
            {
                const long d = cfg_.d_model;
                const long rot = cfg_.rope_dims;
                embeds_ = parameter(ov::element::f32, ov::PartialShape{1, -1, d}, input_embeds);
                cos_ = parameter(ov::element::f32, ov::PartialShape{1, 1, -1, rot}, input_cos);
                sin_ = parameter(ov::element::f32, ov::PartialShape{1, 1, -1, rot}, input_sin);
                beam_ = parameter(ov::element::i32, ov::PartialShape{-1}, input_beam);
                rows_ = parameter(ov::element::i64, ov::PartialShape{-1}, input_rows);
                if (cfg_.n_deepstack > 0)
                    deepstack_ = parameter(ov::element::f32, ov::PartialShape{cfg_.n_deepstack, 1, -1, d},
                        input_deepstack);
                if (cfg_.attn_temp_scale != 0.0)
                    query_scale_ = parameter(ov::element::f32, ov::PartialShape{1, 1, -1, 1}, input_query_scale);
                if (explicit_mask_)
                {
                    mask_input_ = parameter(ov::element::i64, ov::PartialShape{1, -1}, input_mask);
                    mask_ = causal_mask();
                }

                port x = embeds_;
                for (long li = 0; li < cfg_.n_layers; ++li)
                {
                    x = layer(x, li);
                    if (li < cfg_.n_deepstack)
                    {
                        // Slice li + 1 of the inputs of an image token joins its hidden state after layer li.
                        const port slice = std::make_shared<ov::op::v8::Gather>(deepstack_, i64_scalar(li),
                            i64_scalar(0));
                        x = std::make_shared<ov::op::v1::Add>(x, slice);
                    }
                }
                const port picked = std::make_shared<ov::op::v8::Gather>(x, rows_, i64_scalar(1));
                const port hidden = rms(picked, file_.floats("output_norm.weight", static_cast<size_t>(d)));
                ov::ResultVector results;
                results.push_back(result(hidden, output_hidden));
                if (logits_)
                {
                    const tensor_view* out = file_.find(cfg_.tied_embeddings ? "token_embd.weight" : "output.weight");
                    if (!out) throw std::runtime_error("ov_graph: the model has no output matrix");
                    port logits = product(hidden, weights_.convert({weight_part{out, {}}}));
                    if (cfg_.logit_scale != 1.0)
                        logits = std::make_shared<ov::op::v1::Multiply>(logits,
                            f32_scalar(static_cast<float>(1.0 / cfg_.logit_scale)));
                    results.push_back(result(logits, output_logits));
                }
                return std::make_shared<ov::Model>(results, sinks_, parameters_, "albireo-decoder");
            }

        private:
            port parameter(ov::element::Type type, const ov::PartialShape& shape, const char* name)
            {
                auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
                p->set_friendly_name(name);
                p->output(0).get_tensor().set_names({name});
                parameters_.push_back(p);
                return p;
            }

            static std::shared_ptr<ov::op::v0::Result> result(const port& value, const char* name)
            {
                auto r = std::make_shared<ov::op::v0::Result>(value);
                r->set_friendly_name(name);
                r->output(0).get_tensor().set_names({name});
                return r;
            }

            const tensor_view* tensor(const std::string& name) const
            {
                const tensor_view* t = file_.find(name);
                if (!t) throw std::runtime_error("ov_graph: the model lacks the tensor " + name);
                return t;
            }

            // The RMS normalization of the last axis, the pattern the plugins fuse into one node.
            port rms(const port& x, const std::vector<float>& gamma) const
            {
                const port square = std::make_shared<ov::op::v1::Multiply>(x, x);
                const port mean = std::make_shared<ov::op::v1::ReduceMean>(square, i64_values({-1}), true);
                const port root = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(mean,
                    f32_scalar(static_cast<float>(cfg_.rms_eps))));
                const port inverse = std::make_shared<ov::op::v1::Divide>(f32_scalar(1.0f), root);
                return std::make_shared<ov::op::v1::Multiply>(f32_values(gamma),
                    std::make_shared<ov::op::v1::Multiply>(x, inverse));
            }

            /*
                The rotation of the halves of the rotated part of each head of x [1, heads, L,
                head_dim], whose cosines and sines the host gives for both halves: the pattern
                (x cos + rotate_half(x) sin) that the plugins fuse into one node.
            */
            port rope(const port& x) const
            {
                const long hd = cfg_.head_dim;
                const long rot = cfg_.rope_dims;
                port part = x, rest;
                if (rot < hd)
                {
                    auto split = std::make_shared<ov::op::v1::VariadicSplit>(x, i64_scalar(3),
                        i64_values({rot, hd - rot}));
                    part = split->output(0);
                    rest = split->output(1);
                }
                auto halves = std::make_shared<ov::op::v1::VariadicSplit>(part, i64_scalar(3),
                    i64_values({rot / 2, -1}));
                const port negated = std::make_shared<ov::op::v1::Multiply>(halves->output(1), f32_scalar(-1.0f));
                const port rotated = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{negated, halves->output(0)},
                    -1);
                port y = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(part, cos_),
                    std::make_shared<ov::op::v1::Multiply>(rotated, sin_));
                if (rot < hd) y = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{y, rest}, -1);
                return y;
            }

            // The sums of x [1, L, cols] over each group of 32 values, computed once per input.
            port group_sums(const port& x, long cols)
            {
                const std::pair<const ov::Node*, size_t> key(x.get_node(), x.get_index());
                auto it = sums_.find(key);
                if (it != sums_.end()) return it->second;
                const port grouped = std::make_shared<ov::op::v1::Reshape>(x, i64_values({0, 0, cols / group_size,
                    group_size}), true);
                const port sums = std::make_shared<ov::op::v1::ReduceSum>(grouped, i64_values({-1}), false);
                sums_[key] = sums;
                return sums;
            }

            port product(const port& x, const weight_nodes& w)
            {
                port y = std::make_shared<ov::op::v0::MatMul>(x, w.w, false, true);
                if (w.with_minimums)
                    y = std::make_shared<ov::op::v1::Add>(y, std::make_shared<ov::op::v0::MatMul>(
                        group_sums(x, w.cols), w.minimums, false, true));
                return y;
            }

            // Splits the last axis of x into the given widths.
            static std::vector<port> split(const port& x, const std::vector<int64_t>& widths)
            {
                auto s = std::make_shared<ov::op::v1::VariadicSplit>(x, i64_scalar(2), i64_values(widths));
                std::vector<port> out;
                for (size_t i = 0; i < widths.size(); ++i) out.push_back(s->output(i));
                return out;
            }

            /*
                The products of x with the matrices of names, as one product when they share a
                format, and their parts in the order of names.
            */
            std::vector<port> products(const port& x, const std::vector<std::string>& names,
                const std::vector<std::vector<long>>& orders, const std::vector<int64_t>& widths)
            {
                std::vector<weight_part> parts;
                for (size_t i = 0; i < names.size(); ++i) parts.push_back(weight_part{tensor(names[i]), orders[i]});
                if (parts.size() > 1 && weight_converter::joinable(parts))
                    return split(product(x, weights_.convert(parts)), widths);
                // Queries and keys alike share a product more often than the values (Q4_K_M).
                std::vector<port> out;
                if (parts.size() == 3 && weight_converter::joinable({parts[0], parts[1]}))
                {
                    const std::vector<port> qk = split(product(x, weights_.convert({parts[0], parts[1]})),
                        {widths[0], widths[1]});
                    out = qk;
                    out.push_back(product(x, weights_.convert({parts[2]})));
                    return out;
                }
                for (const weight_part& p : parts) out.push_back(product(x, weights_.convert({p})));
                return out;
            }

            port reshape_heads(const port& x, long heads) const
            {
                return std::make_shared<ov::op::v1::Reshape>(x, i64_values({0, 0, heads, cfg_.head_dim}), true);
            }

            // mask [1, 1, L, S]: 0 where a token of the pass sees a position, the lowest float elsewhere.
            port causal_mask() const
            {
                const port s = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(mask_input_),
                    i64_scalar(1), i64_scalar(0));
                const port l = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(embeds_),
                    i64_scalar(1), i64_scalar(0));
                const port i = std::make_shared<ov::op::v4::Range>(i64_scalar(0), l, i64_scalar(1), ov::element::i64);
                const port j = std::make_shared<ov::op::v4::Range>(i64_scalar(0), s, i64_scalar(1), ov::element::i64);
                const port past = std::make_shared<ov::op::v1::Subtract>(s, l);
                const port limit = std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v1::Add>(i, past),
                    i64_scalar(1));
                const port seen = std::make_shared<ov::op::v1::LessEqual>(std::make_shared<ov::op::v0::Unsqueeze>(j,
                    i64_scalar(0)), limit);
                const port mask = std::make_shared<ov::op::v1::Select>(seen, f32_scalar(0.0f), f32_scalar(-FLT_MAX));
                return std::make_shared<ov::op::v0::Unsqueeze>(mask, i64_values({0, 1}));
            }

            // The keys or values of the pass after those of the sequence, kept in the variable of their layer.
            port cache(const port& current, const std::string& id)
            {
                const long kvh = cfg_.n_kv_heads;
                const long hd = cfg_.head_dim;
                auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
                    ov::PartialShape{-1, kvh, -1, hd}, kv_type_, id});
                const port initial = ov::op::v0::Constant::create(kv_type_, ov::Shape{1, static_cast<size_t>(kvh), 0,
                    static_cast<size_t>(hd)}, std::vector<float>());
                port past = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
                if (kv_type_ != ov::element::f32) past = std::make_shared<ov::op::v0::Convert>(past, ov::element::f32);
                past = std::make_shared<ov::op::v8::Gather>(past, beam_, i64_scalar(0));
                const port all = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 2);
                port stored = all;
                if (kv_type_ != ov::element::f32) stored = std::make_shared<ov::op::v0::Convert>(all, kv_type_);
                sinks_.push_back(std::make_shared<ov::op::v6::Assign>(stored, variable));
                return all;
            }

            // The keys or values of the kv heads repeated for the heads that share them.
            port repeat_heads(const port& kv) const
            {
                const long h = cfg_.n_heads;
                const long kvh = cfg_.n_kv_heads;
                if (h == kvh) return kv;
                const port unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, i64_scalar(2));
                const port expanded = std::make_shared<ov::op::v3::Broadcast>(unsqueezed,
                    i64_values({1, 1, h / kvh, 1, 1}), ov::op::BroadcastType::BIDIRECTIONAL);
                return std::make_shared<ov::op::v1::Reshape>(expanded, i64_values({0, h, -1, cfg_.head_dim}), true);
            }

            port layer(const port& x_in, long li)
            {
                const long d = cfg_.d_model;
                const long h = cfg_.n_heads;
                const long kvh = cfg_.n_kv_heads;
                const long hd = cfg_.head_dim;
                const long qd = cfg_.q_dim();
                const long kvd = cfg_.kv_dim();
                const long df = cfg_.d_ffn;
                const std::string p = "blk." + std::to_string(li) + ".";
                const bool norm = cfg_.rope == rope_style::norm;
                const port xn = rms(x_in, file_.floats(p + "attn_norm.weight", static_cast<size_t>(d)));

                std::vector<port> qkv;
                if (cfg_.fused_qkv)
                {
                    std::vector<long> order;
                    if (norm)
                    {
                        order = halves_order(h, hd, cfg_.rope_dims);
                        const std::vector<long> k_order = halves_order(kvh, hd, cfg_.rope_dims, qd);
                        order.insert(order.end(), k_order.begin(), k_order.end());
                        for (long i = 0; i < kvd; ++i) order.push_back(qd + kvd + i);
                    }
                    qkv = products(xn, {p + "attn_qkv.weight"}, {order}, {qd + 2 * kvd});
                    qkv = split(qkv[0], {qd, kvd, kvd});
                }
                else
                {
                    const std::vector<long> none;
                    const std::vector<long> q_order = norm ? halves_order(h, hd, cfg_.rope_dims) : none;
                    const std::vector<long> k_order = norm ? halves_order(kvh, hd, cfg_.rope_dims) : none;
                    qkv = products(xn, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
                        {q_order, k_order, std::vector<long>()}, {qd, kvd, kvd});
                }
                port q = qkv[0], k = qkv[1], v = qkv[2];
                std::vector<float> bq, bk, bv;
                if (file_.optional_floats(p + "attn_q.bias", static_cast<size_t>(qd), bq))
                {
                    bk = file_.floats(p + "attn_k.bias", static_cast<size_t>(kvd));
                    bv = file_.floats(p + "attn_v.bias", static_cast<size_t>(kvd));
                    if (norm)
                    {
                        bq = reordered(bq, halves_order(h, hd, cfg_.rope_dims));
                        bk = reordered(bk, halves_order(kvh, hd, cfg_.rope_dims));
                    }
                    q = std::make_shared<ov::op::v1::Add>(q, f32_values(bq));
                    k = std::make_shared<ov::op::v1::Add>(k, f32_values(bk));
                    v = std::make_shared<ov::op::v1::Add>(v, f32_values(bv));
                }
                q = reshape_heads(q, h);
                k = reshape_heads(k, kvh);
                v = reshape_heads(v, kvh);
                std::vector<float> qn, kn;
                if (file_.optional_floats(p + "attn_q_norm.weight", static_cast<size_t>(hd), qn))
                {
                    kn = file_.floats(p + "attn_k_norm.weight", static_cast<size_t>(hd));
                    if (norm)
                    {
                        qn = reordered(qn, halves_order(1, hd, cfg_.rope_dims));
                        kn = reordered(kn, halves_order(1, hd, cfg_.rope_dims));
                    }
                    q = rms(q, qn);
                    k = rms(k, kn);
                }
                const port to_heads = i64_values({0, 2, 1, 3});
                q = rope(std::make_shared<ov::op::v1::Transpose>(q, to_heads));
                k = rope(std::make_shared<ov::op::v1::Transpose>(k, to_heads));
                v = std::make_shared<ov::op::v1::Transpose>(v, to_heads);
                // The attention takes 1 / sqrt(head_dim) itself; a model that declares another scale corrects it on q.
                if (cfg_.attention_scale > 0.0)
                    q = std::make_shared<ov::op::v1::Multiply>(q, f32_scalar(static_cast<float>(cfg_.attention_scale
                        * std::sqrt(static_cast<double>(hd)))));
                if (query_scale_.get_node()) q = std::make_shared<ov::op::v1::Multiply>(q, query_scale_);

                const port keys = repeat_heads(cache(k, "past_key." + std::to_string(li)));
                const port values = repeat_heads(cache(v, "past_value." + std::to_string(li)));
                port attended = explicit_mask_
                    ? port(std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, keys, values, mask_, false))
                    : port(std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, keys, values, true));
                attended = std::make_shared<ov::op::v1::Transpose>(attended, to_heads);
                attended = std::make_shared<ov::op::v1::Reshape>(attended, i64_values({0, 0, qd}), true);
                port out = product(attended, weights_.convert({weight_part{tensor(p + "attn_output.weight"), {}}}));
                if (cfg_.residual_scale != 1.0)
                    out = std::make_shared<ov::op::v1::Multiply>(out,
                        f32_scalar(static_cast<float>(cfg_.residual_scale)));
                const port x = std::make_shared<ov::op::v1::Add>(x_in, out);

                const port fn = rms(x, file_.floats(p + "ffn_norm.weight", static_cast<size_t>(d)));
                std::vector<port> gate_up = cfg_.fused_gate_up
                    ? split(products(fn, {p + "ffn_up.weight"}, {std::vector<long>()}, {2 * df})[0], {df, df})
                    : products(fn, {p + "ffn_gate.weight", p + "ffn_up.weight"}, {std::vector<long>(),
                        std::vector<long>()}, {df, df});
                const port act = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v4::Swish>(gate_up[0]),
                    gate_up[1]);
                port down = product(act, weights_.convert({weight_part{tensor(p + "ffn_down.weight"), {}}}));
                if (cfg_.residual_scale != 1.0)
                    down = std::make_shared<ov::op::v1::Multiply>(down,
                        f32_scalar(static_cast<float>(cfg_.residual_scale)));
                return std::make_shared<ov::op::v1::Add>(x, down);
            }

            static std::vector<float> reordered(const std::vector<float>& v, const std::vector<long>& order)
            {
                std::vector<float> out(v.size());
                for (size_t i = 0; i < v.size(); ++i) out[i] = v[static_cast<size_t>(order[i])];
                return out;
            }

            const model_file& file_;
            const model_config& cfg_;
            const bool explicit_mask_;
            const ov::element::Type kv_type_;
            const bool logits_;
            weight_converter weights_;
            ov::ParameterVector parameters_;
            ov::SinkVector sinks_;
            port embeds_, cos_, sin_, beam_, rows_, deepstack_, query_scale_, mask_input_, mask_;
            std::map<std::pair<const ov::Node*, size_t>, port> sums_;
        };

        // The position component of each rotary pair (0 time, 1 row, 2 column), as cpu_runtime takes them.
        std::vector<int> pair_components(const model_config& c)
        {
            std::vector<int> comp(static_cast<size_t>(c.rope_dims / 2), 0);
            if (c.multimodal_rope())
                for (long j = 0; j < c.rope_dims / 2; ++j)
                {
                    if (j % 3 == 1 && j < 3 * c.rope_sections[1]) comp[static_cast<size_t>(j)] = 1;
                    else if (j % 3 == 2 && j < 3 * c.rope_sections[2]) comp[static_cast<size_t>(j)] = 2;
                }
            return comp;
        }

        std::string upper(std::string s)
        {
            for (char& ch : s) if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
            return s;
        }

        class ov_runtime final : public ov_graph_runtime
        {
        public:
            ov_runtime(std::shared_ptr<model_file> file, const model_config& config, const ov_graph_settings& settings,
                size_t kv_budget_bytes)
                : file_(std::move(file)), cfg_(config), settings_(settings), kv_budget_(kv_budget_bytes)
            {
                const auto start = std::chrono::steady_clock::now();
                settings_.device = upper(settings_.device);
                cpu_ = settings_.device == "CPU";
                views_ = cpu_ && state_copied_at_once();
                // A device that OpenVINO does not find fails here, before the weights are converted.
                if (!cpu_) check_device();
                // The plugin of the processor fuses the causal attention with its cache, and the others take a mask.
                explicit_mask_ = !cpu_ || settings_.explicit_mask;
                // On the processor, the variables hold half precision and the plugin its own format.
                const ov::element::Type kv_type = ov::element::f16;
                std::shared_ptr<ov::Model> model;
                {
                    graph_builder builder(*file_, cfg_, explicit_mask_, kv_type, !cfg_.is_embedding());
                    model = builder.build();
                    constant_bytes_ = builder.constant_bytes();
                }
                ov::AnyMap properties;
                properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
                if (cpu_)
                {
                    // Single precision whatever the processor offers: bf16 would round the activations to 8 bits.
                    properties[ov::hint::inference_precision.name()] = ov::element::f32;
                    properties[ov::hint::kv_cache_precision.name()] = settings_.kv_half ? ov::element::f16
                        : ov::element::u8;
                    properties[ov::hint::dynamic_quantization_group_size.name()] = static_cast<uint64_t>(group_size);
                    if (settings_.threads > 0) properties[ov::inference_num_threads.name()] = settings_.threads;
                }
                if (!settings_.weights_dir.empty()) model = spill(model);
                compiled_ = core_.compile_model(model, settings_.device, properties);
                model.reset();
                // Linux keeps the pages of a file deleted while it is mapped; Windows refuses to delete it.
                remove_file(bin_.path);
                fused_attention_ = count_fused_attention();
                components_ = pair_components(cfg_);
                const tensor_view* embd = file_->find("token_embd.weight");
                if (!embd || embd->dims.size() != 2 || static_cast<long>(embd->dims[0]) != cfg_.d_model)
                    throw std::runtime_error("ov_graph: the model has no embedding table of its width");
                embedding_ = file_->matrix("token_embd.weight", cfg_.vocab, cfg_.d_model);
                warm_up();
                build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            }

            ~ov_runtime() override
            {
                // The requests and the compiled graph hold the mapping of the file of the weights, which bin_ deletes
                // after them.
                sequences_.clear();
                compiled_ = ov::CompiledModel();
            }

            const model_config& config() const override { return cfg_; }

            int create_sequence() override
            {
                std::unique_ptr<sequence> s(new sequence());
                s->request = compiled_.create_infer_request();
                if (!free_.empty())
                {
                    const int id = free_.back();
                    free_.pop_back();
                    sequences_[static_cast<size_t>(id)] = std::move(s);
                    return id;
                }
                sequences_.push_back(std::move(s));
                return static_cast<int>(sequences_.size() - 1);
            }

            void release_sequence(int seq) override
            {
                if (seq < 0 || static_cast<size_t>(seq) >= sequences_.size() || !sequences_[static_cast<size_t>(seq)])
                    return;
                held_ -= sequences_[static_cast<size_t>(seq)]->planned;
                // The request holds the memory of the keys and values: it goes with the sequence.
                sequences_[static_cast<size_t>(seq)].reset();
                free_.push_back(seq);
            }

            long sequence_length(int seq) const override { return at(seq).length; }

            void truncate_sequence(int seq, long length) override
            {
                sequence& s = at(seq);
                if (length >= s.length) return;
                if (length <= 0)
                {
                    // The plugin keeps the buffers of a cache it resets.
                    s.request.reset_state();
                    s.length = 0;
                    return;
                }
                keep(s.request, s.request, length);
                s.length = length;
                s.capacity = length;
                plan(s, length);
            }

            /*
                The plugin of the processor grows the buffers of a cache that a pass fills beyond them to twice
                the positions of the sequence after the pass, the old buffers being copied into the new ones
                before they go: a pass to length positions beyond the buffers asks for 2 length positions while
                those of the sequence remain.
            */
            bool reserve_sequence(int seq, long length) override
            {
                sequence& s = at(seq);
                const long needed = length <= s.capacity ? s.capacity : 2 * length;
                if (needed <= s.planned) return true;
                const size_t bytes = position_bytes();
                const size_t used = static_cast<size_t>(held_.load()) * bytes;
                const size_t after = used - static_cast<size_t>(s.planned) * bytes
                    + static_cast<size_t>(needed) * bytes;
                if (after > kv_budget_) return false;
                if (guard_ && !guard_(static_cast<size_t>(needed) * bytes, used)) return false;
                plan(s, needed);
                return true;
            }

            bool copy_sequence(int src, int dst, long length) override
            {
                sequence& from = at(src);
                sequence& to = at(dst);
                if (length > from.length) throw std::logic_error("ov_graph: a copy beyond the length of a sequence");
                to.request.reset_state();
                to.length = 0;
                if (length <= 0) return true;
                // The positions copied replace the buffers of dst, which remain while they are written.
                const size_t bytes = position_bytes();
                const size_t used = static_cast<size_t>(held_.load()) * bytes;
                if (used - static_cast<size_t>(to.planned) * bytes + static_cast<size_t>(length) * bytes > kv_budget_)
                    return false;
                if (guard_ && !guard_(static_cast<size_t>(length) * bytes, used)) return false;
                keep(from.request, to.request, length);
                to.length = length;
                to.capacity = length;
                plan(to, length);
                return true;
            }

            void forward(const forward_batch& batch, std::vector<float>& logits) override
            {
                const long T = static_cast<long>(batch.tokens.size());
                const long R = static_cast<long>(batch.logit_rows.size());
                const long d = cfg_.d_model;
                const long V = cfg_.vocab;
                hidden_out_.assign(static_cast<size_t>(R * d), 0.0f);
                if (T == 0)
                {
                    logits.clear();
                    return;
                }
                if (batch.hidden_only || cfg_.is_embedding()) logits.clear();
                else logits.assign(static_cast<size_t>(R * V), 0.0f);
                const bool given = !batch.positions.empty();
                if (given && static_cast<long>(batch.positions.size()) != T * forward_batch::position_components)
                    throw std::runtime_error("ov_graph: the batch gives positions for other tokens than its own");
                const size_t width = static_cast<size_t>(cfg_.input_width());
                if (batch.embeddings.size() != batch.embedding_rows.size() * width)
                    throw std::runtime_error("ov_graph: the embeddings of the batch do not match its rows");
                for (long r : batch.embedding_rows)
                    if (r < 0 || r >= T) throw std::runtime_error("ov_graph: an embedding row lies outside the batch");
                for (long r : batch.logit_rows)
                    if (r < 0 || r >= T) throw std::runtime_error("ov_graph: a row of logits lies outside the batch");
                // The segments are checked before any of them runs, so that a failure leaves every sequence as it was.
                for (const auto& seg : batch.segments)
                {
                    const sequence& s = at(seg.sequence);
                    if (seg.start < 0 || seg.count < 0 || seg.start + seg.count > T)
                        throw std::runtime_error("ov_graph: a segment lies outside the batch");
                    if (s.planned < s.length + seg.count)
                        throw std::runtime_error("ov_graph: sequence storage was not reserved");
                }
                for (const auto& seg : batch.segments)
                    if (seg.count > 0) run_segment(batch, seg, logits);
            }

            const std::vector<float>& hidden() const override { return hidden_out_; }

            size_t kv_bytes_used() const override
            {
                return static_cast<size_t>(held_.load()) * position_bytes();
            }
            size_t kv_bytes_budget() const override { return kv_budget_; }
            // Two positions of the buffers of the plugin for each position of a sequence (reserve_sequence()).
            size_t kv_bytes_per_token() const override { return 2 * position_bytes(); }

            /*
                The intermediate buffers stay with the plugin: after CompiledModel::release_memory(), the
                plugin of the processor of OpenVINO 2026.4 fails the next inference of the requests that
                hold the caches ("Memory has not been allocated").
            */
            void trim_memory() override {}
            void set_busy(bool) override {}
            size_t weight_bytes() const override { return constant_bytes_; }

            std::string description() const override
            {
                std::ostringstream o;
                o << "OpenVINO graph on " << settings_.device << " (OpenVINO " << ov_graph_version()
                  << ", keys and values in " << (settings_.kv_half || !cpu_ ? "half precision" : "8 bits") << ", "
                  << (constant_bytes_ >> 20) << " MB of weights";
                if (!bin_.path.empty()) o << ", weights mapped from their file";
                if (cpu_ && fused_attention_ == 0) o << ", attention not fused";
                o << "), " << describe(cfg_);
                return o.str();
            }

            void set_kv_guard(std::function<bool(size_t, size_t)> guard) override { guard_ = std::move(guard); }

            void set_context(long positions) override
            {
                std::vector<float> previous = cfg_.rope_freq_factors;
                select_rope_factors(cfg_, positions);
                if (cfg_.rope_freq_factors == previous) return;
                if (rope_positions_ > 0)
                {
                    cfg_.rope_freq_factors.swap(previous);
                    throw std::logic_error("ov_graph: the rotary divisors cannot change once positions are computed");
                }
            }

            double build_seconds() const override { return build_seconds_; }

        private:
            struct sequence
            {
                ov::InferRequest request;
                long length = 0;
                // The positions the buffers of the plugin hold for the sequence, and those the passes it reserved
                // make them hold at most (reserve_sequence()), which held_ counts.
                long capacity = 0;
                long planned = 0;
            };

            sequence& at(int seq)
            {
                if (seq < 0 || static_cast<size_t>(seq) >= sequences_.size() || !sequences_[static_cast<size_t>(seq)])
                    throw std::out_of_range("ov_graph: no such sequence");
                return *sequences_[static_cast<size_t>(seq)];
            }

            const sequence& at(int seq) const
            {
                return const_cast<ov_runtime*>(this)->at(seq);
            }

            static void remove_file(const std::filesystem::path& path)
            {
                std::error_code ignored;
                if (!path.empty()) std::filesystem::remove(path, ignored);
            }

            /*
                The file of the weights of the graph, which goes with the runtime, after the members declared
                after it, which map it, and also when the constructor throws once the file is written.
            */
            struct owned_file
            {
                std::filesystem::path path;
                ~owned_file() { remove_file(path); }
            };

            // The bytes of one position of the keys and values of every layer, in the format of the plugin.
            size_t position_bytes() const
            {
                // 8-bit integers carry a scale and a zero point in single precision per head and position.
                const size_t per_head = settings_.kv_half || !cpu_ ? 2 * static_cast<size_t>(cfg_.head_dim)
                    : static_cast<size_t>(cfg_.head_dim) + 8;
                return 2 * static_cast<size_t>(cfg_.n_layers) * static_cast<size_t>(cfg_.n_kv_heads) * per_head;
            }

            // Plans positions positions for s, which held_ counts.
            void plan(sequence& s, long positions)
            {
                held_ += positions - s.planned;
                s.planned = positions;
            }

            // Throws when OpenVINO finds no device settings_.device ("GPU", "GPU.1").
            void check_device()
            {
                const size_t dot = settings_.device.find('.');
                std::vector<std::string> ids;
                try
                {
                    ids = core_.get_property(settings_.device.substr(0, dot), ov::available_devices);
                }
                catch (const std::exception&)
                {
                }
                const std::string id = dot == std::string::npos ? std::string() : settings_.device.substr(dot + 1);
                if (ids.empty() || (!id.empty() && std::find(ids.begin(), ids.end(), id) == ids.end()))
                    throw std::runtime_error("ov_graph: OpenVINO finds no device " + settings_.device);
            }

            /*
                Writes the graph and its weights in settings_.weights_dir and reads them back, the weights
                mapped from their file.  A graph that cannot be written throws: kept in memory, its constants
                would hold the weights a second time beside the copy of the plugin.  The files left by a
                process that ended without deleting them, a day old, go first.
            */
            std::shared_ptr<ov::Model> spill(const std::shared_ptr<ov::Model>& model)
            {
                namespace fs = std::filesystem;
                std::error_code ec;
                const fs::path dir = fs::u8path(settings_.weights_dir);
                fs::create_directories(dir, ec);
                const auto day_ago = fs::file_time_type::clock::now() - std::chrono::hours(24);
                for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
                {
                    const std::string name = it->path().filename().u8string();
                    std::error_code time_ec;
                    if (name.compare(0, 6, "graph-") == 0 && it->last_write_time(time_ec) < day_ago && !time_ec)
                        remove_file(it->path());
                }
                // A name of its own for each graph, whatever the processes and the models.
                std::random_device random;
                const unsigned long long tag = (static_cast<unsigned long long>(random()) << 32)
                    ^ static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count());
                std::ostringstream name;
                name << "graph-" << std::hex << tag;
                const fs::path xml = dir / (name.str() + ".xml");
                const fs::path bin = dir / (name.str() + ".bin");
                bin_.path = bin;
                try
                {
                    ov::serialize(model, xml, bin);
                    std::shared_ptr<ov::Model> mapped = core_.read_model(xml, bin);
                    remove_file(xml);
                    return mapped;
                }
                catch (const std::exception& e)
                {
                    remove_file(xml);
                    remove_file(bin);
                    bin_.path.clear();
                    throw std::runtime_error(std::string("ov_graph: the graph could not be written to ")
                        + dir.u8string() + " (" + e.what() + ")");
                }
            }

            /*
                A pass of a few tokens and a step on a sequence of their own: the plugins prepare their
                kernels and buffers at the first inference of each shape, about a second on the
                processor, which the load then takes rather than the first question.
            */
            void warm_up()
            {
                const int seq = create_sequence();
                try
                {
                    forward_batch b;
                    std::vector<float> logits;
                    for (long count : { 4L, 1L })
                    {
                        b.clear();
                        b.tokens.assign(static_cast<size_t>(count), 0);
                        forward_batch::segment s;
                        s.sequence = seq;
                        s.count = count;
                        b.segments.push_back(s);
                        b.logit_rows.push_back(count - 1);
                        b.hidden_only = cfg_.is_embedding();
                        if (!reserve_sequence(seq, sequence_length(seq) + count)) break;
                        forward(b, logits);
                    }
                }
                catch (...)
                {
                    release_sequence(seq);
                    throw;
                }
                release_sequence(seq);
                // The rotary divisors of a model with LongRoPE may still follow its context (set_context()).
                rope_cos_.clear();
                rope_sin_.clear();
                rope_positions_ = 0;
            }

            /*
                Writes in each variable [1, heads, positions, head_dim] of to the first length positions of the
                same variable of from, one variable at a time, so that a single copy of a layer exists beside the
                caches.  The plugin of the processor copies a state into buffers of its own of these positions,
                and keeps the tensor it was given without reading it again: on the processor, with the versions
                of OpenVINO whose plugin does so (state_copied_at_once()), the tensors given view kept_, which goes
                once the variables are written, so that the cache is not held twice.
            */
            void keep(ov::InferRequest& from, ov::InferRequest& to, long length)
            {
                std::map<std::string, ov::VariableState> targets;
                for (ov::VariableState& state : to.query_state()) targets.emplace(state.get_name(), state);
                for (ov::VariableState& state : from.query_state())
                {
                    auto it = targets.find(state.get_name());
                    if (it == targets.end()) throw std::logic_error("ov_graph: two requests with other variables");
                    const ov::Tensor all = state.get_state();
                    const ov::Shape shape = all.get_shape();
                    if (shape.size() != 4 || static_cast<long>(shape[2]) < length)
                        throw std::logic_error("ov_graph: a variable shorter than its sequence");
                    const ov::Tensor part(all, ov::Coordinate{0, 0, 0, 0},
                        ov::Coordinate{shape[0], shape[1], static_cast<size_t>(length), shape[3]});
                    ov::Tensor copy;
                    if (views_)
                    {
                        kept_.resize(part.get_byte_size());
                        copy = ov::Tensor(all.get_element_type(), part.get_shape(), kept_.data());
                    }
                    else
                    {
                        copy = ov::Tensor(all.get_element_type(), part.get_shape());
                    }
                    part.copy_to(copy);
                    it->second.set_state(copy);
                }
                // Not numbers in half and single precision, which a plugin reading the buffer later would show.
                std::fill(kept_.begin(), kept_.end(), static_cast<uint8_t>(0xFF));
                std::vector<uint8_t>().swap(kept_);
            }

            /*
                Whether the plugin of the processor of this version of OpenVINO copies a state into its own
                buffers at once and never reads the tensor it keeps afterwards, as memory_state.cpp of its
                versions 2025.0, 2025.4 and 2026.4 does: the versions from 2025.0 to 2026.4.
            */
            static bool state_copied_at_once()
            {
                const std::string v = ov_graph_version();
                const size_t dot = v.find('.');
                if (dot == std::string::npos) return false;
                const int major = std::atoi(v.c_str());
                const int minor = std::atoi(v.c_str() + dot + 1);
                return major == 2025 || (major == 2026 && minor <= 4);
            }

            /*
                The attention nodes of the compiled graph that read the cache of their layer
                themselves: the fused node of the processor takes the index of the beams and both
                halves of the cache beside the queries, keys and values of the pass.
            */
            size_t count_fused_attention() const
            {
                size_t n = 0;
                try
                {
                    for (const auto& node : compiled_.get_runtime_model()->get_ordered_ops())
                    {
                        const auto& info = node->get_rt_info();
                        auto it = info.find("layerType");
                        if (it != info.end() && it->second.as<std::string>() == "ScaledDotProductAttention"
                            && node->get_input_size() >= 6)
                            ++n;
                    }
                }
                catch (const std::exception&)
                {
                }
                return n;
            }

            void ensure_rope(long positions)
            {
                if (positions <= rope_positions_) return;
                const long half = cfg_.rope_dims / 2;
                const long target = std::max<long>(positions, rope_positions_ + 1024);
                rope_cos_.resize(static_cast<size_t>(target * half));
                rope_sin_.resize(static_cast<size_t>(target * half));
                rope_rows(cfg_, rope_positions_, target - rope_positions_, rope_cos_.data() + rope_positions_ * half,
                    rope_sin_.data() + rope_positions_ * half);
                rope_positions_ = target;
            }

            void run_segment(const forward_batch& batch, const forward_batch::segment& seg, std::vector<float>& logits)
            {
                sequence& s = at(seg.sequence);
                const long L = seg.count;
                const long d = cfg_.d_model;
                const long half = cfg_.rope_dims / 2;
                const long rot = cfg_.rope_dims;
                const long C = forward_batch::position_components;
                const bool given = !batch.positions.empty();

                // Embeddings of the tokens, those of an encoder in their place, and the DeepStack slices.
                embeds_.resize(static_cast<size_t>(L * d));
                const float es = static_cast<float>(cfg_.embedding_scale);
                for (long j = 0; j < L; ++j)
                {
                    const int id = batch.tokens[static_cast<size_t>(seg.start + j)];
                    if (id < 0 || id >= cfg_.vocab) throw std::runtime_error("token identifier out of range");
                    float* dst = embeds_.data() + j * d;
                    qk_dequantize_row(embedding_.type, embedding_.data + static_cast<size_t>(id) * embedding_.row_bytes,
                        dst, d);
                    if (es != 1.0f) qk_scale(dst, es, d);
                }
                const long nds = cfg_.n_deepstack;
                if (nds > 0) deepstack_.assign(static_cast<size_t>(nds * L * d), 0.0f);
                for (size_t i = 0; i < batch.embedding_rows.size(); ++i)
                {
                    const long row = batch.embedding_rows[i];
                    if (row < seg.start || row >= seg.start + L) continue;
                    const float* src = batch.embeddings.data() + i * static_cast<size_t>(cfg_.input_width());
                    std::memcpy(embeds_.data() + (row - seg.start) * d, src, sizeof(float) * static_cast<size_t>(d));
                    for (long k = 0; k < nds; ++k)
                        std::memcpy(deepstack_.data() + (k * L + (row - seg.start)) * d, src + (k + 1) * d,
                            sizeof(float) * static_cast<size_t>(d));
                }

                // Cosines and sines of both halves of the rotated part, by the slot or by the given components.
                long max_pos = s.length + L - 1;
                if (given)
                    for (long j = 0; j < L * C; ++j)
                    {
                        const long p = batch.positions[static_cast<size_t>(seg.start * C + j)];
                        if (p < 0) throw std::runtime_error("ov_graph: the batch gives a negative position");
                        max_pos = std::max(max_pos, p);
                    }
                ensure_rope(max_pos + 1);
                cos_.resize(static_cast<size_t>(L * rot));
                sin_.resize(static_cast<size_t>(L * rot));
                for (long j = 0; j < L; ++j)
                {
                    float* c = cos_.data() + j * rot;
                    float* sn = sin_.data() + j * rot;
                    for (long i = 0; i < half; ++i)
                    {
                        const long component = components_[static_cast<size_t>(i)];
                        const long pos = given ? batch.positions[static_cast<size_t>((seg.start + j) * C + component)]
                            : s.length + j;
                        const size_t at_pos = static_cast<size_t>(pos * half + i);
                        c[i] = c[i + half] = rope_cos_[at_pos];
                        sn[i] = sn[i + half] = rope_sin_[at_pos];
                    }
                }
                if (cfg_.attn_temp_scale != 0.0)
                {
                    // The scale of the queries of the long positions, by the slot of each token as cpu_runtime
                    // takes it.
                    query_scale_.resize(static_cast<size_t>(L));
                    for (long j = 0; j < L; ++j)
                        query_scale_[static_cast<size_t>(j)] = static_cast<float>(1.0 + cfg_.attn_temp_scale
                            * std::log(1.0 + std::floor(static_cast<double>(s.length + j) / cfg_.attn_temp_length)));
                }

                // The rows of logits of this segment, in the order of the batch, or its last row alone.
                rows_.clear();
                targets_.clear();
                for (size_t r = 0; r < batch.logit_rows.size(); ++r)
                {
                    const long row = batch.logit_rows[r];
                    if (row < seg.start || row >= seg.start + L) continue;
                    rows_.push_back(row - seg.start);
                    targets_.push_back(static_cast<long>(r));
                }
                if (rows_.empty()) rows_.push_back(L - 1);

                const size_t Ls = static_cast<size_t>(L);
                ov::InferRequest& req = s.request;
                req.set_tensor(input_embeds, ov::Tensor(ov::element::f32, ov::Shape{1, Ls, static_cast<size_t>(d)},
                    embeds_.data()));
                req.set_tensor(input_cos, ov::Tensor(ov::element::f32, ov::Shape{1, 1, Ls, static_cast<size_t>(rot)},
                    cos_.data()));
                req.set_tensor(input_sin, ov::Tensor(ov::element::f32, ov::Shape{1, 1, Ls, static_cast<size_t>(rot)},
                    sin_.data()));
                req.set_tensor(input_beam, ov::Tensor(ov::element::i32, ov::Shape{1}, &beam_));
                req.set_tensor(input_rows, ov::Tensor(ov::element::i64, ov::Shape{rows_.size()}, rows_.data()));
                if (nds > 0)
                    req.set_tensor(input_deepstack, ov::Tensor(ov::element::f32,
                        ov::Shape{static_cast<size_t>(nds), 1, Ls, static_cast<size_t>(d)}, deepstack_.data()));
                if (cfg_.attn_temp_scale != 0.0)
                    req.set_tensor(input_query_scale, ov::Tensor(ov::element::f32, ov::Shape{1, 1, Ls, 1},
                        query_scale_.data()));
                if (explicit_mask_)
                {
                    mask_.assign(static_cast<size_t>(s.length + L), 1);
                    req.set_tensor(input_mask, ov::Tensor(ov::element::i64, ov::Shape{1, mask_.size()}, mask_.data()));
                }
                req.infer();
                // The plugin grew the buffers of the cache when the pass filled them (reserve_sequence()).
                if (s.length + L > s.capacity) s.capacity = 2 * (s.length + L);
                if (s.capacity > s.planned) plan(s, s.capacity);
                s.length += L;

                const ov::Tensor hidden = req.get_tensor(output_hidden);
                const float* h = static_cast<const float*>(hidden.data());
                for (size_t i = 0; i < targets_.size(); ++i)
                    std::memcpy(hidden_out_.data() + targets_[i] * d, h + i * static_cast<size_t>(d),
                        sizeof(float) * static_cast<size_t>(d));
                if (!logits.empty() && !targets_.empty())
                {
                    const ov::Tensor out = req.get_tensor(output_logits);
                    const float* lg = static_cast<const float*>(out.data());
                    const size_t V = static_cast<size_t>(cfg_.vocab);
                    for (size_t i = 0; i < targets_.size(); ++i)
                        std::memcpy(logits.data() + static_cast<size_t>(targets_[i]) * V, lg + i * V,
                            sizeof(float) * V);
                }
            }

            std::shared_ptr<model_file> file_;
            model_config cfg_;
            ov_graph_settings settings_;
            size_t kv_budget_ = 0;
            bool cpu_ = true;
            bool explicit_mask_ = false;
            // A cut or a copy gives the plugin views of kept_ (keep()).
            bool views_ = false;
            // The file of the weights of the graph, mapped while the graph runs (spill()), declared before the
            // members that map it.
            owned_file bin_;
            ov::Core core_;
            ov::CompiledModel compiled_;
            size_t constant_bytes_ = 0;
            size_t fused_attention_ = 0;
            double build_seconds_ = 0;
            weight_matrix embedding_;
            std::vector<std::unique_ptr<sequence>> sequences_;
            std::vector<int> free_;
            // The positions planned for all the sequences, which the statistics read from other threads.
            std::atomic<long> held_{0};
            // The positions a truncation or a copy writes back (keep()).
            std::vector<uint8_t> kept_;
            std::function<bool(size_t, size_t)> guard_;
            std::vector<int> components_;
            std::vector<float> rope_cos_, rope_sin_;
            long rope_positions_ = 0;
            // The inputs of a pass, which the tensors of its request point to while it runs.
            std::vector<float> embeds_, deepstack_, cos_, sin_, query_scale_;
            std::vector<int64_t> rows_, mask_;
            std::vector<long> targets_;
            int32_t beam_ = 0;
            std::vector<float> hidden_out_;
        };
    }

    std::string ov_graph_unsupported(const model_config& c, const model_file& f)
    {
        if (c.is_moe()) return "the graph holds dense models only";
        if (c.rope_dims <= 0 || c.rope_dims % 2 != 0) return "the rotary dimensions are not even";
        if (c.head_dim != c.head_dim_v) return "the keys and values of the model differ in width";
        if (c.n_deepstack > c.n_layers) return "the model declares more DeepStack layers than it has";
        if (!f.has("token_embd.weight")) return "the model has no embedding table";
        // The formats the graph keeps (weight_converter), and those it rounds again to 8-bit integers.
        auto takes = [](const tensor_view* t) {
            const bool own = t->type == dlib::ggml_type::F16 || t->type == dlib::ggml_type::BF16
                || t->type == dlib::ggml_type::F32 || t->type == dlib::ggml_type::Q4_0
                || t->type == dlib::ggml_type::Q4_1 || t->type == dlib::ggml_type::Q4_K
                || t->type == dlib::ggml_type::Q5_0 || t->type == dlib::ggml_type::Q8_0;
            return own || (qk_supported(t->type) && !t->dims.empty() && t->dims[0] % group_size == 0);
        };
        if (!c.is_embedding())
        {
            const char* const output = c.tied_embeddings ? "token_embd.weight" : "output.weight";
            const tensor_view* t = f.find(output);
            if (!t) return "the model has no output matrix";
            if (!takes(t)) return std::string("the format of ") + output + " is not supported by the graph";
        }
        const char* const required[] = { "attn_norm.weight", "attn_output.weight", "ffn_norm.weight",
            "ffn_down.weight" };
        for (long i = 0; i < c.n_layers; ++i)
        {
            const std::string p = "blk." + std::to_string(i) + ".";
            for (const char* r : required)
                if (!f.has(p + r)) return "the model lacks the tensor " + p + r;
            const char* const matrices[] = { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight",
                "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
            for (const char* m : matrices)
            {
                const tensor_view* t = f.find(p + m);
                if (t && !takes(t)) return "the format of " + p + m + " is not supported by the graph";
            }
        }
        return std::string();
    }

    std::unique_ptr<ov_graph_runtime> make_ov_graph_runtime(std::shared_ptr<model_file> file,
        const model_config& config, const ov_graph_settings& settings, size_t kv_budget_bytes)
    {
        const std::string why = ov_graph_unsupported(config, *file);
        if (!why.empty()) throw std::runtime_error("ov_graph: " + why);
        return std::unique_ptr<ov_graph_runtime>(new ov_runtime(std::move(file), config, settings, kv_budget_bytes));
    }

    std::vector<std::string> ov_graph_devices()
    {
        ov::Core core;
        return core.get_available_devices();
    }

    bool ov_graph_processor(std::string& name)
    {
        try
        {
            ov::Core core;
            name = core.get_property("CPU", ov::device::full_name);
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    std::string ov_graph_version()
    {
        const ov::Version v = ov::get_openvino_version();
        std::string s = v.buildNumber ? v.buildNumber : "";
        // "2026.4.1-22982-..." gives "2026.4.1".
        const size_t dash = s.find('-');
        return dash == std::string::npos ? s : s.substr(0, dash);
    }

}
