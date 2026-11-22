// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CPU_GGUF_CPP_
#define DLIB_CPU_GGUF_CPP_

#include "cpu_gguf.h"
#include "gguf_blocks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dlib
{
    namespace cpu
    {

    // ----------------------------------------------------------------------------------------

        void gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && rows > 0 && cols > 0);
            DLIB_CASSERT(static_cast<long long>(rows) <= std::numeric_limits<int>::max()
                && static_cast<long long>(cols) <= std::numeric_limits<int>::max());
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights));
            DLIB_CASSERT(src.size() % static_cast<size_t>(cols) == 0);
            const long n = static_cast<long>(src.size() / static_cast<size_t>(cols));
            DLIB_CASSERT(dest.size() == static_cast<size_t>(n) * static_cast<size_t>(rows));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(rows) * row_bytes);
            if (n == 0) return;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            const float* x = src.host();
            float* y = dest.host_write_only();
            const long groups = cols / 32;
            float v[32];
            std::vector<float> acc(static_cast<size_t>(n));
            for (long r = 0; r < rows; ++r)
            {
                const uint8_t* wr = w + static_cast<size_t>(r) * row_bytes;
                for (auto& a : acc) a = 0;
                for (long g = 0; g < groups; ++g)
                {
                    gguf_blocks::decode_group(type, wr, g, v);
                    for (long t = 0; t < n; ++t)
                    {
                        const float* xt = x + static_cast<size_t>(t) * static_cast<size_t>(cols) + g * 32;
                        float s = 0;
                        for (int i = 0; i < 32; ++i) s += v[i] * xt[i];
                        acc[static_cast<size_t>(t)] += s;
                    }
                }
                for (long t = 0; t < n; ++t)
                    y[static_cast<size_t>(t) * static_cast<size_t>(rows) + r] = acc[static_cast<size_t>(t)];
            }
        }

    // ----------------------------------------------------------------------------------------

        void gguf_dequantize (
            tensor& dest,
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && first >= 0 && count >= 0 && cols > 0);
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, weights));
            DLIB_CASSERT(dest.size() == static_cast<size_t>(count) * static_cast<size_t>(cols));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(first + count) * row_bytes);
            if (count == 0) return;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            float* d = dest.host_write_only();
            const long groups = cols / 32;
            for (long r = 0; r < count; ++r)
            {
                const uint8_t* wr = w + static_cast<size_t>(first + r) * row_bytes;
                float* out = d + static_cast<size_t>(r) * static_cast<size_t>(cols);
                for (long g = 0; g < groups; ++g) gguf_blocks::decode_group(type, wr, g, out + g * 32);
            }
        }

    // ----------------------------------------------------------------------------------------

        void rms_norm_rows (
            tensor& dest,
            const tensor& src,
            const tensor& gamma,
            float eps
        )
        {
            const size_t width = gamma.size();
            DLIB_CASSERT(width > 0 && src.size() % width == 0 && dest.size() == src.size() && eps >= 0);
            const size_t rows = src.size() / width;
            const float* x = src.host();
            const float* g = gamma.host();
            // dest may be src: each value is read before it is written.
            float* y = is_same_object(dest, src) ? dest.host() : dest.host_write_only();
            for (size_t r = 0; r < rows; ++r)
            {
                const float* xr = x + r * width;
                float* yr = y + r * width;
                float s = 0;
                for (size_t i = 0; i < width; ++i) s += xr[i] * xr[i];
                const float scale = 1.0f / std::sqrt(s / static_cast<float>(width) + eps);
                for (size_t i = 0; i < width; ++i) yr[i] = xr[i] * scale * g[i];
            }
        }

    // ----------------------------------------------------------------------------------------

        void add_rms_norm_rows (
            tensor& dest,
            tensor& x,
            const tensor& addend,
            float factor,
            const tensor& gamma,
            float eps
        )
        {
            const size_t width = gamma.size();
            DLIB_CASSERT(width > 0 && x.size() % width == 0 && dest.size() == x.size() && addend.size() == x.size()
                && eps >= 0);
            DLIB_CASSERT(!is_same_object(dest, x) && !is_same_object(dest, addend) && !is_same_object(x, addend));
            const size_t rows = x.size() / width;
            float* xs = x.host();
            const float* a = addend.host();
            const float* g = gamma.host();
            float* y = dest.host_write_only();
            for (size_t r = 0; r < rows; ++r)
            {
                float* xr = xs + r * width;
                const float* ar = a + r * width;
                float* yr = y + r * width;
                float s = 0;
                for (size_t i = 0; i < width; ++i)
                {
                    xr[i] += factor * ar[i];
                    s += xr[i] * xr[i];
                }
                const float scale = 1.0f / std::sqrt(s / static_cast<float>(width) + eps);
                for (size_t i = 0; i < width; ++i) yr[i] = xr[i] * scale * g[i];
            }
        }

    // ----------------------------------------------------------------------------------------

        void silu_multiply (
            tensor& gate,
            const tensor& up
        )
        {
            DLIB_CASSERT(gate.size() == up.size() && !is_same_object(gate, up));
            float* g = gate.host();
            const float* u = up.host();
            for (size_t i = 0; i < gate.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        }

    // ----------------------------------------------------------------------------------------

        void rope_rows (
            tensor& rows,
            const tensor& angles,
            long heads,
            long head_dim,
            long rope_dims,
            bool neox
        )
        {
            DLIB_CASSERT(heads > 0 && head_dim > 0 && rope_dims > 0 && rope_dims <= head_dim && rope_dims % 2 == 0);
            DLIB_CASSERT(angles.size() % static_cast<size_t>(rope_dims) == 0);
            const size_t n = angles.size() / static_cast<size_t>(rope_dims);
            const size_t width = static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
            DLIB_CASSERT(rows.size() == n * width && !is_same_object(rows, angles));
            if (n == 0) return;
            const long half = rope_dims / 2;
            const float* a = angles.host();
            float* x = rows.host();
            for (size_t t = 0; t < n; ++t)
            {
                const float* c = a + t * static_cast<size_t>(rope_dims);
                const float* s = c + half;
                for (long h = 0; h < heads; ++h)
                {
                    float* r = x + t * width + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
                    for (long i = 0; i < half; ++i)
                    {
                        const long i0 = neox ? i : 2 * i;
                        const long i1 = neox ? i + half : 2 * i + 1;
                        const float x0 = r[i0], x1 = r[i1];
                        r[i0] = x0 * c[i] - x1 * s[i];
                        r[i1] = x0 * s[i] + x1 * c[i];
                    }
                }
            }
        }

        void rope_rows (
            tensor& first_rows,
            long first_heads,
            tensor& second_rows,
            long second_heads,
            const tensor& angles,
            long head_dim,
            long rope_dims,
            bool neox
        )
        {
            DLIB_CASSERT(!is_same_object(first_rows, second_rows));
            rope_rows(first_rows, angles, first_heads, head_dim, rope_dims, neox);
            rope_rows(second_rows, angles, second_heads, head_dim, rope_dims, neox);
        }

    // ----------------------------------------------------------------------------------------

        namespace
        {
            // The 32-bit integers whose bits the floats of t hold.
            const int32_t* kv_integers(const tensor& t)
            {
                return reinterpret_cast<const int32_t*>(t.host());
            }

            // The whole pages a tensor holds.
            size_t kv_page_count(const tensor& pages, const kv_page_layout& layout)
            {
                return pages.size() * sizeof(float) / layout.page_bytes();
            }

            // The values of row r of a slab of keys or values, as floats.
            void kv_read_row(const uint8_t* slab, long r, const kv_page_layout& layout, float* row)
            {
                const long hd = layout.head_dim;
                const uint8_t* data = slab + static_cast<size_t>(r) * layout.data_bytes();
                if (!layout.q8)
                {
                    for (long i = 0; i < hd; ++i) row[i] = gguf_blocks::read_half(data + 2 * i);
                    return;
                }
                const uint8_t* scales = slab + static_cast<size_t>(layout.block_positions) * layout.data_bytes()
                    + static_cast<size_t>(r) * layout.scale_bytes();
                for (long g = 0; g < hd / 32; ++g)
                {
                    const float d = gguf_blocks::read_half(scales + 2 * g);
                    for (long i = 0; i < 32; ++i)
                        row[g * 32 + i] = d * static_cast<float>(static_cast<int8_t>(data[g * 32 + i]));
                }
            }

            void kv_write_half(uint8_t* p, uint16_t h)
            {
                std::memcpy(p, &h, sizeof(h));
            }
        }

        void paged_kv_store (
            tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& keys,
            const tensor& values,
            const tensor& slots
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            const size_t n = slots.size();
            const long hd = layout.head_dim;
            const size_t width = static_cast<size_t>(layout.heads) * static_cast<size_t>(hd);
            DLIB_CASSERT(keys.size() == n * width && values.size() == n * width);
            DLIB_CASSERT(!is_same_object(pages, keys) && !is_same_object(pages, values)
                && !is_same_object(pages, slots));
            if (n == 0) return;
            const size_t count = kv_page_count(pages, layout);
            const int32_t* s = kv_integers(slots);
            uint8_t* base = reinterpret_cast<uint8_t*>(pages.host());
            const float* k = keys.host();
            const float* v = values.host();
            std::vector<float> row(static_cast<size_t>(hd));
            for (size_t t = 0; t < n; ++t)
            {
                DLIB_CASSERT(s[t] >= 0 && static_cast<size_t>(s[t] / layout.block_positions) < count);
                const size_t page = static_cast<size_t>(s[t] / layout.block_positions);
                const size_t r = static_cast<size_t>(s[t] % layout.block_positions);
                for (long kind = 0; kind < 2; ++kind)
                    for (long h = 0; h < layout.heads; ++h)
                    {
                        const float* src = (kind == 0 ? k : v) + t * width + static_cast<size_t>(h) * hd;
                        uint8_t* slab = base + page * layout.page_bytes() + layout.slab_offset(kind, layer, h);
                        uint8_t* data = slab + r * layout.data_bytes();
                        if (!layout.q8)
                        {
                            for (long i = 0; i < hd; ++i) kv_write_half(data + 2 * i, kv_pages::float_to_half(src[i]));
                            continue;
                        }
                        std::copy(src, src + hd, row.begin());
                        if (kind == 0) kv_pages::rotate_row(row.data(), hd);
                        uint8_t* scales = slab + static_cast<size_t>(layout.block_positions) * layout.data_bytes()
                            + r * layout.scale_bytes();
                        for (long g = 0; g < hd / 32; ++g)
                        {
                            const float* x = row.data() + g * 32;
                            float amax = 0;
                            for (long i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
                            const uint16_t stored = kv_pages::q8_scale(amax);
                            const float d = gguf_blocks::half_to_float(stored);
                            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
                            for (long i = 0; i < 32; ++i)
                                data[g * 32 + i] = static_cast<uint8_t>(kv_pages::q8_level(x[i], inv));
                            kv_write_half(scales + 2 * g, stored);
                        }
                    }
            }
        }

        void paged_kv_gather (
            tensor& dest,
            const tensor& pages,
            const kv_page_layout& layout,
            const tensor& slots
        )
        {
            DLIB_CASSERT(layout.valid());
            const size_t n = slots.size();
            const size_t per_token = 2 * static_cast<size_t>(layout.layers) * static_cast<size_t>(layout.heads)
                * layout.row_bytes();
            DLIB_CASSERT(dest.size() * sizeof(float) >= n * per_token);
            DLIB_CASSERT(!is_same_object(dest, pages) && !is_same_object(dest, slots));
            if (n == 0) return;
            const size_t count = kv_page_count(pages, layout);
            const int32_t* s = kv_integers(slots);
            const uint8_t* base = reinterpret_cast<const uint8_t*>(pages.host());
            uint8_t* out = reinterpret_cast<uint8_t*>(dest.host());
            for (size_t t = 0; t < n; ++t)
            {
                DLIB_CASSERT(s[t] >= 0 && static_cast<size_t>(s[t] / layout.block_positions) < count);
                const size_t page = static_cast<size_t>(s[t] / layout.block_positions);
                const size_t r = static_cast<size_t>(s[t] % layout.block_positions);
                for (long kind = 0; kind < 2; ++kind)
                    for (long l = 0; l < layout.layers; ++l)
                        for (long h = 0; h < layout.heads; ++h)
                        {
                            const uint8_t* slab = base + page * layout.page_bytes() + layout.slab_offset(kind, l, h);
                            std::memcpy(out, slab + r * layout.data_bytes(), layout.data_bytes());
                            out += layout.data_bytes();
                            if (!layout.q8) continue;
                            std::memcpy(out, slab + static_cast<size_t>(layout.block_positions) * layout.data_bytes()
                                + r * layout.scale_bytes(), layout.scale_bytes());
                            out += layout.scale_bytes();
                        }
            }
        }

        namespace
        {
            /*
                The attention of one token (tt::paged_attention()): queries holds the q_heads
                rows of its queries, table the pages of the blocks of its sequence, of which it
                attends the positions [0, p]; out receives its q_heads rows.
            */
            class token_attention
            {
            public:
                token_attention(const tensor& pages, const kv_page_layout& layout, long layer, long q_heads,
                    float scale)
                    : layout_(layout), layer_(layer), q_heads_(q_heads), scale_(scale),
                    base_(reinterpret_cast<const uint8_t*>(pages.host())), count_(kv_page_count(pages, layout)),
                    q_(static_cast<size_t>(layout.head_dim)), row_(static_cast<size_t>(layout.head_dim))
                {
                }

                void operator()(const float* queries, long p, const int32_t* table, float* out)
                {
                    const long hd = layout_.head_dim;
                    const long B = layout_.block_positions;
                    const long rep = q_heads_ / layout_.heads;
                    // The pages of the positions of the token, checked once.
                    keys_.assign(static_cast<size_t>(p / B + 1), nullptr);
                    values_.assign(keys_.size(), nullptr);
                    a_.resize(static_cast<size_t>(p + 1));
                    for (long h = 0; h < q_heads_; ++h)
                    {
                        const long g = h / rep;
                        for (size_t b = 0; b < keys_.size(); ++b)
                        {
                            const int32_t page = table[b];
                            DLIB_CASSERT(page >= 0 && static_cast<size_t>(page) < count_);
                            const uint8_t* start = base_ + static_cast<size_t>(page) * layout_.page_bytes();
                            keys_[b] = start + layout_.slab_offset(0, layer_, g);
                            values_[b] = start + layout_.slab_offset(1, layer_, g);
                        }
                        const float* src = queries + static_cast<size_t>(h) * hd;
                        for (long i = 0; i < hd; ++i) q_[static_cast<size_t>(i)] = src[i] * scale_;
                        if (layout_.q8) kv_pages::rotate_row(q_.data(), hd);
                        float mx = -std::numeric_limits<float>::infinity();
                        for (long j = 0; j <= p; ++j)
                        {
                            kv_read_row(keys_[static_cast<size_t>(j / B)], j % B, layout_, row_.data());
                            float d = 0;
                            for (long i = 0; i < hd; ++i)
                                d += q_[static_cast<size_t>(i)] * row_[static_cast<size_t>(i)];
                            a_[static_cast<size_t>(j)] = d;
                            mx = std::max(mx, d);
                        }
                        float sum = 0;
                        for (long j = 0; j <= p; ++j)
                        {
                            a_[static_cast<size_t>(j)] = std::exp(a_[static_cast<size_t>(j)] - mx);
                            sum += a_[static_cast<size_t>(j)];
                        }
                        float* dst = out + static_cast<size_t>(h) * hd;
                        std::fill(dst, dst + hd, 0.0f);
                        for (long j = 0; j <= p; ++j)
                        {
                            kv_read_row(values_[static_cast<size_t>(j / B)], j % B, layout_, row_.data());
                            const float w = a_[static_cast<size_t>(j)];
                            for (long i = 0; i < hd; ++i) dst[i] += w * row_[static_cast<size_t>(i)];
                        }
                        const float inv = 1.0f / sum;
                        for (long i = 0; i < hd; ++i) dst[i] *= inv;
                    }
                }

            private:
                const kv_page_layout& layout_;
                long layer_;
                long q_heads_;
                float scale_;
                const uint8_t* base_;
                size_t count_;
                std::vector<float> q_, row_, a_;
                std::vector<const uint8_t*> keys_, values_;
            };
        }

        void paged_attention (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tokens,
            const tensor& tables,
            long span
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            const long hd = layout.head_dim;
            DLIB_CASSERT(hd == 32 || hd == 64 || hd == 96 || hd == 128 || hd == 256);
            DLIB_CASSERT(q_heads > 0 && q_heads % layout.heads == 0 && tokens.size() % 2 == 0);
            const size_t n = tokens.size() / 2;
            const size_t width = static_cast<size_t>(q_heads) * static_cast<size_t>(hd);
            DLIB_CASSERT(queries.size() == n * width && out.size() == n * width);
            DLIB_CASSERT(!is_same_object(out, queries) && !is_same_object(out, pages) && !is_same_object(out, tokens)
                && !is_same_object(out, tables));
            if (n == 0) return;
            const long B = layout.block_positions;
            const int32_t* tk = kv_integers(tokens);
            const int32_t* tb = kv_integers(tables);
            const float* qs = queries.host();
            float* o = out.host();
            token_attention attend(pages, layout, layer, q_heads, scale);
            for (size_t t = 0; t < n; ++t)
            {
                const long p = tk[2 * t];
                const long f = tk[2 * t + 1];
                DLIB_CASSERT(p >= 0 && p < span && f >= 0 && static_cast<size_t>(f + p / B) < tables.size());
                attend(qs + t * width, p, tb + f, o + t * width);
            }
        }

        void paged_attention_rows (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tables,
            long first_entry,
            long first_position
        )
        {
            DLIB_CASSERT(layout.valid() && 0 <= layer && layer < layout.layers);
            DLIB_CASSERT(q_heads > 0 && q_heads % layout.heads == 0 && first_entry >= 0 && first_position >= 0);
            const size_t width = static_cast<size_t>(q_heads) * static_cast<size_t>(layout.head_dim);
            DLIB_CASSERT(queries.size() % width == 0 && out.size() == queries.size());
            DLIB_CASSERT(!is_same_object(out, queries) && !is_same_object(out, pages) && !is_same_object(out, tables));
            const size_t n = queries.size() / width;
            if (n == 0) return;
            DLIB_CASSERT(static_cast<size_t>(first_entry) + (static_cast<size_t>(first_position) + n - 1)
                / static_cast<size_t>(layout.block_positions) < tables.size());
            const int32_t* table = kv_integers(tables) + first_entry;
            const float* qs = queries.host();
            float* o = out.host();
            token_attention attend(pages, layout, layer, q_heads, scale);
            for (size_t t = 0; t < n; ++t)
                attend(qs + t * width, first_position + static_cast<long>(t), table, o + t * width);
        }

    // ----------------------------------------------------------------------------------------

        void expert_routes (
            tensor& picks,
            tensor& weights,
            const tensor& logits,
            const tensor& bias,
            long top_k,
            bool normalize,
            float scale
        )
        {
            const long experts = static_cast<long>(bias.size());
            DLIB_CASSERT(experts > 0 && logits.size() % static_cast<size_t>(experts) == 0);
            DLIB_CASSERT(top_k > 0 && top_k <= experts && top_k <= 64);
            const size_t n = logits.size() / static_cast<size_t>(experts);
            DLIB_CASSERT(picks.size() == n * static_cast<size_t>(top_k) && weights.size() == picks.size());
            DLIB_CASSERT(!is_same_object(picks, weights) && !is_same_object(picks, logits)
                && !is_same_object(weights, logits) && !is_same_object(picks, bias)
                && !is_same_object(weights, bias));
            if (n == 0) return;
            const float* l = logits.host();
            const float* b = bias.host();
            float* p = picks.host_write_only();
            float* w = weights.host_write_only();
            std::vector<float> scores(static_cast<size_t>(experts)), ranked(static_cast<size_t>(experts));
            int32_t chosen[64];
            for (size_t t = 0; t < n; ++t)
            {
                const float* row = l + t * static_cast<size_t>(experts);
                for (long e = 0; e < experts; ++e)
                {
                    const float s = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(row[e]))));
                    scores[static_cast<size_t>(e)] = s;
                    ranked[static_cast<size_t>(e)] = s + b[e];
                }
                // The rank of an expert is the number of experts before it, as the kernel of the device counts it.
                for (long k = 0; k < top_k; ++k) chosen[k] = 0;
                for (long e = 0; e < experts; ++e)
                {
                    const float mine = ranked[static_cast<size_t>(e)];
                    long rank = 0;
                    for (long o = 0; o < experts; ++o)
                    {
                        const float other = ranked[static_cast<size_t>(o)];
                        if (other > mine || (other == mine && o < e)) ++rank;
                    }
                    if (rank < top_k) chosen[rank] = static_cast<int32_t>(e);
                }
                float selected = 0;
                for (long k = 0; k < top_k; ++k) selected += scores[static_cast<size_t>(chosen[k])];
                for (long k = 0; k < top_k; ++k)
                {
                    float share = scores[static_cast<size_t>(chosen[k])];
                    if (normalize) share /= selected + 1e-20f;
                    const size_t at = t * static_cast<size_t>(top_k) + static_cast<size_t>(k);
                    std::memcpy(p + at, &chosen[k], sizeof(int32_t));
                    w[at] = share * scale;
                }
            }
        }

    // ----------------------------------------------------------------------------------------

        void gguf_matvec_picked (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes,
            const tensor& picks,
            long per_row
        )
        {
            DLIB_CASSERT(gguf_blocks::known_format(type) && rows > 0 && cols > 0 && per_row > 0);
            DLIB_CASSERT(static_cast<long long>(rows) <= std::numeric_limits<int>::max()
                && static_cast<long long>(cols) <= std::numeric_limits<int>::max());
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights) && !is_same_object(dest, picks));
            const size_t count = picks.size();
            DLIB_CASSERT(count % static_cast<size_t>(per_row) == 0);
            DLIB_CASSERT(src.size() == count / static_cast<size_t>(per_row) * static_cast<size_t>(cols));
            DLIB_CASSERT(dest.size() == count * static_cast<size_t>(rows));
            if (count == 0) return;
            const size_t matrix_bytes = static_cast<size_t>(rows) * row_bytes;
            const size_t matrices = weights.size() * sizeof(float) / matrix_bytes;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            const float* x = src.host();
            const float* pk = picks.host();
            float* y = dest.host_write_only();
            const long groups = cols / 32;
            float v[32];
            for (size_t j = 0; j < count; ++j)
            {
                int32_t pick;
                std::memcpy(&pick, pk + j, sizeof(pick));
                DLIB_CASSERT(pick >= 0 && static_cast<size_t>(pick) < matrices);
                const uint8_t* m = w + static_cast<size_t>(pick) * matrix_bytes;
                const float* xj = x + j / static_cast<size_t>(per_row) * static_cast<size_t>(cols);
                for (long r = 0; r < rows; ++r)
                {
                    const uint8_t* wr = m + static_cast<size_t>(r) * row_bytes;
                    float acc = 0;
                    for (long g = 0; g < groups; ++g)
                    {
                        gguf_blocks::decode_group(type, wr, g, v);
                        const float* xg = xj + g * 32;
                        float s = 0;
                        for (int i = 0; i < 32; ++i) s += v[i] * xg[i];
                        acc += s;
                    }
                    y[j * static_cast<size_t>(rows) + static_cast<size_t>(r)] = acc;
                }
            }
        }

    // ----------------------------------------------------------------------------------------

        void squared_relu (
            tensor& x
        )
        {
            float* p = x.host();
            for (size_t i = 0; i < x.size(); ++i)
            {
                const float r = p[i] > 0.0f ? p[i] : 0.0f;
                p[i] = r * r;
            }
        }

    // ----------------------------------------------------------------------------------------

        void mix_rows (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            long per_row
        )
        {
            DLIB_CASSERT(per_row > 0 && weights.size() % static_cast<size_t>(per_row) == 0);
            const size_t n = weights.size() / static_cast<size_t>(per_row);
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights));
            if (n == 0)
            {
                DLIB_CASSERT(dest.size() == 0 && src.size() == 0);
                return;
            }
            DLIB_CASSERT(dest.size() % n == 0);
            const size_t width = dest.size() / n;
            DLIB_CASSERT(src.size() == weights.size() * width);
            const float* s = src.host();
            const float* w = weights.host();
            float* d = dest.host_write_only();
            for (size_t t = 0; t < n; ++t)
                for (size_t i = 0; i < width; ++i)
                {
                    float sum = 0;
                    for (size_t j = 0; j < static_cast<size_t>(per_row); ++j)
                    {
                        const size_t row = t * static_cast<size_t>(per_row) + j;
                        sum += w[row] * s[row * width + i];
                    }
                    d[t * width + i] = sum;
                }
        }

    // ----------------------------------------------------------------------------------------

    }
}

#endif // DLIB_CPU_GGUF_CPP_
