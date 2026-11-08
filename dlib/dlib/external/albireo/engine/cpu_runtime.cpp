// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cpu_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace albireo
{

    namespace
    {
        float yarn_corr_dim(long n_dims, double n_ctx_orig, double n_rot, double base)
        {
            return static_cast<float>(n_dims * std::log(n_ctx_orig / (n_rot * 2.0 * 3.14159265358979323846))
                / (2.0 * std::log(base)));
        }

        float yarn_ramp(float low, float high, long i0)
        {
            const float y = (static_cast<float>(i0 / 2) - low) / std::max(0.001f, high - low);
            return 1.0f - std::min(1.0f, std::max(0.0f, y));
        }

        // The rows [first, first + count) of a matrix, as a matrix of their own in its memory.
        weight_matrix row_slice(const weight_matrix& w, long first, long count)
        {
            weight_matrix s = w;
            s.data = w.data + static_cast<size_t>(first) * w.row_bytes;
            s.rows = count;
            return s;
        }
    }

    cpu_runtime::cpu_runtime(std::shared_ptr<model_file> file, const model_config& config, thread_pool& pool,
        size_t kv_budget_bytes, long kv_block_positions, kv_format kv_type)
        : file_(std::move(file)), cfg_(config), pool_(pool),
          kv_(config.n_layers, config.n_kv_heads, config.head_dim, kv_block_positions, kv_budget_bytes, kv_type)
    {
        const model_file& f = *file_;
        const long d = cfg_.d_model;
        const long qd = cfg_.q_dim();
        const long kvd = cfg_.kv_dim();
        const long hd = cfg_.head_dim;

        layers_.resize(static_cast<size_t>(cfg_.n_layers));
        for (long i = 0; i < cfg_.n_layers; ++i)
        {
            const std::string p = "blk." + std::to_string(i) + ".";
            layer& L = layers_[static_cast<size_t>(i)];
            L.attn_norm = f.floats(p + "attn_norm.weight", static_cast<size_t>(d));
            if (cfg_.fused_qkv)
            {
                // Three views of one matrix: every product still goes through product().
                const weight_matrix qkv = f.matrix(p + "attn_qkv.weight", qd + 2 * kvd, d);
                L.wq = row_slice(qkv, 0, qd);
                L.wk = row_slice(qkv, qd, kvd);
                L.wv = row_slice(qkv, qd + kvd, kvd);
            }
            else
            {
                L.wq = f.matrix(p + "attn_q.weight", qd, d);
                L.wk = f.matrix(p + "attn_k.weight", kvd, d);
                L.wv = f.matrix(p + "attn_v.weight", kvd, d);
            }
            L.wo = f.matrix(p + "attn_output.weight", d, qd);
            f.optional_floats(p + "attn_q.bias", static_cast<size_t>(qd), L.bq);
            f.optional_floats(p + "attn_k.bias", static_cast<size_t>(kvd), L.bk);
            f.optional_floats(p + "attn_v.bias", static_cast<size_t>(kvd), L.bv);
            f.optional_floats(p + "attn_q_norm.weight", static_cast<size_t>(hd), L.q_norm);
            f.optional_floats(p + "attn_k_norm.weight", static_cast<size_t>(hd), L.k_norm);
            // The forward pass adds the three biases together and normalizes queries and keys together.
            if (L.bq.empty() != L.bk.empty() || L.bq.empty() != L.bv.empty())
                throw std::runtime_error("layer " + std::to_string(i) + " holds some of the attn_q, attn_k and "
                    "attn_v biases without the others");
            if (L.q_norm.empty() != L.k_norm.empty())
                throw std::runtime_error("layer " + std::to_string(i) + " holds one of attn_q_norm and attn_k_norm "
                    "without the other");
            L.ffn_norm = f.floats(p + "ffn_norm.weight", static_cast<size_t>(d));
            if (cfg_.is_moe())
            {
                const long n_exp = cfg_.n_experts;
                const long df = cfg_.d_ffn_expert;
                L.router = f.matrix(p + "ffn_gate_inp.weight", n_exp, d);
                for (long e = 0; e < n_exp; ++e)
                {
                    L.e_gate.push_back(f.expert_matrix(p + "ffn_gate_exps.weight", df, d, e, n_exp));
                    L.e_up.push_back(f.expert_matrix(p + "ffn_up_exps.weight", df, d, e, n_exp));
                    L.e_down.push_back(f.expert_matrix(p + "ffn_down_exps.weight", d, df, e, n_exp));
                }
            }
            else if (cfg_.fused_gate_up)
            {
                // The rows of the gate, then those of the up projection.
                const weight_matrix gate_up = f.matrix(p + "ffn_up.weight", 2 * cfg_.d_ffn, d);
                L.w_gate = row_slice(gate_up, 0, cfg_.d_ffn);
                L.w_up = row_slice(gate_up, cfg_.d_ffn, cfg_.d_ffn);
                L.w_down = f.matrix(p + "ffn_down.weight", d, cfg_.d_ffn);
            }
            else
            {
                L.w_gate = f.matrix(p + "ffn_gate.weight", cfg_.d_ffn, d);
                L.w_up = f.matrix(p + "ffn_up.weight", cfg_.d_ffn, d);
                L.w_down = f.matrix(p + "ffn_down.weight", d, cfg_.d_ffn);
            }
        }
        output_norm_ = f.floats("output_norm.weight", static_cast<size_t>(d));
        token_embd_ = f.matrix("token_embd.weight", cfg_.vocab, d);
        output_ = cfg_.tied_embeddings ? token_embd_ : f.matrix("output.weight", cfg_.vocab, d);

        const long rep = cfg_.n_heads / cfg_.n_kv_heads;
        const size_t rows = static_cast<size_t>(std::max<long>(1, tile_rows / rep) * rep);
        attention_work_.resize(static_cast<size_t>(pool_.size()));
        for (auto& w : attention_work_)
        {
            w.q.resize(rows * static_cast<size_t>(hd));
            w.o.resize(rows * static_cast<size_t>(hd));
            w.s.resize(rows * static_cast<size_t>(kv_.block_positions()));
            w.max.resize(rows);
            w.sum.resize(rows);
            w.scratch.resize(static_cast<size_t>(qk_attention_scratch(hd)));
        }

        // The position component of each rotary pair: the time, except the pairs of the row
        // and column sections, interleaved as the reference implementation of Qwen3-VL lays
        // them out.
        pair_component_.assign(static_cast<size_t>(cfg_.rope_dims / 2), 0);
        if (cfg_.multimodal_rope())
            for (long j = 0; j < cfg_.rope_dims / 2; ++j)
            {
                if (j % 3 == 1 && j < 3 * cfg_.rope_sections[1]) pair_component_[static_cast<size_t>(j)] = 1;
                else if (j % 3 == 2 && j < 3 * cfg_.rope_sections[2]) pair_component_[static_cast<size_t>(j)] = 2;
            }
    }

    void cpu_runtime::set_context(long positions)
    {
        std::vector<float> previous = cfg_.rope_freq_factors;
        select_rope_factors(cfg_, positions);
        if (cfg_.rope_freq_factors == previous) return;
        if (rope_positions_ > 0)
        {
            cfg_.rope_freq_factors.swap(previous);
            throw std::logic_error("cpu_runtime: the rotary divisors cannot change once positions are computed");
        }
    }

    void cpu_runtime::set_offload(std::unique_ptr<matmul_offload> offload)
    {
        layer_pass_.reset();
        offload_ = std::move(offload);
        layer_pass_checked_ = false;
        layer_pass_memory_failures_ = 0;
        layer_pass_wait_ = 0;
        if (!offload_ || cfg_.is_moe()) return;
        dense_model m;
        m.d_model = cfg_.d_model;
        m.q_dim = cfg_.q_dim();
        m.kv_dim = cfg_.kv_dim();
        m.d_ffn = cfg_.d_ffn;
        m.rms_eps = static_cast<float>(cfg_.rms_eps);
        m.residual_scale = static_cast<float>(cfg_.residual_scale);
        for (const layer& L : layers_)
        {
            dense_layer l;
            l.attn_norm = L.attn_norm.data();
            l.ffn_norm = L.ffn_norm.data();
            l.wq = L.wq;
            l.wk = L.wk;
            l.wv = L.wv;
            l.wo = L.wo;
            l.w_gate = L.w_gate;
            l.w_up = L.w_up;
            l.w_down = L.w_down;
            m.layers.push_back(l);
        }
        layer_pass_ = offload_->make_layer_pass(m);
    }

    std::string cpu_runtime::description() const
    {
        std::ostringstream o;
        o << "processor runtime (" << qk_isa() << ", " << pool_.size() << " threads";
        if (offload_) o << ", products offloaded to " << offload_->description();
        o << "), " << describe(cfg_);
        return o.str();
    }

    // Sequences

    int cpu_runtime::create_sequence()
    {
        if (!free_sequences_.empty())
        {
            const int s = free_sequences_.back();
            free_sequences_.pop_back();
            sequences_[static_cast<size_t>(s)].reset(new kv_sequence());
            return s;
        }
        sequences_.emplace_back(new kv_sequence());
        return static_cast<int>(sequences_.size() - 1);
    }

    void cpu_runtime::release_sequence(int seq)
    {
        if (seq < 0 || static_cast<size_t>(seq) >= sequences_.size() || !sequences_[static_cast<size_t>(seq)]) return;
        sequences_[static_cast<size_t>(seq)]->clear(kv_);
        sequences_[static_cast<size_t>(seq)].reset();
        free_sequences_.push_back(seq);
    }

    long cpu_runtime::sequence_length(int seq) const
    {
        return sequences_.at(static_cast<size_t>(seq))->length;
    }

    void cpu_runtime::truncate_sequence(int seq, long length)
    {
        sequences_.at(static_cast<size_t>(seq))->truncate(kv_, length);
    }

    bool cpu_runtime::reserve_sequence(int seq, long length)
    {
        return sequences_.at(static_cast<size_t>(seq))->reserve(kv_, length);
    }

    bool cpu_runtime::copy_sequence(int src, int dst, long length)
    {
        const kv_sequence& s = *sequences_.at(static_cast<size_t>(src));
        kv_sequence& d = *sequences_.at(static_cast<size_t>(dst));
        // The full blocks are shared; each sequence copies a shared block before writing in it.
        return d.share_prefix(kv_, s, length);
    }

    // Building blocks

    void cpu_runtime::rms_norm_rows(const float* x, const float* gamma, float* y, long rows, long width, float eps)
    {
        auto one = [&](long r) {
            const float* xr = x + r * width;
            float* yr = y + r * width;
            const float ss = qk_dot_f32(xr, xr, width);
            const float scale = 1.0f / std::sqrt(ss / static_cast<float>(width) + eps);
            for (long i = 0; i < width; ++i) yr[i] = xr[i] * scale * gamma[i];
        };
        if (rows * width >= 65536)
            pool_.parallel_for(rows, [&](long r, int) { one(r); });
        else
            for (long r = 0; r < rows; ++r) one(r);
    }

    void cpu_runtime::ensure_rope(long positions)
    {
        if (positions <= rope_positions_) return;
        const long half = cfg_.rope_dims / 2;
        const long n_dims = cfg_.rope_dims;
        // Grow in steps, to avoid recomputing for every new token.
        const long target = std::max<long>(positions, rope_positions_ + 1024);
        rope_cos_.resize(static_cast<size_t>(target * half));
        rope_sin_.resize(static_cast<size_t>(target * half));

        const double theta_scale = std::pow(cfg_.rope_base, -2.0 / static_cast<double>(n_dims));
        float low = 0, high = 0;
        if (cfg_.yarn)
        {
            const float fast = yarn_corr_dim(n_dims, cfg_.yarn_orig_ctx, cfg_.yarn_beta_fast, cfg_.rope_base);
            const float slow = yarn_corr_dim(n_dims, cfg_.yarn_orig_ctx, cfg_.yarn_beta_slow, cfg_.rope_base);
            low = std::max(0.0f, std::floor(fast));
            high = std::min(static_cast<float>(n_dims - 1), std::ceil(slow));
        }
        const float freq_scale = static_cast<float>(cfg_.rope_freq_scale);
        const float attn_factor = static_cast<float>(cfg_.rope_attn_factor);
        for (long pos = rope_positions_; pos < target; ++pos)
        {
            double theta_pow = 1.0;
            for (long i = 0; i < half; ++i)
            {
                float theta_extrap = static_cast<float>(static_cast<double>(pos) * theta_pow);
                if (!cfg_.rope_freq_factors.empty()) theta_extrap /= cfg_.rope_freq_factors[static_cast<size_t>(i)];
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (cfg_.yarn)
                {
                    const float mix = yarn_ramp(low, high, 2 * i) * static_cast<float>(cfg_.yarn_ext_factor);
                    theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                    mscale = static_cast<float>(cfg_.yarn_attn_factor * cfg_.rope_attn_factor);
                }
                rope_cos_[static_cast<size_t>(pos * half + i)] = std::cos(theta) * mscale;
                rope_sin_[static_cast<size_t>(pos * half + i)] = std::sin(theta) * mscale;
                theta_pow *= theta_scale;
            }
        }
        rope_positions_ = target;
    }

    void cpu_runtime::apply_rope(float* rows, long n_rows, long n_heads, const std::vector<long>& positions)
    {
        const long hd = cfg_.head_dim;
        const long half = cfg_.rope_dims / 2;
        const bool neox = cfg_.rope == rope_style::neox;
        auto one = [&](long t) {
            const float* c = rope_cos_.data() + positions[static_cast<size_t>(t)] * half;
            const float* s = rope_sin_.data() + positions[static_cast<size_t>(t)] * half;
            for (long h = 0; h < n_heads; ++h)
            {
                float* x = rows + (t * n_heads + h) * hd;
                if (neox)
                {
                    for (long i = 0; i < half; ++i)
                    {
                        const float x0 = x[i], x1 = x[i + half];
                        x[i] = x0 * c[i] - x1 * s[i];
                        x[i + half] = x0 * s[i] + x1 * c[i];
                    }
                }
                else
                {
                    for (long i = 0; i < half; ++i)
                    {
                        const float x0 = x[2 * i], x1 = x[2 * i + 1];
                        x[2 * i] = x0 * c[i] - x1 * s[i];
                        x[2 * i + 1] = x0 * s[i] + x1 * c[i];
                    }
                }
            }
        };
        if (n_rows >= 16)
            pool_.parallel_for(n_rows, [&](long t, int) { one(t); });
        else
            for (long t = 0; t < n_rows; ++t) one(t);
    }

    void cpu_runtime::prepare_given_positions(const forward_batch& batch, long T)
    {
        const long half = cfg_.rope_dims / 2;
        const long C = forward_batch::position_components;
        given_cos_.resize(static_cast<size_t>(T * half));
        given_sin_.resize(static_cast<size_t>(T * half));
        for (long t = 0; t < T; ++t)
        {
            const long* p = batch.positions.data() + t * C;
            float* c = given_cos_.data() + t * half;
            float* s = given_sin_.data() + t * half;
            for (long i = 0; i < half; ++i)
            {
                const size_t at = static_cast<size_t>(p[pair_component_[static_cast<size_t>(i)]] * half + i);
                c[i] = rope_cos_[at];
                s[i] = rope_sin_[at];
            }
        }
    }

    void cpu_runtime::apply_given_rope(float* rows, long n_rows, long n_heads)
    {
        const long hd = cfg_.head_dim;
        const long half = cfg_.rope_dims / 2;
        const bool neox = cfg_.rope == rope_style::neox;
        auto one = [&](long t) {
            const float* c = given_cos_.data() + t * half;
            const float* s = given_sin_.data() + t * half;
            for (long h = 0; h < n_heads; ++h)
            {
                float* x = rows + (t * n_heads + h) * hd;
                if (neox)
                {
                    for (long i = 0; i < half; ++i)
                    {
                        const float x0 = x[i], x1 = x[i + half];
                        x[i] = x0 * c[i] - x1 * s[i];
                        x[i + half] = x0 * s[i] + x1 * c[i];
                    }
                }
                else
                {
                    for (long i = 0; i < half; ++i)
                    {
                        const float x0 = x[2 * i], x1 = x[2 * i + 1];
                        x[2 * i] = x0 * c[i] - x1 * s[i];
                        x[2 * i + 1] = x0 * s[i] + x1 * c[i];
                    }
                }
            }
        };
        if (n_rows >= 16)
            pool_.parallel_for(n_rows, [&](long t, int) { one(t); });
        else
            for (long t = 0; t < n_rows; ++t) one(t);
    }

    void cpu_runtime::check_given_embeddings(const forward_batch& batch, long T) const
    {
        const size_t W = static_cast<size_t>(cfg_.input_width());
        if (batch.embeddings.size() != batch.embedding_rows.size() * W)
            throw std::runtime_error("cpu_runtime: the embeddings of the batch do not match its rows");
        for (long r : batch.embedding_rows)
            if (r < 0 || r >= T) throw std::runtime_error("cpu_runtime: an embedding row lies outside the batch");
    }

    void cpu_runtime::add_given_embeddings(const forward_batch& batch, long slice)
    {
        // Slice 0 replaces the embedding of the token; slice k is added after layer k - 1.
        const long d = cfg_.d_model;
        const long W = cfg_.input_width();
        for (size_t i = 0; i < batch.embedding_rows.size(); ++i)
        {
            float* dst = x_.data() + batch.embedding_rows[i] * d;
            const float* src = batch.embeddings.data() + static_cast<long>(i) * W + slice * d;
            if (slice == 0) std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(d));
            else for (long j = 0; j < d; ++j) dst[j] += src[j];
        }
    }

    void cpu_runtime::product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
        qk_path path)
    {
        if (!offload_ || host_only_)
        {
            qk_matmul(W, X, n, x_stride, Y, y_stride, pool_, scratch_[0], false, path);
            return;
        }
        offload_->matmul(W, X, n, x_stride, Y, y_stride,
            [&]() { qk_matmul(W, X, n, x_stride, Y, y_stride, pool_, scratch_[0], false, path); });
    }

    void cpu_runtime::embed(const std::vector<int>& tokens)
    {
        const long d = cfg_.d_model;
        const long T = static_cast<long>(tokens.size());
        x_.resize(static_cast<size_t>(T * d));
        const float es = static_cast<float>(cfg_.embedding_scale);
        auto one = [&](long t) {
            const int id = tokens[static_cast<size_t>(t)];
            if (id < 0 || id >= cfg_.vocab) throw std::runtime_error("token identifier out of range");
            float* dst = x_.data() + t * d;
            qk_dequantize_row(token_embd_.type, token_embd_.data + static_cast<size_t>(id) * token_embd_.row_bytes,
                dst, d);
            if (es != 1.0f) qk_scale(dst, es, d);
        };
        for (long t = 0; t < T; ++t) one(t);
    }

    // Attention

    void cpu_runtime::store_keys_values(long li, const forward_batch& batch)
    {
        const long T = static_cast<long>(batch.tokens.size());
        const long KVH = cfg_.n_kv_heads;
        const long hd = cfg_.head_dim;
        const long kvd = cfg_.kv_dim();
        const long B = kv_.block_positions();
        const bool q8 = kv_.format() == kv_format::q8;
        const long scales_per_row = hd / kv_q8_group;
        token_sequences_.resize(static_cast<size_t>(T));
        for (const auto& s : batch.segments)
            for (long j = 0; j < s.count; ++j) token_sequences_[static_cast<size_t>(s.start + j)] = s.sequence;
        auto one = [&](long t, int thread) {
            const kv_sequence& seq = *sequences_[static_cast<size_t>(token_sequences_[static_cast<size_t>(t)])];
            const long pos = positions_[static_cast<size_t>(t)];
            const int block = seq.blocks[static_cast<size_t>(pos / B)];
            const long row = pos % B;
            float* rotated = attention_work_[static_cast<size_t>(thread)].scratch.data();
            for (long h = 0; h < KVH; ++h)
            {
                const float* k = k_.data() + t * kvd + h * hd;
                const float* v = v_.data() + t * kvd + h * hd;
                if (q8)
                {
                    // The keys are stored rotated (qk_rotate_row), the queries being rotated alike.
                    std::copy(k, k + hd, rotated);
                    qk_rotate_row(rotated, hd);
                    qk_float_to_q8_row(rotated, reinterpret_cast<int8_t*>(kv_.keys(block, li, h)) + row * hd,
                        kv_.key_scales(block, li, h) + row * scales_per_row, hd);
                    qk_float_to_q8_row(v, reinterpret_cast<int8_t*>(kv_.values(block, li, h)) + row * hd,
                        kv_.value_scales(block, li, h) + row * scales_per_row, hd);
                }
                else
                {
                    qk_float_to_half_row(k, reinterpret_cast<uint16_t*>(kv_.keys(block, li, h)) + row * hd, hd);
                    qk_float_to_half_row(v, reinterpret_cast<uint16_t*>(kv_.values(block, li, h)) + row * hd, hd);
                }
            }
        };
        // Each token writes rows of its own.
        if (T >= 32)
            pool_.parallel_for(T, [&](long t, int thread) { one(t, thread); });
        else
            for (long t = 0; t < T; ++t) one(t, 0);
    }

    void cpu_runtime::attention(long layer_index, const forward_batch& batch, const std::vector<long>& positions)
    {
        const long T = static_cast<long>(batch.tokens.size());
        const long KVH = cfg_.n_kv_heads;
        const long hd = cfg_.head_dim;
        const long qd = cfg_.q_dim();
        const long rep = cfg_.n_heads / KVH;
        const long B = kv_.block_positions();
        const bool q8 = kv_.format() == kv_format::q8;
        const long scales_per_row = hd / kv_q8_group;
        const float scale = cfg_.attention_scale > 0.0
            ? static_cast<float>(cfg_.attention_scale)
            : 1.0f / std::sqrt(static_cast<float>(hd));

        // The tiles: consecutive tokens of one segment, whose positions are consecutive too.
        const long per_tile = std::max<long>(1, tile_rows / rep);
        tiles_.clear();
        long span = 0;
        for (const auto& s : batch.segments)
            for (long j = 0; j < s.count; j += per_tile)
            {
                const long count = std::min(per_tile, s.count - j);
                tiles_.push_back({ s.sequence, s.start + j, count });
                span = std::max(span, positions[static_cast<size_t>(s.start + j + count - 1)] + 1);
            }
        /*
            The parts of a tile: when the tiles and their key/value heads are fewer than the
            threads, as in the generation of a single answer, the positions of each tile are
            split into ranges of part_positions at least, each range read by one task for all
            the heads of the group, and the partial sums of the ranges are merged afterwards
            (the split of FlashDecoding).  The keys and values are then read once for all the
            heads that share them.
        */
        const long units = static_cast<long>(tiles_.size()) * KVH;
        long parts = 1;
        if (units < pool_.size())
            parts = std::max<long>(1, std::min<long>((2 * pool_.size() + units - 1) / units, span / part_positions));
        const long rows_cap = per_tile * rep;
        if (parts > 1) partial_.resize(static_cast<size_t>(units * parts * rows_cap * (hd + 2)));

        ctx_.resize(static_cast<size_t>(T * qd));
        pool_.parallel_for(units * parts, [&](long item, int thread) {
            const attention_tile& tile = tiles_[static_cast<size_t>(item / (KVH * parts))];
            const long kvh = item / parts % KVH;
            const long part = item % parts;
            const long h0 = kvh * rep;
            const kv_sequence& seq = *sequences_[static_cast<size_t>(tile.sequence)];
            attention_work& w = attention_work_[static_cast<size_t>(thread)];
            const long rows = tile.count * rep;
            const long last = positions[static_cast<size_t>(tile.start + tile.count - 1)];
            const long length = (last + parts) / parts;
            const long begin = part * length;
            const long end = std::min(last + 1, begin + length);

            // Row a * rep + i holds token tile.start + a and head h0 + i, scaled, and rotated as the 8-bit keys.
            for (long a = 0; a < tile.count; ++a)
            {
                const float* src = q_.data() + (tile.start + a) * qd + h0 * hd;
                float* dst = w.q.data() + a * rep * hd;
                for (long e = 0; e < rep * hd; ++e) dst[e] = src[e] * scale;
                if (q8)
                    for (long i = 0; i < rep; ++i) qk_rotate_row(dst + i * hd, hd);
            }
            std::fill(w.o.begin(), w.o.begin() + rows * hd, 0.0f);
            std::fill(w.max.begin(), w.max.begin() + rows, -INFINITY);
            std::fill(w.sum.begin(), w.sum.begin() + rows, 0.0f);

            for (long p = begin; p < end;)
            {
                const int block = seq.blocks[static_cast<size_t>(p / B)];
                const long first = p % B;
                const long m = std::min(B - first, end - p);
                if (q8)
                    qk_attention_scores_q8(w.q.data(), rows,
                        reinterpret_cast<const int8_t*>(kv_.keys(block, layer_index, kvh)) + first * hd,
                        kv_.key_scales(block, layer_index, kvh) + first * scales_per_row, m, hd, w.s.data(), B,
                        w.scratch.data());
                else
                    qk_attention_scores(w.q.data(), rows,
                        reinterpret_cast<const uint16_t*>(kv_.keys(block, layer_index, kvh)) + first * hd, m, hd,
                        w.s.data(), B, w.scratch.data());
                for (long a = 0; a < tile.count; ++a)
                {
                    // Each token sees the positions up to its own.
                    const long own = positions[static_cast<size_t>(tile.start + a)];
                    const long seen = std::max<long>(0, std::min(m, own + 1 - p));
                    for (long i = 0; i < rep; ++i)
                    {
                        const long r = a * rep + i;
                        float* sr = w.s.data() + r * B;
                        std::fill(sr + seen, sr + m, 0.0f);
                        if (seen == 0) continue;
                        const float old = w.max[static_cast<size_t>(r)];
                        const float mx = std::max(old, qk_max(sr, seen));
                        const float added = qk_exp_shift_sum(sr, mx, seen);
                        if (mx != old && w.sum[static_cast<size_t>(r)] > 0.0f)
                        {
                            // A larger maximum: what the row gathered so far shrinks.
                            const float shrink = std::exp(old - mx);
                            w.sum[static_cast<size_t>(r)] *= shrink;
                            qk_scale(w.o.data() + r * hd, shrink, hd);
                        }
                        w.max[static_cast<size_t>(r)] = mx;
                        w.sum[static_cast<size_t>(r)] += added;
                    }
                }
                if (q8)
                    qk_attention_values_q8(w.s.data(), B, rows,
                        reinterpret_cast<const int8_t*>(kv_.values(block, layer_index, kvh)) + first * hd,
                        kv_.value_scales(block, layer_index, kvh) + first * scales_per_row, m, hd, w.o.data(),
                        w.scratch.data());
                else
                    qk_attention_values(w.s.data(), B, rows,
                        reinterpret_cast<const uint16_t*>(kv_.values(block, layer_index, kvh)) + first * hd, m, hd,
                        w.o.data(), w.scratch.data());
                p += m;
            }

            if (parts > 1)
            {
                // The sums of the range, unnormalized, with the maximum and the sum of each row.
                float* out = partial_.data() + item * rows_cap * (hd + 2);
                std::copy(w.o.begin(), w.o.begin() + rows * hd, out);
                std::copy(w.max.begin(), w.max.begin() + rows, out + rows_cap * hd);
                std::copy(w.sum.begin(), w.sum.begin() + rows, out + rows_cap * (hd + 1));
                return;
            }
            for (long a = 0; a < tile.count; ++a)
                for (long i = 0; i < rep; ++i)
                {
                    const long r = a * rep + i;
                    const float inv = 1.0f / w.sum[static_cast<size_t>(r)];
                    const float* src = w.o.data() + r * hd;
                    float* dst = ctx_.data() + (tile.start + a) * qd + (h0 + i) * hd;
                    for (long e = 0; e < hd; ++e) dst[e] = src[e] * inv;
                }
        });
        if (parts == 1) return;

        // Each row takes the ranges of its tile with the largest maximum of them as reference.
        pool_.parallel_for(units, [&](long unit, int) {
            const attention_tile& tile = tiles_[static_cast<size_t>(unit / KVH)];
            const long h0 = unit % KVH * rep;
            const float* first = partial_.data() + unit * parts * rows_cap * (hd + 2);
            const long stride = rows_cap * (hd + 2);
            for (long a = 0; a < tile.count; ++a)
                for (long i = 0; i < rep; ++i)
                {
                    const long r = a * rep + i;
                    float mx = -INFINITY;
                    for (long q = 0; q < parts; ++q) mx = std::max(mx, first[q * stride + rows_cap * hd + r]);
                    float* dst = ctx_.data() + (tile.start + a) * qd + (h0 + i) * hd;
                    std::fill(dst, dst + hd, 0.0f);
                    float sum = 0.0f;
                    for (long q = 0; q < parts; ++q)
                    {
                        const float* range = first + q * stride;
                        const float s = range[rows_cap * (hd + 1) + r];
                        if (s <= 0.0f) continue;
                        const float f = std::exp(range[rows_cap * hd + r] - mx);
                        sum += s * f;
                        const float* o = range + r * hd;
                        for (long e = 0; e < hd; ++e) dst[e] += f * o[e];
                    }
                    const float inv = 1.0f / sum;
                    for (long e = 0; e < hd; ++e) dst[e] *= inv;
                }
        });
    }

    // Mixture of experts

    void cpu_runtime::moe_ffn(const layer& L, long T)
    {
        const long d = cfg_.d_model;
        const long n_exp = cfg_.n_experts;
        const long top_k = cfg_.n_experts_used;
        const long df = cfg_.d_ffn_expert;

        router_logits_.resize(static_cast<size_t>(T * n_exp));
        product(L.router, xn_.data(), T, d, router_logits_.data(), n_exp);

        // The lists of the experts keep their memory from one layer and one pass to the next.
        std::vector<std::vector<std::pair<long, float>>>& routed = routed_;
        routed.resize(static_cast<size_t>(n_exp));
        for (auto& list : routed) list.clear();
        std::vector<float>& probs = router_probs_;
        probs.resize(static_cast<size_t>(n_exp));
        std::vector<long>& order = router_order_;
        order.resize(static_cast<size_t>(n_exp));
        for (long t = 0; t < T; ++t)
        {
            const float* row = router_logits_.data() + t * n_exp;
            const float mx = *std::max_element(row, row + n_exp);
            float sum = 0;
            for (long e = 0; e < n_exp; ++e)
            {
                probs[static_cast<size_t>(e)] = std::exp(row[e] - mx);
                sum += probs[static_cast<size_t>(e)];
            }
            for (auto& p : probs) p /= sum;
            for (long e = 0; e < n_exp; ++e) order[static_cast<size_t>(e)] = e;
            std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                [&](long a, long b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
            float selected = 0;
            for (long k = 0; k < top_k; ++k) selected += probs[static_cast<size_t>(order[static_cast<size_t>(k)])];
            for (long k = 0; k < top_k; ++k)
            {
                const long e = order[static_cast<size_t>(k)];
                float w = probs[static_cast<size_t>(e)];
                if (cfg_.expert_weights_norm && selected > 0) w /= selected;
                routed[static_cast<size_t>(e)].push_back(std::make_pair(t, w));
            }
        }

        tmp_.assign(static_cast<size_t>(T * d), 0.0f);
        aligned_floats& xe = expert_x_;
        aligned_floats& ge = expert_gate_;
        aligned_floats& ue = expert_up_;
        aligned_floats& ye = expert_y_;
        const qk_path path = T >= qk_float_product_tokens ? qk_path::floats : qk_path::integer;
        for (long e = 0; e < n_exp; ++e)
        {
            const auto& list = routed[static_cast<size_t>(e)];
            const long m = static_cast<long>(list.size());
            if (m == 0) continue;
            xe.resize(static_cast<size_t>(m * d));
            for (long i = 0; i < m; ++i)
                std::memcpy(xe.data() + i * d, xn_.data() + list[static_cast<size_t>(i)].first * d, sizeof(float) * d);
            ge.resize(static_cast<size_t>(m * df));
            ue.resize(static_cast<size_t>(m * df));
            product(L.e_gate[static_cast<size_t>(e)], xe.data(), m, d, ge.data(), df, path);
            product(L.e_up[static_cast<size_t>(e)], xe.data(), m, d, ue.data(), df, path);
            qk_silu_mul(ge.data(), ue.data(), static_cast<long>(ge.size()));
            ye.resize(static_cast<size_t>(m * d));
            product(L.e_down[static_cast<size_t>(e)], ge.data(), m, df, ye.data(), d, path);
            for (long i = 0; i < m; ++i)
            {
                const float w = list[static_cast<size_t>(i)].second;
                float* dst = tmp_.data() + list[static_cast<size_t>(i)].first * d;
                const float* src = ye.data() + i * d;
                for (long j = 0; j < d; ++j) dst[j] += w * src[j];
            }
        }
    }

    // Forward pass

    bool cpu_runtime::takes_layer_pass(const forward_batch& batch) const
    {
        // The DeepStack slices of the images are added to the hidden states between the first layers.
        return !(cfg_.n_deepstack > 0 && !batch.embedding_rows.empty());
    }

    void cpu_runtime::drop_layer_pass(const std::string& why)
    {
        layer_pass_->report(false, why);
        layer_pass_.reset();
    }

    /*
        Runs n tokens in a sequence of their own through the layers of the device, then through
        the processor alone, and compares their logits: 1 when they agree, 0 when they differ, -1
        when the keys and values or the device lack room for the probe.  detail describes the
        comparison.
    */
    int cpu_runtime::probe_layer_pass(long n, std::string& detail)
    {
        const long V = cfg_.vocab;
        const int seq = create_sequence();
        struct release
        {
            cpu_runtime& rt;
            int seq;
            ~release() { rt.release_sequence(seq); }
        } guard{ *this, seq };
        if (!reserve_sequence(seq, n)) return -1;
        forward_batch b;
        for (long i = 0; i < n; ++i) b.tokens.push_back(static_cast<int>((1009 + 7919 * i) % V));
        b.segments.push_back({ seq, 0, n });
        for (long i = 0; i < n; ++i) b.logit_rows.push_back(i);
        std::vector<float> reference, device;
        if (!run(b, device, true)) return -1;
        // The truncation gives the blocks back.
        truncate_sequence(seq, 0);
        if (!reserve_sequence(seq, n)) return -1;
        {
            struct processor_only
            {
                bool& flag;
                ~processor_only() { flag = false; }
            } scope{ host_only_ };
            host_only_ = true;
            run(b, reference, false);
        }
        double worst = 0;
        long same = 0;
        bool apart = false;
        for (long r = 0; r < n; ++r)
        {
            const float* a = reference.data() + r * V;
            const float* d = device.data() + r * V;
            double diff = 0, norm = 0;
            long ia = 0, id = 0;
            for (long v = 0; v < V; ++v)
            {
                diff += (static_cast<double>(a[v]) - d[v]) * (static_cast<double>(a[v]) - d[v]);
                norm += static_cast<double>(a[v]) * a[v];
                if (a[v] > a[ia]) ia = v;
                if (d[v] > d[id]) id = v;
            }
            const double relative = std::sqrt(diff / std::max(norm, 1e-30));
            // Written so that a value that is not a number sets the row apart.
            if (!(relative <= 0.01)) apart = true;
            if (relative > worst) worst = relative;
            if (ia == id) ++same;
        }
        std::ostringstream o;
        o << "logits of " << n << " probe tokens within " << worst * 100 << "% of those of the processor"
            << (apart && worst <= 0.01 ? ", some not numbers" : "") << ", " << same << " of " << n
            << " most likely tokens the same";
        detail = o.str();
        // Two candidates whose logits are nearly equal may change places: one row in sixteen may differ.
        return !apart && same >= n - std::max<long>(1, n / 16) ? 1 : 0;
    }

    /*
        Keeps the layer pass when passes of the device agree with the processor, a defect of the
        device or of its kernels otherwise reaching every answer: a probe of probe_tokens tokens,
        whose products read the blocks of the weights in place, and one of tiled_tokens() tokens,
        whose products go through the tiles of the weights expanded.  The state stays unchecked
        when the keys and values or the device lack room for a probe, so that a later pass checks
        again.
    */
    void cpu_runtime::check_layer_pass()
    {
        // A copy of the constant, which std::max() takes by reference.
        long first = probe_tokens;
        first = std::max(first, layer_pass_->min_tokens());
        if (cfg_.ctx_train > 0) first = std::min(first, cfg_.ctx_train);
        std::vector<long> sizes(1, first);
        const long tiled = layer_pass_->tiled_tokens();
        if (tiled > first && (cfg_.ctx_train <= 0 || tiled <= cfg_.ctx_train)) sizes.push_back(tiled);
        std::string details;
        for (long n : sizes)
        {
            std::string detail;
            const int outcome = probe_layer_pass(n, detail);
            if (outcome < 0) return;
            details += (details.empty() ? "" : "; ") + detail;
            if (outcome == 0)
            {
                drop_layer_pass(details);
                return;
            }
        }
        layer_pass_checked_ = true;
        layer_pass_->report(true, details);
    }

    void cpu_runtime::forward(const forward_batch& batch, std::vector<float>& logits)
    {
        if (layer_pass_ && !batch.tokens.empty() && takes_layer_pass(batch))
        {
            try
            {
                // A check that the keys and values or the device put off waits for a few passes.
                if (!layer_pass_checked_ && layer_pass_wait_ > 0)
                    --layer_pass_wait_;
                else if (!layer_pass_checked_)
                {
                    check_layer_pass();
                    if (!layer_pass_checked_) layer_pass_wait_ = layer_pass_check_interval;
                }
                if (layer_pass_ && layer_pass_checked_ && run(batch, logits, true))
                {
                    layer_pass_memory_failures_ = 0;
                    return;
                }
            }
            catch (const device_failure& e)
            {
                // The pass is computed again below; its keys and values are written again.
                if (!e.memory)
                    drop_layer_pass(std::string("the device failed: ") + e.what());
                else if (++layer_pass_memory_failures_ >= max_layer_pass_memory_failures)
                    drop_layer_pass(std::string("the device lacked memory several times in a row: ") + e.what());
            }
        }
        run(batch, logits, false);
    }

    bool cpu_runtime::run(const forward_batch& batch, std::vector<float>& logits, bool on_device)
    {
        const long T = static_cast<long>(batch.tokens.size());
        if (T == 0) { logits.clear(); return true; }
        const long d = cfg_.d_model;
        const long qd = cfg_.q_dim();
        const long kvd = cfg_.kv_dim();
        const long hd = cfg_.head_dim;
        const long H = cfg_.n_heads;
        const long KVH = cfg_.n_kv_heads;
        const long B = kv_.block_positions();
        const float eps = static_cast<float>(cfg_.rms_eps);
        const float rs = static_cast<float>(cfg_.residual_scale);
        if (scratch_.empty()) scratch_.resize(1);

        // Positions of the tokens, which continue their sequences.
        positions_.assign(static_cast<size_t>(T), 0);
        long max_pos = 0;
        for (const auto& s : batch.segments)
        {
            kv_sequence& seq = *sequences_.at(static_cast<size_t>(s.sequence));
            if (static_cast<long>(seq.blocks.size()) * B < seq.length + s.count)
                throw std::runtime_error("cpu_runtime: sequence storage was not reserved");
            if (s.count > 0 && kv_.shared(seq.blocks[static_cast<size_t>(seq.length / B)]))
                throw std::runtime_error("cpu_runtime: a sequence would write in a block other sequences hold");
            for (long j = 0; j < s.count; ++j)
            {
                positions_[static_cast<size_t>(s.start + j)] = seq.length + j;
                max_pos = std::max(max_pos, seq.length + j);
            }
        }
        // The positions and the embeddings the batch gives, for the tokens of images.
        const bool given = !batch.positions.empty();
        if (given)
        {
            if (static_cast<long>(batch.positions.size()) != T * forward_batch::position_components)
                throw std::runtime_error("cpu_runtime: the batch gives positions for other tokens than its own");
            for (long p : batch.positions)
            {
                if (p < 0) throw std::runtime_error("cpu_runtime: the batch gives a negative position");
                max_pos = std::max(max_pos, p);
            }
        }
        if (!batch.embedding_rows.empty()) check_given_embeddings(batch, T);
        ensure_rope(max_pos + 1);
        if (given) prepare_given_positions(batch, T);

        embed(batch.tokens);
        if (!batch.embedding_rows.empty()) add_given_embeddings(batch, 0);
        xn_.resize(static_cast<size_t>(T * d));
        q_.resize(static_cast<size_t>(T * qd));
        k_.resize(static_cast<size_t>(T * kvd));
        v_.resize(static_cast<size_t>(T * kvd));
        tmp_.resize(static_cast<size_t>(T * d));
        if (on_device && !layer_pass_->begin(x_.data(), T)) return false;

        for (long li = 0; li < cfg_.n_layers; ++li)
        {
            const layer& L = layers_[static_cast<size_t>(li)];

            // Attention block.
            if (on_device)
                layer_pass_->attention_inputs(li, q_.data(), k_.data(), v_.data());
            else
            {
                rms_norm_rows(x_.data(), L.attn_norm.data(), xn_.data(), T, d, eps);
                product(L.wq, xn_.data(), T, d, q_.data(), qd);
                product(L.wk, xn_.data(), T, d, k_.data(), kvd);
                product(L.wv, xn_.data(), T, d, v_.data(), kvd);
            }
            if (!L.bq.empty())
                for (long t = 0; t < T; ++t)
                {
                    for (long i = 0; i < qd; ++i) q_[static_cast<size_t>(t * qd + i)] += L.bq[static_cast<size_t>(i)];
                    for (long i = 0; i < kvd; ++i) k_[static_cast<size_t>(t * kvd + i)] += L.bk[static_cast<size_t>(i)];
                    for (long i = 0; i < kvd; ++i) v_[static_cast<size_t>(t * kvd + i)] += L.bv[static_cast<size_t>(i)];
                }
            if (!L.q_norm.empty())
            {
                rms_norm_rows(q_.data(), L.q_norm.data(), q_.data(), T * H, hd, eps);
                rms_norm_rows(k_.data(), L.k_norm.data(), k_.data(), T * KVH, hd, eps);
            }
            if (given)
            {
                apply_given_rope(q_.data(), T, H);
                apply_given_rope(k_.data(), T, KVH);
            }
            else
            {
                apply_rope(q_.data(), T, H, positions_);
                apply_rope(k_.data(), T, KVH, positions_);
            }
            if (cfg_.attn_temp_scale != 0.0)
                for (long t = 0; t < T; ++t)
                {
                    const double pos = static_cast<double>(positions_[static_cast<size_t>(t)]);
                    const float f = static_cast<float>(1.0 + cfg_.attn_temp_scale
                        * std::log(1.0 + std::floor(pos / cfg_.attn_temp_length)));
                    if (f != 1.0f) qk_scale(q_.data() + t * qd, f, qd);
                }

            // Keys and values of the batch go into the sequences before attention reads
            // them, so that each token sees the earlier tokens of its own chunk.
            store_keys_values(li, batch);

            attention(li, batch, positions_);
            if (on_device)
            {
                // The projection, the feed-forward block and both residual sums.
                layer_pass_->attention_output(li, ctx_.data());
                continue;
            }
            product(L.wo, ctx_.data(), T, qd, tmp_.data(), d);
            for (size_t i = 0; i < x_.size(); ++i) x_[i] += rs * tmp_[i];

            // Feed-forward block.
            rms_norm_rows(x_.data(), L.ffn_norm.data(), xn_.data(), T, d, eps);
            if (cfg_.is_moe())
            {
                moe_ffn(L, T);
            }
            else
            {
                const long df = cfg_.d_ffn;
                gate_.resize(static_cast<size_t>(T * df));
                up_.resize(static_cast<size_t>(T * df));
                product(L.w_gate, xn_.data(), T, d, gate_.data(), df);
                product(L.w_up, xn_.data(), T, d, up_.data(), df);
                const long n = T * df;
                auto gated = [&](long a, long b) { qk_silu_mul(gate_.data() + a, up_.data() + a, b - a); };
                if (n >= 65536)
                    pool_.parallel_for((n + 4095) / 4096, [&](long c, int) {
                        gated(c * 4096, std::min(n, c * 4096 + 4096));
                    });
                else
                    gated(0, n);
                product(L.w_down, gate_.data(), T, df, tmp_.data(), d);
            }
            for (size_t i = 0; i < x_.size(); ++i) x_[i] += rs * tmp_[i];
            if (li < cfg_.n_deepstack && !batch.embedding_rows.empty()) add_given_embeddings(batch, li + 1);
        }

        if (on_device) layer_pass_->end(batch.logit_rows, x_.data());
        for (const auto& s : batch.segments)
            sequences_[static_cast<size_t>(s.sequence)]->length += s.count;

        // Final normalization and logits of the requested rows only.
        const long R = static_cast<long>(batch.logit_rows.size());
        hidden_out_.resize(static_cast<size_t>(R * d));
        for (long r = 0; r < R; ++r)
        {
            const long t = batch.logit_rows[static_cast<size_t>(r)];
            rms_norm_rows(x_.data() + t * d, output_norm_.data(), hidden_out_.data() + r * d, 1, d, eps);
        }
        if (batch.hidden_only)
        {
            logits.clear();
            return true;
        }
        logits.resize(static_cast<size_t>(R * cfg_.vocab));
        if (R > 0)
        {
            product(output_, hidden_out_.data(), R, d, logits.data(), cfg_.vocab);
            if (cfg_.logit_scale != 1.0)
                qk_scale(logits.data(), static_cast<float>(1.0 / cfg_.logit_scale), R * cfg_.vocab);
        }
        return true;
    }

}
