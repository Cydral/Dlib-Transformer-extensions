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
        inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

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
    }

    cpu_runtime::cpu_runtime(std::shared_ptr<model_file> file, const model_config& config, thread_pool& pool,
        size_t kv_budget_bytes, long kv_block_positions)
        : file_(std::move(file)), cfg_(config), pool_(pool),
          kv_(config.n_layers, config.n_kv_heads, config.head_dim, kv_block_positions, kv_budget_bytes)
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
            L.wq = f.matrix(p + "attn_q.weight", qd, d);
            L.wk = f.matrix(p + "attn_k.weight", kvd, d);
            L.wv = f.matrix(p + "attn_v.weight", kvd, d);
            L.wo = f.matrix(p + "attn_output.weight", d, qd);
            f.optional_floats(p + "attn_q.bias", static_cast<size_t>(qd), L.bq);
            f.optional_floats(p + "attn_k.bias", static_cast<size_t>(kvd), L.bk);
            f.optional_floats(p + "attn_v.bias", static_cast<size_t>(kvd), L.bv);
            f.optional_floats(p + "attn_q_norm.weight", static_cast<size_t>(hd), L.q_norm);
            f.optional_floats(p + "attn_k_norm.weight", static_cast<size_t>(hd), L.k_norm);
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
        scores_.resize(static_cast<size_t>(pool_.size()));
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
        kv_sequence& s = *sequences_.at(static_cast<size_t>(src));
        kv_sequence& d = *sequences_.at(static_cast<size_t>(dst));
        d.clear(kv_);
        length = std::max<long>(0, std::min(length, s.length));
        if (!d.reserve(kv_, length)) return false;
        const long B = kv_.block_positions();
        const size_t blocks = static_cast<size_t>((length + B - 1) / B);
        for (size_t b = 0; b < blocks; ++b) kv_.copy_block(s.blocks[b], d.blocks[b]);
        d.length = length;
        return true;
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
        for (long pos = rope_positions_; pos < target; ++pos)
        {
            double theta_pow = 1.0;
            for (long i = 0; i < half; ++i)
            {
                float theta_extrap = static_cast<float>(static_cast<double>(pos) * theta_pow);
                if (!cfg_.rope_freq_factors.empty()) theta_extrap /= cfg_.rope_freq_factors[static_cast<size_t>(i)];
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = 1.0f;
                if (cfg_.yarn)
                {
                    const float mix = yarn_ramp(low, high, 2 * i) * static_cast<float>(cfg_.yarn_ext_factor);
                    theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                    mscale = static_cast<float>(cfg_.yarn_attn_factor);
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

    void cpu_runtime::product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
        qk_path path)
    {
        if (!offload_)
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

    void cpu_runtime::attention(long layer_index, const forward_batch& batch, const std::vector<long>& positions)
    {
        const long T = static_cast<long>(batch.tokens.size());
        const long H = cfg_.n_heads;
        const long hd = cfg_.head_dim;
        const long qd = cfg_.q_dim();
        const long rep = H / cfg_.n_kv_heads;
        const long B = kv_.block_positions();
        const float scale = cfg_.attention_scale > 0.0
            ? static_cast<float>(cfg_.attention_scale)
            : 1.0f / std::sqrt(static_cast<float>(hd));

        // The sequence of every token, for the tasks below.
        std::vector<int> seq_of(static_cast<size_t>(T), -1);
        long max_len = 0;
        for (const auto& s : batch.segments)
            for (long j = 0; j < s.count; ++j)
            {
                seq_of[static_cast<size_t>(s.start + j)] = s.sequence;
                max_len = std::max(max_len, positions[static_cast<size_t>(s.start + j)] + 1);
            }
        for (auto& sc : scores_)
            if (static_cast<long>(sc.size()) < max_len) sc.resize(static_cast<size_t>(max_len));

        ctx_.resize(static_cast<size_t>(T * qd));
        pool_.parallel_for(T * H, [&](long item, int thread) {
            const long t = item / H;
            const long h = item % H;
            const long kvh = h / rep;
            const kv_sequence& seq = *sequences_[static_cast<size_t>(seq_of[static_cast<size_t>(t)])];
            const long n = positions[static_cast<size_t>(t)] + 1;
            const float* q = q_.data() + t * qd + h * hd;
            float* sc = scores_[static_cast<size_t>(thread)].data();

            for (long p = 0; p < n; p += B)
            {
                const int block = seq.blocks[static_cast<size_t>(p / B)];
                const long m = std::min(B, n - p);
                const uint16_t* K = kv_.keys(block, layer_index, kvh);
                for (long i = 0; i < m; ++i)
                    sc[p + i] = qk_dot_f32_f16(q, K + i * hd, hd) * scale;
            }
            const float mx = qk_max(sc, n);
            const float sum = qk_exp_shift_sum(sc, mx, n);
            const float inv = 1.0f / sum;

            float* out = ctx_.data() + t * qd + h * hd;
            std::fill(out, out + hd, 0.0f);
            for (long p = 0; p < n; p += B)
            {
                const int block = seq.blocks[static_cast<size_t>(p / B)];
                const long m = std::min(B, n - p);
                const uint16_t* V = kv_.values(block, layer_index, kvh);
                for (long i = 0; i < m; ++i)
                    qk_axpy_f16(sc[p + i] * inv, V + i * hd, out, hd);
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

        std::vector<std::vector<std::pair<long, float>>> routed(static_cast<size_t>(n_exp));
        std::vector<float> probs(static_cast<size_t>(n_exp));
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
            std::vector<long> order(static_cast<size_t>(n_exp));
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
        std::vector<float> xe, ge, ue, ye;
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
            for (size_t i = 0; i < ge.size(); ++i) ge[i] = silu(ge[i]) * ue[i];
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

    void cpu_runtime::forward(const forward_batch& batch, std::vector<float>& logits)
    {
        const long T = static_cast<long>(batch.tokens.size());
        if (T == 0) { logits.clear(); return; }
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
            for (long j = 0; j < s.count; ++j)
            {
                positions_[static_cast<size_t>(s.start + j)] = seq.length + j;
                max_pos = std::max(max_pos, seq.length + j);
            }
        }
        ensure_rope(max_pos + 1);

        embed(batch.tokens);
        xn_.resize(static_cast<size_t>(T * d));
        q_.resize(static_cast<size_t>(T * qd));
        k_.resize(static_cast<size_t>(T * kvd));
        v_.resize(static_cast<size_t>(T * kvd));
        tmp_.resize(static_cast<size_t>(T * d));

        for (long li = 0; li < cfg_.n_layers; ++li)
        {
            const layer& L = layers_[static_cast<size_t>(li)];

            // Attention block.
            rms_norm_rows(x_.data(), L.attn_norm.data(), xn_.data(), T, d, eps);
            product(L.wq, xn_.data(), T, d, q_.data(), qd);
            product(L.wk, xn_.data(), T, d, k_.data(), kvd);
            product(L.wv, xn_.data(), T, d, v_.data(), kvd);
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
            apply_rope(q_.data(), T, H, positions_);
            apply_rope(k_.data(), T, KVH, positions_);
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
            for (const auto& s : batch.segments)
            {
                kv_sequence& seq = *sequences_[static_cast<size_t>(s.sequence)];
                for (long j = 0; j < s.count; ++j)
                {
                    const long t = s.start + j;
                    const long pos = positions_[static_cast<size_t>(t)];
                    const int block = seq.blocks[static_cast<size_t>(pos / B)];
                    const long row = pos % B;
                    for (long h = 0; h < KVH; ++h)
                    {
                        qk_float_to_half_row(k_.data() + t * kvd + h * hd, kv_.keys(block, li, h) + row * hd, hd);
                        qk_float_to_half_row(v_.data() + t * kvd + h * hd, kv_.values(block, li, h) + row * hd, hd);
                    }
                }
            }

            attention(li, batch, positions_);
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
                auto gated = [&](long a, long b) {
                    for (size_t i = static_cast<size_t>(a); i < static_cast<size_t>(b); ++i)
                        gate_[i] = silu(gate_[i]) * up_[i];
                };
                if (n >= 65536)
                    pool_.parallel_for((n + 4095) / 4096, [&](long c, int) {
                        gated(c * 4096, std::min(n, c * 4096 + 4096));
                    });
                else
                    gated(0, n);
                product(L.w_down, gate_.data(), T, df, tmp_.data(), d);
            }
            for (size_t i = 0; i < x_.size(); ++i) x_[i] += rs * tmp_[i];
        }

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
        logits.resize(static_cast<size_t>(R * cfg_.vocab));
        if (R > 0)
        {
            product(output_, hidden_out_.data(), R, d, logits.data(), cfg_.vocab);
            if (cfg_.logit_scale != 1.0)
                qk_scale(logits.data(), static_cast<float>(1.0 / cfg_.logit_scale), R * cfg_.vocab);
        }
    }

}
