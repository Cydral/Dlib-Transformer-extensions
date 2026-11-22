// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cpu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace albireo
{

    namespace
    {
        // The rows [first, first + count) of a matrix, as a matrix of their own in its memory.
        weight_matrix row_slice(const weight_matrix& w, long first, long count)
        {
            weight_matrix s = w;
            s.data = w.data + static_cast<size_t>(first) * w.row_bytes;
            s.rows = count;
            return s;
        }

        const double pi = 3.14159265358979323846;

        /*
            The scalars of a head of a Mamba-3 layer (step, decay, trapezoid weight, angles) are
            computed in double precision: a step of several units multiplies the rounding of
            single precision into angles off by 1e-5, and they cost a few operations per head
            and per token.
        */
        inline double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
        inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

        // The pairs (i, i + n / 2) of a row of n values turned by the angles of cos and sin.
        inline void rotate_halves(float* row, const float* cos, const float* sin, long n, long angles)
        {
            float* second = row + n / 2;
            for (long i = 0; i < angles; ++i)
            {
                const float a = row[i], b = second[i];
                row[i] = a * cos[i] - b * sin[i];
                second[i] = a * sin[i] + b * cos[i];
            }
        }
    }

    cpu_runtime::cpu_runtime(std::shared_ptr<model_file> file, const model_config& config, thread_pool& pool,
        size_t kv_budget_bytes, long kv_block_positions, kv_format kv_type)
        : file_(std::move(file)), cfg_(config), pool_(pool),
          kv_(config.attention_layers(), config.n_kv_heads, config.head_dim, kv_block_positions, kv_budget_bytes,
              kv_type)
    {
        const model_file& f = *file_;
        const long d = cfg_.d_model;
        const long qd = cfg_.q_dim();
        const long kvd = cfg_.kv_dim();
        const long hd = cfg_.head_dim;

        layers_.resize(static_cast<size_t>(cfg_.n_layers));
        for (long i = 0; i < cfg_.n_layers; ++i)
        {
            if (cfg_.hybrid())
            {
                load_hybrid_layer(f, i);
                continue;
            }
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
        if (!cfg_.hybrid()) output_norm_ = f.floats("output_norm.weight", static_cast<size_t>(d));
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
        if (cfg_.hybrid())
        {
            const size_t R = static_cast<size_t>(cfg_.ssm_rank), N = static_cast<size_t>(cfg_.ssm_state);
            const size_t P = static_cast<size_t>(cfg_.ssm_head_dim);
            mixer_work_.resize(static_cast<size_t>(pool_.size()));
            for (auto& w : mixer_work_)
            {
                w.k.resize(R * N);
                w.q.resize(R * N);
                w.cos.resize(static_cast<size_t>(cfg_.ssm_angles));
                w.sin.resize(static_cast<size_t>(cfg_.ssm_angles));
                w.before.resize(R);
                w.now.resize(R);
                w.y.resize(R);
                w.o.resize(R * P);
                w.gate.resize(R * P);
            }
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

    void cpu_runtime::load_hybrid_layer(const model_file& f, long i)
    {
        const std::string p = "blk." + std::to_string(i) + ".";
        layer& L = layers_[static_cast<size_t>(i)];
        const long d = cfg_.d_model;
        const long hd = cfg_.head_dim;
        const size_t H = static_cast<size_t>(cfg_.ssm_heads), P = static_cast<size_t>(cfg_.ssm_head_dim);
        const size_t N = static_cast<size_t>(cfg_.ssm_state), R = static_cast<size_t>(cfg_.ssm_rank);
        const size_t S = static_cast<size_t>(cfg_.ssm_angles);
        if (cfg_.ssm_rank > qk_state_rank)
            throw std::runtime_error("the states of the Mamba-3 layers take more inputs than the kernels hold");
        auto scalar = [&](const char* name) { return f.floats(p + name, 1)[0]; };
        L.mixer_scale = scalar("geo_mixer_scale");
        L.mixer_bias = scalar("geo_mixer_bias");
        L.ffn_scale = scalar("geo_mlp_scale");
        L.ffn_bias = scalar("geo_mlp_bias");
        if (kv_layers_.empty()) kv_layers_.assign(static_cast<size_t>(cfg_.n_layers), -1);
        state_offsets_.resize(static_cast<size_t>(cfg_.n_layers));
        state_offsets_[static_cast<size_t>(i)] = state_floats_;
        if (cfg_.layer_types[static_cast<size_t>(i)] == 'M')
        {
            /*
                A row of ssm_in holds, for each head, its gate and its input (ssm_head_dim values
                each), then its step, its decay and its trapezoid weight; a row of ssm_in_dyn
                holds the inputs of the states, their outputs (ssm_rank rows of ssm_state values
                each), then the turns of the angles.
            */
            const long inner = cfg_.ssm_inner();
            L.ssm_in = f.matrix(p + "ssm_in.weight", static_cast<long>(H * (2 * P + 3)), d);
            L.ssm_dyn = f.matrix(p + "ssm_in_dyn.weight", static_cast<long>(2 * R * N + S), d);
            L.wo = f.matrix(p + "attn_output.weight", d, inner);
            L.ssm_b_norm = f.floats(p + "ssm_b_norm.weight", N);
            L.ssm_c_norm = f.floats(p + "ssm_c_norm.weight", N);
            L.ssm_b_bias = f.floats(p + "ssm_b_bias", H * R * N);
            L.ssm_c_bias = f.floats(p + "ssm_c_bias", H * R * N);
            L.ssm_dt_bias = f.floats(p + "ssm_dt_bias", H);
            L.ssm_d = f.floats(p + "ssm_d", H);
            L.mimo_x = f.floats(p + "ssm_mimo_x", H * R * P);
            L.mimo_z = f.floats(p + "ssm_mimo_z", H * R * P);
            L.mimo_o = f.floats(p + "ssm_mimo_o", H * R * P);
            // The angles, the states of the channels, then the inputs and the input of the previous token.
            state_floats_ += H * S + H * P * N + H * R * N + H * P;
        }
        else
        {
            const long KV = cfg_.n_kv_heads;
            const long rank = cfg_.tpa_rank;
            L.wq = f.matrix(p + "attn_q.weight", cfg_.q_dim(), d);
            L.wa_k = f.matrix(p + "attn_wa_k.weight", KV * rank, d);
            L.wb_k = f.matrix(p + "attn_wb_k.weight", rank * hd, d);
            L.wa_v = f.matrix(p + "attn_wa_v.weight", KV * rank, d);
            L.wb_v = f.matrix(p + "attn_wb_v.weight", rank * hd, d);
            L.shift_k = f.matrix(p + "attn_shift_k.weight", KV, d);
            L.shift_v = f.matrix(p + "attn_shift_v.weight", KV, d);
            L.noise = f.matrix(p + "attn_lambda.weight", KV, d);
            L.attn_gate = f.matrix(p + "attn_gate.weight", cfg_.signal_heads() * hd, d);
            L.wo = f.matrix(p + "attn_output.weight", d, cfg_.signal_heads() * hd);
            L.q_norm = f.floats(p + "attn_q_norm.weight", static_cast<size_t>(hd));
            L.k_norm = f.floats(p + "attn_k_norm.weight", static_cast<size_t>(hd));
            L.softmax_scaler = f.floats(p + "attn_softmax_scaler", static_cast<size_t>(cfg_.n_heads));
            long used = 0;
            for (long v : kv_layers_) used += v >= 0 ? 1 : 0;
            kv_layers_[static_cast<size_t>(i)] = used;
            // The key and the value of the last token, before their shift.
            state_floats_ += 2 * static_cast<size_t>(cfg_.kv_dim());
        }
        const long n_exp = cfg_.n_experts;
        const long dl = cfg_.d_latent;
        L.router = f.matrix(p + "ffn_gate_inp.weight", n_exp, d);
        L.expert_bias = f.floats(p + "exp_probs_b", static_cast<size_t>(n_exp));
        L.latent_down = f.matrix(p + "ffn_latent_down.weight", dl, d);
        L.latent_up = f.matrix(p + "ffn_latent_up.weight", d, dl);
        for (long e = 0; e < n_exp; ++e)
        {
            L.e_up.push_back(f.expert_matrix(p + "ffn_up_exps.weight", cfg_.d_ffn_expert, dl, e, n_exp));
            L.e_down.push_back(f.expert_matrix(p + "ffn_down_exps.weight", dl, cfg_.d_ffn_expert, e, n_exp));
        }
        L.shared_up = f.matrix(p + "ffn_up_shexp.weight", cfg_.d_ffn_shared, d);
        L.shared_down = f.matrix(p + "ffn_down_shexp.weight", d, cfg_.d_ffn_shared);
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
        // The states the device of the previous offload alone holds come back first.
        if (hybrid_pass_) fetch_all_states();
        hybrid_pass_.reset();
        offload_ = std::move(offload);
        layer_pass_checked_ = false;
        layer_pass_memory_failures_ = 0;
        layer_pass_wait_ = 0;
        attention_wait_ = 0;
        attention_interval_ = 0;
        device_few_ = true;
        device_many_ = true;
        if (!offload_) return;
        if (cfg_.hybrid())
        {
            hybrid_model h;
            if (describe_hybrid(h)) hybrid_pass_ = offload_->make_hybrid_pass(h);
            return;
        }
        // The other layers of a device are those of a dense decoder made of attentions.
        if (cfg_.is_moe()) return;
        dense_model m;
        m.d_model = cfg_.d_model;
        m.q_dim = cfg_.q_dim();
        m.kv_dim = cfg_.kv_dim();
        m.d_ffn = cfg_.d_ffn;
        m.rms_eps = static_cast<float>(cfg_.rms_eps);
        m.residual_scale = static_cast<float>(cfg_.residual_scale);
        m.n_heads = cfg_.n_heads;
        m.n_kv_heads = cfg_.n_kv_heads;
        m.head_dim = cfg_.head_dim;
        m.rope_dims = cfg_.rope_dims;
        m.neox = cfg_.rope == rope_style::neox;
        m.attention_scale = cfg_.attention_scale > 0.0
            ? static_cast<float>(cfg_.attention_scale)
            : 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
        // The temperature of the attention of the long positions stays with the processor.
        m.attention = cfg_.attn_temp_scale == 0.0;
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
            if (!L.bq.empty())
            {
                l.bq = L.bq.data();
                l.bk = L.bk.data();
                l.bv = L.bv.data();
            }
            if (!L.q_norm.empty())
            {
                l.q_norm = L.q_norm.data();
                l.k_norm = L.k_norm.data();
            }
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

    bool cpu_runtime::describe_hybrid(hybrid_model& m) const
    {
        m.d_model = cfg_.d_model;
        m.rms_eps = static_cast<float>(cfg_.rms_eps);
        m.ssm_heads = cfg_.ssm_heads;
        m.ssm_head_dim = cfg_.ssm_head_dim;
        m.ssm_state = cfg_.ssm_state;
        m.ssm_rank = cfg_.ssm_rank;
        m.ssm_angles = cfg_.ssm_angles;
        m.n_experts = cfg_.n_experts;
        m.n_experts_used = cfg_.n_experts_used;
        m.expert_weights_norm = cfg_.expert_weights_norm;
        m.expert_weights_scale = static_cast<float>(cfg_.expert_weights_scale);
        m.state_floats = state_floats_;
        m.output = output_;
        // The matrices of the experts of a layer as one matrix, when they follow one another in the file.
        auto joined = [&](const std::vector<weight_matrix>& experts, weight_matrix& all) {
            if (experts.empty() || !experts[0].valid()) return false;
            const weight_matrix& first = experts[0];
            const size_t bytes = static_cast<size_t>(first.rows) * first.row_bytes;
            for (size_t e = 1; e < experts.size(); ++e)
                if (experts[e].type != first.type || experts[e].rows != first.rows || experts[e].cols != first.cols
                    || experts[e].row_bytes != first.row_bytes || experts[e].data != first.data + e * bytes)
                    return false;
            all = first;
            all.rows = first.rows * static_cast<long>(experts.size());
            return true;
        };
        for (long i = 0; i < cfg_.n_layers; ++i)
        {
            const layer& L = layers_[static_cast<size_t>(i)];
            hybrid_layer l;
            l.mamba = cfg_.layer_types[static_cast<size_t>(i)] == 'M';
            if (l.mamba)
            {
                l.ssm_in = L.ssm_in;
                l.ssm_dyn = L.ssm_dyn;
                l.b_norm = L.ssm_b_norm.data();
                l.c_norm = L.ssm_c_norm.data();
                l.b_bias = L.ssm_b_bias.data();
                l.c_bias = L.ssm_c_bias.data();
                l.dt_bias = L.ssm_dt_bias.data();
                l.skip = L.ssm_d.data();
                l.mimo_x = L.mimo_x.data();
                l.mimo_z = L.mimo_z.data();
                l.mimo_o = L.mimo_o.data();
            }
            else
            {
                l.wq = L.wq;
                l.wa_k = L.wa_k;
                l.wb_k = L.wb_k;
                l.wa_v = L.wa_v;
                l.wb_v = L.wb_v;
                l.shift_k = L.shift_k;
                l.shift_v = L.shift_v;
                l.noise = L.noise;
                l.gate = L.attn_gate;
            }
            l.wo = L.wo;
            l.router = L.router;
            l.latent_down = L.latent_down;
            l.latent_up = L.latent_up;
            if (!joined(L.e_up, l.experts_up) || !joined(L.e_down, l.experts_down)) return false;
            l.shared_up = L.shared_up;
            l.shared_down = L.shared_down;
            l.expert_bias = L.expert_bias.data();
            l.mixer_scale = L.mixer_scale;
            l.mixer_bias = L.mixer_bias;
            l.ffn_scale = L.ffn_scale;
            l.ffn_bias = L.ffn_bias;
            l.state_offset = state_offsets_[static_cast<size_t>(i)];
            m.layers.push_back(l);
        }
        return true;
    }

    void cpu_runtime::set_busy(bool busy)
    {
        // A model without requests may give the room of its copies: its states leave the device first.
        if (!busy && hybrid_pass_) fetch_all_states();
        if (offload_) offload_->set_busy(busy);
    }

    void cpu_runtime::trim_memory()
    {
        if (hybrid_pass_) fetch_all_states();
        kv_.trim();
    }

    // Sequences

    int cpu_runtime::create_sequence()
    {
        // The states of a hybrid decoder first, which may lack memory: no slot is then taken.
        recurrent_sequence states;
        if (cfg_.hybrid()) states.live.values.assign(state_floats_, 0.0f);
        if (!free_sequences_.empty())
        {
            const int s = free_sequences_.back();
            sequences_[static_cast<size_t>(s)].reset(new kv_sequence());
            free_sequences_.pop_back();
            sequence_pools_[static_cast<size_t>(s)] = &kv_;
            if (cfg_.hybrid()) recurrent_[static_cast<size_t>(s)] = std::move(states);
            return s;
        }
        sequences_.reserve(sequences_.size() + 1);
        sequence_pools_.reserve(sequence_pools_.size() + 1);
        if (cfg_.hybrid()) recurrent_.reserve(recurrent_.size() + 1);
        sequences_.emplace_back(new kv_sequence());
        sequence_pools_.push_back(&kv_);
        if (cfg_.hybrid()) recurrent_.push_back(std::move(states));
        return static_cast<int>(sequences_.size() - 1);
    }

    void cpu_runtime::release_sequence(int seq)
    {
        if (seq < 0 || static_cast<size_t>(seq) >= sequences_.size() || !sequences_[static_cast<size_t>(seq)]) return;
        sequences_[static_cast<size_t>(seq)]->clear(pool_of(seq));
        sequences_[static_cast<size_t>(seq)].reset();
        sequence_pools_[static_cast<size_t>(seq)] = &kv_;
        // The states of the sequence go back to the system, and leave the device.
        if (cfg_.hybrid()) recurrent_[static_cast<size_t>(seq)] = recurrent_sequence();
        if (hybrid_pass_) hybrid_pass_->forget(seq);
        free_sequences_.push_back(seq);
    }

    long cpu_runtime::sequence_length(int seq) const
    {
        return sequences_.at(static_cast<size_t>(seq))->length;
    }

    void cpu_runtime::truncate_sequence(int seq, long length)
    {
        kv_sequence& s = *sequences_.at(static_cast<size_t>(seq));
        kv_pool& pool = pool_of(seq);
        const long B = pool.block_positions();
        // The copies of the device forget the rows the sequence gives up in the block it keeps last.
        if (layer_pass_ && length >= 0 && length < s.length && length % B != 0)
            layer_pass_->truncated(pool, s.blocks[static_cast<size_t>(length / B)], length % B);
        if (cfg_.hybrid() && length < s.length)
        {
            recurrent_sequence& r = recurrent_[static_cast<size_t>(seq)];
            const recurrent_state* kept = state_at(seq, length);
            if (length > 0 && !kept)
                throw std::logic_error("cpu_runtime: a hybrid decoder goes back to the lengths of its saved states "
                    "only (restorable_length())");
            if (length <= 0) std::fill(r.live.values.begin(), r.live.values.end(), 0.0f);
            else if (kept != &r.live) r.live.values = kept->values;
            r.live.length = std::max<long>(0, length);
            while (!r.saved.empty() && r.saved.back().length > length) r.saved.pop_back();
            r.reading = -1;
            r.own = false;
            // The host holds the states again: those of the device, further in the sequence, go.
            if (length <= 0 || kept != &r.live)
            {
                r.on_host = true;
                r.lost = false;
                if (hybrid_pass_) hybrid_pass_->forget(seq);
            }
        }
        s.truncate(pool, length);
    }

    bool cpu_runtime::reserve_sequence(int seq, long length)
    {
        return sequences_.at(static_cast<size_t>(seq))->reserve(pool_of(seq), length);
    }

    bool cpu_runtime::copy_sequence(int src, int dst, long length)
    {
        const kv_sequence& s = *sequences_.at(static_cast<size_t>(src));
        kv_sequence& d = *sequences_.at(static_cast<size_t>(dst));
        if (&pool_of(src) != &pool_of(dst)) throw std::logic_error("cpu_runtime: sequences of two pools");
        if (cfg_.hybrid())
        {
            const recurrent_state* from = state_at(src, length);
            if (length > 0 && !from)
                throw std::logic_error("cpu_runtime: a hybrid decoder copies a sequence up to the lengths of its "
                    "saved states only (restorable_length())");
            // The states the device alone holds come back before their copy; lost, they are not copied.
            if (length > 0 && from == &recurrent_[static_cast<size_t>(src)].live && !fetch_states(src)) return false;
            // The states are copied apart first: the source may be the destination, and the keys may lack room.
            recurrent_sequence copy;
            if (length > 0) copy.live.values = from->values;
            else copy.live.values.assign(state_floats_, 0.0f);
            copy.live.length = std::max<long>(0, length);
            // The copy may go back to the prefix it shares.
            if (length > 0) copy.saved.push_back(copy.live);
            if (!d.share_prefix(pool_of(dst), s, length)) return false;
            recurrent_[static_cast<size_t>(dst)] = std::move(copy);
            if (hybrid_pass_) hybrid_pass_->forget(dst);
            return true;
        }
        // The full blocks are shared; each sequence copies a shared block before writing in it.
        return d.share_prefix(pool_of(dst), s, length);
    }

    const cpu_runtime::recurrent_state* cpu_runtime::state_at(int seq, long length) const
    {
        const recurrent_sequence& r = recurrent_.at(static_cast<size_t>(seq));
        if (!r.lost && r.live.length == length) return &r.live;
        for (const recurrent_state& s : r.saved)
            if (s.length == length) return &s;
        return nullptr;
    }

    long cpu_runtime::restorable_length(int seq, long length) const
    {
        if (!cfg_.hybrid()) return length;
        const recurrent_sequence& r = recurrent_.at(static_cast<size_t>(seq));
        long best = 0;
        if (!r.lost && r.live.length <= length) best = r.live.length;
        for (const recurrent_state& s : r.saved)
            if (s.length <= length) best = std::max(best, s.length);
        return best;
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
        // Grow in steps, to avoid recomputing for every new token.
        const long target = std::max<long>(positions, rope_positions_ + 1024);
        rope_cos_.resize(static_cast<size_t>(target * half));
        rope_sin_.resize(static_cast<size_t>(target * half));
        rope_rows(cfg_, rope_positions_, target - rope_positions_, rope_cos_.data() + rope_positions_ * half,
            rope_sin_.data() + rope_positions_ * half);
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
            /*
                The reference of the probes of the device computes the products of the prompts on
                floats, the closest to the products of the device, rather than on 8-bit integers.
            */
            const bool prompt = path == qk_path::prompt
                || (path == qk_path::automatic && n >= qk_float_product_tokens);
            qk_matmul(W, X, n, x_stride, Y, y_stride, pool_, scratch_[0], false,
                host_only_ && prompt ? qk_path::floats : path);
            return;
        }
        offload_->matmul(W, X, n, x_stride, Y, y_stride,
            [&]() { qk_matmul(W, X, n, x_stride, Y, y_stride, pool_, scratch_[0], false, path); });
    }

    void cpu_runtime::add_product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y,
        long y_stride)
    {
        qk_product p;
        p.W = &W;
        p.X = X;
        p.n_tokens = n;
        p.x_stride = x_stride;
        p.Y = Y;
        p.y_stride = y_stride;
        group_.push_back(p);
    }

    void cpu_runtime::products(long n_tokens)
    {
        const long count = static_cast<long>(group_.size());
        const bool few = n_tokens < qk_float_product_tokens;
        if (few && (!offload_ || host_only_))
            qk_matmul_group(group_.data(), count, pool_, scratch_[0]);
        else if (!few || !offload_->matmul_group(group_.data(), count,
            [&]() { qk_matmul_group(group_.data(), count, pool_, scratch_[0]); }))
        {
            // The kernels of the size of the pass, which the tokens of an expert may not reach.
            const qk_path path = few ? qk_path::integer : qk_path::prompt;
            for (const qk_product& p : group_) product(*p.W, p.X, p.n_tokens, p.x_stride, p.Y, p.y_stride, path);
        }
        group_.clear();
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
        const long scales_per_row = hd / kv_q8_group;
        token_sequences_.resize(static_cast<size_t>(T));
        for (const auto& s : batch.segments)
            for (long j = 0; j < s.count; ++j) token_sequences_[static_cast<size_t>(s.start + j)] = s.sequence;
        auto one = [&](long t, int thread) {
            const int sequence = token_sequences_[static_cast<size_t>(t)];
            const kv_sequence& seq = *sequences_[static_cast<size_t>(sequence)];
            kv_pool& kv = pool_of(sequence);
            const bool q8 = kv.format() == kv_format::q8;
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
                    qk_float_to_q8_row(rotated, reinterpret_cast<int8_t*>(kv.keys(block, li, h)) + row * hd,
                        kv.key_scales(block, li, h) + row * scales_per_row, hd);
                    qk_float_to_q8_row(v, reinterpret_cast<int8_t*>(kv.values(block, li, h)) + row * hd,
                        kv.value_scales(block, li, h) + row * scales_per_row, hd);
                }
                else
                {
                    qk_float_to_half_row(k, reinterpret_cast<uint16_t*>(kv.keys(block, li, h)) + row * hd, hd);
                    qk_float_to_half_row(v, reinterpret_cast<uint16_t*>(kv.values(block, li, h)) + row * hd, hd);
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
        const long scales_per_row = hd / kv_q8_group;
        const float scale = cfg_.attention_scale > 0.0
            ? static_cast<float>(cfg_.attention_scale)
            : 1.0f / std::sqrt(static_cast<float>(hd));
        const float cap = static_cast<float>(cfg_.attn_softcap);

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
            kv_pool& kv = pool_of(tile.sequence);
            const bool q8 = kv.format() == kv_format::q8;
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
                        reinterpret_cast<const int8_t*>(kv.keys(block, layer_index, kvh)) + first * hd,
                        kv.key_scales(block, layer_index, kvh) + first * scales_per_row, m, hd, w.s.data(), B,
                        w.scratch.data());
                else
                    qk_attention_scores(w.q.data(), rows,
                        reinterpret_cast<const uint16_t*>(kv.keys(block, layer_index, kvh)) + first * hd, m, hd,
                        w.s.data(), B, w.scratch.data());
                if (cap > 0.0f)
                    for (long r = 0; r < rows; ++r) qk_softcap(w.s.data() + r * B, cap, m);
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
                        reinterpret_cast<const int8_t*>(kv.values(block, layer_index, kvh)) + first * hd,
                        kv.value_scales(block, layer_index, kvh) + first * scales_per_row, m, hd, w.o.data(),
                        w.scratch.data());
                else
                    qk_attention_values(w.s.data(), B, rows,
                        reinterpret_cast<const uint16_t*>(kv.values(block, layer_index, kvh)) + first * hd, m, hd,
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

        /*
            The rows of the experts follow one another in the buffers, in the order of the
            experts.  With the few tokens of a generation step, the experts compute their gates
            and their first products together, then their second ones; with a prompt, each
            expert in turn, in buffers that stay in the caches.
        */
        tmp_.assign(static_cast<size_t>(T * d), 0.0f);
        aligned_floats& xe = expert_x_;
        aligned_floats& ge = expert_gate_;
        aligned_floats& ue = expert_up_;
        aligned_floats& ye = expert_y_;
        const bool few = T < qk_float_product_tokens;
        const bool together = few && qk_grouped_products();
        const long reached = lay_out_experts(n_exp, together);
        xe.resize(static_cast<size_t>(reached * d));
        ge.resize(static_cast<size_t>(reached * df));
        ue.resize(static_cast<size_t>(reached * df));
        ye.resize(static_cast<size_t>(reached * d));
        auto gather = [&](long e, float* x) {
            const auto& list = routed[static_cast<size_t>(e)];
            for (size_t i = 0; i < list.size(); ++i)
                std::memcpy(x + static_cast<long>(i) * d, xn_.data() + list[i].first * d,
                    sizeof(float) * static_cast<size_t>(d));
        };
        auto scatter = [&](long e, const float* y) {
            const auto& list = routed[static_cast<size_t>(e)];
            for (size_t i = 0; i < list.size(); ++i)
            {
                const float w = list[i].second;
                float* dst = tmp_.data() + list[i].first * d;
                const float* src = y + static_cast<long>(i) * d;
                for (long j = 0; j < d; ++j) dst[j] += w * src[j];
            }
        };
        if (together)
        {
            for (long e = 0; e < n_exp; ++e)
            {
                const long first = expert_rows_[static_cast<size_t>(e)];
                if (first < 0) continue;
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                gather(e, xe.data() + first * d);
                add_product(L.e_gate[static_cast<size_t>(e)], xe.data() + first * d, m, d, ge.data() + first * df, df);
                add_product(L.e_up[static_cast<size_t>(e)], xe.data() + first * d, m, d, ue.data() + first * df, df);
            }
            products(T);
            for (long e = 0; e < n_exp; ++e)
            {
                const long first = expert_rows_[static_cast<size_t>(e)];
                if (first < 0) continue;
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                // One call per expert, as in the loop below, since the kernel ends a run of values with other code.
                qk_silu_mul(ge.data() + first * df, ue.data() + first * df, m * df);
                add_product(L.e_down[static_cast<size_t>(e)], ge.data() + first * df, m, df, ye.data() + first * d, d);
            }
            products(T);
            for (long e = 0; e < n_exp; ++e)
                if (expert_rows_[static_cast<size_t>(e)] >= 0)
                    scatter(e, ye.data() + expert_rows_[static_cast<size_t>(e)] * d);
        }
        else
        {
            // The experts follow the size of the pass, which their own tokens may not reach.
            const qk_path path = few ? qk_path::integer : qk_path::prompt;
            for (long e = 0; e < n_exp; ++e)
            {
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                if (m == 0) continue;
                gather(e, xe.data());
                product(L.e_gate[static_cast<size_t>(e)], xe.data(), m, d, ge.data(), df, path);
                product(L.e_up[static_cast<size_t>(e)], xe.data(), m, d, ue.data(), df, path);
                qk_silu_mul(ge.data(), ue.data(), m * df);
                product(L.e_down[static_cast<size_t>(e)], ge.data(), m, df, ye.data(), d, path);
                scatter(e, ye.data());
            }
        }
    }

    long cpu_runtime::lay_out_experts(long n_experts, bool together)
    {
        long rows = 0;
        if (together) expert_rows_.assign(static_cast<size_t>(n_experts), -1);
        for (long e = 0; e < n_experts; ++e)
        {
            const long m = static_cast<long>(routed_[static_cast<size_t>(e)].size());
            if (!together)
                rows = std::max(rows, m);
            else if (m > 0)
            {
                expert_rows_[static_cast<size_t>(e)] = rows;
                rows += m;
            }
        }
        return rows;
    }

    // Forward pass

    bool cpu_runtime::takes_layer_pass(const forward_batch& batch) const
    {
        // The DeepStack slices of the images are added to the hidden states between the first layers.
        return !(cfg_.n_deepstack > 0 && !batch.embedding_rows.empty());
    }

    bool cpu_runtime::plan_attention(const forward_batch& batch, bool given)
    {
        const long T = static_cast<long>(batch.tokens.size());
        const long R = cfg_.rope_dims;
        const long half = R / 2;
        attention_plan& p = plan_;
        p.pool = nullptr;
        p.positions.assign(static_cast<size_t>(T), 0);
        p.firsts.assign(static_cast<size_t>(T), 0);
        p.blocks.clear();
        p.segments.clear();
        p.angles.resize(static_cast<size_t>(T * R));
        for (const auto& s : batch.segments)
        {
            // The sequences of a pass share a pool, whose blocks the device copies.
            kv_pool& pool = pool_of(s.sequence);
            if (p.pool && p.pool != &pool) return false;
            p.pool = &pool;
            const kv_sequence& seq = *sequences_[static_cast<size_t>(s.sequence)];
            const long B = pool.block_positions();
            attention_plan::segment g;
            g.first = static_cast<long>(p.blocks.size());
            g.length = seq.length;
            g.count = s.count;
            const size_t blocks = static_cast<size_t>((seq.length + s.count + B - 1) / B);
            if (blocks > seq.blocks.size()) return false;
            p.blocks.insert(p.blocks.end(), seq.blocks.begin(), seq.blocks.begin() + static_cast<long>(blocks));
            p.segments.push_back(g);
            for (long j = 0; j < s.count; ++j)
            {
                const size_t t = static_cast<size_t>(s.start + j);
                p.positions[t] = positions_[t];
                p.firsts[t] = g.first;
            }
        }
        // The angles of each token, from the positions the batch gives or from the position of the token.
        for (long t = 0; t < T; ++t)
        {
            const size_t at = given ? static_cast<size_t>(t * half)
                : static_cast<size_t>(positions_[static_cast<size_t>(t)] * half);
            const float* c = (given ? given_cos_ : rope_cos_).data() + at;
            const float* sn = (given ? given_sin_ : rope_sin_).data() + at;
            float* a = p.angles.data() + t * R;
            std::copy(c, c + half, a);
            std::copy(sn, sn + half, a + half);
        }
        return p.pool != nullptr;
    }

    void cpu_runtime::drop_layer_pass(const std::string& why)
    {
        layer_pass_->report(false, why);
        layer_pass_.reset();
    }

    double cpu_runtime::time_pass(long n, int warm, int timed, bool on_device)
    {
        const bool hybrid = cfg_.hybrid();
        const long V = cfg_.vocab;
        const long B = kv_.block_positions();
        const long layers = hybrid ? cfg_.attention_layers() : cfg_.n_layers;
        const long tokens = n * (warm + timed);
        const size_t per_token = 2 * static_cast<size_t>(layers) * static_cast<size_t>(cfg_.n_kv_heads)
            * kv_row_bytes(kv_.format(), cfg_.head_dim);
        std::unique_ptr<kv_pool> own(new kv_pool(layers, cfg_.n_kv_heads, cfg_.head_dim, B,
            per_token * static_cast<size_t>((tokens + B - 1) / B * B), kv_.format()));
        const int seq = create_sequence();
        sequence_pools_[static_cast<size_t>(seq)] = own.get();
        struct release
        {
            cpu_runtime& rt;
            int seq;
            ~release() { rt.release_sequence(seq); }
        } guard{ *this, seq };
        if (!reserve_sequence(seq, tokens)) return -1;
        forward_batch b;
        b.segments.push_back({ seq, 0, n });
        b.logit_rows.push_back(n - 1);
        std::vector<float> logits;
        double best = -1;
        for (int pass = 0; pass < warm + timed; ++pass)
        {
            b.tokens.clear();
            for (long i = 0; i < n; ++i)
                b.tokens.push_back(static_cast<int>((2003 + 7919 * (static_cast<long>(pass) * n + i)) % V));
            const auto start = std::chrono::steady_clock::now();
            if (!(hybrid ? run_hybrid(b, logits, on_device) : run(b, logits, on_device))) return -1;
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (pass >= warm && (best < 0 || seconds < best)) best = seconds;
        }
        return best;
    }

    bool cpu_runtime::time_passes(long many, std::string& detail)
    {
        device_few_ = true;
        device_many_ = true;
        std::ostringstream o;
        o.setf(std::ios::fixed);
        o.precision(1);
        // Two passes settle the routes of the products of the processor, then the shortest of six counts.
        int warm = 2, timed = 6;
        if (cfg_.ctx_train > 0 && cfg_.ctx_train < warm + timed)
        {
            warm = cfg_.ctx_train > 1 ? 1 : 0;
            timed = static_cast<int>(std::max(1L, cfg_.ctx_train - warm));
        }
        const double few_device = time_pass(1, warm, timed, true);
        const double few_host = few_device < 0 ? -1 : time_pass(1, warm, timed, false);
        const bool few = few_device >= 0 && few_host >= 0;
        if (few)
        {
            device_few_ = few_device <= few_host;
            o << "a token in " << few_device * 1e3 << " ms on the device and " << few_host * 1e3
                << " ms on the processor";
        }
        double many_device = many > 1 ? time_pass(many, 0, 1, true) : -1;
        double many_host = many_device < 0 ? -1 : time_pass(many, 0, 1, false);
        const bool both = many_device >= 0 && many_host >= 0;
        if (both)
        {
            // Measures close to each other are taken once more, each kept at its shortest.
            if (many_device < 1.3 * many_host && many_host < 1.3 * many_device)
            {
                const double device_again = time_pass(many, 0, 1, true);
                const double host_again = time_pass(many, 0, 1, false);
                if (device_again >= 0) many_device = std::min(many_device, device_again);
                if (host_again >= 0) many_host = std::min(many_host, host_again);
            }
            device_many_ = many_device <= many_host;
            o << (few ? ", " : "") << many << " tokens in " << many_device * 1e3 << " ms and " << many_host * 1e3
                << " ms";
        }
        // A test names the passes the device keeps whatever the measures.
        if (const char* test = std::getenv("ALBIREO_TEST_TIMED_PASSES"))
        {
            const std::string kept(test);
            device_few_ = kept == "few" || kept == "both";
            device_many_ = kept == "many" || kept == "both";
            o << ", a test keeping for the device the passes named '" << kept << "'";
        }
        if (!device_few_ && !device_many_)
            o << ", the processor computing its passes faster";
        else if (!device_few_)
            o << ", the processor serving the passes of at most " << attention_tokens << " tokens, which it computes "
                "faster";
        else if (!device_many_)
            o << ", the processor serving the passes of more than " << attention_tokens << " tokens, which it "
                "computes faster";
        detail = o.str();
        return device_few_ || device_many_;
    }

    bool cpu_runtime::device_serves(const forward_batch& batch) const
    {
        return static_cast<long>(batch.tokens.size()) <= attention_tokens ? device_few_ : device_many_;
    }

    /*
        Runs n tokens in a sequence of their own through the layers of the device, then through
        the processor alone, and compares their logits: 1 when they agree, the relative distance
        of the logits of each token within tolerance, 0 when they differ, -1 when the keys and
        values or the device lack room for the probe.  The keys and values of the probe go into
        a pool of its own, in the given format, and the device attends the tokens itself when
        attention is true and it takes the pass (attended).  detail describes the comparison.
    */
    int cpu_runtime::probe_layer_pass(long n, kv_format format, bool attention, double tolerance, bool& attended,
        std::string& detail)
    {
        const long V = cfg_.vocab;
        const long B = kv_.block_positions();
        const size_t per_token = 2 * static_cast<size_t>(cfg_.n_layers) * static_cast<size_t>(cfg_.n_kv_heads)
            * kv_row_bytes(format, cfg_.head_dim);
        std::unique_ptr<kv_pool> own(new kv_pool(cfg_.n_layers, cfg_.n_kv_heads, cfg_.head_dim, B,
            per_token * static_cast<size_t>((n + B - 1) / B * B), format));
        const int seq = create_sequence();
        sequence_pools_[static_cast<size_t>(seq)] = own.get();
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
        {
            struct restore
            {
                bool& flag;
                ~restore() { flag = true; }
            } scope{ attention_allowed_ };
            attention_allowed_ = attention;
            attended_ = false;
            if (!run(b, device, true)) return -1;
            attended = attended_;
        }
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
            if (!(relative <= tolerance)) apart = true;
            if (relative > worst) worst = relative;
            if (ia == id) ++same;
        }
        std::ostringstream o;
        o << "logits of " << n << " probe tokens" << (attended ? " attended by the device" : "")
            << (format == kv_format::q8 ? " with keys and values in 8 bits" : "") << " within " << worst * 100
            << "% of those of the processor" << (apart && worst <= tolerance ? ", some not numbers" : "") << ", "
            << same << " of " << n << " most likely tokens the same";
        detail = o.str();
        /*
            Two candidates whose logits are nearly equal may change places: one row in sixteen may
            differ, one in eight with the looser tolerance of the keys and values in 8 bits.
        */
        const long misses = std::max<long>(1, tolerance > probe_tolerance ? n / 8 : n / 16);
        return !apart && same >= n - misses ? 1 : 0;
    }

    /*
        Keeps the layer pass when passes of the device agree with the processor, a defect of the
        device or of its kernels otherwise reaching every answer: a probe of probe_tokens tokens,
        whose products read the blocks of the weights in place and which the device attends
        token by token when it can, and one of tiled_tokens() tokens, whose products go through
        the tiles of the weights expanded and which the device attends over all its tokens
        together when it can.  The keys and values of these probes stay in half precision:
        rounded to 8 bits, the keys of the two passes would turn the small differences of the
        products of the device and of the processor into differences of whole steps of their
        rounding.  With keys and values in 8 bits, two more probes check the attentions of the
        device over them, within a looser tolerance: that of the few tokens, and that of the
        longer passes on a pass of long_attention_probe_tokens tokens.  The device keeps the
        layers while the attention of the processor serves the passes whose attention alone
        differs, and gives up an arithmetic of lower precision of its products before the layers
        return to the processor.  The state stays unchecked when the keys and values or the
        device lack room for a probe, so that a later pass checks again.
    */
    long cpu_runtime::probe_size() const
    {
        // A copy of the constant, which std::max() takes by reference.
        long first = probe_tokens;
        first = std::max(first, layer_pass_->min_tokens());
        if (cfg_.ctx_train > 0) first = std::min(first, cfg_.ctx_train);
        return first;
    }

    void cpu_runtime::check_layer_pass()
    {
        const long first = probe_size();
        const long tiled = layer_pass_->tiled_tokens();
        std::string details;
        auto note = [&details](const std::string& d) { details += (details.empty() ? "" : "; ") + d; };
        std::string detail;
        bool attended = false;

        // The passes of a few tokens, which the device attends itself when it can.
        bool attention = layer_pass_->attends() && first <= attention_tokens;
        // Without passes of a few tokens on the device, its attention would serve none.
        if (!attention) layer_pass_->allow_attention(false);
        int outcome = probe_layer_pass(first, kv_format::f16, attention, probe_tolerance, attended, detail);
        if (outcome < 0) return;
        if (attention && !attended)
        {
            // No room for the copies of the keys and values: the probe checked the products alone.
            note("no room on the device for its copies of the keys and values, the attention of the processor "
                "serving until a later probe");
            layer_pass_->allow_attention(false);
            attention = false;
            attention_interval_ = layer_pass_check_interval;
            attention_wait_ = attention_interval_;
        }
        else if (attention && outcome == 0)
        {
            // The attention of the device differs, or its products do: the products are checked alone.
            note(detail + ", the attention of the processor serving");
            layer_pass_->allow_attention(false);
            attention = false;
            outcome = probe_layer_pass(first, kv_format::f16, false, probe_tolerance, attended, detail);
            if (outcome < 0) return;
        }
        note(detail);
        if (outcome == 0)
        {
            drop_layer_pass(details);
            return;
        }

        /*
            The longer passes, whose products go through the tiles, and which the device attends
            over all their tokens together when it can.  When such a pass differs, the device
            first leaves its attention to the processor.  When the pass still differs, the device
            gives up an arithmetic of lower precision of its products when it used one
            (layer_pass::retreat()) and takes back the attention it left, which that arithmetic
            may have been alone to disturb.  The pass is compared again each time.
        */
        long_attention_differs_ = false;
        if (tiled > first && (cfg_.ctx_train <= 0 || tiled <= cfg_.ctx_train))
        {
            bool together = attention && layer_pass_->attends_many();
            // Whether a pass that differed left the attention of the longer passes to the processor.
            bool left = false;
            for (;;)
            {
                outcome = probe_layer_pass(tiled, kv_format::f16, together, probe_tolerance, attended, detail);
                if (outcome < 0) return;
                if (together && !attended)
                {
                    // No room for the work of that attention: the probe checked the products alone.
                    note("no room on the device to attend the longer passes, whose attention stays on the processor");
                    layer_pass_->allow_long_attention(false);
                    together = false;
                }
                if (outcome != 0) break;
                if (together)
                {
                    note(detail + ", the attention of the processor serving the longer passes");
                    long_attention_differs_ = true;
                    layer_pass_->allow_long_attention(false);
                    together = false;
                    left = true;
                    continue;
                }
                if (!layer_pass_->retreat()) break;
                if (left)
                {
                    note(detail + ", the products of the device returning to single precision and the device "
                        "attending the longer passes again");
                    long_attention_differs_ = false;
                    layer_pass_->allow_long_attention(true);
                    together = layer_pass_->attends_many();
                    left = false;
                }
                else
                {
                    note(detail + ", the products of the device returning to single precision");
                }
            }
            note(detail);
            if (outcome == 0)
            {
                drop_layer_pass(details);
                return;
            }
        }
        else
        {
            // Without a longer probe, the attention of the longer passes stays unchecked: the processor keeps it.
            layer_pass_->allow_long_attention(false);
        }

        if (attention && kv_.format() == kv_format::q8)
        {
            outcome = probe_layer_pass(first, kv_format::q8, true, q8_probe_tolerance, attended, detail);
            if (outcome < 0) return;
            if (outcome == 0 || !attended)
            {
                note(attended ? detail + ", the attention of the processor serving"
                    : "no room on the device for its copies of the keys and values in 8 bits, the attention of the "
                    "processor serving until a later probe");
                layer_pass_->allow_attention(false);
                if (!attended)
                {
                    attention_interval_ = layer_pass_check_interval;
                    attention_wait_ = attention_interval_;
                }
            }
            else
            {
                note(detail);
            }
        }
        // The attention of the longer passes over keys and values in 8 bits, on a shorter pass.
        const long longer = long_attention_probe_tokens;
        if (layer_pass_->attends_many() && kv_.format() == kv_format::q8 && longer > attention_tokens
            && (cfg_.ctx_train <= 0 || longer <= cfg_.ctx_train))
        {
            outcome = probe_layer_pass(longer, kv_format::q8, true, q8_probe_tolerance, attended, detail);
            if (outcome < 0) return;
            if (outcome == 0 || !attended)
            {
                note(attended ? detail + ", the attention of the processor serving the longer passes"
                    : "no room on the device to attend the longer passes over keys and values in 8 bits");
                long_attention_differs_ = attended;
                layer_pass_->allow_long_attention(false);
            }
            else
            {
                note(detail);
            }
        }
        if (layer_pass_->timed())
        {
            const long many = tiled > first && (cfg_.ctx_train <= 0 || tiled <= cfg_.ctx_train) ? tiled : 0;
            const bool kept = time_passes(many, detail);
            if (!detail.empty()) note(detail);
            if (!kept)
            {
                drop_layer_pass(details);
                return;
            }
            if (!device_few_ && !layer_pass_->attends_many())
            {
                // The device attended the passes of a few tokens alone, which it leaves to the processor.
                layer_pass_->allow_attention(false);
                attention_wait_ = 0;
            }
        }
        layer_pass_checked_ = true;
        layer_pass_->report(true, details);
    }

    /*
        The probes of the attention of the device, as check_layer_pass() makes them, once the
        layers of the device serve: the device attends the passes of a few tokens when it agrees
        with the processor, keeps the attention with the processor when it differs, and probes
        again later, twice as late, when its copies of the keys and values find no room again.
    */
    void cpu_runtime::check_attention()
    {
        const long first = probe_size();
        layer_pass_->allow_attention(true);
        std::string details, detail;
        bool attended = false;
        int outcome = probe_layer_pass(first, kv_format::f16, true, probe_tolerance, attended, detail);
        if (outcome > 0 && attended && kv_.format() == kv_format::q8)
        {
            details = detail;
            outcome = probe_layer_pass(first, kv_format::q8, true, q8_probe_tolerance, attended, detail);
        }
        if (outcome > 0 && attended)
        {
            // allow_attention() lets the device attend every pass again: the longer ones stay as their probe left them.
            if (long_attention_differs_) layer_pass_->allow_long_attention(false);
            details += (details.empty() ? "" : "; ") + detail;
            layer_pass_->report(true, details);
            return;
        }
        layer_pass_->allow_attention(false);
        if (outcome < 0 || !attended)
        {
            // A copy of the constant, which std::min() takes by reference.
            const int most = max_attention_check_interval;
            attention_interval_ = std::min(2 * attention_interval_, most);
            attention_wait_ = attention_interval_;
            return;
        }
        layer_pass_->report(true, detail + ", the attention of the processor serving");
    }

    void cpu_runtime::forward(const forward_batch& batch, std::vector<float>& logits)
    {
        if (cfg_.hybrid())
        {
            if (hybrid_pass_ && !batch.tokens.empty())
            {
                try
                {
                    // A check that the keys and values or the device put off waits for a few passes.
                    if (!layer_pass_checked_ && layer_pass_wait_ > 0)
                        --layer_pass_wait_;
                    else if (!layer_pass_checked_)
                    {
                        check_hybrid_pass();
                        if (hybrid_pass_ && !layer_pass_checked_) layer_pass_wait_ = layer_pass_check_interval;
                    }
                    if (hybrid_pass_ && layer_pass_checked_ && device_serves(batch) && run_hybrid(batch, logits, true))
                    {
                        layer_pass_memory_failures_ = 0;
                        return;
                    }
                }
                catch (const device_failure& e)
                {
                    // The pass is computed again below, from the states its sequences had before it.
                    if (e.changed)
                        ;
                    else if (hybrid_pass_ && !e.memory)
                        drop_hybrid_pass(std::string("the device failed: ") + e.what());
                    else if (hybrid_pass_ && ++layer_pass_memory_failures_ >= max_layer_pass_memory_failures)
                        drop_hybrid_pass(std::string("the device lacked memory several times in a row: ") + e.what());
                }
            }
            run_hybrid(batch, logits, false);
            return;
        }
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
                else if (attention_wait_ > 0 && --attention_wait_ == 0)
                {
                    check_attention();
                }
                if (layer_pass_ && layer_pass_checked_ && device_serves(batch) && run(batch, logits, true))
                {
                    layer_pass_memory_failures_ = 0;
                    return;
                }
            }
            catch (const device_failure& e)
            {
                /*
                    The pass is computed again below; its keys and values are written again.  A
                    device that changed its arithmetic stays in use.
                */
                if (e.changed)
                    ;
                else if (!e.memory)
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
            if (s.count > 0 && pool_of(s.sequence).shared(seq.blocks[static_cast<size_t>(seq.length / B)]))
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
        /*
            A pass runs whole on the device when the device attends it: token by token for a pass
            of a few tokens, as a step of a generation, and over all its tokens together for a
            longer one, as a chunk of a prompt.
        */
        const bool attending = on_device && attention_allowed_
            && (T <= attention_tokens ? layer_pass_->attends() : layer_pass_->attends_many())
            && plan_attention(batch, given) && layer_pass_->begin_attention(plan_);
        attended_ = attending;

        for (long li = 0; li < cfg_.n_layers; ++li)
        {
            const layer& L = layers_[static_cast<size_t>(li)];
            if (attending)
            {
                layer_pass_->attend(li);
                continue;
            }

            // Attention block.
            if (on_device)
                layer_pass_->attention_inputs(li, q_.data(), k_.data(), v_.data());
            else
            {
                rms_norm_rows(x_.data(), L.attn_norm.data(), xn_.data(), T, d, eps);
                // The queries, keys and values read the same rows and nothing of one another.
                add_product(L.wq, xn_.data(), T, d, q_.data(), qd);
                add_product(L.wk, xn_.data(), T, d, k_.data(), kvd);
                add_product(L.wv, xn_.data(), T, d, v_.data(), kvd);
                products(T);
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
                add_product(L.w_gate, xn_.data(), T, d, gate_.data(), df);
                add_product(L.w_up, xn_.data(), T, d, up_.data(), df);
                products(T);
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

    // Hybrid decoders

    void cpu_runtime::geodesic_update(const float* g, float scale, float bias, long layer, long T)
    {
        /*
            Each hidden state x turns on its sphere towards the part of g that is tangent to it,
            by the angle the size of that part gives, a quarter of a turn of the half plane at
            most, scaled, shifted and divided by the depth of the layer.  The sums run in double
            precision: the outputs of the mixers reach 1e14.
        */
        const long d = cfg_.d_model;
        const double quarter = pi / 4.0;
        auto one = [&](long t) {
            float* x = x_.data() + t * d;
            const float* u = g + t * d;
            double xx = 0.0, xu = 0.0;
            for (long i = 0; i < d; ++i)
            {
                xx += static_cast<double>(x[i]) * x[i];
                xu += static_cast<double>(x[i]) * u[i];
            }
            const double along = xu / std::max(xx, 1e-12);
            double tt = 0.0;
            for (long i = 0; i < d; ++i)
            {
                const double v = u[i] - along * x[i];
                tt += v * v;
            }
            const double norm = std::max(std::sqrt(tt), 1e-8);
            const double radius = std::max(std::sqrt(xx), 1e-6);
            double theta = std::min(norm / radius, quarter);
            theta = std::min((theta * scale + bias) / static_cast<double>(layer + 1), quarter);
            const double c = std::cos(theta), s = std::sin(theta) * radius / norm;
            for (long i = 0; i < d; ++i) x[i] = static_cast<float>(x[i] * c + (u[i] - along * x[i]) * s);
        };
        if (T >= 16)
            pool_.parallel_for(T, [&](long t, int) { one(t); });
        else
            for (long t = 0; t < T; ++t) one(t);
    }

    void cpu_runtime::mamba3_mixer(long li, const forward_batch& batch)
    {
        const layer& L = layers_[static_cast<size_t>(li)];
        const long T = static_cast<long>(batch.tokens.size());
        const long d = cfg_.d_model;
        const long H = cfg_.ssm_heads, P = cfg_.ssm_head_dim, N = cfg_.ssm_state, R = cfg_.ssm_rank;
        const long S = cfg_.ssm_angles;
        const long head_row = 2 * P + 3;
        const long in_width = H * head_row;
        const long dyn_width = 2 * R * N + S;
        const long inner = H * P;
        const float eps = static_cast<float>(cfg_.rms_eps);

        proj_.resize(static_cast<size_t>(T * in_width));
        dyn_.resize(static_cast<size_t>(T * dyn_width));
        add_product(L.ssm_in, x_.data(), T, d, proj_.data(), in_width);
        add_product(L.ssm_dyn, x_.data(), T, d, dyn_.data(), dyn_width);
        products(T);

        // The inputs and the outputs of the states, normalized, and the turn of each angle for a step of one.
        turns_.resize(static_cast<size_t>(T * S));
        for (long t = 0; t < T; ++t)
        {
            float* row = dyn_.data() + t * dyn_width;
            rms_norm_rows(row, L.ssm_b_norm.data(), row, R, N, eps);
            rms_norm_rows(row + R * N, L.ssm_c_norm.data(), row + R * N, R, N, eps);
            for (long i = 0; i < S; ++i)
                turns_[static_cast<size_t>(t * S + i)] = std::tanh(static_cast<double>(row[2 * R * N + i])) * pi;
        }

        ctx_.resize(static_cast<size_t>(T * inner));
        const long n_segments = static_cast<long>(batch.segments.size());
        const size_t row_bytes = sizeof(float) * static_cast<size_t>(R * N);
        // A head reads the tokens of a sequence in order; the heads and the sequences run in parallel.
        pool_.parallel_for(n_segments * H, [&](long item, int thread) {
            const forward_batch::segment& seg = batch.segments[static_cast<size_t>(item / H)];
            const long h = item % H;
            mixer_work& w = mixer_work_[static_cast<size_t>(thread)];
            float* state = recurrent_[static_cast<size_t>(seg.sequence)].live.values.data()
                + state_offsets_[static_cast<size_t>(li)];
            float* angle = state + h * S;
            float* channels = state + H * S + h * P * N;
            float* k_before = state + H * S + H * P * N + h * R * N;
            float* x_before = state + H * S + H * P * N + H * R * N + h * P;
            const float* b_bias = L.ssm_b_bias.data() + h * R * N;
            const float* c_bias = L.ssm_c_bias.data() + h * R * N;
            const float* mx = L.mimo_x.data() + h * R * P;
            const float* mz = L.mimo_z.data() + h * R * P;
            const float* mo = L.mimo_o.data() + h * R * P;
            const float skip = L.ssm_d[static_cast<size_t>(h)];
            for (long j = 0; j < seg.count; ++j)
            {
                const long t = seg.start + j;
                const float* z = proj_.data() + t * in_width + h * head_row;
                const float* x = z + P;
                const double step = softplus(static_cast<double>(x[P]) + L.ssm_dt_bias[static_cast<size_t>(h)]);
                const double decay = std::min(-softplus(static_cast<double>(x[P + 1])), -1e-4);
                const double trapezoid = sigmoid(static_cast<double>(x[P + 2]));
                const double kept = std::exp(decay * step);
                const float alpha = static_cast<float>(kept);
                const float beta = static_cast<float>((1.0 - trapezoid) * step * kept);
                const float gamma = static_cast<float>(trapezoid * step);

                /*
                    The angles of the head, kept within one turn.  The cosines and the sines come
                    from the angles as the state holds them: a sequence that takes back a saved
                    state computes the same numbers.
                */
                const double* turn = turns_.data() + t * S;
                for (long i = 0; i < S; ++i)
                {
                    double sum = angle[i] + turn[i] * step;
                    sum -= 2.0 * pi * std::floor((sum + pi) / (2.0 * pi));
                    angle[i] = static_cast<float>(sum);
                    w.cos[static_cast<size_t>(i)] = std::cos(angle[i]);
                    w.sin[static_cast<size_t>(i)] = std::sin(angle[i]);
                }

                // The inputs and the outputs of the states of the head: their biases, then the rotation.
                const float* bn = dyn_.data() + t * dyn_width;
                const float* cn = bn + R * N;
                for (long r = 0; r < R; ++r)
                {
                    float* k = w.k.data() + r * N;
                    float* q = w.q.data() + r * N;
                    for (long n = 0; n < N; ++n)
                    {
                        k[n] = bn[r * N + n] + b_bias[r * N + n];
                        q[n] = cn[r * N + n] + c_bias[r * N + n];
                    }
                    rotate_halves(k, w.cos.data(), w.sin.data(), N, S);
                    rotate_halves(q, w.cos.data(), w.sin.data(), N, S);
                }

                /*
                    The state of each channel takes its inputs by the trapezoid rule, the input of
                    this token and the one of the previous token, and gives its outputs, which
                    the input of the channel joins directly.
                */
                for (long p = 0; p < P; ++p)
                {
                    for (long r = 0; r < R; ++r)
                    {
                        w.now[static_cast<size_t>(r)] = gamma * x[p] * mx[r * P + p];
                        w.before[static_cast<size_t>(r)] = beta * x_before[p] * mx[r * P + p];
                    }
                    qk_state_step(channels + p * N, alpha, w.before.data(), k_before, w.now.data(), w.k.data(),
                        w.q.data(), R, N, w.y.data());
                    for (long r = 0; r < R; ++r)
                    {
                        w.o[static_cast<size_t>(r * P + p)] = w.y[static_cast<size_t>(r)] + skip * x[p] * mx[r * P + p];
                        w.gate[static_cast<size_t>(r * P + p)] = z[p] * mz[r * P + p];
                    }
                }
                qk_silu_mul(w.gate.data(), w.o.data(), R * P);
                float* out = ctx_.data() + t * inner + h * P;
                for (long p = 0; p < P; ++p)
                {
                    float sum = 0.0f;
                    for (long r = 0; r < R; ++r) sum += w.gate[static_cast<size_t>(r * P + p)] * mo[r * P + p];
                    out[p] = sum;
                }

                // What the next token reads of this one.
                std::memcpy(k_before, w.k.data(), row_bytes);
                std::memcpy(x_before, x, sizeof(float) * static_cast<size_t>(P));
            }
        });

        tmp_.resize(static_cast<size_t>(T * d));
        product(L.wo, ctx_.data(), T, inner, tmp_.data(), d);
    }

    void cpu_runtime::differential_attention(long li, const forward_batch& batch, bool on_device)
    {
        const layer& L = layers_[static_cast<size_t>(li)];
        const long T = static_cast<long>(batch.tokens.size());
        const long d = cfg_.d_model;
        const long H = cfg_.n_heads, KV = cfg_.n_kv_heads, hd = cfg_.head_dim;
        const long qd = cfg_.q_dim(), kvd = cfg_.kv_dim();
        const long rank = cfg_.tpa_rank;
        const long group = H / KV;
        const float eps = static_cast<float>(cfg_.rms_eps);

        /*
            The products that read the hidden states, none of which reads another: the queries,
            the factors of the keys then those of the values, the shifts, the share of the
            noise and the gate of the layer.
        */
        const long signal = group - 1;
        const long width = cfg_.signal_heads() * hd;
        const long fa = T * KV * rank, fb = T * rank * hd;
        q_.resize(static_cast<size_t>(T * qd));
        k_.resize(static_cast<size_t>(T * kvd));
        v_.resize(static_cast<size_t>(T * kvd));
        factor_a_.resize(static_cast<size_t>(2 * fa));
        factor_b_.resize(static_cast<size_t>(2 * fb));
        shift_.resize(static_cast<size_t>(T * 2 * KV));
        noise_.resize(static_cast<size_t>(T * KV));
        gate_.resize(static_cast<size_t>(T * width));
        if (on_device)
        {
            // The device holds the hidden states, and sends these products in one copy.
            hybrid_pass_->attention_inputs(li, q_.data(), factor_a_.data(), factor_b_.data(), shift_.data(),
                noise_.data(), gate_.data());
        }
        else
        {
            add_product(L.wq, x_.data(), T, d, q_.data(), qd);
            add_product(L.wa_k, x_.data(), T, d, factor_a_.data(), KV * rank);
            add_product(L.wb_k, x_.data(), T, d, factor_b_.data(), rank * hd);
            add_product(L.wa_v, x_.data(), T, d, factor_a_.data() + fa, KV * rank);
            add_product(L.wb_v, x_.data(), T, d, factor_b_.data() + fb, rank * hd);
            add_product(L.shift_k, x_.data(), T, d, shift_.data(), 2 * KV);
            add_product(L.shift_v, x_.data(), T, d, shift_.data() + KV, 2 * KV);
            add_product(L.noise, x_.data(), T, d, noise_.data(), KV);
            add_product(L.attn_gate, x_.data(), T, d, gate_.data(), width);
            products(T);
        }

        // A key, or a value, of a head: the mean of rank products of a factor of the head and a row of its width.
        auto tensor_products = [&](const float* factors_a, const float* factors_b, float* out) {
            const float inv = 1.0f / static_cast<float>(rank);
            for (long t = 0; t < T; ++t)
                for (long h = 0; h < KV; ++h)
                {
                    float* row = out + t * kvd + h * hd;
                    std::fill(row, row + hd, 0.0f);
                    for (long r = 0; r < rank; ++r)
                    {
                        const float a = factors_a[(t * KV + h) * rank + r] * inv;
                        const float* b = factors_b + (t * rank + r) * hd;
                        for (long e = 0; e < hd; ++e) row[e] += a * b[e];
                    }
                }
        };
        tensor_products(factor_a_.data(), factor_b_.data(), k_.data());
        tensor_products(factor_a_.data() + fa, factor_b_.data() + fb, v_.data());

        /*
            The shift: each key and each value leans on the one of the previous token of its
            sequence, as it stood before its own shift.  The tokens of a segment are taken from
            the last one, so that the row before each still holds what it held.
        */
        last_keys_.resize(static_cast<size_t>(2 * kvd));
        for (const auto& seg : batch.segments)
        {
            if (seg.count <= 0) continue;
            float* last = recurrent_[static_cast<size_t>(seg.sequence)].live.values.data()
                + state_offsets_[static_cast<size_t>(li)];
            const long end = seg.start + seg.count - 1;
            std::copy(k_.data() + end * kvd, k_.data() + (end + 1) * kvd, last_keys_.data());
            std::copy(v_.data() + end * kvd, v_.data() + (end + 1) * kvd, last_keys_.data() + kvd);
            for (long t = end; t >= seg.start; --t)
            {
                const float* k_before = t > seg.start ? k_.data() + (t - 1) * kvd : last;
                const float* v_before = t > seg.start ? v_.data() + (t - 1) * kvd : last + kvd;
                if (t == seg.start && positions_[static_cast<size_t>(t)] == 0) break;
                for (long h = 0; h < KV; ++h)
                {
                    const float ak = static_cast<float>(sigmoid(shift_[static_cast<size_t>(t * 2 * KV + h)]));
                    const float av = static_cast<float>(sigmoid(shift_[static_cast<size_t>(t * 2 * KV + KV + h)]));
                    float* k = k_.data() + t * kvd + h * hd;
                    float* v = v_.data() + t * kvd + h * hd;
                    for (long e = 0; e < hd; ++e)
                    {
                        k[e] = ak * k_before[h * hd + e] + (1.0f - ak) * k[e];
                        v[e] = av * v_before[h * hd + e] + (1.0f - av) * v[e];
                    }
                }
            }
            std::copy(last_keys_.begin(), last_keys_.end(), last);
        }

        rms_norm_rows(q_.data(), L.q_norm.data(), q_.data(), T * H, hd, eps);
        rms_norm_rows(k_.data(), L.k_norm.data(), k_.data(), T * KV, hd, eps);
        // The queries grow with the logarithm of the positions they see, each head by its own factor.
        for (long t = 0; t < T; ++t)
        {
            long seen = positions_[static_cast<size_t>(t)] + 1;
            if (cfg_.attention_window > 0) seen = std::min(seen, cfg_.attention_window);
            const float reach = static_cast<float>(std::log(static_cast<double>(seen)));
            for (long h = 0; h < H; ++h)
                qk_scale(q_.data() + t * qd + h * hd, L.softmax_scaler[static_cast<size_t>(h)] * reach, hd);
        }

        const long pool_layer = kv_layers_[static_cast<size_t>(li)];
        store_keys_values(pool_layer, batch);
        attention(pool_layer, batch, positions_);

        /*
            Each key and value head serves a group of query heads: the last one of the group
            reads the noise, which the others give up in the proportion their token chooses;
            the gate of the layer then opens each value of the result.
        */
        up_.resize(static_cast<size_t>(T * width));
        const float gate_bias = static_cast<float>(cfg_.gate_bias);
        if (gate_bias != 0.0f)
            for (float& v : gate_) v += gate_bias;
        for (long t = 0; t < T; ++t)
            for (long g = 0; g < KV; ++g)
            {
                const float share = static_cast<float>(sigmoid(noise_[static_cast<size_t>(t * KV + g)]));
                const float* heads = ctx_.data() + t * qd + g * group * hd;
                const float* noise = heads + signal * hd;
                float* out = up_.data() + t * width + g * signal * hd;
                for (long i = 0; i < signal; ++i)
                    for (long e = 0; e < hd; ++e) out[i * hd + e] = heads[i * hd + e] - share * noise[e];
            }
        qk_silu_mul(gate_.data(), up_.data(), T * width);

        if (on_device)
        {
            // The device projects the result and turns the hidden states towards it.
            hybrid_pass_->attention_output(li, gate_.data());
            return;
        }
        tmp_.resize(static_cast<size_t>(T * d));
        product(L.wo, gate_.data(), T, width, tmp_.data(), d);
    }

    void cpu_runtime::latent_experts(const layer& L, long T)
    {
        const long d = cfg_.d_model;
        const long n_exp = cfg_.n_experts;
        const long top_k = cfg_.n_experts_used;
        const long df = cfg_.d_ffn_expert;
        const long dl = cfg_.d_latent;
        const long ds = cfg_.d_ffn_shared;
        const float weights_scale = static_cast<float>(cfg_.expert_weights_scale);
        auto squared_relu = [](float* v, long n) {
            for (long i = 0; i < n; ++i)
            {
                const float r = v[i] > 0.0f ? v[i] : 0.0f;
                v[i] = r * r;
            }
        };

        // The products that read the hidden states: the router, the entry of the latent space and the shared expert.
        router_logits_.resize(static_cast<size_t>(T * n_exp));
        latent_.resize(static_cast<size_t>(T * dl));
        gate_.resize(static_cast<size_t>(T * ds));
        add_product(L.router, x_.data(), T, d, router_logits_.data(), n_exp);
        add_product(L.latent_down, x_.data(), T, d, latent_.data(), dl);
        add_product(L.shared_up, x_.data(), T, d, gate_.data(), ds);
        products(T);

        /*
            The router gives each expert a sigmoid; a token takes the experts of the largest
            sigmoids once their biases are added, with the sigmoids themselves, brought to a sum
            of one, as their weights.
        */
        std::vector<std::vector<std::pair<long, float>>>& routed = routed_;
        routed.resize(static_cast<size_t>(n_exp));
        for (auto& list : routed) list.clear();
        std::vector<float>& scores = router_probs_;
        scores.resize(static_cast<size_t>(2 * n_exp));
        float* ranked = scores.data() + n_exp;
        std::vector<long>& order = router_order_;
        order.resize(static_cast<size_t>(n_exp));
        for (long t = 0; t < T; ++t)
        {
            const float* row = router_logits_.data() + t * n_exp;
            for (long e = 0; e < n_exp; ++e)
            {
                scores[static_cast<size_t>(e)] = static_cast<float>(sigmoid(row[e]));
                ranked[e] = scores[static_cast<size_t>(e)] + L.expert_bias[static_cast<size_t>(e)];
                order[static_cast<size_t>(e)] = e;
            }
            std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                [&](long a, long b) { return ranked[a] > ranked[b] || (ranked[a] == ranked[b] && a < b); });
            float selected = 0.0f;
            for (long k = 0; k < top_k; ++k) selected += scores[static_cast<size_t>(order[static_cast<size_t>(k)])];
            for (long k = 0; k < top_k; ++k)
            {
                const long e = order[static_cast<size_t>(k)];
                float w = scores[static_cast<size_t>(e)];
                if (cfg_.expert_weights_norm) w /= selected + 1e-20f;
                routed[static_cast<size_t>(e)].push_back(std::make_pair(t, w * weights_scale));
            }
        }

        /*
            The experts work in the latent space, which one projection enters and one leaves.
            Their rows follow one another in the buffers, in the order of the experts.  With the
            few tokens of a generation step, the experts compute their first products together,
            then their second ones; with a prompt, each expert in turn, in buffers that stay in
            the caches.
        */
        routed_latent_.assign(static_cast<size_t>(T * dl), 0.0f);
        aligned_floats& xe = expert_x_;
        aligned_floats& ue = expert_up_;
        aligned_floats& ye = expert_y_;
        const bool few = T < qk_float_product_tokens;
        const bool together = few && qk_grouped_products();
        const long reached = lay_out_experts(n_exp, together);
        xe.resize(static_cast<size_t>(reached * dl));
        ue.resize(static_cast<size_t>(reached * df));
        ye.resize(static_cast<size_t>(reached * dl));
        auto gather = [&](long e, float* x) {
            const auto& list = routed[static_cast<size_t>(e)];
            for (size_t i = 0; i < list.size(); ++i)
                std::memcpy(x + static_cast<long>(i) * dl, latent_.data() + list[i].first * dl,
                    sizeof(float) * static_cast<size_t>(dl));
        };
        auto scatter = [&](long e, const float* y) {
            const auto& list = routed[static_cast<size_t>(e)];
            for (size_t i = 0; i < list.size(); ++i)
            {
                const float w = list[i].second;
                float* dst = routed_latent_.data() + list[i].first * dl;
                const float* src = y + static_cast<long>(i) * dl;
                for (long j = 0; j < dl; ++j) dst[j] += w * src[j];
            }
        };
        if (together)
        {
            for (long e = 0; e < n_exp; ++e)
            {
                const long first = expert_rows_[static_cast<size_t>(e)];
                if (first < 0) continue;
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                gather(e, xe.data() + first * dl);
                add_product(L.e_up[static_cast<size_t>(e)], xe.data() + first * dl, m, dl, ue.data() + first * df, df);
            }
            products(T);
            squared_relu(ue.data(), reached * df);
            for (long e = 0; e < n_exp; ++e)
            {
                const long first = expert_rows_[static_cast<size_t>(e)];
                if (first < 0) continue;
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                add_product(L.e_down[static_cast<size_t>(e)], ue.data() + first * df, m, df, ye.data() + first * dl,
                    dl);
            }
            products(T);
            for (long e = 0; e < n_exp; ++e)
                if (expert_rows_[static_cast<size_t>(e)] >= 0)
                    scatter(e, ye.data() + expert_rows_[static_cast<size_t>(e)] * dl);
        }
        else
        {
            // The experts follow the size of the pass, which their own tokens may not reach.
            const qk_path path = few ? qk_path::integer : qk_path::prompt;
            for (long e = 0; e < n_exp; ++e)
            {
                const long m = static_cast<long>(routed[static_cast<size_t>(e)].size());
                if (m == 0) continue;
                gather(e, xe.data());
                product(L.e_up[static_cast<size_t>(e)], xe.data(), m, dl, ue.data(), df, path);
                squared_relu(ue.data(), m * df);
                product(L.e_down[static_cast<size_t>(e)], ue.data(), m, df, ye.data(), dl, path);
                scatter(e, ye.data());
            }
        }

        // The exit of the latent space, and the shared expert, which reads every token.
        tmp_.resize(static_cast<size_t>(T * d));
        up_.resize(static_cast<size_t>(T * d));
        squared_relu(gate_.data(), T * ds);
        add_product(L.latent_up, routed_latent_.data(), T, dl, tmp_.data(), d);
        add_product(L.shared_down, gate_.data(), T, ds, up_.data(), d);
        products(T);
        qk_add(tmp_.data(), up_.data(), T * d);
    }

    bool cpu_runtime::fetch_states(int seq)
    {
        if (!cfg_.hybrid()) return true;
        recurrent_sequence& r = recurrent_.at(static_cast<size_t>(seq));
        if (r.lost) return false;
        if (r.on_host) return true;
        if (!hybrid_pass_ || !hybrid_pass_->holds(seq))
        {
            r.lost = true;
            return false;
        }
        try
        {
            hybrid_pass_->fetch(seq, r.live.values.data());
        }
        catch (const device_failure&)
        {
            r.lost = true;
            return false;
        }
        r.on_host = true;
        return true;
    }

    void cpu_runtime::fetch_all_states()
    {
        for (size_t seq = 0; seq < recurrent_.size(); ++seq)
            if (!recurrent_[seq].on_host && !recurrent_[seq].lost) fetch_states(static_cast<int>(seq));
    }

    void cpu_runtime::save_attention_states(const forward_batch& batch)
    {
        const size_t kept = 2 * static_cast<size_t>(cfg_.kv_dim());
        attention_states_.clear();
        for (const auto& s : batch.segments)
        {
            const float* values = recurrent_[static_cast<size_t>(s.sequence)].live.values.data();
            for (long li = 0; li < cfg_.n_layers; ++li)
                if (cfg_.layer_types[static_cast<size_t>(li)] != 'M')
                    attention_states_.insert(attention_states_.end(), values + state_offsets_[static_cast<size_t>(li)],
                        values + state_offsets_[static_cast<size_t>(li)] + kept);
        }
    }

    /*
        After a pass the device failed: what the attention layers kept of the last token of each
        sequence returns, and the states of the Mamba-3 layers come back from the copy the device
        made before the pass.  A sequence whose states the device cannot give is marked lost.
    */
    void cpu_runtime::restore_states(const forward_batch& batch)
    {
        const size_t kept = 2 * static_cast<size_t>(cfg_.kv_dim());
        size_t at = 0;
        for (const auto& s : batch.segments)
        {
            recurrent_sequence& r = recurrent_[static_cast<size_t>(s.sequence)];
            float* values = r.live.values.data();
            for (long li = 0; li < cfg_.n_layers; ++li)
            {
                if (cfg_.layer_types[static_cast<size_t>(li)] == 'M') continue;
                if (at + kept <= attention_states_.size())
                    std::copy(attention_states_.begin() + static_cast<std::ptrdiff_t>(at),
                        attention_states_.begin() + static_cast<std::ptrdiff_t>(at + kept),
                        values + state_offsets_[static_cast<size_t>(li)]);
                at += kept;
            }
            if (hybrid_pass_ && hybrid_pass_->recover(s.sequence, values))
                r.on_host = true;
            else
                r.lost = true;
        }
    }

    void cpu_runtime::drop_hybrid_pass(const std::string& why)
    {
        // The states the device alone holds come back while it may still give them.
        fetch_all_states();
        hybrid_pass_->report(false, why);
        hybrid_pass_.reset();
    }

    /*
        probe_layer_pass() for a hybrid decoder: n tokens in a sequence of their own through the
        layers of the device, then through the processor alone, and the comparison of their
        logits.  The attempts read different tokens.
    */
    int cpu_runtime::probe_hybrid_pass(long n, int attempt, std::string& detail)
    {
        const long V = cfg_.vocab;
        const long B = kv_.block_positions();
        const size_t per_token = 2 * static_cast<size_t>(cfg_.attention_layers())
            * static_cast<size_t>(cfg_.n_kv_heads) * kv_row_bytes(kv_.format(), cfg_.head_dim);
        std::unique_ptr<kv_pool> own(new kv_pool(cfg_.attention_layers(), cfg_.n_kv_heads, cfg_.head_dim, B,
            per_token * static_cast<size_t>((n + B - 1) / B * B), kv_.format()));
        const int seq = create_sequence();
        sequence_pools_[static_cast<size_t>(seq)] = own.get();
        struct release
        {
            cpu_runtime& rt;
            int seq;
            ~release() { rt.release_sequence(seq); }
        } guard{ *this, seq };
        if (!reserve_sequence(seq, n)) return -1;
        forward_batch b;
        for (long i = 0; i < n; ++i)
            b.tokens.push_back(static_cast<int>((1009 + 7919 * (i + 131 * static_cast<long>(attempt))) % V));
        b.segments.push_back({ seq, 0, n });
        for (long i = 0; i < n; ++i) b.logit_rows.push_back(i);
        std::vector<float> reference, device;
        if (!run_hybrid(b, device, true)) return -1;
        // The truncation gives the blocks back and starts the states again.
        truncate_sequence(seq, 0);
        if (!reserve_sequence(seq, n)) return -1;
        {
            struct processor_only
            {
                bool& flag;
                ~processor_only() { flag = false; }
            } scope{ host_only_ };
            host_only_ = true;
            run_hybrid(b, reference, false);
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
            if (!(relative <= probe_tolerance)) apart = true;
            if (relative > worst) worst = relative;
            if (ia == id) ++same;
        }
        std::ostringstream o;
        o << "logits of " << n << " probe tokens within " << worst * 100 << "% of those of the processor"
            << (apart && worst <= probe_tolerance ? ", some not numbers" : "") << ", " << same << " of " << n
            << " most likely tokens the same";
        detail = o.str();
        const long misses = std::max<long>(1, n / 16);
        return !apart && same >= n - misses ? 1 : 0;
    }

    /*
        Keeps the hybrid pass when passes of the device agree with the processor: a probe of
        probe_tokens tokens, whose products read the blocks of the weights in place, and one of
        tiled_tokens() tokens, whose products go through the tiles of the weights expanded.  Two
        experts whose scores stand within the rounding of the products may change places between
        the device and the processor, the token then reading another expert: a probe that
        differs runs once more on other tokens before the device gives the layers up.  The state
        stays unchecked when the keys and values or the device lack room for a probe.
    */
    void cpu_runtime::check_hybrid_pass()
    {
        long first = probe_tokens;
        first = std::max(first, hybrid_pass_->min_tokens());
        if (cfg_.ctx_train > 0) first = std::min(first, cfg_.ctx_train);
        const long tiled = hybrid_pass_->tiled_tokens();
        std::string details, detail;
        auto note = [&details](const std::string& d) { details += (details.empty() ? "" : "; ") + d; };
        for (long n : { first, tiled })
        {
            if (n != first && (n <= first || (cfg_.ctx_train > 0 && n > cfg_.ctx_train))) continue;
            int outcome = probe_hybrid_pass(n, 0, detail);
            if (outcome < 0) return;
            if (outcome == 0)
            {
                note(detail + ", probed again");
                outcome = probe_hybrid_pass(n, 1, detail);
                if (outcome < 0) return;
            }
            // A pass that differs with products of lower precision is compared again without them.
            if (outcome == 0 && hybrid_pass_->retreat())
            {
                note(detail + ", the products of the device returning to single precision");
                outcome = probe_hybrid_pass(n, 2, detail);
                if (outcome < 0) return;
            }
            note(detail);
            if (outcome == 0)
            {
                drop_hybrid_pass(details);
                return;
            }
        }
        if (hybrid_pass_->timed())
        {
            const long many = tiled > first && (cfg_.ctx_train <= 0 || tiled <= cfg_.ctx_train) ? tiled : 0;
            const bool kept = time_passes(many, detail);
            if (!detail.empty()) note(detail);
            if (!kept)
            {
                drop_hybrid_pass(details);
                return;
            }
        }
        layer_pass_checked_ = true;
        hybrid_pass_->report(true, details);
    }

    bool cpu_runtime::run_hybrid(const forward_batch& batch, std::vector<float>& logits, bool on_device)
    {
        const long T = static_cast<long>(batch.tokens.size());
        if (T == 0) { logits.clear(); return true; }
        if (!batch.positions.empty() || !batch.embedding_rows.empty())
            throw std::runtime_error("cpu_runtime: a hybrid decoder reads tokens of text only");
        const long d = cfg_.d_model;
        const long B = kv_.block_positions();
        if (scratch_.empty()) scratch_.resize(1);

        // Positions of the tokens, which continue their sequences and the states of their layers.
        positions_.assign(static_cast<size_t>(T), 0);
        for (const auto& s : batch.segments)
        {
            kv_sequence& seq = *sequences_.at(static_cast<size_t>(s.sequence));
            if (static_cast<long>(seq.blocks.size()) * B < seq.length + s.count)
                throw std::runtime_error("cpu_runtime: sequence storage was not reserved");
            if (s.count > 0 && pool_of(s.sequence).shared(seq.blocks[static_cast<size_t>(seq.length / B)]))
                throw std::runtime_error("cpu_runtime: a sequence would write in a block other sequences hold");
            const recurrent_sequence& r = recurrent_.at(static_cast<size_t>(s.sequence));
            if (r.lost)
                throw std::runtime_error("cpu_runtime: the states of a sequence were lost with the device");
            if (r.live.length != seq.length)
                throw std::logic_error("cpu_runtime: the states of a sequence stand at another length than its keys");
            for (long j = 0; j < s.count; ++j) positions_[static_cast<size_t>(s.start + j)] = seq.length + j;
        }
        /*
            The processor reads the states of the sequences of its passes from the host and moves
            them ahead: the device drops the copies it kept, which a later pass of its own
            would otherwise take for the states of the sequences.
        */
        if (!on_device)
            for (const auto& s : batch.segments)
            {
                if (!fetch_states(s.sequence))
                    throw std::runtime_error("cpu_runtime: the states of a sequence were lost with the device");
                if (hybrid_pass_ && s.count > 0) hybrid_pass_->forget(s.sequence);
            }

        // The embeddings enter the sphere the blocks keep them on.
        embed(batch.tokens);
        for (long t = 0; t < T; ++t)
        {
            float* x = x_.data() + t * d;
            double ss = 0.0;
            for (long i = 0; i < d; ++i) ss += static_cast<double>(x[i]) * x[i];
            qk_scale(x, static_cast<float>(1.0 / std::max(std::sqrt(ss), 1e-12)), d);
        }

        const long R = static_cast<long>(batch.logit_rows.size());
        bool device_logits = false;
        if (on_device)
        {
            device_segments_.clear();
            for (const auto& s : batch.segments)
            {
                recurrent_sequence& r = recurrent_[static_cast<size_t>(s.sequence)];
                hybrid_pass::segment g;
                g.sequence = s.sequence;
                g.start = s.start;
                g.count = s.count;
                g.state = r.live.values.data();
                g.current = r.on_host;
                device_segments_.push_back(g);
            }
            // What the attention layers change on the host during the pass, kept for a pass the device fails.
            save_attention_states(batch);
            try
            {
                if (!hybrid_pass_->begin(x_.data(), T, device_segments_)) return false;
                for (long li = 0; li < cfg_.n_layers; ++li)
                {
                    if (cfg_.layer_types[static_cast<size_t>(li)] == 'M') hybrid_pass_->mixer(li);
                    else differential_attention(li, batch, true);
                    hybrid_pass_->experts(li);
                }
                const bool wanted = !batch.hidden_only && R > 0;
                if (wanted) logits.resize(static_cast<size_t>(R * cfg_.vocab));
                device_logits = hybrid_pass_->end(batch.logit_rows, x_.data(), wanted ? logits.data() : nullptr);
            }
            catch (const device_failure&)
            {
                restore_states(batch);
                throw;
            }
        }
        else
        {
            for (long li = 0; li < cfg_.n_layers; ++li)
            {
                const layer& L = layers_[static_cast<size_t>(li)];
                if (cfg_.layer_types[static_cast<size_t>(li)] == 'M') mamba3_mixer(li, batch);
                else differential_attention(li, batch, false);
                geodesic_update(tmp_.data(), L.mixer_scale, L.mixer_bias, li, T);
                latent_experts(L, T);
                geodesic_update(tmp_.data(), L.ffn_scale, L.ffn_bias, li, T);
            }
        }

        for (const auto& s : batch.segments)
        {
            sequences_[static_cast<size_t>(s.sequence)]->length += s.count;
            recurrent_sequence& r = recurrent_[static_cast<size_t>(s.sequence)];
            const long start = r.live.length;
            r.live.length += s.count;
            // The device moved the states of the Mamba-3 layers ahead of those of the host.
            if (on_device && s.count > 0) r.on_host = false;
            if (s.count < 2 && !s.prompt)
            {
                // A generated token: the next prompt saves states of its own.
                r.reading = -1;
                r.own = false;
                continue;
            }
            // A pass of the prompt the pass before was reading, or the first one of another prompt.
            if (r.reading != start) r.own = false;
            r.reading = r.live.length;
            // A lone token saves no state: the one of the chunk before it stays that of its prompt.
            if (s.count < 2) continue;
            // The states come back from the device to be saved; lost, the sequence keeps its older ones.
            if (!fetch_states(s.sequence)) continue;
            // The end of a chunk of a prompt: its state replaces the one of the chunk before it.
            if (r.own && !r.saved.empty()) r.saved.pop_back();
            else if (r.saved.size() >= saved_states) r.saved.erase(r.saved.begin());
            r.saved.push_back(r.live);
            r.own = true;
        }

        // The logits of the requested rows read the hidden states as they stand.
        hidden_out_.resize(static_cast<size_t>(R * d));
        for (long r = 0; r < R; ++r)
        {
            const float* x = x_.data() + batch.logit_rows[static_cast<size_t>(r)] * d;
            std::copy(x, x + d, hidden_out_.begin() + r * d);
        }
        if (batch.hidden_only)
        {
            logits.clear();
            return true;
        }
        if (device_logits) return true;
        logits.resize(static_cast<size_t>(R * cfg_.vocab));
        if (R > 0) product(output_, hidden_out_.data(), R, d, logits.data(), cfg_.vocab);
        return true;
    }

}
