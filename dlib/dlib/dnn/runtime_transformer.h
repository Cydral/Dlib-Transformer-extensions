// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Runtime (shape-dynamic) decoder-only transformer for imported open-weight models.
//
// The template network stack fixes every architectural dimension (depth, heads,
// embedding width) in the C++ type, which gives full integration with Dlib's
// training machinery at the price of one long compilation per model. This engine is
// the inference-only counterpart: every dimension comes from the model_spec at load
// time, so the engine compiles once and loads any supported GGUF model dynamically.
// It runs on the same tensor primitives (gemm, split/merge heads, rotary embedding,
// rms normalization, plane-wise softmax) as the template stack, on CPU or CUDA
// according to the build, and is validated by logit parity against external
// reference implementations.
//
// Two weight residencies are supported:
//   - resident (default): every matrix is dequantized once at load time; fastest
//     forward, memory footprint of the full float32 model;
//   - quantized at rest: the raw GGUF bytes stay in memory and each matrix is
//     dequantized just in time into a small shared pool at every use, keeping the
//     footprint close to the on-disk file size (e.g. ~1.1 GB instead of ~7 GB for a
//     1.7B model), traded against one dequantization per matrix and per forward.
//
// Generation runs in two phases: forward_prefill() processes the prompt and, when a
// context capacity is set, fills the KV cache; step() then extends the sequence one
// token at a time. Keys are cached before rotary encoding and encoded at read time
// on their compacted index, so the attention-sink eviction (keep the first rows,
// discard a block after them) keeps a coherent self-attention geometry without any
// shifting machinery, exactly like the template path.

#ifndef DLIB_DNN_RUNTIME_TRANSFORMER_H_
#define DLIB_DNN_RUNTIME_TRANSFORMER_H_

#include "runtime_transformer_abstract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#include "../dnn.h"
#include "../data_io/gguf_weight_loader.h"

namespace dlib
{
    class runtime_transformer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A decoder-only transformer (Llama/Qwen family) whose architecture is
                resolved at load time from a model_spec. load() ingests the GGUF
                weights; forward_prefill() runs the full forward over a token
                sequence and returns the logits of every position.

                The numeric path mirrors the template attention layer exactly: Q
                scaled by 1/sqrt(head_dim) at projection, rotary encoding applied to
                Q and K with positions 0..N-1, GQA repeat by channel copy, causal
                masking by additive per-head bias, plane-wise softmax, and a SwiGLU
                feed-forward.
        !*/

        /* Storage of one weight matrix. In resident mode the dequantized, transposed
           tensor lives in `resident`. At rest, the raw GGUF bytes live in `raw` and
           the matrix is dequantized and transposed just in time into the shared pool
           at every use. */
        struct stored_matrix
        {
            resizable_tensor resident;
            std::vector<uint8_t> raw;
            ggml_type type = ggml_type::F32;
            std::vector<uint64_t> dims;
            long in_dim = 0, out_dim = 0;
            bool permute = false;
        };

        /* One transformer block: norms, projections and its KV cache. Declared here
           because member functions take it by reference, and parameter types must be
           visible at the point of declaration. */
        struct layer
        {
            resizable_tensor attn_norm_g, ffn_norm_g;
            stored_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
            resizable_tensor bq, bk, bv;         // optional QKV biases (Qwen2 family)
            resizable_tensor qnorm_g, knorm_g;   // optional per-head QK-norm gammas (Qwen3)
            resizable_tensor k_cache, v_cache;   // pre-rotary keys, values

            /* Mixture-of-experts feed-forward (spec.n_experts > 0): the router
               projects to one logit per expert, and each expert is an independent
               SwiGLU triple sliced out of the 3D *_exps tensors. Dense models leave
               these empty and use w_gate/w_up/w_down above. */
            stored_matrix router;
            std::vector<stored_matrix> e_gate, e_up, e_down;
        };

        static void serialize_matrix(const stored_matrix& m, std::ostream& out)
        {
            const int kind = m.raw.empty() ? 0 : 1;
            dlib::serialize(kind, out);
            dlib::serialize(m.in_dim, out);
            dlib::serialize(m.out_dim, out);
            dlib::serialize(m.permute, out);
            if (kind == 1)
            {
                dlib::serialize(static_cast<uint32_t>(m.type), out);
                dlib::serialize(m.dims, out);
                dlib::serialize(m.raw, out);
            }
            else
            {
                dlib::serialize(m.resident, out);
            }
        }

        static void deserialize_matrix(stored_matrix& m, std::istream& in)
        {
            int kind = 0;
            dlib::deserialize(kind, in);
            dlib::deserialize(m.in_dim, in);
            dlib::deserialize(m.out_dim, in);
            dlib::deserialize(m.permute, in);
            m.raw.clear();
            m.resident.set_size(0, 0, 0, 0);
            if (kind == 1)
            {
                uint32_t t = 0;
                dlib::deserialize(t, in);
                m.type = static_cast<ggml_type>(t);
                dlib::deserialize(m.dims, in);
                dlib::deserialize(m.raw, in);
            }
            else
            {
                dlib::deserialize(m.resident, in);
            }
        }

    public:

        // Weight residency, to set before load(). Quantized at rest is the default:
        // memory stays close to the file size and, on CUDA builds, device memory
        // holds only the materialization pool. Disable it to trade memory for the
        // fastest forward (no per-use dequantization or upload).
        void set_quantized_at_rest(bool enabled) { at_rest_ = enabled; }
        bool quantized_at_rest() const { return at_rest_; }

        void load(gguf_reader& g, const model_spec& spec, const gguf_load_options& opt)
        {
            spec_ = spec;
            /* Rotary convention: llama-family GGUFs carry Q/K weights pre-permuted for
               the interleaved kernel, while the qwen families keep the split-half
               (NeoX) layout; the load-time permutation maps the latter onto the
               interleaved kernel. Selected from the architecture, with the option as
               a manual override for exotic exports. */
            rope_permute_ = opt.rope_permute
                || spec.arch_name == "qwen2" || spec.arch_name == "qwen3";
            const long d  = spec.d_model;
            const long qd = spec.n_heads * head_dim();
            const long kv = spec.n_kv_heads * head_dim();

            layers_.clear();
            layers_.resize(static_cast<size_t>(spec.n_layers));
            for (long i = 0; i < spec.n_layers; ++i)
            {
                const std::string p = "blk." + std::to_string(i) + ".";
                layer& L = layers_[static_cast<size_t>(i)];
                load_gamma (g, p + "attn_norm.weight", d, L.attn_norm_g);
                load_matrix(g, p + "attn_q.weight",      d, qd, L.wq, opt);
                load_matrix(g, p + "attn_k.weight",      d, kv, L.wk, opt);
                load_matrix(g, p + "attn_v.weight",      d, kv, L.wv, opt);
                load_optional_bias(g, p + "attn_q.bias", qd, L.bq, rope_permute_);
                load_optional_bias(g, p + "attn_k.bias", kv, L.bk, rope_permute_);
                load_optional_bias(g, p + "attn_v.bias", kv, L.bv, false);
                load_optional_head_gamma(g, p + "attn_q_norm.weight", L.qnorm_g, true);
                load_optional_head_gamma(g, p + "attn_k_norm.weight", L.knorm_g, false);
                load_matrix(g, p + "attn_output.weight", qd, d, L.wo, opt);
                load_gamma (g, p + "ffn_norm.weight", d, L.ffn_norm_g);
                if (spec.n_experts > 0)
                {
                    load_matrix(g, p + "ffn_gate_inp.weight", d, spec.n_experts, L.router, opt);
                    load_expert_matrices(g, p + "ffn_gate_exps.weight", d, spec.d_ffn, L.e_gate);
                    load_expert_matrices(g, p + "ffn_up_exps.weight",   d, spec.d_ffn, L.e_up);
                    load_expert_matrices(g, p + "ffn_down_exps.weight", spec.d_ffn, d, L.e_down);
                }
                else
                {
                    load_matrix(g, p + "ffn_gate.weight", d, spec.d_ffn, L.w_gate, opt);
                    load_matrix(g, p + "ffn_up.weight",   d, spec.d_ffn, L.w_up,   opt);
                    load_matrix(g, p + "ffn_down.weight", spec.d_ffn, d, L.w_down, opt);
                }
            }
            load_gamma(g, "output_norm.weight", d, final_norm_g_);
            load_matrix(g, spec.tied_embeddings ? "token_embd.weight" : "output.weight",
                d, spec.vocab_size, lm_head_, opt);

            /* The embedding table stays on the host. Resident mode keeps it as a flat
               float array; at rest, the raw quantized rows are kept and gathered rows
               are dequantized on demand (a token row spans only a few blocks). */
            const gguf_tensor_info* e = g.find_tensor("token_embd.weight");
            if (!e) throw std::runtime_error("runtime_transformer: token_embd.weight not found");
            if (at_rest_)
            {
                embd_raw_.type = e->type;
                embd_raw_.dims = e->dims;
                g.read_tensor_raw(*e, embd_raw_.raw);
                const ggml_type_traits tr = get_ggml_type_traits(e->type);
                if (static_cast<uint64_t>(d) % tr.block_size != 0)
                    throw std::runtime_error("runtime_transformer: embedding row not block-aligned");
                embd_row_bytes_ = static_cast<size_t>(d) / tr.block_size * tr.type_size;
            }
            else
            {
                gguf_read_dequantized(g, *e, embd_);
                if (embd_.size() != static_cast<size_t>(spec.vocab_size) * d)
                    throw std::runtime_error("runtime_transformer: embedding table size mismatch");
            }
        }

        // Configure the KV cache for incremental generation: capacity in tokens, and
        // the number of leading positions (attention sinks: BOS and system prompt)
        // pinned across evictions. Call before forward_prefill(); resets the cache.
        void set_context(long capacity, long keep_length = 0)
        {
            cap_ = capacity;
            keep_ = keep_length;
            reset_cache();
        }

        void reset_cache()
        {
            cur_len_ = 0;
            for (layer& L : layers_)
            {
                if (cap_ > 0)
                {
                    L.k_cache.set_size(1, spec_.n_kv_heads, cap_, head_dim());
                    L.v_cache.set_size(1, spec_.n_kv_heads, cap_, head_dim());
                }
                else
                {
                    L.k_cache.set_size(0, 0, 0, 0);
                    L.v_cache.set_size(0, 0, 0, 0);
                }
            }
        }

        long cache_length() const { return cur_len_; }

        // Full prefill forward. Returns the logits of every position in a
        // [1, 1, N, vocab] tensor. When a context capacity is set, the pre-rotary
        // keys and the values of every layer are appended to the KV cache.
        const tensor& forward_prefill(const std::vector<int>& tokens)
        {
            const long N  = static_cast<long>(tokens.size());
            const long d  = spec_.d_model;
            const long H  = spec_.n_heads;
            const long KV = spec_.n_kv_heads;
            const long hd = head_dim();
            const long rep = H / KV;
            const float qk_scale = spec_.attention_scale > 0.0
                ? static_cast<float>(spec_.attention_scale)
                : 1.0f / std::sqrt(static_cast<float>(hd));
            const float eps = static_cast<float>(spec_.rms_eps);
            DLIB_CASSERT(N > 0);
            if (cap_ > 0 && cur_len_ + N > cap_)
                throw std::runtime_error("runtime_transformer: prompt exceeds the context capacity");
            /* The prefill computes attention within the chunk only, which is exact on
               an empty cache; later chunks must go through step() so every token
               attends to the whole cached context. */
            DLIB_CASSERT(cur_len_ == 0, "forward_prefill() requires an empty cache; use step() to extend");

            /* Embedding gather on the host, single upload. */
            x_.set_size(1, 1, N, d);
            {
                float* xh = x_.host_write_only();
                for (long t = 0; t < N; ++t)
                {
                    const long id = tokens[static_cast<size_t>(t)];
                    DLIB_CASSERT(id >= 0 && id < spec_.vocab_size, "token id out of range");
                    gather_embedding_row(id, xh + t * d);
                }
            }
            if (spec_.embedding_scale != 1.0)
                tt::affine_transform(x_, x_, static_cast<float>(spec_.embedding_scale), 0.0f);

            build_trig_caches(N);
            build_causal_mask(N);

            for (layer& L : layers_)
            {
                /* ---- attention block ---- */
                tt::rms_normalize(eps, xn_, rms_scale_, x_, L.attn_norm_g, true);

                q_flat_.set_size(1, 1, N, H * hd);
                k_flat_.set_size(1, 1, N, KV * hd);
                v_flat_.set_size(1, 1, N, KV * hd);
                {
                    alias_tensor x_2d(N, d), q_2d(N, H * hd), kv_2d(N, KV * hd);
                    auto xv = x_2d(xn_, 0);
                    auto qv = q_2d(q_flat_, 0);
                    auto kv_v = kv_2d(k_flat_, 0);
                    auto vv = kv_2d(v_flat_, 0);
                    tt::gemm(0.0f, qv, qk_scale, xv, false, weight(L.wq), false);
                    tt::gemm(0.0f, kv_v, 1.0f, xv, false, weight(L.wk), false);
                    tt::gemm(0.0f, vv, 1.0f, xv, false, weight(L.wv), false);
                    if (L.bq.size())
                    {
                        ensure_ones(N);
                        tt::gemm(1.0f, qv, qk_scale, ones_, false, L.bq, false);
                        tt::gemm(1.0f, kv_v, 1.0f, ones_, false, L.bk, false);
                        tt::gemm(1.0f, vv, 1.0f, ones_, false, L.bv, false);
                    }
                }

                q4_.set_size(1, H, N, hd);
                k4_.set_size(1, KV, N, hd);
                v4_.set_size(1, KV, N, hd);
                tt::split_heads(false, q4_, q_flat_);
                tt::split_heads(false, k4_, k_flat_);
                tt::split_heads(false, v4_, v_flat_);

                apply_qk_norm(L, eps);

                if (cap_ > 0)
                    append_to_cache(L, k4_, v4_, N);   // normalized, pre-rotary keys

                tt::apply_rotary_positional_embedding(false, q4_, cos_cache_, sin_cache_);
                tt::apply_rotary_positional_embedding(false, k4_, cos_cache_, sin_cache_);

                k_rep_.set_size(1, H, N, hd);
                v_rep_.set_size(1, H, N, hd);
                for (long h = 0; h < H; ++h)
                {
                    tt::copy_tensor(false, k_rep_, h, k4_, h / rep, 1);
                    tt::copy_tensor(false, v_rep_, h, v4_, h / rep, 1);
                }

                scores_.set_size(1, H, N, N);
                tt::gemm(0.0f, scores_, 1.0f, q4_, false, k_rep_, true, operation_mode::PLANE_WISE);
                /* Causal mask, added per head plane: dlib's add() does not broadcast
                   unit dimensions, it zero-pads sources outside their bounds, so a
                   [1,1,N,N] mask added to [1,H,N,N] scores would only mask head 0. */
                {
                    alias_tensor plane(1, 1, N, N);
                    for (long h = 0; h < H; ++h)
                    {
                        auto sp = plane(scores_, static_cast<size_t>(h) * N * N);
                        tt::add(sp, sp, mask_);
                    }
                }
                attn_.copy_size(scores_);
                tt::softmax(attn_, scores_, operation_mode::PLANE_WISE);

                ctx4_.set_size(1, H, N, hd);
                tt::gemm(0.0f, ctx4_, 1.0f, attn_, false, v_rep_, false, operation_mode::PLANE_WISE);
                ctx_flat_.set_size(1, 1, N, H * hd);
                tt::merge_heads(false, ctx_flat_, ctx4_);

                attn_out_.set_size(1, 1, N, d);
                {
                    alias_tensor c_2d(N, H * hd), o_2d(N, d);
                    auto cv = c_2d(ctx_flat_, 0);
                    auto ov = o_2d(attn_out_, 0);
                    tt::gemm(0.0f, ov, 1.0f, cv, false, weight(L.wo), false);
                }
                add_residual(x_, attn_out_);

                /* ---- feed-forward block (dense SwiGLU or mixture of experts) ---- */
                tt::rms_normalize(eps, xn_, rms_scale_, x_, L.ffn_norm_g, true);
                if (spec_.n_experts > 0)
                    moe_ffn(L, N);
                else
                {
                    gate_.set_size(1, 1, N, spec_.d_ffn);
                    up_.set_size(1, 1, N, spec_.d_ffn);
                    {
                        alias_tensor n_2d(N, d), f_2d(N, spec_.d_ffn);
                        auto nv = n_2d(xn_, 0);
                        auto gv = f_2d(gate_, 0);
                        auto uv = f_2d(up_, 0);
                        tt::gemm(0.0f, gv, 1.0f, nv, false, weight(L.w_gate), false);
                        tt::gemm(0.0f, uv, 1.0f, nv, false, weight(L.w_up), false);
                    }
                    sig_.copy_size(gate_);
                    tt::sigmoid(sig_, gate_);
                    tt::multiply(false, gate_, gate_, sig_);   // silu(g) = g * sigmoid(g)
                    tt::multiply(false, gate_, gate_, up_);    // swiglu = silu(g) * up
                    ffn_out_.set_size(1, 1, N, d);
                    {
                        alias_tensor h_2d(N, spec_.d_ffn), f_2d(N, d);
                        auto hv = h_2d(gate_, 0);
                        auto fv = f_2d(ffn_out_, 0);
                        tt::gemm(0.0f, fv, 1.0f, hv, false, weight(L.w_down), false);
                    }
                }
                add_residual(x_, ffn_out_);
            }

            if (cap_ > 0)
                cur_len_ += N;

            /* ---- final norm and logits ---- */
            tt::rms_normalize(eps, xn_, rms_scale_, x_, final_norm_g_, true);
            logits_.set_size(1, 1, N, spec_.vocab_size);
            {
                alias_tensor n_2d(N, d), l_2d(N, spec_.vocab_size);
                auto nv = n_2d(xn_, 0);
                auto lv = l_2d(logits_, 0);
                const float logit_alpha = spec_.logit_scale > 0.0
                    ? 1.0f / static_cast<float>(spec_.logit_scale) : 1.0f;
                tt::gemm(0.0f, lv, logit_alpha, nv, false, weight(lm_head_), false);
            }
            return logits_;
        }

        // One-token incremental forward. Requires a prior forward_prefill() with a
        // context capacity set. Appends the token to the KV cache, evicting a middle
        // block first when the cache is full, and returns the [1, 1, 1, vocab] logits.
        const tensor& step(int token)
        {
            DLIB_CASSERT(cap_ > 0, "set_context() must be called before step()");
            DLIB_CASSERT(cur_len_ > 0, "forward_prefill() must run before step()");
            if (cur_len_ + 1 > cap_)
                evict_middle_block();

            const long d  = spec_.d_model;
            const long H  = spec_.n_heads;
            const long KV = spec_.n_kv_heads;
            const long hd = head_dim();
            const long rep = H / KV;
            const long pos = cur_len_;
            const long len = cur_len_ + 1;
            const float qk_scale = spec_.attention_scale > 0.0
                ? static_cast<float>(spec_.attention_scale)
                : 1.0f / std::sqrt(static_cast<float>(hd));
            const float eps = static_cast<float>(spec_.rms_eps);

            x_.set_size(1, 1, 1, d);
            gather_embedding_row(token, x_.host_write_only());
            if (spec_.embedding_scale != 1.0)
                tt::affine_transform(x_, x_, static_cast<float>(spec_.embedding_scale), 0.0f);
            build_trig_caches(len);   // covers the read positions 0..len-1

            for (layer& L : layers_)
            {
                /* ---- attention block ---- */
                tt::rms_normalize(eps, xn_, rms_scale_, x_, L.attn_norm_g, true);

                q_flat_.set_size(1, 1, 1, H * hd);
                k_flat_.set_size(1, 1, 1, KV * hd);
                v_flat_.set_size(1, 1, 1, KV * hd);
                {
                    alias_tensor x_2d(1, d), q_2d(1, H * hd), kv_2d(1, KV * hd);
                    auto xv = x_2d(xn_, 0);
                    auto qv = q_2d(q_flat_, 0);
                    auto kv_v = kv_2d(k_flat_, 0);
                    auto vv = kv_2d(v_flat_, 0);
                    tt::gemm(0.0f, qv, qk_scale, xv, false, weight(L.wq), false);
                    tt::gemm(0.0f, kv_v, 1.0f, xv, false, weight(L.wk), false);
                    tt::gemm(0.0f, vv, 1.0f, xv, false, weight(L.wv), false);
                    if (L.bq.size())
                    {
                        ensure_ones(1);
                        tt::gemm(1.0f, qv, qk_scale, ones_, false, L.bq, false);
                        tt::gemm(1.0f, kv_v, 1.0f, ones_, false, L.bk, false);
                        tt::gemm(1.0f, vv, 1.0f, ones_, false, L.bv, false);
                    }
                }
                q4_.set_size(1, H, 1, hd);
                k4_.set_size(1, KV, 1, hd);
                v4_.set_size(1, KV, 1, hd);
                tt::split_heads(false, q4_, q_flat_);
                tt::split_heads(false, k4_, k_flat_);
                tt::split_heads(false, v4_, v_flat_);
                apply_qk_norm(L, eps);
                append_to_cache(L, k4_, v4_, 1);   // normalized, pre-rotary key at row pos

                /* Rotate the query at its position: a one-row slice of the caches. */
                {
                    alias_tensor row(1, 1, 1, hd / 2);
                    auto cr = row(cos_cache_, static_cast<size_t>(pos) * (hd / 2));
                    auto sr = row(sin_cache_, static_cast<size_t>(pos) * (hd / 2));
                    tt::apply_rotary_positional_embedding(false, q4_, cr, sr);
                }

                /* Gather the cached window (contiguous per channel), rotate the keys
                   on their compacted indices 0..len-1. */
                k_win_.set_size(1, KV, len, hd);
                v_win_.set_size(1, KV, len, hd);
                {
                    alias_tensor w_2d(1, 1, len, hd);
                    for (long c = 0; c < KV; ++c)
                    {
                        auto kw = w_2d(k_win_, static_cast<size_t>(c) * len * hd);
                        auto vw = w_2d(v_win_, static_cast<size_t>(c) * len * hd);
                        auto ks = w_2d(L.k_cache, static_cast<size_t>(c) * cap_ * hd);
                        auto vs = w_2d(L.v_cache, static_cast<size_t>(c) * cap_ * hd);
                        memcpy(kw, ks);
                        memcpy(vw, vs);
                    }
                }
                tt::apply_rotary_positional_embedding(false, k_win_, cos_cache_, sin_cache_);

                k_rep_.set_size(1, H, len, hd);
                v_rep_.set_size(1, H, len, hd);
                for (long h = 0; h < H; ++h)
                {
                    tt::copy_tensor(false, k_rep_, h, k_win_, h / rep, 1);
                    tt::copy_tensor(false, v_rep_, h, v_win_, h / rep, 1);
                }

                /* Single query row: every cached position is in the past, no mask. */
                scores_.set_size(1, H, 1, len);
                tt::gemm(0.0f, scores_, 1.0f, q4_, false, k_rep_, true, operation_mode::PLANE_WISE);
                attn_.copy_size(scores_);
                tt::softmax(attn_, scores_, operation_mode::PLANE_WISE);

                ctx4_.set_size(1, H, 1, hd);
                tt::gemm(0.0f, ctx4_, 1.0f, attn_, false, v_rep_, false, operation_mode::PLANE_WISE);
                ctx_flat_.set_size(1, 1, 1, H * hd);
                tt::merge_heads(false, ctx_flat_, ctx4_);

                attn_out_.set_size(1, 1, 1, d);
                {
                    alias_tensor c_2d(1, H * hd), o_2d(1, d);
                    auto cv = c_2d(ctx_flat_, 0);
                    auto ov = o_2d(attn_out_, 0);
                    tt::gemm(0.0f, ov, 1.0f, cv, false, weight(L.wo), false);
                }
                add_residual(x_, attn_out_);

                /* ---- feed-forward block (SwiGLU) ---- */
                tt::rms_normalize(eps, xn_, rms_scale_, x_, L.ffn_norm_g, true);
                if (spec_.n_experts > 0)
                    moe_ffn(L, 1);
                else
                {
                    gate_.set_size(1, 1, 1, spec_.d_ffn);
                    up_.set_size(1, 1, 1, spec_.d_ffn);
                    {
                        alias_tensor n_2d(1, d), f_2d(1, spec_.d_ffn);
                        auto nv = n_2d(xn_, 0);
                        auto gv = f_2d(gate_, 0);
                        auto uv = f_2d(up_, 0);
                        tt::gemm(0.0f, gv, 1.0f, nv, false, weight(L.w_gate), false);
                        tt::gemm(0.0f, uv, 1.0f, nv, false, weight(L.w_up), false);
                    }
                    sig_.copy_size(gate_);
                    tt::sigmoid(sig_, gate_);
                    tt::multiply(false, gate_, gate_, sig_);
                    tt::multiply(false, gate_, gate_, up_);
                    ffn_out_.set_size(1, 1, 1, d);
                    {
                        alias_tensor h_2d(1, spec_.d_ffn), f_2d(1, d);
                        auto hv = h_2d(gate_, 0);
                        auto fv = f_2d(ffn_out_, 0);
                        tt::gemm(0.0f, fv, 1.0f, hv, false, weight(L.w_down), false);
                    }
                }
                add_residual(x_, ffn_out_);
            }
            cur_len_ = len;

            tt::rms_normalize(eps, xn_, rms_scale_, x_, final_norm_g_, true);
            logits_.set_size(1, 1, 1, spec_.vocab_size);
            {
                alias_tensor n_2d(1, d), l_2d(1, spec_.vocab_size);
                auto nv = n_2d(xn_, 0);
                auto lv = l_2d(logits_, 0);
                const float logit_alpha = spec_.logit_scale > 0.0
                    ? 1.0f / static_cast<float>(spec_.logit_scale) : 1.0f;
                tt::gemm(0.0f, lv, logit_alpha, nv, false, weight(lm_head_), false);
            }
            return logits_;
        }

        const model_spec& spec() const { return spec_; }

        /* Self-contained runtime archive: the model spec and every weight in its
           current storage form, quantized blocks at rest or dequantized floats. This
           format belongs to the engine, so loading never depends on the template
           network types nor on their serialization internals; models fine-tuned
           through the template path reach it via an export visitor on that side. */
        void save(std::ostream& out) const
        {
            const bool moe = spec_.n_experts > 0;
            dlib::serialize(moe ? "rtm_2" : "rtm_1", out);
            dlib::serialize(spec_.arch_name, out);
            dlib::serialize(spec_.model_name, out);
            dlib::serialize(static_cast<int>(spec_.family), out);
            dlib::serialize(spec_.n_layers, out);
            dlib::serialize(spec_.n_heads, out);
            dlib::serialize(spec_.n_kv_heads, out);
            dlib::serialize(spec_.d_model, out);
            dlib::serialize(spec_.d_ffn, out);
            dlib::serialize(spec_.head_dim, out);
            dlib::serialize(spec_.vocab_size, out);
            dlib::serialize(spec_.ffn_num, out);
            dlib::serialize(spec_.ffn_den, out);
            dlib::serialize(spec_.rms_eps, out);
            dlib::serialize(spec_.rope_freq_base, out);
            dlib::serialize(spec_.n_experts, out);
            dlib::serialize(spec_.n_experts_used, out);
            dlib::serialize(spec_.rope_scaling_type, out);
            dlib::serialize(spec_.rope_scaling_factor, out);
            dlib::serialize(spec_.rope_orig_ctx, out);
            dlib::serialize(spec_.tied_embeddings, out);
            dlib::serialize(spec_.quantized, out);
            dlib::serialize(spec_.qk_norm, out);
            if (moe)
            {
                dlib::serialize(spec_.embedding_scale, out);
                dlib::serialize(spec_.residual_scale, out);
                dlib::serialize(spec_.attention_scale, out);
                dlib::serialize(spec_.logit_scale, out);
            }
            for (const layer& L : layers_)
            {
                dlib::serialize(L.attn_norm_g, out);
                dlib::serialize(L.ffn_norm_g, out);
                serialize_matrix(L.wq, out);
                serialize_matrix(L.wk, out);
                serialize_matrix(L.wv, out);
                dlib::serialize(L.bq, out);
                dlib::serialize(L.bk, out);
                dlib::serialize(L.bv, out);
                dlib::serialize(L.qnorm_g, out);
                dlib::serialize(L.knorm_g, out);
                serialize_matrix(L.wo, out);
                serialize_matrix(L.w_gate, out);
                serialize_matrix(L.w_up, out);
                serialize_matrix(L.w_down, out);
                if (moe)
                {
                    serialize_matrix(L.router, out);
                    for (const stored_matrix& m : L.e_gate) serialize_matrix(m, out);
                    for (const stored_matrix& m : L.e_up)   serialize_matrix(m, out);
                    for (const stored_matrix& m : L.e_down) serialize_matrix(m, out);
                }
            }
            dlib::serialize(final_norm_g_, out);
            serialize_matrix(lm_head_, out);
            dlib::serialize(embd_, out);
            dlib::serialize(embd_raw_.raw, out);
            dlib::serialize(static_cast<uint32_t>(embd_raw_.type), out);
            dlib::serialize(embd_raw_.dims, out);
            dlib::serialize(static_cast<uint64_t>(embd_row_bytes_), out);
        }

        void load(std::istream& in)
        {
            std::string tag;
            dlib::deserialize(tag, in);
            if (tag != "rtm_1" && tag != "rtm_2")
                throw std::runtime_error("runtime_transformer: not a runtime archive (tag '" + tag + "')");
            const bool moe = (tag == "rtm_2");
            spec_ = model_spec();
            dlib::deserialize(spec_.arch_name, in);
            dlib::deserialize(spec_.model_name, in);
            int fam = 0;
            dlib::deserialize(fam, in);
            spec_.family = static_cast<arch_family>(fam);
            dlib::deserialize(spec_.n_layers, in);
            dlib::deserialize(spec_.n_heads, in);
            dlib::deserialize(spec_.n_kv_heads, in);
            dlib::deserialize(spec_.d_model, in);
            dlib::deserialize(spec_.d_ffn, in);
            dlib::deserialize(spec_.head_dim, in);
            dlib::deserialize(spec_.vocab_size, in);
            dlib::deserialize(spec_.ffn_num, in);
            dlib::deserialize(spec_.ffn_den, in);
            dlib::deserialize(spec_.rms_eps, in);
            dlib::deserialize(spec_.rope_freq_base, in);
            dlib::deserialize(spec_.n_experts, in);
            dlib::deserialize(spec_.n_experts_used, in);
            dlib::deserialize(spec_.rope_scaling_type, in);
            dlib::deserialize(spec_.rope_scaling_factor, in);
            dlib::deserialize(spec_.rope_orig_ctx, in);
            dlib::deserialize(spec_.tied_embeddings, in);
            dlib::deserialize(spec_.quantized, in);
            dlib::deserialize(spec_.qk_norm, in);
            if (moe)
            {
                dlib::deserialize(spec_.embedding_scale, in);
                dlib::deserialize(spec_.residual_scale, in);
                dlib::deserialize(spec_.attention_scale, in);
                dlib::deserialize(spec_.logit_scale, in);
            }
            layers_.clear();
            layers_.resize(static_cast<size_t>(spec_.n_layers));
            for (layer& L : layers_)
            {
                dlib::deserialize(L.attn_norm_g, in);
                dlib::deserialize(L.ffn_norm_g, in);
                deserialize_matrix(L.wq, in);
                deserialize_matrix(L.wk, in);
                deserialize_matrix(L.wv, in);
                dlib::deserialize(L.bq, in);
                dlib::deserialize(L.bk, in);
                dlib::deserialize(L.bv, in);
                dlib::deserialize(L.qnorm_g, in);
                dlib::deserialize(L.knorm_g, in);
                deserialize_matrix(L.wo, in);
                deserialize_matrix(L.w_gate, in);
                deserialize_matrix(L.w_up, in);
                deserialize_matrix(L.w_down, in);
                if (moe)
                {
                    deserialize_matrix(L.router, in);
                    L.e_gate.assign(static_cast<size_t>(spec_.n_experts), stored_matrix());
                    L.e_up.assign(static_cast<size_t>(spec_.n_experts), stored_matrix());
                    L.e_down.assign(static_cast<size_t>(spec_.n_experts), stored_matrix());
                    for (stored_matrix& m : L.e_gate) deserialize_matrix(m, in);
                    for (stored_matrix& m : L.e_up)   deserialize_matrix(m, in);
                    for (stored_matrix& m : L.e_down) deserialize_matrix(m, in);
                }
            }
            dlib::deserialize(final_norm_g_, in);
            deserialize_matrix(lm_head_, in);
            dlib::deserialize(embd_, in);
            dlib::deserialize(embd_raw_.raw, in);
            uint32_t t = 0;
            dlib::deserialize(t, in);
            embd_raw_.type = static_cast<ggml_type>(t);
            dlib::deserialize(embd_raw_.dims, in);
            uint64_t rb = 0;
            dlib::deserialize(rb, in);
            embd_row_bytes_ = static_cast<size_t>(rb);
            at_rest_ = !layers_.empty() && !layers_.front().wq.raw.empty();
            reset_cache();
        }

    private:

        /* Append n pre-rotary key rows and value rows to every-channel caches at the
           current length. Rows are contiguous within a channel. */
        void append_to_cache(layer& L, const resizable_tensor& k, const resizable_tensor& v, long n)
        {
            const long hd = head_dim();
            DLIB_CASSERT(cur_len_ + n <= cap_);
            alias_tensor blk(1, 1, n, hd);
            for (long c = 0; c < spec_.n_kv_heads; ++c)
            {
                auto ks = blk(const_cast<resizable_tensor&>(k), static_cast<size_t>(c) * n * hd);
                auto vs = blk(const_cast<resizable_tensor&>(v), static_cast<size_t>(c) * n * hd);
                auto kd = blk(L.k_cache, static_cast<size_t>(c) * cap_ * hd + cur_len_ * hd);
                auto vd = blk(L.v_cache, static_cast<size_t>(c) * cap_ * hd + cur_len_ * hd);
                memcpy(kd, ks);
                memcpy(vd, vs);
            }
        }

        /* Attention-sink eviction: keep the first keep_ rows, discard half of the
           remainder, slide the tail down. Keys are stored pre-rotary and encoded on
           their compacted index at read time, so the surviving rows keep a coherent
           relative geometry without any shifting pass. */
        void evict_middle_block()
        {
            const long discard = std::max<long>(1, (cap_ - keep_) / 2);
            const long tail = cur_len_ - keep_ - discard;
            DLIB_CASSERT(tail >= 0);
            const long hd = head_dim();
            alias_tensor blk(1, 1, tail, hd);
            for (layer& L : layers_)
            {
                for (long c = 0; c < spec_.n_kv_heads; ++c)
                {
                    const size_t base = static_cast<size_t>(c) * cap_ * hd;
                    auto ks = blk(L.k_cache, base + (keep_ + discard) * hd);
                    auto kd = blk(L.k_cache, base + keep_ * hd);
                    memcpy(kd, ks);
                    auto vs = blk(L.v_cache, base + (keep_ + discard) * hd);
                    auto vd = blk(L.v_cache, base + keep_ * hd);
                    memcpy(vd, vs);
                }
            }
            cur_len_ -= discard;
        }


        long head_dim() const
        {
            return spec_.head_dim > 0 ? spec_.head_dim : spec_.d_model / spec_.n_heads;
        }

        void load_matrix(gguf_reader& g, const std::string& name, long in_dim, long out_dim,
            stored_matrix& dst, const gguf_load_options& opt)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_transformer: missing tensor " + name);
            if (t->n_elements() != static_cast<uint64_t>(in_dim) * out_dim)
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            dst.in_dim = in_dim;
            dst.out_dim = out_dim;
            /* GGUF stores [out, in] row-major; the engine multiplies x[N,in] @ W[in,out],
               so the materialization transposes, honoring the optional rotate-half
               permutation exactly like the template-path loader. */
            dst.permute = rope_permute_
                && (name.find("attn_q") != std::string::npos || name.find("attn_k") != std::string::npos);
            if (at_rest_)
            {
                dst.type = t->type;
                dst.dims = t->dims;
                g.read_tensor_raw(*t, dst.raw);
            }
            else
            {
                std::vector<float> src;
                gguf_read_dequantized(g, *t, src);
                dst.resident.set_size(in_dim, out_dim);
                transpose_matrix(src, dst, dst.resident.host());
            }
        }

        /* Load a 3D expert tensor ([in, out, n_experts] in ggml dims order, one
           [out, in] matrix per expert) and split it per expert. Expert boundaries
           fall on row boundaries, and each row is block-aligned by the GGUF format,
           so at rest the raw bytes are sliced without requantization; the per-expert
           slices then behave exactly like ordinary 2D matrices (same weight()
           materialization, same transposition, no rotary permutation). */
        void load_expert_matrices(gguf_reader& g, const std::string& name,
            long in_dim, long out_dim, std::vector<stored_matrix>& dst)
        {
            const long n_exp = spec_.n_experts;
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_transformer: missing tensor " + name
                + " (legacy per-expert GGUF splits are not supported; re-convert with a recent tool)");
            if (t->n_elements() != static_cast<uint64_t>(in_dim) * out_dim * n_exp)
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            dst.assign(static_cast<size_t>(n_exp), stored_matrix());
            if (at_rest_)
            {
                std::vector<uint8_t> raw;
                g.read_tensor_raw(*t, raw);
                const size_t exp_bytes = raw.size() / static_cast<size_t>(n_exp);
                for (long e = 0; e < n_exp; ++e)
                {
                    stored_matrix& m = dst[static_cast<size_t>(e)];
                    m.in_dim = in_dim;
                    m.out_dim = out_dim;
                    m.type = t->type;
                    m.dims = { static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim) };
                    m.raw.assign(raw.begin() + static_cast<size_t>(e) * exp_bytes,
                                 raw.begin() + static_cast<size_t>(e + 1) * exp_bytes);
                }
            }
            else
            {
                std::vector<float> src;
                gguf_read_dequantized(g, *t, src);
                const size_t exp_elems = static_cast<size_t>(in_dim) * out_dim;
                std::vector<float> slice(exp_elems);
                for (long e = 0; e < n_exp; ++e)
                {
                    stored_matrix& m = dst[static_cast<size_t>(e)];
                    m.in_dim = in_dim;
                    m.out_dim = out_dim;
                    std::memcpy(slice.data(), src.data() + static_cast<size_t>(e) * exp_elems,
                        exp_elems * sizeof(float));
                    m.resident.set_size(in_dim, out_dim);
                    transpose_matrix(slice, m, m.resident.host());
                }
            }
        }

        /* Source tensor for one gemm: the resident copy, or a just-in-time
           materialization into the rotating pool. The returned reference stays valid
           until the same pool slot is handed out again, i.e. beyond the single gemm
           that consumes it. */        const tensor& weight(stored_matrix& m)
        {
            if (m.resident.size() != 0)
                return m.resident;
            gguf_tensor_info info;
            info.type = m.type;
            info.dims = m.dims;
            gguf_dequantize(info, m.raw, scratch_);
            resizable_tensor& slot = pool_[pool_next_];
            pool_next_ = (pool_next_ + 1) % 2;
            slot.set_size(m.in_dim, m.out_dim);
            /* Pure overwrite: host_write_only() skips the device-to-host refresh that
               host() would perform on a device-current tensor. */
            transpose_matrix(scratch_, m, slot.host_write_only());
            return slot;
        }

        void transpose_matrix(const std::vector<float>& src, const stored_matrix& m, float* dst) const
        {
            const long hd = const_cast<runtime_transformer*>(this)->head_dim();
            gguf_load_impl::transpose_into(src, m.out_dim, m.in_dim, dst,
                m.permute ? m.out_dim / hd : 1,
                m.permute ? hd : m.out_dim,
                m.permute);
        }

        /* Optional per-projection bias vector, stored as [1, dim] so the addition is
           a rank-one gemm: ones[N,1] @ b[1,dim] accumulated into the projection. The
           tensor stays empty when the model carries no bias. When the rotary layout
           permutation applies to the projection, it must apply to its bias too: the
           bias lives in the same output space as the permuted weight rows, and every
           quantity indexed by that space follows the same reordering. */
        void load_optional_bias(gguf_reader& g, const std::string& name, long dim,
            resizable_tensor& dst, bool permute)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) { dst.set_size(0, 0, 0, 0); return; }
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            if (src.size() != static_cast<size_t>(dim))
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            dst.set_size(1, dim);
            float* out = dst.host_write_only();
            if (!permute)
            {
                std::memcpy(out, src.data(), src.size() * sizeof(float));
                return;
            }
            const long hd = head_dim();
            const long half = hd / 2;
            for (long o = 0; o < dim; ++o)
            {
                const long h = o / hd;
                const long r = o % hd;
                const long ri = (r < half) ? (2 * r) : (2 * (r - half) + 1);
                out[h * hd + ri] = src[static_cast<size_t>(o)];
            }
        }

        /* Optional per-head QK-norm gamma (Qwen3), one vector of head_dim values shared
           by every head, stored [1,1,1,hd] for rms_normalize. Two adjustments happen at
           load: the rotary layout permutation (the gamma indexes the same permuted
           space as the Q/K rows), and, for the Q side, the 1/sqrt(head_dim) attention
           prescale folded into the gamma, since normalization erases any scale applied
           upstream (rms_norm(s*q) == rms_norm(q)) and would silently drop the prescale
           carried by the projection gemm. */
        void load_optional_head_gamma(gguf_reader& g, const std::string& name,
            resizable_tensor& dst, bool fold_qk_scale)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) { dst.set_size(0, 0, 0, 0); return; }
            const long hd = head_dim();
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            if (src.size() != static_cast<size_t>(hd))
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            const float s = fold_qk_scale ? 1.0f / std::sqrt(static_cast<float>(hd)) : 1.0f;
            const long half = hd / 2;
            dst.set_size(1, 1, 1, hd);
            float* out = dst.host_write_only();
            for (long r = 0; r < hd; ++r)
            {
                const long ri = rope_permute_ ? ((r < half) ? (2 * r) : (2 * (r - half) + 1)) : r;
                out[ri] = src[static_cast<size_t>(r)] * s;
            }
        }

        /* Per-head RMS normalization of Q and K (Qwen3): rms_normalize over the last
           dimension of the [1, heads, rows, head_dim] tensors, gammas carrying the
           layout permutation and, on the Q side, the attention prescale. */
        void apply_qk_norm(const layer& L, float eps)
        {
            if (L.qnorm_g.size() == 0) return;
            tt::rms_normalize(eps, qk_norm_tmp_, rms_scale_, q4_, L.qnorm_g, true);
            memcpy(q4_, qk_norm_tmp_);
            tt::rms_normalize(eps, qk_norm_tmp_, rms_scale_, k4_, L.knorm_g, true);
            memcpy(k4_, qk_norm_tmp_);
        }

        void ensure_ones(long n)
        {
            if (ones_.num_samples() == n) return;
            ones_.set_size(n, 1);
            float* p = ones_.host_write_only();
            for (long i = 0; i < n; ++i) p[i] = 1.0f;
        }

        /* Residual accumulation, honoring the residual scale declared by some
           families (granitemoe): x += residual_scale * y. */
        void add_residual(resizable_tensor& x, const resizable_tensor& y)
        {
            const float rs = static_cast<float>(spec_.residual_scale);
            if (rs == 1.0f) tt::add(x, x, y);
            else tt::affine_transform(x, x, y, 1.0f, rs);
        }

        /* Mixture-of-experts feed-forward over the normalized activations
           xn_ [1, 1, N, d], writing the combined result into ffn_out_ [1, 1, N, d].
           Routing follows the mixtral/granitemoe convention: softmax over the router
           logits, top-k selection, renormalization of the selected weights to sum 1.
           Tokens routed to the same expert are gathered into one batch, so each
           active expert materializes its three matrices once per call whatever the
           token count; in quantized-at-rest mode the dequantization cost is thus
           bounded by the number of active experts, not by tokens x top_k. */
        void moe_ffn(layer& L, long N)
        {
            const long d = spec_.d_model;
            const long n_exp = spec_.n_experts;
            const long top_k = spec_.n_experts_used;

            /* Router logits, then per-token selection on the host (n_exp is small). */
            router_logits_.set_size(1, 1, N, n_exp);
            {
                alias_tensor n_2d(N, d), r_2d(N, n_exp);
                auto nv = n_2d(xn_, 0);
                auto rv = r_2d(router_logits_, 0);
                tt::gemm(0.0f, rv, 1.0f, nv, false, weight(L.router), false);
            }
            const float* rl = router_logits_.host();
            route_e_.assign(static_cast<size_t>(N) * top_k, 0);
            route_w_.assign(static_cast<size_t>(N) * top_k, 0.0f);
            probs_.resize(static_cast<size_t>(n_exp));
            for (long t = 0; t < N; ++t)
            {
                const float* row = rl + t * n_exp;
                float mx = row[0];
                for (long e = 1; e < n_exp; ++e) mx = std::max(mx, row[e]);
                float sum = 0.0f;
                for (long e = 0; e < n_exp; ++e)
                {
                    probs_[static_cast<size_t>(e)] = std::exp(row[e] - mx);
                    sum += probs_[static_cast<size_t>(e)];
                }
                float sel = 0.0f;
                for (long k = 0; k < top_k; ++k)
                {
                    long best = 0;
                    for (long e = 1; e < n_exp; ++e)
                        if (probs_[static_cast<size_t>(e)] > probs_[static_cast<size_t>(best)]) best = e;
                    route_e_[static_cast<size_t>(t) * top_k + k] = best;
                    route_w_[static_cast<size_t>(t) * top_k + k] = probs_[static_cast<size_t>(best)] / sum;
                    sel += probs_[static_cast<size_t>(best)] / sum;
                    probs_[static_cast<size_t>(best)] = -1.0f;   // exclude from further picks
                }
                for (long k = 0; k < top_k; ++k)
                    route_w_[static_cast<size_t>(t) * top_k + k] /= sel;
            }

            ffn_out_.set_size(1, 1, N, d);
            ffn_out_ = 0;
            alias_tensor row(1, 1, 1, d);
            for (long e = 0; e < n_exp; ++e)
            {
                /* Gather the tokens routed to this expert. */
                sel_tok_.clear();
                sel_wgt_.clear();
                for (long t = 0; t < N; ++t)
                    for (long k = 0; k < top_k; ++k)
                        if (route_e_[static_cast<size_t>(t) * top_k + k] == e)
                        {
                            sel_tok_.push_back(t);
                            sel_wgt_.push_back(route_w_[static_cast<size_t>(t) * top_k + k]);
                        }
                const long m = static_cast<long>(sel_tok_.size());
                if (m == 0) continue;

                xg_.set_size(1, 1, m, d);
                for (long i = 0; i < m; ++i)
                {
                    auto src = row(xn_, static_cast<size_t>(sel_tok_[static_cast<size_t>(i)]) * d);
                    auto dst = row(xg_, static_cast<size_t>(i) * d);
                    memcpy(dst, src);
                }

                /* SwiGLU of the gathered batch through this expert. */
                eg_.set_size(1, 1, m, spec_.d_ffn);
                eu_.set_size(1, 1, m, spec_.d_ffn);
                {
                    alias_tensor x_2d(m, d), f_2d(m, spec_.d_ffn);
                    auto xv = x_2d(xg_, 0);
                    auto gv = f_2d(eg_, 0);
                    auto uv = f_2d(eu_, 0);
                    tt::gemm(0.0f, gv, 1.0f, xv, false, weight(L.e_gate[static_cast<size_t>(e)]), false);
                    tt::gemm(0.0f, uv, 1.0f, xv, false, weight(L.e_up[static_cast<size_t>(e)]), false);
                }
                esig_.copy_size(eg_);
                tt::sigmoid(esig_, eg_);
                tt::multiply(false, eg_, eg_, esig_);
                tt::multiply(false, eg_, eg_, eu_);
                ey_.set_size(1, 1, m, d);
                {
                    alias_tensor h_2d(m, spec_.d_ffn), y_2d(m, d);
                    auto hv = h_2d(eg_, 0);
                    auto yv = y_2d(ey_, 0);
                    tt::gemm(0.0f, yv, 1.0f, hv, false, weight(L.e_down[static_cast<size_t>(e)]), false);
                }

                /* Scatter-accumulate the weighted expert output. */
                for (long i = 0; i < m; ++i)
                {
                    auto y = row(ey_, static_cast<size_t>(i) * d);
                    auto o = row(ffn_out_, static_cast<size_t>(sel_tok_[static_cast<size_t>(i)]) * d);
                    tt::affine_transform(o, o, y, 1.0f, sel_wgt_[static_cast<size_t>(i)]);
                }
            }
        }

        void load_gamma(gguf_reader& g, const std::string& name, long d, resizable_tensor& dst)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_transformer: missing tensor " + name);
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            if (src.size() != static_cast<size_t>(d))
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            dst.set_size(1, 1, 1, d);
            std::memcpy(dst.host(), src.data(), src.size() * sizeof(float));
        }

        /* Copy the embedding row of one token into dst. At rest, only the blocks of
           that row are dequantized (a row spans row_bytes on disk). */
        void gather_embedding_row(long id, float* dst)
        {
            const long d = spec_.d_model;
            if (!at_rest_)
            {
                std::memcpy(dst, embd_.data() + static_cast<size_t>(id) * d,
                    static_cast<size_t>(d) * sizeof(float));
                return;
            }
            gguf_tensor_info info;
            info.type = embd_raw_.type;
            info.dims = { static_cast<uint64_t>(d) };
            row_bytes_.assign(embd_raw_.raw.begin() + static_cast<size_t>(id) * embd_row_bytes_,
                embd_raw_.raw.begin() + static_cast<size_t>(id + 1) * embd_row_bytes_);
            gguf_dequantize(info, row_bytes_, row_values_);
            std::memcpy(dst, row_values_.data(), static_cast<size_t>(d) * sizeof(float));
        }

        void build_trig_caches(long N)
        {
            const long half = head_dim() / 2;
            cos_cache_.set_size(1, 1, N, half);
            sin_cache_.set_size(1, 1, N, half);
            float* c = cos_cache_.host_write_only();
            float* s = sin_cache_.host_write_only();
            for (long pos = 0; pos < N; ++pos)
                for (long i = 0; i < half; ++i)
                {
                    const float inv_freq = std::pow(static_cast<float>(spec_.rope_freq_base),
                        -2.0f * i / static_cast<float>(head_dim()));
                    const float a = pos * inv_freq;
                    c[pos * half + i] = std::cos(a);
                    s[pos * half + i] = std::sin(a);
                }
        }

        void build_causal_mask(long N)
        {
            mask_.set_size(1, 1, N, N);
            float* m = mask_.host_write_only();
            for (long r = 0; r < N; ++r)
                for (long cidx = 0; cidx < N; ++cidx)
                    m[r * N + cidx] = (cidx <= r) ? 0.0f : -1e9f;
        }


        model_spec spec_;
        bool at_rest_ = true;
        bool rope_permute_ = false;
        long cap_ = 0, keep_ = 0, cur_len_ = 0;
        std::vector<layer> layers_;
        resizable_tensor final_norm_g_;
        stored_matrix lm_head_;

        /* Embedding table: resident floats, or raw rows at rest. */
        std::vector<float> embd_;
        stored_matrix embd_raw_;
        size_t embd_row_bytes_ = 0;
        std::vector<uint8_t> row_bytes_;
        std::vector<float> row_values_;

        /* Just-in-time materialization pool and scratch. */
        resizable_tensor qk_norm_tmp_;
        resizable_tensor ones_;
        resizable_tensor pool_[2];
        int pool_next_ = 0;
        std::vector<float> scratch_;

        /* Mixture-of-experts routing and gather buffers. */
        resizable_tensor router_logits_, xg_, eg_, eu_, esig_, ey_;
        std::vector<float> probs_, route_w_, sel_wgt_;
        std::vector<long> route_e_, sel_tok_;

        /* Work buffers, reused across layers and calls. */
        resizable_tensor x_, xn_, rms_scale_;
        resizable_tensor q_flat_, k_flat_, v_flat_, q4_, k4_, v4_, k_rep_, v_rep_;
        resizable_tensor k_win_, v_win_;
        resizable_tensor scores_, attn_, ctx4_, ctx_flat_, attn_out_;
        resizable_tensor gate_, up_, sig_, ffn_out_;
        resizable_tensor cos_cache_, sin_cache_, mask_, logits_;
    };
}

#endif // DLIB_DNN_RUNTIME_TRANSFORMER_H_
