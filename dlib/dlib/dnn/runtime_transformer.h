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
#include <sstream>
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

                A hybrid decoder (model_spec::layer_types, the olala family) takes a path
                of its own: its tokens are read one after the other, each layer keeping
                its state.  The hidden states are normalized by no layer: each block
                turns them on their sphere towards the output of its mixer, then towards
                the output of its mixture of experts (tt::geodesic_update).  A Mamba-3
                layer keeps the states of its channels (tt::mamba3_step); an attention
                layer keeps keys and values that are means of tensor products, shifted
                towards those of the previous token, and subtracts from the heads of each
                group the head that reads the noise; the experts work in a latent space
                beside a shared expert.  Such a decoder holds no position to evict: its
                context does not slide, and step() throws once the capacity is reached.
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

            /* Hybrid decoders, which use wq, wo, qnorm_g, knorm_g, router, e_up, e_down
               and the caches above with their own tensors. A Mamba-3 layer: its two
               projections, the constants of its heads packed as tt::mamba3_step() takes
               them, and its state. An attention layer: the factors of the tensor
               products of its keys and values, the projections of their shifts, of the
               weights of the noise heads and of the gate, the factor of each query head,
               and the key and value of the last token before their shift. The experts:
               the projections to and from their latent space, the bias of their choice
               and the shared expert. Then the scale and bias of the two geodesic
               updates of the block. */
            stored_matrix ssm_in, ssm_dyn;
            resizable_tensor ssm_norms, ssm_bias, ssm_misc, ssm_mxz, ssm_mo, ssm_state;
            stored_matrix wa_k, wb_k, wa_v, wb_v, shift_k, shift_v, noise, attn_gate;
            std::vector<float> softmax_scaler, expert_bias;
            resizable_tensor last_k, last_v;
            stored_matrix latent_down, latent_up, shared_up, shared_down;
            float mixer_scale = 1.0f, mixer_bias = 0.0f, ffn_scale = 1.0f, ffn_bias = 0.0f;
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
            drop_adapters();
            spec_ = spec;
            // The layers start anew: no context, no position read.
            cap_ = keep_ = cur_len_ = 0;
            own_context_ = false;
            if (spec.hybrid())
            {
                rope_permute_ = false;
                load_hybrid(g, opt);
                load_embeddings(g);
                return;
            }
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
                load_optional_head_gamma(g, p + "attn_q_norm.weight", L.qnorm_g);
                load_optional_head_gamma(g, p + "attn_k_norm.weight", L.knorm_g);
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
            load_embeddings(g);
        }

        /* The embedding table stays on the host. Resident mode keeps it as a flat
           float array; at rest, the raw quantized rows are kept and gathered rows
           are dequantized on demand (a token row spans only a few blocks). */
        void load_embeddings(gguf_reader& g)
        {
            const model_spec& spec = spec_;
            const long d = spec.d_model;
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
        /* Supplies embeddings for chosen positions of the next prefill, in place of the
           table lookup those positions would otherwise get.

           This is how a modality that produces vectors rather than identifiers enters the
           stream. The caller reserves the positions with placeholder tokens, so the
           sequence stays a plain sequence of integers and nothing downstream, neither the
           attention, nor the rotary positions, nor the cache, has to know that some rows
           came from elsewhere.

           Consumed by the next forward_prefill() and forgotten afterwards, so a second
           prompt cannot inherit the images of the first. */
        void set_input_embeddings(const tensor& rows, const std::vector<long>& positions)
        {
            DLIB_CASSERT(static_cast<size_t>(rows.num_samples()) == positions.size(),
                "one position is needed per supplied embedding row");
            DLIB_CASSERT(rows.size() == positions.size() * static_cast<size_t>(spec_.d_model),
                "the supplied embeddings do not have the width of this model");
            injected_rows_.set_size(static_cast<long>(positions.size()), spec_.d_model);
            memcpy(injected_rows_, rows);
            injected_ = positions;
        }

        void clear_input_embeddings() { injected_.clear(); }

        // pinned across evictions. Call before forward_prefill(); resets the cache.
        void set_context(long capacity, long keep_length = 0)
        {
            if (spec_.hybrid() && spec_.attention_window > 0 && capacity > spec_.attention_window)
                throw std::runtime_error("runtime_transformer: the context of this hybrid decoder holds "
                    + std::to_string(spec_.attention_window) + " positions at most, the window of its attention");
            cap_ = capacity;
            keep_ = keep_length;
            own_context_ = false;
            reset_cache();
        }

        void reset_cache()
        {
            cur_len_ = 0;
            for (size_t i = 0; i < layers_.size(); ++i)
            {
                layer& L = layers_[i];
                if (spec_.hybrid())
                {
                    // The states of the layers start from zero; the attention layers alone keep keys and values.
                    if (L.ssm_state.size() != 0) L.ssm_state = 0;
                    if (spec_.layer_types[i] != 'V') continue;
                }
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
            if (spec_.hybrid()) return hybrid_prefill(tokens);
            if (cap_ > 0 && cur_len_ + N > cap_)
                throw std::runtime_error("runtime_transformer: prompt exceeds the context capacity");
            /* The prefill computes attention within the chunk only, which is exact on
               an empty cache; later chunks must go through step() so every token
               attends to the whole cached context. */
            DLIB_CASSERT(cur_len_ == 0, "forward_prefill() requires an empty cache; use step() to extend");

            /* Embedding gather on the host, single upload. Rows that were given an
               embedding directly are written over afterwards rather than looked up: a
               vision tower produces vectors, not identifiers, and the placeholder tokens
               standing in for them in the stream carry no meaning of their own. */
            x_.set_size(1, 1, N, d);
            {
                float* xh = x_.host_write_only();
                for (long t = 0; t < N; ++t)
                {
                    const long id = tokens[static_cast<size_t>(t)];
                    DLIB_CASSERT(id >= 0 && id < spec_.vocab_size, "token id out of range");
                    gather_embedding_row(id, xh + t * d);
                }
                if (!injected_.empty())
                {
                    const float* src = injected_rows_.host();
                    const long rows = injected_rows_.num_samples();
                    for (long r = 0; r < rows; ++r)
                    {
                        const long t = injected_[static_cast<size_t>(r)];
                        DLIB_CASSERT(t >= 0 && t < N,
                            "an injected embedding falls outside the prompt");
                        std::memcpy(xh + t * d, src + r * d, sizeof(float) * d);
                    }
                    injected_.clear();
                }
            }
            if (spec_.embedding_scale != 1.0)
                tt::affine_transform(x_, x_, static_cast<float>(spec_.embedding_scale), 0.0f);

            build_trig_caches(N);
            build_causal_mask(N);

            for (layer& L : layers_)
            {
                const size_t li = static_cast<size_t>(&L - layers_.data());
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
                    tt::gemm(0.0f, qv, 1.0f, xv, false, weight(L.wq), false);
                    tt::gemm(0.0f, kv_v, 1.0f, xv, false, weight(L.wk), false);
                    tt::gemm(0.0f, vv, 1.0f, xv, false, weight(L.wv), false);
                    adapt(li, 0, xv, L.wq, qv);
                    adapt(li, 1, xv, L.wk, kv_v);
                    adapt(li, 2, xv, L.wv, vv);
                    if (L.bq.size())
                    {
                        ensure_ones(N);
                        tt::gemm(1.0f, qv, 1.0f, ones_, false, L.bq, false);
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

                /* Scaled Q @ K^T. The factor is applied here, downstream of the optional
                   QK normalization which would otherwise erase it, and not folded into the
                   Q projection. Same placement as the template attention layer. */
                scores_.set_size(1, H, N, N);
                tt::gemm(0.0f, scores_, qk_scale, q4_, false, k_rep_, true, operation_mode::PLANE_WISE);
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
                    adapt(li, 3, cv, L.wo, ov);
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
                        adapt(li, 4, nv, L.w_gate, gv);
                        adapt(li, 5, nv, L.w_up, uv);
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
                        adapt(li, 6, hv, L.w_down, fv);
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

        /* The hidden states of the last forward, after the final normalization and before
           the vocabulary projection.

           This is what an embedding model produces. Such a container carries no
           output.weight at all: the vocabulary projection was removed when the task
           adapter was merged, and what remains ends at output_norm. The logits computed
           above are then meaningless, being a projection through the tied embedding table,
           and are simply not read.

           Laid out as (1, 1, seq_len, embedding_dim), row t holding the state of position
           t. Which row to take is the pooling question, and the container answers it:
           see model_spec::pooling. */
        const tensor& hidden_states() const { return xn_; }

        // One-token incremental forward. Requires a prior forward_prefill() with a
        // context capacity set. Appends the token to the KV cache, evicting a middle
        // block first when the cache is full, and returns the [1, 1, 1, vocab] logits.
        const tensor& step(int token)
        {
            DLIB_CASSERT(cap_ > 0, "set_context() must be called before step()");
            DLIB_CASSERT(cur_len_ > 0, "forward_prefill() must run before step()");
            if (spec_.hybrid())
            {
                if (own_context_ || cur_len_ + 1 > cap_)
                    throw std::runtime_error("runtime_transformer: the context of a hybrid decoder is full, and its "
                        "layers cannot give up the tokens they read; set_context() takes a larger one");
                hybrid_token(token);
                return logits_;
            }
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
                const size_t li = static_cast<size_t>(&L - layers_.data());
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
                    tt::gemm(0.0f, qv, 1.0f, xv, false, weight(L.wq), false);
                    tt::gemm(0.0f, kv_v, 1.0f, xv, false, weight(L.wk), false);
                    tt::gemm(0.0f, vv, 1.0f, xv, false, weight(L.wv), false);
                    adapt(li, 0, xv, L.wq, qv);
                    adapt(li, 1, xv, L.wk, kv_v);
                    adapt(li, 2, xv, L.wv, vv);
                    if (L.bq.size())
                    {
                        ensure_ones(1);
                        tt::gemm(1.0f, qv, 1.0f, ones_, false, L.bq, false);
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
                tt::gemm(0.0f, scores_, qk_scale, q4_, false, k_rep_, true, operation_mode::PLANE_WISE);
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
                    adapt(li, 3, cv, L.wo, ov);
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
                        adapt(li, 4, nv, L.w_gate, gv);
                        adapt(li, 5, nv, L.w_up, uv);
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
                        adapt(li, 6, hv, L.w_down, fv);
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

        // Low-rank fine-tuning of a dense decoder

        /* One adapted projection: the tensor of the container it adapts, its block and
           projection, its geometry, and the position of its [A | B | m] block in the
           packed parameters. */
        struct adapter_site
        {
            std::string tensor_name;
            long layer = 0;
            int projection = 0;
            adapter_geometry geometry;
            size_t offset = 0;
        };

        static const char* projection_name(int p)
        {
            static const char* names[] = { "attn_q", "attn_k", "attn_v", "attn_output",
                                           "ffn_gate", "ffn_up", "ffn_down" };
            DLIB_CASSERT(p >= 0 && p < 7);
            return names[p];
        }

        size_t configure_adapters(const adapter_plan& plan, unsigned long seed = 0)
        {
            if (layers_.empty())
                throw std::runtime_error("runtime_transformer: load a model before configuring adapters");
            if (spec_.hybrid() || spec_.n_experts > 0)
                throw std::runtime_error("runtime_transformer: adapters are offered for dense decoders only");
            adapters_.clear();
            sites_.clear();
            site_of_.assign(layers_.size() * 7, -1);
            traces_.clear();
            plan_ = plan;
            adapter_params_.set_size(0, 0, 0, 0);
            adapter_grads_.set_size(0, 0, 0, 0);
            if (!plan.active())
                return 0;

            size_t total = 0;
            for (size_t li = 0; li < layers_.size(); ++li)
            {
                for (int p = 0; p < 7; ++p)
                {
                    const stored_matrix& m = projection(layers_[li], p);
                    /* The query and the value follow their own switches; the key, the output
                       and the feed-forward projections follow the switch of the plain
                       projections. */
                    const bool wanted = (p == 0 && plan.attention_query) || (p == 2 && plan.attention_value)
                        || ((p == 1 || p >= 3) && plan.projection);
                    if (!wanted || !plan.covers(m.out_dim))
                        continue;
                    low_rank_adapter a;
                    a.configure(m.in_dim, m.out_dim, plan.rank, plan.method, plan.alpha);
                    adapter_site site;
                    site.tensor_name = "blk." + std::to_string(li) + "." + projection_name(p) + ".weight";
                    site.layer = static_cast<long>(li);
                    site.projection = p;
                    site.geometry = a.geometry();
                    site.offset = total;
                    total += a.parameter_count();
                    site_of_[li * 7 + static_cast<size_t>(p)] = static_cast<long>(adapters_.size());
                    adapters_.push_back(a);
                    sites_.push_back(site);
                }
            }
            if (total == 0)
                return 0;
            adapter_params_.set_size(static_cast<long>(total));
            adapter_params_ = 0;
            adapter_grads_.set_size(static_cast<long>(total));
            adapter_grads_ = 0;
            dlib::rand rnd(seed);
            for (size_t s = 0; s < adapters_.size(); ++s)
            {
                const adapter_site& site = sites_[s];
                low_rank_adapter& a = adapters_[s];
                auto av = a.a_view()(adapter_params_, site.geometry.a_offset(site.offset));
                auto bv = a.b_view()(adapter_params_, site.geometry.b_offset(site.offset));
                auto mv = a.m_view()(adapter_params_, site.geometry.m_offset(site.offset));
                a.initialize(weight(projection(layers_[static_cast<size_t>(site.layer)], site.projection)),
                    av, bv, mv, rnd);
            }
            return adapters_.size();
        }

        bool adapters_active() const { return !adapters_.empty(); }
        const adapter_plan& adapter_configuration() const { return plan_; }
        const std::vector<adapter_site>& adapter_sites() const { return sites_; }
        size_t adapter_parameter_count() const { return adapter_params_.size(); }
        tensor& adapter_parameters() { return adapter_params_; }
        const tensor& adapter_parameters() const { return adapter_params_; }
        const tensor& adapter_gradients() const { return adapter_grads_; }
        void zero_adapter_gradients() { if (adapter_grads_.size()) adapter_grads_ = 0; }

        /* The packed parameters as consecutive blocks, A as (in_dim, rank), B as
           (out_dim, rank) and m as one row, in the order of adapter_sites(). */
        parameter_layout adapter_layout() const
        {
            parameter_layout layout;
            for (const adapter_site& s : sites_)
            {
                layout.add_block(s.geometry.in_dim, s.geometry.rank);
                layout.add_block(s.geometry.out_dim, s.geometry.rank);
                if (s.geometry.m_count())
                    layout.add_block(1, s.geometry.out_dim);
            }
            return layout;
        }

        /* Runs the sequence forward, keeping what the backward pass needs, then adds to the
           adapter gradients gradient_scale times the gradient of the summed cross-entropy of
           the supervised positions: those whose target is a token, a negative target leaving
           its position out. Returns the summed loss and sets supervised to the number of
           supervised positions. The adapters must be configured; no context may be open. */
        double accumulate_adapter_gradients(const std::vector<int>& tokens, const std::vector<int>& targets,
            long& supervised, double gradient_scale = 1.0)
        {
            supervised = 0;
            if (!adapters_active())
                throw std::runtime_error("runtime_transformer: no adapter to train");
            DLIB_CASSERT(!tokens.empty() && tokens.size() == targets.size());
            DLIB_CASSERT(cur_len_ == 0, "accumulate_adapter_gradients() needs an empty cache");
            forward_train(tokens);
            const double loss = cross_entropy(targets, supervised, true, static_cast<float>(gradient_scale));
            if (supervised > 0)
                backward_train();
            return loss;
        }

        /* The summed cross-entropy of the supervised positions of a sequence, adapters
           included, and their number. Leaves the cache empty. */
        double evaluate_loss(const std::vector<int>& tokens, const std::vector<int>& targets, long& supervised)
        {
            DLIB_CASSERT(!tokens.empty() && tokens.size() == targets.size());
            if (spec_.hybrid())
                throw std::runtime_error("runtime_transformer: evaluate_loss() is offered for dense decoders");
            reset_cache();
            forward_prefill(tokens);
            const double loss = cross_entropy(targets, supervised, false, 1.0f);
            reset_cache();
            return loss;
        }

        /* Folds every adapter into the weight it adapts and removes the adapters. The
           adapted weights become resident in single precision. */
        void merge_adapters()
        {
            for (size_t s = 0; s < adapters_.size(); ++s)
            {
                const adapter_site& site = sites_[s];
                stored_matrix& m = projection(layers_[static_cast<size_t>(site.layer)], site.projection);
                make_resident(m);
                low_rank_adapter& a = adapters_[s];
                auto av = a.a_view()(adapter_params_, site.geometry.a_offset(site.offset));
                auto bv = a.b_view()(adapter_params_, site.geometry.b_offset(site.offset));
                auto mv = a.m_view()(adapter_params_, site.geometry.m_offset(site.offset));
                a.merge_into_base(av, bv, mv, m.resident);
                merged_sites_.push_back(site);
            }
            const std::vector<adapter_site> merged = merged_sites_;
            adapter_plan none;
            configure_adapters(none);
            merged_sites_ = merged;
        }

        /* A copy of the container the model was loaded from, with the adapted weights
           merged and encoded in type (F32, F16, BF16 or Q8_0; a weight whose rows cannot
           hold whole Q8_0 blocks is written in F16). Every other tensor and every
           metadata value is copied as it is. */
        void write_merged_gguf(gguf_reader& source, const std::string& path, ggml_type type = ggml_type::Q8_0)
        {
            gguf_writer w;
            w.copy_metadata(source);
            std::vector<uint8_t> raw;
            std::vector<float> gg;
            for (const gguf_tensor_info& t : source.tensors())
            {
                long site_index = -1;
                for (size_t s = 0; s < sites_.size(); ++s)
                    if (sites_[s].tensor_name == t.name)
                        site_index = static_cast<long>(s);
                // A weight an earlier merge_adapters() changed is written as the engine holds it.
                const adapter_site* folded = nullptr;
                for (const adapter_site& f : merged_sites_)
                    if (f.tensor_name == t.name)
                        folded = &f;
                if (site_index < 0 && folded == nullptr)
                {
                    source.read_tensor_raw(t, raw);
                    w.add_tensor(t.name, t.type, t.dims, raw);
                    continue;
                }
                const adapter_site& site = site_index >= 0 ? sites_[static_cast<size_t>(site_index)] : *folded;
                stored_matrix& m = projection(layers_[static_cast<size_t>(site.layer)], site.projection);
                resizable_tensor merged;
                {
                    const tensor& base = weight(m);
                    merged.copy_size(base);
                    memcpy(merged, base);
                }
                if (site_index >= 0)
                {
                    low_rank_adapter a = adapters_[static_cast<size_t>(site_index)];
                    auto av = a.a_view()(adapter_params_, site.geometry.a_offset(site.offset));
                    auto bv = a.b_view()(adapter_params_, site.geometry.b_offset(site.offset));
                    auto mv = a.m_view()(adapter_params_, site.geometry.m_offset(site.offset));
                    a.merge_into_base(av, bv, mv, merged);
                }
                // Back to the [out, in] rows of the container, undoing the rotary permutation.
                const long in = m.in_dim, out = m.out_dim, hd = head_dim(), half = hd / 2;
                gg.assign(static_cast<size_t>(in) * out, 0.0f);
                const float* src = merged.host();
                for (long o = 0; o < out; ++o)
                {
                    long od = o;
                    if (m.permute)
                    {
                        const long r = o % hd;
                        od = (o / hd) * hd + ((r < half) ? (2 * r) : (2 * (r - half) + 1));
                    }
                    float* row = gg.data() + static_cast<size_t>(o) * in;
                    for (long i = 0; i < in; ++i)
                        row[i] = src[static_cast<size_t>(i) * out + od];
                }
                const ggml_type as = (type == ggml_type::Q8_0 && in % 32 != 0) ? ggml_type::F16 : type;
                w.add_tensor(t.name, t.dims, gg.data(), as);
            }
            w.write(path);
        }

        void save_adapters(std::ostream& out) const
        {
            dlib::serialize("runtime_adapters", out);
            dlib::serialize(model_signature(), out);
            dlib::serialize(plan_.rank, out);
            dlib::serialize(static_cast<int>(plan_.method), out);
            dlib::serialize(plan_.alpha, out);
            dlib::serialize(plan_.attention_query, out);
            dlib::serialize(plan_.attention_value, out);
            dlib::serialize(plan_.projection, out);
            dlib::serialize(plan_.max_width, out);
            dlib::serialize(adapter_params_, out);
        }

        void load_adapters(std::istream& in)
        {
            std::string tag, signature;
            dlib::deserialize(tag, in);
            if (tag != "runtime_adapters")
                throw serialization_error("runtime_transformer: not an adapter archive");
            dlib::deserialize(signature, in);
            if (signature != model_signature())
                throw serialization_error("runtime_transformer: the adapters were trained on another model");
            adapter_plan plan;
            int method = 0;
            dlib::deserialize(plan.rank, in);
            dlib::deserialize(method, in);
            if (method < 0 || method > 2 || plan.rank < 0 || plan.rank > 4096)
                throw serialization_error("runtime_transformer: invalid adapter configuration");
            plan.method = static_cast<adapter_method>(method);
            dlib::deserialize(plan.alpha, in);
            dlib::deserialize(plan.attention_query, in);
            dlib::deserialize(plan.attention_value, in);
            dlib::deserialize(plan.projection, in);
            dlib::deserialize(plan.max_width, in);
            resizable_tensor params;
            dlib::deserialize(params, in);
            // A failure leaves the adapters as they were.
            const adapter_plan old_plan = plan_;
            resizable_tensor old_params;
            old_params.copy_size(adapter_params_);
            if (adapter_params_.size())
                memcpy(old_params, adapter_params_);
            configure_adapters(plan);
            if (params.size() != adapter_params_.size())
            {
                configure_adapters(old_plan);
                if (old_params.size() == adapter_params_.size() && old_params.size())
                    memcpy(adapter_params_, old_params);
                throw serialization_error("runtime_transformer: adapter parameters of a wrong size");
            }
            memcpy(adapter_params_, params);
        }


        /* Self-contained runtime archive: the model spec and every weight in its
           current storage form, quantized blocks at rest or dequantized floats. This
           format belongs to the engine, so loading never depends on the template
           network types nor on their serialization internals; models fine-tuned
           through the template path reach it via an export visitor on that side. */
        void save(std::ostream& out) const
        {
            if (spec_.hybrid())
                throw std::runtime_error("runtime_transformer: the runtime archive does not hold the hybrid decoders; "
                    "they load from their GGUF file");
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
            drop_adapters();
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
            /* The family follows the name of the architecture, which the archive holds: the
               number stored beside it is the rank the family had in the list of the library
               that wrote the archive, and the list grows. */
            spec_.family = arch_family_from_name(spec_.arch_name);
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
            stored_matrix& dst, const gguf_load_options& /*opt*/)
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
        /* Per-head QK-Norm gamma. It indexes the same output space as the projection rows
           it scales, so it carries the rotary layout permutation. No scale is folded in: the
           attention prescale is applied at the score product, downstream of the
           normalization, so the stored gamma is the one the source model declares. */
        void load_optional_head_gamma(gguf_reader& g, const std::string& name,
            resizable_tensor& dst)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) { dst.set_size(0, 0, 0, 0); return; }
            const long hd = head_dim();
            std::vector<float> src;
            gguf_read_dequantized(g, *t, src);
            if (src.size() != static_cast<size_t>(hd))
                throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            const long half = hd / 2;
            dst.set_size(1, 1, 1, hd);
            float* out = dst.host_write_only();
            for (long r = 0; r < hd; ++r)
            {
                const long ri = rope_permute_ ? ((r < half) ? (2 * r) : (2 * (r - half) + 1)) : r;
                out[ri] = src[static_cast<size_t>(r)];
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

        // Hybrid decoders

        // out [1, 1, 1, w.out_dim] = in [1, 1, 1, w.in_dim] through a weight matrix.
        void project(resizable_tensor& out, resizable_tensor& in, stored_matrix& w)
        {
            DLIB_CASSERT(static_cast<long>(in.size()) == w.in_dim);
            out.set_size(1, 1, 1, w.out_dim);
            alias_tensor i_2d(1, w.in_dim), o_2d(1, w.out_dim);
            auto iv = i_2d(in, 0);
            auto ov = o_2d(out, 0);
            tt::gemm(0.0f, ov, 1.0f, iv, false, weight(w), false);
        }

        // The values of a tensor of the file, count of them, or throws.
        std::vector<float> read_floats(gguf_reader& g, const std::string& name, size_t count)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("runtime_transformer: missing tensor " + name);
            std::vector<float> v;
            gguf_read_dequantized(g, *t, v);
            if (v.size() != count) throw std::runtime_error("runtime_transformer: shape mismatch for " + name);
            return v;
        }

        /* The constants of a Mamba-3 layer as tt::mamba3_step() takes them: the tensor the
           file packs them in, or, for a file without it, its two parts side by side, each
           values of first then each values of second for each of the outer blocks. */
        void load_packed(gguf_reader& g, const std::string& packed, const std::string& first,
            const std::string& second, long outer, long each, resizable_tensor& dst)
        {
            const size_t n = static_cast<size_t>(2 * outer * each);
            dst.set_size(1, 1, 1, static_cast<long>(n));
            float* out = dst.host_write_only();
            if (g.find_tensor(packed))
            {
                const std::vector<float> v = read_floats(g, packed, n);
                std::memcpy(out, v.data(), n * sizeof(float));
                return;
            }
            const std::vector<float> a = read_floats(g, first, n / 2), b = read_floats(g, second, n / 2);
            for (long o = 0; o < outer; ++o)
            {
                std::memcpy(out + 2 * o * each, a.data() + o * each, static_cast<size_t>(each) * sizeof(float));
                std::memcpy(out + (2 * o + 1) * each, b.data() + o * each, static_cast<size_t>(each) * sizeof(float));
            }
        }

        void load_hybrid(gguf_reader& g, const gguf_load_options& opt)
        {
            const model_spec& s = spec_;
            if (!s.hybrid_supported)
                throw std::runtime_error("runtime_transformer: this variant of the hybrid layers is not supported");
            const long d = s.d_model, hd = head_dim();
            const long H = s.ssm_heads, P = s.ssm_head_dim, N = s.ssm_state, R = s.ssm_rank, S = s.ssm_angles;
            const long KV = s.n_kv_heads, rank = s.tpa_rank;
            const long signal = (s.n_heads - KV) * hd;
            layers_.clear();
            layers_.resize(static_cast<size_t>(s.n_layers));
            for (long i = 0; i < s.n_layers; ++i)
            {
                const std::string p = "blk." + std::to_string(i) + ".";
                layer& L = layers_[static_cast<size_t>(i)];
                L.mixer_scale = read_floats(g, p + "geo_mixer_scale", 1)[0];
                L.mixer_bias = read_floats(g, p + "geo_mixer_bias", 1)[0];
                L.ffn_scale = read_floats(g, p + "geo_mlp_scale", 1)[0];
                L.ffn_bias = read_floats(g, p + "geo_mlp_bias", 1)[0];
                if (s.layer_types[static_cast<size_t>(i)] == 'M')
                {
                    load_matrix(g, p + "ssm_in.weight", d, H * (2 * P + 3), L.ssm_in, opt);
                    load_matrix(g, p + "ssm_in_dyn.weight", d, 2 * R * N + S, L.ssm_dyn, opt);
                    load_matrix(g, p + "attn_output.weight", H * P, d, L.wo, opt);
                    load_packed(g, p + "ssm_m3_norms", p + "ssm_b_norm.weight", p + "ssm_c_norm.weight", 1, N,
                        L.ssm_norms);
                    load_packed(g, p + "ssm_m3_bias", p + "ssm_b_bias", p + "ssm_c_bias", H, R * N, L.ssm_bias);
                    load_packed(g, p + "ssm_m3_misc", p + "ssm_dt_bias", p + "ssm_d", 1, H, L.ssm_misc);
                    load_packed(g, p + "ssm_m3_mxz", p + "ssm_mimo_x", p + "ssm_mimo_z", H, R * P, L.ssm_mxz);
                    load_gamma(g, p + "ssm_mimo_o", H * R * P, L.ssm_mo);
                    L.ssm_state.set_size(1, 1, 1, H * S + H * P * N + H * R * N + H * P);
                    L.ssm_state = 0;
                }
                else
                {
                    load_matrix(g, p + "attn_q.weight", d, s.n_heads * hd, L.wq, opt);
                    load_matrix(g, p + "attn_wa_k.weight", d, KV * rank, L.wa_k, opt);
                    load_matrix(g, p + "attn_wb_k.weight", d, rank * hd, L.wb_k, opt);
                    load_matrix(g, p + "attn_wa_v.weight", d, KV * rank, L.wa_v, opt);
                    load_matrix(g, p + "attn_wb_v.weight", d, rank * hd, L.wb_v, opt);
                    load_matrix(g, p + "attn_shift_k.weight", d, KV, L.shift_k, opt);
                    load_matrix(g, p + "attn_shift_v.weight", d, KV, L.shift_v, opt);
                    load_matrix(g, p + "attn_lambda.weight", d, KV, L.noise, opt);
                    load_matrix(g, p + "attn_gate.weight", d, signal, L.attn_gate, opt);
                    load_matrix(g, p + "attn_output.weight", signal, d, L.wo, opt);
                    load_optional_head_gamma(g, p + "attn_q_norm.weight", L.qnorm_g);
                    load_optional_head_gamma(g, p + "attn_k_norm.weight", L.knorm_g);
                    if (L.qnorm_g.size() == 0 || L.knorm_g.size() == 0)
                        throw std::runtime_error("runtime_transformer: missing tensor " + p
                            + "attn_q_norm.weight");
                    L.softmax_scaler = read_floats(g, p + "attn_softmax_scaler", static_cast<size_t>(s.n_heads));
                    L.last_k.set_size(1, 1, 1, KV * hd);
                    L.last_v.set_size(1, 1, 1, KV * hd);
                }
                load_matrix(g, p + "ffn_gate_inp.weight", d, s.n_experts, L.router, opt);
                L.expert_bias = read_floats(g, p + "exp_probs_b", static_cast<size_t>(s.n_experts));
                load_matrix(g, p + "ffn_latent_down.weight", d, s.d_latent, L.latent_down, opt);
                load_matrix(g, p + "ffn_latent_up.weight", s.d_latent, d, L.latent_up, opt);
                load_expert_matrices(g, p + "ffn_up_exps.weight", s.d_latent, s.d_ffn_expert, L.e_up);
                load_expert_matrices(g, p + "ffn_down_exps.weight", s.d_ffn_expert, s.d_latent, L.e_down);
                load_matrix(g, p + "ffn_up_shexp.weight", d, s.d_ffn_shared, L.shared_up, opt);
                load_matrix(g, p + "ffn_down_shexp.weight", s.d_ffn_shared, d, L.shared_down, opt);
            }
            load_matrix(g, "output.weight", d, s.vocab_size, lm_head_, opt);
        }

        // The logits of every token of a prompt, read one token after the other.
        const tensor& hybrid_prefill(const std::vector<int>& tokens)
        {
            const long N = static_cast<long>(tokens.size());
            if (cap_ == 0 || own_context_)
            {
                // Without a context, the pass keeps its keys, values and states for its own tokens.
                cap_ = N;
                own_context_ = true;
                reset_cache();
            }
            DLIB_CASSERT(cur_len_ == 0, "forward_prefill() requires an empty cache; use step() to extend");
            if (N > cap_) throw std::runtime_error("runtime_transformer: prompt exceeds the context capacity");
            injected_.clear();
            const long V = spec_.vocab_size, d = spec_.d_model;
            all_logits_.set_size(1, 1, N, V);
            all_hidden_.set_size(1, 1, N, d);
            alias_tensor row(1, 1, 1, V), state(1, 1, 1, d);
            for (long t = 0; t < N; ++t)
            {
                hybrid_token(tokens[static_cast<size_t>(t)]);
                auto dst = row(all_logits_, static_cast<size_t>(t) * V);
                memcpy(dst, logits_);
                auto hs = state(all_hidden_, static_cast<size_t>(t) * d);
                memcpy(hs, x_);
            }
            // The hidden states of the pass, which hidden_states() returns.
            xn_ = all_hidden_;
            return all_logits_;
        }

        // Reads one token: its logits in logits_, its hidden state in xn_.
        void hybrid_token(int token)
        {
            DLIB_CASSERT(token >= 0 && token < spec_.vocab_size, "token id out of range");
            const long d = spec_.d_model;
            x_.set_size(1, 1, 1, d);
            {
                // The embedding enters the sphere the blocks keep the hidden state on.
                float* x = x_.host_write_only();
                gather_embedding_row(token, x);
                double ss = 0.0;
                for (long i = 0; i < d; ++i) ss += static_cast<double>(x[i]) * x[i];
                const float inv = static_cast<float>(1.0 / std::max(std::sqrt(ss), 1e-12));
                for (long i = 0; i < d; ++i) x[i] *= inv;
            }
            for (size_t i = 0; i < layers_.size(); ++i)
            {
                layer& L = layers_[i];
                const long depth = static_cast<long>(i) + 1;
                if (spec_.layer_types[i] == 'M') mamba3_mixer(L);
                else differential_attention(L);
                tt::geodesic_update(x_, attn_out_, L.mixer_scale, L.mixer_bias, depth);
                latent_experts(L);
                tt::geodesic_update(x_, ffn_out_, L.ffn_scale, L.ffn_bias, depth);
            }
            ++cur_len_;
            xn_ = x_;
            project(logits_, x_, lm_head_);
        }

        void mamba3_mixer(layer& L)
        {
            project(m_in_, x_, L.ssm_in);
            project(m_dyn_, x_, L.ssm_dyn);
            m_out_.set_size(1, 1, 1, spec_.ssm_heads * spec_.ssm_head_dim);
            tt::mamba3_step(m_out_, L.ssm_state, m_in_, m_dyn_, L.ssm_norms, L.ssm_bias, L.ssm_misc, L.ssm_mxz,
                L.ssm_mo, static_cast<float>(spec_.rms_eps));
            project(attn_out_, m_out_, L.wo);
        }

        // A key, or a value, of each head: the mean of the products of a factor of the head and a row of its width.
        void tensor_products(stored_matrix& wa, stored_matrix& wb, resizable_tensor& out)
        {
            const long KV = spec_.n_kv_heads, rank = spec_.tpa_rank, hd = head_dim();
            project(factor_a_, x_, wa);
            project(factor_b_, x_, wb);
            out.set_size(1, 1, 1, KV * hd);
            alias_tensor a_2d(KV, rank), b_2d(rank, hd), o_2d(KV, hd);
            auto av = a_2d(factor_a_, 0);
            auto bv = b_2d(factor_b_, 0);
            auto ov = o_2d(out, 0);
            tt::gemm(0.0f, ov, 1.0f / static_cast<float>(rank), av, false, bv, false);
        }

        void differential_attention(layer& L)
        {
            const long H = spec_.n_heads, KV = spec_.n_kv_heads, hd = head_dim();
            const long group = H / KV;
            const long pos = cur_len_, len = cur_len_ + 1;
            const float eps = static_cast<float>(spec_.rms_eps);
            auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

            project(q_flat_, x_, L.wq);
            tensor_products(L.wa_k, L.wb_k, k_flat_);
            tensor_products(L.wa_v, L.wb_v, v_flat_);

            /* The shift: each key and each value leans on the one of the previous token,
               as it stood before its own shift. */
            project(shift_, x_, L.shift_k);
            project(noise_, x_, L.shift_v);
            {
                float* k = k_flat_.host();
                float* v = v_flat_.host();
                float* lk = L.last_k.host();
                float* lv = L.last_v.host();
                const float* sk = shift_.host();
                const float* sv = noise_.host();
                for (long h = 0; h < KV; ++h)
                {
                    const float ak = sigmoid(sk[h]), av = sigmoid(sv[h]);
                    for (long e = 0; e < hd; ++e)
                    {
                        const float raw_k = k[h * hd + e], raw_v = v[h * hd + e];
                        if (pos > 0)
                        {
                            k[h * hd + e] = ak * lk[h * hd + e] + (1.0f - ak) * raw_k;
                            v[h * hd + e] = av * lv[h * hd + e] + (1.0f - av) * raw_v;
                        }
                        lk[h * hd + e] = raw_k;
                        lv[h * hd + e] = raw_v;
                    }
                }
            }

            q4_.set_size(1, H, 1, hd);
            k4_.set_size(1, KV, 1, hd);
            v4_.set_size(1, KV, 1, hd);
            tt::split_heads(false, q4_, q_flat_);
            tt::split_heads(false, k4_, k_flat_);
            tt::split_heads(false, v4_, v_flat_);
            apply_qk_norm(L, eps);
            {
                // The queries grow with the logarithm of the positions they see, each head by its own factor.
                long seen = len;
                if (spec_.attention_window > 0) seen = std::min(seen, spec_.attention_window);
                const float reach = static_cast<float>(std::log(static_cast<double>(seen)));
                float* q = q4_.host();
                for (long h = 0; h < H; ++h)
                {
                    const float f = L.softmax_scaler[static_cast<size_t>(h)] * reach;
                    for (long e = 0; e < hd; ++e) q[h * hd + e] *= f;
                }
            }
            append_to_cache(L, k4_, v4_, 1);

            // The keys and values of the sequence, each key and value head given to its group of query heads.
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
            k_rep_.set_size(1, H, len, hd);
            v_rep_.set_size(1, H, len, hd);
            for (long h = 0; h < H; ++h)
            {
                tt::copy_tensor(false, k_rep_, h, k_win_, h / group, 1);
                tt::copy_tensor(false, v_rep_, h, v_win_, h / group, 1);
            }
            scores_.set_size(1, H, 1, len);
            tt::gemm(0.0f, scores_, static_cast<float>(spec_.attention_scale), q4_, false, k_rep_, true,
                operation_mode::PLANE_WISE);
            attn_.copy_size(scores_);
            if (spec_.attn_softcap > 0.0)
            {
                // The scores are capped: cap * tanh(score / cap).
                const float cap = static_cast<float>(spec_.attn_softcap);
                tt::affine_transform(attn_, scores_, 1.0f / cap, 0.0f);
                tt::tanh(scores_, attn_);
                tt::affine_transform(scores_, scores_, cap, 0.0f);
            }
            tt::softmax(attn_, scores_, operation_mode::PLANE_WISE);
            ctx4_.set_size(1, H, 1, hd);
            tt::gemm(0.0f, ctx4_, 1.0f, attn_, false, v_rep_, false, operation_mode::PLANE_WISE);

            /* Each group of heads gives up the noise its last head reads, in the proportion
               the token chooses; the gate of the layer then opens each value of the result. */
            const long signal = group - 1;
            project(noise_, x_, L.noise);
            diff_.set_size(1, 1, 1, KV * signal * hd);
            {
                const float* c = ctx4_.host();
                const float* share = noise_.host();
                float* o = diff_.host_write_only();
                for (long g = 0; g < KV; ++g)
                {
                    const float lambda = sigmoid(share[g]);
                    const float* heads = c + g * group * hd;
                    const float* noise = heads + signal * hd;
                    for (long i = 0; i < signal; ++i)
                        for (long e = 0; e < hd; ++e)
                            o[(g * signal + i) * hd + e] = heads[i * hd + e] - lambda * noise[e];
                }
            }
            project(gate_, x_, L.attn_gate);
            if (spec_.gate_bias != 0.0)
                tt::affine_transform(gate_, gate_, 1.0f, static_cast<float>(spec_.gate_bias));
            sig_.copy_size(gate_);
            tt::sigmoid(sig_, gate_);
            tt::multiply(false, gate_, gate_, sig_);
            tt::multiply(false, gate_, gate_, diff_);
            project(attn_out_, gate_, L.wo);
        }

        /* The mixture of experts of a hybrid block, written in ffn_out_: the router gives
           each expert a sigmoid, the token takes the experts of the largest sigmoids once
           their biases are added, with the sigmoids themselves, brought to a sum of one and
           scaled, as their weights; the experts work in a latent space, with the square of
           a rectified linear unit between their two matrices, as does the shared expert,
           which reads the hidden state itself. */
        void latent_experts(layer& L)
        {
            const long n_exp = spec_.n_experts;
            const long top_k = spec_.n_experts_used;
            project(router_logits_, x_, L.router);
            const float* rl = router_logits_.host();
            probs_.resize(static_cast<size_t>(2 * n_exp));
            for (long e = 0; e < n_exp; ++e)
            {
                probs_[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-rl[e]));
                probs_[static_cast<size_t>(n_exp + e)] = probs_[static_cast<size_t>(e)]
                    + L.expert_bias[static_cast<size_t>(e)];
            }
            route_e_.assign(static_cast<size_t>(top_k), 0);
            route_w_.assign(static_cast<size_t>(top_k), 0.0f);
            float chosen = 0.0f;
            for (long k = 0; k < top_k; ++k)
            {
                long best = -1;
                for (long e = 0; e < n_exp; ++e)
                {
                    bool taken = false;
                    for (long j = 0; j < k; ++j) taken = taken || route_e_[static_cast<size_t>(j)] == e;
                    const float* ranked = probs_.data() + n_exp;
                    if (!taken && (best < 0 || ranked[e] > ranked[best])) best = e;
                }
                route_e_[static_cast<size_t>(k)] = best;
                route_w_[static_cast<size_t>(k)] = probs_[static_cast<size_t>(best)];
                chosen += probs_[static_cast<size_t>(best)];
            }
            for (long k = 0; k < top_k; ++k)
            {
                if (top_k > 1) route_w_[static_cast<size_t>(k)] /= chosen + 1e-20f;
                route_w_[static_cast<size_t>(k)] *= static_cast<float>(spec_.expert_weights_scale);
            }

            project(latent_, x_, L.latent_down);
            routed_.set_size(1, 1, 1, spec_.d_latent);
            routed_ = 0;
            for (long k = 0; k < top_k; ++k)
            {
                const size_t e = static_cast<size_t>(route_e_[static_cast<size_t>(k)]);
                project(eu_, latent_, L.e_up[e]);
                esig_.copy_size(eu_);
                tt::relu(esig_, eu_);
                tt::multiply(false, eu_, esig_, esig_);
                project(ey_, eu_, L.e_down[e]);
                tt::affine_transform(routed_, routed_, ey_, 1.0f, route_w_[static_cast<size_t>(k)]);
            }
            project(ffn_out_, routed_, L.latent_up);

            project(gate_, x_, L.shared_up);
            sig_.copy_size(gate_);
            tt::relu(sig_, gate_);
            tt::multiply(false, gate_, sig_, sig_);
            project(up_, gate_, L.shared_down);
            tt::add(ffn_out_, ffn_out_, up_);
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
        /* Embeddings supplied for chosen positions of the next prefill, and where they go.
           One shot: cleared as soon as they are consumed. */
        resizable_tensor injected_rows_;
        std::vector<long> injected_;
        resizable_tensor gate_, up_, sig_, ffn_out_;
        resizable_tensor cos_cache_, sin_cache_, mask_, logits_;

        /* Hybrid decoders: the work buffers of their layers, and whether the context is
           the one a prefill sized for its own tokens (no set_context()). */
        resizable_tensor m_in_, m_dyn_, m_out_, factor_a_, factor_b_, shift_, noise_, diff_, latent_, routed_;
        resizable_tensor all_logits_, all_hidden_;
        bool own_context_ = false;

        // Adapters and training

        void drop_adapters()
        {
            adapters_.clear();
            sites_.clear();
            merged_sites_.clear();
            site_of_.clear();
            traces_.clear();
            plan_ = adapter_plan();
            adapter_params_.set_size(0, 0, 0, 0);
            adapter_grads_.set_size(0, 0, 0, 0);
        }

        stored_matrix& projection(layer& L, int p)
        {
            switch (p)
            {
            case 0: return L.wq;
            case 1: return L.wk;
            case 2: return L.wv;
            case 3: return L.wo;
            case 4: return L.w_gate;
            case 5: return L.w_up;
            default: return L.w_down;
            }
        }

        long site_index(size_t li, int p) const
        {
            if (adapters_.empty()) return -1;
            return site_of_[li * 7 + static_cast<size_t>(p)];
        }

        /* Adds the adapter of projection p of layer li to y = x.W, where one is configured.
           The base weight is read by DoRA only, so LoRA never materializes it again. */
        void adapt(size_t li, int p, const tensor& x, stored_matrix& w, tensor& y)
        {
            const long s = site_index(li, p);
            if (s < 0) return;
            low_rank_adapter& a = adapters_[static_cast<size_t>(s)];
            const adapter_site& site = sites_[static_cast<size_t>(s)];
            auto av = a.a_view()(adapter_params_, site.geometry.a_offset(site.offset));
            auto bv = a.b_view()(adapter_params_, site.geometry.b_offset(site.offset));
            auto mv = a.m_view()(adapter_params_, site.geometry.m_offset(site.offset));
            const tensor& base = site.geometry.method == adapter_method::dora
                ? weight(w) : static_cast<const tensor&>(adapter_params_);
            a.forward(x, base, av, bv, mv, y);
        }

        /* dx += dy.W' for projection p of layer li, through its adapter when it has one,
           whose parameter gradients then receive their part. */
        void project_backward(size_t li, int p, const tensor& x, stored_matrix& w, const tensor& dy, tensor& dx)
        {
            const long s = site_index(li, p);
            if (s < 0)
            {
                tt::gemm(1.0f, dx, 1.0f, dy, false, weight(w), true);
                return;
            }
            low_rank_adapter& a = adapters_[static_cast<size_t>(s)];
            const adapter_site& site = sites_[static_cast<size_t>(s)];
            auto av = a.a_view()(adapter_params_, site.geometry.a_offset(site.offset));
            auto bv = a.b_view()(adapter_params_, site.geometry.b_offset(site.offset));
            auto mv = a.m_view()(adapter_params_, site.geometry.m_offset(site.offset));
            auto dav = a.a_view()(adapter_grads_, site.geometry.a_offset(site.offset));
            auto dbv = a.b_view()(adapter_grads_, site.geometry.b_offset(site.offset));
            auto dmv = a.m_view()(adapter_grads_, site.geometry.m_offset(site.offset));
            a.backward(x, weight(w), av, bv, mv, dy, dx, dav, dbv, dmv);
        }

        void make_resident(stored_matrix& m)
        {
            if (m.resident.size() != 0)
                return;
            const tensor& w = weight(m);
            m.resident.copy_size(w);
            memcpy(m.resident, w);
            m.raw.clear();
            m.raw.shrink_to_fit();
        }

        std::string model_signature() const
        {
            std::ostringstream sout;
            sout << spec_.arch_name << ':' << spec_.n_layers << ':' << spec_.d_model << ':' << spec_.d_ffn << ':'
                 << spec_.n_heads << ':' << spec_.n_kv_heads << ':' << head_dim() << ':' << spec_.vocab_size;
            return sout.str();
        }

        /* What the backward pass of one block reads: its inputs, the normalizations and
           their scales, the queries and keys before and after their normalization and
           rotation, the attention weights, and the activations of the feed-forward block.
           The attention weights take n_heads*N*N floats per block, which bounds the length
           of a training sequence before anything else does. */
        struct layer_trace
        {
            resizable_tensor x_in, attn_scale, xn_attn;
            resizable_tensor q_pre, k_pre, q_scale, k_scale;
            resizable_tensor q_rot, k_rot, v, attn, ctx_flat;
            resizable_tensor x_mid, ffn_scale, xn_ffn, gate, up, hidden;
        };

        void forward_train(const std::vector<int>& tokens)
        {
            const long N  = static_cast<long>(tokens.size());
            const long d  = spec_.d_model;
            const long H  = spec_.n_heads;
            const long KV = spec_.n_kv_heads;
            const long hd = head_dim();
            const long F  = spec_.d_ffn;
            const long rep = H / KV;
            const float qk_scale = spec_.attention_scale > 0.0
                ? static_cast<float>(spec_.attention_scale)
                : 1.0f / std::sqrt(static_cast<float>(hd));
            const float eps = static_cast<float>(spec_.rms_eps);

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
            traces_.resize(layers_.size());

            for (size_t li = 0; li < layers_.size(); ++li)
            {
                layer& L = layers_[li];
                layer_trace& T = traces_[li];
                T.x_in.copy_size(x_);
                memcpy(T.x_in, x_);
                tt::rms_normalize(eps, T.xn_attn, T.attn_scale, x_, L.attn_norm_g, true);

                q_flat_.set_size(1, 1, N, H * hd);
                k_flat_.set_size(1, 1, N, KV * hd);
                v_flat_.set_size(1, 1, N, KV * hd);
                {
                    alias_tensor x_2d(N, d), q_2d(N, H * hd), kv_2d(N, KV * hd);
                    auto xv = x_2d(T.xn_attn, 0);
                    auto qv = q_2d(q_flat_, 0);
                    auto kv_v = kv_2d(k_flat_, 0);
                    auto vv = kv_2d(v_flat_, 0);
                    tt::gemm(0.0f, qv, 1.0f, xv, false, weight(L.wq), false);
                    tt::gemm(0.0f, kv_v, 1.0f, xv, false, weight(L.wk), false);
                    tt::gemm(0.0f, vv, 1.0f, xv, false, weight(L.wv), false);
                    adapt(li, 0, xv, L.wq, qv);
                    adapt(li, 1, xv, L.wk, kv_v);
                    adapt(li, 2, xv, L.wv, vv);
                    if (L.bq.size())
                    {
                        ensure_ones(N);
                        tt::gemm(1.0f, qv, 1.0f, ones_, false, L.bq, false);
                        tt::gemm(1.0f, kv_v, 1.0f, ones_, false, L.bk, false);
                        tt::gemm(1.0f, vv, 1.0f, ones_, false, L.bv, false);
                    }
                }
                T.q_rot.set_size(1, H, N, hd);
                T.k_rot.set_size(1, KV, N, hd);
                T.v.set_size(1, KV, N, hd);
                tt::split_heads(false, T.q_rot, q_flat_);
                tt::split_heads(false, T.k_rot, k_flat_);
                tt::split_heads(false, T.v, v_flat_);
                if (L.qnorm_g.size() != 0)
                {
                    T.q_pre.copy_size(T.q_rot);
                    memcpy(T.q_pre, T.q_rot);
                    T.k_pre.copy_size(T.k_rot);
                    memcpy(T.k_pre, T.k_rot);
                    tt::rms_normalize(eps, T.q_rot, T.q_scale, T.q_pre, L.qnorm_g, true);
                    tt::rms_normalize(eps, T.k_rot, T.k_scale, T.k_pre, L.knorm_g, true);
                }
                tt::apply_rotary_positional_embedding(false, T.q_rot, cos_cache_, sin_cache_);
                tt::apply_rotary_positional_embedding(false, T.k_rot, cos_cache_, sin_cache_);

                k_rep_.set_size(1, H, N, hd);
                v_rep_.set_size(1, H, N, hd);
                for (long h = 0; h < H; ++h)
                {
                    tt::copy_tensor(false, k_rep_, h, T.k_rot, h / rep, 1);
                    tt::copy_tensor(false, v_rep_, h, T.v, h / rep, 1);
                }
                scores_.set_size(1, H, N, N);
                tt::gemm(0.0f, scores_, qk_scale, T.q_rot, false, k_rep_, true, operation_mode::PLANE_WISE);
                {
                    alias_tensor plane(1, 1, N, N);
                    for (long h = 0; h < H; ++h)
                    {
                        auto sp = plane(scores_, static_cast<size_t>(h) * N * N);
                        tt::add(sp, sp, mask_);
                    }
                }
                T.attn.copy_size(scores_);
                tt::softmax(T.attn, scores_, operation_mode::PLANE_WISE);
                ctx4_.set_size(1, H, N, hd);
                tt::gemm(0.0f, ctx4_, 1.0f, T.attn, false, v_rep_, false, operation_mode::PLANE_WISE);
                T.ctx_flat.set_size(1, 1, N, H * hd);
                tt::merge_heads(false, T.ctx_flat, ctx4_);
                attn_out_.set_size(1, 1, N, d);
                {
                    alias_tensor c_2d(N, H * hd), o_2d(N, d);
                    auto cv = c_2d(T.ctx_flat, 0);
                    auto ov = o_2d(attn_out_, 0);
                    tt::gemm(0.0f, ov, 1.0f, cv, false, weight(L.wo), false);
                    adapt(li, 3, cv, L.wo, ov);
                }
                add_residual(x_, attn_out_);

                T.x_mid.copy_size(x_);
                memcpy(T.x_mid, x_);
                tt::rms_normalize(eps, T.xn_ffn, T.ffn_scale, x_, L.ffn_norm_g, true);
                T.gate.set_size(1, 1, N, F);
                T.up.set_size(1, 1, N, F);
                {
                    alias_tensor n_2d(N, d), f_2d(N, F);
                    auto nv = n_2d(T.xn_ffn, 0);
                    auto gv = f_2d(T.gate, 0);
                    auto uv = f_2d(T.up, 0);
                    tt::gemm(0.0f, gv, 1.0f, nv, false, weight(L.w_gate), false);
                    tt::gemm(0.0f, uv, 1.0f, nv, false, weight(L.w_up), false);
                    adapt(li, 4, nv, L.w_gate, gv);
                    adapt(li, 5, nv, L.w_up, uv);
                }
                sig_.copy_size(T.gate);
                tt::sigmoid(sig_, T.gate);
                T.hidden.copy_size(T.gate);
                tt::multiply(false, T.hidden, T.gate, sig_);
                tt::multiply(false, T.hidden, T.hidden, T.up);
                ffn_out_.set_size(1, 1, N, d);
                {
                    alias_tensor h_2d(N, F), f_2d(N, d);
                    auto hv = h_2d(T.hidden, 0);
                    auto fv = f_2d(ffn_out_, 0);
                    tt::gemm(0.0f, fv, 1.0f, hv, false, weight(L.w_down), false);
                    adapt(li, 6, hv, L.w_down, fv);
                }
                add_residual(x_, ffn_out_);
            }

            final_x_.copy_size(x_);
            memcpy(final_x_, x_);
            tt::rms_normalize(eps, xn_, final_scale_, x_, final_norm_g_, true);
            logits_.set_size(1, 1, N, spec_.vocab_size);
            {
                alias_tensor n_2d(N, d), l_2d(N, spec_.vocab_size);
                auto nv = n_2d(xn_, 0);
                auto lv = l_2d(logits_, 0);
                tt::gemm(0.0f, lv, logit_alpha(), nv, false, weight(lm_head_), false);
            }
        }

        float logit_alpha() const
        {
            return spec_.logit_scale > 0.0 ? 1.0f / static_cast<float>(spec_.logit_scale) : 1.0f;
        }

        /* Summed cross-entropy of the supervised positions of logits_. With gradient set,
           dlogits_ receives scale times its gradient. */
        double cross_entropy(const std::vector<int>& targets, long& supervised, bool gradient, float scale)
        {
            const long N = logits_.nr();
            const long V = spec_.vocab_size;
            DLIB_CASSERT(static_cast<long>(targets.size()) == N);
            const float* lg = logits_.host();
            float* dl = nullptr;
            if (gradient)
            {
                dlogits_.copy_size(logits_);
                dl = dlogits_.host_write_only();
                std::fill(dl, dl + dlogits_.size(), 0.0f);
            }
            double loss = 0;
            supervised = 0;
            for (long t = 0; t < N; ++t)
            {
                const int y = targets[static_cast<size_t>(t)];
                if (y < 0)
                    continue;
                DLIB_CASSERT(y < V, "target out of range");
                const float* row = lg + static_cast<size_t>(t) * V;
                float mx = row[0];
                for (long v = 1; v < V; ++v) mx = std::max(mx, row[v]);
                double sum = 0;
                for (long v = 0; v < V; ++v) sum += std::exp(static_cast<double>(row[v] - mx));
                loss += std::log(sum) - (row[y] - mx);
                ++supervised;
                if (dl)
                {
                    float* g = dl + static_cast<size_t>(t) * V;
                    const double inv = 1.0 / sum;
                    for (long v = 0; v < V; ++v)
                        g[v] = static_cast<float>(scale * std::exp(static_cast<double>(row[v] - mx)) * inv);
                    g[y] -= scale;
                }
            }
            return loss;
        }

        void backward_train()
        {
            const long N  = logits_.nr();
            const long d  = spec_.d_model;
            const long H  = spec_.n_heads;
            const long KV = spec_.n_kv_heads;
            const long hd = head_dim();
            const long F  = spec_.d_ffn;
            const long rep = H / KV;
            const float qk_scale = spec_.attention_scale > 0.0
                ? static_cast<float>(spec_.attention_scale)
                : 1.0f / std::sqrt(static_cast<float>(hd));
            const float rs = static_cast<float>(spec_.residual_scale);
            gamma_grad_.set_size(1, 1, 1, d);
            head_gamma_grad_.set_size(1, 1, 1, hd);

            // logits = alpha . xn.W_head, then the final normalization.
            dxn_.set_size(1, 1, N, d);
            {
                alias_tensor l_2d(N, spec_.vocab_size), n_2d(N, d);
                auto gl = l_2d(dlogits_, 0);
                auto gn = n_2d(dxn_, 0);
                tt::gemm(0.0f, gn, logit_alpha(), gl, false, weight(lm_head_), true);
            }
            dx_.copy_size(final_x_);
            dx_ = 0;
            tt::rms_normalize_gradient(dxn_, final_scale_, final_x_, final_norm_g_, dx_, gamma_grad_, dscale_, true);

            for (size_t step = layers_.size(); step-- > 0; )
            {
                layer& L = layers_[step];
                layer_trace& T = traces_[step];

                /* ---- feed-forward block: x_out = x_mid + rs.down(silu(gate) * up) ---- */
                dres_.copy_size(dx_);
                tt::affine_transform(dres_, dx_, rs, 0.0f);
                dhidden_.copy_size(T.hidden);
                dhidden_ = 0;
                {
                    alias_tensor h_2d(N, F), f_2d(N, d);
                    auto hv = h_2d(T.hidden, 0);
                    auto gy = f_2d(dres_, 0);
                    auto gh = h_2d(dhidden_, 0);
                    project_backward(step, 6, hv, L.w_down, gy, gh);
                }
                sig_.copy_size(T.gate);
                tt::sigmoid(sig_, T.gate);
                silu_.copy_size(T.gate);
                tt::multiply(false, silu_, T.gate, sig_);
                dup_.copy_size(T.gate);
                tt::multiply(false, dup_, dhidden_, silu_);
                // silu'(g) = s.(1 + g.(1 - s)), with s = sigmoid(g)
                work_.copy_size(T.gate);
                tt::affine_transform(work_, sig_, -1.0f, 1.0f);
                tt::multiply(false, work_, work_, T.gate);
                tt::affine_transform(work_, work_, 1.0f, 1.0f);
                tt::multiply(false, work_, work_, sig_);
                dgate_.copy_size(T.gate);
                tt::multiply(false, dgate_, dhidden_, T.up);
                tt::multiply(false, dgate_, dgate_, work_);
                dxn_.copy_size(T.xn_ffn);
                dxn_ = 0;
                {
                    alias_tensor n_2d(N, d), f_2d(N, F);
                    auto nv = n_2d(T.xn_ffn, 0);
                    auto gn = n_2d(dxn_, 0);
                    auto gg = f_2d(dgate_, 0);
                    auto gu = f_2d(dup_, 0);
                    project_backward(step, 4, nv, L.w_gate, gg, gn);
                    project_backward(step, 5, nv, L.w_up, gu, gn);
                }
                tt::rms_normalize_gradient(dxn_, T.ffn_scale, T.x_mid, L.ffn_norm_g, dx_, gamma_grad_, dscale_, true);

                /* ---- attention block: x_mid = x_in + rs.(attention . W_o) ---- */
                tt::affine_transform(dres_, dx_, rs, 0.0f);
                dctx_flat_.copy_size(T.ctx_flat);
                dctx_flat_ = 0;
                {
                    alias_tensor c_2d(N, H * hd), o_2d(N, d);
                    auto cv = c_2d(T.ctx_flat, 0);
                    auto go = o_2d(dres_, 0);
                    auto gc = c_2d(dctx_flat_, 0);
                    project_backward(step, 3, cv, L.wo, go, gc);
                }
                dctx4_.set_size(1, H, N, hd);
                tt::split_heads(false, dctx4_, dctx_flat_);

                k_rep_.set_size(1, H, N, hd);
                v_rep_.set_size(1, H, N, hd);
                for (long h = 0; h < H; ++h)
                {
                    tt::copy_tensor(false, k_rep_, h, T.k_rot, h / rep, 1);
                    tt::copy_tensor(false, v_rep_, h, T.v, h / rep, 1);
                }
                dattn_.set_size(1, H, N, N);
                tt::gemm(0.0f, dattn_, 1.0f, dctx4_, false, v_rep_, true, operation_mode::PLANE_WISE);
                dv_rep_.set_size(1, H, N, hd);
                tt::gemm(0.0f, dv_rep_, 1.0f, T.attn, true, dctx4_, false, operation_mode::PLANE_WISE);
                tt::softmax_gradient(dattn_, T.attn, dattn_, operation_mode::PLANE_WISE);
                dq_.set_size(1, H, N, hd);
                tt::gemm(0.0f, dq_, qk_scale, dattn_, false, k_rep_, false, operation_mode::PLANE_WISE);
                dk_rep_.set_size(1, H, N, hd);
                tt::gemm(0.0f, dk_rep_, qk_scale, dattn_, true, T.q_rot, false, operation_mode::PLANE_WISE);

                // The heads that share a key and value channel sum their gradients into it.
                dk_.set_size(1, KV, N, hd);
                dv_.set_size(1, KV, N, hd);
                dk_ = 0;
                dv_ = 0;
                for (long h = 0; h < H; ++h)
                {
                    tt::copy_tensor(true, dk_, h / rep, dk_rep_, h, 1);
                    tt::copy_tensor(true, dv_, h / rep, dv_rep_, h, 1);
                }
                tt::apply_rotary_positional_embedding(true, dq_, cos_cache_, sin_cache_);
                tt::apply_rotary_positional_embedding(true, dk_, cos_cache_, sin_cache_);
                if (L.qnorm_g.size() != 0)
                {
                    dpre_.copy_size(dq_);
                    dpre_ = 0;
                    tt::rms_normalize_gradient(dq_, T.q_scale, T.q_pre, L.qnorm_g, dpre_, head_gamma_grad_, dscale_, true);
                    memcpy(dq_, dpre_);
                    dpre_.copy_size(dk_);
                    dpre_ = 0;
                    tt::rms_normalize_gradient(dk_, T.k_scale, T.k_pre, L.knorm_g, dpre_, head_gamma_grad_, dscale_, true);
                    memcpy(dk_, dpre_);
                }
                q_flat_.set_size(1, 1, N, H * hd);
                k_flat_.set_size(1, 1, N, KV * hd);
                v_flat_.set_size(1, 1, N, KV * hd);
                tt::merge_heads(false, q_flat_, dq_);
                tt::merge_heads(false, k_flat_, dk_);
                tt::merge_heads(false, v_flat_, dv_);
                dxn_.copy_size(T.xn_attn);
                dxn_ = 0;
                {
                    alias_tensor x_2d(N, d), q_2d(N, H * hd), kv_2d(N, KV * hd);
                    auto xv = x_2d(T.xn_attn, 0);
                    auto gx = x_2d(dxn_, 0);
                    auto gq = q_2d(q_flat_, 0);
                    auto gk = kv_2d(k_flat_, 0);
                    auto gv = kv_2d(v_flat_, 0);
                    project_backward(step, 0, xv, L.wq, gq, gx);
                    project_backward(step, 1, xv, L.wk, gk, gx);
                    project_backward(step, 2, xv, L.wv, gv, gx);
                }
                tt::rms_normalize_gradient(dxn_, T.attn_scale, T.x_in, L.attn_norm_g, dx_, gamma_grad_, dscale_, true);
            }
        }

        std::vector<low_rank_adapter> adapters_;
        std::vector<adapter_site> sites_;
        std::vector<adapter_site> merged_sites_;
        std::vector<long> site_of_;
        adapter_plan plan_;
        resizable_tensor adapter_params_, adapter_grads_;
        std::vector<layer_trace> traces_;
        resizable_tensor final_x_, final_scale_, dlogits_, dx_, dxn_, dres_, dscale_, gamma_grad_, head_gamma_grad_;
        resizable_tensor dhidden_, silu_, dup_, dgate_, work_, dctx_flat_, dctx4_, dattn_, dv_rep_, dq_, dk_rep_;
        resizable_tensor dk_, dv_, dpre_;
    };
}

#endif // DLIB_DNN_RUNTIME_TRANSFORMER_H_
