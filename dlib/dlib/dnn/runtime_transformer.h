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
// according to the build, and is validated by logit parity against llama.cpp.
//
// Two weight residencies are supported:
//   - resident (default): every matrix is dequantized once at load time; fastest
//     forward, memory footprint of the full float32 model;
//   - quantized at rest: the raw GGUF bytes stay in memory and each matrix is
//     dequantized just in time into a small shared pool at every use, keeping the
//     footprint close to the on-disk file size (e.g. ~1.1 GB instead of ~7 GB for a
//     1.7B model), traded against one dequantization per matrix and per forward.
//
// This increment covers the prefill forward and full-sequence logits, which is what
// the probe-based validation protocol needs; the incremental KV-cache path for chat
// generation is the next increment.

#ifndef DLIB_DNN_RUNTIME_TRANSFORMER_H_
#define DLIB_DNN_RUNTIME_TRANSFORMER_H_

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
                load_matrix(g, p + "attn_output.weight", qd, d, L.wo, opt);
                load_gamma (g, p + "ffn_norm.weight", d, L.ffn_norm_g);
                load_matrix(g, p + "ffn_gate.weight", d, spec.d_ffn, L.w_gate, opt);
                load_matrix(g, p + "ffn_up.weight",   d, spec.d_ffn, L.w_up,   opt);
                load_matrix(g, p + "ffn_down.weight", spec.d_ffn, d, L.w_down, opt);
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

        // Full prefill forward. Returns the logits of every position in a
        // [1, 1, N, vocab] tensor.
        const tensor& forward_prefill(const std::vector<int>& tokens)
        {
            const long N  = static_cast<long>(tokens.size());
            const long d  = spec_.d_model;
            const long H  = spec_.n_heads;
            const long KV = spec_.n_kv_heads;
            const long hd = head_dim();
            const long rep = H / KV;
            const float qk_scale = 1.0f / std::sqrt(static_cast<float>(hd));
            const float eps = static_cast<float>(spec_.rms_eps);
            DLIB_CASSERT(N > 0);

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
                }

                q4_.set_size(1, H, N, hd);
                k4_.set_size(1, KV, N, hd);
                v4_.set_size(1, KV, N, hd);
                tt::split_heads(false, q4_, q_flat_);
                tt::split_heads(false, k4_, k_flat_);
                tt::split_heads(false, v4_, v_flat_);

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
                tt::add(x_, x_, attn_out_);

                /* ---- feed-forward block (SwiGLU) ---- */
                tt::rms_normalize(eps, xn_, rms_scale_, x_, L.ffn_norm_g, true);
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
                tt::add(x_, x_, ffn_out_);
            }

            /* ---- final norm and logits ---- */
            tt::rms_normalize(eps, xn_, rms_scale_, x_, final_norm_g_, true);
            logits_.set_size(1, 1, N, spec_.vocab_size);
            {
                alias_tensor n_2d(N, d), l_2d(N, spec_.vocab_size);
                auto nv = n_2d(xn_, 0);
                auto lv = l_2d(logits_, 0);
                tt::gemm(0.0f, lv, 1.0f, nv, false, weight(lm_head_), false);
            }
            return logits_;
        }

        const model_spec& spec() const { return spec_; }

    private:

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
            dst.permute = opt.rope_permute
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

        /* Source tensor for one gemm: the resident copy, or a just-in-time
           materialization into the rotating pool. The returned reference stays valid
           until the same pool slot is handed out again, i.e. beyond the single gemm
           that consumes it. */
        const tensor& weight(stored_matrix& m)
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

        struct layer
        {
            resizable_tensor attn_norm_g, ffn_norm_g;
            stored_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
        };

        model_spec spec_;
        bool at_rest_ = true;
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
        resizable_tensor pool_[2];
        int pool_next_ = 0;
        std::vector<float> scratch_;

        /* Work buffers, reused across layers and calls. */
        resizable_tensor x_, xn_, rms_scale_;
        resizable_tensor q_flat_, k_flat_, v_flat_, q4_, k4_, v4_, k_rep_, v_rep_;
        resizable_tensor scores_, attn_, ctx4_, ctx_flat_, attn_out_;
        resizable_tensor gate_, up_, sig_, ffn_out_;
        resizable_tensor cos_cache_, sin_cache_, mask_, logits_;
    };
}

#endif // DLIB_DNN_RUNTIME_TRANSFORMER_H_
