// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF -> Dlib weight import.
//
// Repacking stage of the import pipeline: dequantizes the tensors of a Llama-family
// GGUF file and copies them, with the required transpositions and RoPE permutations,
// into a network built from decoder_transformer_config. See the abstract header for
// the weight layout and the validation procedure.

#ifndef DLIB_GGUF_WEIGHT_LOADER_H_
#define DLIB_GGUF_WEIGHT_LOADER_H_

#include "gguf_weight_loader_abstract.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iostream>

#include "gguf_reader.h"
#include "gguf_dequantize.h"
#include "gguf_model_spec.h"
#include "../dnn.h"

namespace dlib
{
    struct gguf_load_options
    {
        bool rope_permute = false;
        bool swap_gate_up = false;
        bool verbose = true;
    };

    namespace gguf_load_impl
    {
        /* Transpose a GGUF [out_dim, in_dim] row-major matrix into a Dlib [in_dim, out_dim]
           row-major buffer. Optionally permute the output rows per head for RoPE. */
        inline void transpose_into(const std::vector<float>& src, long out_dim, long in_dim,
            float* dst, long num_heads, long head_dim, bool rope_permute)
        {
            for (long o = 0; o < out_dim; ++o)
            {
                long od = o;
                if (rope_permute && head_dim > 0 && num_heads > 0)
                {
                    /* Map GGUF (split-half) head ordering to interleaved head ordering. */
                    const long h = o / head_dim;
                    const long r = o % head_dim;
                    const long half = head_dim / 2;
                    const long ri = (r < half) ? (2 * r) : (2 * (r - half) + 1);
                    od = h * head_dim + ri;
                }
                const float* srow = src.data() + static_cast<size_t>(o) * in_dim;
                for (long i = 0; i < in_dim; ++i)
                    dst[static_cast<size_t>(i) * out_dim + od] = srow[i];
            }
        }

        inline std::vector<float> fetch(gguf_reader& g, const std::string& name)
        {
            const gguf_tensor_info* t = g.find_tensor(name);
            if (!t) throw std::runtime_error("import_gguf_weights: missing tensor '" + name + "'");
            std::vector<float> out;
            gguf_read_dequantized(g, *t, out);
            return out;
        }

        /* Copy src into the layer parameter tensor. Llama weights carry no bias, so when the
           Dlib layer is larger than the source (a linear layer keeps weights followed by a
           bias), the weight part is filled and the trailing bias is zeroed. */
        inline void copy_into(tensor& dst, const std::vector<float>& src, const std::string& what)
        {
            if (src.size() > dst.size())
                throw std::runtime_error("import_gguf_weights: source larger than layer for " + what
                    + " (layer " + std::to_string(dst.size()) + " vs source " + std::to_string(src.size()) + ")");
            float* h = dst.host();
            std::memcpy(h, src.data(), src.size() * sizeof(float));
            if (src.size() < dst.size())
                std::memset(h + src.size(), 0, (dst.size() - src.size()) * sizeof(float));
        }

        /* The embedding layer keeps its table in a private member exposed by get_embeddings,
           not in get_layer_params. This visitor targets exactly that layer (SFINAE on the
           presence of get_embeddings) and copies the table directly, no transpose, since GGUF
           token_embd is [vocab, d_model] and the Dlib table is [vocab, d_model] too. */
        struct embedding_loader
        {
            const std::vector<float>* src;
            bool* done;
            template <typename T>
            auto set(T& layer, int) -> decltype((void)layer.get_embeddings())
            {
                tensor& e = layer.get_embeddings();
                if (e.size() == src->size())
                {
                    std::memcpy(e.host(), src->data(), src->size() * sizeof(float));
                    *done = true;
                }
            }
            template <typename T> void set(T&, long) {}
            template <typename T> void operator()(T& layer) { set(layer, 0); }
        };

        /* Pushes the model's runtime attention settings into every attention layer that
           exposes them: the RoPE frequency base, the QK-Norm epsilon, and classical RoPE
           (YaRN disabled). SFINAE on set_rope_theta_base (only the fused attention layer
           has it) makes this a no-op for every other layer. Must run before the first
           forward pass, since the RoPE trig caches are built then and are not recomputed
           on a later base change. Disabling YaRN explicitly matters: the import pipeline
           runs a 1-token allocation forward, which would otherwise capture original_len=1
           and make any YaRN-style rescaling length-dependent and destructive. */
        struct attention_config_setter
        {
            float rope_base;
            double qk_eps;
            bool qkv_bias = false;
            template <typename T>
            auto set(T& layer, int) -> decltype((void)layer.set_rope_theta_base(0.f))
            {
                layer.set_rope_theta_base(rope_base);
                layer.set_qk_norm_eps(qk_eps);
                layer.set_yarn_params(0.0f, 0.0f, 0, false); // classical RoPE only
                layer.set_qkv_bias_enabled(qkv_bias);
            }
            template <typename T> void set(T&, long) {}
            template <typename T> void operator()(T& layer) { set(layer, 0); }
        };

        /* Pushes the model's RMSNorm epsilon into every rms_norm layer of the network
           (attention norms, FFN norms and the final norm). SFINAE on set_eps makes this
           a no-op for every other layer. Imported models carry their own value (e.g.
           1e-6 for Qwen3) which may differ from the library default. */
        struct norm_eps_setter
        {
            double eps;
            template <typename T>
            auto set(T& layer, int) -> decltype((void)layer.set_eps(0.0))
            {
                layer.set_eps(eps);
            }
            template <typename T> void set(T&, long) {}
            template <typename T> void operator()(T& layer) { set(layer, 0); }
        };

        /* Reads back the RoPE state actually held by the first attention layer: the
           theta base and three cosine values of the trig cache at position 1. Used by
           the verbose import diagnostic; a member-template visitor must live at
           namespace scope, since local classes cannot hold template members. */
        struct rope_probe_reader
        {
            float* theta; float* c0; float* c1; float* cl;
            template <typename T>
            auto rd(T& layer, int) -> decltype((void)layer.get_rope_cos_probe(0, 0))
            {
                if (*theta < 0.0f)
                {
                    *theta = layer.get_rope_theta_base();
                    *c0 = layer.get_rope_cos_probe(1, 0);
                    *c1 = layer.get_rope_cos_probe(1, 1);
                    *cl = layer.get_rope_cos_probe(1, -1);
                }
            }
            template <typename T> void rd(T&, long) {}
            template <typename T> void operator()(T& layer) { rd(layer, 0); }
        };
    }

    template <typename net_type>
    void import_gguf_weights(net_type& net, gguf_reader& g, const model_spec& spec,
        const gguf_load_options& opt = gguf_load_options())
    {
        /* QKV bias tensors present (qwen2 family)? Detected up front from the reader,
           since the config setter pushing the flag into the attention layers runs
           before the packing loop. */
        const bool has_qkv_bias = g.find_tensor("blk.0.attn_q.bias") != nullptr;
        using namespace gguf_load_impl;

        const long d = spec.d_model;
        const long kv = spec.n_kv_heads * spec.head_dim;
        const long ff = spec.d_ffn;
        const long nh = spec.n_heads;
        const long hd = spec.head_dim;
        const long qd = nh * hd;          // query/output projection width (decoupled head_dim)
        const long V = spec.vocab_size;
        const long L = spec.n_layers;

        /* Push RoPE base and QK-Norm epsilon into the attention layers before the first
           forward, so the RoPE caches are built with the right base (they are not rebuilt on a
           later change) and QK-Norm uses the model's epsilon. No-op for other layers. */
        {
            attention_config_setter cfg{ static_cast<float>(spec.rope_freq_base), spec.rms_eps,
                has_qkv_bias };
            visit_computational_layers(net, cfg);
            norm_eps_setter neps{ spec.rms_eps };
            visit_computational_layers(net, neps);
        }

        /* Read back the RoPE state actually held by the first attention layer: the theta
           base and three cosine values of the trig cache at position 1. cos(dim 0) must
           equal cos(1) ~= 0.5403 whatever the base (that frequency is base^0 = 1); a
           printed 1.0 means the rotation is effectively the identity. cos(dim 1) equals
           cos(base^(-2/head_dim)) and therefore identifies the base numerically. */
        if (opt.verbose)
        {
            float applied = -1.0f, c0 = 0, c1 = 0, cl = 0;
            rope_probe_reader pr{ &applied, &c0, &c1, &cl };
            visit_computational_layers(net, pr);
            std::cout << "  RoPE theta applied : " << applied
                      << "  cos@pos1: dim0=" << c0 << " dim1=" << c1 << " last=" << cl << "\n";
        }

        /* Allocate every parameter tensor with a single forward pass on one token. */
        network_context::reset();
        network_context::set_kv_cache_capacity(1);
        network_context::set_inference_mode(network_context::inference_mode::prefill);
        {
            matrix<int, 0, 1> dummy(1, 1);
            dummy(0) = 0;
            net(dummy);
        }
        network_context::reset();

        /* Collect the writable parameter tensors of the parameter-bearing computational
           layers, in network visit order. The vector is captured by reference so the result
           survives even though visit_computational_layers takes the visitor by value. */
        std::vector<tensor*> params;
        visit_computational_layers(net, [&params](auto& layer) {
            tensor& p = layer.get_layer_params();
            if (p.size() > 0) params.push_back(&p);
        });

        /* Expected order, from the output side inward (the order visit_computational_layers
           produces): lm_head, output_norm, then for each block from the last to the first
           [ffn_down, ffn_a, ffn_b, ffn_norm, attention, attn_norm], then the embedding. */
        size_t k = 0;
        auto next = [&](const std::string& what) -> tensor& {
            if (k >= params.size())
                throw std::runtime_error("import_gguf_weights: ran out of layers at " + what);
            return *params[k++];
        };

        /* The SwiGLU is down( silu(gate(x)) * up(x) ). In visit order (output to input) the
           two inner linears appear as up first, then gate. swap_gate_up flips the mapping. */
        const std::string up_inner   = opt.swap_gate_up ? "ffn_gate" : "ffn_up";
        const std::string gate_inner = opt.swap_gate_up ? "ffn_up" : "ffn_gate";

        /* lm_head: output.weight is [vocab, d_model] in GGUF -> transpose to [d_model, vocab]. */
        {
            const std::string name = spec.tied_embeddings ? "token_embd.weight" : "output.weight";
            std::vector<float> src = fetch(g, name), dst(static_cast<size_t>(d) * V);
            transpose_into(src, V, d, dst.data(), 0, 0, false);
            copy_into(next("lm_head"), dst, "lm_head");
        }
        copy_into(next("output_norm"), fetch(g, "output_norm.weight"), "output_norm");

        for (long b = L - 1; b >= 0; --b)
        {
            const std::string p = "blk." + std::to_string(b) + ".";

            /* ffn_down: [d_model, ff] in GGUF -> [ff, d_model]. */
            {
                std::vector<float> src = fetch(g, p + "ffn_down.weight"), dst(static_cast<size_t>(ff) * d);
                transpose_into(src, d, ff, dst.data(), 0, 0, false);
                copy_into(next("ffn_down"), dst, p + "ffn_down");
            }
            /* up then gate: the visit order of the two inner SwiGLU linears. Each is
               [ff, d_model] in GGUF -> [d_model, ff]. */
            {
                std::vector<float> src = fetch(g, p + up_inner + ".weight"), dst(static_cast<size_t>(d) * ff);
                transpose_into(src, ff, d, dst.data(), 0, 0, false);
                copy_into(next("ffn_up"), dst, p + up_inner);
            }
            {
                std::vector<float> src = fetch(g, p + gate_inner + ".weight"), dst(static_cast<size_t>(d) * ff);
                transpose_into(src, ff, d, dst.data(), 0, 0, false);
                copy_into(next("ffn_gate"), dst, p + gate_inner);
            }
            copy_into(next("ffn_norm"), fetch(g, p + "ffn_norm.weight"), p + "ffn_norm");

            /* Attention: one packed tensor [wq | wk | wv | wo (| gamma_q | gamma_k)]. wq is
               [d, qd] and wo is [qd, d] with qd = n_heads * head_dim, which equals d for the
               usual head_dim = d / n_heads and differs only when head_dim is decoupled (Qwen3). */
            {
                const size_t nq = static_cast<size_t>(d) * qd;
                const size_t nkv = static_cast<size_t>(d) * kv;
                const size_t ng = spec.qk_norm ? static_cast<size_t>(hd) : 0;
                /* The bias slots [bq|bk|bv] close the packed blob of the families that
                   carry QKV biases; a bias-free layer holds none. */
                const size_t nb = has_qkv_bias
                    ? static_cast<size_t>(qd) + 2 * static_cast<size_t>(kv) : 0;
                std::vector<float> fused(2 * nq + 2 * nkv + 2 * ng + nb, 0.0f);
                std::vector<float> sq = fetch(g, p + "attn_q.weight");
                std::vector<float> sk = fetch(g, p + "attn_k.weight");
                std::vector<float> sv = fetch(g, p + "attn_v.weight");
                std::vector<float> so = fetch(g, p + "attn_output.weight");
                /* qwen families keep the split-half (NeoX) Q/K layout; llama-family
                   files come pre-permuted for the interleaved kernel. */
                const bool neox_layout = opt.rope_permute
                    || spec.arch_name == "qwen2" || spec.arch_name == "qwen3";
                transpose_into(sq, qd, d, fused.data(), nh, hd, neox_layout);          // wq (RoPE)
                transpose_into(sk, kv, d, fused.data() + nq, spec.n_kv_heads, hd, neox_layout); // wk (RoPE)
                transpose_into(sv, kv, d, fused.data() + nq + nkv, 0, 0, false);            // wv
                transpose_into(so, d, qd, fused.data() + nq + nkv + nkv, 0, 0, false);      // wo
                if (spec.qk_norm)
                {
                    /* Per-head Q/K RMSNorm gammas (head_dim each), stored after W_O to match the
                       attention [wq|wk|wv|wo|gamma_q|gamma_k] parameter packing. Two adjustments
                       are applied here, both consequences of where the normalization sits in the
                       attention block.

                       The gamma indexes the same output space as the projection rows it scales,
                       so it carries the same rotary layout permutation. Copied in the container's
                       order while the rows are permuted, each coefficient would land on the wrong
                       channel.

                       No scale is folded in: the attention layer applies the
                       1 / sqrt(head_dim) factor at the score product, downstream of the
                       normalization, so the gamma stored here is exactly the gamma the source
                       model declares. */
                    std::vector<float> gq = fetch(g, p + "attn_q_norm.weight");
                    std::vector<float> gk = fetch(g, p + "attn_k_norm.weight");
                    if (static_cast<long>(gq.size()) != hd || static_cast<long>(gk.size()) != hd)
                        throw std::runtime_error("import_gguf_weights: unexpected QK-Norm gamma size in " + p);
                    const long half = hd / 2;
                    float* dst_gq = fused.data() + 2 * nq + 2 * nkv;
                    float* dst_gk = dst_gq + ng;
                    for (long r = 0; r < hd; ++r)
                    {
                        const long ri = neox_layout ? ((r < half) ? (2 * r) : (2 * (r - half) + 1)) : r;
                        dst_gq[ri] = gq[static_cast<size_t>(r)];
                        dst_gk[ri] = gk[static_cast<size_t>(r)];
                    }
                }
                if (has_qkv_bias)
                {
                    /* QKV biases (qwen2 family). The Q and K vectors live in the same
                       output space as the permuted projection rows, so the NeoX layout
                       reordering applies to them identically; V is rotation-free. */
                    std::vector<float> bq = fetch(g, p + "attn_q.bias");
                    std::vector<float> bk = fetch(g, p + "attn_k.bias");
                    std::vector<float> bv = fetch(g, p + "attn_v.bias");
                    if (static_cast<long>(bq.size()) != qd || static_cast<long>(bk.size()) != kv
                        || static_cast<long>(bv.size()) != kv)
                        throw std::runtime_error("import_gguf_weights: unexpected QKV bias size in " + p);
                    const size_t bb = 2 * nq + 2 * nkv + 2 * ng;
                    auto put = [&](const std::vector<float>& src, size_t off)
                    {
                        const long half = hd / 2;
                        for (long o = 0; o < static_cast<long>(src.size()); ++o)
                        {
                            const long h = o / hd, r = o % hd;
                            const long ri = neox_layout ? ((r < half) ? 2 * r : 2 * (r - half) + 1) : r;
                            fused[off + static_cast<size_t>(h) * hd + ri] = src[static_cast<size_t>(o)];
                        }
                    };
                    put(bq, bb);
                    put(bk, bb + static_cast<size_t>(qd));
                    /* V is rotation-free: copied verbatim, never permuted. */
                    std::memcpy(fused.data() + bb + qd + kv, bv.data(), bv.size() * sizeof(float));
                }
                copy_into(next("attention"), fused, p + "attention");
            }
            copy_into(next("attn_norm"), fetch(g, p + "attn_norm.weight"), p + "attn_norm");

            if (opt.verbose && (b % 8 == 0 || b == L - 1))
                std::cout << "  loaded block " << b << "\r" << std::flush;
        }

        /* The embedding table is the parameter tensor of the embedding layer, last in the
           visit order, and is filled below through get_embeddings(). */
        if (k + 1 == params.size() && params.back()->size() == static_cast<size_t>(V) * d)
            params.pop_back();
        if (k != params.size())
            throw std::runtime_error("import_gguf_weights: assigned " + std::to_string(k)
                + " of " + std::to_string(params.size())
                + " parameter tensors; the layer order assumption is wrong");

        /* The embedding table is held by the embeddings layer outside get_layer_params, so it
           is assigned separately through get_embeddings. */
        {
            std::vector<float> emb = fetch(g, "token_embd.weight");
            bool done = false;
            embedding_loader el{ &emb, &done };
            visit_computational_layers(net, el);
            if (!done)
                throw std::runtime_error("import_gguf_weights: failed to set the embedding table");
        }

        if (opt.verbose)
            std::cout << "  loaded " << k << " parameter tensors plus the embedding table\n";
    }

    namespace gguf_load_impl
    {
        /* The inverse of transpose_into: a Dlib [in_dim, out_dim] row-major matrix back to the
           [out_dim, in_dim] rows of the container, the per-head rotary permutation undone. */
        inline void transpose_from(const float* src, long out_dim, long in_dim, std::vector<float>& dst,
            long head_dim, bool rope_permute)
        {
            dst.assign(static_cast<size_t>(out_dim) * in_dim, 0.0f);
            const long half = head_dim / 2;
            for (long o = 0; o < out_dim; ++o)
            {
                long od = o;
                if (rope_permute && head_dim > 0)
                {
                    const long r = o % head_dim;
                    od = (o / head_dim) * head_dim + ((r < half) ? (2 * r) : (2 * (r - half) + 1));
                }
                float* drow = dst.data() + static_cast<size_t>(o) * in_dim;
                for (long i = 0; i < in_dim; ++i)
                    drow[i] = src[static_cast<size_t>(i) * out_dim + od];
            }
        }

        // A vector indexed like the rows of a permuted projection, back to the container order.
        inline std::vector<float> unpermute_vector(const float* src, long dim, long head_dim, bool rope_permute)
        {
            std::vector<float> dst(static_cast<size_t>(dim));
            const long half = head_dim / 2;
            for (long o = 0; o < dim; ++o)
            {
                long od = o;
                if (rope_permute && head_dim > 0)
                {
                    const long r = o % head_dim;
                    od = (o / head_dim) * head_dim + ((r < half) ? (2 * r) : (2 * (r - half) + 1));
                }
                dst[static_cast<size_t>(o)] = src[od];
            }
            return dst;
        }

        /* The parameter tensors of a decoder_transformer_config network in the order of
           import_gguf_weights(): the output head, the final norm, then for each block from
           the last [ffn_down, ffn_up, ffn_gate, ffn_norm, attention, attn_norm], and the
           embedding table last. */
        template <typename net_type>
        std::vector<tensor*> decoder_parameter_tensors(net_type& net, const model_spec& spec)
        {
            std::vector<tensor*> params;
            visit_learnable_parameters(net, [&params](tensor& t) { if (t.size() > 0) params.push_back(&t); });
            const size_t expected = 3 + 6 * static_cast<size_t>(spec.n_layers);
            if (params.size() != expected)
                throw std::runtime_error("export_gguf_weights: the network holds " + std::to_string(params.size())
                    + " parameter tensors where a decoder of " + std::to_string(spec.n_layers)
                    + " blocks holds " + std::to_string(expected) + "; run a forward pass first");
            return params;
        }
    }

    /* The layout of the parameters of a decoder_transformer_config network, with each
       packed attention tensor split into its query, key, value and output matrices, as
       consolidation and spectral operations need them. */
    template <typename net_type>
    parameter_layout decoder_parameter_layout(net_type& net, const model_spec& spec)
    {
        parameter_layout layout = get_parameter_layout(net);
        const long d = spec.d_model, hd = spec.head_dim > 0 ? spec.head_dim : spec.d_model / spec.n_heads;
        const long qd = spec.n_heads * hd, kv = spec.n_kv_heads * hd;
        if (layout.num_blocks() != 3 + 6 * static_cast<size_t>(spec.n_layers))
            throw std::runtime_error("decoder_parameter_layout: the network does not match the specification");
        /* A split moves the blocks after the one it splits: the attention tensors are split
           from the last in the layout, block 0 of the decoder, to the first. */
        for (long b = 0; b < spec.n_layers; ++b)
        {
            const size_t at = 2 + 6 * static_cast<size_t>(spec.n_layers - 1 - b) + 4;
            if (layout.block(at).size < 2 * static_cast<size_t>(d) * (qd + kv))
                throw std::runtime_error("decoder_parameter_layout: unexpected attention parameters");
            layout.split_block(at, { { d, qd }, { d, kv }, { d, kv }, { qd, d } });
        }
        return layout;
    }

    /* Writes the weights of a decoder_transformer_config network into a container under the
       architecture of spec, with the metadata of that architecture: the inverse of
       import_gguf_weights(). Matrices are encoded in matrix_type (a matrix whose rows cannot
       hold whole Q8_0 blocks falls back to F16), vectors in F32. The tokenizer and the chat
       template are left to the caller, who copies them from where the vocabulary came from. */
    template <typename net_type>
    void export_gguf_weights(net_type& net, const model_spec& spec, gguf_writer& w,
        ggml_type matrix_type = ggml_type::F32, const gguf_load_options& opt = gguf_load_options())
    {
        using namespace gguf_load_impl;
        std::vector<tensor*> params = decoder_parameter_tensors(net, spec);
        const long d = spec.d_model;
        const long hd = spec.head_dim > 0 ? spec.head_dim : spec.d_model / spec.n_heads;
        const long qd = spec.n_heads * hd, kv = spec.n_kv_heads * hd, ff = spec.d_ffn;
        const long V = spec.vocab_size, L = spec.n_layers;
        const bool neox = opt.rope_permute || spec.arch_name == "qwen2" || spec.arch_name == "qwen3";
        const std::string a = spec.arch_name;

        w.set_string("general.architecture", a);
        if (!w.has("general.name"))
            w.set_string("general.name", spec.model_name.empty() ? a : spec.model_name);
        w.set_uint32(a + ".block_count", static_cast<uint32_t>(L));
        w.set_uint32(a + ".embedding_length", static_cast<uint32_t>(d));
        w.set_uint32(a + ".feed_forward_length", static_cast<uint32_t>(ff));
        w.set_uint32(a + ".attention.head_count", static_cast<uint32_t>(spec.n_heads));
        w.set_uint32(a + ".attention.head_count_kv", static_cast<uint32_t>(spec.n_kv_heads));
        w.set_float32(a + ".attention.layer_norm_rms_epsilon", static_cast<float>(spec.rms_eps));
        w.set_float32(a + ".rope.freq_base", static_cast<float>(spec.rope_freq_base));
        w.set_uint32(a + ".vocab_size", static_cast<uint32_t>(V));
        if (!w.has(a + ".context_length"))
            w.set_uint32(a + ".context_length", 2048);
        if (hd * spec.n_heads != d)
        {
            w.set_uint32(a + ".attention.key_length", static_cast<uint32_t>(hd));
            w.set_uint32(a + ".attention.value_length", static_cast<uint32_t>(hd));
        }
        const uint32_t file_type = matrix_type == ggml_type::F16 ? 1 : matrix_type == ggml_type::Q8_0 ? 7
            : matrix_type == ggml_type::BF16 ? 32 : 0;
        w.set_uint32("general.file_type", file_type);

        auto put_matrix = [&](const std::string& name, const float* src, long out_dim, long in_dim, bool permute)
        {
            std::vector<float> rows;
            transpose_from(src, out_dim, in_dim, rows, hd, permute);
            const ggml_type t = (matrix_type == ggml_type::Q8_0 && in_dim % 32 != 0) ? ggml_type::F16 : matrix_type;
            w.add_tensor(name, { static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim) }, rows.data(), t);
        };
        auto put_vector = [&](const std::string& name, const float* src, long n)
        {
            w.add_tensor(name, { static_cast<uint64_t>(n) }, src, ggml_type::F32);
        };

        size_t k = 0;
        const tensor& head = *params[k++];
        if (head.size() != static_cast<size_t>(d) * V)
            throw std::runtime_error("export_gguf_weights: unexpected output head");
        const tensor& out_norm = *params[k++];
        const tensor& embd = *params.back();
        if (embd.size() != static_cast<size_t>(V) * d || out_norm.size() != static_cast<size_t>(d))
            throw std::runtime_error("export_gguf_weights: unexpected embedding table or final norm");

        /* The embedding table is [vocab, d_model] in both layouts: the rows of the container
           are the rows of the table. */
        w.add_tensor("token_embd.weight", { static_cast<uint64_t>(d), static_cast<uint64_t>(V) }, embd.host(),
            (matrix_type == ggml_type::Q8_0 && d % 32 != 0) ? ggml_type::F16 : matrix_type);
        put_vector("output_norm.weight", out_norm.host(), d);
        // The compiled decoder trains an output head of its own: the container always carries it.
        put_matrix("output.weight", head.host(), V, d, false);

        for (long b = L - 1; b >= 0; --b)
        {
            const std::string p = "blk." + std::to_string(b) + ".";
            const tensor& down = *params[k++];
            const tensor& up = *params[k++];
            const tensor& gate = *params[k++];
            const tensor& ffn_norm = *params[k++];
            const tensor& attn = *params[k++];
            const tensor& attn_norm = *params[k++];
            if (down.size() != static_cast<size_t>(ff) * d || up.size() != static_cast<size_t>(d) * ff
                || gate.size() != static_cast<size_t>(d) * ff)
                throw std::runtime_error("export_gguf_weights: unexpected feed-forward tensors in " + p);
            put_matrix(p + (opt.swap_gate_up ? "ffn_gate" : "ffn_up") + ".weight", up.host(), ff, d, false);
            put_matrix(p + (opt.swap_gate_up ? "ffn_up" : "ffn_gate") + ".weight", gate.host(), ff, d, false);
            put_matrix(p + "ffn_down.weight", down.host(), d, ff, false);
            put_vector(p + "ffn_norm.weight", ffn_norm.host(), d);
            put_vector(p + "attn_norm.weight", attn_norm.host(), d);

            const size_t nq = static_cast<size_t>(d) * qd, nkv = static_cast<size_t>(d) * kv;
            const size_t ng = spec.qk_norm ? static_cast<size_t>(hd) : 0;
            if (attn.size() < 2 * nq + 2 * nkv + 2 * ng)
                throw std::runtime_error("export_gguf_weights: unexpected attention tensor in " + p);
            const float* f = attn.host();
            put_matrix(p + "attn_q.weight", f, qd, d, neox);
            put_matrix(p + "attn_k.weight", f + nq, kv, d, neox);
            put_matrix(p + "attn_v.weight", f + nq + nkv, kv, d, false);
            put_matrix(p + "attn_output.weight", f + nq + 2 * nkv, d, qd, false);
            if (spec.qk_norm)
            {
                const float* gq = f + 2 * nq + 2 * nkv;
                std::vector<float> vq = unpermute_vector(gq, hd, hd, neox);
                std::vector<float> vk = unpermute_vector(gq + ng, hd, hd, neox);
                put_vector(p + "attn_q_norm.weight", vq.data(), hd);
                put_vector(p + "attn_k_norm.weight", vk.data(), hd);
            }
            const size_t nb = static_cast<size_t>(qd) + 2 * static_cast<size_t>(kv);
            if (attn.size() == 2 * nq + 2 * nkv + 2 * ng + nb)
            {
                const float* bb = f + 2 * nq + 2 * nkv + 2 * ng;
                std::vector<float> bq = unpermute_vector(bb, qd, hd, neox);
                std::vector<float> bk = unpermute_vector(bb + qd, kv, hd, neox);
                put_vector(p + "attn_q.bias", bq.data(), qd);
                put_vector(p + "attn_k.bias", bk.data(), kv);
                put_vector(p + "attn_v.bias", bb + qd + kv, kv);
            }
        }
    }
}

#endif // DLIB_GGUF_WEIGHT_LOADER_H_
