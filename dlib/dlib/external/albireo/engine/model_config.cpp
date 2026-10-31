// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "model_config.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <dlib/data_io/gguf_model_spec.h>

namespace albireo
{

    namespace
    {
        /*
            An integer of the metadata within [lo, hi], checked before any narrowing to long,
            which has 32 bits on Windows.  The bounds leave a wide margin over the published
            models; a value beyond them comes from a damaged file, whose sizes would overflow
            the products that size the memory of the runtime.
        */
        long bounded_int(const dlib::gguf_reader& g, const std::string& key, int64_t def, int64_t lo, int64_t hi)
        {
            const int64_t v = g.get_int(key, def);
            if (v < lo || v > hi)
                throw std::runtime_error("the model metadata holds an invalid " + key + " (" + std::to_string(v)
                    + ", expected " + std::to_string(lo) + " to " + std::to_string(hi) + ")");
            return static_cast<long>(v);
        }

        // A real number of the metadata, which must be finite.
        double finite_double(const dlib::gguf_reader& g, const std::string& key, double def)
        {
            const double v = g.get_double(key, def);
            if (!std::isfinite(v)) throw std::runtime_error("the model metadata holds an invalid " + key);
            return v;
        }

        const int64_t max_layers = 1024;
        const int64_t max_heads = 1024;
        const int64_t max_width = 1 << 17;
        const int64_t max_ffn = 1 << 20;
        const int64_t max_head_dim = 4096;
        const int64_t max_context = int64_t(1) << 26;
        const int64_t max_vocab = int64_t(1) << 24;
        const int64_t max_experts = 4096;
    }

    model_config read_model_config(const model_file& f)
    {
        const dlib::gguf_reader& g = f.reader();
        model_config c;
        c.arch = g.get_str("general.architecture", "");
        c.name = dlib::clean_model_name(g.get_str("general.name", c.arch));
        const std::string a = c.arch;

        const bool llama_like = a == "llama" || a == "mistral" || a == "mistral3" || a == "granite";
        const bool qwen_like = a == "qwen2" || a == "qwen3";
        const bool moe_arch = a == "mixtral" || a == "granitemoe" || a == "qwen3moe";
        if (!llama_like && !qwen_like && !moe_arch)
            throw std::runtime_error("the architecture '" + a + "' is not supported by this engine");

        if (!g.has(a + ".block_count") || !g.has(a + ".attention.head_count") || !g.has(a + ".embedding_length"))
            throw std::runtime_error("the model metadata is incomplete (missing dimensions)");
        c.n_layers = bounded_int(g, a + ".block_count", 0, 1, max_layers);
        // Every layer holds tensors of its own.
        if (static_cast<size_t>(c.n_layers) > g.tensors().size())
            throw std::runtime_error("the model declares more layers than it holds tensors");
        c.n_heads = bounded_int(g, a + ".attention.head_count", 0, 1, max_heads);
        c.n_kv_heads = bounded_int(g, a + ".attention.head_count_kv", c.n_heads, 1, c.n_heads);
        c.d_model = bounded_int(g, a + ".embedding_length", 0, 1, max_width);
        c.d_ffn = bounded_int(g, a + ".feed_forward_length", 0, 0, max_ffn);
        const long split = c.d_model / c.n_heads;
        c.head_dim = bounded_int(g, a + ".attention.key_length", split, 0, max_head_dim);
        c.head_dim_v = bounded_int(g, a + ".attention.value_length", c.head_dim, 0, max_head_dim);
        c.ctx_train = bounded_int(g, a + ".context_length", 4096, 0, max_context);
        c.rms_eps = finite_double(g, a + ".attention.layer_norm_rms_epsilon", 1e-5);
        if (g.has("tokenizer.ggml.tokens"))
        {
            const size_t n = g.at("tokenizer.ggml.tokens").arr_str.size();
            if (n > static_cast<size_t>(max_vocab))
                throw std::runtime_error("the vocabulary of the model is too large");
            c.vocab = static_cast<long>(n);
        }
        else
        {
            c.vocab = bounded_int(g, a + ".vocab_size", 0, 0, max_vocab);
        }

        if (c.vocab <= 0) throw std::runtime_error("the model metadata is incomplete (missing dimensions)");
        if (c.n_heads % c.n_kv_heads != 0)
            throw std::runtime_error("the number of heads is not a multiple of the number of key/value heads");
        if (c.head_dim <= 0 || c.head_dim % 2 != 0 || c.head_dim_v != c.head_dim)
            throw std::runtime_error("unsupported head geometry");
        if (c.rms_eps < 0) throw std::runtime_error("the model metadata holds a negative normalization epsilon");

        // Rotary embedding.
        c.rope = qwen_like || a == "qwen3moe" ? rope_style::neox : rope_style::norm;
        c.rope_dims = bounded_int(g, a + ".rope.dimension_count", c.head_dim, 0, max_head_dim);
        if (c.rope_dims <= 0 || c.rope_dims > c.head_dim || c.rope_dims % 2 != 0)
            throw std::runtime_error("unsupported rotary dimension count");
        c.rope_base = finite_double(g, a + ".rope.freq_base", 10000.0);
        if (c.rope_base <= 0) throw std::runtime_error("the model metadata holds an invalid rotary base");
        const std::string scaling = g.get_str(a + ".rope.scaling.type", "");
        double factor = finite_double(g, a + ".rope.scaling.factor", 0.0);
        if (scaling.empty() && g.has(a + ".rope.scale_linear"))
            factor = finite_double(g, a + ".rope.scale_linear", 0.0);
        if ((scaling == "linear" || (scaling.empty() && factor > 0.0)) && factor > 0.0)
            c.rope_freq_scale = 1.0 / factor;
        if (scaling == "yarn" && factor > 0.0)
        {
            c.yarn = true;
            c.rope_freq_scale = 1.0 / factor;
            c.yarn_orig_ctx = static_cast<double>(bounded_int(g, a + ".rope.scaling.original_context_length",
                c.ctx_train, 0, max_context));
            c.yarn_beta_fast = finite_double(g, a + ".rope.scaling.yarn_beta_fast", 32.0);
            c.yarn_beta_slow = finite_double(g, a + ".rope.scaling.yarn_beta_slow", 1.0);
            c.yarn_ext_factor = 1.0;
            const double base_mscale = 1.0 + 0.1 * std::log(factor);
            if (g.has(a + ".rope.scaling.yarn_log_multiplier"))
            {
                // The magnitude correction of the reference implementation when both
                // mscale and mscale_all_dim are declared: their ratio.
                const double mul = finite_double(g, a + ".rope.scaling.yarn_log_multiplier", 1.0);
                c.yarn_attn_factor = base_mscale / (1.0 + 0.1 * mul * std::log(factor));
            }
            else
            {
                c.yarn_attn_factor = base_mscale;
            }
        }
        f.optional_floats("rope_freqs.weight", static_cast<size_t>(c.rope_dims / 2), c.rope_freq_factors);

        c.attn_temp_scale = finite_double(g, a + ".attention.temperature_scale", 0.0);
        if (c.attn_temp_scale != 0.0)
        {
            c.attn_temp_length = static_cast<double>(bounded_int(g, a + ".attention.temperature_length",
                c.yarn ? static_cast<int64_t>(c.yarn_orig_ctx) : 8192, 0, max_context));
            if (c.attn_temp_length <= 0) c.attn_temp_length = 8192;
        }

        c.qkv_bias = f.has("blk.0.attn_q.bias");
        c.qk_norm = f.has("blk.0.attn_q_norm.weight");
        c.tied_embeddings = !f.has("output.weight");

        if (a == "granite" || a == "granitemoe")
        {
            c.embedding_scale = finite_double(g, a + ".embedding_scale", 1.0);
            c.residual_scale = finite_double(g, a + ".residual_scale", 1.0);
            c.attention_scale = finite_double(g, a + ".attention.scale", 0.0);
            c.logit_scale = finite_double(g, a + ".logit_scale", 1.0);
            if (c.logit_scale == 0.0) c.logit_scale = 1.0;
        }

        c.n_experts = bounded_int(g, a + ".expert_count", 0, 0, max_experts);
        c.n_experts_used = bounded_int(g, a + ".expert_used_count", 0, 0, max_experts);
        c.d_ffn_expert = bounded_int(g, a + ".expert_feed_forward_length", c.d_ffn, 0, max_ffn);
        if (c.n_experts > 0)
        {
            if (c.n_experts_used <= 0 || c.n_experts_used > c.n_experts)
                throw std::runtime_error("invalid number of experts used per token");
            if (f.has("blk.0.ffn_gate_shexp.weight") || f.has("blk.0.ffn_up_shexp.weight"))
                throw std::runtime_error("shared experts are not supported by this engine");
            if (!f.has("blk.0.ffn_gate_exps.weight"))
                throw std::runtime_error("this mixture of experts stores its experts in a legacy layout; "
                    "convert it again with a recent tool");
        }
        else if (moe_arch && a != "granitemoe")
        {
            throw std::runtime_error("the model declares a mixture of experts without experts");
        }

        c.chat_template = g.get_str("tokenizer.chat_template", "");
        c.tokenizer_pre = g.get_str("tokenizer.ggml.pre", "");
        const int64_t pooling = g.get_int(a + ".pooling_type", 0);
        if (pooling < 0 || pooling > 4) throw std::runtime_error("unknown pooling type");
        c.pooling = static_cast<int>(pooling);
        return c;
    }

    std::string describe(const model_config& c)
    {
        std::ostringstream o;
        o << c.name << " (" << c.arch << ", " << c.n_layers << " layers, width " << c.d_model
          << ", heads " << c.n_heads << "/" << c.n_kv_heads << " of " << c.head_dim
          << ", ffn " << c.d_ffn;
        if (c.is_moe()) o << ", " << c.n_experts << " experts top-" << c.n_experts_used;
        o << ", vocabulary " << c.vocab << ", trained context " << c.ctx_train
          << ", rope " << (c.rope == rope_style::neox ? "neox" : "norm") << " base " << c.rope_base;
        if (c.yarn) o << " yarn x" << 1.0 / c.rope_freq_scale;
        else if (c.rope_freq_scale != 1.0) o << " linear x" << 1.0 / c.rope_freq_scale;
        if (!c.rope_freq_factors.empty()) o << " with frequency factors";
        if (c.qkv_bias) o << ", qkv bias";
        if (c.qk_norm) o << ", qk norm";
        if (c.tied_embeddings) o << ", tied embeddings";
        if (c.is_embedding())
        {
            static const char* const pooling[] = { "none", "mean", "first token", "last token", "rank" };
            o << ", embedding model, pooling " << pooling[c.pooling];
        }
        o << ")";
        return o.str();
    }

}
