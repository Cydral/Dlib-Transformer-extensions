// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "model_config.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <dlib/data_io/gguf_model_spec.h>

namespace albireo
{

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

        c.n_layers = static_cast<long>(g.get_int(a + ".block_count"));
        c.n_heads = static_cast<long>(g.get_int(a + ".attention.head_count"));
        c.n_kv_heads = static_cast<long>(g.get_int(a + ".attention.head_count_kv", c.n_heads));
        c.d_model = static_cast<long>(g.get_int(a + ".embedding_length"));
        c.d_ffn = static_cast<long>(g.get_int(a + ".feed_forward_length"));
        const long split = c.n_heads > 0 ? c.d_model / c.n_heads : 0;
        c.head_dim = static_cast<long>(g.get_int(a + ".attention.key_length", split));
        c.head_dim_v = static_cast<long>(g.get_int(a + ".attention.value_length", c.head_dim));
        c.ctx_train = static_cast<long>(g.get_int(a + ".context_length", 4096));
        c.rms_eps = g.get_double(a + ".attention.layer_norm_rms_epsilon", 1e-5);
        if (g.has("tokenizer.ggml.tokens"))
            c.vocab = static_cast<long>(g.at("tokenizer.ggml.tokens").arr_str.size());
        else
            c.vocab = static_cast<long>(g.get_int(a + ".vocab_size", 0));

        if (c.n_layers <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0 || c.d_model <= 0 || c.vocab <= 0)
            throw std::runtime_error("the model metadata is incomplete (missing dimensions)");
        if (c.n_heads % c.n_kv_heads != 0)
            throw std::runtime_error("the number of heads is not a multiple of the number of key/value heads");
        if (c.head_dim <= 0 || c.head_dim % 2 != 0 || c.head_dim_v != c.head_dim)
            throw std::runtime_error("unsupported head geometry");

        // Rotary embedding.
        c.rope = qwen_like || a == "qwen3moe" ? rope_style::neox : rope_style::norm;
        c.rope_dims = static_cast<long>(g.get_int(a + ".rope.dimension_count", c.head_dim));
        if (c.rope_dims <= 0 || c.rope_dims > c.head_dim || c.rope_dims % 2 != 0)
            throw std::runtime_error("unsupported rotary dimension count");
        c.rope_base = g.get_double(a + ".rope.freq_base", 10000.0);
        const std::string scaling = g.get_str(a + ".rope.scaling.type", "");
        double factor = g.get_double(a + ".rope.scaling.factor", 0.0);
        if (scaling.empty() && g.has(a + ".rope.scale_linear"))
            factor = g.get_double(a + ".rope.scale_linear", 0.0);
        if ((scaling == "linear" || (scaling.empty() && factor > 0.0)) && factor > 0.0)
            c.rope_freq_scale = 1.0 / factor;
        if (scaling == "yarn" && factor > 0.0)
        {
            c.yarn = true;
            c.rope_freq_scale = 1.0 / factor;
            c.yarn_orig_ctx = static_cast<double>(g.get_int(a + ".rope.scaling.original_context_length", c.ctx_train));
            c.yarn_beta_fast = g.get_double(a + ".rope.scaling.yarn_beta_fast", 32.0);
            c.yarn_beta_slow = g.get_double(a + ".rope.scaling.yarn_beta_slow", 1.0);
            c.yarn_ext_factor = 1.0;
            const double base_mscale = 1.0 + 0.1 * std::log(factor);
            if (g.has(a + ".rope.scaling.yarn_log_multiplier"))
            {
                // The magnitude correction of the reference implementation when both
                // mscale and mscale_all_dim are declared: their ratio.
                const double mul = g.get_double(a + ".rope.scaling.yarn_log_multiplier", 1.0);
                c.yarn_attn_factor = base_mscale / (1.0 + 0.1 * mul * std::log(factor));
            }
            else
            {
                c.yarn_attn_factor = base_mscale;
            }
        }
        f.optional_floats("rope_freqs.weight", static_cast<size_t>(c.rope_dims / 2), c.rope_freq_factors);

        c.attn_temp_scale = g.get_double(a + ".attention.temperature_scale", 0.0);
        if (c.attn_temp_scale != 0.0)
        {
            c.attn_temp_length = static_cast<double>(g.get_int(a + ".attention.temperature_length",
                c.yarn ? static_cast<int64_t>(c.yarn_orig_ctx) : 8192));
            if (c.attn_temp_length <= 0) c.attn_temp_length = 8192;
        }

        c.qkv_bias = f.has("blk.0.attn_q.bias");
        c.qk_norm = f.has("blk.0.attn_q_norm.weight");
        c.tied_embeddings = !f.has("output.weight");

        if (a == "granite" || a == "granitemoe")
        {
            c.embedding_scale = g.get_double(a + ".embedding_scale", 1.0);
            c.residual_scale = g.get_double(a + ".residual_scale", 1.0);
            c.attention_scale = g.get_double(a + ".attention.scale", 0.0);
            c.logit_scale = g.get_double(a + ".logit_scale", 1.0);
            if (c.logit_scale == 0.0) c.logit_scale = 1.0;
        }

        c.n_experts = static_cast<long>(g.get_int(a + ".expert_count", 0));
        c.n_experts_used = static_cast<long>(g.get_int(a + ".expert_used_count", 0));
        c.d_ffn_expert = static_cast<long>(g.get_int(a + ".expert_feed_forward_length", c.d_ffn));
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
        o << ")";
        return o.str();
    }

}
