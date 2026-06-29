// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF -> Dlib model specification.
//
// Reads the architecture metadata of a GGUF file into a neutral model_spec, checks
// compatibility against the available Dlib layers, and emits a Dlib model header.
// This is the detection and header-generation stage of the import pipeline; weight
// dequantization/repacking and tokenizer extraction are separate stages.

#ifndef DLIB_GGUF_MODEL_SPEC_H_
#define DLIB_GGUF_MODEL_SPEC_H_

#include "gguf_reader.h"
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>

namespace dlib
{
    enum class arch_family { llama, mistral, gemma, gemma2, qwen2, qwen3, mixtral, unknown };

    inline arch_family arch_family_from_name(const std::string& a)
    {
        if (a == "llama")   return arch_family::llama;     // Llama 1/2/3 (and Mistral in many GGUFs)
        if (a == "mistral") return arch_family::mistral;
        if (a == "gemma")   return arch_family::gemma;
        if (a == "gemma2")  return arch_family::gemma2;
        if (a == "qwen2")   return arch_family::qwen2;
        if (a == "qwen3")   return arch_family::qwen3;
        return arch_family::unknown;
    }

    // Neutral description of a transformer model, independent of any framework.
    struct model_spec
    {
        std::string arch_name;
        arch_family family = arch_family::unknown;
        long n_layers = 0, n_heads = 0, n_kv_heads = 0;
        long d_model = 0, d_ffn = 0, head_dim = 0, vocab_size = 0;
        long ffn_num = 0, ffn_den = 1;          // d_ffn = d_model * ffn_num / ffn_den (reduced)
        double rms_eps = 1e-5, rope_freq_base = 10000.0;
        long n_experts = 0, n_experts_used = 0; // > 0 => mixture of experts
        bool tied_embeddings = false;           // no separate output.weight tensor
        bool quantized = false;                 // at least one tensor is neither F32 nor F16
        bool qk_norm = false;                   // per-head RMSNorm on Q and K (Qwen3)
    };

    inline long gcd_long(long a, long b)
    {
        while (b) { long t = a % b; a = b; b = t; }
        return a < 0 ? -a : a;
    }

    inline model_spec detect_model(const gguf_reader& g)
    {
        model_spec s;
        s.arch_name = g.get_str("general.architecture", "unknown");
        s.family = arch_family_from_name(s.arch_name);

        const std::string p = s.arch_name;   // metadata keys are prefixed by the architecture
        s.n_layers   = static_cast<long>(g.get_int(p + ".block_count"));
        s.n_heads    = static_cast<long>(g.get_int(p + ".attention.head_count"));
        s.n_kv_heads = static_cast<long>(g.get_int(p + ".attention.head_count_kv", s.n_heads));
        s.d_model    = static_cast<long>(g.get_int(p + ".embedding_length"));
        s.d_ffn      = static_cast<long>(g.get_int(p + ".feed_forward_length"));
        /* Qwen3 (and some others) decouple head_dim from d_model / n_heads; prefer the explicit
           attention.key_length metadata when present, else fall back to the derived value. */
        {
            const long kl = static_cast<long>(g.get_int(p + ".attention.key_length", 0));
            s.head_dim = kl > 0 ? kl : (s.n_heads > 0 ? s.d_model / s.n_heads : 0);
        }
        s.rms_eps    = g.get_double(p + ".attention.layer_norm_rms_epsilon", 1e-5);
        s.rope_freq_base = g.get_double(p + ".rope.freq_base", 10000.0);
        s.n_experts  = static_cast<long>(g.get_int(p + ".expert_count", 0));
        s.n_experts_used = static_cast<long>(g.get_int(p + ".expert_used_count", 0));

        if (g.has("tokenizer.ggml.tokens"))
            s.vocab_size = static_cast<long>(g.at("tokenizer.ggml.tokens").arr_str.size());
        else
            s.vocab_size = static_cast<long>(g.get_int(p + ".vocab_size", 0));

        if (s.d_model > 0 && s.d_ffn > 0)
        {
            const long gg = gcd_long(s.d_ffn, s.d_model);
            s.ffn_num = s.d_ffn / gg;
            s.ffn_den = s.d_model / gg;
        }

        s.tied_embeddings = (g.find_tensor("output.weight") == nullptr);
        s.qk_norm = (g.find_tensor("blk.0.attn_q_norm.weight") != nullptr);
        for (const auto& t : g.tensors()) if (t.is_quantized()) { s.quantized = true; break; }
        return s;
    }

    inline std::string describe(const model_spec& s)
    {
        std::ostringstream o;
        o << "Architecture       : " << s.arch_name << "\n"
          << "Layers             : " << s.n_layers << "\n"
          << "Attention heads    : " << s.n_heads << "\n"
          << "KV heads           : " << s.n_kv_heads
          << " (GQA repeat " << (s.n_kv_heads ? s.n_heads / s.n_kv_heads : 0) << "x)\n"
          << "Embedding dim      : " << s.d_model << "\n"
          << "Head dim           : " << s.head_dim << "\n"
          << "FFN hidden         : " << s.d_ffn
          << " (= d_model * " << s.ffn_num << " / " << s.ffn_den << ")\n"
          << "Vocab size         : " << s.vocab_size << "\n"
          << "RMSNorm epsilon    : " << s.rms_eps << "\n"
          << "RoPE freq base     : " << s.rope_freq_base << "\n"
          << "Experts            : " << s.n_experts
          << (s.n_experts ? " (used " + std::to_string(s.n_experts_used) + ")" : "") << "\n"
          << "QK-Norm            : " << (s.qk_norm ? "yes" : "no") << "\n"
          << "Tied embeddings    : " << (s.tied_embeddings ? "yes" : "no") << "\n"
          << "Quantized weights  : " << (s.quantized ? "yes" : "no") << "\n";
        return o.str();
    }

    struct compat_result
    {
        bool ok = true;
        std::vector<std::string> blockers;
        std::vector<std::string> notes;
    };

    inline compat_result check_compatibility(const model_spec& s)
    {
        compat_result r;
        if (s.family == arch_family::gemma || s.family == arch_family::gemma2)
            r.blockers.push_back("Gemma family needs (1+w) RMSNorm, embedding scaling and GeGLU, not available yet");
        if (s.n_experts > 0)
            r.blockers.push_back("MoE routing convention must be aligned with moe_ before enabling import");
        if (s.head_dim <= 0 || (s.head_dim % 2) != 0)
            r.blockers.push_back("head_dim must be even for RoPE");
        if (s.n_kv_heads <= 0 || (s.n_heads % s.n_kv_heads) != 0)
            r.blockers.push_back("n_heads must be a multiple of n_kv_heads");
        if (s.d_model <= 0 || s.n_layers <= 0 || s.vocab_size <= 0)
            r.blockers.push_back("incomplete metadata (missing dimensions)");
        if (s.quantized)
            r.notes.push_back("Weights are quantized; the converter dequantizes legacy (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) and k-quant (Q2_K..Q6_K) formats; i-quants (IQ*) are not supported");
        r.ok = r.blockers.empty();
        return r;
    }

    // Emit a Dlib model header. The network uses the unified GQA stack with ACT disabled
    // and the exact FFN ratio, so it depends on the FFN-ratio parameters added to
    // gqa_transformer_unified::transformer_stack.
    inline void emit_header(const model_spec& s, const std::string& path,
        const std::string& ns = "imported_model")
    {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("emit_header: cannot write " + path);

        out << "// Generated by slm_gguf_import. Source architecture: " << s.arch_name << ".\n"
            << "// Edit with care; regenerate from the source model when the spec changes.\n"
            << "#ifndef IMPORTED_MODEL_H_\n#define IMPORTED_MODEL_H_\n\n"
            << "#include <dlib/dnn/decoder_transformer_config.h>\n\n"
            << "namespace " << ns << "\n{\n"
            << "    static constexpr long VOCAB_SIZE    = " << s.vocab_size << ";\n"
            << "    static constexpr long NUM_LAYERS    = " << s.n_layers << ";\n"
            << "    static constexpr long NUM_HEADS     = " << s.n_heads << ";\n"
            << "    static constexpr long NUM_KV_HEADS  = " << s.n_kv_heads << ";\n"
            << "    static constexpr long EMBEDDING_DIM = " << s.d_model << ";\n"
            << "    static constexpr long HEAD_DIM      = " << s.head_dim << ";\n"
            << "    static constexpr bool USE_QK_NORM   = " << (s.qk_norm ? "true" : "false") << ";\n"
            << "    static constexpr long FFN_NUM       = " << s.ffn_num
            << ";  // hidden = EMBEDDING_DIM * FFN_NUM / FFN_DEN = " << s.d_ffn << "\n"
            << "    static constexpr long FFN_DEN       = " << s.ffn_den << ";\n\n"
            << "    using config = dlib::decoder_transformer_config<\n"
            << "        VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, EMBEDDING_DIM, FFN_NUM, FFN_DEN,\n"
            << "        HEAD_DIM, USE_QK_NORM>;\n\n"
            << "    // Strict model definition. network_type<false> is the inference network\n"
            << "    // (generation / chatbot); network_type<true> is the training network used\n"
            << "    // for fine-tuning. The imported weights are loaded into this exact type.\n"
            << "    template <bool is_training>\n"
            << "    using network_type = config::network_type<is_training>;\n\n"
            << "    // GGUF tensor names consumed by the weight-repacking stage:\n"
            << "    //   token_embd.weight, output_norm.weight"
            << (s.tied_embeddings ? " (output tied to token_embd)" : ", output.weight") << "\n"
            << "    //   blk.{i}.attn_norm.weight\n"
            << "    //   blk.{i}.attn_q.weight, blk.{i}.attn_k.weight, blk.{i}.attn_v.weight, blk.{i}.attn_output.weight\n"
            << (s.qk_norm ? "    //   blk.{i}.attn_q_norm.weight, blk.{i}.attn_k_norm.weight\n" : "")
            << "    //   blk.{i}.ffn_norm.weight\n"
            << "    //   blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight, blk.{i}.ffn_down.weight\n"
            << "}\n\n#endif // IMPORTED_MODEL_H_\n";
    }
}

#endif // DLIB_GGUF_MODEL_SPEC_H_
