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
        std::string model_name;                 // human-readable identity (general.name)
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

    /* GGUF files converted from HuggingFace often carry general.name in the
       "<organization>_<repository>" form; when the repository itself starts with the
       organization name (e.g. "tinyllama_tinyllama-1.1b-chat-v1.0"), the prefix is
       redundant and only the repository part is kept. */
    inline std::string clean_model_name(const std::string& name)
    {
        const size_t p = name.find('_');
        if (p == std::string::npos || p == 0 || p + 1 >= name.size()) return name;
        auto lower = [](std::string v) {
            for (char& c : v) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            return v;
        };
        const std::string org = lower(name.substr(0, p));
        const std::string rest = name.substr(p + 1);
        if (rest.size() >= org.size() && lower(rest).compare(0, org.size(), org) == 0)
            return rest;
        return name;
    }

    inline model_spec detect_model(const gguf_reader& g)
    {
        model_spec s;
        s.arch_name = g.get_str("general.architecture", "unknown");
        s.model_name = clean_model_name(g.get_str("general.name", s.arch_name));
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

    /* Lower-case a model name into a valid C++ identifier: alphanumerics are kept,
       everything else collapses into single underscores. "TinyLlama-1.1B-Chat-v1.0"
       becomes "tinyllama_1_1b_chat_v1_0". */
    inline std::string sanitize_identifier(const std::string& s)
    {
        std::string id;
        for (char c : s)
        {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) id += c;
            else if (c >= 'A' && c <= 'Z') id += static_cast<char>(c - 'A' + 'a');
            else if (!id.empty() && id.back() != '_') id += '_';
        }
        while (!id.empty() && id.back() == '_') id.pop_back();
        if (id.empty()) id = "model";
        if (id[0] >= '0' && id[0] <= '9') id = "m_" + id;
        return id;
    }

    /* The canonical identifier of a model: sanitized general.name, falling back to the
       architecture name. Used to derive the generated namespace, the header guard and
       the default output file names, so successive imports do not collide. */
    inline std::string model_identifier(const model_spec& s)
    {
        return sanitize_identifier(s.model_name.empty() ? s.arch_name : s.model_name);
    }

    inline std::string describe(const model_spec& s)
    {
        std::ostringstream o;
        o << "Model name         : " << s.model_name << "\n"
          << "Architecture       : " << s.arch_name << "\n"
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
    // gqa_transformer_unified::transformer_stack. The namespace and the include guard are
    // derived from the model identity unless an explicit namespace is given, so several
    // generated headers can coexist in a program embedding a catalog of imported models.
    inline void emit_header(const model_spec& s, const std::string& path,
        const std::string& ns = "")
    {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("emit_header: cannot write " + path);

        const std::string id = ns.empty() ? model_identifier(s) : ns;
        std::string guard = "DLIB_IMPORTED_";
        for (char c : id)
            guard += (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
        guard += "_H_";

        out << "// Generated by slm_gguf_import. Source architecture: " << s.arch_name << ".\n"
            << "// Edit with care; regenerate from the source model when the spec changes.\n"
            << "#ifndef " << guard << "\n#define " << guard << "\n\n"
            << "#include <dlib/dnn/decoder_transformer_config.h>\n\n"
            << "namespace " << id << "\n{\n"
            << "    // Identity of the imported model, for display and catalog selection.\n"
            << "    static const char* const MODEL_NAME = \"" << s.model_name << "\";\n\n"
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
            << "}\n\n"
            << "/* The first imported-model header included in a translation unit claims the\n"
            << "   generic `imported_model` alias, so single-model programs compile unchanged\n"
            << "   whatever the model-specific namespace is. Programs embedding several models\n"
            << "   include several generated headers and use the explicit namespaces. */\n"
            << "#ifndef DLIB_IMPORTED_MODEL_ALIAS_DEFINED\n"
            << "#define DLIB_IMPORTED_MODEL_ALIAS_DEFINED\n"
            << "namespace imported_model = " << id << ";\n"
            << "#endif\n"
            << "\n#endif // " << guard << "\n";
    }
}

#endif // DLIB_GGUF_MODEL_SPEC_H_
