// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GGUF_MODEL_SPEC_ABSTRACT_H_
#ifdef DLIB_GGUF_MODEL_SPEC_ABSTRACT_H_

#include <string>
#include <vector>
#include "gguf_reader_abstract.h"

namespace dlib
{
    /*!
        OVERVIEW
            This header is the detection and header-generation stage of the GGUF
            import pipeline. From the metadata of an opened GGUF file it builds a
            neutral model_spec (architecture family, dimensions, RoPE and
            normalization settings, structural options), checks that description
            against the capabilities of the available Dlib transformer layers, and
            can emit a ready-to-include Dlib model header. Weight dequantization
            and repacking (gguf_weight_loader.h) and tokenizer extraction are
            separate stages.

        TYPICAL USAGE
            gguf_reader g("model.gguf");
            model_spec spec = detect_model(g);
            std::cout << describe(spec);

            compat_result cr = check_compatibility(spec);
            if (!cr.ok)
                report_blockers_and_stop(cr);

            emit_header(spec, model_identifier(spec) + ".h");
            // then: import_gguf_weights(net, g, spec);   (see gguf_weight_loader.h)
    !*/

    // ---------------------------------------------------------------------------------

    enum class arch_family
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The known transformer architecture families a GGUF file may declare
                through its general.architecture metadata key.

            VALUES
                llama    - Llama 1/2/3 (many Mistral GGUFs also declare "llama")
                mistral  - Mistral models declaring the "mistral" architecture
                gemma    - Gemma first generation
                gemma2   - Gemma second generation
                qwen2    - Qwen2 family (QKV biases, tied embeddings on small sizes)
                qwen3    - Qwen3 family (per-head QK-Norm, decoupled head_dim)
                mixtral  - Mixture-of-experts Mistral variants
                unknown  - Any architecture string not listed above
        !*/
    };

    arch_family arch_family_from_name(
        const std::string& a
    );
    /*!
        ensures
            - Returns the arch_family matching the given general.architecture string,
              or arch_family::unknown if the string is not recognized.
    !*/

    // ---------------------------------------------------------------------------------

    struct model_spec
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A neutral, framework-independent description of a decoder-only
                transformer model, extracted from GGUF metadata by detect_model().
                It carries everything the compatibility check, the header generator
                and the weight loader need to know about the model.

            FIELDS
                arch_name           - Raw general.architecture string
                model_name          - Human-readable identity (cleaned general.name)
                family              - Parsed architecture family
                n_layers            - Number of transformer blocks
                n_heads             - Number of attention (query) heads
                n_kv_heads          - Number of key/value heads (GQA); equals n_heads
                                      for classical multi-head attention
                d_model             - Embedding (hidden) dimension
                d_ffn               - Feed-forward hidden dimension
                head_dim            - Per-head dimension; usually d_model / n_heads
                                      but decoupled on some families (e.g. Qwen3)
                vocab_size          - Vocabulary size (from the embedded tokenizer
                                      when present, else from the metadata)
                ffn_num, ffn_den    - Reduced fraction such that
                                      d_ffn == d_model * ffn_num / ffn_den; used by
                                      the generated header to express the FFN ratio
                                      as template parameters
                rms_eps             - RMSNorm epsilon declared by the model
                rope_freq_base      - RoPE frequency base (theta)
                rope_scaling_type   - "", "none", "linear" or "yarn"
                rope_scaling_factor - Position scaling factor when declared
                rope_orig_ctx       - Original context length when declared
                n_experts           - Expert count; > 0 means mixture of experts
                n_experts_used      - Experts active per token
                tied_embeddings     - True when there is no separate output.weight
                                      tensor (the lm_head shares the embedding table)
                quantized           - True when at least one tensor is neither F32
                                      nor F16
                qk_norm             - True when the model applies per-head RMSNorm
                                      to Q and K (detected from the presence of
                                      blk.0.attn_q_norm.weight)
        !*/

        std::string arch_name;
        std::string model_name;
        arch_family family;
        long n_layers, n_heads, n_kv_heads;
        long d_model, d_ffn, head_dim, vocab_size;
        long ffn_num, ffn_den;
        double rms_eps, rope_freq_base;
        long n_experts, n_experts_used;
        std::string rope_scaling_type;
        double rope_scaling_factor;
        long rope_orig_ctx;
        bool tied_embeddings;
        bool quantized;
        bool qk_norm;
    };

    // ---------------------------------------------------------------------------------

    long gcd_long(
        long a,
        long b
    );
    /*!
        ensures
            - Returns the greatest common divisor of a and b, as a non-negative value.
            - Helper used to reduce the d_ffn / d_model ratio; kept public for reuse.
    !*/

    std::string clean_model_name(
        const std::string& name
    );
    /*!
        ensures
            - Returns the model name with a redundant organization prefix removed.
            - GGUF files converted from model hubs often carry general.name in the
              "<organization>_<repository>" form; when the repository itself starts
              with the organization name (case-insensitively), only the repository
              part is kept, otherwise name is returned unchanged.
            - Example: "tinyllama_tinyllama-1.1b-chat-v1.0" becomes
              "tinyllama-1.1b-chat-v1.0".
    !*/

    model_spec detect_model(
        const gguf_reader& g
    );
    /*!
        ensures
            - Reads the architecture metadata of g and returns the corresponding
              model_spec, with every field documented above populated.
            - Missing optional metadata falls back to conventional defaults
              (rms_eps == 1e-5, rope_freq_base == 10000, n_kv_heads == n_heads,
              head_dim == d_model / n_heads).
            - The legacy rope.scale_linear key is recognized and mapped onto
              rope_scaling_type == "linear" when the modern rope.scaling.* keys are
              absent.
            - Structural flags (tied_embeddings, qk_norm, quantized) are detected
              from the tensor table, not from the metadata.
    !*/

    // ---------------------------------------------------------------------------------

    std::string sanitize_identifier(
        const std::string& s
    );
    /*!
        ensures
            - Returns s lowered into a valid C++ identifier: alphanumerics are kept
              (upper case is lowered), every other run of characters collapses into a
              single underscore, leading/trailing underscores are trimmed.
            - Returns "model" if nothing remains, and prefixes "m_" if the result
              would start with a digit.
            - Example: "TinyLlama-1.1B-Chat-v1.0" becomes "tinyllama_1_1b_chat_v1_0".
    !*/

    std::string model_identifier(
        const model_spec& s
    );
    /*!
        ensures
            - Returns the canonical identifier of the model: the sanitized model
              name, falling back to the sanitized architecture name when the model
              name is empty.
            - This identifier derives the generated namespace, the include guard and
              the default output file names, so successive imports do not collide.
    !*/

    std::string describe(
        const model_spec& s
    );
    /*!
        ensures
            - Returns a multi-line, human-readable summary of the model
              specification, suitable for console display.
    !*/

    // ---------------------------------------------------------------------------------

    struct compat_result
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The outcome of a compatibility check between a model specification
                and the available Dlib transformer layers.

            FIELDS
                ok       - True when there is no blocker
                blockers - Conditions that prevent a faithful import; the import
                           tool must stop when any is present
                notes    - Informational remarks that do not prevent the import
                           (e.g. quantization formats handled by the dequantizer)
        !*/

        bool ok;
        std::vector<std::string> blockers;
        std::vector<std::string> notes;
    };

    compat_result check_compatibility(
        const model_spec& s,
        bool fused_attention_path = true
    );
    /*!
        ensures
            - Checks s against the capabilities of the current Dlib transformer
              layers and returns the blockers and notes.
            - #compat_result::ok == true if and only if no blocker was found.
            - Current blockers include: the Gemma families (require (1+w) RMSNorm,
              embedding scaling and GeGLU), mixture-of-experts models (routing
              convention not aligned yet), declared linear RoPE scaling, odd or
              missing head_dim, n_heads not a multiple of n_kv_heads, and incomplete
              dimension metadata.
            - fused_attention_path distinguishes the two consumers of the check: the
              statically generated template network (true) and the shape-dynamic
              runtime engine (false). Both currently share the same rules; the
              parameter is kept so consumer-specific constraints (such as bias
              handling in the fused attention packing) can diverge without an API
              change.
    !*/

    // ---------------------------------------------------------------------------------

    void emit_header(
        const model_spec& s,
        const std::string& path,
        const std::string& ns = ""
    );
    /*!
        ensures
            - Writes a self-contained Dlib model header for s at the given path.
            - The generated header defines the model constants (vocabulary, layers,
              heads, dimensions, FFN ratio), a decoder_transformer_config alias, and
              a network_type<is_training> alias covering both the inference and the
              fine-tuning networks; it also lists the GGUF tensor names the weight
              loader consumes.
            - The namespace and the include guard are derived from
              model_identifier(s) unless ns is non-empty, so several generated
              headers can coexist in one program.
            - The first generated header included in a translation unit claims the
              generic `imported_model` namespace alias, so single-model programs
              compile unchanged whatever the model-specific namespace is.
        throws
            - std::runtime_error if the file cannot be created or fully written.
    !*/

    // ---------------------------------------------------------------------------------

}

#endif // DLIB_GGUF_MODEL_SPEC_ABSTRACT_H_
