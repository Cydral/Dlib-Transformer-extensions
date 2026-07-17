// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Catalog of ready-to-instantiate decoder model structures.
//
// Each alias below is a complete, named network structure matching a published
// open-weight model exactly: layer count, head geometry, embedding width, feed-forward
// ratio and vocabulary. They are usable on their own, independently of the GGUF import
// machinery, for training from scratch, fine-tuning, or as the typed counterpart of an
// imported model:
//
//     using net_type = tinyllama_1_1b::network_type<false>;
//     net_type net;
//
// Two run-time settings complete each structure, because they are values rather than
// shapes: the RMSNorm epsilon (norm_eps_setter visitor, or set_eps on the layers) and
// the rotary frequency base (attention_config_setter). The documented per-model values
// are listed with each alias; the GGUF import path applies them automatically from the
// container metadata.
//
// Compilation cost scales with the layer count (every block is a distinct type); the
// runtime engine (runtime_transformer.h) is the shape-dynamic alternative when only
// inference is needed.

#ifndef DLIB_DECODER_MODELS_H_
#define DLIB_DECODER_MODELS_H_

#include "decoder_models_abstract.h"

#include "decoder_transformer_config.h"

namespace dlib
{
    /* TinyLlama 1.1B (chat v1.0 and derivatives).
       RMSNorm eps 1e-5, RoPE theta 10000. Derivative fine-tunes sometimes extend the
       vocabulary (e.g. 32003); substitute the first parameter accordingly. */
    using tinyllama_1_1b = decoder_transformer_config<
        /*vocab*/32000, /*layers*/22, /*heads*/32, /*kv_heads*/4,
        /*d_model*/2048, /*ffn*/11, 4>;

    /* SmolLM2 1.7B (instruct). Plain multi-head attention (kv == heads), tied
       embeddings on import. RMSNorm eps 1e-5, RoPE theta 130000. */
    using smollm2_1_7b = decoder_transformer_config<
        /*vocab*/49152, /*layers*/24, /*heads*/32, /*kv_heads*/32,
        /*d_model*/2048, /*ffn*/4, 1>;

    /* Llama-2 7B. RMSNorm eps 1e-5, RoPE theta 10000. */
    using llama2_7b = decoder_transformer_config<
        /*vocab*/32000, /*layers*/32, /*heads*/32, /*kv_heads*/32,
        /*d_model*/4096, /*ffn*/43, 16>;

    /* Mistral 7B v0.1/v0.2 geometry. RMSNorm eps 1e-5, RoPE theta 10000 (v0.1)
       or 1e6 (v0.2+). */
    using mistral_7b = decoder_transformer_config<
        /*vocab*/32000, /*layers*/32, /*heads*/32, /*kv_heads*/8,
        /*d_model*/4096, /*ffn*/7, 2>;

    /* Llama-3 8B. RMSNorm eps 1e-5, RoPE theta 500000. */
    using llama3_8b = decoder_transformer_config<
        /*vocab*/128256, /*layers*/32, /*heads*/32, /*kv_heads*/8,
        /*d_model*/4096, /*ffn*/7, 2>;

    /* Qwen2 0.5B (instruct). RMSNorm eps 1e-6, RoPE theta 1e6, tied embeddings.
       This family carries QKV projection biases: the attention layer's bias slots
       receive them on GGUF import (enabled automatically from the tensor table);
       standalone use enables them via set_qkv_bias_enabled or the attention
       configuration visitor. */
    using qwen2_0_5b = decoder_transformer_config<
        /*vocab*/151936, /*layers*/24, /*heads*/14, /*kv_heads*/2,
        /*d_model*/896, /*ffn*/38, 7>;

    /* Qwen3 0.6B. RMSNorm eps 1e-6, RoPE theta 1e6. Explicit head_dim of 128
       (larger than d_model / heads) and per-head QK normalization. */
    using qwen3_0_6b = decoder_transformer_config<
        /*vocab*/151936, /*layers*/28, /*heads*/16, /*kv_heads*/8,
        /*d_model*/1024, /*ffn*/3, 1, /*head_dim*/128, /*qk_norm*/true>;
}

#endif // DLIB_DECODER_MODELS_H_
