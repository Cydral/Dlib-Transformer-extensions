// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_RUNTIME_TRANSFORMER_ABSTRACT_H_
#ifdef DLIB_DNN_RUNTIME_TRANSFORMER_ABSTRACT_H_

#include <iosfwd>
#include <string>
#include <vector>
#include "../cuda/tensor_abstract.h"
#include "../data_io/gguf_reader_abstract.h"
#include "../data_io/gguf_model_spec_abstract.h"
#include "../data_io/gguf_weight_loader_abstract.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

    class runtime_transformer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A shape-dynamic, inference-only decoder-only transformer
                (Llama/Qwen family) whose architecture is resolved at load time
                from a model_spec. It is the runtime counterpart of the template
                network stack: where the template path fixes every architectural
                dimension in the C++ type (full training integration, one long
                compilation per model), this engine compiles once and loads any
                supported GGUF model dynamically.

                The numeric path mirrors the template attention layer exactly:
                Q scaled by 1/sqrt(head_dim) at projection, optional QKV biases,
                optional per-head QK RMSNorm, rotary encoding of Q and K, GQA
                repeat by channel copy, causal masking by additive per-head
                bias, plane-wise softmax, SwiGLU feed-forward (dense, or routed
                across experts for mixture-of-experts models), pre-norm
                residuals, final RMSNorm and lm_head projection. The global
                scale factors declared by some families (granitemoe: embedding,
                residual, attention and logit scales) are applied at their
                respective points. It runs on the
                same tensor primitives (gemm, rotary embedding, rms
                normalization, plane-wise softmax) as the template stack, on
                CPU or CUDA according to the build, and is validated by logit
                parity against external reference implementations.

            WEIGHT RESIDENCY
                Two storage modes are supported, selected before load():
                  - resident (default) : every matrix is dequantized once at
                    load time; fastest forward, memory footprint of the full
                    float32 model.
                  - quantized at rest : the raw GGUF blocks stay in memory and
                    each matrix is dequantized just in time into a small shared
                    rotating pool at every use, keeping the footprint close to
                    the on-disk file size, traded against one dequantization
                    per matrix and per forward.

            MIXTURE OF EXPERTS
                When spec().n_experts > 0, the feed-forward of each block routes
                every token through its top spec().n_experts_used experts:
                softmax over the router logits, top-k selection, renormalization
                of the selected weights to sum 1 (the mixtral / granitemoe
                convention), then the weighted sum of the selected experts'
                SwiGLU outputs. Tokens routed to the same expert are batched, so
                each active expert materializes its matrices once per forward;
                in quantized-at-rest mode only the experts actually selected are
                dequantized, which keeps the per-token cost proportional to the
                active parameters rather than the total parameters.

            KV CACHE AND GENERATION
                Generation runs in two phases: forward_prefill() processes the
                prompt and, when a context capacity is set, fills the KV cache;
                step() then extends the sequence one token at a time. Keys are
                cached before rotary encoding and encoded at read time on their
                compacted index, so the attention-sink eviction (keep the first
                keep_length rows, discard a block after them) keeps a coherent
                self-attention geometry without any shifting machinery, exactly
                like the template path.

            THREAD SAFETY
                A single generation thread is assumed. Distinct instances are
                independent.
        !*/

    public:

        void set_quantized_at_rest(bool enabled);
        bool quantized_at_rest() const;
        /*!
            ensures
                - Selects / reports the quantized-at-rest weight residency.
                - Must be set before load(): the flag decides, tensor by
                  tensor, whether load() stores raw quantized blocks or
                  dequantized floats. Changing it afterwards has no effect on
                  already loaded weights.
                - Tensors stored as F32/F16 in the container are always kept
                  resident; at-rest storage applies to the quantized formats.
        !*/

        void load(
            gguf_reader& g,
            const model_spec& spec,
            const gguf_load_options& opt
        );
        /*!
            requires
                - spec == detect_model(g), and check_compatibility(spec, false)
                  reported no blocker.
            ensures
                - #spec() == spec (the engine adopts the given specification).
                - Ingests every weight of g: per-block norms, attention and
                  feed-forward projections, optional QKV biases and QK-Norm
                  gammas (detected from the tensor table), final norm, lm_head
                  (the embedding table when spec.tied_embeddings is true) and
                  the token embedding table. For mixture-of-experts models the
                  feed-forward weights are the per-block router and the 3D
                  expert tensors, sliced into one matrix per expert (raw
                  quantized slices in at-rest mode).
                - Projection matrices are stored transposed to the [in, out]
                  orientation used by the forward path; the RoPE row permutation
                  is applied according to the architecture family and
                  opt.rope_permute, with the same conventions as
                  import_gguf_weights().
                - In quantized-at-rest mode the raw blocks are kept and the
                  embedding table is gathered row by row at lookup time;
                  otherwise everything is dequantized now.
            throws
                - std::runtime_error if a required tensor is missing or does
                  not match the dimensions announced by spec.
        !*/

        void set_context(
            long capacity,
            long keep_length = 0
        );
        /*!
            requires
                - capacity >= 0, keep_length >= 0
            ensures
                - Configures the KV cache for incremental generation: capacity
                  in tokens, and the number of leading positions (attention
                  sinks: typically BOS and the system block) pinned across
                  evictions.
                - Calls reset_cache(). Call before forward_prefill().
                - A capacity of 0 disables the cache: forward_prefill() is then
                  a pure full forward and step() is unavailable.
        !*/

        void reset_cache();
        /*!
            ensures
                - #cache_length() == 0
                - Reallocates the per-layer K/V caches at the configured
                  capacity (or releases them when the capacity is 0). Weights
                  are untouched.
        !*/

        long cache_length() const;
        /*!
            ensures
                - Returns the number of positions currently held in the KV
                  cache (identical for every layer).
        !*/

        const tensor& forward_prefill(
            const std::vector<int>& tokens
        );
        /*!
            requires
                - tokens is not empty and every id is in [0, spec().vocab_size)
                - If a context capacity is set, tokens.size() <= capacity.
            ensures
                - Runs the full forward over the token sequence and returns the
                  logits of every position as a [1, 1, N, vocab] tensor, valid
                  until the next forward call on this object.
                - When a context capacity is set, appends the pre-rotary keys
                  and the values of every layer to the KV cache and sets
                  #cache_length() == tokens.size().
        !*/

        const tensor& step(
            int token
        );
        /*!
            requires
                - forward_prefill() has been called since the last
                  reset_cache(), with a context capacity set.
                - token is in [0, spec().vocab_size)
            ensures
                - Runs the incremental forward for one new token against the
                  cached keys/values and returns its logits as a
                  [1, 1, 1, vocab] tensor, valid until the next forward call.
                - Appends the new position to the cache. When the cache is
                  full, first evicts a block of the oldest evictable positions
                  (preserving the keep_length attention sinks) to make room,
                  compacting the remaining rows; rotary positions follow the
                  compacted indices, so the attention geometry stays coherent.
                - #cache_length() grows by one (net of any eviction).
        !*/

        const model_spec& spec() const;
        /*!
            ensures
                - Returns the model specification the engine is running
                  (adopted by load(), or rebuilt from a runtime archive).
        !*/

        void save(std::ostream& out) const;
        /*!
            ensures
                - Writes a self-contained runtime archive: the model
                  specification and every weight in its current storage form
                  (quantized blocks at rest, or dequantized floats). Loading it
                  never depends on the template network types nor on their
                  serialization internals.
                - Format tag: "rtm_1" for dense models (byte-identical to the
                  previous format), "rtm_2" for mixture-of-experts models
                  (adds the global scale factors and the router and expert
                  weights).
        !*/

        void load(std::istream& in);
        /*!
            ensures
                - Restores an engine from a runtime archive written by save()
                  (accepting both the "rtm_1" and "rtm_2" formats):
                  specification, weights in their saved storage form, and an
                  empty KV cache (set_context() must be called again for
                  generation).
            throws
                - std::runtime_error if the stream does not start with the
                  "rtm_1" tag or is truncated.
        !*/
    };

    // ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_RUNTIME_TRANSFORMER_ABSTRACT_H_
