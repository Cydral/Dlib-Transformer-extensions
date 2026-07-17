// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GGUF_DEQUANTIZE_ABSTRACT_H_
#ifdef DLIB_GGUF_DEQUANTIZE_ABSTRACT_H_

#include <cstdint>
#include <vector>
#include "gguf_reader_abstract.h"

namespace dlib
{
    /*!
        OVERVIEW
            This header converts the raw on-disk bytes of a GGUF tensor (as returned
            by gguf_reader::read_tensor_raw) into a flat float buffer, in the tensor's
            own logical orientation (ggml dims[0] contiguous). It performs no matrix
            transposition, no RoPE head permutation and no layer-specific packing;
            those belong to the repacking stage (gguf_weight_loader.h).

        SUPPORTED TENSOR TYPES
            - F32 and F16,
            - the legacy 32-element blocks: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
            - the 256-element k-quant super-blocks: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
            - IQ4_NL (4-bit over a fixed non-linear value table), which converters use
              as the fallback when a tensor row is not a multiple of the k-quant
              super-block size.
            The _S/_M/_L quantization mixes (e.g. Q4_K_M) are file-level recipes, not
            tensor types: every tensor in such a file carries one of the types above,
            so they require no extra code. The grid-based i-quants (IQ1/IQ2/IQ3 and
            IQ4_XS) are not covered and cause an exception.

        NUMERICAL FIDELITY
            The fp16 conversion handles signed zero, subnormals, infinities and NaN.
            Correct subnormal handling matters in practice: k-quant super-block scales
            commonly fall in the fp16 subnormal range for low-magnitude blocks, and an
            off-by-one in the subnormal exponent silently halves the affected weights.
            The implementation is validated against the gguf-py reference.
    !*/

    // ---------------------------------------------------------------------------------

    float gguf_half_to_float(
        uint16_t h
    );
    /*!
        ensures
            - Interprets h as an IEEE 754 half-precision (binary16) value and returns
              the exactly corresponding single-precision float.
            - Signed zero, subnormal values, infinities and NaN payloads are all
              converted faithfully (every binary16 value is representable in binary32,
              so the conversion is exact).
    !*/

    float gguf_rd_half(
        const uint8_t* p
    );
    /*!
        requires
            - p points to at least 2 readable bytes
        ensures
            - Reads a little-endian binary16 value at p and returns
              gguf_half_to_float() of it. p does not need to be 2-byte aligned.
    !*/

    // ---------------------------------------------------------------------------------

    /*!
        BLOCK DEQUANTIZATION KERNELS
            The functions
                gguf_dq_q4_0,  gguf_dq_q4_1,  gguf_dq_q5_0, gguf_dq_q5_1,
                gguf_dq_q8_0,  gguf_dq_q8_1,  gguf_dq_iq4_nl,
                gguf_dq_q2_K,  gguf_dq_q3_K, gguf_dq_q4_K, gguf_dq_q5_K,
                gguf_dq_q6_K,  gguf_dq_q8_K
            all share the signature
                void (const uint8_t* p, float* y)
            and decode exactly one quantization block: p points to the raw block bytes
            and y receives the decoded values (32 floats for the legacy and IQ4_NL
            formats, 256 floats for the k-quant super-blocks). They are the per-block
            workers used by gguf_dq_blocks() and are exposed mainly for testing
            against reference implementations; typical code should call
            gguf_dequantize() instead.
    !*/

    void gguf_get_scale_min_k4(
        int j,
        const uint8_t* q,
        uint8_t& d,
        uint8_t& m
    );
    /*!
        requires
            - 0 <= j < 8
            - q points to the 12-byte packed scale/min array of a k-quant super-block
        ensures
            - #d == the 6-bit sub-block scale at index j
            - #m == the 6-bit sub-block minimum at index j
    !*/

    // ---------------------------------------------------------------------------------

    void gguf_dq_blocks(
        const gguf_tensor_info& t,
        const std::vector<uint8_t>& raw,
        std::vector<float>& out,
        uint64_t qk,
        uint64_t bytes,
        void (*fn)(const uint8_t*, float*)
    );
    /*!
        requires
            - fn decodes one block of qk elements from bytes raw bytes
            - out.size() >= t.n_elements()
        ensures
            - Decodes all t.n_elements()/qk blocks of raw into out, invoking fn once
              per block.
        throws
            - std::runtime_error if t.n_elements() is not a multiple of qk or if raw
              is too small to hold all blocks.
    !*/

    void gguf_dequantize(
        const gguf_tensor_info& t,
        const std::vector<uint8_t>& raw,
        std::vector<float>& out
    );
    /*!
        requires
            - raw holds the on-disk bytes of the tensor described by t (as produced
              by gguf_reader::read_tensor_raw)
        ensures
            - #out.size() == t.n_elements()
            - #out holds the dequantized tensor values as floats, in the tensor's
              on-disk element order (ggml dims[0] contiguous).
        throws
            - std::runtime_error if t.type is not one of the supported types, or if
              raw is smaller than the tensor requires.
    !*/

    void gguf_read_dequantized(
        gguf_reader& g,
        const gguf_tensor_info& t,
        std::vector<float>& out
    );
    /*!
        requires
            - t was obtained from g
        ensures
            - Reads the raw tensor bytes from g and dequantizes them, i.e. performs
              g.read_tensor_raw() followed by gguf_dequantize() into out.
        throws
            - std::runtime_error on read failure or unsupported tensor type.
    !*/

    // ---------------------------------------------------------------------------------

}

#endif // DLIB_GGUF_DEQUANTIZE_ABSTRACT_H_
