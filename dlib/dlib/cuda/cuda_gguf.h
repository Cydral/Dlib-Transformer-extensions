// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_GGUF_H_
#define DLIB_CUDA_GGUF_H_

#include "tensor.h"
#include "kv_pages.h"

#ifdef DLIB_USE_CUDA

namespace dlib
{
    namespace cuda
    {
        /*
            The kernels of the operations of tensor_tools.h on weight matrices kept in their
            GGUF formats (gguf_blocks.h), of the row operations of the forward pass of a
            decoder, of its mixtures of experts, and of its keys and values in pages
            (kv_pages.h).  tensor_tools.h documents them.
        */

        void gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        );

        void gguf_matvec_group (
            tensor* const* dests,
            const tensor& src,
            const tensor* const* weights,
            int count,
            int type,
            const long* rows,
            long cols,
            size_t row_bytes
        );

        void gguf_matvec_swiglu (
            tensor& dest,
            tensor& work,
            const tensor& src,
            const tensor& gate_weights,
            const tensor& up_weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        );

        void gguf_dequantize (
            tensor& dest,
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes
        );

        void rms_norm_rows (
            tensor& dest,
            const tensor& src,
            const tensor& gamma,
            float eps
        );

        void add_rms_norm_rows (
            tensor& dest,
            tensor& x,
            const tensor& addend,
            float factor,
            const tensor& gamma,
            float eps
        );

        void silu_multiply (
            tensor& gate,
            const tensor& up
        );

        void rope_rows (
            tensor& rows,
            const tensor& angles,
            long heads,
            long head_dim,
            long rope_dims,
            bool neox
        );

        void rope_rows (
            tensor& first_rows,
            long first_heads,
            tensor& second_rows,
            long second_heads,
            const tensor& angles,
            long head_dim,
            long rope_dims,
            bool neox
        );

        void paged_kv_store (
            tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& keys,
            const tensor& values,
            const tensor& slots
        );

        void paged_kv_gather (
            tensor& dest,
            const tensor& pages,
            const kv_page_layout& layout,
            const tensor& slots
        );

        void paged_attention (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tokens,
            const tensor& tables,
            long span
        );

        void expert_routes (
            tensor& picks,
            tensor& weights,
            const tensor& logits,
            const tensor& bias,
            long top_k,
            bool normalize,
            float scale
        );

        void gguf_matvec_picked (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes,
            const tensor& picks,
            long per_row
        );

        void squared_relu (
            tensor& x
        );

        void mix_rows (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            long per_row
        );

        /*
            The kernels of the products that cuBLAS computes (cublas_dlibapi.h): gguf_gemm() and
            paged_attention_rows().  They read and write memory of the device, which the caller
            holds, and return once their kernel is launched.
        */

        // tile receives the rows [first, first + count) of the matrix as floats, cols values each.
        void gguf_expand_tile (
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes,
            float* tile
        );

        /*
            gguf_expand_tile() with the values rounded to half precision; *saturated becomes 1
            when a value lies beyond the halves or is not a number.
        */
        void gguf_expand_half_tile (
            const tensor& weights,
            int type,
            long first,
            long count,
            long cols,
            size_t row_bytes,
            uint16_t* tile,
            float* saturated
        );

        // dst[i] = src[i] rounded to half precision for i < n, *saturated as above.
        void floats_to_halves (
            const float* src,
            uint16_t* dst,
            size_t n,
            float* saturated
        );

        /*
            dst receives the keys then the values of layer at the positions [0, span) of a
            sequence whose pages table lists, as floats, head after head:
            2*layout.heads*span*layout.head_dim floats.  The keys of the 8-bit format stay
            rotated.
        */
        void paged_kv_expand (
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const int32_t* table,
            long span,
            float* dst
        );

        /*
            dst receives the rows of the heads of tokens tokens, head after head, rotated as the
            keys of the 8-bit format when rotate is true: q_heads*tokens rows of hd floats.
        */
        void attention_queries (
            const float* queries,
            long q_heads,
            long tokens,
            long hd,
            bool rotate,
            float* dst
        );

        /*
            Each of the rows of span scores becomes its softmax over the positions [0, limit),
            with zeros from limit on, limit == first_limit + r % tokens for row r.
        */
        void attention_softmax (
            float* scores,
            long span,
            long rows,
            long tokens,
            long first_limit
        );

        // dst[(t*heads + h)*hd + e] = src[(h*tokens + t)*hd + e].
        void rows_from_heads (
            const float* src,
            long heads,
            long tokens,
            long hd,
            float* dst
        );
    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_CUDA_GGUF_H_
