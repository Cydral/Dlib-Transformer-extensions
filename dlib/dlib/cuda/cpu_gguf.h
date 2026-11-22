// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CPU_GGUF_H_
#define DLIB_CPU_GGUF_H_

#include "tensor.h"
#include "kv_pages.h"

namespace dlib
{
    namespace cpu
    {
        /*
            The processor versions of the operations of tensor_tools.h on weight matrices kept
            in their GGUF formats (gguf_blocks.h), of the row operations of the forward pass of
            a decoder, of its mixtures of experts, and of its keys and values in pages
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

        void paged_attention_rows (
            tensor& out,
            const tensor& queries,
            long q_heads,
            float scale,
            const tensor& pages,
            const kv_page_layout& layout,
            long layer,
            const tensor& tables,
            long first_entry,
            long first_position
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
    }
}

#ifdef NO_MAKEFILE
#include "cpu_gguf.cpp"
#endif

#endif // DLIB_CPU_GGUF_H_
