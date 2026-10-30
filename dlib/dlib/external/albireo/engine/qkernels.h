// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_QKERNELS_H_
#define ALBIREO_ENGINE_QKERNELS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <dlib/data_io/gguf_reader.h>

#include "../common/system_info.h"

namespace albireo
{
    class thread_pool;

    /*!
        Products of quantized weight matrices with activations, on the processor.

        A weight matrix is used in the layout a GGUF file stores it: one row per output
        feature, each row a sequence of blocks covering the input features.  Its bytes are
        read where they lie, in the memory mapping of the file, and never expanded to
        floats.  The activations are quantized once per product to 8-bit blocks matching
        the block structure of the weights (32 values for the legacy formats, 256 values
        with partial sums for the k-quants), and every dot product is then computed in
        integers, block by block, with one floating multiplication per block.  This is the
        scheme that makes a 4-bit model run at the speed of memory: a token reads each
        weight once, in its compact form.

        Formats with a dedicated kernel: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
        IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K and Q6_K.  The AVX2 build carries vector versions
        of the formats that the published models use most (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K,
        F16 and F32); the others run their portable version, which the compiler still
        vectorizes in part.

        Products are distributed over the threads of a thread_pool by groups of output
        rows, so that each thread reads a disjoint part of the matrix.
    !*/

    enum class act_format { f32, q8_0, q8_k };

    struct weight_matrix
    {
        dlib::ggml_type type = dlib::ggml_type::F32;
        const uint8_t* data = nullptr;   // rows * row_bytes bytes
        long rows = 0;                   // output features
        long cols = 0;                   // input features
        size_t row_bytes = 0;

        bool valid() const { return data != nullptr && rows > 0 && cols > 0; }
    };

    // Whether the processor kernels support a tensor type, and the activation format they
    // need for it.
    bool qk_supported(dlib::ggml_type type);
    act_format qk_activation_format(dlib::ggml_type type);
    size_t qk_row_bytes(dlib::ggml_type type, long cols);

    // Bytes of one activation row of n values quantized to format f.
    size_t qk_activation_row_bytes(act_format f, long n);

    // Quantizes one row of n floats (n a multiple of the block size of f) into dst.
    void qk_quantize_row(act_format f, const float* src, void* dst, long n);

    // Dot product of weight row w (type t, n values) with an activation row prepared in
    // the matching format by qk_quantize_row(), or given as floats for F32, F16 and BF16.
    float qk_dot(dlib::ggml_type t, const uint8_t* w, const void* act, long n);

    // The portable version of qk_dot(), whatever the build: the reference against which
    // the vector kernels are checked.
    float qk_dot_portable(dlib::ggml_type t, const uint8_t* w, const void* act, long n);

    // Expands one weight row to floats (reference path and embedding lookup).
    void qk_dequantize_row(dlib::ggml_type t, const uint8_t* w, float* dst, long n);

    /*!
        Scratch memory a thread reuses between products, so that a token allocates nothing.
    !*/
    struct qk_scratch
    {
        std::vector<uint8_t> act;
        std::vector<float> row;
    };

    /*!
        From this number of tokens on, qk_matmul() expands the weights to floats, a tile of
        rows at a time, and multiplies them with the activations as they are: the expansion
        is then shared by enough tokens to cost little, and the products run at the full
        rate of the floating point units.  Below it, the activations are quantized and the
        integer kernels compute each dot product from the compact weights, which is what a
        generation step, bound by the reading of the weights, needs.
    !*/
    const long qk_float_product_tokens = 8;

    /*!
        ensures
            - Computes Y[t][o] = sum_i W[o][i] * X[t][i] for t < n_tokens and o < W.rows.
            - X holds n_tokens rows of W.cols floats, x_stride floats apart; Y receives
              n_tokens rows of W.rows floats, y_stride floats apart.
            - When accumulate is true, the products are added to Y instead of replacing it.
            - path selects the kernels.  The experts of a mixture receive a varying share
              of the tokens of a pass; they follow the size of the whole pass, so that a
              token is computed the same way whichever expert it reaches.
    !*/
    enum class qk_path
    {
        automatic,  // floats from qk_float_product_tokens tokens on, integers below
        integer,    // quantized activations and integer kernels
        floats      // weights expanded to floats
    };

    void qk_matmul(const weight_matrix& W, const float* X, long n_tokens, long x_stride,
        float* Y, long y_stride, thread_pool& pool, qk_scratch& scratch, bool accumulate = false,
        qk_path path = qk_path::automatic);

    // Name of the instruction set the kernels were compiled for.
    const char* qk_isa();

    /*!
        ensures
            - Returns the qk_isa_flags of this build, which the processor must have.  The
              function holds no instruction of these sets: a program checks them before it
              calls a kernel.
    !*/
    unsigned qk_required_isa();

    /*!
        ensures
            - Returns the first of the qk_isa_flags that the kernels of this build, plus the
              flags of extra, use and that cpu lacks, or 0 when cpu runs them.  Defined here,
              it is compiled with the instructions of the program that calls it, as are the
              two functions that follow.
    !*/
    inline unsigned qk_missing_isa(const cpu_info& cpu, unsigned extra = 0)
    {
        return qk_lacking_isa(cpu, qk_required_isa() | extra);
    }

    /*!
        ensures
            - When on is true, qk_matmul() expands every weight row to floats and computes
              single precision dot products with the activations as given, instead of
              quantizing the activations.  It is slow and only meant for validation: the
              results can then be compared with a floating point reference at the level of
              rounding errors.
    !*/
    void qk_set_exact_products(bool on);

    // Half and bfloat16 conversions used throughout the engine.
    float qk_half_to_float(uint16_t h);
    uint16_t qk_float_to_half(float f);
    void qk_half_to_float_row(const uint16_t* src, float* dst, long n);
    void qk_float_to_half_row(const float* src, uint16_t* dst, long n);

    // Vector helpers of the attention and normalization code.
    float qk_dot_f32(const float* a, const float* b, long n);
    float qk_dot_f32_f16(const float* a, const uint16_t* b, long n);
    void qk_axpy_f16(float alpha, const uint16_t* x, float* y, long n);   // y += alpha * x
    void qk_scale(float* x, float s, long n);
    float qk_max(const float* x, long n);
    // x[i] = exp(x[i] - m), returns the sum of the results.
    float qk_exp_shift_sum(float* x, float m, long n);

    /*!
        The attention by tiles of the processor runtime.  A tile holds rows of queries, the
        heads that share one key/value head for consecutive tokens of one sequence, and
        reads the cache block after block: every key and every value taken from memory
        serves all the rows of the tile.  K and V hold m rows of hd halves, Q and O rows of
        hd floats, and scratch qk_attention_scratch(hd) floats private to the caller.

        qk_attention_scores() ensures:
            - S[r * lds + j] == dot(Q + r * hd, K + j * hd), for r < rows and j < m.
        qk_attention_values() ensures:
            - O[r * hd + i] += sum over j < m of P[r * lds + j] * V[j * hd + i], for
              r < rows and i < hd.
    !*/
    long qk_attention_scratch(long hd);
    void qk_attention_scores(const float* Q, long rows, const uint16_t* K, long m, long hd, float* S, long lds,
        float* scratch);
    void qk_attention_values(const float* P, long lds, long rows, const uint16_t* V, long m, long hd, float* O,
        float* scratch);

}

#endif // ALBIREO_ENGINE_QKERNELS_H_
