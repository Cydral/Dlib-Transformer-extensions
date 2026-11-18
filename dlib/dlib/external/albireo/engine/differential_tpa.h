// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_DIFFERENTIAL_TPA_H_
#define ALBIREO_ENGINE_DIFFERENTIAL_TPA_H_

namespace albireo
{

    /*!
        Two forms of attention a decoder may put in place of the queries, keys and values of
        one matrix each, and of the single map of a head.  The first changes where the rows
        of an attention come from, the second what its heads read and what becomes of their
        outputs.  These functions compute what is proper to each form; the scores, the
        softmax and the weighted sums of the values stay those of an attention.

        TENSOR PRODUCT ATTENTION (Zhang et al., "Tensor Product Attention Is All You Need", 2025)
            The queries of a token are a sum of rank outer products of a factor over the heads
            and a factor over the dimensions of a head, both products of the input of the layer
            with small matrices, and so are its keys and its values, with one rank for the
            queries and one for the keys and the values:

                row[h] = 1 / rank * sum over r of a[h][r] * b[r]

            with a of n_heads * rank values, head by head, and b of rank * head_dim values, rank
            by rank, as the module CPLinear of the reference lays them out.  The reference
            turns the factors b of the queries and of the keys by the rotary positions, which
            turns each row of the sum, the rotation being linear.  Its rotation pairs dimension
            i with i + head_dim / 2 and gives (x1 cos + x2 sin, x2 cos - x1 sin), the angle
            opposite to that of the rotary positions of the engine: the runtime of a model has
            to follow the rotation its weights were trained with.  These functions rotate
            nothing.

        DIFFERENTIAL ATTENTION (Ye et al., "Differential Transformer", 2025)
            A head subtracts two softmax maps, each with queries and keys of its own, before
            they weigh the values:

                out = norm(first - lambda * second) * (1 - lambda_init)

            where first and second are the outputs of the two maps over values of 2 * head_dim
            floats, norm divides by the root mean square over these floats and scales by a
            weight per dimension, lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init, and
            lambda_init = 0.8 - 0.6 exp(-0.3 depth) for the layer of index depth.

            A layer of n_heads heads over n_kv key/value heads thus has 2 * n_heads query
            sub-heads and 2 * n_kv key sub-heads of head_dim floats, and n_kv value heads of
            2 * head_dim floats; the maps 2h and 2h + 1 of head h read the values of the head
            h / (n_heads / n_kv).  The keys of a map differ between the sources of the
            reference when n_kv is below n_heads: map m reads the key sub-head
            m / (n_heads / n_kv) in multihead_diffattn.py, and map 2h + s the key sub-head
            2 * (h / (n_heads / n_kv)) + s in multihead_flashdiff_1.py, the two being the same
            with as many key heads as heads.  The attention of a model follows the source its
            weights were trained with; the functions below take the outputs of the maps and
            do not depend on it.
    !*/

    /*!
        requires
            - rank >= 1
            - a holds n_heads * rank floats, b rank * head_dim floats, out n_heads * head_dim floats.
        ensures
            - out[h * head_dim + j] = 1 / rank * sum over r of a[h * rank + r] * b[r * head_dim + j]
    !*/
    void tpa_combine(const float* a, const float* b, long n_heads, long rank, long head_dim, float* out);

    // tpa_combine() for n tokens, the rows of each array being a_stride, b_stride and out_stride floats apart.
    void tpa_combine_rows(const float* a, long a_stride, const float* b, long b_stride, long n, long n_heads,
        long rank, long head_dim, float* out, long out_stride);

    // 0.8 - 0.6 exp(-0.3 depth), the lambda a layer starts from; depth is 0 for the first layer.
    float differential_lambda_init(long depth);

    // exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init, over vectors of n floats.
    float differential_lambda(const float* lq1, const float* lk1, const float* lq2, const float* lk2, long n,
        float lambda_init);

    /*!
        requires
            - width >= 1, eps > 0
            - maps holds, for each of n_heads heads, the width outputs of its first map then
              those of its second one; out holds n_heads * width floats and may be maps.
            - weight holds width floats, or is null for a normalization without weights.
        ensures
            - For each head, out = (first - lambda * second) / sqrt(mean of its squares + eps)
              * weight * scale.
    !*/
    void differential_combine(const float* maps, long n_heads, long width, float lambda, const float* weight,
        float eps, float scale, float* out);

    /*!
        requires
            - width >= 1, eps > 0
            - x holds n_heads rows of width floats; weight holds width floats, or is null.
        ensures
            - Each row is divided by the root mean square of its values (eps added to their mean
              square) and scaled by weight: the normalization of the outputs of each head.
    !*/
    void head_rms_norm(float* x, long n_heads, long width, const float* weight, float eps);

}

#endif // ALBIREO_ENGINE_DIFFERENTIAL_TPA_H_
