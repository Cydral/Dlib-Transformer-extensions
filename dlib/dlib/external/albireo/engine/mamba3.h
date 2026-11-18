// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_MAMBA3_H_
#define ALBIREO_ENGINE_MAMBA3_H_

#include <cstddef>
#include <vector>

#include "qkernels.h"

namespace albireo
{

    struct mamba3_geometry
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The sizes and the options of one Mamba-3 layer (Lahoti et al., "Mamba-3:
                Improved Sequence Modeling using State Space Principles", 2026), with the
                names of its reference module, mamba_ssm/modules/mamba3.py:

                  d_inner       channels of the layer, expand * d_model
                  head_dim      channels of a head (headdim); n_heads = d_inner / head_dim
                  d_state       width of the projections B and C, and of the state of a channel
                  groups        heads of B and C (ngroups): the n_heads / groups heads of a
                                group, which follow one another, share its B and its C
                  rank          B and C a token brings (mimo_rank); 1 without MIMO
                  n_angles      pairs of B and C the layer rotates, d_state * rope_fraction / 2
                  mimo          the multi-input multi-output layer, whose channels take a
                                weight per B (mimo_x), per gate (mimo_z) and per output (mimo_o)
                  pairwise      the rotation pairs dimension 2i with 2i + 1; otherwise i with
                                i + d_state / 2.  The reference pairs them the first way in its
                                single-input single-output layer and the second way with MIMO.
                                Its inference kernel splits a state whose width is no power of
                                two at half the next power of two; this layer splits it at
                                d_state / 2, as its training kernel and its reference function do
                  norm_before_gate
                                the outputs of each head are divided by their root mean square
                                and scaled before the gate (is_outproj_norm)

                The row a token brings to the layer is the product of its input with in_proj,
                projection_width() floats in this order: the gates z (d_inner), the inputs x
                (d_inner), B then C (rank * groups * d_state each, rank by rank and group by
                group), then per head the step dt, the decay A and the trapezoid weight, then
                the n_angles angles.

                A layer has at most mamba3_max_channels channels, a state of mamba3_max_state
                floats per channel and mamba3_max_rank ranks, and its state, its rotated B and
                the B of a token hold at most mamba3_max_floats floats each: check_mamba3()
                refuses the sizes beyond, which a file could give.
        !*/
        long d_inner = 0;
        long head_dim = 0;
        long n_heads = 0;
        long d_state = 0;
        long groups = 1;
        long rank = 1;
        long n_angles = 0;
        bool mimo = false;
        bool pairwise = true;
        bool norm_before_gate = false;
        float a_floor = 1e-4f;
        float norm_eps = 1e-5f;

        long bc_width() const { return d_state * groups * rank; }
        long projection_width() const { return 2 * d_inner + 2 * bc_width() + 3 * n_heads + n_angles; }
    };

    const long mamba3_max_channels = 1L << 18;
    const long mamba3_max_state = 1L << 12;
    const long mamba3_max_rank = 64;
    const long long mamba3_max_floats = 1LL << 28;

    /*!
        ensures
            - Returns the geometry of the layer the reference module builds from these
              arguments: d_inner = expand * d_model, n_angles from rope_fraction (0.5 or 1),
              the rank 1 and the pairs (2i, 2i + 1) without MIMO, the pairs (i, i + d_state / 2)
              with it.
        throws
            - std::runtime_error, with the reason, when the sizes do not make a layer.
    !*/
    mamba3_geometry mamba3_layer(long d_model, double expand, long head_dim, long d_state, long groups,
        double rope_fraction, bool mimo, long rank, bool norm_before_gate);

    // Throws std::runtime_error, with the reason, when g is not the geometry of a layer.
    void check_mamba3(const mamba3_geometry& g);

    struct mamba3_parameters
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The parameters of a layer beside its two matrices, in_proj and out_proj, with
                the layout of the reference module:

                  dt_bias               n_heads
                  b_bias, c_bias        n_heads * rank * d_state, head by head then rank by rank
                  b_norm, c_norm        d_state, the weights of the normalizations of B and C
                  d                     n_heads, the weight of the input in the output
                  mimo_x, mimo_z, mimo_o
                                        n_heads * rank * head_dim, head by head then rank by
                                        rank; empty without MIMO
                  out_norm              d_inner, the weights of the normalization of the
                                        outputs; empty unless norm_before_gate
        !*/
        std::vector<float> dt_bias, b_bias, c_bias, b_norm, c_norm, d;
        std::vector<float> mimo_x, mimo_z, mimo_o;
        std::vector<float> out_norm;
    };

    struct mamba3_state
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What a layer keeps of the tokens of one sequence, whatever their number:

                  angle     n_heads * n_angles, the angles of the rotation, within [0, 2 pi)
                  ssm       n_heads * head_dim * d_state, the state of the recurrence
                  k         rank * n_heads * d_state, the rotated B of the last token
                  v         n_heads * head_dim, the inputs x of the last token

                A copy of a state is a snapshot of the sequence at its last token: a sequence
                goes back to an earlier token by taking the state copied there, and two
                sequences that start with the same tokens start from copies of one state.
        !*/
        std::vector<float> angle, ssm, k, v;

        size_t bytes() const { return sizeof(float) * (angle.size() + ssm.size() + k.size() + v.size()); }
    };

    struct mamba3_work
    {
        // The buffers of one thread of a layer.
        aligned_floats b, c, k, q, y, gate, cos, sin;
        std::vector<double> turn;
    };

    class mamba3_mixer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The recurrence of a Mamba-3 layer between its two matrices: forward() takes
                the rows in_proj gives for consecutive tokens of one sequence and writes the
                rows out_proj takes, while the state of the sequence follows.

                For a token, with the notations of the reference:

                    A = -f(A_row), at most -a_floor, f(x) = 1 + x for x >= 0, 1 / (1 - x) below
                    dt = softplus(dt_row + dt_bias),  trap = sigmoid(trap_row)
                    alpha = exp(A dt),  beta = (1 - trap) dt alpha,  gamma = trap dt
                    B, C = their rows divided by their root mean square over d_state, times
                           b_norm and c_norm, plus b_bias and c_bias
                    angle += tanh(angle_row) pi dt, and B and C are rotated by the angles
                    h = alpha h + beta (B' x') + gamma (B x), with B' and x' the rotated B and
                        the inputs of the previous token, each product being the outer product
                        of the inputs of a head, times mimo_x, with B
                    y = h C + d x (x times mimo_x), then y silu(z) (z times mimo_z), after the
                        normalization of y over the channels of each head when the geometry
                        asks for it, and the sum of the ranks weighted by mimo_o

                The line of h is the exponential-trapezoidal discretization of the paper, which
                adds the term of the previous token to the update of Mamba-2; the rotation of B
                and C by the summed angles is its complex-valued state.

                The heads are independent: threads may run forward() at once on disjoint
                ranges of heads of one state, each with a mamba3_work of its own.
        !*/
    public:
        mamba3_mixer(const mamba3_geometry& g, const mamba3_parameters& p);

        const mamba3_geometry& geometry() const { return g_; }

        // The state of a sequence without tokens.
        mamba3_state initial_state() const;

        /*!
            requires
                - s is initial_state(), or the state of this layer after earlier tokens of
                  the sequence.
                - projections holds count rows of geometry().projection_width() floats.
                - outputs holds count rows of geometry().d_inner floats.
                - 0 <= head_begin <= head_end <= geometry().n_heads
            ensures
                - Reads the count tokens in their order for the heads of the range, writes
                  their channels in each row of outputs and leaves in s their state after the
                  last token.
            throws
                - std::runtime_error when s has the sizes of another layer, or the range heads
                  the layer does not have.
        !*/
        void forward(mamba3_state& s, const float* projections, long count, float* outputs, mamba3_work& work,
            long head_begin, long head_end) const;

        void forward(mamba3_state& s, const float* projections, long count, float* outputs, mamba3_work& work) const
        {
            forward(s, projections, count, outputs, work, 0, g_.n_heads);
        }

    private:
        mamba3_geometry g_;
        std::vector<float> dt_bias_, b_norm_, c_norm_, d_, out_norm_;
        // The biases and the weights of the ranks, rank by rank then head by head, as the loops read them.
        std::vector<float> b_bias_, c_bias_;
        aligned_floats mimo_x_, mimo_z_, mimo_o_;
    };

    /*!
        requires
            - row holds d_state floats, cos and sin n_angles floats.
            - 2 * n_angles <= d_state, and d_state is even when pairwise is false.
        ensures
            - Rotates the n_angles first pairs of row by their angles: the pair (a, b) becomes
              (a cos - b sin, a sin + b cos).  The pairs are (2i, 2i + 1) when pairwise is true,
              (i, i + d_state / 2) otherwise; the other values stay.
    !*/
    void mamba3_rotate(float* row, const float* cos, const float* sin, long d_state, long n_angles, bool pairwise);

}

#endif // ALBIREO_ENGINE_MAMBA3_H_
