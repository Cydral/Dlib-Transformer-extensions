#!/usr/bin/env python3
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""Writes the reference vectors of tests/test_mixers.cpp (tests/data/mixers.golden).

    python3 tools/make_mixer_golden.py [output file]

The vectors come from an implementation, in double precision with numpy, of three published
mechanisms, written after the reference functions of their authors:

  - the decoding step of a Mamba-3 layer, single-input single-output and multi-input multi-output
    (mamba3_siso_step_ref and mamba3_MIMO_step_ref of the tests of github.com/state-spaces/mamba,
    with the preparation of their inputs by the module mamba_ssm/modules/mamba3.py), and the two
    pairings of its rotation (apply_rotary_qk_inference_reference);
  - the projections of Tensor Product Attention (CPLinear of github.com/tensorgi/TPA, model/T6.py),
    without the rotary positions;
  - the differential attention of the Differential Transformer (github.com/microsoft/unilm,
    Diff-Transformer), up to its output projection, without the rotary positions and without the
    replacement of the scores that are no numbers.

Where these sources differ among themselves, the cases follow them as below:

  - the angles of a Mamba-3 layer are kept within one turn, as its step kernels keep them, where
    its reference functions keep their sum;
  - the layers with several inputs and outputs pair dimension i with i + d_state / 2, as the module
    and its training kernel do; the case mimo_pairwise pairs 2i with 2i + 1, as the step reference
    of the tests does;
  - head h reads the B and C of group h // (heads // groups), as the kernels do; the module itself
    runs one group or one group per head;
  - a state holds the inputs of the last token as the module keeps them, before their weights
    mimo_x, and its rotated B rank by rank then head by head;
  - with fewer key heads than heads, the case gqa repeats each key sub-head for the maps that follow
    one another (multihead_diffattn.py: map m reads the keys m // n_rep), and the case flash gives
    the two maps of a head the two key sub-heads of its group (multihead_flashdiff_1.py: map s of
    head h reads the keys 2 * (h // n_rep) + s).  Both read the values h // n_rep.

Each case holds its inputs, its parameters and its results, as arrays of single precision floats;
the results are computed from the inputs and the parameters as the file holds them.  The cases are
drawn from fixed seeds: the file comes out the same at each run of a version of numpy.  The program
needs numpy and nothing else.
"""

import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "..", "tests", "data", "mixers.golden")
TWO_PI = 2.0 * math.pi

tensors = {}


def put(name, value):
    tensors[name] = np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def single(value):
    """The values as the file holds them, in single precision."""
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def heavy_tail(x):
    """1 + x for x >= 0, 1 / (1 - x) below: the activation of the data-dependent A of Mamba-3."""
    return np.maximum(x, 0.0) + 1.0 / (1.0 - np.minimum(x, 0.0))


def rms_norm(x, weight, eps, group=None):
    """x / sqrt(mean(x^2) + eps) * weight, over the last axis or over its groups of group values."""
    if group is None:
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight
    xg = x.reshape(x.shape[:-1] + (-1, group))
    xg = xg / np.sqrt((xg * xg).mean(axis=-1, keepdims=True) + eps)
    return xg.reshape(x.shape) * weight


def rotate(t, cos, sin, pairwise):
    """Rotates the pairs of the last axis of t (rank, heads, d_state) by the angles of each head.

    The pairs are (2i, 2i + 1) when pairwise, (i, i + d_state / 2) otherwise; the dimensions
    beyond the angles are left as they are.
    """
    out = t.copy()
    n_angles = cos.shape[-1]
    if pairwise:
        a, b = t[..., 0:2 * n_angles:2], t[..., 1:2 * n_angles:2]
        out[..., 0:2 * n_angles:2] = a * cos - b * sin
        out[..., 1:2 * n_angles:2] = a * sin + b * cos
    else:
        half = t.shape[-1] // 2
        a, b = t[..., :n_angles], t[..., half:half + n_angles]
        out[..., :n_angles] = a * cos - b * sin
        out[..., half:half + n_angles] = a * sin + b * cos
    return out


def mamba3_case(name, seed, d_inner, head_dim, d_state, groups, rope_fraction, rank, mimo, norm, pairwise, length,
                scaled=False):
    """One layer over length tokens.

    scaled multiplies the rows of A, of the steps, of B and C and of the inputs, so that the floor
    of A, the linear part of the softplus and the eps of the normalizations decide the results.
    """
    rng = np.random.default_rng(seed)
    heads = d_inner // head_dim
    rotated = int(d_state * rope_fraction)
    rotated -= rotated % 2
    n_angles = rotated // 2
    r = rank if mimo else 1
    bc = d_state * groups * r
    width = 2 * d_inner + 2 * bc + 3 * heads + n_angles
    proj = rng.standard_normal((length, width))
    if scaled:
        proj[:, d_inner:2 * d_inner] *= 0.0005                                  # x
        proj[:, 2 * d_inner:2 * d_inner + 2 * bc] *= 0.002                      # B and C
        proj[:, 2 * d_inner + 2 * bc:2 * d_inner + 2 * bc + heads] *= 12.0      # steps
        proj[:, 2 * d_inner + 2 * bc + heads:2 * d_inner + 2 * bc + 2 * heads] *= 30000.0     # A
    proj = single(proj)
    dt_bias = single(0.5 * rng.standard_normal(heads) - 2.0)
    b_bias = single(1.0 + 0.3 * rng.standard_normal((heads, r, d_state)))
    c_bias = single(1.0 + 0.3 * rng.standard_normal((heads, r, d_state)))
    b_norm = single(1.0 + 0.2 * rng.standard_normal(d_state))
    c_norm = single(1.0 + 0.2 * rng.standard_normal(d_state))
    d_skip = single(1.0 + 0.3 * rng.standard_normal(heads))
    if mimo:
        mimo_x = single((1.0 + 0.3 * rng.standard_normal((heads, r, head_dim))) / r)
        mimo_z = single(1.0 + 0.3 * rng.standard_normal((heads, r, head_dim)))
        mimo_o = single((1.0 + 0.3 * rng.standard_normal((heads, r, head_dim))) / r)
    else:
        mimo_x = mimo_z = mimo_o = np.ones((heads, 1, head_dim))
    out_norm = single(1.0 + 0.2 * rng.standard_normal(d_inner))
    a_floor, eps = 1e-4, 1e-5

    angle = np.zeros((heads, n_angles))
    ssm = np.zeros((heads, head_dim, d_state))
    k_prev = np.zeros((r, heads, d_state))
    x_prev = np.zeros((heads, head_dim))
    vx = mimo_x.transpose(1, 0, 2)          # (rank, heads, head_dim)
    vz = mimo_z.transpose(1, 0, 2)
    vo = mimo_o.transpose(1, 0, 2)
    outputs = []
    for t in range(length):
        row = proj[t]
        at = 0
        z = row[at:at + d_inner].reshape(heads, head_dim); at += d_inner
        x = row[at:at + d_inner].reshape(heads, head_dim); at += d_inner
        b = row[at:at + bc].reshape(r, groups, d_state); at += bc
        c = row[at:at + bc].reshape(r, groups, d_state); at += bc
        dd_dt = row[at:at + heads]; at += heads
        dd_a = row[at:at + heads]; at += heads
        trap = sigmoid(row[at:at + heads]); at += heads
        angles = row[at:at + n_angles]

        a = np.minimum(-heavy_tail(dd_a), -a_floor)
        dt = softplus(dd_dt + dt_bias)
        b = rms_norm(b, b_norm, eps)
        c = rms_norm(c, c_norm, eps)
        # The heads of a group follow one another.
        k = np.repeat(b, heads // groups, axis=1) + b_bias.transpose(1, 0, 2)
        q = np.repeat(c, heads // groups, axis=1) + c_bias.transpose(1, 0, 2)
        angle = np.mod(angle + np.tanh(angles)[None, :] * dt[:, None] * math.pi, TWO_PI)
        cos, sin = np.cos(angle), np.sin(angle)
        k = rotate(k, cos, sin, pairwise)
        q = rotate(q, cos, sin, pairwise)

        alpha = np.exp(a * dt)
        beta = (1.0 - trap) * dt * alpha
        gamma = trap * dt
        v = x[None] * vx
        v_prev = x_prev[None] * vx
        ssm = (alpha[:, None, None] * ssm
               + beta[:, None, None] * np.einsum("rhn,rhp->hpn", k_prev, v_prev)
               + gamma[:, None, None] * np.einsum("rhn,rhp->hpn", k, v))
        y = np.einsum("hpn,rhn->rhp", ssm, q) + d_skip[None, :, None] * v
        gate = silu(z[None] * vz)
        if norm:
            y = y / np.sqrt((y * y).mean(axis=-1, keepdims=True) + eps) * out_norm.reshape(heads, head_dim)[None]
        y = y * gate
        outputs.append((y * vo).sum(axis=0).reshape(-1))
        k_prev, x_prev = k, x

    p = "m3." + name + "."
    put(p + "geometry", [d_inner, head_dim, d_state, groups, r, 1 if mimo else 0, n_angles, 1 if norm else 0,
                         1 if pairwise else 0, length])
    put(p + "proj", proj)
    put(p + "dt_bias", dt_bias)
    put(p + "b_bias", b_bias)
    put(p + "c_bias", c_bias)
    put(p + "b_norm", b_norm)
    put(p + "c_norm", c_norm)
    put(p + "d", d_skip)
    put(p + "mimo_x", mimo_x)
    put(p + "mimo_z", mimo_z)
    put(p + "mimo_o", mimo_o)
    put(p + "out_norm", out_norm)
    put(p + "y", np.stack(outputs))
    put(p + "angle", angle)
    put(p + "ssm", ssm)
    put(p + "k", k_prev)
    put(p + "v", x_prev)


def rotation_case():
    rng = np.random.default_rng(21)
    r, heads, d_state, n_angles = 2, 3, 16, 4
    q = single(rng.standard_normal((r, heads, d_state)))
    k = single(rng.standard_normal((r, heads, d_state)))
    angle = single(TWO_PI * rng.random((heads, n_angles)))
    cos, sin = np.cos(angle), np.sin(angle)
    put("rot.q", q)
    put("rot.k", k)
    put("rot.angle", angle)
    for label, pairwise in (("halved", False), ("pairwise", True)):
        put("rot." + label + ".q_out", rotate(q, cos, sin, pairwise))
        put("rot." + label + ".k_out", rotate(k, cos, sin, pairwise))


def tpa_case():
    rng = np.random.default_rng(31)
    n_in, heads, head_dim, rank, q_rank, tokens = 20, 4, 6, 2, 3, 5
    x = single(rng.standard_normal((tokens, n_in)))
    w = {
        "W_A_q": single(rng.standard_normal((heads * q_rank, n_in)) * 0.3),
        "W_A_k": single(rng.standard_normal((heads * rank, n_in)) * 0.3),
        "W_A_v": single(rng.standard_normal((heads * rank, n_in)) * 0.3),
        "W_B_q": single(rng.standard_normal((q_rank * head_dim, n_in)) * 0.3),
        "W_B_k": single(rng.standard_normal((rank * head_dim, n_in)) * 0.3),
        "W_B_v": single(rng.standard_normal((rank * head_dim, n_in)) * 0.3),
    }
    put("tpa.geometry", [n_in, heads, head_dim, rank, q_rank, tokens])
    put("tpa.x", x)
    for key, value in w.items():
        put("tpa." + key, value)
    for which, ranks in (("q", q_rank), ("k", rank), ("v", rank)):
        a = (x @ w["W_A_" + which].T).reshape(tokens, heads, ranks)
        b = (x @ w["W_B_" + which].T).reshape(tokens, ranks, head_dim)
        put("tpa." + which, np.einsum("thr,trd->thd", a, b) / ranks)


def diff_case(name, seed, embed, depth, heads, kv_heads, tokens, paired=False):
    """One differential attention; paired gives the two maps of a head the two key sub-heads of its group."""
    rng = np.random.default_rng(seed)
    kv_heads = heads if kv_heads is None else kv_heads
    n_rep = heads // kv_heads
    head_dim = embed // heads // 2
    x = single(rng.standard_normal((tokens, embed)))
    wq = single(rng.standard_normal((embed, embed)) * 0.25)
    wk = single(rng.standard_normal((embed // n_rep, embed)) * 0.25)
    wv = single(rng.standard_normal((embed // n_rep, embed)) * 0.25)
    lambdas = [single(0.1 * rng.standard_normal(head_dim)) for _ in range(4)]
    subln = single(1.0 + 0.2 * rng.standard_normal(2 * head_dim))
    lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

    q = (x @ wq.T).reshape(tokens, 2 * heads, head_dim).transpose(1, 0, 2)          # (2 heads, tokens, head_dim)
    k = (x @ wk.T).reshape(tokens, 2 * kv_heads, head_dim).transpose(1, 0, 2)
    v = (x @ wv.T).reshape(tokens, kv_heads, 2 * head_dim).transpose(1, 0, 2)
    if paired:
        k = k[[2 * ((m // 2) // n_rep) + m % 2 for m in range(2 * heads)]]
    else:
        k = np.repeat(k, n_rep, axis=0)
    v = np.repeat(v, n_rep, axis=0)
    scores = (q * head_dim ** -0.5) @ k.transpose(0, 2, 1)
    scores = scores + np.triu(np.full((tokens, tokens), -np.inf), 1)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    lam = math.exp(float(lambdas[0] @ lambdas[1])) - math.exp(float(lambdas[2] @ lambdas[3])) + lambda_init
    weights = weights.reshape(heads, 2, tokens, tokens)
    weights = weights[:, 0] - lam * weights[:, 1]
    attn = weights @ v
    attn = rms_norm(attn, subln, 1e-5) * (1.0 - lambda_init)
    y = attn.transpose(1, 0, 2).reshape(tokens, heads * 2 * head_dim)

    p = "diff." + name + "."
    put(p + "geometry", [embed, depth, heads, kv_heads, head_dim, tokens, 1 if paired else 0])
    put(p + "x", x)
    put(p + "q_proj", wq)
    put(p + "k_proj", wk)
    put(p + "v_proj", wv)
    for label, value in zip(("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"), lambdas):
        put(p + label, value)
    put(p + "subln", subln)
    put(p + "lambda_init", [lambda_init])
    put(p + "lambda", [lam])
    put(p + "y", y)


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    mamba3_case("siso_gate", 11, d_inner=48, head_dim=8, d_state=16, groups=1, rope_fraction=0.5, rank=1,
                mimo=False, norm=False, pairwise=True, length=14)
    mamba3_case("siso_norm", 12, d_inner=48, head_dim=8, d_state=16, groups=1, rope_fraction=1.0, rank=1,
                mimo=False, norm=True, pairwise=True, length=14)
    mamba3_case("siso_groups", 13, d_inner=48, head_dim=8, d_state=18, groups=2, rope_fraction=0.5, rank=1,
                mimo=False, norm=False, pairwise=True, length=11)
    mamba3_case("siso_wide", 14, d_inner=64, head_dim=32, d_state=64, groups=1, rope_fraction=0.5, rank=1,
                mimo=False, norm=False, pairwise=True, length=9)
    mamba3_case("mimo_gate", 15, d_inner=48, head_dim=8, d_state=16, groups=1, rope_fraction=0.5, rank=3,
                mimo=True, norm=False, pairwise=False, length=12)
    mamba3_case("mimo_norm", 16, d_inner=40, head_dim=10, d_state=20, groups=1, rope_fraction=1.0, rank=2,
                mimo=True, norm=True, pairwise=False, length=10)
    mamba3_case("mimo_pairwise", 17, d_inner=48, head_dim=8, d_state=16, groups=2, rope_fraction=0.5, rank=4,
                mimo=True, norm=False, pairwise=True, length=8)
    mamba3_case("siso_scaled", 18, d_inner=48, head_dim=8, d_state=16, groups=1, rope_fraction=0.5, rank=1,
                mimo=False, norm=True, pairwise=True, length=16, scaled=True)
    mamba3_case("mimo_scaled", 19, d_inner=48, head_dim=8, d_state=16, groups=1, rope_fraction=0.5, rank=2,
                mimo=True, norm=True, pairwise=False, length=16, scaled=True)
    rotation_case()
    tpa_case()
    diff_case("mha", 41, embed=24, depth=3, heads=3, kv_heads=None, tokens=7)
    diff_case("gqa", 42, embed=32, depth=0, heads=4, kv_heads=2, tokens=6)
    diff_case("flash", 43, embed=32, depth=5, heads=4, kv_heads=2, tokens=6, paired=True)

    with open(output, "wb") as f:
        f.write(b"ALBGOLD1")
        f.write(struct.pack("<I", len(tensors)))
        for name in sorted(tensors):
            t = tensors[name]
            raw = name.encode("ascii")
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<I", t.ndim))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            f.write(t.astype("<f4").tobytes())
    print("%d arrays, %d values, %s" % (len(tensors), sum(t.size for t in tensors.values()), os.path.normpath(output)))


if __name__ == "__main__":
    main()
