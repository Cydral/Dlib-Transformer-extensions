#!/usr/bin/env python3
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Reference forward pass of the decoder families served by the Albireo engine, written
with numpy from the published model definitions, to validate the C++ runtime.

The weights are read and dequantized by the gguf package of the GGUF ecosystem, so
neither the file parsing nor the quantized kernels of the engine are involved; the
products run in single precision on dequantized weights.  The script writes the logits
of every position of a token sequence as raw float32 rows.

    python3 reference_forward.py model.gguf out.bin id id id ...

Supported: llama, mistral, mistral3, qwen2, qwen3, qwen3moe, granite, granitemoe (and
llama files that carry experts, such as Mixtral).
"""

import math
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.contents()


class Weights:
    def __init__(self, reader):
        self.tensors = {t.name: t for t in reader.tensors}

    def has(self, name):
        return name in self.tensors

    def get(self, name):
        t = self.tensors[name]
        a = dequantize(t.data, t.tensor_type)
        return np.asarray(a, dtype=np.float32)

    def rows(self, name, start, stop):
        """Rows [start, stop) of a 2D tensor (outer dimension), dequantized."""
        t = self.tensors[name]
        a = dequantize(t.data[start:stop], t.tensor_type)
        return np.asarray(a, dtype=np.float32)


def rms_norm(x, w, eps):
    x64 = x.astype(np.float32)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def yarn_corr_dim(n_dims, n_ctx_orig, n_rot, base):
    return n_dims * math.log(n_ctx_orig / (n_rot * 2.0 * math.pi)) / (2.0 * math.log(base))


def rope_tables(cfg, positions):
    """cos and sin of shape (T, rope_dims/2), the magnitude correction included."""
    n_dims = cfg["rope_dims"]
    half = n_dims // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = cfg["rope_base"] ** (-2.0 * i / n_dims)
    if cfg["freq_factors"] is not None:
        inv_freq = inv_freq / cfg["freq_factors"].astype(np.float64)
    pos = np.asarray(positions, dtype=np.float64)[:, None]
    theta_extrap = pos * inv_freq[None, :]
    theta_interp = cfg["freq_scale"] * theta_extrap
    mscale = 1.0
    if cfg["yarn"]:
        fast = yarn_corr_dim(n_dims, cfg["yarn_orig_ctx"], cfg["yarn_beta_fast"], cfg["rope_base"])
        slow = yarn_corr_dim(n_dims, cfg["yarn_orig_ctx"], cfg["yarn_beta_slow"], cfg["rope_base"])
        low = max(0.0, math.floor(fast))
        high = min(n_dims - 1.0, math.ceil(slow))
        ramp = (i - low) / max(0.001, high - low)
        mix = 1.0 - np.clip(ramp, 0.0, 1.0)
        theta = theta_interp * (1.0 - mix[None, :]) + theta_extrap * mix[None, :]
        mscale = cfg["yarn_mscale"]
    else:
        theta = theta_interp
    return (np.cos(theta) * mscale).astype(np.float32), (np.sin(theta) * mscale).astype(np.float32)


def apply_rope(x, cos, sin, cfg):
    """x: (T, heads, head_dim); rotates the leading rope_dims of each head."""
    half = cfg["rope_dims"] // 2
    out = x.copy()
    c = cos[:, None, :]
    s = sin[:, None, :]
    if cfg["neox"]:
        x0 = x[:, :, :half]
        x1 = x[:, :, half:2 * half]
        out[:, :, :half] = x0 * c - x1 * s
        out[:, :, half:2 * half] = x0 * s + x1 * c
    else:
        x0 = x[:, :, 0:2 * half:2]
        x1 = x[:, :, 1:2 * half:2]
        out[:, :, 0:2 * half:2] = x0 * c - x1 * s
        out[:, :, 1:2 * half:2] = x0 * s + x1 * c
    return out


def read_config(reader, weights):
    a = field(reader, "general.architecture")
    k = lambda name, d=None: field(reader, a + "." + name, d)
    cfg = {"arch": a}
    cfg["n_layers"] = int(k("block_count"))
    cfg["n_heads"] = int(k("attention.head_count"))
    cfg["n_kv_heads"] = int(k("attention.head_count_kv", cfg["n_heads"]))
    cfg["d_model"] = int(k("embedding_length"))
    cfg["head_dim"] = int(k("attention.key_length", cfg["d_model"] // cfg["n_heads"]))
    cfg["eps"] = float(k("attention.layer_norm_rms_epsilon", 1e-5))
    cfg["rope_dims"] = int(k("rope.dimension_count", cfg["head_dim"]))
    cfg["rope_base"] = float(k("rope.freq_base", 10000.0))
    cfg["neox"] = a in ("qwen2", "qwen3", "qwen3moe")
    scaling = k("rope.scaling.type", "")
    factor = float(k("rope.scaling.factor", 0.0) or 0.0)
    cfg["freq_scale"] = 1.0
    cfg["yarn"] = False
    if scaling in ("linear", "yarn") and factor > 0:
        cfg["freq_scale"] = 1.0 / factor
    if scaling == "yarn" and factor > 0:
        cfg["yarn"] = True
        cfg["yarn_orig_ctx"] = float(k("rope.scaling.original_context_length", k("context_length")))
        cfg["yarn_beta_fast"] = float(k("rope.scaling.yarn_beta_fast", 32.0))
        cfg["yarn_beta_slow"] = float(k("rope.scaling.yarn_beta_slow", 1.0))
        get_mscale = lambda s, m: 1.0 if s <= 1.0 else 0.1 * m * math.log(s) + 1.0
        log_mul = float(k("rope.scaling.yarn_log_multiplier", 0.0) or 0.0)
        if log_mul != 0.0:
            cfg["yarn_mscale"] = get_mscale(factor, 1.0) / get_mscale(factor, log_mul)
        else:
            cfg["yarn_mscale"] = get_mscale(factor, 1.0)
    cfg["freq_factors"] = weights.get("rope_freqs.weight") if weights.has("rope_freqs.weight") else None
    cfg["attn_temp_scale"] = float(k("attention.temperature_scale", 0.0) or 0.0)
    if cfg["attn_temp_scale"]:
        if a == "mistral3":
            cfg["attn_temp_length"] = cfg["yarn_orig_ctx"]
        else:
            cfg["attn_temp_length"] = float(k("attention.temperature_length", 8192))
    granite = a in ("granite", "granitemoe")
    cfg["embedding_scale"] = float(k("embedding_scale", 1.0)) if granite else 1.0
    cfg["residual_scale"] = float(k("residual_scale", 1.0)) if granite else 1.0
    att_scale = float(k("attention.scale", 0.0) or 0.0) if granite else 0.0
    cfg["kq_scale"] = att_scale if att_scale else 1.0 / math.sqrt(cfg["head_dim"])
    cfg["logit_scale"] = float(k("logit_scale", 1.0) or 1.0) if granite else 1.0
    cfg["n_experts"] = int(k("expert_count", 0) or 0)
    cfg["n_experts_used"] = int(k("expert_used_count", 0) or 0)
    return cfg


def forward(path, tokens):
    reader = GGUFReader(path)
    w = Weights(reader)
    cfg = read_config(reader, w)
    T = len(tokens)
    H, KVH, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    eps = cfg["eps"]

    x = w.get("token_embd.weight")[np.asarray(tokens)] * cfg["embedding_scale"]
    x = x.astype(np.float32)
    positions = np.arange(T)
    cos, sin = rope_tables(cfg, positions)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)

    for li in range(cfg["n_layers"]):
        p = "blk.%d." % li
        h = rms_norm(x, w.get(p + "attn_norm.weight"), eps)
        q = h @ w.get(p + "attn_q.weight").T
        kk = h @ w.get(p + "attn_k.weight").T
        v = h @ w.get(p + "attn_v.weight").T
        if w.has(p + "attn_q.bias"):
            q = q + w.get(p + "attn_q.bias")
            kk = kk + w.get(p + "attn_k.bias")
            v = v + w.get(p + "attn_v.bias")
        q = q.reshape(T, H, hd)
        kk = kk.reshape(T, KVH, hd)
        v = v.reshape(T, KVH, hd)
        if w.has(p + "attn_q_norm.weight"):
            q = rms_norm(q, w.get(p + "attn_q_norm.weight"), eps)
            kk = rms_norm(kk, w.get(p + "attn_k_norm.weight"), eps)
        q = apply_rope(q, cos, sin, cfg)
        kk = apply_rope(kk, cos, sin, cfg)
        if cfg["attn_temp_scale"]:
            f = 1.0 + cfg["attn_temp_scale"] * np.log(1.0 + np.floor(positions / cfg["attn_temp_length"]))
            q = q * f[:, None, None].astype(np.float32)
        rep = H // KVH
        ctx = np.empty((T, H, hd), dtype=np.float32)
        for hh in range(H):
            kv = hh // rep
            s = (q[:, hh, :] @ kk[:, kv, :].T) * cfg["kq_scale"] + mask
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            e = e / e.sum(axis=1, keepdims=True)
            ctx[:, hh, :] = e @ v[:, kv, :]
        o = ctx.reshape(T, H * hd) @ w.get(p + "attn_output.weight").T
        x = x + cfg["residual_scale"] * o

        h = rms_norm(x, w.get(p + "ffn_norm.weight"), eps)
        if cfg["n_experts"] > 0:
            router = h @ w.get(p + "ffn_gate_inp.weight").T          # (T, E)
            probs = np.exp(router - router.max(axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            gate_e = w.get(p + "ffn_gate_exps.weight")                 # (E, df, d)
            up_e = w.get(p + "ffn_up_exps.weight")
            down_e = w.get(p + "ffn_down_exps.weight")                 # (E, d, df)
            out = np.zeros_like(x)
            for t in range(T):
                top = np.argsort(-probs[t], kind="stable")[: cfg["n_experts_used"]]
                wts = probs[t, top]
                wts = wts / wts.sum()
                for e, wt in zip(top, wts):
                    g = gate_e[e] @ h[t]
                    u = up_e[e] @ h[t]
                    out[t] += wt * (down_e[e] @ (silu(g) * u))
            f_out = out
        else:
            g = h @ w.get(p + "ffn_gate.weight").T
            u = h @ w.get(p + "ffn_up.weight").T
            f_out = (silu(g) * u) @ w.get(p + "ffn_down.weight").T
        x = x + cfg["residual_scale"] * f_out

    x = rms_norm(x, w.get("output_norm.weight"), eps)
    head = "output.weight" if w.has("output.weight") else "token_embd.weight"
    n_vocab = int(w.tensors[head].shape[1])
    logits = np.empty((T, n_vocab), dtype=np.float32)
    chunk = 16384
    for s in range(0, n_vocab, chunk):
        e = min(n_vocab, s + chunk)
        logits[:, s:e] = x @ w.rows(head, s, e).T
    if cfg["logit_scale"] != 1.0:
        logits = logits / cfg["logit_scale"]
    return logits


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    tokens = [int(t) for t in sys.argv[3:]]
    logits = forward(sys.argv[1], tokens)
    logits.astype(np.float32).tofile(sys.argv[2])
    print("tokens %d vocabulary %d" % logits.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
