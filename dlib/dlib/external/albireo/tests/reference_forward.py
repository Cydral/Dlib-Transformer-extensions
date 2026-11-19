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
    python3 reference_forward.py model.gguf out.bin --generate 24 id id id ...

Supported: llama, mistral, mistral3, qwen2, qwen3, qwen3moe, granite, granitemoe (and
llama files that carry experts, such as Mixtral), and olala.

olala (Olala 1.0 mini of OVHai LLM) is written after modeling_olala.py of its authors: no
position encoding and no normalization of the residual stream, which each block turns on
its sphere (geodesic update); blocks whose mixer is a Mamba-3 layer with several inputs
and outputs, or a differential attention whose keys and values are tensor products; a
mixture of experts in a latent space beside a shared expert.  Its tokens are read one
after the other, each layer keeping its state, as a decoder does.  --generate appends the
most likely token as many times, and prints the ids and the text of the result.
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
        self.kept = {}

    def has(self, name):
        return name in self.tensors

    def get(self, name):
        t = self.tensors[name]
        a = dequantize(t.data, t.tensor_type)
        return np.asarray(a, dtype=np.float32)

    def keep(self, name):
        """A tensor dequantized once, for the layers that read it at every token."""
        if name not in self.kept:
            self.kept[name] = self.get(name)
        return self.kept[name]

    def expert(self, name, e):
        """The matrix of one expert of a tensor that stacks them (outer dimension)."""
        t = self.tensors[name]
        return np.asarray(dequantize(t.data[e], t.tensor_type), dtype=np.float32)

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


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def geodesic(x, g, scale, bias, layer):
    """The residual x turned on its sphere towards the output g of a mixer or of a feed-forward block."""
    x, g = x.astype(np.float64), g.astype(np.float64)
    tangent = g - (np.dot(x, g) / max(np.dot(x, x), 1e-12)) * x
    norm = max(np.linalg.norm(tangent), 1e-8)
    radius = max(np.linalg.norm(x), 1e-6)
    theta = min(norm / radius, math.pi / 4)
    theta = min((theta * scale + bias) / (layer + 1), math.pi / 4)
    return (x * math.cos(theta) + (tangent / norm) * radius * math.sin(theta)).astype(np.float32)


def olala_config(reader):
    a = "olala."
    cfg = {key: field(reader, a + name) for key, name in (
        ("n_layers", "block_count"), ("d", "embedding_length"), ("n_heads", "attention.head_count"),
        ("n_kv_heads", "attention.head_count_kv"), ("head_dim", "attention.key_length"),
        ("eps", "attention.layer_norm_rms_epsilon"), ("window", "attention.sliding_window"),
        ("softcap", "attn_logit_softcapping"), ("n_used", "expert_used_count"),
        ("expert_scale", "expert_weights_scale"), ("d_state", "ssm.state_size"), ("rank", "mamba_mimo_dim"),
        ("ssm_head_dim", "mamba_headdim"), ("n_signal", "num_signal_heads"), ("tpa_rank", "tpa_rank"),
        ("n_angles", "num_rope_angles"), ("layer_types", "layer_types"), ("gate_bias", "gate_bias"))}
    cfg["softcap"] = cfg["softcap"] or 0.0
    cfg["gate_bias"] = cfg["gate_bias"] or 0.0
    return cfg


class OlalaState:
    """What the layers keep of a sequence: the states of the Mamba-3 layers, the keys and values of the others."""

    def __init__(self, cfg):
        self.position = 0
        self.layers = [None] * cfg["n_layers"]


def olala_mamba3(cfg, w, p, x, state):
    N, R, P, S = cfg["d_state"], cfg["rank"], cfg["ssm_head_dim"], cfg["n_angles"]
    proj = (w.keep(p + "ssm_in.weight") @ x).reshape(-1, 2 * P + 3)
    H = proj.shape[0]
    z, xin = proj[:, :P], proj[:, P:2 * P]
    step = softplus(proj[:, 2 * P] + w.keep(p + "ssm_dt_bias")).astype(np.float64)
    decay = np.minimum(-softplus(proj[:, 2 * P + 1]), -1e-4).astype(np.float64)
    trap = sigmoid(proj[:, 2 * P + 2].astype(np.float64))
    dyn = w.keep(p + "ssm_in_dyn.weight") @ x
    b = rms_norm(dyn[:R * N].reshape(R, N), w.keep(p + "ssm_b_norm.weight"), cfg["eps"])
    c = rms_norm(dyn[R * N:2 * R * N].reshape(R, N), w.keep(p + "ssm_c_norm.weight"), cfg["eps"])
    if state is None:
        state = {"angle": np.zeros((H, S)), "ssm": np.zeros((H, P, N)), "k": np.zeros((R, H, N)), "v": np.zeros((H, P))}
    angle = state["angle"] + np.tanh(dyn[2 * R * N:].astype(np.float64))[None, :] * step[:, None] * math.pi
    cos, sin = np.cos(angle)[None], np.sin(angle)[None]

    def rotate(v):
        # The pairs (i, i + N / 2) of the first S dimensions of each half.
        out = v.copy()
        a, b2 = v[..., :S], v[..., N // 2:N // 2 + S]
        out[..., :S] = a * cos - b2 * sin
        out[..., N // 2:N // 2 + S] = a * sin + b2 * cos
        return out

    # (R, H, N): the weights keep their biases head by head then rank by rank.
    k = rotate(b[:, None, :].astype(np.float64) + w.keep(p + "ssm_b_bias").transpose(1, 0, 2))
    q = rotate(c[:, None, :].astype(np.float64) + w.keep(p + "ssm_c_bias").transpose(1, 0, 2))
    alpha = np.exp(decay * step)
    beta = (1.0 - trap) * step * alpha
    gamma = trap * step
    mx = w.keep(p + "ssm_mimo_x").transpose(1, 0, 2).astype(np.float64)        # (R, H, P)
    mz = w.keep(p + "ssm_mimo_z").transpose(1, 0, 2).astype(np.float64)
    mo = w.keep(p + "ssm_mimo_o").transpose(1, 0, 2).astype(np.float64)
    xv = xin[None].astype(np.float64) * mx
    xs = state["v"][None] * mx
    ssm = (state["ssm"] * alpha[:, None, None]
           + np.einsum("rhp,rhn->hpn", xv * gamma[None, :, None], k)
           + np.einsum("rhp,rhn->hpn", xs * beta[None, :, None], state["k"]))
    o = np.einsum("hpn,rhn->rhp", ssm, q) + xv * w.keep(p + "ssm_d").astype(np.float64)[None, :, None]
    o = o * silu(z[None].astype(np.float64) * mz)
    y = np.einsum("rhp,rhp->hp", o, mo)
    angle = np.mod(angle + math.pi, 2 * math.pi) - math.pi
    return y.reshape(-1), {"angle": angle, "ssm": ssm, "k": k, "v": xin.astype(np.float64)}


def olala_attention(cfg, w, p, x, state, position):
    H, KV, hd, rank, signal = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["tpa_rank"], cfg["n_signal"]
    group = H // KV
    q = (w.keep(p + "attn_q.weight") @ x).reshape(H, hd)

    def factors(which):
        a = (w.keep(p + "attn_wa_" + which + ".weight") @ x).reshape(KV, rank)
        b = (w.keep(p + "attn_wb_" + which + ".weight") @ x).reshape(rank, hd)
        return (a @ b) / rank

    k_raw, v_raw = factors("k"), factors("v")
    if state is None:
        state = {"k": [], "v": [], "last": None}
    k, v = k_raw, v_raw
    if w.has(p + "attn_shift_k.weight") and state["last"] is not None:
        # Each key and value leans on the one of the previous token, before its own shift.
        ak = sigmoid(w.keep(p + "attn_shift_k.weight") @ x)[:, None]
        av = sigmoid(w.keep(p + "attn_shift_v.weight") @ x)[:, None]
        k = ak * state["last"][0] + (1.0 - ak) * k_raw
        v = av * state["last"][1] + (1.0 - av) * v_raw
    q = rms_norm(q, w.keep(p + "attn_q_norm.weight"), cfg["eps"])
    k = rms_norm(k, w.keep(p + "attn_k_norm.weight"), cfg["eps"])
    window = cfg["window"] or 0
    if w.has(p + "attn_softmax_scaler"):
        count = position + 1.0
        q = q * (w.keep(p + "attn_softmax_scaler")[:, None] * math.log(min(count, window) if window > 0 else count))
    keys, values = state["k"] + [k], state["v"] + [v]
    first = max(0, len(keys) - 1 - window) if window > 0 else 0
    K = np.stack(keys[first:]).astype(np.float32)          # (T, KV, hd)
    V = np.stack(values[first:]).astype(np.float32)
    out = np.zeros((H, hd), dtype=np.float64)
    for h in range(H):
        scores = (K[:, h // group, :] @ q[h]).astype(np.float64) / hd
        if cfg["softcap"] > 0.0:
            scores = np.tanh(scores / cfg["softcap"]) * cfg["softcap"]
        scores = np.exp(scores - scores.max())
        out[h] = (scores / scores.sum()) @ V[:, h // group, :]
    # Each group of heads on one key head: its first heads carry the signal, its last one the noise.
    out = out.reshape(KV, group, hd)
    lam = sigmoid((w.keep(p + "attn_lambda.weight") @ x).astype(np.float64))[:, None, None]
    y = (out[:, :group - 1] - lam * out[:, group - 1:]).reshape(-1)
    assert y.size == signal * hd
    if w.has(p + "attn_gate.weight"):
        y = y * silu((w.keep(p + "attn_gate.weight") @ x).astype(np.float64) + cfg["gate_bias"])
    return y, {"k": keys, "v": values, "last": (k_raw, v_raw)}


def olala_experts(cfg, w, p, x):
    scores = sigmoid((w.keep(p + "ffn_gate_inp.weight") @ x).astype(np.float64))
    routed = scores + w.keep(p + "exp_probs_b") if w.has(p + "exp_probs_b") else scores
    chosen = np.argsort(-routed, kind="stable")[:cfg["n_used"]]
    share = scores[chosen]
    if cfg["n_used"] > 1:
        share = share / (share.sum() + 1e-20)
    share = share * cfg["expert_scale"]
    latent = w.keep(p + "ffn_latent_down.weight") @ x if w.has(p + "ffn_latent_down.weight") else x
    y = np.zeros(latent.shape[0], dtype=np.float64)
    for e, s in zip(chosen, share):
        hidden = np.maximum(w.expert(p + "ffn_up_exps.weight", int(e)) @ latent, 0.0).astype(np.float64) ** 2
        y += s * (w.expert(p + "ffn_down_exps.weight", int(e)).astype(np.float64) @ hidden)
    if w.has(p + "ffn_latent_up.weight"):
        y = w.keep(p + "ffn_latent_up.weight").astype(np.float64) @ y
    if w.has(p + "ffn_up_shexp.weight"):
        hidden = np.maximum(w.keep(p + "ffn_up_shexp.weight") @ x, 0.0).astype(np.float64) ** 2
        y = y + w.keep(p + "ffn_down_shexp.weight").astype(np.float64) @ hidden
    return y, chosen


def olala_step(cfg, w, token, state):
    """Reads one token and returns the hidden state the logits come from."""
    x = w.rows("token_embd.weight", token, token + 1)[0]
    x = x / max(float(np.linalg.norm(x.astype(np.float64))), 1e-12)
    for li, kind in enumerate(cfg["layer_types"]):
        p = "blk.%d." % li
        if kind == "M":
            y, state.layers[li] = olala_mamba3(cfg, w, p, x, state.layers[li])
        elif kind == "V":
            y, state.layers[li] = olala_attention(cfg, w, p, x, state.layers[li], state.position)
        else:
            raise ValueError("unsupported layer type: %s" % kind)
        g = w.keep(p + "attn_output.weight").astype(np.float64) @ y
        x = geodesic(x, g, float(w.keep(p + "geo_mixer_scale")[0]), float(w.keep(p + "geo_mixer_bias")[0]), li)
        y, _ = olala_experts(cfg, w, p, x)
        x = geodesic(x, y, float(w.keep(p + "geo_mlp_scale")[0]), float(w.keep(p + "geo_mlp_bias")[0]), li)
    state.position += 1
    return x


def olala_forward(reader, w, tokens, generate=0):
    cfg = olala_config(reader)
    if field(reader, "olala.final_norm"):
        raise ValueError("a final normalization is not implemented for olala")
    head = w.keep("output.weight")
    state = OlalaState(cfg)
    rows, tokens = [], list(tokens)
    for i in range(len(tokens) + generate):
        if i >= len(tokens):
            tokens.append(int(np.argmax(rows[-1])))
        rows.append(head @ olala_step(cfg, w, tokens[i], state))
    return np.stack(rows[:len(rows) - generate]) if generate else np.stack(rows), tokens


def forward(path, tokens, generate=0):
    reader = GGUFReader(path)
    w = Weights(reader)
    if field(reader, "general.architecture") == "olala":
        logits, tokens = olala_forward(reader, w, tokens, generate)
        if generate:
            words = field(reader, "tokenizer.ggml.tokens")
            text = "".join(words[t] for t in tokens).replace("\u0120", " ").replace("\u010a", "\n")
            print("ids:", " ".join(str(t) for t in tokens))
            print("text:", text)
        return logits
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
    rest, generate = sys.argv[3:], 0
    if rest[0] == "--generate":
        generate, rest = int(rest[1]), rest[2:]
    tokens = [int(t) for t in rest]
    logits = forward(sys.argv[1], tokens, generate)
    logits.astype(np.float32).tofile(sys.argv[2])
    print("tokens %d vocabulary %d" % logits.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
