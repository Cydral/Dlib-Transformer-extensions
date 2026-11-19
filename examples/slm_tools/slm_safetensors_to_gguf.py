#!/usr/bin/env python3
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Converts a language model published as open weights (safetensors, the format of training and
fine-tuning) into GGUF files for inference, in full precision or quantized.

    python3 slm_safetensors_to_gguf.py fetch OVHaiLLM/Olala-v1.0-mini-7A1B Olala-v1.0-mini-7A1B
    python3 slm_safetensors_to_gguf.py plan Olala-v1.0-mini-7A1B --type q5_k_m
    python3 slm_safetensors_to_gguf.py convert Olala-v1.0-mini-7A1B --type q8_0 --type q5_k_m --out gguf
    python3 slm_safetensors_to_gguf.py verify Olala-v1.0-mini-7A1B gguf/Olala-v1.0-mini-7A1B-q5_k_m.gguf
    python3 slm_safetensors_to_gguf.py inspect gguf/Olala-v1.0-mini-7A1B-q5_k_m.gguf
    python3 slm_safetensors_to_gguf.py compare gguf/Olala-v1.0-mini-7A1B-q5_k_m.gguf official-q5_k_m.gguf
    python3 slm_safetensors_to_gguf.py selftest

fetch     downloads the files of a checkpoint from the Hugging Face hub (configuration, tokenizer,
          chat template, weights, license and notices), resuming the files already begun.
plan      prints the tensors a conversion would write, with their shapes and formats, from the
          header of the weights alone.
convert   writes one GGUF file per --type, named after the folder of the checkpoint (or --name)
          and its type, and copies the license and the notices of the checkpoint beside them.  A
          tensor whose values the format of the plan cannot hold is kept in BF16, and named.
verify    sets each tensor of a GGUF file against the values its checkpoint gives it: a tensor
          written from other values, in another layout or beyond the rounding of its format fails.
inspect   prints the metadata and the tensors of a GGUF file.
compare   compares two GGUF files: metadata, then each tensor, by its bytes when both store it in
          the same format and by its values otherwise.
selftest  checks the quantizers against their own decoders and converts a small random model.

A checkpoint is a folder with config.json, the weights (model.safetensors, or the shards
model-00001-of-0000N.safetensors), tokenizer.json, tokenizer_config.json and, when the model has
one, chat_template.jinja.

Formats (--type): f32, f16, bf16, q8_0, q6_k, q5_k_m and q4_k_m.  A quantized format is a base
format for the matrices of the model; the vectors (normalization weights, biases, scales) stay in
single precision, and each architecture names the matrices that keep more bits than the base
format gives, where its authors measured that fewer bits break the model.  A matrix whose rows do
not hold whole blocks of its format takes the 8-bit format, or bfloat16 when its rows do not hold
blocks of 32 values either.

The quantized formats and F16 hold their values through numbers in half precision: the scale of
a block of 6 bits rounds to zero for values under about 1e-4, and the block then reads as zeros.
convert measures each tensor it writes against its values and writes again in BF16 a tensor its
format leaves beyond the distance verify accepts, so that every tensor of a file follows the
weights.  The latent down projections of five layers of Olala 1.0 mini (7, 8, 9, 10 and 14) hold
such values, near 2e-5, as do the other tensors of the routed experts of these layers, whose part
in the output of their block stays under 2e-18 of that of the shared expert.  The 6-bit file of
the authors reads these projections as zeros; the one written here keeps them in BF16, and its
other tensors have the formats of theirs.

Architectures: olala (Olala 1.0 mini of OVHai LLM: Mamba-3 layers with several inputs and outputs,
differential attention with keys and values as tensor products, a mixture of experts in a latent
space, geodesic updates of the residual stream), and the models its authors fine-tuned from it
(Olala-v1.0-summarise, Olala-v1.0-translate), whose checkpoints store in BF16 the biases and the
other small tensors that a file keeps in single precision.  The files follow the GGUF container of
the authors (github.com/OVHai-LLM/llama.cpp: architecture name, metadata keys, tensor names, the
packed copies of the constants of the Mamba-3 layers), so that a program written for their files
reads these too, and they follow their recipes for the tensors that keep more bits.  The full
precision tensors and the 8-bit ones have the bytes of the files of the authors.  The k-quants are
computed here with the search of the ggml library and without the importance matrix their files
use: each weight keeps the rounding of its format, where theirs give the precision to the columns
that the activations of a calibration text weight most.  On the first two layers of Olala 1.0 mini
in 6 bits, the outputs of the experts stand within 0.2 % of those of the weights either way.  The
authors publish no q4_k_m file: they measured it worse than q5_k_m for 0.3 GB less.

The program needs Python 3.8 and numpy, nothing else.  It reads the weights through a memory map
and writes one tensor at a time: its memory stays near the size of the largest tensor in single
precision, times the number of --jobs.
"""

import argparse
import hashlib
import http.client
import json
import math
import mmap
import os
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ---------------------------------------------------------------------------------------------
# GGUF: tensor formats

F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, BF16 = 0, 1, 8, 12, 13, 14, 30
TYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K", BF16: "BF16"}
# Values per block and bytes per block.
BLOCKS = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q8_0: (32, 34), Q4_K: (256, 144), Q5_K: (256, 176),
          Q6_K: (256, 210)}
# general.file_type of each --type, as llama.cpp numbers them.
FILE_TYPES = {"f32": 0, "f16": 1, "q8_0": 7, "q4_k_m": 15, "q5_k_m": 17, "q6_k": 18, "bf16": 32}
BASE_TYPES = {"f32": F32, "f16": F16, "bf16": BF16, "q8_0": Q8_0, "q6_k": Q6_K, "q5_k_m": Q5_K, "q4_k_m": Q4_K}


def tensor_bytes(shape, kind):
    values, size = BLOCKS[kind]
    count = int(np.prod(shape)) if len(shape) else 1
    if shape and shape[-1] % values != 0:
        raise ValueError("rows of %d values do not hold whole %s blocks" % (shape[-1], TYPE_NAMES[kind]))
    return count // values * size


def fitting(kind, shape):
    """kind, or the format that takes its place when the rows of a tensor do not hold whole blocks of it."""
    if shape[-1] % BLOCKS[kind][0] == 0:
        return kind
    return Q8_0 if shape[-1] % 32 == 0 else BF16


def f32_to_bf16(x):
    """Single precision to bfloat16, rounded to nearest even, as the ggml library rounds."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)
    quiet = ((u >> 16) | 64).astype(np.uint16)
    return np.where((u & 0x7FFFFFFF) > 0x7F800000, quiet, rounded)


def bf16_to_f32(u16):
    return (np.ascontiguousarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def nearest(x):
    return np.rint(x)


def quantize_q8_0(x):
    x = x.reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    v = x * inv[:, None]
    q = np.trunc(v + np.copysign(0.5, v)).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)


def dequantize_q8_0(raw):
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def affine_levels(x, weights, nmax, rmin, rdelta, nstep):
    """
    For each row of x, the scale and the offset of nmax + 1 levels that best give its values, the
    weights telling the values that matter most: the search of make_qkx2_quants of the ggml
    library, over all the rows at once.  Returns the scales and the offsets as positive numbers,
    x being near scale * level - offset.
    """
    lo = np.minimum(x.min(axis=1), 0.0)
    hi = x.max(axis=1)
    span = hi - lo
    flat = span <= 0
    span = np.where(flat, 1.0, span)
    sum_w = weights.sum(axis=1)
    sum_x = (weights * x).sum(axis=1)

    def error(scale, offset, levels):
        diff = scale[:, None] * levels + offset[:, None] - x
        return (weights * diff * diff).sum(axis=1)

    iscale = nmax / span
    levels = np.clip(nearest(iscale[:, None] * (x - lo[:, None])), 0, nmax)
    best_scale, best_min = 1.0 / iscale, lo.copy()
    best = error(best_scale, best_min, levels)
    for step in range(nstep + 1):
        # The offset a step found better is the one the next steps start from, as in the library.
        span = hi - best_min
        span = np.where(span > 0, span, 1.0)
        iscale = (rmin + rdelta * step + nmax) / span
        l = np.clip(nearest(iscale[:, None] * (x - best_min[:, None])), 0, nmax)
        sum_l = (weights * l).sum(axis=1)
        sum_l2 = (weights * l * l).sum(axis=1)
        sum_xl = (weights * l * x).sum(axis=1)
        det = sum_w * sum_l2 - sum_l * sum_l
        ok = det > 0
        safe = np.where(ok, det, 1.0)
        scale = (sum_w * sum_xl - sum_x * sum_l) / safe
        offset = (sum_l2 * sum_x - sum_l * sum_xl) / safe
        positive = offset > 0
        scale = np.where(positive, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), scale)
        offset = np.where(positive, 0.0, offset)
        cur = error(scale, offset, l)
        better = ok & ~flat & (cur < best)
        best = np.where(better, cur, best)
        best_scale = np.where(better, scale, best_scale)
        best_min = np.where(better, offset, best_min)
    best_scale = np.where(flat, 0.0, best_scale)
    best_min = np.where(flat, lo, best_min)
    return best_scale.astype(np.float32), (-best_min).astype(np.float32)


def quantize_k45(x, bits):
    """Q4_K (bits 4) and Q5_K (bits 5): 8 groups of 32 values, each with a scale and an offset of 6 bits."""
    nmax = (1 << bits) - 1
    x = x.reshape(-1, 32).astype(np.float32)
    weights = np.sqrt((x * x).mean(axis=1, keepdims=True)) + np.abs(x)
    if bits == 4:
        scales, mins = affine_levels(x, weights, nmax, -1.0, 0.1, 20)
    else:
        scales, mins = affine_levels(x, weights, nmax, -0.5, 0.1, 15)
    scales, mins = scales.reshape(-1, 8), mins.reshape(-1, 8)
    max_scale, max_min = scales.max(axis=1), mins.max(axis=1)
    inv_scale = np.divide(63.0, max_scale, out=np.zeros_like(max_scale), where=max_scale > 0)
    inv_min = np.divide(63.0, max_min, out=np.zeros_like(max_min), where=max_min > 0)
    ls = np.minimum(63, nearest(inv_scale[:, None] * scales)).astype(np.uint8)
    lm = np.minimum(63, nearest(inv_min[:, None] * mins)).astype(np.uint8)
    d16 = (max_scale / 63.0).astype(np.float16)
    m16 = (max_min / 63.0).astype(np.float16)
    d = d16.astype(np.float32)[:, None] * ls
    dm = m16.astype(np.float32)[:, None] * lm
    xs = x.reshape(-1, 8, 32)
    safe = np.where(d > 0, d, 1.0)[:, :, None]
    levels = np.clip(nearest((xs + dm[:, :, None]) / safe), 0, nmax)
    levels = np.where((d > 0)[:, :, None], levels, 0).astype(np.uint8)
    blocks = xs.shape[0]
    packed = np.zeros((blocks, 12), dtype=np.uint8)
    packed[:, 0:4] = ls[:, 0:4] | ((ls[:, 4:8] >> 4) << 6)
    packed[:, 4:8] = lm[:, 0:4] | ((lm[:, 4:8] >> 4) << 6)
    packed[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    pairs = levels.reshape(blocks, 4, 2, 32)
    low = ((pairs[:, :, 0, :] & 0xF) | ((pairs[:, :, 1, :] & 0xF) << 4)).reshape(blocks, 128)
    head = np.concatenate([d16.view(np.uint8).reshape(-1, 2), m16.view(np.uint8).reshape(-1, 2), packed], axis=1)
    if bits == 4:
        return np.concatenate([head, low], axis=1).reshape(-1)
    # The fifth bit of the eight groups of a block, one byte per value of a group.
    high = np.zeros((blocks, 32), dtype=np.uint8)
    for group in range(4):
        high |= ((pairs[:, group, 0, :] >> 4) << (2 * group)) | ((pairs[:, group, 1, :] >> 4) << (2 * group + 1))
    return np.concatenate([head, high, low], axis=1).reshape(-1)


def k45_scales(packed):
    """The 8 scales and the 8 offsets of 6 bits of each block, from their 12 bytes."""
    ls = np.empty((packed.shape[0], 8), dtype=np.uint8)
    lm = np.empty_like(ls)
    ls[:, 0:4] = packed[:, 0:4] & 63
    lm[:, 0:4] = packed[:, 4:8] & 63
    ls[:, 4:8] = (packed[:, 8:12] & 0xF) | ((packed[:, 0:4] >> 6) << 4)
    lm[:, 4:8] = (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 6) << 4)
    return ls, lm


def dequantize_k45(raw, bits):
    size = 144 if bits == 4 else 176
    b = raw.reshape(-1, size)
    blocks = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    ls, lm = k45_scales(b[:, 4:16])
    low = b[:, size - 128:].reshape(blocks, 4, 32)
    levels = np.empty((blocks, 4, 2, 32), dtype=np.uint8)
    levels[:, :, 0, :] = low & 0xF
    levels[:, :, 1, :] = low >> 4
    if bits == 5:
        high = b[:, 16:48]
        for group in range(4):
            levels[:, group, 0, :] |= ((high >> (2 * group)) & 1) << 4
            levels[:, group, 1, :] |= ((high >> (2 * group + 1)) & 1) << 4
    levels = levels.reshape(blocks, 8, 32).astype(np.float32)
    out = (d * ls)[:, :, None] * levels - (dmin * lm)[:, :, None]
    return out.reshape(-1)


def symmetric_levels(x, nmax):
    """The scale of 2 * nmax signed levels that best gives each row of x (make_qx_quants of the ggml library)."""
    index = np.abs(x).argmax(axis=1)
    peak = x[np.arange(x.shape[0]), index]
    zero = np.abs(peak) < 1e-15
    peak = np.where(zero, 1.0, peak)
    weights = x * x
    best_scale = np.zeros(x.shape[0], dtype=np.float64)
    best = np.full(x.shape[0], -1.0)
    for step in [0] + [s for s in range(-9, 10) if s != 0]:
        iscale = -(nmax + 0.1 * step) / peak
        l = np.clip(nearest(iscale[:, None] * x), -nmax, nmax - 1)
        sumlx = (weights * x * l).sum(axis=1)
        suml2 = (weights * l * l).sum(axis=1)
        ok = suml2 > 0
        scale = sumlx / np.where(ok, suml2, 1.0)
        gain = scale * sumlx
        better = ok & (gain > best)
        best = np.where(better, gain, best)
        best_scale = np.where(better, scale, best_scale)
    return np.where(zero, 0.0, best_scale).astype(np.float32)


def quantize_q6_k(x):
    """Q6_K: 16 groups of 16 values, each with a signed scale of 8 bits, values of 6 bits."""
    x = x.reshape(-1, 16).astype(np.float32)
    scales = symmetric_levels(x, 32).reshape(-1, 16)
    index = np.abs(scales).argmax(axis=1)
    peak = scales[np.arange(scales.shape[0]), index]
    zero = np.abs(peak) < 1e-15
    iscale = -128.0 / np.where(zero, 1.0, peak)
    d16 = np.where(zero, 0.0, 1.0 / iscale).astype(np.float16)
    ls = np.where(zero[:, None], 0, np.minimum(127, nearest(iscale[:, None] * scales))).astype(np.int8)
    d = d16.astype(np.float32)[:, None] * ls
    xs = x.reshape(-1, 16, 16)
    safe = np.where(d != 0, d, 1.0)[:, :, None]
    levels = np.clip(nearest(xs / safe), -32, 31)
    levels = (np.where((d != 0)[:, :, None], levels, 0) + 32).astype(np.uint8)
    blocks = xs.shape[0]
    # Two halves of 128 values, each as four runs of 32.
    runs = levels.reshape(blocks, 2, 4, 32)
    ql = np.empty((blocks, 2, 2, 32), dtype=np.uint8)
    ql[:, :, 0, :] = (runs[:, :, 0, :] & 0xF) | ((runs[:, :, 2, :] & 0xF) << 4)
    ql[:, :, 1, :] = (runs[:, :, 1, :] & 0xF) | ((runs[:, :, 3, :] & 0xF) << 4)
    qh = ((runs[:, :, 0, :] >> 4) | ((runs[:, :, 1, :] >> 4) << 2) | ((runs[:, :, 2, :] >> 4) << 4)
          | ((runs[:, :, 3, :] >> 4) << 6))
    return np.concatenate([ql.reshape(blocks, 128), qh.reshape(blocks, 64), ls.view(np.uint8),
                           d16.view(np.uint8).reshape(-1, 2)], axis=1).reshape(-1)


def dequantize_q6_k(raw):
    b = raw.reshape(-1, 210)
    blocks = b.shape[0]
    ql = b[:, 0:128].reshape(blocks, 2, 2, 32)
    qh = b[:, 128:192].reshape(blocks, 2, 32)
    ls = b[:, 192:208].view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
    runs = np.empty((blocks, 2, 4, 32), dtype=np.int16)
    runs[:, :, 0, :] = (ql[:, :, 0, :] & 0xF) | ((qh & 3) << 4)
    runs[:, :, 1, :] = (ql[:, :, 1, :] & 0xF) | (((qh >> 2) & 3) << 4)
    runs[:, :, 2, :] = (ql[:, :, 0, :] >> 4) | (((qh >> 4) & 3) << 4)
    runs[:, :, 3, :] = (ql[:, :, 1, :] >> 4) | (((qh >> 6) & 3) << 4)
    levels = (runs.reshape(blocks, 16, 16) - 32).astype(np.float32)
    return ((d * ls)[:, :, None] * levels).reshape(-1)


def encode(x, kind):
    """The bytes of float32 values in a tensor format, as a uint8 array."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if kind == F32:
        return x.view(np.uint8).reshape(-1)
    if kind == F16:
        return x.astype(np.float16).view(np.uint8).reshape(-1)
    if kind == BF16:
        return f32_to_bf16(x).view(np.uint8).reshape(-1)
    if kind == Q8_0:
        return quantize_q8_0(x)
    if kind == Q4_K:
        return quantize_k45(x, 4)
    if kind == Q5_K:
        return quantize_k45(x, 5)
    if kind == Q6_K:
        return quantize_q6_k(x)
    raise ValueError("no encoder for tensor type %d" % kind)


def decode(raw, kind):
    """The float32 values of the bytes of a tensor."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if kind == F32:
        return raw.view(np.float32).copy()
    if kind == F16:
        return raw.view(np.float16).astype(np.float32)
    if kind == BF16:
        return bf16_to_f32(raw.view(np.uint16))
    if kind == Q8_0:
        return dequantize_q8_0(raw)
    if kind == Q4_K:
        return dequantize_k45(raw, 4)
    if kind == Q5_K:
        return dequantize_k45(raw, 5)
    if kind == Q6_K:
        return dequantize_q6_k(raw)
    raise ValueError("no decoder for tensor type %d" % kind)


# ---------------------------------------------------------------------------------------------
# safetensors

SAFETENSORS_TYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64,
                     "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


class Checkpoint:
    """The tensors of the safetensors files of a folder, read where they lie."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        names = sorted(n for n in os.listdir(folder) if n.endswith(".safetensors"))
        if not names:
            raise ValueError("no .safetensors file in %s" % folder)
        self.tensors, self.maps = {}, []
        for name in names:
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                if length > 256 * 1024 * 1024:
                    raise ValueError("%s: a header of %d bytes is not one of a safetensors file" % (name, length))
                header = json.loads(f.read(length).decode("utf-8"))
            header.pop("__metadata__", None)
            for tensor, entry in header.items():
                begin, end = entry["data_offsets"]
                self.tensors[tensor] = (len(self.maps), entry["dtype"], tuple(entry["shape"]), 8 + length + begin,
                                        end - begin)
            self.maps.append((path, None))

    def raw(self, name):
        """The stored array of a tensor, in its own type."""
        index, dtype, shape, offset, size = self.tensors[name]
        path, mapped = self.maps[index]
        if mapped is None:
            with open(path, "rb") as f:
                mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps[index] = (path, mapped)
        if offset + size > len(mapped):
            raise ValueError("%s: the weights of %s lie beyond the end of the file" % (path, name))
        kind = SAFETENSORS_TYPES[dtype]
        return np.frombuffer(mapped, dtype=kind, count=size // np.dtype(kind).itemsize, offset=offset).reshape(shape)

    def dtype(self, name):
        return self.tensors[name][1]

    def shape(self, name):
        return self.tensors[name][2]

    def values(self, name):
        """A tensor in single precision."""
        a = self.raw(name)
        if self.dtype(name) == "BF16":
            return bf16_to_f32(a).reshape(a.shape)
        return np.asarray(a, dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# GGUF: writer and reader

GGUF_TYPES = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "array": 9,
              "u64": 10, "i64": 11, "f64": 12}
GGUF_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
ALIGNMENT = 32


def pack_string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def pack_value(kind, value):
    code = GGUF_TYPES[kind]
    if kind == "str":
        return pack_string(value)
    return struct.pack("<" + GGUF_FORMATS[code], value)


def pack_metadata(items):
    """items: (key, kind, value) with kind a scalar type, or ("array", element kind)."""
    out = []
    for key, kind, value in items:
        out.append(pack_string(key))
        if isinstance(kind, tuple):
            element = kind[1]
            out.append(struct.pack("<IIQ", GGUF_TYPES["array"], GGUF_TYPES[element], len(value)))
            if element == "str":
                out.extend(pack_string(v) for v in value)
            else:
                out.append(np.asarray(value, dtype="<" + {"i32": "i4", "u32": "u4", "f32": "f4"}[element]).tobytes())
        else:
            out.append(struct.pack("<I", GGUF_TYPES[kind]))
            out.append(pack_value(kind, value))
    return b"".join(out)


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


class GGUFFile:
    """The header of a GGUF file and its tensors, read where they lie."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "rb")
        self.mapped = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        d = self.mapped
        if d[:4] != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        self.at = 4
        version, n_tensors, n_kv = self.take("I"), self.take("Q"), self.take("Q")
        if version not in (2, 3):
            raise ValueError("%s: GGUF version %d is not supported" % (path, version))
        self.metadata, self.kinds = {}, {}
        for _ in range(n_kv):
            key = self.string()
            kind = self.take("I")
            self.kinds[key] = kind
            self.metadata[key] = self.value(kind)
        self.tensors = {}
        order = []
        for _ in range(n_tensors):
            name = self.string()
            dims = [self.take("Q") for _ in range(self.take("I"))]
            kind, offset = self.take("I"), self.take("Q")
            self.tensors[name] = (tuple(reversed(dims)), kind, offset)
            order.append(name)
        self.order = order
        alignment = int(self.metadata.get("general.alignment", ALIGNMENT))
        self.data = (self.at + alignment - 1) // alignment * alignment

    def take(self, fmt):
        (v,) = struct.unpack_from("<" + fmt, self.mapped, self.at)
        self.at += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.take("Q")
        s = self.mapped[self.at:self.at + n].decode("utf-8", "replace")
        self.at += n
        return s

    def value(self, kind):
        if kind in GGUF_FORMATS:
            return self.take(GGUF_FORMATS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, n = self.take("I"), self.take("Q")
            if element in GGUF_FORMATS and element != 7:
                fmt = "<" + GGUF_FORMATS[element]
                a = np.frombuffer(self.mapped, dtype=np.dtype(fmt), count=n, offset=self.at)
                self.at += n * a.itemsize
                return a.tolist()
            return [self.value(element) for _ in range(n)]
        raise ValueError("%s: metadata type %d is not one of GGUF" % (self.path, kind))

    def raw(self, name):
        shape, kind, offset = self.tensors[name]
        if kind not in BLOCKS:
            raise ValueError("%s: tensor type %d of %s is not read by this program" % (self.path, kind, name))
        size = tensor_bytes(shape, kind)
        if self.data + offset + size > len(self.mapped):
            raise ValueError("%s: the bytes of %s lie beyond the end of the file" % (self.path, name))
        return np.frombuffer(self.mapped, dtype=np.uint8, count=size, offset=self.data + offset)

    def values(self, name):
        shape, kind, _ = self.tensors[name]
        return decode(self.raw(name), kind).reshape(shape)


# The largest relative distance a tensor of each format may leave to the weights of its checkpoint: several
# times what its rounding gives.  A conversion keeps in BF16 a tensor its format leaves beyond it, and the check
# of a file tells with it a tensor written from other values or in another layout, whose distance nears 1, and
# not the quality of a quantization.
VERIFY_BOUNDS = {F32: 0.0, F16: 0.002, BF16: 0.01, Q8_0: 0.03, Q6_K: 0.08, Q5_K: 0.12, Q4_K: 0.25}


def squared_distance(got, want):
    """The squared distance of two arrays of one shape and the squared size of the second, in double precision."""
    w = np.asarray(want).astype(np.float64)
    return float(((np.asarray(got).astype(np.float64) - w) ** 2).sum()), float((w ** 2).sum())


def distance_of(difference, size):
    """The relative distance these two sums give."""
    return math.sqrt(difference / size) if size > 0.0 else (0.0 if difference == 0.0 else float("inf"))


def encode_task(args):
    """
    The bytes of rows in a format, with the two sums of squared_distance() between the values these bytes
    give back and the rows, for the formats that may leave their values: (bytes, difference, size).
    """
    values, kind = args
    raw = encode(values, kind)
    if kind in (F32, BF16):
        return raw.tobytes(), 0.0, 0.0
    difference, size = squared_distance(decode(raw, kind).reshape(values.shape), values)
    return raw.tobytes(), difference, size


def write_gguf(path, metadata, tensors, jobs=1, log=None):
    """
    tensors: (name, shape, kind, source) in the order of the file, source() giving the values in
    single precision, or bytes already in the format of the file.  The shapes are those of numpy:
    GGUF stores their dimensions reversed.

    A quantized format holds its values through scales in half precision, which round to zero for
    faint values.  A tensor written from values that its format leaves beyond VERIFY_BOUNDS is
    written again in BF16, so that no tensor of the file leaves its weights: the formats and the
    offsets of the table of the tensors are those of the data once written.  Returns the tensors
    kept in BF16, as (name, format given up, relative distance it left).
    """
    def table(kinds, offsets):
        return b"".join(pack_string(name) + struct.pack("<I", len(shape))
                        + b"".join(struct.pack("<Q", n) for n in reversed(shape)) + struct.pack("<IQ", kind, offset)
                        for (name, shape, _, _), kind, offset in zip(tensors, kinds, offsets))

    kinds, offsets, kept = [kind for _, _, kind, _ in tensors], [0] * len(tensors), []
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) + pack_metadata(metadata)
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        with open(path + ".part", "wb") as out:
            out.write(head)
            # The room of the table, whose size depends neither on the formats nor on the offsets.
            out.write(table(kinds, offsets))
            out.write(b"\0" * (align(out.tell()) - out.tell()))
            base = out.tell()
            for index, (name, shape, kind, source) in enumerate(tensors):
                data = source()
                begin = out.tell()
                offsets[index] = begin - base
                if isinstance(data, (bytes, bytearray, memoryview)) or data.dtype == np.uint8:
                    out.write(data)
                else:
                    # Rows by chunks of about 4 million values; the workers quantize a few chunks ahead.
                    rows = data.reshape(-1, shape[-1]) if len(shape) else data.reshape(1, 1)
                    step = max(1, (1 << 22) // max(1, rows.shape[1]))
                    pending = deque()
                    difference = size = 0.0

                    def write(result):
                        nonlocal difference, size
                        out.write(result[0])
                        difference, size = difference + result[1], size + result[2]

                    for first in range(0, rows.shape[0], step):
                        chunk = (np.ascontiguousarray(rows[first:first + step], dtype=np.float32), kind)
                        if pool is None or kind in (F32, F16, BF16):
                            write(encode_task(chunk))
                            continue
                        pending.append(pool.submit(encode_task, chunk))
                        if len(pending) >= 2 * jobs:
                            write(pending.popleft().result())
                    while pending:
                        write(pending.popleft().result())
                    distance = distance_of(difference, size)
                    if kind not in (F32, BF16) and not distance <= VERIFY_BOUNDS[kind]:
                        out.seek(begin)
                        out.truncate()
                        for first in range(0, rows.shape[0], step):
                            out.write(encode(rows[first:first + step], BF16).tobytes())
                        kinds[index] = BF16
                        kept.append((name, kind, distance))
                if out.tell() - begin != tensor_bytes(shape, kinds[index]):
                    raise RuntimeError("%s: %d bytes written where its shape and format take %d"
                                       % (name, out.tell() - begin, tensor_bytes(shape, kinds[index])))
                written = out.tell() - base
                out.write(b"\0" * (align(written) - written))
                if log and (index % 50 == 0 or index + 1 == len(tensors)):
                    log("  %4d / %d tensors, %.1f GB, %.0f s" % (index + 1, len(tensors), out.tell() / 1e9,
                                                                    time.time() - start))
            out.seek(len(head))
            out.write(table(kinds, offsets))
    finally:
        if pool is not None:
            pool.shutdown()
    os.replace(path + ".part", path)
    return kept


# ---------------------------------------------------------------------------------------------
# Tokenizer (byte-pair encoding of the GPT-2 family, as tokenizer.json stores it)

def tokenizer_metadata(folder, vocab_size, pre):
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        spec = json.load(f)
    settings = {}
    path = os.path.join(folder, "tokenizer_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            settings = json.load(f)
    model = spec["model"]
    if model.get("type") != "BPE":
        raise ValueError("tokenizer.json holds a %s tokenizer; this program writes BPE ones" % model.get("type"))
    by_id = {index: token for token, index in model["vocab"].items()}
    added = {entry["id"]: entry for entry in spec.get("added_tokens", [])}
    for index, entry in added.items():
        by_id[index] = entry["content"]
    size = max(vocab_size, max(by_id) + 1)
    tokens, kinds = [], []
    for index in range(size):
        if index not in by_id:
            tokens.append("[PAD%d]" % index)
            kinds.append(5)                         # unused
        elif index in added:
            # A control token is one the tokenizer marks as special, or one written <|name|>, which
            # some tokenizers leave unmarked; the other added tokens are defined by the user.
            text = by_id[index]
            control = added[index].get("special") or (text.startswith("<|") and text.endswith("|>"))
            tokens.append(text)
            kinds.append(3 if control else 4)
        else:
            tokens.append(by_id[index])
            kinds.append(1)                         # normal
    merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", [])]
    ids = {token: index for index, token in enumerate(tokens)}

    def token_id(key):
        value = settings.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        return ids.get(value) if value is not None else None

    items = [("tokenizer.ggml.model", "str", "gpt2"), ("tokenizer.ggml.pre", "str", pre),
             ("tokenizer.ggml.tokens", ("array", "str"), tokens),
             ("tokenizer.ggml.token_type", ("array", "i32"), kinds),
             ("tokenizer.ggml.merges", ("array", "str"), merges)]
    for key, name in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id"), ("unk_token", "unknown_token_id"),
                      ("pad_token", "padding_token_id")):
        index = token_id(key)
        if index is not None:
            items.append(("tokenizer.ggml." + name, "u32", index))
    items.append(("tokenizer.ggml.add_bos_token", "bool", bool(settings.get("add_bos_token", False))))
    items.append(("tokenizer.ggml.add_eos_token", "bool", bool(settings.get("add_eos_token", False))))
    template = os.path.join(folder, "chat_template.jinja")
    if os.path.exists(template):
        items.append(("tokenizer.chat_template", "str", open(template, encoding="utf-8").read()))
    elif isinstance(settings.get("chat_template"), str):
        items.append(("tokenizer.chat_template", "str", settings["chat_template"]))
    return items


# ---------------------------------------------------------------------------------------------
# Architectures

def use_more_bits(layer, n_layers):
    """The layers whose down projections keep more bits in the _M mixes of the k-quants (llama.cpp)."""
    return layer < n_layers // 8 or layer >= 7 * n_layers // 8 or (layer - n_layers // 8) % 3 == 2


class Olala:
    """
    Olala 1.0 mini of OVHai LLM (model_type olala), in the GGUF container of its authors.

    Layers M are Mamba-3 layers with several inputs and outputs, layers V differential attentions
    whose keys and values are tensor products; every block ends with a mixture of experts in a
    latent space beside a shared expert, and turns the residual stream on its sphere.
    """

    name = "olala"
    PUBLISHED = {
        "normalize_embeddings_ngpt": True, "normalize_embeddings": False, "use_completed_p": True,
        "layer_norm_scaling": False, "gate_act": "silu", "xsa": False, "complete_slw": False,
        "token_shift_attn": True, "scalable_softmax": True, "qk_norm": True, "gate_attn": True,
        "mamba3_add_trapezoid": True, "mamba3_rope": True, "mamba3_remove_conv": True,
        "mamba3_postgate_norm": False, "mamba3_remove_BC_bias": False, "moe_shared_expert_gate": False,
    }
    LAYER = {
        "mixer_proj.weight": "attn_output.weight", "gate_proj.weight": "attn_gate.weight",
        "geodesic_mixer.scale": "geo_mixer_scale", "geodesic_mixer.bias": "geo_mixer_bias",
        "geodesic_mlp.scale": "geo_mlp_scale", "geodesic_mlp.bias": "geo_mlp_bias",
        "mixer.in_proj.weight": "ssm_in.weight", "mixer.in_proj_dyn.weight": "ssm_in_dyn.weight",
        "mixer.B_norm.norm.weight": "ssm_b_norm.weight", "mixer.C_norm.norm.weight": "ssm_c_norm.weight",
        "mixer.B_bias": "ssm_b_bias", "mixer.C_bias": "ssm_c_bias", "mixer.dt_bias": "ssm_dt_bias",
        "mixer.D": "ssm_d", "mixer.in_proj_mimo_x": "ssm_mimo_x", "mixer.in_proj_mimo_z": "ssm_mimo_z",
        "mixer.out_proj_mimo": "ssm_mimo_o",
        "mixer.c_q.weight": "attn_q.weight", "mixer.W_A_k.weight": "attn_wa_k.weight",
        "mixer.W_A_v.weight": "attn_wa_v.weight", "mixer.W_B_k.weight": "attn_wb_k.weight",
        "mixer.W_B_v.weight": "attn_wb_v.weight", "mixer.q_norm.norm.weight": "attn_q_norm.weight",
        "mixer.k_norm.norm.weight": "attn_k_norm.weight", "mixer.shift_proj_k.weight": "attn_shift_k.weight",
        "mixer.shift_proj_v.weight": "attn_shift_v.weight", "mixer.lambda_proj.weight": "attn_lambda.weight",
        "mixer.softmax_scaler": "attn_softmax_scaler",
        "mlp.moe_gate.weight": "ffn_gate_inp.weight", "mlp.expert_bias": "exp_probs_b",
        "mlp.down_proj.weight": "ffn_latent_down.weight", "mlp.up_proj.weight": "ffn_latent_up.weight",
        "mlp.experts.experts.weight": "ffn_up_exps.weight",
        "mlp.experts.output_experts.weight": "ffn_down_exps.weight",
        "mlp.shared_experts.fc_1.weight": "ffn_up_shexp.weight",
        "mlp.shared_experts.fc_2.weight": "ffn_down_shexp.weight",
    }
    GLOBAL = {"model.embedding.weight": "token_embd.weight", "lm_head.weight": "output.weight"}
    DROPPED = ("geodesic_mixer.prosres_scalar", "geodesic_mlp.prosres_scalar")
    # The normalization weights are stored minus one (zero_centered_gamma): the files hold them whole.
    FOLDED = ("ssm_b_norm.weight", "ssm_c_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")
    # Kept in single precision whatever the format: the router of the experts reads small differences.
    SINGLE = ("ffn_gate_inp.weight",)
    # The packed copies of the constants of a Mamba-3 layer that the kernels of the authors read:
    # name: (first, second, axis of the concatenation, or None for a stack of two rows).
    PACKS = {"ssm_m3_bias": ("ssm_b_bias", "ssm_c_bias", 1), "ssm_m3_mxz": ("ssm_mimo_x", "ssm_mimo_z", 1),
             "ssm_m3_norms": ("ssm_b_norm.weight", "ssm_c_norm.weight", None),
             "ssm_m3_misc": ("ssm_dt_bias", "ssm_d", None)}
    """
    What the authors keep above the base format (docs/olala.md of their llama.cpp): the tensors of
    384 columns cannot take the k-quants, whose blocks hold 256 values; the inputs of the output
    projection of a mixer and of the down projections reach 1e9 to 1e14, which the 8-bit blocks of
    32 values read badly.
    """
    RECIPES = {
        "q8_0": {"attn_output.weight": BF16, "ffn_down_exps.weight": Q6_K, "ffn_down_shexp.weight": BF16,
                 "ffn_latent_up.weight": BF16},
        "q6_k": {"ffn_latent_up.weight": BF16, "ffn_up_exps.weight": Q8_0},
        "q5_k_m": {"ffn_latent_up.weight": BF16, "ffn_up_exps.weight": Q8_0, "output.weight": Q6_K},
        "q4_k_m": {"ffn_latent_up.weight": BF16, "ffn_up_exps.weight": Q8_0, "output.weight": Q6_K},
    }
    MORE_BITS = ("ffn_down_exps.weight", "ffn_down_shexp.weight")

    def __init__(self, checkpoint):
        self.ck = checkpoint
        c = checkpoint.config
        self.layers = c["layers_config"]
        if len(self.layers) != c["num_hidden_layers"] or set(self.layers) - set("MV"):
            raise ValueError("layers_config %r is not %d layers of the types M and V"
                             % (self.layers, c["num_hidden_layers"]))
        if not c.get("geodesic_update") or not c.get("moe"):
            raise ValueError("this program converts the Olala models with geodesic updates and experts")
        if c.get("final_norm") or c.get("rope_theta", 0.0) > 0.0 or c.get("use_value_embedding"):
            raise ValueError("a final normalization, rotary positions or value embeddings are not converted")
        # The settings that change the computation of a layer and leave its tensors as they are: the
        # engines compute the published variant, and a file of another one would give other answers.
        for key, published in self.PUBLISHED.items():
            if key in c and c[key] != published:
                raise ValueError("config.json sets %s to %r where the published models have %r: this variant "
                                 "is not converted" % (key, c[key], published))
        self.folded = bool(c.get("zero_centered_gamma"))

    def metadata(self):
        c, a = self.ck.config, "olala."
        heads = c["num_attention_heads"]
        signal = c.get("num_signal_heads_diff") or heads // 2
        # The Mamba-3 layers of the authors rotate half of the state, by pairs.
        rotated = int(c["mamba_d_state"] * 0.5)
        rotated -= rotated % 2
        items = [
            (a + "block_count", "u32", c["num_hidden_layers"]),
            (a + "context_length", "u32", c["max_position_embeddings"]),
            (a + "embedding_length", "u32", c["hidden_size"]),
            (a + "feed_forward_length", "u32", 0),
            (a + "attention.head_count", "u32", heads),
            (a + "attention.head_count_kv", "u32", heads - signal),
            (a + "attention.key_length", "u32", c["head_dim"]),
            (a + "attention.value_length", "u32", c["head_dim"]),
            (a + "attention.layer_norm_rms_epsilon", "f32", c["norm_epsilon"]),
            (a + "attention.sliding_window", "u32", c["slw_wsize"]),
        ]
        if c.get("softcap_attn", 0.0) > 0.0:
            items.append((a + "attn_logit_softcapping", "f32", c["softcap_attn"]))
        items += [
            (a + "expert_count", "u32", c["moe_num_routed_experts"]),
            (a + "expert_used_count", "u32", c["moe_num_active_experts"]),
            (a + "expert_feed_forward_length", "u32", c["moe_routed_intermediate_size"]),
        ]
        if c.get("moe_shared_intermediate_size", 0) > 0:
            items += [(a + "expert_shared_feed_forward_length", "u32", c["moe_shared_intermediate_size"]),
                      (a + "expert_shared_count", "u32", 1)]
        items += [(a + "expert_weights_scale", "f32", c["moe_routed_scaling_factor"]),
                  (a + "expert_gating_func", "u32", 2)]                    # sigmoid
        if c.get("moe_routed_input_dim", 0) > 0:
            items.append((a + "moe_latent_size", "u32", c["moe_routed_input_dim"]))
        items += [
            (a + "ssm.inner_size", "u32", 2 * c["hidden_size"]),
            (a + "ssm.state_size", "u32", c["mamba_d_state"]),
            (a + "ssm.group_count", "u32", c["mamba_ngroups"]),
            (a + "ssm.time_step_rank", "u32", c["mamba_headdim"]),
            (a + "mamba_mimo_dim", "u32", c["mamba_mimo_dim"]),
            (a + "mamba_headdim", "u32", c["mamba_headdim"]),
            (a + "num_signal_heads", "u32", signal),
            (a + "num_noise_heads", "u32", heads - signal),
            (a + "tpa_rank", "u32", c["tpa_rank"]),
            (a + "num_rope_angles", "u32", rotated // 2),
            (a + "layer_types", "str", self.layers),
            (a + "zero_centered_gamma", "bool", self.folded),
            (a + "gate_bias", "f32", 1.15 if c.get("zero_centered_gate") else 0.0),
            (a + "gate_attn", "bool", bool(c.get("gate_attn"))),
            (a + "gate_gdn", "bool", bool(c.get("gate_gdn"))),
            (a + "geodesic_update", "bool", True),
            (a + "final_norm", "bool", False),
        ]
        return items

    def tokenizer_pre(self):
        return "qwen2"

    def kind(self, target, short, layer, source_dtype, shape):
        """The format of a tensor of the file for a --type."""
        # The weight matrices alone take the format of the file.  Biases, scalars and the projections of
        # the several inputs and outputs stay in single precision, whichever type a checkpoint stores
        # them in: the fine-tuned models of the family hold them in BF16.
        if (len(shape) < 2 or source_dtype == "F32" or not short.endswith(".weight") or short in self.SINGLE
                or short in self.FOLDED):
            return F32
        kind = BASE_TYPES[target]
        recipe = self.RECIPES.get(target, {})
        if short in recipe:
            kind = recipe[short]
        elif target in ("q5_k_m", "q4_k_m") and short in self.MORE_BITS and use_more_bits(layer, len(self.layers)):
            kind = Q6_K
        return fitting(kind, shape)

    def plan(self, target):
        """(name, shape, kind, source) of every tensor, in the order of the quantized files of the authors."""
        ck = self.ck
        per_layer = {}
        globals_ = {}
        for name in ck.tensors:
            if name in self.GLOBAL:
                globals_[self.GLOBAL[name]] = name
                continue
            parts = name.split(".", 3)
            if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers" or not parts[2].isdigit():
                raise ValueError("unexpected tensor in the checkpoint: %s" % name)
            if parts[3] in self.DROPPED:
                continue
            if parts[3] not in self.LAYER:
                raise ValueError("unexpected tensor in the checkpoint: %s" % name)
            per_layer.setdefault(int(parts[2]), {})[self.LAYER[parts[3]]] = name
        if sorted(per_layer) != list(range(len(self.layers))) or len(globals_) != 2:
            raise ValueError("the checkpoint does not hold the embeddings, the head and %d layers" % len(self.layers))
        out = []

        def add(base, short, layer, source):
            shape = ck.shape(source) or (1,)
            kind = self.kind(target, short, layer, ck.dtype(source), shape)
            out.append((base, tuple(shape), kind, self.reader(source, short, kind, shape)))

        for base in sorted(globals_):
            add(base, base, -1, globals_[base])
        for layer in range(len(self.layers)):
            names = dict(per_layer[layer])
            prefix = "blk.%d." % layer
            packs = {}
            if self.layers[layer] == "M":
                for packed, (first, second, axis) in self.PACKS.items():
                    a, b = ck.shape(names[first]), ck.shape(names[second])
                    shape = (a[:axis] + (a[axis] + b[axis],) + a[axis + 1:]) if axis is not None \
                        else (2, int(np.prod(a)))
                    packs[packed] = (shape, self.packer(names[first], first, names[second], second, axis))
            for short in sorted(set(names) | set(packs)):
                if short in packs:
                    out.append((prefix + short, packs[short][0], F32, packs[short][1]))
                else:
                    add(prefix + short, short, layer, names[short])
        return out

    def constant(self, source, short):
        """A tensor of the checkpoint in single precision, as the file holds it."""
        values = self.ck.values(source)
        if self.folded and short in self.FOLDED:
            values = (1.0 + values).astype(np.float32)
        return values.reshape(-1) if values.ndim == 0 else values

    def reader(self, source, short, kind, shape):
        def read():
            if kind == BF16 and self.ck.dtype(source) == "BF16":
                # The bytes of the checkpoint, unchanged.
                return np.ascontiguousarray(self.ck.raw(source)).view(np.uint8).reshape(-1)
            return self.constant(source, short).reshape(shape)
        return read

    def packer(self, first_source, first, second_source, second, axis):
        def read():
            a, b = self.constant(first_source, first), self.constant(second_source, second)
            if axis is None:
                return np.stack([a.reshape(-1), b.reshape(-1)])
            return np.concatenate([a, b], axis=axis)
        return read

ARCHITECTURES = {"olala": Olala}


def architecture_of(checkpoint):
    kind = checkpoint.config.get("model_type")
    if kind not in ARCHITECTURES:
        raise ValueError("model_type %r is not converted by this program (known: %s)"
                         % (kind, ", ".join(sorted(ARCHITECTURES))))
    return ARCHITECTURES[kind](checkpoint)


def rounded_count(n):
    """A count of parameters with two digits: 1.4B, 21M, 350K."""
    for limit, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if n > limit:
            scaled = n / limit
            break
    else:
        scaled, unit = n / 1e3, "K"
    digits = max(2 - len(str(round(scaled)).lstrip("0")), 0)
    return "%.*f%s" % (digits, scaled, unit)


def size_label(tensors):
    """
    The size label of llama.cpp: the parameters one token reads, the experts counted once, after the
    number of experts when the model has some.
    """
    shared = per_expert = experts = stacks = 0
    for name, shape, _, _ in tensors:
        size = int(np.prod(shape))
        if "_exps." in name and len(shape) >= 3:
            per_expert += size // shape[-3]
            experts += shape[-3]
            stacks += 1
        else:
            shared += size
    if stacks:
        return "%dx%s" % (experts // stacks, rounded_count(shared + per_expert))
    return rounded_count(shared)


def general_metadata(arch, name, tensors, extra):
    items = [("general.architecture", "str", arch.name), ("general.type", "str", "model"),
             ("general.name", "str", name), ("general.size_label", "str", size_label(tensors))]
    for key in ("license", "author", "url", "source.url", "description"):
        if extra.get(key):
            items.append(("general." + key, "str", extra[key]))
    return items


def build(checkpoint, target, name, extra):
    arch = architecture_of(checkpoint)
    tensors = arch.plan(target)
    metadata = general_metadata(arch, name, tensors, extra) + arch.metadata()
    metadata += tokenizer_metadata(checkpoint.folder, checkpoint.config.get("vocab_size", 0), arch.tokenizer_pre())
    metadata += [("general.quantization_version", "u32", 2), ("general.file_type", "u32", FILE_TYPES[target])]
    return metadata, tensors


# ---------------------------------------------------------------------------------------------
# Commands

HUB_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "LICENSE", "NOTICE")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url, path, token):
    """Downloads url to path, from the byte where an earlier download stopped."""
    have = os.path.getsize(path + ".part") if os.path.exists(path + ".part") else 0
    request = urllib.request.Request(url, headers={"User-Agent": "slm_safetensors_to_gguf"})
    if token:
        # The token goes to the hub alone, not to the storage the hub redirects to.
        request.add_unredirected_header("Authorization", "Bearer " + token)
    if have:
        request.add_header("Range", "bytes=%d-" % have)
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            if have and reply.status != 206:
                have = 0
            total = have + int(reply.headers.get("Content-Length", 0))
            with open(path + ".part", "ab" if have else "wb") as out:
                done, shown = have, time.time()
                while True:
                    block = reply.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if time.time() - shown > 5:
                        shown = time.time()
                        print("  %s: %.1f of %.1f GB" % (os.path.basename(path), done / 1e9, total / 1e9), flush=True)
            # A transfer that ends before its length keeps its bytes for the next attempt.
            if done < total:
                raise OSError("the transfer of %s ended at %d of %d bytes" % (os.path.basename(path), done, total))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        if e.code == 416 and have:
            pass                                    # the file was whole already
        else:
            raise
    os.replace(path + ".part", path)
    return True


def command_fetch(args):
    os.makedirs(args.folder, exist_ok=True)
    base = "https://huggingface.co/%s/resolve/%s/" % (args.repository, args.revision)
    token = os.environ.get("HF_TOKEN", "")
    listing = urllib.request.Request("https://huggingface.co/api/models/%s/tree/%s" % (args.repository, args.revision),
                                     headers={"User-Agent": "slm_safetensors_to_gguf"})
    if token:
        listing.add_unredirected_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(listing, timeout=60) as reply:
        entries = json.load(reply)
    if not isinstance(entries, list):
        raise ValueError("the hub lists no files for %s" % args.repository)
    # The sizes of the files and, for the large ones, the SHA-256 the hub gives them.
    files, hashes = {}, {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "file" and isinstance(entry.get("path"), str):
            files[entry["path"]] = entry.get("size", 0)
            oid = (entry.get("lfs") or {}).get("oid")
            if isinstance(oid, str):
                hashes[entry["path"]] = oid.split(":")[-1].lower()
    # The files of the root of the repository: a name with a folder in it is never written.
    weights = sorted(f for f in files if f.endswith(".safetensors") and os.path.basename(f) == f and "\\" not in f)
    wanted = [f for f in HUB_FILES if f in files] + weights
    if not weights:
        print("no .safetensors file at the root of %s" % args.repository)
        return 1
    for name in wanted:
        path = os.path.join(args.folder, name)
        if os.path.exists(path) and os.path.getsize(path) == files[name]:
            print("%s: present" % name)
            continue
        print("%s: %.2f GB" % (name, files[name] / 1e9), flush=True)
        # A transfer that the network cuts goes on from the bytes received, a few times.
        for attempt in range(5):
            try:
                fetch_file(base + name, path, token)
                break
            except (OSError, http.client.HTTPException) as e:
                if attempt == 4 or (isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404)):
                    raise
                print("  %s: %s; going on from the bytes received" % (name, e), flush=True)
                time.sleep(5 * (attempt + 1))
        if not os.path.exists(path) or os.path.getsize(path) != files[name]:
            print("%s: %d bytes received where the hub lists %d; the file is removed"
                  % (name, os.path.getsize(path) if os.path.exists(path) else 0, files[name]))
            if os.path.exists(path):
                os.remove(path)
            return 1
        # A file resumed from the bytes of another revision has the size of its file and another content.
        if name in hashes and file_sha256(path) != hashes[name]:
            print("%s: its SHA-256 is not the one the hub lists; the file is removed" % name)
            os.remove(path)
            return 1
    print("checkpoint in %s" % args.folder)
    return 0


def describe(tensors):
    sizes = {}
    for name, shape, kind, _ in tensors:
        sizes[kind] = sizes.get(kind, 0) + tensor_bytes(shape, kind)
    return ", ".join("%s %.2f GB" % (TYPE_NAMES[k], v / 1e9) for k, v in sorted(sizes.items(), key=lambda i: -i[1]))


def default_name(folder):
    return os.path.basename(os.path.abspath(folder))


def command_plan(args):
    ck = Checkpoint(args.checkpoint)
    for target in args.type or ["bf16"]:
        metadata, tensors = build(ck, target, args.name or default_name(args.checkpoint), {})
        print("%s: %d tensors, %s" % (target, len(tensors), describe(tensors)))
        if args.list:
            for name, shape, kind, _ in tensors:
                print("  %-44s %-5s %s" % (name, TYPE_NAMES[kind], list(reversed(shape))))
    return 0


def command_convert(args):
    ck = Checkpoint(args.checkpoint)
    name = args.name or default_name(args.checkpoint)
    extra = {"license": args.license, "author": args.author, "url": args.url, "source.url": args.source_url}
    os.makedirs(args.out, exist_ok=True)
    for target in args.type or ["bf16"]:
        metadata, tensors = build(ck, target, name, extra)
        path = os.path.join(args.out, "%s-%s.gguf" % (name, target))
        print("%s: %d tensors, %s" % (path, len(tensors), describe(tensors)), flush=True)
        kept = write_gguf(path, metadata, tensors, jobs=args.jobs, log=lambda line: print(line, flush=True))
        for tensor, kind, distance in kept:
            print("  %s: kept in BF16, %s left its values at a relative distance of %.4f"
                  % (tensor, TYPE_NAMES[kind], distance), flush=True)
        print("%s: %d bytes, sha256 %s" % (path, os.path.getsize(path), file_sha256(path)), flush=True)
    # The license and the notices of the checkpoint follow the files made from it.
    for notice in ("LICENSE", "NOTICE"):
        source, copy = os.path.join(args.checkpoint, notice), os.path.join(args.out, notice)
        if os.path.isfile(source) and os.path.abspath(source) != os.path.abspath(copy):
            shutil.copyfile(source, copy)
            print("%s: copied from the checkpoint" % copy, flush=True)
    return 0


def command_inspect(args):
    g = GGUFFile(args.file)
    print("%s: %d metadata values, %d tensors" % (args.file, len(g.metadata), len(g.tensors)))
    for key, value in g.metadata.items():
        if isinstance(value, list):
            print("  %s = [%d] %s" % (key, len(value), repr(value[:4])[:100]))
        else:
            print("  %s = %s" % (key, repr(value)[:160]))
    if args.list:
        for name in g.order:
            shape, kind, _ = g.tensors[name]
            print("  %-44s %-5s %s" % (name, TYPE_NAMES.get(kind, str(kind)), list(reversed(shape))))
    return 0


def command_compare(args):
    a, b = GGUFFile(args.first), GGUFFile(args.second)
    differences = 0
    skipped = set(args.ignore or [])
    for key in sorted(set(a.metadata) | set(b.metadata)):
        if key in skipped:
            continue
        if key not in a.metadata or key not in b.metadata:
            print("metadata %s: only in %s" % (key, args.first if key in a.metadata else args.second))
            differences += 1
        elif a.metadata[key] != b.metadata[key]:
            close = isinstance(a.metadata[key], float) and math.isclose(a.metadata[key], b.metadata[key], rel_tol=1e-6)
            if not close:
                print("metadata %s: %s | %s" % (key, repr(a.metadata[key])[:60], repr(b.metadata[key])[:60]))
                differences += 1
    same = close = 0
    worst = (0.0, "")
    for name in sorted(set(a.tensors) | set(b.tensors)):
        if name not in a.tensors or name not in b.tensors:
            print("tensor %s: only in %s" % (name, args.first if name in a.tensors else args.second))
            differences += 1
            continue
        (sa, ka, _), (sb, kb, _) = a.tensors[name], b.tensors[name]
        if sa != sb:
            print("tensor %s: shapes %s | %s" % (name, list(sa), list(sb)))
            differences += 1
            continue
        if ka == kb and np.array_equal(a.raw(name), b.raw(name)):
            same += 1
            continue
        # By values: the distance of the two tensors against the size of the second.
        distance = relative_distance(a.values(name), b.values(name))
        close += 1
        if distance > worst[0]:
            worst = (distance, name)
        if distance > args.tolerance:
            print("tensor %s (%s | %s): relative distance %.4f" % (name, TYPE_NAMES.get(ka), TYPE_NAMES.get(kb),
                                                                   distance))
            differences += 1
    print("%d tensors with the same bytes, %d compared by values (largest relative distance %.4f, %s)"
          % (same, close, worst[0], worst[1] or "none"))
    print("no difference" if differences == 0 else "%d differences" % differences)
    return 0 if differences == 0 else 1


def relative_distance(got, want):
    """The distance of two arrays of one shape against the size of the second, by slices of rows."""
    rows = want.shape[0] if want.ndim > 1 else 1
    got, want = got.reshape(rows, -1), want.reshape(rows, -1)
    difference = size = 0.0
    step = max(1, (1 << 22) // max(1, want.shape[1]))
    for at in range(0, rows, step):
        d, s = squared_distance(got[at:at + step], want[at:at + step])
        difference, size = difference + d, size + s
    return distance_of(difference, size)


def verify_tensors(ck, g):
    """
    Sets the tensors of the GGUF file g against the values the checkpoint ck gives them in single
    precision.  Returns the lines that tell the tensors that fail, and for each format of the file
    the number of its tensors, their largest distance and the tensor that has it.
    """
    _, truth = build(ck, "f32", default_name(ck.folder), {})
    problems, seen, worst = [], set(), {}
    for name, shape, kind, source in truth:
        seen.add(name)
        if name not in g.tensors:
            problems.append("tensor %s: not in the file" % name)
            continue
        got_shape, got_kind, _ = g.tensors[name]
        if tuple(got_shape) != tuple(shape):
            problems.append("tensor %s: shape %s where the checkpoint gives %s" % (name, list(got_shape), list(shape)))
            continue
        want = source()
        if not isinstance(want, np.ndarray) or want.dtype == np.uint8:
            want = decode(np.frombuffer(bytes(want), dtype=np.uint8), kind)
        want = np.asarray(want, dtype=np.float32).reshape(shape)
        distance = relative_distance(g.values(name), want)
        count, largest, where = worst.get(got_kind, (0, 0.0, name))
        worst[got_kind] = (count + 1, max(largest, distance), name if distance > largest else where)
        if not distance <= VERIFY_BOUNDS.get(got_kind, 0.0):
            problems.append("tensor %s (%s): relative distance %.4f to the weights of the checkpoint"
                            % (name, TYPE_NAMES.get(got_kind, got_kind), distance))
    for name in sorted(set(g.tensors) - seen):
        problems.append("tensor %s: not among those the checkpoint gives" % name)
    return problems, worst


def command_verify(args):
    problems, worst = verify_tensors(Checkpoint(args.checkpoint), GGUFFile(args.file))
    for line in problems:
        print(line)
    for kind, (count, largest, where) in sorted(worst.items()):
        print("%-5s %4d tensors, largest relative distance %.4f (%s), at most %.4f"
              % (TYPE_NAMES.get(kind, kind), count, largest, where, VERIFY_BOUNDS.get(kind, 0.0)))
    print("every tensor follows the checkpoint" if not problems else "%d tensors FAILED" % len(problems))
    return 0 if not problems else 1


def tiny_olala(folder, seed=1, small=None, faint=()):
    """
    Writes a small random checkpoint with the tensors and the settings of an Olala model.  small is
    the type of the biases and of the other small tensors: single precision as Olala 1.0 mini stores
    them, or "BF16" as the models fine-tuned from it do.  faint names the matrices written with
    values 10,000 times smaller, as the layers of Olala 1.0 mini whose routed experts faded hold.
    """
    rng = np.random.default_rng(seed)
    # Rows of 256 values or of a multiple, as the blocks of the k-quants ask.
    d, hd, heads, signal, rank = 256, 64, 16, 12, 2
    ssm_head, d_state, mimo, experts, latent = 32, 32, 2, 8, 64
    routed, shared = 256, 256
    layers = "MMVMV"
    config = {
        "model_type": "olala", "layers_config": layers, "num_hidden_layers": len(layers), "hidden_size": d,
        "vocab_size": 300, "max_position_embeddings": 512, "num_attention_heads": heads,
        "num_signal_heads_diff": signal, "head_dim": hd, "tpa_rank": rank, "norm_epsilon": 1e-6, "slw_wsize": 64,
        "softcap_attn": 150.0, "moe": True, "moe_num_routed_experts": experts, "moe_num_active_experts": 3,
        "moe_routed_intermediate_size": routed, "moe_shared_intermediate_size": shared,
        "moe_routed_scaling_factor": 2.45,
        "moe_routed_input_dim": latent, "mamba_d_state": d_state, "mamba_ngroups": 1, "mamba_headdim": ssm_head,
        "mamba_mimo_dim": mimo, "zero_centered_gamma": True, "zero_centered_gate": True, "gate_attn": True,
        "gate_gdn": True, "geodesic_update": True, "final_norm": False, "rope_theta": 0.0,
        "use_value_embedding": False,
    }
    inner = 2 * d
    ssm_heads = inner // ssm_head
    kv = heads - signal
    angles = (d_state // 2 - (d_state // 2) % 2) // 2
    tensors = {}

    def bf16(name, *shape, scale=0.2):
        if name in faint:
            scale *= 1e-4
        tensors[name] = ("BF16", shape, f32_to_bf16((scale * rng.standard_normal(shape)).astype(np.float32)))

    def f32(name, *shape, scale=0.2, around=0.0):
        values = (around + scale * rng.standard_normal(shape)).astype(np.float32)
        # The router keeps single precision in every checkpoint of the family.
        if small == "BF16" and not name.endswith("moe_gate.weight"):
            tensors[name] = ("BF16", shape, f32_to_bf16(values))
        else:
            tensors[name] = ("F32", shape, values)

    bf16("model.embedding.weight", 300, d, scale=0.5)
    bf16("lm_head.weight", 300, d, scale=0.5)
    for i, kind in enumerate(layers):
        p = "model.layers.%d." % i
        for part in ("geodesic_mixer", "geodesic_mlp"):
            f32(p + part + ".scale", scale=0.1, around=1.0)
            f32(p + part + ".bias", scale=0.02)
            f32(p + part + ".prosres_scalar", scale=0.0, around=1.0)
        if kind == "M":
            bf16(p + "mixer.in_proj.weight", ssm_heads * (2 * ssm_head + 3), d)
            bf16(p + "mixer.in_proj_dyn.weight", 2 * mimo * d_state + angles, d)
            bf16(p + "mixer_proj.weight", d, inner)
            bf16(p + "mixer.B_norm.norm.weight", d_state, scale=0.1)
            bf16(p + "mixer.C_norm.norm.weight", d_state, scale=0.1)
            f32(p + "mixer.B_bias", ssm_heads, mimo, d_state, scale=0.3, around=1.0)
            f32(p + "mixer.C_bias", ssm_heads, mimo, d_state, scale=0.3, around=1.0)
            f32(p + "mixer.dt_bias", ssm_heads, scale=0.5, around=-2.0)
            f32(p + "mixer.D", ssm_heads, scale=0.3, around=1.0)
            f32(p + "mixer.in_proj_mimo_x", ssm_heads, mimo, ssm_head, scale=0.1, around=1.0 / mimo)
            f32(p + "mixer.in_proj_mimo_z", ssm_heads, mimo, ssm_head, scale=0.3, around=1.0)
            f32(p + "mixer.out_proj_mimo", ssm_heads, mimo, ssm_head, scale=0.1, around=1.0 / mimo)
        else:
            bf16(p + "mixer.c_q.weight", heads * hd, d)
            for which in "kv":
                bf16(p + "mixer.W_A_%s.weight" % which, kv * rank, d, scale=0.5)
                bf16(p + "mixer.W_B_%s.weight" % which, rank * hd, d, scale=0.5)
                bf16(p + "mixer.shift_proj_%s.weight" % which, kv, d, scale=0.5)
            bf16(p + "mixer.q_norm.norm.weight", hd, scale=0.1)
            bf16(p + "mixer.k_norm.norm.weight", hd, scale=0.1)
            bf16(p + "mixer.lambda_proj.weight", kv, d, scale=0.5)
            f32(p + "mixer.softmax_scaler", heads, scale=0.2, around=1.0)
            bf16(p + "gate_proj.weight", signal * hd, d)
            bf16(p + "mixer_proj.weight", d, signal * hd)
        f32(p + "mlp.moe_gate.weight", experts, d, scale=1.0)
        f32(p + "mlp.expert_bias", experts, scale=0.05)
        bf16(p + "mlp.down_proj.weight", latent, d)
        bf16(p + "mlp.up_proj.weight", d, latent)
        bf16(p + "mlp.experts.experts.weight", experts, routed, latent)
        bf16(p + "mlp.experts.output_experts.weight", experts, latent, routed)
        bf16(p + "mlp.shared_experts.fc_1.weight", shared, d)
        bf16(p + "mlp.shared_experts.fc_2.weight", d, shared)
    os.makedirs(folder, exist_ok=True)
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + data.nbytes]}
        offset += data.nbytes
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(os.path.join(folder, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        for dtype, shape, data in tensors.values():
            f.write(data.tobytes())
    json.dump(config, open(os.path.join(folder, "config.json"), "w"), indent=1)
    vocab = {chr(0x100 + i) if i < 33 or 126 < i < 161 or i == 173 else chr(i): i for i in range(256)}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|channel_start|>", "<|channel_end|>", "<|content|>"]
    added = [{"id": 256 + i, "content": s, "special": True} for i, s in enumerate(specials)]
    json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added},
              open(os.path.join(folder, "tokenizer.json"), "w"))
    json.dump({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None},
              open(os.path.join(folder, "tokenizer_config.json"), "w"))
    open(os.path.join(folder, "chat_template.jinja"), "w").write(
        "{% for m in messages %}<|im_start|>{{ m.role }}<|channel_start|>final<|content|>{{ m.content }}"
        "<|channel_end|><|im_end|>{% endfor %}")
    return config


def command_selftest(args):
    failures = 0

    def check(ok, what):
        nonlocal failures
        if not ok:
            failures += 1
        print("%s  %s" % ("ok    " if ok else "FAILED", what))

    # Rows of several sizes, with a run of zeros and a row of zeros; then the same with one large value.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 512)).astype(np.float32) * np.exp(rng.standard_normal((64, 1))).astype(np.float32)
    x[3, 40:90] = 0.0
    x[5] = 0.0
    spiked = x.copy()
    spiked[7, 17] = 300.0
    for kind, bound in ((Q8_0, 0.01), (Q6_K, 0.025), (Q5_K, 0.045), (Q4_K, 0.09), (BF16, 0.004), (F16, 0.0005)):
        raw = encode(x, kind)
        back = decode(raw, kind).reshape(x.shape)
        error = float(np.sqrt(((back - x) ** 2).mean() / (x ** 2).mean()))
        large = decode(encode(spiked, kind), kind).reshape(x.shape)
        check(raw.size == tensor_bytes(x.shape, kind) and error <= bound and np.isfinite(large).all()
              and abs(large[7, 17] - 300.0) <= 12.0 and not back[5].any(),
              "%-5s %d bytes, relative error %.4f (at most %.4f)" % (TYPE_NAMES[kind], raw.size, error, bound))
    check(np.array_equal(bf16_to_f32(f32_to_bf16(bf16_to_f32(f32_to_bf16(x)))), bf16_to_f32(f32_to_bf16(x))),
          "BF16 values keep their bits")
    # The small model lies in the folder given, where it stays, or in a temporary one, removed at the end.
    folder = args.folder or tempfile.mkdtemp(prefix="slm-selftest-")
    try:
        config = tiny_olala(os.path.join(folder, "checkpoint"))
        ck = Checkpoint(os.path.join(folder, "checkpoint"))
        for target in ("bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"):
            metadata, tensors = build(ck, target, "tiny-olala", {"license": "apache-2.0"})
            path = os.path.join(folder, "tiny-olala-%s.gguf" % target)
            write_gguf(path, metadata, tensors, jobs=1)
            g = GGUFFile(path)
            worst, exact = 0.0, True
            for name, shape, kind, source in tensors:
                want = source()
                if not isinstance(want, np.ndarray) or want.dtype == np.uint8:
                    want = decode(np.frombuffer(bytes(want), dtype=np.uint8), kind)
                want = np.asarray(want, dtype=np.float32).reshape(shape)
                got = g.values(name)
                scale = float(np.sqrt((want.astype(np.float64) ** 2).mean())) or 1.0
                distance = float(np.sqrt(((got - want).astype(np.float64) ** 2).mean())) / scale
                worst = max(worst, distance)
                exact = exact and (kind not in (F32, BF16) or distance == 0.0)
            layers = len(config["layers_config"])
            check(len(g.tensors) == len(tensors) and exact and worst <= 0.1
                  and g.metadata["olala.layer_types"] == config["layers_config"]
                  and g.metadata["olala.block_count"] == layers and len(g.metadata["tokenizer.ggml.tokens"]) == 300,
                  "%-7s %d tensors, single precision and BF16 kept, largest relative distance %.4f"
                  % (target, len(g.tensors), worst))
            # The check of a conversion: every tensor of the file follows the checkpoint, and a file whose
            # tensor holds other values fails it.
            problems, _ = verify_tensors(ck, g)
            check(not problems, "%-7s every tensor follows the checkpoint%s"
                  % (target, "" if not problems else ": " + problems[0]))
        damaged = os.path.join(folder, "tiny-olala-damaged.gguf")
        with open(os.path.join(folder, "tiny-olala-q8_0.gguf"), "rb") as f:
            image = bytearray(f.read())
        g = GGUFFile(os.path.join(folder, "tiny-olala-q8_0.gguf"))
        at = g.data + g.tensors["blk.0.ssm_in.weight"][2]
        image[at:at + 4096] = bytes(4096)
        with open(damaged, "wb") as f:
            f.write(image)
        problems, _ = verify_tensors(ck, GGUFFile(damaged))
        check(len(problems) == 1 and "blk.0.ssm_in.weight" in problems[0], "a tensor that holds other values fails")
        os.remove(damaged)
        # A checkpoint that stores its small tensors in BF16, as the fine-tuned models of the family do:
        # its files hold the same tensors in the same formats.
        tiny_olala(os.path.join(folder, "checkpoint-bf16"), small="BF16")
        half = Checkpoint(os.path.join(folder, "checkpoint-bf16"))
        same = True
        for target in ("bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"):
            _, first = build(ck, target, "tiny-olala", {})
            _, second = build(half, target, "tiny-olala", {})
            same = same and [t[:3] for t in first] == [t[:3] for t in second]
        metadata, tensors = build(half, "q6_k", "tiny-olala", {})
        path = os.path.join(folder, "tiny-olala-bf16-q6_k.gguf")
        write_gguf(path, metadata, tensors, jobs=1)
        problems, _ = verify_tensors(half, GGUFFile(path))
        check(same and not problems, "a checkpoint with its small tensors in BF16 gives the same tensors%s"
              % ("" if not problems else ": " + problems[0]))
        # A matrix of faint values, which the scales of the 8-bit and 6-bit formats round coarsely or to zero:
        # it is kept in BF16, alone, and every tensor of the file follows the checkpoint.  The 5-bit and 4-bit
        # formats, whose scales are larger numbers, may still hold values of that size.
        faint = "blk.1.ffn_latent_down.weight"
        tiny_olala(os.path.join(folder, "checkpoint-faint"), faint=("model.layers.1.mlp.down_proj.weight",))
        low = Checkpoint(os.path.join(folder, "checkpoint-faint"))
        for target in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
            metadata, tensors = build(low, target, "tiny-olala", {})
            path = os.path.join(folder, "tiny-olala-faint-%s.gguf" % target)
            kept = write_gguf(path, metadata, tensors, jobs=1)
            g = GGUFFile(path)
            names = [k[0] for k in kept]
            changed = [t[0] for t in tensors if g.tensors[t[0]][1] != t[2]]
            problems, _ = verify_tensors(low, g)
            expected = names == [faint] or (not names and target in ("q5_k_m", "q4_k_m"))
            check(expected and changed == names and all(g.tensors[n][1] == BF16 for n in names) and not problems
                  and g.order == [t[0] for t in tensors],
                  "%-7s a matrix of faint values %s, the other tensors keep their formats"
                  % (target, "is kept in BF16 (%s left it at %.2f)" % (TYPE_NAMES[kept[0][1]], kept[0][2]) if kept
                     else "is held by its format"))
    finally:
        if not args.folder:
            shutil.rmtree(folder, ignore_errors=True)
    print("all checks passed" if failures == 0 else "%d checks FAILED" % failures)
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Open weights in safetensors to GGUF files for inference.")
    commands = parser.add_subparsers(dest="command")
    p = commands.add_parser("fetch", help="download a checkpoint from the Hugging Face hub")
    p.add_argument("repository")
    p.add_argument("folder")
    p.add_argument("--revision", default="main")
    types = sorted(FILE_TYPES)
    for name in ("plan", "convert"):
        p = commands.add_parser(name, help="list the tensors of a conversion" if name == "plan" else "write GGUF files")
        p.add_argument("checkpoint")
        p.add_argument("--type", action="append", choices=types, help="format of the file (default bf16); repeatable")
        p.add_argument("--name", help="name of the model in the files (default: the name of the folder)")
        if name == "plan":
            p.add_argument("--list", action="store_true", help="print every tensor")
        else:
            p.add_argument("--out", default=".", help="folder of the GGUF files")
            p.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                           help="processes that quantize")
            p.add_argument("--license", default="", help="general.license of the files, such as apache-2.0")
            p.add_argument("--author", default="", help="general.author of the files")
            p.add_argument("--url", default="", help="general.url of the files")
            p.add_argument("--source-url", default="", help="general.source.url: where the checkpoint comes from")
    p = commands.add_parser("inspect", help="print the metadata and the tensors of a GGUF file")
    p.add_argument("file")
    p.add_argument("--list", action="store_true", help="print every tensor")
    p = commands.add_parser("compare", help="compare two GGUF files")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--tolerance", type=float, default=0.1, help="relative distance two formats may leave")
    p.add_argument("--ignore", action="append", help="metadata key left out of the comparison; repeatable")
    p = commands.add_parser("verify", help="set the tensors of a GGUF file against the weights of its checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("file")
    p = commands.add_parser("selftest", help="check the quantizers and convert a small random model")
    p.add_argument("--folder", help="where the small model is written and kept (default: a temporary folder, "
                                     "removed at the end)")
    args = parser.parse_args()
    if not args.command:
        print(__doc__)
        return 1
    try:
        return {"fetch": command_fetch, "plan": command_plan, "convert": command_convert, "inspect": command_inspect,
                "compare": command_compare, "verify": command_verify,
                "selftest": command_selftest}[args.command](args)
    except BrokenPipeError:
        # The reader of the output closed it, as head does: nothing more to print.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
