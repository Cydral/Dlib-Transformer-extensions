#!/usr/bin/env python3
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Writes a small mixture-of-experts model (qwen3moe) with a chosen number of experts, to
check the engine on mixtures no published model of the catalog has, such as 256 experts
per layer of which 16 answer a token.

    python3 make_moe_model.py model.gguf experts used [seed [architecture]]

The model has two layers of 64 dimensions, four heads over two key/value heads, experts
of 24 dimensions and a vocabulary of the 256 bytes with the control tokens of ChatML.
Its weights are random, drawn from the seed (1 by default): the file is the same at every
run.  The architecture is qwen3moe by default, with the normalization of the queries and
of the keys; llama gives the layers of Mixtral, which the runtime engine of the library
reads too (slm_gguf_runtime_ex --probe-ids).  The logits of tests/reference_forward.py on this file are the reference of
test_runtime, which takes 17 token ids at least:

    ids="72 101 108 108 111 44 32 119 111 114 108 100 33 32 84 104 101 32 113 117"
    python3 make_moe_model.py moe.gguf 256 16
    python3 reference_forward.py moe.gguf moe.ref $ids
    test_runtime moe.gguf moe.ref exact $ids
    test_runtime moe.gguf moe.ref $ids

Random weights give some positions two logits that nearly tie, where the most likely token
of the engine and that of the reference may differ while their distributions agree: the
sequence above has no such position.

The gguf package of the GGUF ecosystem writes the file.
"""

import sys

import gguf
import numpy as np


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    path, n_experts, used = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    architecture = sys.argv[5] if len(sys.argv) > 5 else "qwen3moe"
    if not 1 <= used <= n_experts:
        print("the experts that answer a token are between 1 and the experts of a layer")
        return 2
    if architecture not in ("qwen3moe", "llama"):
        print("the architecture is qwen3moe or llama")
        return 2
    rng = np.random.default_rng(seed)
    d, hd, heads, kv, layers, df, vocab = 64, 16, 4, 2, 2, 24, 288
    w = gguf.GGUFWriter(path, architecture)
    w.add_name("moe%d" % n_experts)
    w.add_block_count(layers)
    w.add_context_length(512)
    w.add_embedding_length(d)
    w.add_feed_forward_length(df)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_expert_count(n_experts)
    w.add_expert_used_count(used)
    w.add_expert_feed_forward_length(df)
    tokens = [bytes([i]).decode("latin-1") for i in range(256)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    types = [1] * 256 + [3] * 3
    while len(tokens) < vocab:
        tokens.append("[PAD%d]" % len(tokens))
        types.append(5)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_eos_token_id(256)

    def tensor(name, *shape, scale=0.1, ones=False):
        values = (1.0 + 0.1 * rng.standard_normal(shape)) if ones else scale * rng.standard_normal(shape)
        w.add_tensor(name, values.astype(np.float32))

    tensor("token_embd.weight", vocab, d, scale=0.5)
    for layer in range(layers):
        p = "blk.%d." % layer
        tensor(p + "attn_norm.weight", d, ones=True)
        tensor(p + "attn_q.weight", heads * hd, d)
        tensor(p + "attn_k.weight", kv * hd, d)
        tensor(p + "attn_v.weight", kv * hd, d)
        tensor(p + "attn_output.weight", d, heads * hd)
        if architecture == "qwen3moe":
            tensor(p + "attn_q_norm.weight", hd, ones=True)
            tensor(p + "attn_k_norm.weight", hd, ones=True)
        tensor(p + "ffn_norm.weight", d, ones=True)
        # A wide router, so that the experts of a token change along a sequence.
        tensor(p + "ffn_gate_inp.weight", n_experts, d, scale=0.6)
        tensor(p + "ffn_gate_exps.weight", n_experts, df, d, scale=0.25)
        tensor(p + "ffn_up_exps.weight", n_experts, df, d, scale=0.25)
        tensor(p + "ffn_down_exps.weight", n_experts, d, df, scale=0.25)
    tensor("output_norm.weight", d, ones=True)
    tensor("output.weight", vocab, d, scale=0.3)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("%s: %d experts per layer, %d per token" % (path, n_experts, used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
