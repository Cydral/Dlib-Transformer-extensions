#!/usr/bin/env python3
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Reference evaluation tool for the GGUF import work. It runs the same model file through
# llama.cpp (via llama-cpp-python) so its output can be compared with slm_gguf_import_ex.
# It mirrors what the Dlib program does: the TinyLlama / Zephyr chat template (system block
# on the first turn, then user/assistant turns), the same default sampling settings, and a
# single prefill followed by incremental decoding over a persistent KV cache (handled by
# llama.cpp internally).
#
# Three modes:
#   chat (default)   : interactive multi-turn chat using the zephyr template.
#   --probe TEXT     : print the most probable next tokens for TEXT, like --probe-logits.
#   --show-tokens T  : tokenize T and print the ids, to compare with the Dlib tokenizer.
#
# For a token-exact comparison use greedy decoding on both sides (--deterministic here and
# --deterministic in the Dlib program). Greedy is argmax of the logits and does not depend on
# the sampler order, so the two programs should agree if the Dlib pipeline is correct. Plain
# sampling will not match token for token, because the random draws differ.

import argparse
import sys

import numpy as np
from llama_cpp import Llama


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def last_logits(llm):
    # Requires logits_all=True so the scores buffer is populated.
    return np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)


def run_probe(llm, text, top_n=5):
    llm.reset()
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
    llm.eval(tokens)
    p = softmax(last_logits(llm))
    order = np.argsort(p)[::-1][:top_n]
    print(f'Prompt ({len(tokens)} tokens): "{text}"')
    print("Most probable next tokens:")
    for t in order:
        piece = llm.detokenize([int(t)]).decode("utf-8", "replace")
        print(f"  {p[t]:.4f}  id {int(t)}  {piece!r}")


def run_show_tokens(llm, text):
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
    print(f"{len(tokens)} tokens: {tokens}")
    for t in tokens:
        piece = llm.detokenize([int(t)]).decode("utf-8", "replace")
        print(f"  {t}  {piece!r}")


def run_chat(llm, args):
    common = dict(
        max_tokens=args.max_response,
        temperature=0.0 if args.deterministic else args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
    )
    print("Ready. Type 'quit' or 'exit' to stop.\n")

    if args.raw:
        # Raw completion: no template, the text simply continues.
        prompt = ""
        while True:
            try:
                line = input("You: ").strip()
            except EOFError:
                break
            if not line or line in ("quit", "exit"):
                if line in ("quit", "exit"):
                    break
                continue
            prompt += line
            out = llm.create_completion(prompt, **common)
            text = out["choices"][0]["text"]
            print("Model:", text, "\n")
            prompt += text
        return

    messages = [{"role": "system", "content": args.system}]
    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": line})
        out = llm.create_chat_completion(messages, **common)
        text = out["choices"][0]["message"]["content"] or ""
        print("Model:", text, "\n")
        messages.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser(description="llama.cpp reference chat / probe for GGUF import comparison")
    ap.add_argument("--model", required=True, help="path to the GGUF model file")
    ap.add_argument("--system", default="You are a helpful assistant.", help="system prompt for --chat")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", dest="top_k", type=int, default=40)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--min-p", dest="min_p", type=float, default=0.05)
    ap.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=1.1)
    ap.add_argument("--deterministic", action="store_true", help="greedy decoding (temperature 0)")
    ap.add_argument("--raw", action="store_true", help="chat without the template (raw completion)")
    ap.add_argument("--n-ctx", dest="n_ctx", type=int, default=2048, help="context window (Dlib --context defaults to 512)")
    ap.add_argument("--max-response", dest="max_response", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-gpu-layers", dest="n_gpu_layers", type=int, default=0,
                    help="layers to offload to the GPU (0 = CPU, -1 = all); needs a Vulkan or SYCL build")
    ap.add_argument("--verbose", action="store_true", help="print backend/device info (confirms GPU use)")
    ap.add_argument("--probe", metavar="TEXT", default=None, help="print top next tokens for TEXT, then exit")
    ap.add_argument("--show-tokens", dest="show_tokens", metavar="TEXT", default=None, help="print token ids for TEXT, then exit")
    args = ap.parse_args()

    need_logits = args.probe is not None
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=need_logits,
        chat_format="zephyr",
        seed=args.seed,
        verbose=args.verbose,
    )

    if args.show_tokens is not None:
        run_show_tokens(llm, args.show_tokens)
        return
    if args.probe is not None:
        run_probe(llm, args.probe)
        return
    run_chat(llm, args)


if __name__ == "__main__":
    sys.exit(main())
