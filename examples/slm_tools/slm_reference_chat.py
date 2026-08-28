#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Reference evaluation of a GGUF model through llama.cpp, for comparison with the C++ path.
#
# An import pipeline that nothing contradicts is a pipeline nobody has verified. This tool
# runs the same GGUF file through an independent implementation and reports the same
# figures as slm_gguf_import_ex and slm_gguf_runtime_ex, so a disagreement points at a
# defect rather than being absorbed as noise. Every quantization format, rotary
# convention, QKV bias and per-head normalization added to the C++ path was settled here.
#
# Four modes, mirroring the probes of the C++ programs:
#   chat (default)    interactive multi-turn chat
#   --probe TEXT      most probable next tokens for TEXT, like --probe-logits
#   --probe-ids IDS   the same on an explicit id sequence, like --probe-ids
#   --show-tokens T   tokenize T and print the ids, to compare tokenizers
#
# --probe-ids is the mode that settles an argument. It bypasses tokenization and the chat
# template, so the two implementations are fed the exact same integers and any difference
# is in the model evaluation itself. Take the ids from the C++ side with --trace-prompt.
#
# Protocol for a token-exact comparison: greedy decoding on both sides, that is
# --deterministic here and --deterministic in the C++ program. Greedy is the argmax of the
# logits and does not depend on the sampler, so the two must agree if the pipeline is
# right. Plain sampling will not match, the random draws being independent.
#
# Requirements:
#   python3 -m pip install --upgrade pip
#   pip install llama-cpp-python numpy \
#       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#
#   The prebuilt index is worth the extra argument: without it pip compiles llama.cpp from
#   source, which needs a toolchain and, on distributions shipping pip below 23, fails in
#   the build backend long before any compiler runs. Upgrading pip first is what fixes
#   that failure; the index is what avoids the compilation altogether.
#
#   Falling back to a source build:
#       pip install --upgrade pip setuptools wheel "packaging>=23" "scikit-build-core>=0.9"
#       CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install llama-cpp-python --no-build-isolation
#
#   For a GPU build see the llama-cpp-python installation notes; --n-gpu-layers then
#   offloads that many layers. Note that GPU evaluation is not bit-reproducible, so a
#   comparison against the CPU path should keep --n-gpu-layers at 0.
#
# Usage:
#   slm_reference_chat.py --model model.gguf --probe "The capital of France is"
#   slm_reference_chat.py --model model.gguf --probe-ids "785 6722 315 9625 374"
#   slm_reference_chat.py --model model.gguf --chat --deterministic

import argparse
import sys

try:
    import numpy as np
    from llama_cpp import Llama
except ImportError as exc:
    sys.exit(f"missing dependency: {exc}\n"
             "install with:\n"
             "  python3 -m pip install --upgrade pip\n"
             "  pip install llama-cpp-python numpy \\\n"
             "      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
             "The prebuilt index avoids compiling llama.cpp from source, which fails in the\n"
             "build backend on distributions shipping pip below 23.")


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def last_logits(llm):
    """Logits of the last evaluated position.

    Requires the model to have been built with all logits kept: llama.cpp only stores the
    last row otherwise, and the whole point of a probe is to look at a chosen position.
    """
    return np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)


def report_top(llm, logits, top_n):
    p = softmax(logits)
    order = np.argsort(p)[::-1][:top_n]
    print("Most probable next tokens:")
    for t in order:
        piece = llm.detokenize([int(t)]).decode("utf-8", "replace")
        print(f"  {p[t]:.6f}  id {int(t)}  {piece!r}")


def run_probe(llm, text, top_n, add_bos):
    llm.reset()
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)
    llm.eval(tokens)
    print(f'Prompt ({len(tokens)} tokens): "{text}"')
    print("Token ids fed:", " ".join(str(t) for t in tokens))
    report_top(llm, last_logits(llm), top_n)


def run_probe_ids(llm, id_string, top_n):
    """Feed an explicit id sequence, bypassing tokenization and the chat template.

    No BOS is added: the ids are taken as they are, so they must already carry whatever
    the compared run carried.
    """
    ids = [int(x) for x in id_string.replace(",", " ").split()]
    if not ids:
        sys.exit("--probe-ids received no id")
    llm.reset()
    llm.eval(ids)
    print(f"Fed {len(ids)} explicit token ids.")
    report_top(llm, last_logits(llm), top_n)


def run_show_tokens(llm, text, add_bos):
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)
    print(f"{len(tokens)} tokens: {' '.join(str(t) for t in tokens)}")
    for t in tokens:
        piece = llm.detokenize([int(t)]).decode("utf-8", "replace")
        print(f"  {t}  {piece!r}")


def run_chat(llm, args):
    sampling = dict(
        max_tokens=args.max_response,
        temperature=0.0 if args.deterministic else args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
    )
    print("Ready. Type 'quit' or 'exit' to stop.\n")

    if args.raw:
        # No template: the text simply continues, which is how to see exactly what the
        # model emits when the template cleaning of a chat path hides it.
        prompt = ""
        while True:
            try:
                line = input("You: ").strip()
            except EOFError:
                break
            if line in ("quit", "exit"):
                break
            if not line:
                continue
            prompt += line
            text = llm.create_completion(prompt, **sampling)["choices"][0]["text"]
            print("Model:", text, "\n")
            prompt += text
        return

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if line in ("quit", "exit"):
            break
        if not line:
            continue
        messages.append({"role": "user", "content": line})
        out = llm.create_chat_completion(messages, **sampling)
        text = out["choices"][0]["message"]["content"] or ""
        print("Model:", text, "\n")
        messages.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser(
        description="llama.cpp reference evaluation of a GGUF model, for comparison "
                    "with the C++ import and runtime paths.")
    ap.add_argument("--model", required=True, help="path to the GGUF model file")
    ap.add_argument("--probe", metavar="TEXT", default=None,
                    help="print the top next tokens for TEXT, then exit")
    ap.add_argument("--probe-ids", dest="probe_ids", metavar="IDS", default=None,
                    help="same, on an explicit space- or comma-separated id sequence")
    ap.add_argument("--show-tokens", dest="show_tokens", metavar="TEXT", default=None,
                    help="print the token ids of TEXT, then exit")
    ap.add_argument("--chat", action="store_true",
                    help="interactive chat; the default when no probe is given")
    ap.add_argument("--raw", action="store_true",
                    help="chat without the template, to see the raw completion")
    ap.add_argument("--system", default="You are a helpful assistant.",
                    help="system block; empty omits it entirely")
    ap.add_argument("--chat-format", dest="chat_format", default="auto",
                    help="template override: auto uses the one the GGUF declares, "
                         "otherwise a llama.cpp format name such as chatml or zephyr")
    ap.add_argument("--top-n", dest="top_n", type=int, default=5,
                    help="candidates listed by the probes (default: 5)")
    ap.add_argument("--no-bos", dest="add_bos", action="store_false",
                    help="do not prepend BOS when tokenizing, to match a run that did not")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", dest="top_k", type=int, default=40)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--min-p", dest="min_p", type=float, default=0.05)
    ap.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=1.1)
    ap.add_argument("--deterministic", action="store_true",
                    help="greedy decoding; required for a token-exact comparison")
    ap.add_argument("--n-ctx", dest="n_ctx", type=int, default=2048)
    ap.add_argument("--max-response", dest="max_response", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-gpu-layers", dest="n_gpu_layers", type=int, default=0,
                    help="layers offloaded to the GPU; keep at 0 when comparing, "
                         "GPU evaluation not being bit-reproducible")
    ap.add_argument("--verbose", action="store_true",
                    help="print the backend and device information")
    args = ap.parse_args()

    probing = args.probe is not None or args.probe_ids is not None

    options = dict(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        verbose=args.verbose,
    )
    if args.chat_format != "auto":
        options["chat_format"] = args.chat_format
    if probing:
        # The probes read a chosen position, which llama.cpp only keeps when asked. The
        # option was renamed across releases, so both spellings are tried.
        options["logits_all"] = True

    try:
        llm = Llama(**options)
    except TypeError as exc:
        if "logits_all" not in str(exc):
            raise
        options.pop("logits_all")
        llm = Llama(**options)
        print("note: this llama-cpp-python build has no logits_all; the probes read the "
              "last position only, which is what they use anyway", file=sys.stderr)

    if args.show_tokens is not None:
        run_show_tokens(llm, args.show_tokens, args.add_bos)
        return
    if args.probe_ids is not None:
        run_probe_ids(llm, args.probe_ids, args.top_n)
        return
    if args.probe is not None:
        run_probe(llm, args.probe, args.top_n, args.add_bos)
        return
    run_chat(llm, args)


if __name__ == "__main__":
    sys.exit(main())
