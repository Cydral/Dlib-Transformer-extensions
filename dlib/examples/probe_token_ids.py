# Feed an explicit token-id sequence to llama.cpp and print the top next-token
# predictions at the last position. This bypasses tokenization and any chat
# template, so it compares llama.cpp against Dlib on the exact same token ids.
#
# Usage:
#   python probe_token_ids.py --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
#       --ids "1 529 29989 5205 ... 13"
#
# The ids must already include BOS if the Dlib run included it. No extra BOS is added.

import argparse
import numpy as np
from llama_cpp import Llama


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description="Probe llama.cpp on an explicit token-id sequence")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="space- or comma-separated token ids (BOS included)")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.replace(",", " ").split()]
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, logits_all=True, verbose=False)

    llm.reset()
    llm.eval(ids)
    p = softmax(np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64))
    order = np.argsort(p)[::-1][: args.top]

    print(f"Fed {len(ids)} explicit token ids. Top-{args.top} next-token predictions:")
    for t in order:
        piece = llm.detokenize([int(t)]).decode("utf-8", "replace")
        print(f"  {p[t]:.4f}  id {int(t)}  {piece!r}")


if __name__ == "__main__":
    main()
