#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Sequence-level distillation: build a training corpus from what teachers answer.
#
# WHAT THIS IS, AND HOW IT DIFFERS FROM THE OTHER KIND
#
# slm_distill_ex records a teacher's distribution at every position, which needs its
# weights. This script needs only its answers. Prompts go in, generated text comes out, and
# the pairs become an ordinary supervised corpus that slm_lora_finetune_ex reads. The
# signal is far thinner per token, one hard label instead of an ordering over the whole
# vocabulary, and the method compensates with the volume and the diversity of the prompts.
#
# It is the same procedure behind Stanford Alpaca and Vicuna, and it is what "distillation"
# usually means when the teacher is reachable only through an interface.
#
# WHY SEVERAL TEACHERS
#
# One teacher passes on its knowledge and its mannerisms alike: its refusals, its favourite
# openings, its length habits. Rotating over several dilutes what is idiosyncratic while
# what they agree on survives, which is the part worth having. The teachers do not need to
# be the same size or family; they need to answer the same prompts.
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
#
# It queries local GGUF models through llama-cpp-python and nothing else. Pointing it at a
# commercial API would be a few lines, and those lines are not here on purpose: the terms of
# service of most such services forbid using their output to train a competing model, and a
# tool that makes the wrong thing effortless invites it.
#
# Usage:
#   slm_sequence_distill.py --teacher a.gguf --teacher b.gguf \
#                           --prompts instructions.txt --out ~/corpus/synth.txt
#
#   slm_sequence_distill.py --teacher a.gguf --prompts seeds.txt --out synth.txt \
#                           --system "You are a cybersecurity analyst." \
#                           --temperature 0.8 --max-tokens 512 --resume

import argparse
import hashlib
import os
import random
import re
import sys
import time

try:
    from llama_cpp import Llama
except ImportError:
    sys.exit("this script needs llama-cpp-python: pip install llama-cpp-python")

RECORD_SENTINEL = "<<<record>>>"
SYSTEM_SENTINEL = "<<<system>>>"
USER_SENTINEL = "<<<user>>>"
ASSISTANT_SENTINEL = "<<<assistant>>>"

# Openings a model produces when it is declining rather than answering. A refusal is a
# perfectly good answer for the teacher and a poor training example for the student, which
# would learn the shape of a refusal without the judgment behind it.
REFUSAL_PATTERNS = [
    r"^\s*i(?:'m| am)\s+sorry",
    r"^\s*i\s+(?:cannot|can'?t|won'?t)\b",
    r"^\s*i(?:'m| am)\s+(?:unable|not able)\b",
    r"^\s*as an ai\b",
    r"^\s*i(?: do not|'?t| don'?t) have\b",
    r"^\s*sorry,?\s+but\b",
    r"^\s*unfortunately,?\s+i\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def read_prompts(path, limit):
    """One instruction per line, blank lines and # comments ignored.

    A line is taken verbatim: whatever prompt engineering the corpus deserves belongs
    upstream of this script, where it can be reviewed.
    """
    seen, prompts = set(), []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line in seen:
                continue
            seen.add(line)
            prompts.append(line)
            if limit and len(prompts) >= limit:
                break
    return prompts


def already_done(path):
    """Prompts a previous run already answered, so that --resume costs nothing twice."""
    done = set()
    if not os.path.exists(path):
        return done
    field = None
    current = ""
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if line == USER_SENTINEL:
                field, current = "user", ""
                continue
            if line in (RECORD_SENTINEL, SYSTEM_SENTINEL, ASSISTANT_SENTINEL):
                if field == "user" and current.strip():
                    done.add(current.strip())
                field = None
                continue
            if field == "user":
                current += line + "\n"
    if field == "user" and current.strip():
        done.add(current.strip())
    return done


def is_repetitive(text, window=60, threshold=3):
    """True when one span keeps coming back.

    Small models loop. A looping answer teaches looping, so it is dropped rather than
    truncated: a truncated loop is still mostly loop.
    """
    if len(text) < window * 2:
        return False
    counts = {}
    for i in range(0, len(text) - window, window // 2):
        span = text[i:i + window]
        counts[span] = counts.get(span, 0) + 1
        if counts[span] >= threshold:
            return True
    return False


def quality_verdict(answer, args):
    """Why an answer is kept or dropped, as a short reason or None."""
    a = answer.strip()
    if len(a) < args.min_chars:
        return "too short"
    if args.max_chars and len(a) > args.max_chars:
        return "too long"
    if not args.keep_refusals and REFUSAL_RE.search(a):
        return "refusal"
    if is_repetitive(a):
        return "repetitive"
    letters = sum(1 for c in a if c.isalpha() or c.isspace())
    if letters / max(len(a), 1) < args.min_alpha:
        return "not prose"
    return None


def write_record(out, system, user, assistant):
    out.write(RECORD_SENTINEL + "\n")
    if system:
        out.write(SYSTEM_SENTINEL + "\n" + system.strip() + "\n")
    out.write(USER_SENTINEL + "\n" + user.strip() + "\n")
    out.write(ASSISTANT_SENTINEL + "\n" + assistant.strip() + "\n")
    out.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Build a supervised corpus from what local teachers answer.")
    parser.add_argument("--teacher", action="append", required=True, metavar="GGUF",
                        help="teacher model; repeat it to rotate over several, which "
                             "dilutes the mannerisms of any single one")
    parser.add_argument("--prompts", required=True,
                        help="one instruction per line; blank lines and # comments ignored")
    parser.add_argument("--out", required=True, help="corpus in the sentinel format")
    parser.add_argument("--system", default="",
                        help="system block written on every record")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many prompts; 0 reads them all")
    parser.add_argument("--rotation", choices=["cycle", "random", "all"], default="cycle",
                        help="cycle gives each prompt to the next teacher, random picks "
                             "one, all asks every teacher every prompt (default: cycle)")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=4096,
                        help="context window given to the teacher (default: 4096)")
    parser.add_argument("--threads", type=int, default=0,
                        help="0 lets llama.cpp decide")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--min-chars", type=int, default=80,
                        help="answers shorter than this are dropped (default: 80)")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="answers longer than this are dropped; 0 keeps them all")
    parser.add_argument("--min-alpha", type=float, default=0.75,
                        help="minimum share of letters and spaces (default: 0.75)")
    parser.add_argument("--keep-refusals", action="store_true",
                        help="keep answers that decline; they are dropped by default, "
                             "since a student learns the shape of a refusal without the "
                             "judgment behind it")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing corpus, skipping prompts already "
                             "answered in it")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts, args.limit)
    if not prompts:
        sys.exit(f"no prompt could be read from {args.prompts}")

    skip = already_done(args.out) if args.resume else set()
    todo = [p for p in prompts if p not in skip]

    print(f"prompts read     : {len(prompts)}")
    if skip:
        print(f"already answered : {len(skip)}, skipped")
    print(f"to generate      : {len(todo)}")
    print(f"teachers         : {len(args.teacher)} ({args.rotation})")

    if not todo:
        print("nothing to do.")
        return

    random.seed(args.seed)

    # Teachers are loaded one at a time rather than all at once: several billion-parameter
    # models resident together is how a machine starts swapping, and swapping is slower than
    # reloading. With 'cycle' or 'random' the prompts are grouped by teacher so that each
    # one is loaded exactly once.
    assignment = {i: [] for i in range(len(args.teacher))}
    if args.rotation == "all":
        for i in range(len(args.teacher)):
            assignment[i] = list(todo)
    elif args.rotation == "random":
        for p in todo:
            assignment[random.randrange(len(args.teacher))].append(p)
    else:
        for n, p in enumerate(todo):
            assignment[n % len(args.teacher)].append(p)

    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    kept = 0
    dropped = {}
    started = time.time()

    with open(args.out, mode, encoding="utf-8") as out:
        for idx, path in enumerate(args.teacher):
            mine = assignment[idx]
            if not mine:
                continue
            name = os.path.basename(path)
            print(f"\nloading {name} ...")
            llm = Llama(model_path=path, n_ctx=args.context, seed=args.seed,
                        n_threads=(args.threads or None), verbose=False)
            print(f"  {len(mine)} prompts for this teacher")

            for n, prompt in enumerate(mine, 1):
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                messages.append({"role": "user", "content": prompt})

                try:
                    # The teacher's own chat template is applied by llama.cpp, which is what
                    # makes the answer look like the answer it would give in service.
                    reply = llm.create_chat_completion(
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens)
                    answer = reply["choices"][0]["message"]["content"]
                except Exception as e:                      # noqa: BLE001
                    dropped["generation error"] = dropped.get("generation error", 0) + 1
                    print(f"\r  [{n}/{len(mine)}] error: {e}")
                    continue

                why = quality_verdict(answer, args)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                else:
                    write_record(out, args.system, prompt, answer)
                    kept += 1

                if n % 10 == 0 or n == len(mine):
                    rate = (time.time() - started) / max(kept + sum(dropped.values()), 1)
                    print(f"\r  [{n}/{len(mine)}] kept {kept}, "
                          f"{rate:.1f} s per prompt   ", end="", flush=True)
            print()
            del llm

    elapsed = time.time() - started
    print(f"\nSequence-level corpus")
    print(f"  records written : {kept}")
    if dropped:
        detail = ", ".join(f"{v} {k}" for k, v in sorted(dropped.items()))
        print(f"  dropped         : {detail}")
    print(f"  elapsed         : {elapsed / 60:.1f} min")
    print(f"  written to      : {args.out}")
    print("\nThis corpus is read by slm_lora_finetune_ex --dataset, and by")
    print("slm_distill_ex --supervised when a teacher's distributions are also wanted.")


if __name__ == "__main__":
    main()
