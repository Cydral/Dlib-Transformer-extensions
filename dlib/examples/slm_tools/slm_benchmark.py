#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Benchmark a model and place it against published small language models.
#
# WHAT THIS MEASURES, AND WHAT IT DOES NOT
#
# The divergence a distillation run reports says how close a student stayed to its teacher.
# It says nothing about whether either of them is any good, and a student can track a weak
# teacher perfectly. These benchmarks answer the other question: what the model actually
# knows and can do, on tasks the field has agreed to compare on.
#
# The suite below is the common one. ARC and its challenge split test grade-school science
# questions; HellaSwag and PIQA test whether a plausible continuation is preferred to an
# implausible one; WinoGrande tests pronoun resolution that needs world knowledge; MMLU
# spans fifty-seven subjects; GSM8K tests arithmetic reasoning and is the only one here a
# model under a billion parameters usually fails outright.
#
# ON THE COMPARISON FIGURES
#
# The reference numbers are read from a file, not from the network. Leaderboards move, their
# formats change, and a number scraped without its evaluation settings compares nothing:
# MMLU at five shots and MMLU at zero shots differ by several points on the same model. Each
# entry therefore carries its source and the date it was read, and updating it is a
# deliberate act rather than a silent one.
#
# The figures shipped with this file were written in 2026 and are already ageing. They are
# there to give a sense of scale, not to settle anything.
#
# Usage:
#   slm_benchmark.py --model ~/models/qwen3-24b-instruct-pruned
#   slm_benchmark.py --model ~/models/qwen3-24b-instruct-pruned --quick
#   slm_benchmark.py --model X --tasks arc_easy,hellaswag --limit 200
#   slm_benchmark.py --show-reference

import argparse
import json
import os
import sys
from datetime import datetime

REFERENCE_FILE = "slm_reference_scores.json"

# The default suite, chosen to finish in a reasonable time on a model of this size.
QUICK_TASKS = ["arc_easy", "hellaswag", "winogrande", "piqa"]
FULL_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "mmlu"]

# How each task is normally reported, so that a number here means the same as a number
# quoted elsewhere. Getting this wrong is the commonest way to publish a misleading table.
TASK_SHOTS = {"arc_easy": 0, "arc_challenge": 25, "hellaswag": 10, "winogrande": 5,
              "piqa": 0, "mmlu": 5, "gsm8k": 5, "truthfulqa_mc2": 0}

TASK_METRIC = {"arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm",
               "winogrande": "acc", "piqa": "acc_norm", "mmlu": "acc",
               "gsm8k": "exact_match", "truthfulqa_mc2": "acc"}


def load_reference(path):
    """The comparison table, with its provenance."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def show_reference(table):
    if not table:
        print("No reference table found. One is written on first use.")
        return
    print(f"{'model':<34} {'params':>8}  {'read':>10}  scores")
    print("-" * 96)
    for name, entry in sorted(table.items()):
        scores = ", ".join(f"{k} {v:.1f}" for k, v in sorted(entry.get("scores", {}).items()))
        print(f"{name:<34} {entry.get('params', '?'):>8}  "
              f"{entry.get('read', '?'):>10}  {scores}")
    print("\nEach entry carries the source it was read from:")
    for name, entry in sorted(table.items()):
        if entry.get("source"):
            print(f"  {name:<34} {entry['source']}")


def write_default_reference(path):
    """Writes a starting table, clearly dated, for the caller to correct and extend.

    These are figures as they stood when this file was written. They are a starting point
    and a reminder of the shape of the data, not an authority: verify each one against its
    source before quoting it anywhere that matters.
    """
    table = {
        "Qwen2.5-0.5B-Instruct": {
            "params": "0.5B", "read": "2026-05",
            "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
            "scores": {"arc_challenge": 32.0, "hellaswag": 52.1, "mmlu": 47.5,
                       "winogrande": 56.3}},
        "Qwen3-0.6B": {
            "params": "0.6B", "read": "2026-05",
            "source": "https://huggingface.co/Qwen/Qwen3-0.6B",
            "scores": {"arc_challenge": 36.0, "hellaswag": 55.0, "mmlu": 52.8,
                       "winogrande": 59.0}},
        "SmolLM2-360M-Instruct": {
            "params": "0.36B", "read": "2026-05",
            "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
            "scores": {"arc_challenge": 34.0, "hellaswag": 55.0, "mmlu": 31.0,
                       "winogrande": 56.0}},
        "SmolLM2-1.7B-Instruct": {
            "params": "1.7B", "read": "2026-05",
            "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "scores": {"arc_challenge": 51.0, "hellaswag": 68.7, "mmlu": 48.0,
                       "winogrande": 64.0}},
        "Llama-3.2-1B-Instruct": {
            "params": "1.2B", "read": "2026-05",
            "source": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
            "scores": {"arc_challenge": 38.0, "hellaswag": 60.8, "mmlu": 46.0,
                       "winogrande": 60.0}},
        "gemma-2-2b-it": {
            "params": "2.6B", "read": "2026-05",
            "source": "https://huggingface.co/google/gemma-2-2b-it",
            "scores": {"arc_challenge": 55.0, "hellaswag": 73.0, "mmlu": 56.1,
                       "winogrande": 69.0}},
        "Phi-3-mini-4k-instruct": {
            "params": "3.8B", "read": "2026-05",
            "source": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
            "scores": {"arc_challenge": 62.0, "hellaswag": 76.0, "mmlu": 68.8,
                       "winogrande": 71.0}},
    }
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(table, fout, indent=2, ensure_ascii=False)
    return table


def run_benchmarks(model_path, tasks, limit, batch_size, device, shots_override):
    """Runs lm-evaluation-harness in process and returns one score per task."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        sys.exit("this script needs lm-eval: pip install lm-eval")
    except TypeError as e:
        # A stale typing_extensions is the usual cause, and the message says nothing
        # about it: lm-eval declares typed dictionaries with a keyword the version
        # shipped by the distribution does not know.
        if "extra_items" in str(e):
            sys.exit("lm-eval needs a newer typing_extensions than this environment has.\n"
                     "  pip install --break-system-packages -U typing_extensions\n"
                     f"(the underlying error was: {e})")
        raise

    results = {}
    if str(batch_size) == "auto":
        print("\n  note: --batch-size auto probes for the largest batch the card will take,")
        print("        which fails outright if anything else is using it. A fixed size is")
        print("        slower and survives sharing the card with a training run.")
    try:
        lm = HFLM(pretrained=model_path, batch_size=batch_size, device=device)
    except Exception as e:                                        # noqa: BLE001
        sys.exit(f"could not load the model: {e}")

    for task in tasks:
        shots = shots_override if shots_override is not None else TASK_SHOTS.get(task, 0)
        print(f"\n  {task} ({shots}-shot"
              f"{f', {limit} samples' if limit else ''})...", flush=True)
        try:
            out = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=shots,
                                          limit=limit, verbosity="ERROR")
        except Exception as e:                                    # noqa: BLE001
            # Out of memory is the common failure and says nothing useful on its own:
            # the card is usually shared with a training run, or the batch is too large
            # for what is left of it.
            if "out of memory" in str(e).lower():
                print(f"    out of memory on {task}. The card may be shared with another")
                print( "    process; try --batch-size 1, or --device cpu, or wait.")
                continue
            print(f"    {task} failed: {e}")
            continue
        row = out["results"].get(task, {})
        metric = TASK_METRIC.get(task, "acc")
        value = None
        for key, val in row.items():
            # lm-eval names metrics "acc,none" and similar; match on the stem.
            if key.split(",")[0] == metric and isinstance(val, (int, float)):
                value = float(val) * 100.0
                break
        if value is None:
            for key, val in row.items():
                if key.split(",")[0] in ("acc", "exact_match") and isinstance(val, (int, float)):
                    value = float(val) * 100.0
                    break
        if value is not None:
            results[task] = value
            print(f"    {metric} = {value:.1f}")
        else:
            print(f"    no usable metric in {sorted(row)}")
    return results


def report(name, scores, table, limit):
    """Places the measured model in the reference table, task by task."""
    tasks = sorted(scores)
    print("\n" + "=" * 96)
    print(f"{'model':<34} {'params':>8}  " + "  ".join(f"{t[:12]:>12}" for t in tasks))
    print("-" * 96)

    rows = [(name, "measured", scores, True)]
    for ref, entry in table.items():
        rows.append((ref, entry.get("params", "?"), entry.get("scores", {}), False))

    for label, params, values, mine in rows:
        cells = []
        for t in tasks:
            cells.append(f"{values[t]:>12.1f}" if t in values else f"{'-':>12}")
        mark = " <-" if mine else ""
        print(f"{label:<34} {params:>8}  " + "  ".join(cells) + mark)

    print("\nReference figures come from the file named by --compare, with the source and")
    print("date recorded in it. They are not fetched: a number quoted without its")
    print("evaluation settings compares nothing, five-shot and zero-shot MMLU differing by")
    print("several points on the same model.")
    if limit:
        print(f"\nThe measured row used --limit {limit}, so it is an estimate on a subset and")
        print("carries a sampling error of a few points. Drop --limit before quoting it.")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Benchmark a model and place it against published small models.")
    p.add_argument("--model", help="model directory in Hugging Face layout")
    p.add_argument("--name", help="label for the report (default: the directory name)")
    p.add_argument("--tasks", help="comma-separated task list; defaults to the standard suite")
    p.add_argument("--quick", action="store_true",
                   help="four fast tasks instead of the full suite")
    p.add_argument("--limit", type=int, default=0,
                   help="samples per task; 0 runs them all. A few hundred gives a usable "
                        "estimate in a fraction of the time")
    p.add_argument("--shots", type=int, default=None,
                   help="override the few-shot count; the defaults match how each task is "
                        "normally reported")
    p.add_argument("--batch-size", default="4",
                   help="requests per forward pass. 'auto' lets lm-eval probe for the "
                        "largest that fits, which allocates aggressively and fails on a "
                        "card another process is already using (default: 4)")
    p.add_argument("--device", default=None, help="cuda or cpu; detected when omitted")
    p.add_argument("--compare", default=os.path.join(here, REFERENCE_FILE))
    p.add_argument("--show-reference", action="store_true",
                   help="print the comparison table and stop")
    p.add_argument("--out", help="write the measured scores to this JSON file")
    args = p.parse_args()

    table = load_reference(args.compare)
    if not table:
        print(f"No reference table at {args.compare}; writing a starting one.")
        table = write_default_reference(args.compare)

    if args.show_reference:
        show_reference(table)
        return

    if not args.model:
        sys.exit("--model is required unless --show-reference is given")

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = QUICK_TASKS if args.quick else FULL_TASKS

    name = args.name or os.path.basename(os.path.normpath(args.model))
    print(f"model   : {args.model}")
    print(f"device  : {device}")
    print(f"tasks   : {', '.join(tasks)}")

    scores = run_benchmarks(args.model, tasks, args.limit or None, args.batch_size,
                            device, args.shots)
    if not scores:
        sys.exit("no task produced a score")

    report(name, scores, table, args.limit)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fout:
            json.dump({"model": args.model, "name": name,
                       "measured": datetime.now().strftime("%Y-%m-%d"),
                       "limit": args.limit, "scores": scores}, fout, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
