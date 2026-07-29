#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Reference loss of the source model, measured before any conversion.
#
# This exists to answer one question that the C++ side cannot answer about itself: when
# slm_lora_finetune_ex reports a loss before training, is that number the model's own or
# an artefact of the pipeline that feeds it? The only way to know is to put the same
# records through the published weights, with the same masking, and compare.
#
# What is held identical to the C++ path, because a difference in any of them would make
# the comparison meaningless:
#
#   the records, read from the same sentinel-separated file;
#   the turn rendering, which reproduces what chat_template_formatter emits for the
#   idefics3 family and which --trace-prompt showed literally;
#   the split between prompt and response, the prompt being masked out so that only the
#   answer is scored;
#   the window length and the left padding, and the fact that padded positions are both
#   masked in the loss and hidden from the attention;
#   the number of windows, taken from the head of the file in the same order.
#
# What necessarily differs is the arithmetic. The reference runs on bfloat16 or float32
# weights straight from the hub, the C++ path on weights that went through a GGUF
# container. A few hundredths of difference are expected and mean nothing; a factor of two
# means the pipeline is wrong.
#
# Usage:
#   slm_eval_loss.py --model HuggingFaceTB/SmolVLM-256M-Instruct \
#                    --dataset ~/corpus/cve_qa_valid.txt --windows 64 --window 512

import argparse
import sys

try:
    import torch
except ImportError:
    sys.exit("this script needs PyTorch: pip install torch")

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("this script needs transformers: pip install transformers")

RECORD_SENTINEL = "<<<record>>>"
SYSTEM_SENTINEL = "<<<system>>>"
USER_SENTINEL = "<<<user>>>"
ASSISTANT_SENTINEL = "<<<assistant>>>"

IGNORE = -100


def read_records(path):
    """Parse the sentinel-separated file the C++ side reads.

    A sentinel counts only when it is the whole line, which is what lets an answer contain
    a line that looks like one without breaking the record.
    """
    records, current, field = [], None, None

    def flush():
        nonlocal current, field
        if current and current.get("user") and current.get("assistant"):
            records.append({k: v.strip() for k, v in current.items()})
        current, field = None, None

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if line == RECORD_SENTINEL:
                flush()
                current = {"system": "", "user": "", "assistant": ""}
                continue
            if current is None:
                continue
            if line == SYSTEM_SENTINEL:
                field = "system"
                continue
            if line == USER_SENTINEL:
                field = "user"
                continue
            if line == ASSISTANT_SENTINEL:
                field = "assistant"
                continue
            if field:
                current[field] += line + "\n"
    flush()
    return records


def render_turn(system, user):
    """The prompt as chat_template_formatter emits it for the idefics3 family.

    Reproduced here rather than taken from the tokenizer's own template, so that the two
    sides are compared on the same string. --trace-prompt prints exactly this.
    """
    head = "<|im_start|>"
    if system:
        head += "System: " + system + "<end_of_utterance>\n"
    head += "User:" + user + "<end_of_utterance>\nAssistant:"
    return head


def build_window(tokenizer, record, window, pad_id):
    """One padded window with its labels, or None when the answer alone overflows.

    The prompt is truncated from its head when the pair does not fit, which is the policy
    the C++ side names truncate_prompt_head: an answer cut short would be scored against
    tokens it was never shown.
    """
    prompt_ids = tokenizer(render_turn(record["system"], record["user"]),
                           add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(record["assistant"] + "<end_of_utterance>",
                           add_special_tokens=False)["input_ids"]

    if len(answer_ids) >= window:
        return None
    room = window - len(answer_ids)
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    full = prompt_ids + answer_ids
    pad = window - len(full)
    ids = [pad_id] * pad + full
    labels = [IGNORE] * (pad + len(prompt_ids)) + answer_ids
    mask = [0] * pad + [1] * len(full)
    return ids, labels, mask


def main():
    parser = argparse.ArgumentParser(
        description="Measure the reference loss of the source model on a prepared dataset.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct",
                        help="model repository or local directory")
    parser.add_argument("--dataset", required=True,
                        help="sentinel-separated file, normally the validation one")
    parser.add_argument("--window", type=int, default=512,
                        help="window length, matching the C++ run (default: 512)")
    parser.add_argument("--windows", type=int, default=64,
                        help="windows measured, matching --initial-windows (default: 64)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="windows per forward pass (default: 1)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                        help="weight precision (default: float32, closest to the C++ path)")
    parser.add_argument("--pad-id", type=int, default=-1,
                        help="padding token; -1 takes the tokenizer's. The C++ side prints "
                             "the value it uses as 'Special ids ... pad=N'")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the first rendered prompt, to compare with --trace-prompt")
    args = parser.parse_args()

    records = read_records(args.dataset)
    if not records:
        sys.exit(f"no record could be read from {args.dataset}")
    print(f"records read : {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The padding token has to be the one the C++ side used, which it reads from the GGUF
    # and prints as 'Special ids'. A different filler changes nothing about the loss, the
    # positions being masked either way, but it keeps the two runs literally comparable.
    pad_id = args.pad_id
    if pad_id < 0:
        pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0
    print(f"padding token : {pad_id}")

    # The text decoder is what is being measured, but a vision-language checkpoint does
    # not load through AutoModelForCausalLM: its configuration is an image-text one, and
    # that factory only knows text architectures. The entry points are tried in the order
    # that matches the newer naming first, since which of them exists depends on the
    # installed version of transformers. Either way the model accepts input_ids alone and
    # the vision tower is never reached without pixel values.
    model = None
    errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                 "AutoModelForCausalLM"):
        try:
            factory = getattr(__import__("transformers", fromlist=[name]), name)
        except AttributeError:
            continue
        try:
            model = factory.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
            print(f"loaded through : {name}")
            break
        except (ValueError, TypeError) as e:
            errors.append(f"{name}: {e}")
            try:
                model = factory.from_pretrained(args.model,
                                                torch_dtype=getattr(torch, args.dtype))
                print(f"loaded through : {name}")
                break
            except (ValueError, TypeError) as e2:
                errors.append(f"{name} (torch_dtype): {e2}")
    if model is None:
        sys.exit("could not load the model:\n  " + "\n  ".join(errors))
    model.to(args.device)
    model.eval()

    windows = []
    for record in records:
        built = build_window(tokenizer, record, args.window, pad_id)
        if built is not None:
            windows.append(built)
        if len(windows) >= args.windows:
            break
    if not windows:
        sys.exit("no window could be built; is --window too small?")
    print(f"windows built : {len(windows)} of {args.window} tokens")

    if args.show_prompt:
        print("\nFirst rendered prompt:")
        print(render_turn(records[0]["system"], records[0]["user"]))
        print()

    total, counted = 0.0, 0
    scored_positions = 0
    with torch.no_grad():
        for i in range(0, len(windows), args.batch_size):
            chunk = windows[i:i + args.batch_size]
            ids = torch.tensor([w[0] for w in chunk], device=args.device)
            labels = torch.tensor([w[1] for w in chunk], device=args.device)
            mask = torch.tensor([w[2] for w in chunk], device=args.device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            n = int((labels != IGNORE).sum().item())
            total += float(out.loss.item()) * n
            scored_positions += n
            counted += len(chunk)

    print(f"scored positions : {scored_positions}")
    print(f"reference loss   : {total / max(scored_positions, 1):.5f}")
    print("\nCompare with the 'loss before training' line of slm_lora_finetune_ex, run\n"
          "with the same --window and --initial-windows on the same validation file.\n"
          "A difference of a few hundredths is the arithmetic; a factor of two is a bug.")


if __name__ == "__main__":
    main()
