#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Python counterpart of slm_lora_finetune_ex.cpp.
#
# Same corpora, same masking, same adapter geometry, same reported figures, run through
# the reference ecosystem. Its purpose is not to replace the C++ path but to give it
# something to be checked against: a loss curve is only meaningful next to another one
# obtained on the same data with the same conventions, and a fine-tuning pipeline that
# nothing contradicts is a pipeline nobody has verified.
#
# The conventions are matched deliberately, one by one:
#   - the corpora are the sentinel files written by nist_corpus_prepare.py and
#     cve_qa_prepare.py, read here by the same rules,
#   - the prompt is rendered by the model's own chat template, as encode_turn does,
#   - only the response is scored, prompt and padding carrying the ignore label,
#   - the window is derived from the data by the same percentile rule,
#   - the adapters sit on the same projections at the same rank, scale alpha / rank.
#
# What differs, and matters when reading a comparison: this path loads the model from the
# Hugging Face hub in bfloat16 or float32, where the C++ path loads a GGUF that was
# quantized and dequantized. The two therefore start from weights that agree to about
# 1e-3, not exactly, so identical losses are not expected. What should agree is the shape
# of the curve, the number of trainable parameters, and the ranking of two methods.
#
# Requirements:
#   pip install "torch>=2.2" "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \
#               "jinja2>=3.1"
#   Optional, for a GPU run in 4-bit: pip install bitsandbytes
#
#   jinja2 is not optional even though nothing here imports it: the chat template of a
#   modern model is a jinja document, and apply_chat_template refuses versions below 3.1.
#   Distribution packages are routinely older, so it is checked below rather than left to
#   surface as a stack trace in the middle of a corpus pass.
#
# Usage:
#   slm_lora_finetune.py --model Qwen/Qwen3-0.6B --dataset ~/corpus/cve_qa.txt --dry-run
#   slm_lora_finetune.py --model Qwen/Qwen3-0.6B --dataset ~/corpus/cve_qa.txt \
#                        --limit 256 --window 384 --batch-size 2 --epochs 2 \
#                        --out ./qwen3_cve_lora
#   slm_lora_finetune.py --model Qwen/Qwen3-0.6B --corpus ~/corpus/nist_corpus.txt \
#                        --dataset ~/corpus/cve_qa.txt --method dora --merge

import argparse
import json
import math
import os
import random
import sys
import time

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as exc:
    sys.exit(f"missing dependency: {exc}\n"
             'install with: pip install "torch>=2.2" "transformers>=4.44" '
             '"peft>=0.12" "accelerate>=0.33" "jinja2>=3.1"')


def _require_jinja2():
    """Fail here rather than three hundred lines into a corpus pass.

    apply_chat_template compiles the model's own template, which is jinja, and refuses
    anything below 3.1. Nothing in this file imports jinja2, so the requirement is
    invisible until it fires.
    """
    try:
        import jinja2
    except ImportError:
        sys.exit('missing dependency: jinja2\n'
                 'install with: pip install --upgrade "jinja2>=3.1"')
    parts = []
    for piece in jinja2.__version__.split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts[:2]) < (3, 1):
        sys.exit(f"jinja2 {jinja2.__version__} is too old for apply_chat_template\n"
                 'upgrade with: pip install --upgrade "jinja2>=3.1"')


_require_jinja2()

DOC_SENTINEL = "<<<doc>>>"
RECORD_SENTINEL = "<<<record>>>"
SYSTEM_SENTINEL = "<<<system>>>"
USER_SENTINEL = "<<<user>>>"
ASSISTANT_SENTINEL = "<<<assistant>>>"
IGNORE_LABEL = -100          # what the loss skips, the counterpart of IGNORE_LABEL in C++


# --------------------------------------------------------------------------------------
# Corpus files, read by the same rules as language_model_data.h
# --------------------------------------------------------------------------------------

def load_document_corpus(path):
    """Documents of a plain corpus file. Anything before the first sentinel is ignored."""
    documents, current, open_doc = [], [], False
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if line == DOC_SENTINEL:
                if open_doc:
                    text = "\n".join(current).strip()
                    if text:
                        documents.append(text)
                current, open_doc = [], True
                continue
            if open_doc:
                current.append(line)
    if open_doc:
        text = "\n".join(current).strip()
        if text:
            documents.append(text)
    return documents


def load_chat_records(path):
    """Records of a supervised corpus file, dropping any with an empty user or answer."""
    records, current, field = [], {"system": [], "user": [], "assistant": []}, None
    out = []

    def flush():
        rec = {k: "\n".join(v).strip() for k, v in current.items()}
        if rec["user"] and rec["assistant"]:
            out.append(rec)
        for v in current.values():
            v.clear()

    open_rec = False
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if line == RECORD_SENTINEL:
                if open_rec:
                    flush()
                open_rec, field = True, None
                continue
            if not open_rec:
                continue
            if line == SYSTEM_SENTINEL:
                field = "system"; continue
            if line == USER_SENTINEL:
                field = "user"; continue
            if line == ASSISTANT_SENTINEL:
                field = "assistant"; continue
            if field:
                current[field].append(line)
    if open_rec:
        flush()
    records = out
    return records


# --------------------------------------------------------------------------------------
# Tokenized datasets, matching finetuning_data.h
# --------------------------------------------------------------------------------------

def encode_supervised_example(tok, record, supervise_eos=True, thinking=False):
    """Prompt and response token ids of one record.

    The prompt is rendered by the model's own chat template with the generation prompt
    appended, which is what the inference path builds. Training on anything else spends
    capacity on the difference.

    enable_thinking is passed explicitly because its default is not the one a plain
    question-and-answer corpus needs. On a reasoning-capable model the reference template
    appends an empty think block when it is false, and that block is not decoration: it is
    how the family switches the reasoning trace off. Leaving it out puts the model in
    reasoning mode, where it opens a trace the reference answers do not contain, so the
    stage teaches it to skip the reasoning it would otherwise produce. Templates that do
    not define the variable ignore it.
    """
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["user"]})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=thinking)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tok(record["assistant"], add_special_tokens=False)["input_ids"]
    if supervise_eos and tok.eos_token_id is not None:
        response_ids = response_ids + [tok.eos_token_id]
    return prompt_ids, response_ids


def suggest_window_length(totals, coverage=0.95, granularity=64, max_window=0):
    """Shortest window keeping the requested fraction of the examples whole.

    The window is a property of the data, not of the model: rotary positions extend to any
    length and the cache is sized at inference. Coverage stops short of one on purpose,
    since a handful of outliers would otherwise set the cost of every batch.
    """
    if not totals:
        return granularity
    ordered = sorted(totals)
    i = min(len(ordered) - 1, int(coverage * (len(ordered) - 1) + 0.5))
    needed = max(ordered[i] - 1, 1)
    window = ((needed + granularity - 1) // granularity) * granularity
    if max_window > 0:
        window = min(window, max_window)
    return max(window, granularity)


class WindowDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs, self.labels = inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {"input_ids": self.inputs[i], "labels": self.labels[i]}


def build_supervised_dataset(pairs, window, pad_id, policy="truncate_prompt_head"):
    """One window per example, prompt masked out.

    Position t predicts token t + 1, so the first scored position is the last token of the
    prompt: that is where the answer must begin. Scoring the prompt as well would teach the
    model to reproduce the questions.
    """
    inputs, labels = [], []
    kept = skipped = truncated = scored = ignored = 0
    capacity = window + 1

    for prompt, response in pairs:
        if not response:
            skipped += 1
            continue
        if len(prompt) + len(response) > capacity:
            if policy == "skip":
                skipped += 1
                continue
            room = capacity - len(response)
            if room <= 0:
                skipped += 1
                continue
            prompt = prompt[-room:]
            truncated += 1

        full = prompt + response
        total = len(full)
        first_scored = len(prompt) - 1

        x = [full[t] if t < total else pad_id for t in range(window)]
        y = []
        for t in range(window):
            if t >= first_scored and t + 1 < total:
                y.append(full[t + 1]); scored += 1
            else:
                y.append(IGNORE_LABEL); ignored += 1

        inputs.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
        kept += 1

    report = dict(kept=kept, skipped=skipped, truncated=truncated,
                  windows=len(inputs), scored=scored, ignored=ignored)
    return WindowDataset(inputs, labels), report


def build_causal_dataset(streams, window, stride, pad_id, pack=True):
    """Causal windows over a corpus, every position scored."""
    inputs, labels = [], []
    scored = ignored = 0

    def emit(stream, start):
        nonlocal scored, ignored
        n = len(stream)
        x, y = [], []
        for t in range(window):
            i = start + t
            x.append(stream[i] if i < n else pad_id)
            if i + 1 < n:
                y.append(stream[i + 1]); scored += 1
            else:
                y.append(IGNORE_LABEL); ignored += 1
        inputs.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))

    sources = [[t for doc in streams for t in doc]] if pack else streams
    for stream in sources:
        n = len(stream)
        start = 0
        while start + 1 < n:
            emit(stream, start)
            if start + window >= n:
                break
            start += stride

    return WindowDataset(inputs, labels), dict(
        kept=len(streams), skipped=0, truncated=0,
        windows=len(inputs), scored=scored, ignored=ignored)


def describe_report(rep):
    total = rep["scored"] + rep["ignored"]
    pct = 100.0 * rep["scored"] / total if total else 0.0
    line = f"  examples    : {rep['kept']} kept"
    if rep["skipped"]:
        line += f", {rep['skipped']} skipped"
    if rep["truncated"]:
        line += f", {rep['truncated']} truncated"
    line += f"\n  windows     : {rep['windows']}"
    line += f"\n  supervision : {rep['scored']} scored, {rep['ignored']} ignored ({pct:.0f}% scored)"
    return line


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def collate(batch):
    return (torch.stack([b["input_ids"] for b in batch]),
            torch.stack([b["labels"] for b in batch]))


def report_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters  : {total} total, {trainable} trainable "
          f"({100.0 * trainable / total:.3f}%)")
    if trainable == 0:
        print("  warning     : nothing is trainable; check the rank and the targets")
    return total, trainable


@torch.no_grad()
def evaluate(model, dataset, device, batch_size):
    """Validation loss, one window at a time by default.

    The peak of a forward is set by the output head, which allocates
    samples x window x vocabulary; on a 150k vocabulary that is gigabytes for a handful of
    windows, so the evaluation is chunked like the training steps rather than run whole.
    """
    if len(dataset) == 0:
        return None
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    total, counted = 0.0, 0
    for x, y in loader:
        out = model(input_ids=x.to(device), labels=y.to(device))
        total += out.loss.item() * x.size(0)
        counted += x.size(0)
    model.train()
    return total / counted if counted else None


def run_training(model, train_set, valid_set, args, device):
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              betas=(args.beta1, args.beta2))
    steps = max(1, len(loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(optim, int(0.03 * steps), steps)

    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        running, seen = 0.0, 0
        for x, y in loader:
            out = model(input_ids=x.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += out.loss.item()
            seen += 1
        train_loss = running / max(seen, 1)
        print(f"  epoch {epoch + 1}/{args.epochs}  learning rate "
              f"{sched.get_last_lr()[0]:.2e}  average loss {train_loss:.5f}")

        v = evaluate(model, valid_set, device, args.valid_batch)
        if v is not None:
            note = "  (well above the training loss: the stage is memorizing)" \
                if v > 1.5 * train_loss else ""
            print(f"  validation loss {v:.5f}{note}")

    print(f"  trained in  : {int(time.time() - started)} s")


# --------------------------------------------------------------------------------------

def split_dataset(dataset, fraction, seed=1):
    """Deterministic split, shuffled before the cut so an ordered corpus is not sliced by
    position."""
    n = len(dataset)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_val = int(n * fraction)
    val = [order[i] for i in range(n_val)]
    tr = [order[i] for i in range(n_val, n)]
    pick = lambda idx: WindowDataset([dataset.inputs[i] for i in idx],
                                     [dataset.labels[i] for i in idx])
    return pick(tr), pick(val)


def make_peft(model, args):
    """LoRA or DoRA on the requested projections.

    peft names the projections as the model does, so the letters of --targets map onto
    q_proj, v_proj and the three feed-forward ones, which is the same set the C++ path
    reaches through its adapter plan. use_dora turns the same adapter into its
    magnitude-decomposed form, exactly as adapter_method::dora does.
    """
    modules = []
    if "q" in args.targets: modules.append("q_proj")
    if "k" in args.targets: modules.append("k_proj")
    if "v" in args.targets: modules.append("v_proj")
    if "o" in args.targets: modules.append("o_proj")
    if "f" in args.targets: modules += ["gate_proj", "up_proj", "down_proj"]
    if not modules:
        sys.exit("--targets selects no projection")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        use_dora=(args.method == "dora"),
        target_modules=modules,
    )
    return get_peft_model(model, config)


def main():
    p = argparse.ArgumentParser(
        description="Parameter-efficient fine-tuning, the Python counterpart of "
                    "slm_lora_finetune_ex.cpp.")
    p.add_argument("--model", required=True, help="model id or local path")
    p.add_argument("--corpus", default="", help="plain corpus; runs the knowledge stage")
    p.add_argument("--dataset", default="", help="question-and-answer records; runs the task stage")
    p.add_argument("--valid", default="", help="validation records for the task stage")
    p.add_argument("--out", default="", help="where to write the result")
    p.add_argument("--system", default="", help="system block forced on every record")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--method", default="lora", choices=["lora", "dora"])
    p.add_argument("--targets", default="qv",
                   help="letters among q, k, v, o and f for the feed-forward (default: qv)")
    p.add_argument("--window", type=int, default=0, help="0 derives it from the data")
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--max-window", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    # Four, to match the C++ default: a bare invocation of either program must describe
    # the same run, and the step count is what a comparison is read against.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--valid-batch", type=int, default=1)
    p.add_argument("--valid-fraction", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    p.add_argument("--think", action="store_true",
                   help="train a reasoning-capable model in reasoning mode; the reference "
                        "answers must then carry their own reasoning traces")
    p.add_argument("--merge", action="store_true",
                   help="fold the adapters into the weights before writing")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.corpus and not args.dataset:
        sys.exit("supply --corpus, --dataset, or both")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(
        args.dtype, torch.bfloat16 if device == "cuda" else torch.float32)

    print(f"Loading {args.model} on {device} in {dtype} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    model.to(device)

    model = make_peft(model, args)
    print(f"  adapters    : {args.method}, rank {args.rank}, alpha {args.alpha} "
          f"on {args.targets}")
    report_trainable(model)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # --- knowledge alignment -----------------------------------------------------------
    if args.corpus:
        print(f"\n=== Knowledge alignment: {args.corpus}")
        documents = load_document_corpus(args.corpus)
        streams = []
        for d in documents:
            ids = tok(d, add_special_tokens=False)["input_ids"]
            if ids:
                if tok.eos_token_id is not None:
                    ids = ids + [tok.eos_token_id]
                streams.append(ids)
        print(f"  documents   : {len(streams)}, {sum(len(s) for s in streams)} tokens")

        window = args.window if args.window > 0 else min(args.max_window, 512)
        train_set, rep = build_causal_dataset(streams, window, window, pad_id, pack=True)
        print(f"  window      : {window}\n{describe_report(rep)}")
        print(f"  cost        : {len(train_set) * window} positions per epoch, "
              f"{len(train_set) // max(1, args.batch_size)} steps")
        if not args.dry_run:
            run_training(model, train_set, WindowDataset([], []), args, device)
            if args.dataset:
                print("  merging the knowledge stage into the weights before the next one")
                model = model.merge_and_unload()
                model = make_peft(model, args)

    # --- task alignment ----------------------------------------------------------------
    if args.dataset:
        print(f"\n=== Task alignment: {args.dataset}")
        records = load_chat_records(args.dataset)
        if args.limit > 0:
            records = records[:args.limit]
            print(f"  limited to  : {len(records)} records")
        if args.system:
            for r in records:
                r["system"] = args.system

        pairs = [encode_supervised_example(tok, r, thinking=args.think) for r in records]
        totals = [len(a) + len(b) for a, b in pairs]
        ordered = sorted(totals)
        q = lambda f: ordered[min(len(ordered) - 1, int(f * (len(ordered) - 1) + 0.5))]
        print(f"  examples    : {len(pairs)}\n"
              f"  total tokens: median {q(.5)}, p90 {q(.9)}, p99 {q(.99)}, max {ordered[-1]}\n"
              f"  mean split  : {sum(len(a) for a, _ in pairs) // len(pairs)} prompt + "
              f"{sum(len(b) for _, b in pairs) // len(pairs)} response")

        window = args.window if args.window > 0 else suggest_window_length(
            totals, args.coverage, 64, args.max_window)
        covered = sum(1 for t in totals if t <= window + 1) / len(totals)
        print(f"  window      : {window} ({100 * covered:.0f}% covered)")

        train_set, rep = build_supervised_dataset(pairs, window, pad_id)
        print(describe_report(rep))

        if args.valid:
            vrec = load_chat_records(args.valid)
            if args.system:
                for r in vrec:
                    r["system"] = args.system
            valid_set, _ = build_supervised_dataset(
                [encode_supervised_example(tok, r, thinking=args.think) for r in vrec],
                window, pad_id)
            print(f"  validation  : {len(valid_set)} windows, from {args.valid}")
        elif args.valid_fraction > 0 and len(train_set) > 8:
            train_set, valid_set = split_dataset(train_set, args.valid_fraction)
            print(f"  validation  : {len(valid_set)} windows, held out from the training set")
        else:
            valid_set = WindowDataset([], [])

        print(f"  cost        : {len(train_set) * window} positions per epoch, "
              f"{len(train_set) // max(1, args.batch_size)} steps")
        if not args.dry_run:
            run_training(model, train_set, valid_set, args, device)

    if args.dry_run:
        print("\nDry run complete; nothing was written.")
        return

    if args.out:
        if args.merge:
            print("\nMerging the adapters into the weights ...")
            model = model.merge_and_unload()
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        note = "a full model" if args.merge else "adapters only, a few megabytes"
        print(f"Written to {args.out} ({note}).")
        if args.merge:
            print("Convert it to GGUF to load it with the C++ path:\n"
                  "  python llama.cpp/convert_hf_to_gguf.py " + args.out)


if __name__ == "__main__":
    main()
