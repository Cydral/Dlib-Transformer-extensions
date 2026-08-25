#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Depth pruning with distillation: keep a model's architecture, remove blocks, recover.
#
# WHY DEPTH RATHER THAN WIDTH
#
# A model can be made smaller by removing whole blocks or by shrinking every matrix inside
# them. The second is tempting, and factorizing each projection by SVD is the usual way to
# do it, but it produces something that is no longer the architecture it came from: every
# projection becomes two, and no llama-shaped container can describe that. The result would
# need a loader of its own.
#
# Removing whole blocks changes exactly one number. A Qwen with sixteen blocks instead of
# twenty-eight is still a Qwen: the same tensor names, the same attention, the same rotary
# encoding, one different entry in the metadata. It converts to GGUF through the ordinary
# tooling and loads in anything that reads the family, this library included.
#
# WHICH BLOCKS TO REMOVE
#
# Not every block does the same amount of work. A block that returns its input nearly
# unchanged costs a twenty-eighth of the compute and contributes almost nothing, and there
# are usually several of them in the middle of a deep model. Measuring that is cheap: run a
# few batches, and for each block record the angle between what enters it and what leaves
# it. Blocks whose output stays closest to their input are the ones to drop.
#
# The alternative, dropping blocks at regular intervals, is what a script writes when it
# has not looked. It removes useful blocks and keeps idle ones with equal enthusiasm.
#
# WHY THE STUDENT INHERITS RATHER THAN STARTS FRESH
#
# The blocks that remain keep the teacher's weights. A student built from a reduced
# configuration and randomly initialized would have to learn everything again, which is a
# pre-training budget rather than a recovery one. Inheriting means the student begins a few
# nats away from the teacher instead of at log(vocabulary), and the distillation only has to
# repair what the missing blocks used to do.
#
# WHAT THE RECOVERY OPTIMIZES
#
#     L = alpha * T^2 * KL(teacher || student) + (1 - alpha) * MSE(hidden states)
#
# The first term matches the output distributions, the second keeps the surviving blocks
# producing representations the later ones can still use. The mapping between student and
# teacher layers is the one the pruning produced, so block i of the student is compared with
# the teacher block it replaced.
#
# Usage:
#   slm_prune_distill.py --teacher Qwen/Qwen3-0.6B-Base --keep 12 --out ./qwen3-pruned
#   slm_prune_distill.py --teacher Qwen/Qwen3-0.6B-Base --keep 12 --steps 2000 \
#                        --selection importance --alpha 0.7 --temperature 2

import argparse
import json
import math
import os
import shutil
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("this script needs PyTorch: pip install torch")

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("this script needs transformers: pip install transformers")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("this script needs datasets: pip install datasets")


def block_list(model):
    """The decoder blocks, wherever this architecture keeps them.

    Every causal model in transformers holds them in a ModuleList, and the attribute path
    differs by family. Probing rather than assuming keeps the tool working across families
    without a table to maintain.
    """
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList):
                return path, obj
        except AttributeError:
            continue
    raise RuntimeError("cannot find the decoder blocks of this model")


def stream_batches(tokenizer, dataset_name, config_name, split, batch_size, seq_len,
                   count, skip=0):
    """Tokenized batches from a streaming corpus, so nothing is downloaded whole."""
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    buf, out = [], []
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text.strip():
            continue
        ids = tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < seq_len // 4:
            continue
        ids = ids + [tokenizer.pad_token_id] * (seq_len - len(ids))
        buf.append(ids[:seq_len])
        if len(buf) == batch_size:
            out.append(torch.tensor(buf))
            buf = []
            if len(out) >= count:
                break
    return out


@torch.no_grad()
def measure_block_influence(model, batches, device):
    """How much each block changes what passes through it.

    The score is one minus the cosine between a block's input and its output, averaged over
    positions and batches. A block whose output points the same way as its input is doing
    almost nothing, and is the cheapest to remove. This is the block-influence criterion,
    and it costs a handful of forward passes.
    """
    _, blocks = block_list(model)
    n = len(blocks)
    influence = [0.0] * n
    seen = 0

    for batch in batches:
        ids = batch.to(device)
        out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states          # n + 1 entries: the embedding, then each block
        for i in range(n):
            a = hs[i].float().flatten(0, 1)
            b = hs[i + 1].float().flatten(0, 1)
            cos = F.cosine_similarity(a, b, dim=-1).mean().item()
            influence[i] += 1.0 - cos
        seen += 1

    return [x / max(seen, 1) for x in influence]


def choose_blocks(n_total, n_keep, influence, mode):
    """Which block indices survive, in order.

    Importance keeps the highest scoring blocks. Uniform spreads the survivors evenly. In
    both cases the first and last blocks are kept whatever their score: the first sees the
    raw embedding and the last feeds the output norm, and removing either costs far more
    than their influence suggests.
    """
    if n_keep >= n_total:
        return list(range(n_total))

    if mode == "uniform":
        step = n_total / n_keep
        return sorted({min(n_total - 1, int(i * step)) for i in range(n_keep)})

    order = sorted(range(n_total), key=lambda i: influence[i], reverse=True)
    keep = {0, n_total - 1}
    for i in order:
        if len(keep) >= n_keep:
            break
        keep.add(i)
    return sorted(keep)


def build_student(teacher_name, kept, device, dtype):
    """A model of the same family with fewer blocks, inheriting the ones that stayed."""
    config = AutoConfig.from_pretrained(teacher_name)
    config.num_hidden_layers = len(kept)

    # Anything the configuration states once per layer has to be subset the same way.
    #
    # Recent architectures describe their blocks individually: layer_types says which
    # attention each one uses, and a hybrid model alternates full and sliding attention
    # along its depth. Those lists are derived at load time even when the published
    # config.json does not carry them, and they are validated against num_hidden_layers,
    # so a model pruned without them fails to save with a message about a mismatch rather
    # than about pruning. Subsetting by the kept indices also preserves the alternation:
    # dropping the last twelve entries instead would silently change which blocks attend
    # how.
    for field in ("layer_types", "cross_attention_layers", "full_attn_idxs"):
        value = getattr(config, field, None)
        if isinstance(value, (list, tuple)) and len(value) >= max(kept) + 1:
            setattr(config, field, [value[i] for i in kept])
    if getattr(config, "max_window_layers", None) is not None:
        config.max_window_layers = min(config.max_window_layers, len(kept))
    student = AutoModelForCausalLM.from_config(config)
    student = student.to(dtype=dtype)

    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, dtype=dtype)
    path, t_blocks = block_list(teacher)
    _, s_blocks = block_list(student)

    # Everything outside the blocks is copied as it is: embeddings, final norm, head.
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    prefix = path + "."
    carried = 0
    for name, tensor in t_state.items():
        if name.startswith(prefix):
            continue
        if name in s_state and s_state[name].shape == tensor.shape:
            s_state[name].copy_(tensor)
            carried += 1

    # Then the surviving blocks, in the order they were kept.
    for new_i, old_i in enumerate(kept):
        s_blocks[new_i].load_state_dict(t_blocks[old_i].state_dict())

    student.load_state_dict(s_state, strict=False)
    del teacher
    return student.to(device), carried


def save_tokenizer(tokenizer, out_dir):
    """Writes the tokenizer, then repairs what the round trip breaks.

    save_pretrained does not always produce a directory from_pretrained can read back.
    extra_special_tokens is written as an empty list and expected as a mapping, so a model
    saved this way fails to reload with an error about a list having no keys, far from
    anything a caller did. The converter hits it before it ever looks at a weight.

    The field is normalized here rather than left to whoever meets it next, and the result
    is reloaded to prove the directory is usable before the run reports success.
    """
    tokenizer.save_pretrained(out_dir)

    path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fin:
            cfg = json.load(fin)
        changed = False
        value = cfg.get("extra_special_tokens")
        if isinstance(value, list):
            cfg["extra_special_tokens"] = {}
            changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as fout:
                json.dump(cfg, fout, indent=2, ensure_ascii=False)

    try:
        AutoTokenizer.from_pretrained(out_dir)
    except Exception as e:                                   # noqa: BLE001
        print(f"  warning: the saved tokenizer does not reload ({e}).")
        print("  The conversion to GGUF will fail until this is resolved.")


def distillation_loss(s_out, t_out, mapping, temperature, alpha):
    """The two terms, and the T^2 that keeps them comparable."""
    T = temperature
    # Divided by the sequence length as well as the batch.
    #
    # batchmean divides by the first dimension only, so on a (batch, positions, vocabulary)
    # tensor it yields a divergence per sequence: a number in the hundreds that varies with
    # the window and compares with nothing. Per token, it reads in nats like every other
    # loss in this project, and a value near one means the student is one nat of surprise
    # away from its teacher.
    tokens = s_out.logits.shape[1]
    kl = F.kl_div(
        F.log_softmax(s_out.logits / T, dim=-1),
        F.log_softmax(t_out.logits / T, dim=-1),
        reduction="batchmean",
        log_target=True,
    ) * (T * T) / tokens

    hidden = torch.zeros((), device=s_out.logits.device)
    if alpha < 1.0 and mapping:
        for s_i, t_i in mapping.items():
            hidden = hidden + F.mse_loss(
                s_out.hidden_states[s_i + 1].float(),
                t_out.hidden_states[t_i + 1].float())
        hidden = hidden / len(mapping)

    return alpha * kl + (1.0 - alpha) * hidden, kl.item(), hidden.detach().item()


@torch.no_grad()
def evaluate(student, teacher, batches, device, temperature):
    """Fidelity on held-out text, measured as the divergence the training minimizes.

    Not the squared error between logits: two models can differ by a constant on every
    logit and produce the same distribution, and that constant would dominate a squared
    error while meaning nothing.
    """
    total = 0.0
    for batch in batches:
        ids = batch.to(device)
        t_out = teacher(input_ids=ids, return_dict=True)
        s_out = student(input_ids=ids, return_dict=True)
        total += (F.kl_div(
            F.log_softmax(s_out.logits / temperature, dim=-1),
            F.log_softmax(t_out.logits / temperature, dim=-1),
            reduction="batchmean", log_target=True)
            * temperature ** 2 / s_out.logits.shape[1]).item()
    return total / max(len(batches), 1)


def main():
    p = argparse.ArgumentParser(
        description="Prune a causal model in depth and distil the survivors back into shape.")
    p.add_argument("--teacher", required=True, help="model repository or local directory")
    p.add_argument("--keep", type=int, required=True, help="blocks the student retains")
    p.add_argument("--out", required=True, help="directory for the student, in HF layout")
    p.add_argument("--selection", choices=["importance", "uniform"], default="importance",
                   help="how the surviving blocks are chosen (default: importance)")

    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=500, help="training batches (default: 500)")
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--probe-batches", type=int, default=8,
                   help="batches used to measure block influence (default: 8)")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.9,
                   help="weight of the output divergence against the hidden-state term")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = p.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)

    # Said out loud, because the alternative is an hour of silence.
    #
    # Falling back to the processor when no accelerator answers is the right behaviour and
    # the wrong thing to keep quiet about: the run still works, a hundred times slower, and
    # nothing on screen distinguishes that from a hang. The most common cause is a torch
    # built against a CUDA version older than the card requires, which reports no device at
    # all rather than complaining.
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARNING: --device asked for cuda and torch reports no CUDA device.")
            print(f"         torch {torch.__version__}, built against CUDA "
                  f"{torch.version.cuda}. Falling back to the processor, which will be")
            print("         roughly two orders of magnitude slower for this work.")
            device = "cpu"
        else:
            name = torch.cuda.get_device_name(0)
            cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"device       : {name}, compute {cap}, {total:.1f} GB, "
                  f"torch {torch.__version__} on CUDA {torch.version.cuda}")
    else:
        print(f"device       : {device} (torch {torch.__version__})")

    print(f"teacher      : {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype).to(device).eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    _, t_blocks = block_list(teacher)
    n_total = len(t_blocks)
    print(f"blocks       : {n_total} -> {args.keep}")

    print("reading the corpus...")
    probe = stream_batches(tokenizer, args.dataset, args.dataset_config, args.split,
                           args.batch_size, args.seq_len, args.probe_batches)
    train = stream_batches(tokenizer, args.dataset, args.dataset_config, args.split,
                           args.batch_size, args.seq_len, args.steps,
                           skip=args.probe_batches * args.batch_size)
    held = stream_batches(tokenizer, args.dataset, args.dataset_config, args.split,
                          args.batch_size, args.seq_len, args.eval_batches,
                          skip=(args.probe_batches + args.steps) * args.batch_size)
    if not train:
        sys.exit("the corpus yielded no training batch")
    print(f"batches      : {len(train)} training, {len(held)} held out")

    influence = [1.0] * n_total
    if args.selection == "importance":
        print("measuring block influence...")
        influence = measure_block_influence(teacher, probe, device)
        ranked = sorted(range(n_total), key=lambda i: influence[i])
        print("  least influential :", ", ".join(f"{i}({influence[i]:.3f})" for i in ranked[:5]))
        print("  most influential  :", ", ".join(f"{i}({influence[i]:.3f})" for i in ranked[-5:]))

    kept = choose_blocks(n_total, args.keep, influence, args.selection)
    print(f"keeping      : {kept}")

    student, carried = build_student(args.teacher, kept, device, dtype)
    mapping = {new_i: old_i for new_i, old_i in enumerate(kept)}
    print(f"inherited    : {carried} tensors outside the blocks, plus {len(kept)} blocks")

    n_s = sum(x.numel() for x in student.parameters())
    n_t = sum(x.numel() for x in teacher.parameters())
    print(f"parameters   : {n_t} -> {n_s}  ({100.0 * n_s / n_t:.1f}%)")

    before = evaluate(student, teacher, held, device, args.temperature)
    print(f"divergence before recovery : {before:.5f} nats per token")

    print(f"training     : {len(train)} batches of {args.batch_size} x {args.seq_len} "
          f"tokens on {device}")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    # The schedule counts optimizer steps, not batches.
    #
    # With gradient accumulation the optimizer runs once every grad_accum batches, so a
    # cosine period sized on the number of batches traverses only that fraction of itself
    # and the rate never anneals. The symptom is a run that ends at nearly its starting
    # rate, still bouncing around a plateau it would have settled below.
    updates = max(len(train) // max(args.grad_accum, 1), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    student.train()

    running = 0.0
    t0 = time.time()
    for step, batch in enumerate(train, 1):
        ids = batch.to(device)
        with torch.no_grad():
            t_out = teacher(input_ids=ids, output_hidden_states=(args.alpha < 1.0),
                            return_dict=True)
        s_out = student(input_ids=ids, output_hidden_states=(args.alpha < 1.0),
                        return_dict=True)
        loss, kl, hid = distillation_loss(s_out, t_out, mapping if args.alpha < 1.0 else {},
                                          args.temperature, args.alpha)
        (loss / args.grad_accum).backward()
        running += loss.item()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

        # The first few steps are reported one by one, then every twenty-five.
        #
        # A run whose first message arrives after twenty-five steps looks identical to a run
        # that is stuck, and the difference matters most exactly when the steps are slow.
        # Announcing the first one tells a caller within seconds whether anything is moving,
        # and at what pace.
        report = step <= 3 or step % 25 == 0 or step == len(train)
        if report:
            window = 1 if step <= 3 else min(25, step)
            elapsed = time.time() - t0
            print(f"  step {step:>5}/{len(train)}  loss {running / window:.4f}"
                  f"  kl {kl:.4f}  hidden {hid:.4f}  lr {sched.get_last_lr()[0]:.2e}"
                  f"  {elapsed / step:.2f} s/step", flush=True)
            if step > 3:
                running = 0.0
            else:
                running = 0.0

    student.eval()
    after = evaluate(student, teacher, held, device, args.temperature)
    print(f"divergence after recovery  : {after:.5f} nats per token  "
          f"({100.0 * (before - after) / max(before, 1e-9):.1f}% closer)")

    os.makedirs(args.out, exist_ok=True)
    student.save_pretrained(args.out, safe_serialization=True)
    save_tokenizer(tokenizer, args.out)

    with open(os.path.join(args.out, "pruning.json"), "w", encoding="utf-8") as fout:
        json.dump({"teacher": args.teacher, "kept_blocks": kept,
                   "selection": args.selection, "influence": influence,
                   "divergence_before": before, "divergence_after": after,
                   "steps": len(train), "alpha": args.alpha,
                   "temperature": args.temperature}, fout, indent=2)

    print(f"\nwritten to {args.out}")
    print("\nThe student is an ordinary model of its family with fewer blocks, so the")
    print("standard converter produces a container this library reads:")
    print(f"  python3 convert_hf_to_gguf.py {args.out} --outtype f16 --outfile student.gguf")
    print("\nThen, from the build directory:")
    print("  ./slm_gguf_runtime_ex --input student.gguf --probe-logits")
    print("  ./slm_gguf_import_ex  --input student.gguf --out-prefix slm_imported_model")


if __name__ == "__main__":
    main()
