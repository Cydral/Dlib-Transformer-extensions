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


# Dataset samples skipped per batch reserved, to keep the regions of the stream apart.
#
# Not every sample yields a usable sequence: blanks and short entries are dropped, so a
# batch consumes more samples than it holds. Skipping generously is cheap on a streaming
# corpus and is what keeps the evaluation from meeting the training text.
OVERSHOOT = 8


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
    """Batches of tokens with their attention masks, yielded as they are produced.

    Two things here are not incidental.

    The mask travels with the batch. A sequence shorter than the window is padded, and a
    model given no mask attends over that padding as if it were text. Both models would do
    so identically, so the divergence between them stays small and says nothing: the loss
    would be measured largely on positions that carry no text at all, and on this corpus
    that is up to three quarters of them.

    And batches are yielded rather than accumulated. Tokenizing forty thousand of them
    before the first step is several minutes of silence, and holding them is several hundred
    megabytes that the training loop never needs more than one of.
    """
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    pad = tokenizer.pad_token_id
    ids_buf, mask_buf, made = [], [], 0
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text.strip():
            continue
        ids = tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < seq_len // 4:
            continue
        real = len(ids)
        ids = ids[:seq_len] + [pad] * (seq_len - real)
        ids_buf.append(ids)
        mask_buf.append([1] * min(real, seq_len) + [0] * (seq_len - real))
        if len(ids_buf) == batch_size:
            yield torch.tensor(ids_buf), torch.tensor(mask_buf)
            ids_buf, mask_buf = [], []
            made += 1
            if made >= count:
                return


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

    for ids, mask in batches:
        ids, mask = ids.to(device), mask.to(device)
        out = model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=True, return_dict=True)
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


def save_tokenizer(tokenizer, teacher_name, out_dir):
    """Copies the teacher's tokenizer files rather than re-serializing them.

    save_pretrained rebuilds tokenizer.json from the loaded objects, and the rebuilt file is
    not always byte-identical to the published one: transformers rewrites the pre-tokenizer
    regex in a form its own loader then flags as incorrect, quoting a discussion about a
    Mistral model because that is where the pattern was first reported. The warning names a
    company that has nothing to do with the model at hand, which is why it reads as noise
    and is worth taking seriously anyway: a rewritten regex is a rewritten tokenizer.

    A student inherits its teacher's vocabulary unchanged, so there is nothing to rebuild.
    The files are copied as they were published, and the result is byte-identical
    tokenization by construction rather than by hope.
    """
    os.makedirs(out_dir, exist_ok=True)
    wanted = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
              "chat_template.jinja"]
    copied = []

    if os.path.isdir(teacher_name):
        for name in wanted:
            src = os.path.join(teacher_name, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, name))
                copied.append(name)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            hf_hub_download = None
        if hf_hub_download is not None:
            for name in wanted:
                try:
                    src = hf_hub_download(repo_id=teacher_name, filename=name)
                    shutil.copy2(src, os.path.join(out_dir, name))
                    copied.append(name)
                except Exception:                                # noqa: BLE001
                    continue      # the file simply does not exist for this model

    if not copied:
        print("  note: the teacher's tokenizer files could not be copied; falling back to")
        print("        re-serializing them, which may alter the pre-tokenizer regex.")
        tokenizer.save_pretrained(out_dir)

    # Whichever path was taken, prove the directory reloads before reporting success.
    try:
        AutoTokenizer.from_pretrained(out_dir)
        print(f"  tokenizer    : {len(copied)} files copied verbatim from the teacher"
              if copied else "  tokenizer    : re-serialized")
    except Exception as e:                                       # noqa: BLE001
        print(f"  warning: the saved tokenizer does not reload ({e}).")
        print("  The conversion to GGUF will fail until this is resolved.")


def masked_kl(s_logits, t_logits, mask, temperature):
    """Divergence over the positions that carry text, per token.

    Padding is excluded rather than tolerated. Both models agree trivially on filler, so
    including it drags the average towards zero and hides how far apart they are where it
    matters; it also spends gradient on teaching a student to imitate a teacher's opinion
    about nothing.
    """
    T = temperature
    per_pos = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.log_softmax(t_logits / T, dim=-1),
        reduction="none", log_target=True).sum(dim=-1)
    m = mask.to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum().clamp(min=1) * (T * T)


def distillation_loss(s_out, t_out, mask, mapping, temperature, alpha):
    """The two terms, and the T^2 that keeps them comparable."""
    T = temperature
    kl = masked_kl(s_out.logits, t_out.logits, mask, T)

    hidden = torch.zeros((), device=s_out.logits.device)
    if alpha < 1.0 and mapping:
        # Averaged over the real positions and over the width, which is what makes this
        # a mean squared error rather than a sum. Dividing by anything else leaves the
        # term scaled by the hidden dimension, and at a width of a thousand it then
        # dwarfs the divergence it is meant to accompany.
        m = mask.to(torch.float32).unsqueeze(-1)
        width = s_out.hidden_states[0].shape[-1]
        denom = (m.sum() * width).clamp(min=1)
        for s_i, t_i in mapping.items():
            diff = (s_out.hidden_states[s_i + 1].float()
                    - t_out.hidden_states[t_i + 1].float())
            hidden = hidden + ((diff * diff) * m).sum() / denom
        hidden = hidden / len(mapping)

    return alpha * kl + (1.0 - alpha) * hidden, kl.item(), hidden.detach().item()


@torch.no_grad()
def evaluate(student, teacher, batches, device, temperature):
    """Fidelity on held-out text, measured as the divergence the training minimizes.

    Not the squared error between logits: two models can differ by a constant on every
    logit and produce the same distribution, and that constant would dominate a squared
    error while meaning nothing.
    """
    was_training = student.training
    student.eval()
    total, seen = 0.0, 0
    for ids, mask in batches:
        ids, mask = ids.to(device), mask.to(device)
        t_out = teacher(input_ids=ids, attention_mask=mask, return_dict=True)
        s_out = student(input_ids=ids, attention_mask=mask, return_dict=True)
        total += masked_kl(s_out.logits, t_out.logits, mask, temperature).item()
        seen += 1
    if was_training:
        student.train()
    return total / max(seen, 1)


def main():
    p = argparse.ArgumentParser(
        description="Prune a causal model in depth and distil the survivors back into shape.")
    p.add_argument("--teacher", required=True, help="model repository or local directory")
    p.add_argument("--keep", type=int, required=True, help="blocks the student retains")
    p.add_argument("--out", help="directory for the student; derived from the naming "
                   "convention when omitted")
    p.add_argument("--model-name", help="name written inside the model, and used to derive "
                   "the directory. Defaults to the convention below")
    p.add_argument("--family", default="Dlib-SLM",
                   help="first element of the name (default: Dlib-SLM)")
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
    p.add_argument("--seed", type=int, default=1234,
                   help="fixes the initialization of what the student does not inherit, so "
                        "two runs of the same command can be compared (default: 1234)")
    args = p.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

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

    teacher_cfg = AutoConfig.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype).to(device).eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    _, t_blocks = block_list(teacher)
    n_total = len(t_blocks)
    print(f"blocks       : {n_total} -> {args.keep}")

    # The held-out batches come first in the stream, then the probe, then training.
    #
    # A held-out set taken after the training set has to skip past it, and the skip counts
    # dataset samples while the training loop counts batches: samples too short are dropped,
    # so the two never line up and the evaluation ends up measuring text the student was
    # trained on. Taking the evaluation from the head of the stream makes the boundary exact
    # in the only direction that matters, and OVERSHOOT keeps a wide margin between the
    # regions so the filtering cannot close it.
    def make(count, skip):
        return stream_batches(tokenizer, args.dataset, args.dataset_config, args.split,
                              args.batch_size, args.seq_len, count, skip=skip)

    held_skip = 0
    probe_skip = args.eval_batches * args.batch_size * OVERSHOOT
    train_skip = probe_skip + args.probe_batches * args.batch_size * OVERSHOOT

    print("reading the corpus...")
    held = list(make(args.eval_batches, held_skip))
    probe = list(make(args.probe_batches, probe_skip))
    if not held:
        sys.exit("the corpus yielded no held-out batch")
    print(f"batches      : {args.steps} training (streamed), {len(held)} held out")

    influence = [1.0] * n_total
    if args.selection == "importance":
        print("measuring block influence...")
        influence = measure_block_influence(teacher, probe, device)
        ranked = sorted(range(n_total), key=lambda i: influence[i])
        print("  least influential :", ", ".join(f"{i}({influence[i]:.3f})" for i in ranked[:5]))
        print("  most influential  :", ", ".join(f"{i}({influence[i]:.3f})" for i in ranked[-5:]))

    kept = choose_blocks(n_total, args.keep, influence, args.selection)
    print(f"keeping      : {kept}")

    # The name, and the directory derived from it.
    #
    # A model that circulates is identified by what it is, not by the folder it happened to
    # be written into. The convention states the family, the width and the depth:
    # Dlib-SLM-1024x24.
    #
    # The teacher is deliberately absent from it. The geometry already implies the lineage
    # for anyone who looks, the model is meant to stand on its own rather than as a derived
    # work, and a name that carries a provenance ages badly the moment the same shape is
    # reached another way. What produced it stays recorded in pruning.json beside the
    # weights, where it belongs.
    #
    # The same string goes into config.json, so the name travels into the GGUF container and
    # shows up in every report this library prints. A file renamed on disk still says what
    # it is.
    name = args.model_name or f"{args.family}-{teacher_cfg.hidden_size}x{len(kept)}"
    out_dir = args.out or os.path.join(os.path.expanduser("~/models"), name.lower())
    print(f"name         : {name}")
    print(f"directory    : {out_dir}")

    student, carried = build_student(args.teacher, kept, device, dtype)
    student.config.name_or_path = name
    if hasattr(student.config, "_name_or_path"):
        student.config._name_or_path = name
    mapping = {new_i: old_i for new_i, old_i in enumerate(kept)}
    print(f"inherited    : {carried} tensors outside the blocks, plus {len(kept)} blocks")

    n_s = sum(x.numel() for x in student.parameters())
    n_t = sum(x.numel() for x in teacher.parameters())
    print(f"parameters   : {n_t} -> {n_s}  ({100.0 * n_s / n_t:.1f}%)")

    before = evaluate(student, teacher, held, device, args.temperature)
    print(f"divergence before recovery : {before:.5f} nats per token")

    print(f"training     : {args.steps} batches of {args.batch_size} x {args.seq_len} "
          f"tokens on {device}")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    # The schedule counts optimizer steps, not batches.
    #
    # With gradient accumulation the optimizer runs once every grad_accum batches, so a
    # cosine period sized on the number of batches traverses only that fraction of itself
    # and the rate never anneals. The symptom is a run that ends at nearly its starting
    # rate, still bouncing around a plateau it would have settled below.
    updates = max(args.steps // max(args.grad_accum, 1), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    student.train()

    running = 0.0
    t0 = time.time()
    train = make(args.steps, train_skip)
    interrupted = False
    step = 0
    try:
      for step, (ids, mask) in enumerate(train, 1):
          ids, mask = ids.to(device), mask.to(device)
          with torch.no_grad():
              t_out = teacher(input_ids=ids, attention_mask=mask,
                              output_hidden_states=(args.alpha < 1.0), return_dict=True)
          s_out = student(input_ids=ids, attention_mask=mask,
                          output_hidden_states=(args.alpha < 1.0), return_dict=True)
          loss, kl, hid = distillation_loss(s_out, t_out, mask,
                                            mapping if args.alpha < 1.0 else {},
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
          report = step <= 3 or step % 25 == 0 or step == args.steps
          if report:
              window = 1 if step <= 3 else min(25, step)
              elapsed = time.time() - t0
              print(f"  step {step:>5}/{args.steps}  loss {running / window:.4f}"
                    f"  kl {kl:.4f}  hidden {hid:.4f}  lr {sched.get_last_lr()[0]:.2e}"
                    f"  {elapsed / step:.2f} s/step", flush=True)
              if step > 3:
                  running = 0.0
              else:
                  running = 0.0

    except KeyboardInterrupt:
        # Interrupted, not abandoned: what the student has learned is measured and
        # written, so hours of recovery are not lost to a keystroke.
        interrupted = True
        print(f"\n  interrupted at step {step}; the student is kept as it stands")

    student.eval()
    after = evaluate(student, teacher, held, device, args.temperature)
    print(f"divergence after recovery  : {after:.5f} nats per token  "
          f"({100.0 * (before - after) / max(before, 1e-9):.1f}% closer)")

    os.makedirs(out_dir, exist_ok=True)
    student.save_pretrained(out_dir, safe_serialization=True)
    save_tokenizer(tokenizer, args.teacher, out_dir)

    with open(os.path.join(out_dir, "pruning.json"), "w", encoding="utf-8") as fout:
        json.dump({"teacher": args.teacher, "kept_blocks": kept,
                   "selection": args.selection, "influence": influence,
                   "divergence_before": before, "divergence_after": after,
                   "steps": step, "steps_requested": args.steps,
                   "interrupted": interrupted, "seed": args.seed, "alpha": args.alpha,
                   "temperature": args.temperature}, fout, indent=2)

    print(f"\nwritten to {out_dir}")
    print("\nThe student is an ordinary model of its family with fewer blocks, so the")
    print("standard converter produces a container this library reads:")
    print(f"  python3 convert_hf_to_gguf.py {out_dir} --outtype f16 --outfile student.gguf")
    print("\nThen, from the build directory:")
    print("  ./slm_gguf_runtime_ex --input student.gguf --probe-logits")
    print("  ./slm_gguf_import_ex  --input student.gguf --out-prefix slm_imported_model")


if __name__ == "__main__":
    main()
