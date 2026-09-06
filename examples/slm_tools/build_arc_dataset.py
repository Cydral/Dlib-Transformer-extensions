#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# ARC-AGI dataset preparation for the hierarchical reasoning example.
#
# Turns the ARC-AGI JSON challenges into the flat arrays the HRM training loop reads,
# in the layout the reference implementation uses, so that a dataset built here can be
# fed to either codebase and the two sets of results compared on the same data.
#
# WHAT THE LAYOUT IS, AND WHY IT LOOKS LIKE THIS
#
# Each input-output pair is one example. A grid is flattened row by row into a fixed
# window of 900 cells, which is the largest ARC grid, 30 by 30. Cells outside the grid
# hold the pad token, so every example has the same length and no length bookkeeping
# is needed at training time.
#
# The conditioning is the part worth understanding. The model is shown the input grid
# and nothing else: no worked examples of the same task accompany it. What tells it
# which transformation to apply is an integer, the puzzle identifier, which indexes a
# learned embedding prepended to the sequence. The model therefore learns to associate
# an identifier with a rule, rather than to infer a rule from demonstrations.
#
# This has a consequence that has to be accepted deliberately: an identifier the model
# has never trained on means nothing to it, so evaluation puzzles must be present in
# the training set. That is how the reference results were obtained and it is what
# makes them comparable; it is also why those results do not describe a model that
# generalises to unseen tasks. The example program can be run either way, and the
# choice belongs to whoever reads the numbers.
#
# AUGMENTATION
#
# ARC ships a few hundred tasks with a handful of pairs each, which is far too little.
# The reference work multiplies this by applying the transformations that leave the
# nature of a puzzle intact: the eight rotations and reflections of the square, a
# permutation of the ten colours, and a translation within the grid. Each augmentation
# receives its own puzzle identifier, since it is a different rule from the model's
# point of view even though a human would call it the same task.
#
# Augmentation is optional here. Building without it is the honest baseline and shows
# how much of the result the augmentation is responsible for.
#
# GROUPING
#
# Two index arrays describe the nesting. puzzle_indices gives, for each puzzle, where
# its examples begin; group_indices gives, for each group, where its puzzles begin. A
# group holds one original task together with all of its augmentations, which lets a
# sampler draw one variant per task per epoch rather than letting a heavily augmented
# task dominate.
#
# OUTPUT
#
# Two formats are written by default. The .npy set is the reference layout, readable
# by numpy and by the reference training code. The .bin file packs the same arrays
# behind a small header for the C++ side, which needs no numpy and no parsing.
#
# Usage:
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --augmentations 300
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --augmentations 0
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --protocol reference
#   build_arc_dataset.py --download --out-dir arc-data --augmentations 1000

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

# The window is the largest ARC grid, so no example ever has to be truncated.
GRID_SIDE = 30
SEQ_LEN = GRID_SIDE * GRID_SIDE

# Token 0 is the pad, colours 0 to 9 become tokens 1 to 10. Keeping the pad at zero
# means an unwritten cell is already correct and the fill costs nothing.
PAD_TOKEN = 0
NUM_COLOURS = 10
VOCAB_SIZE = NUM_COLOURS + 1

# Identifier 0 is reserved for "no puzzle", which is what an evaluation run uses when
# it wants to ask the model to work without being told which rule to apply.
BLANK_PUZZLE_ID = 0

ARC1_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"


# ----------------------------------------------------------------------------------
# Reading the source data


def read_task_directory(path: str) -> Dict[str, dict]:
    """Reads one directory of ARC task files, keyed by the task name."""
    tasks = {}
    if not os.path.isdir(path):
        return tasks
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(path, name), "r", encoding="utf-8") as f:
            tasks[os.path.splitext(name)[0]] = json.load(f)
    return tasks


def read_challenge_file(challenges: str, solutions: str) -> Dict[str, dict]:
    """Reads the paired challenge and solution files used by the later ARC releases."""
    with open(challenges, "r", encoding="utf-8") as f:
        ch = json.load(f)
    sol = {}
    if solutions and os.path.exists(solutions):
        with open(solutions, "r", encoding="utf-8") as f:
            sol = json.load(f)

    tasks = {}
    for name, task in ch.items():
        merged = {"train": task.get("train", []), "test": []}
        for i, item in enumerate(task.get("test", [])):
            pair = {"input": item["input"]}
            if name in sol and i < len(sol[name]):
                pair["output"] = sol[name][i]
            elif "output" in item:
                pair["output"] = item["output"]
            else:
                continue                      # no answer available, so nothing to learn
            merged["test"].append(pair)
        tasks[name] = merged
    return tasks


def load_split(data_dir: str, split: str) -> Dict[str, dict]:
    """Finds whichever of the two ARC distributions is on disk."""
    per_task = os.path.join(data_dir, split)
    tasks = read_task_directory(per_task)
    if tasks:
        return tasks

    challenges = os.path.join(data_dir, f"arc-agi_{split}_challenges.json")
    solutions = os.path.join(data_dir, f"arc-agi_{split}_solutions.json")
    if os.path.exists(challenges):
        return read_challenge_file(challenges, solutions)
    return {}


def download_arc1(dest: str) -> None:
    """Fetches the original ARC-AGI task files, which are one JSON per task."""
    index_url = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data"
    for split in ("training", "evaluation"):
        out = os.path.join(dest, split)
        os.makedirs(out, exist_ok=True)
        with urllib.request.urlopen(f"{index_url}/{split}") as r:
            listing = json.load(r)
        print(f"  {split}: {len(listing)} tasks", flush=True)
        for i, entry in enumerate(listing):
            target = os.path.join(out, entry["name"])
            if os.path.exists(target):
                continue
            with urllib.request.urlopen(entry["download_url"]) as r:
                data = r.read()
            with open(target, "wb") as f:
                f.write(data)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(listing)}", flush=True)


# ----------------------------------------------------------------------------------
# Grid handling


Grid = List[List[int]]


def encode_grid(grid: Grid) -> List[int]:
    """Flattens a grid into the fixed window, padding what the grid does not reach."""
    out = [PAD_TOKEN] * SEQ_LEN
    for r, row in enumerate(grid[:GRID_SIDE]):
        base = r * GRID_SIDE
        for c, v in enumerate(row[:GRID_SIDE]):
            out[base + c] = v + 1                     # colour 0 must not collide with pad
    return out


def rotate(grid: Grid) -> Grid:
    """A quarter turn clockwise."""
    return [list(row) for row in zip(*grid[::-1])]


def flip(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def dihedral(grid: Grid, index: int) -> Grid:
    """One of the eight symmetries of the square, index 0 being the identity."""
    g = grid
    for _ in range(index % 4):
        g = rotate(g)
    return flip(g) if index >= 4 else g


def recolour(grid: Grid, mapping: List[int]) -> Grid:
    return [[mapping[v] for v in row] for row in grid]


def translate(grid: Grid, dr: int, dc: int) -> Grid:
    """Shifts within the window, filling what is vacated with colour zero.

    A translation only makes sense while the grid still fits, so the caller checks the
    room available before asking for one.
    """
    h, w = len(grid), len(grid[0])
    out = [[0] * (w + abs(dc)) for _ in range(h + abs(dr))]
    for r in range(h):
        for c in range(w):
            out[r + max(dr, 0)][c + max(dc, 0)] = grid[r][c]
    return out


def augment_pair(pair: Tuple[Grid, Grid], rng, allow_translate: bool
                 ) -> Tuple[Grid, Grid]:
    """Applies one draw of the transformations that leave a puzzle's nature intact.

    The same draw is applied to the input and to the output, since a rule is only
    preserved when both sides move together.
    """
    gi, go = pair
    d = rng.randrange(8)
    gi, go = dihedral(gi, d), dihedral(go, d)

    perm = list(range(NUM_COLOURS))
    rng.shuffle(perm)
    gi, go = recolour(gi, perm), recolour(go, perm)

    if allow_translate:
        room_r = GRID_SIDE - max(len(gi), len(go))
        room_c = GRID_SIDE - max(len(gi[0]), len(go[0]))
        if room_r > 0 or room_c > 0:
            dr = rng.randint(0, room_r) if room_r > 0 else 0
            dc = rng.randint(0, room_c) if room_c > 0 else 0
            if dr or dc:
                gi, go = translate(gi, dr, dc), translate(go, dr, dc)
    return gi, go


# ----------------------------------------------------------------------------------
# Building


class Builder:
    """Accumulates examples and the two index arrays that describe their nesting."""

    def __init__(self):
        self.inputs: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.puzzle_identifiers: List[int] = [BLANK_PUZZLE_ID]
        self.puzzle_indices: List[int] = [0]
        self.group_indices: List[int] = [0]
        self.names: List[str] = ["<blank>"]

    def add_puzzle(self, name: str, pairs: List[Tuple[Grid, Grid]]) -> None:
        for gi, go in pairs:
            self.inputs.append(encode_grid(gi))
            self.labels.append(encode_grid(go))
        self.puzzle_indices.append(len(self.inputs))
        self.puzzle_identifiers.append(len(self.puzzle_identifiers))
        self.names.append(name)

    def close_group(self) -> None:
        self.group_indices.append(len(self.puzzle_indices) - 1)


def task_rng(name: str, seed: int):
    """A generator tied to the task name, so a task augments identically in both splits.

    This is what lets the reference protocol work: a puzzle keeps the same identifier
    and the same transformation whether it is met while training or while being asked
    for its held-out answer.
    """
    import random
    return random.Random(seed ^ int(hashlib.md5(name.encode()).hexdigest()[:8], 16))


def build_split(tasks: Dict[str, dict], augmentations: int, seed: int,
                use_test_pairs: bool, allow_translate: bool) -> Builder:
    """Independent identifier space, one puzzle per task and per augmentation."""
    b = Builder()
    for name in sorted(tasks):
        task = tasks[name]
        pairs = [(p["input"], p["output"]) for p in task.get("train", [])
                 if "output" in p]
        if use_test_pairs:
            pairs += [(p["input"], p["output"]) for p in task.get("test", [])
                      if "output" in p]
        if not pairs:
            continue

        b.add_puzzle(name, pairs)                     # the original, always identifier one
        rng = task_rng(name, seed)
        for a in range(augmentations):
            b.add_puzzle(f"{name}#{a}",
                         [augment_pair(p, rng, allow_translate) for p in pairs])
        b.close_group()
    return b


def build_reference_protocol(train_tasks: Dict[str, dict], eval_tasks: Dict[str, dict],
                             augmentations: int, seed: int, allow_translate: bool
                             ) -> Tuple[Builder, Builder]:
    """Builds both splits over one shared identifier space.

    An identifier the model never trained on carries no meaning: its row of the
    embedding is still at its initial value, so asking a question by naming it asks
    nothing. Two independently numbered splits therefore cannot be used to evaluate
    one another, which is the constraint the reference protocol answers.

    The answer is to let every task, evaluation ones included, receive its identifier
    while training, from its demonstration pairs alone. The held-out pair of each
    evaluation task goes to the other split under the same identifier, so the question
    put at evaluation time is one the model can understand and has not seen answered.

    This is a form of learning at test time and it is why those numbers are what they
    are. It is offered because it is what makes a comparison possible, not because it
    is the harder setting; --protocol held-out is that.
    """
    tr, ev = Builder(), Builder()

    def emit(builder, name, pairs, rng_state_name):
        """Adds a task and its augmentations, drawing the same transformations both times."""
        builder.add_puzzle(name, pairs)
        rng = task_rng(rng_state_name, seed)
        for a in range(augmentations):
            builder.add_puzzle(f"{name}#{a}",
                               [augment_pair(p, rng, allow_translate) for p in pairs])
        builder.close_group()

    # The training tasks contribute everything they have and are never asked again.
    for name in sorted(train_tasks):
        pairs = [(p["input"], p["output"]) for p in train_tasks[name].get("train", [])
                 if "output" in p]
        pairs += [(p["input"], p["output"]) for p in train_tasks[name].get("test", [])
                  if "output" in p]
        if pairs:
            emit(tr, name, pairs, name)

    # An evaluation task gives its demonstrations to the training split and its held-out
    # pair to the evaluation split. Both use the same name, so both draw the same
    # augmentations and land on the same identifiers.
    for name in sorted(eval_tasks):
        demo = [(p["input"], p["output"]) for p in eval_tasks[name].get("train", [])
                if "output" in p]
        held = [(p["input"], p["output"]) for p in eval_tasks[name].get("test", [])
                if "output" in p]
        if not demo or not held:
            continue
        emit(tr, name, demo, name)
        emit(ev, name, held, name)

    # The evaluation identifiers must be the ones the training split assigned, which the
    # shared naming guarantees; the check is cheap and a silent mismatch would be fatal.
    index = {n: i for i, n in enumerate(tr.names)}
    ev.puzzle_identifiers = [index.get(n, BLANK_PUZZLE_ID) for n in ev.names]
    missing = [n for n in ev.names[1:] if n not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} evaluation puzzles have no identifier in the "
                           f"training split, starting with {missing[0]}")
    return tr, ev


# ----------------------------------------------------------------------------------
# Writing


def write_npy(out_dir: str, split: str, b: Builder) -> None:
    if np is None:
        print("  numpy is not installed, so the .npy set is skipped", file=sys.stderr)
        return
    d = os.path.join(out_dir, split)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "all__inputs.npy"), np.array(b.inputs, dtype=np.int32))
    np.save(os.path.join(d, "all__labels.npy"), np.array(b.labels, dtype=np.int32))
    np.save(os.path.join(d, "all__puzzle_identifiers.npy"),
            np.array(b.puzzle_identifiers, dtype=np.int32))
    np.save(os.path.join(d, "all__puzzle_indices.npy"),
            np.array(b.puzzle_indices, dtype=np.int32))
    np.save(os.path.join(d, "all__group_indices.npy"),
            np.array(b.group_indices, dtype=np.int32))


def write_bin(out_dir: str, split: str, b: Builder) -> str:
    """Packs the same arrays behind a header the C++ loader reads in one pass.

    Everything is little-endian int32 after a fixed header, so the reader needs no
    parsing and no third-party library.
    """
    path = os.path.join(out_dir, f"{split}.bin")
    with open(path, "wb") as f:
        f.write(b"ARCD")
        f.write(struct.pack("<7i", 1, SEQ_LEN, VOCAB_SIZE,
                            len(b.inputs), len(b.puzzle_identifiers),
                            len(b.puzzle_indices), len(b.group_indices)))
        for seq in b.inputs:
            f.write(struct.pack(f"<{SEQ_LEN}i", *seq))
        for seq in b.labels:
            f.write(struct.pack(f"<{SEQ_LEN}i", *seq))
        f.write(struct.pack(f"<{len(b.puzzle_identifiers)}i", *b.puzzle_identifiers))
        f.write(struct.pack(f"<{len(b.puzzle_indices)}i", *b.puzzle_indices))
        f.write(struct.pack(f"<{len(b.group_indices)}i", *b.group_indices))
    return path


def write_metadata(out_dir: str, split: str, b: Builder, args) -> None:
    meta = {
        "split": split,
        "seq_len": SEQ_LEN,
        "grid_side": GRID_SIDE,
        "vocab_size": VOCAB_SIZE,
        "pad_token": PAD_TOKEN,
        "blank_puzzle_id": BLANK_PUZZLE_ID,
        "num_examples": len(b.inputs),
        "num_puzzles": len(b.puzzle_identifiers) - 1,
        "num_groups": len(b.group_indices) - 1,
        "protocol": args.protocol,
        "augmentations_per_task": args.augmentations,
        "translations": bool(args.translations),
        "test_pairs_included": bool(args.include_test_pairs),
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ----------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Prepare ARC-AGI for the hierarchical reasoning example.")
    p.add_argument("--data-dir", default="arc-agi",
                   help="directory holding the ARC JSON files")
    p.add_argument("--out-dir", default="arc-data",
                   help="where the prepared arrays are written")
    p.add_argument("--download", action="store_true",
                   help="fetch the original ARC-AGI task files into --data-dir first")
    p.add_argument("--augmentations", type=int, default=1000,
                   help="variants generated per task, 0 for none (default: 1000)")
    p.add_argument("--translations", action="store_true", default=True,
                   help="include translation among the augmentations (default: on)")
    p.add_argument("--no-translations", dest="translations", action="store_false")
    p.add_argument("--protocol", choices=("held-out", "reference"), default="held-out",
                   help="held-out numbers the two splits independently, so evaluation "
                        "must be run with the blank identifier and the model is asked to "
                        "work without being told the rule. reference gives every task, "
                        "evaluation ones included, an identifier while training, from its "
                        "demonstration pairs alone, and puts only the held-out pair in the "
                        "evaluation split; that is what makes the published numbers "
                        "comparable and it is learning at test time (default: held-out)")
    p.add_argument("--include-test-pairs", action="store_true",
                   help="under held-out only, add each task's held-out pairs to its own "
                        "training examples")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--format", choices=("npy", "bin", "both"), default="both")
    args = p.parse_args()

    if args.download:
        print(f"Fetching ARC-AGI into {args.data_dir}")
        os.makedirs(args.data_dir, exist_ok=True)
        download_arc1(args.data_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    train_tasks = load_split(args.data_dir, "training")
    eval_tasks = load_split(args.data_dir, "evaluation")
    if not train_tasks and not eval_tasks:
        print(f"nothing found in {args.data_dir}")
        return 1

    if args.protocol == "reference":
        tr, ev = build_reference_protocol(train_tasks, eval_tasks, args.augmentations,
                                          args.seed, args.translations)
        built = {"training": tr, "evaluation": ev}
        counts = {"training": len(train_tasks) + len(eval_tasks),
                  "evaluation": len(eval_tasks)}
    else:
        built, counts = {}, {}
        for split, tasks in (("training", train_tasks), ("evaluation", eval_tasks)):
            if not tasks:
                continue
            use_test = args.include_test_pairs and split == "training"
            built[split] = build_split(tasks, args.augmentations, args.seed, use_test,
                                       args.translations)
            counts[split] = len(tasks)

    for split, b in built.items():
        if args.format in ("npy", "both"):
            write_npy(args.out_dir, split, b)
        if args.format in ("bin", "both"):
            write_bin(args.out_dir, split, b)
        meta = write_metadata(args.out_dir, split, b, args)
        print(f"{split}: {counts[split]} tasks, {meta['num_puzzles']} puzzles, "
              f"{meta['num_examples']} examples, {meta['num_groups']} groups")

    if args.protocol == "reference":
        print("\nThe two splits share one identifier space: every evaluation puzzle was "
              "named during training,\nfrom its demonstration pairs only. Its held-out "
              "answer is what the evaluation split asks for.")
    else:
        print("\nThe two splits are numbered independently, so evaluation identifiers "
              "mean nothing to a model\ntrained on the other split. Evaluate with the "
              "blank identifier, or build with --protocol reference.")

    print(f"\nWritten to {args.out_dir}")
    print(f"window {SEQ_LEN} cells, vocabulary {VOCAB_SIZE}, pad token {PAD_TOKEN}")


if __name__ == "__main__":
    main()
