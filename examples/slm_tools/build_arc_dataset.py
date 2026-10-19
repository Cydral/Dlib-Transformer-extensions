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
# Each input-output pair is one example. A grid is written into a fixed window of 900
# cells, the size of the largest ARC grid, 30 by 30, and the window is flattened row by
# row. Cells outside the grid hold the pad token, so every example has the same length
# and no length bookkeeping is needed at training time.
#
# The conditioning is the part worth understanding. The model is shown the input grid
# and nothing else: no worked examples of the same task accompany it. What tells it
# which transformation to apply is an integer, the puzzle identifier, which indexes a
# learned embedding prepended to the sequence. The model therefore learns which rule
# each identifier names.
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
# The reference work multiplies them by the transformations that leave the nature of a
# puzzle intact. A variant of a task is one draw of two of them, one of the eight
# symmetries of the square and a permutation of the colours one to nine, colour zero
# keeping its place as the background. The draw applies to every pair of the task, input
# and output alike, in the training split as in the evaluation split, since a rule is
# preserved only when everything it relates moves together. Each variant receives its own
# puzzle identifier and a name recording the draw, the task's name followed by the
# symmetry and the colour mapping, as in the reference. Two draws giving the same grids
# make one variant, so a task with symmetries of its own receives fewer variants than
# asked for once a bounded number of further draws has failed to find new ones.
#
# A translation comes on top, in the training split only. Every pair of a puzzle but one
# moves to a random place inside the window, its input and its output to the same place,
# and padding surrounds it. The evaluation split is never moved, so an answer is read
# from the top-left corner of the window, and undoing a variant is undoing its symmetry
# and its colours.
#
# Augmentation is optional here. Building without it gives the baseline that shows how
# much of the result the augmentation is responsible for.
#
# GROUPING
#
# Two index arrays describe the nesting. puzzle_indices gives, for each puzzle, where
# its examples begin; group_indices gives, for each group, where its puzzles begin. A
# group holds one original task together with all of its variants, the task itself
# first, which lets a sampler draw one variant per task per epoch, so that a task weighs
# the same however many variants it has.
#
# OUTPUT
#
# For each split, a .bin file packs the arrays behind a small header for the C++ side,
# which needs no numpy and no parsing, and a .names file holds the name of every puzzle,
# one per line in the order of the puzzles, the blank one first. A name records its
# variant, which is what an evaluation reads to bring the answers of all the variants of
# a task back to the task itself before they vote.
#
# The reference layout is written beside them when numpy is present: a train and a test
# directory of .npy arrays with their dataset.json, which the reference's training code
# reads as they are, and identifiers.json, the name of every identifier, which its
# evaluation reads to undo the variants.
#
# The examples are produced while the files are written, one at a time, so building a
# thousand variants of every task needs memory for the tasks alone.
#
# Usage:
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --augmentations 300
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --augmentations 0
#   build_arc_dataset.py --data-dir arc-agi --out-dir arc-data --protocol reference
#   build_arc_dataset.py --data-dir arc-agi --extra-corpus ConceptARC/corpus \
#       --out-dir arc-data --protocol reference --augmentations 300
#   build_arc_dataset.py --download --out-dir arc-data --augmentations 1000

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import urllib.request
from array import array
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

# The window is the largest ARC grid, so no example ever has to be truncated.
GRID_SIDE = 30
SEQ_LEN = GRID_SIDE * GRID_SIDE

# Token 0 is the padding outside the grid, token 1 marks where the grid ends, and the ten
# colours follow as 2 to 11. The end marker lets a model state the size of its answer. A
# window of 900 cells carries no other sign of where a 3 by 3 grid stops, and an answer
# without it could only be read back by guessing its extent from where colours happen to
# be. Keeping the padding at zero means an unwritten cell already holds it.
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOUR_OFFSET = 2
NUM_COLOURS = 10
VOCAB_SIZE = NUM_COLOURS + COLOUR_OFFSET

# Identifier 0 is reserved for "no puzzle", which is what an evaluation run uses when
# it wants to ask the model to work without being told which rule to apply.
BLANK_PUZZLE_ID = 0
BLANK_NAME = "<blank>"

# A task with symmetries of its own gives the same grids under several draws. The draws
# for a task stop after this many times the number of variants asked for, as in the
# reference.
RETRIES_FACTOR = 5

ARC1_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"


# ----------------------------------------------------------------------------------
# Reading the source data


def read_task_directory(path: str, recursive: bool = False) -> Dict[str, dict]:
    """Reads a directory of ARC task files, keyed by the task name.

    ConceptARC and similar corpora file their tasks under one directory per concept, so
    the recursive form walks the tree and prefixes each name with the directory it came
    from. The prefix keeps two tasks that happen to share a filename apart, which matters
    because the name is what an augmentation draw is tied to.
    """
    tasks = {}
    if not os.path.isdir(path):
        return tasks

    if not recursive:
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                    tasks[os.path.splitext(name)[0]] = json.load(f)
        return tasks

    for root, _, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            rel = os.path.relpath(os.path.join(root, name), path)
            key = os.path.splitext(rel)[0].replace(os.sep, "/")
            with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                try:
                    task = json.load(f)
                except json.JSONDecodeError:
                    continue
            if isinstance(task, dict) and ("train" in task or "test" in task):
                tasks[key] = task
    return tasks


def task_fingerprint(task: dict) -> str:
    """Identifies a task by its demonstration pairs, whatever its filename.

    Names travel between releases and so do tasks, but neither reliably: a task may be
    renamed, and two corpora may name different things alike. What does not change is the
    grids, so the pairs are what a duplicate is decided on.
    """
    demo = task.get("train", [])
    payload = json.dumps([[p.get("input"), p.get("output")] for p in demo],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def drop_overlap(extra: Dict[str, dict], evaluation: Dict[str, dict]) -> Tuple[Dict[str, dict], int]:
    """Removes from a corpus anything that is also in the evaluation split.

    This matters as soon as a corpus is drawn from a later release of the same benchmark.
    A release carries its predecessor's tasks forward, evaluation ones included, so adding
    its training set wholesale would inject through those tasks the answers the run is
    about to be graded on. The score would rise and would collapse against any set the
    model had not seen. Declaring the protocol does not repair that, since a leak is an
    error of measurement, which no protocol accounts for.

    Matching is by name and by the fingerprint of the demonstration pairs, since either
    alone can miss.
    """
    if not evaluation:
        return extra, 0

    names = {n.rsplit("/", 1)[-1] for n in evaluation}
    prints = {task_fingerprint(t) for t in evaluation.values()}

    kept, dropped = {}, 0
    for name, task in extra.items():
        base = name.rsplit("/", 1)[-1]
        if base in names or task_fingerprint(task) in prints:
            dropped += 1
            continue
        kept[name] = task
    return kept, dropped


def load_extra_corpus(path: str) -> Dict[str, dict]:
    """Reads a corpus of ARC-format tasks from wherever they happen to sit.

    The reference work trains on 960 tasks: the 400 of the ARC-AGI-1 training set, the
    400 of its evaluation set through their demonstration pairs alone, and 160 from
    ConceptARC. Only the first two come from the ARC-AGI tree, so the third has to be
    named separately. Anything holding tasks in the same JSON shape can be added the same
    way, including the training split of a later ARC release.
    """
    tasks = read_task_directory(path, recursive=True)
    if tasks:
        return tasks

    # a challenges and solutions pair, the layout of the later releases
    for base in ("training", "evaluation", "test"):
        ch = os.path.join(path, f"arc-agi_{base}_challenges.json")
        if os.path.exists(ch):
            tasks.update(read_challenge_file(
                ch, os.path.join(path, f"arc-agi_{base}_solutions.json")))
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
Pair = Tuple[Grid, Grid]

# A variant is a symmetry of the square and a colour mapping, colour c becoming
# mapping[c]. The task itself is the identity of both.
Variant = Tuple[int, Tuple[int, ...]]
IDENTITY: Variant = (0, tuple(range(NUM_COLOURS)))


def valid_grid(grid) -> bool:
    """A grid the window can hold: rectangular, at most 30 by 30, colours 0 to 9."""
    if not isinstance(grid, list) or not grid or len(grid) > GRID_SIDE:
        return False
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    if width == 0 or width > GRID_SIDE:
        return False
    return all(isinstance(row, list) and len(row) == width and
               all(isinstance(v, int) and 0 <= v < NUM_COLOURS for v in row)
               for row in grid)


def rotate(grid: Grid) -> Grid:
    """A quarter turn counter-clockwise, the turn of numpy's rot90."""
    return [list(row) for row in zip(*grid)][::-1]


def dihedral(grid: Grid, index: int) -> Grid:
    """One of the eight symmetries of the square, numbered as the reference numbers them.

    0 is the identity, 1 to 3 are one to three quarter turns counter-clockwise, 4 mirrors
    left to right, 5 top to bottom, 6 transposes and 7 reflects across the other diagonal.
    The number is part of a variant's name, where the reference's evaluation reads it to
    undo the variant, so the numbering follows numpy's rot90, fliplr, flipud and transpose.
    """
    if index == 0:
        return [list(row) for row in grid]
    if 1 <= index <= 3:
        g = grid
        for _ in range(index):
            g = rotate(g)
        return g
    if index == 4:
        return [row[::-1] for row in grid]
    if index == 5:
        return [list(row) for row in grid[::-1]]
    if index == 6:
        return [list(row) for row in zip(*grid)]
    if index == 7:
        return [row[::-1] for row in rotate(grid)]
    raise ValueError(f"there is no symmetry numbered {index}")


def apply_variant(grid: Grid, variant: Variant) -> Grid:
    """The colours first, then the symmetry, in the order of the reference."""
    index, mapping = variant
    return dihedral([[mapping[v] for v in row] for row in grid], index)


def variant_name(task: str, variant: Variant) -> str:
    """The reference's name for a variant: the task, the symmetry, then the colour mapping."""
    if variant == IDENTITY:
        return task
    index, mapping = variant
    return f"{task}_t{index}_{''.join(str(c) for c in mapping)}"


def grid_hash(grid: Grid) -> str:
    h = hashlib.sha256(bytes([len(grid), len(grid[0])]))
    h.update(bytes(v for row in grid for v in row))
    return h.hexdigest()


def puzzle_hash(pairs: List[Pair]) -> str:
    """What two variants are compared on: their pairs, in whatever order they come."""
    keys = sorted(f"{grid_hash(gi)}|{grid_hash(go)}" for gi, go in pairs)
    return hashlib.sha256("|".join(keys).encode()).hexdigest()


def stream(seed: int, *parts: str) -> random.Random:
    """A generator of its own for each use, so that what a task draws depends on its name
    and the seed alone, and never on the other tasks or on the order they come in."""
    return random.Random("/".join([str(seed)] + list(parts)))


def draw_variants(pairs: List[Pair], count: int, rng: random.Random) -> List[Variant]:
    """The variants of one task, the task itself first.

    pairs holds every pair of the task, whichever split each one goes to, since a single
    draw applies to all of them. A draw that gives the grids of the task or of a variant
    already kept is dropped, and the draws stop after RETRIES_FACTOR times count.
    """
    variants = [IDENTITY]
    if count <= 0:
        return variants
    seen = {puzzle_hash(pairs)}
    for _ in range(RETRIES_FACTOR * count):
        index = rng.randrange(8)
        others = list(range(1, NUM_COLOURS))
        rng.shuffle(others)
        variant = (index, (0,) + tuple(others))
        h = puzzle_hash([(apply_variant(gi, variant), apply_variant(go, variant))
                         for gi, go in pairs])
        if h not in seen:
            seen.add(h)
            variants.append(variant)
            if len(variants) > count:
                break
    return variants


def placements(pairs: List[Pair], rng: random.Random) -> List[Tuple[int, int]]:
    """Where each pair of a training puzzle sits in the window, as a row and a column.

    One pair, drawn at random, stays in the top-left corner, as in the reference. Every
    other pair moves to a random place where both of its grids fit, its input and its
    output to the same place.
    """
    kept = rng.randrange(len(pairs))
    out = []
    for k, (gi, go) in enumerate(pairs):
        if k == kept:
            out.append((0, 0))
            continue
        room_rows = GRID_SIDE - max(len(gi), len(go))
        room_cols = GRID_SIDE - max(len(gi[0]), len(go[0]))
        out.append((rng.randint(0, room_rows), rng.randint(0, room_cols)))
    return out


def encode_grid(grid: Grid, row: int = 0, col: int = 0) -> bytearray:
    """Writes a grid into the window, its top-left cell at the given row and column.

    The cells take their colour shifted past the two reserved tokens. The row just below
    the grid and the column just to its right hold the end marker, along the grid's own
    width and height and as far as the window allows, so that the grid's extent is written
    into the sequence. Everything else is padding, and padding is what the loss ignores.
    """
    out = bytearray(SEQ_LEN)                          # every cell starts as the padding
    height, width = len(grid), len(grid[0])
    for r in range(height):
        base = (row + r) * GRID_SIDE + col
        out[base:base + width] = bytes(v + COLOUR_OFFSET for v in grid[r])
    if row + height < GRID_SIDE:                      # the row below the grid
        base = (row + height) * GRID_SIDE + col
        out[base:base + width] = bytes([EOS_TOKEN]) * width
    if col + width < GRID_SIDE:                       # the column to its right
        for r in range(height):
            out[(row + r) * GRID_SIDE + col + width] = EOS_TOKEN
    return out


# ----------------------------------------------------------------------------------
# Building


class Task:
    """One task, the pairs each split receives from it, and its variants."""

    def __init__(self, name: str, splits: Dict[str, List[Pair]]):
        self.name = name
        self.splits = splits
        self.variants: List[Variant] = [IDENTITY]

    def all_pairs(self) -> List[Pair]:
        return self.splits.get("training", []) + self.splits.get("evaluation", [])


class Split:
    """The puzzles of one split and the two index arrays that describe their nesting.

    Only names and counts are held here. The grids are produced while the split is
    written, one example at a time, so that a split of millions of examples never sits in
    memory.
    """

    def __init__(self, name: str, translate: bool):
        self.name = name
        self.translate = translate
        self.names: List[str] = [BLANK_NAME]
        self.puzzle_identifiers: List[int] = [BLANK_PUZZLE_ID]
        self.puzzle_indices: List[int] = [0]
        self.group_indices: List[int] = [0]
        self.members: List[Tuple[Task, int]] = []      # each puzzle's task and variant
        self.num_tasks = 0

    @property
    def num_examples(self) -> int:
        return self.puzzle_indices[-1]

    def add_task(self, task: Task) -> None:
        """The task and its variants become one group, the task itself first."""
        count = len(task.splits[self.name])
        for k, variant in enumerate(task.variants):
            self.names.append(variant_name(task.name, variant))
            self.puzzle_identifiers.append(len(self.puzzle_identifiers))
            self.puzzle_indices.append(self.num_examples + count)
            self.members.append((task, k))
        self.group_indices.append(len(self.puzzle_indices) - 1)
        self.num_tasks += 1

    def examples(self, seed: int):
        """Every example of the split in order, as the encoded input and label."""
        for p, (task, k) in enumerate(self.members):
            variant = task.variants[k]
            pairs = [(apply_variant(gi, variant), apply_variant(go, variant))
                     for gi, go in task.splits[self.name]]
            if self.translate:
                places = placements(pairs, stream(seed, "placement", self.name,
                                                  self.names[p + 1]))
            else:
                places = [(0, 0)] * len(pairs)
            for (gi, go), (row, col) in zip(pairs, places):
                yield encode_grid(gi, row, col), encode_grid(go, row, col)


def pairs_of(task: dict, keys: Tuple[str, ...], rejected: List[str], name: str) -> List[Pair]:
    """The complete pairs a task holds under the given keys, in their order."""
    out = []
    for key in keys:
        for p in task.get(key, []):
            if "output" not in p:
                continue                      # no answer available, so nothing to learn
            if valid_grid(p.get("input")) and valid_grid(p["output"]):
                out.append((p["input"], p["output"]))
            else:
                rejected.append(name)
    return out


def plan_tasks(protocol: str, train_tasks: Dict[str, dict], eval_tasks: Dict[str, dict],
               extra_tasks: Dict[str, dict], include_test_pairs: bool,
               rejected: List[str]) -> List[Task]:
    """Decides which pairs of which task go to which split.

    Under the reference protocol every task, evaluation ones included, receives its
    identifier while training. A complementary corpus and the training tasks give the
    training split everything they have and are never asked again. An evaluation task
    gives its demonstrations to the training split and its test pairs to the evaluation
    split, under the same identifiers, so the question put at evaluation time is one the
    model can understand and has not seen answered. This is a form of learning at test
    time and it is why those numbers are what they are. It is offered because it makes a
    comparison possible; --protocol held-out is the harder setting.

    Under the held-out protocol the evaluation tasks give their test pairs to the
    evaluation split alone, and the model meets them without an identifier.

    The complementary corpus comes first, so that its identifiers stay put when the ARC
    tasks change, which lets two datasets built from the same corpora share a model.
    """
    plan = []
    for name in sorted(extra_tasks):
        pairs = pairs_of(extra_tasks[name], ("train", "test"), rejected, name)
        if pairs:
            plan.append(Task("extra/" + name, {"training": pairs}))

    train_keys = ("train", "test") if protocol == "reference" or include_test_pairs else ("train",)
    for name in sorted(train_tasks):
        pairs = pairs_of(train_tasks[name], train_keys, rejected, name)
        if pairs:
            plan.append(Task(name, {"training": pairs}))

    for name in sorted(eval_tasks):
        test = pairs_of(eval_tasks[name], ("test",), rejected, name)
        if protocol == "reference":
            demo = pairs_of(eval_tasks[name], ("train",), rejected, name)
            if demo and test:
                plan.append(Task(name, {"training": demo, "evaluation": test}))
        elif test:
            plan.append(Task(name, {"evaluation": test}))
    return plan


def build_splits(plan: List[Task], protocol: str, translate: bool) -> Tuple[Split, Split]:
    tr, ev = Split("training", translate), Split("evaluation", False)
    for task in plan:
        if "training" in task.splits:
            tr.add_task(task)
    for task in plan:
        if "evaluation" in task.splits:
            ev.add_task(task)

    if protocol == "reference":
        # The evaluation puzzles are named by the identifiers the training split gave
        # them. A silent mismatch would be fatal, so it is checked.
        index = {n: i for i, n in enumerate(tr.names)}
        missing = [n for n in ev.names[1:] if n not in index]
        if missing:
            raise RuntimeError(f"{len(missing)} evaluation puzzles have no identifier in the "
                               f"training split, starting with {missing[0]}")
        ev.puzzle_identifiers = [index[n] for n in ev.names]
    else:
        # The evaluation puzzles were never named during training, so they carry the blank
        # identifier, which is how the model is asked about them.
        ev.puzzle_identifiers = [BLANK_PUZZLE_ID] * len(ev.names)
    return tr, ev


# ----------------------------------------------------------------------------------
# Writing


def int32_bytes(values) -> bytes:
    """Little-endian int32, the type of every value of the .bin file."""
    block = array("i", array("B", values))
    if sys.byteorder != "little":
        block.byteswap()
    return block.tobytes()


def open_npy_rows(path: str, rows: int):
    """An .npy file of rows of SEQ_LEN bytes, its header written, for the rows to follow in
    order."""
    f = open(path, "wb", buffering=1 << 20)
    header = {"descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
              "fortran_order": False, "shape": (rows, SEQ_LEN)}
    np.lib.format.write_array_header_1_0(f, header)
    return f


def write_split(out_dir: str, split: Split, seed: int, bin_out: bool, npy_out: bool,
                num_identifiers: int) -> None:
    """Writes the .bin file, the .names file and the reference layout of one split.

    The .bin file holds a header, then the inputs, the labels, the identifiers, the puzzle
    index and the group index, all int32, so that the reader needs no parsing and no
    third-party library. The labels are gathered in a temporary file while the inputs are
    written, and appended after them.

    The reference layout stores the cells as bytes. Its identifier array holds one entry
    per puzzle, where the .bin file opens with the blank puzzle.

    Every file is written in order as the examples are produced, so the memory a split
    takes is that of its names and indices.
    """
    n = split.num_examples
    with open(os.path.join(out_dir, f"{split.name}.names"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(split.names) + "\n")

    npy_dir = os.path.join(out_dir, "train" if split.name == "training" else "test")
    with contextlib.ExitStack() as stack:
        if bin_out:
            bin_file = stack.enter_context(open(os.path.join(out_dir, f"{split.name}.bin"),
                                                "wb", buffering=1 << 20))
            labels_tmp = stack.enter_context(tempfile.TemporaryFile(dir=out_dir))
            bin_file.write(b"ARCD")
            bin_file.write(struct.pack("<7i", 1, SEQ_LEN, VOCAB_SIZE, n,
                                       len(split.puzzle_identifiers), len(split.puzzle_indices),
                                       len(split.group_indices)))
        if npy_out:
            os.makedirs(npy_dir, exist_ok=True)
            npy_inputs = stack.enter_context(
                open_npy_rows(os.path.join(npy_dir, "all__inputs.npy"), n))
            npy_labels = stack.enter_context(
                open_npy_rows(os.path.join(npy_dir, "all__labels.npy"), n))

        e = 0
        for gi, go in split.examples(seed):
            if bin_out:
                bin_file.write(int32_bytes(gi))
                labels_tmp.write(int32_bytes(go))
            if npy_out:
                npy_inputs.write(gi)
                npy_labels.write(go)
            e += 1
        assert e == n, f"{split.name}: {e} examples written where the index counts {n}"

        if bin_out:
            labels_tmp.seek(0)
            shutil.copyfileobj(labels_tmp, bin_file, 1 << 20)
            for values in (split.puzzle_identifiers, split.puzzle_indices, split.group_indices):
                bin_file.write(struct.pack(f"<{len(values)}i", *values))

    if npy_out:
        np.save(os.path.join(npy_dir, "all__puzzle_identifiers.npy"),
                np.array(split.puzzle_identifiers[1:], dtype=np.int32))
        np.save(os.path.join(npy_dir, "all__puzzle_indices.npy"),
                np.array(split.puzzle_indices, dtype=np.int32))
        np.save(os.path.join(npy_dir, "all__group_indices.npy"),
                np.array(split.group_indices, dtype=np.int32))
        puzzles = max(1, len(split.names) - 1)
        meta = {
            "pad_id": PAD_TOKEN,
            "ignore_label_id": PAD_TOKEN,
            "blank_identifier_id": BLANK_PUZZLE_ID,
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "num_puzzle_identifiers": num_identifiers,
            "total_groups": len(split.group_indices) - 1,
            "mean_puzzle_examples": n / puzzles,
            "sets": ["all"],
        }
        with open(os.path.join(npy_dir, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)


def write_metadata(out_dir: str, split: Split, args, dropped: int, short: int) -> dict:
    meta = {
        "split": split.name,
        "seq_len": SEQ_LEN,
        "grid_side": GRID_SIDE,
        "vocab_size": VOCAB_SIZE,
        "pad_token": PAD_TOKEN,
        "eos_token": EOS_TOKEN,
        "colour_offset": COLOUR_OFFSET,
        "blank_puzzle_id": BLANK_PUZZLE_ID,
        "num_tasks": split.num_tasks,
        "num_examples": split.num_examples,
        "num_puzzles": len(split.names) - 1,
        "num_groups": len(split.group_indices) - 1,
        "protocol": args.protocol,
        "extra_corpora": args.extra_corpus,
        "extra_tasks_dropped_as_overlap": dropped,
        "augmentations_per_task": args.augmentations,
        "tasks_with_fewer_variants": short,
        "translated": split.translate,
        "test_pairs_included": bool(args.include_test_pairs),
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, f"{split.name}.json"), "w", encoding="utf-8") as f:
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
                   help="variants drawn per task, 0 for none; a task whose own symmetries "
                        "leave fewer distinct variants receives fewer (default: 1000)")
    p.add_argument("--translations", action="store_true", default=True,
                   help="move the pairs of the training split inside the window "
                        "(default: on)")
    p.add_argument("--no-translations", dest="translations", action="store_false")
    p.add_argument("--protocol", choices=("held-out", "reference"), default="held-out",
                   help="held-out puts the test pairs of the evaluation tasks in the "
                        "evaluation split alone, under the blank identifier, so the model is "
                        "asked to work without being told the rule. reference gives every "
                        "task, evaluation ones included, an identifier while training, from "
                        "its demonstration pairs alone, and puts only its test pairs in the "
                        "evaluation split; that is what makes the published numbers "
                        "comparable and it is learning at test time (default: held-out)")
    p.add_argument("--include-test-pairs", action="store_true",
                   help="under held-out only, add the test pairs of the training tasks to "
                        "their training examples, as the reference protocol does")
    p.add_argument("--extra-corpus", action="append", default=[], metavar="DIR",
                   help="a further corpus of ARC-format tasks, added to the training split "
                        "and never evaluated on. Repeatable. The reference work trains on "
                        "960 tasks: the 400 of the ARC-AGI-1 training set, the 400 of its "
                        "evaluation set through their demonstration pairs alone, and 160 "
                        "from ConceptARC, which is what this option is for. A corpus drawn "
                        "from a later release of the same benchmark carries the earlier "
                        "tasks forward, so anything also present in the evaluation split is "
                        "dropped and the count reported")
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

    extra_tasks = {}
    for path in args.extra_corpus:
        found = load_extra_corpus(path)
        if not found:
            print(f"extra corpus: nothing found in {path}")
        else:
            print(f"extra corpus: {len(found)} tasks in {path}")
        for k, v in found.items():
            extra_tasks[f"{os.path.basename(os.path.normpath(path))}/{k}"] = v

    if not train_tasks and not eval_tasks and not extra_tasks:
        print(f"nothing found in {args.data_dir}")
        return 1

    extra_tasks, dropped = drop_overlap(extra_tasks, eval_tasks)
    if dropped:
        print(f"extra corpus: {dropped} tasks dropped, they are in the evaluation split")

    rejected: List[str] = []
    plan = plan_tasks(args.protocol, train_tasks, eval_tasks, extra_tasks,
                      args.include_test_pairs, rejected)
    if rejected:
        print(f"{len(rejected)} pairs left out, a grid being empty, uneven, larger than "
              f"{GRID_SIDE} by {GRID_SIDE} or holding a value outside 0 to 9, starting "
              f"with task {rejected[0]}")

    short = 0
    for task in plan:
        task.variants = draw_variants(task.all_pairs(), args.augmentations,
                                      stream(args.seed, "variants", task.name))
        short += len(task.variants) <= args.augmentations
    if short:
        print(f"{short} tasks received fewer than {args.augmentations} variants, their own "
              f"symmetries leaving fewer distinct ones")

    tr, ev = build_splits(plan, args.protocol, args.translations)

    npy_out = args.format in ("npy", "both")
    if npy_out and np is None:
        print("numpy is not installed, so the reference layout is skipped", file=sys.stderr)
        npy_out = False
    bin_out = args.format in ("bin", "both")
    num_identifiers = len(tr.names)

    for split in (tr, ev):
        if split.num_tasks == 0:
            continue
        write_split(args.out_dir, split, args.seed, bin_out, npy_out, num_identifiers)
        meta = write_metadata(args.out_dir, split, args, dropped,
                              sum(len(t.variants) <= args.augmentations
                                  for t, k in split.members if k == 0))
        print(f"{split.name}: {meta['num_tasks']} tasks, {meta['num_puzzles']} puzzles, "
              f"{meta['num_examples']} examples, {meta['num_groups']} groups")

    if npy_out:
        with open(os.path.join(args.out_dir, "identifiers.json"), "w", encoding="utf-8") as f:
            json.dump(tr.names, f)

    if args.protocol == "reference":
        print("\nThe two splits share one identifier space: every evaluation puzzle was "
              "named during training,\nfrom its demonstration pairs only. Its test pairs "
              "are what the evaluation split asks for.")
    else:
        print("\nThe evaluation puzzles carry the blank identifier, since none of them was "
              "named during\ntraining, so the model is asked about them without being told "
              "the rule.")

    print(f"\nWritten to {args.out_dir}")
    print(f"window {SEQ_LEN} cells, vocabulary {VOCAB_SIZE}, pad token {PAD_TOKEN}")


if __name__ == "__main__":
    main()
