#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Sudoku-Extreme dataset preparation for the hierarchical reasoning examples.
#
# Turns the Sudoku-Extreme CSV files into the arrays the HRM training loop reads, in the
# .bin layout that build_arc_dataset.py writes and the C++ loader already reads, and in
# the .npy layout of the reference implementation, so that a dataset built here can be
# fed to either codebase and the two sets of results compared on the same data.
#
# WHY SUDOKU
#
# The reference work reports its Sudoku result from a model trained on one thousand
# puzzles, each augmented a thousand times, on a single GPU, where its ARC result needs
# several. Both are trained by the same code, with the same model and the same halting,
# so Sudoku tests the reproduction of the whole training procedure at a cost a single
# machine can carry.
#
# WHAT THE LAYOUT IS
#
# A puzzle is its 81 cells read row by row. Token 0 is the padding, which Sudoku never
# needs, token 1 a blank cell and tokens 2 to 10 the digits 1 to 9, which is the
# reference's encoding, digit plus one. The input is the puzzle and the label its solution, so a
# model answers all 81 cells at once.
#
# Every puzzle is its own puzzle in the index arrays and carries the blank identifier, as
# in the reference, where a single learned vector stands in front of every grid. A group
# holds one original puzzle and its augmentations, so a sampler that draws one variant per
# group sees every original puzzle once per epoch, however many variants it has.
#
# AUGMENTATION
#
# Only on the training split, and only through the symmetries of the game, applied
# together to the puzzle and its solution, as the reference applies them. They are a
# permutation of the nine digits, a transposition with probability one half, a
# permutation of the three bands of rows and of the rows within each band, and the same
# for the columns. Each turns a valid Sudoku into a valid Sudoku with the same clues.
#
# OUTPUT
#
#   <out-dir>/train.bin, test.bin      the arrays for the C++ loader, little-endian int32
#   <out-dir>/train.json, test.json    what was built, and from what
#   <out-dir>/train/, test/            the reference layout, when numpy is installed
#
# Usage:
#   build_sudoku_dataset.py --download --out-dir sudoku-data
#   build_sudoku_dataset.py --data-dir sudoku-extreme --out-dir sudoku-data
#   build_sudoku_dataset.py --out-dir sudoku-full --subsample-size 0 --num-aug 0
#
# The defaults build the configuration the reference reports on, with one thousand
# training puzzles drawn from the 3.8 million, a thousand augmentations of each, and the
# whole test split of about 423 thousand puzzles left as it is.

import argparse
import csv
import json
import operator
import os
import random
import struct
import sys
import urllib.request
from array import array

try:
    import numpy as np
except ImportError:
    np = None

SOURCE_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main"

SEQ_LEN = 81
PAD_TOKEN = 0           # never written, since every puzzle fills its 81 cells
BLANK_CELL_TOKEN = 1    # a cell the puzzle leaves empty
DIGIT_OFFSET = 1        # digit d is token d + 1
VOCAB_SIZE = 11         # the padding, the blank and the nine digits
BLANK_PUZZLE_ID = 0     # the one identifier every puzzle carries


# ----------------------------------------------------------------------------------
# Reading the source data


def download(data_dir: str) -> None:
    """Fetches train.csv and test.csv, the two splits of Sudoku-Extreme.

    The files are those the reference reads through the Hugging Face hub, taken here over
    plain HTTPS so that nothing beyond the standard library is needed.
    """
    os.makedirs(data_dir, exist_ok=True)
    for split in ("train", "test"):
        dest = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(dest):
            print(f"  {dest} is already here")
            continue
        url = f"{SOURCE_URL}/{split}.csv"
        print(f"  {url}")
        tmp = dest + ".part"
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)


def eligible_rows(path: str, min_difficulty):
    """Yields (question, answer) for each row whose rating reaches min_difficulty.

    The columns are source, question, answer and rating. A blank cell is written '.'.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)   # the header
        for row in reader:
            if len(row) < 4:
                continue
            _, q, a, rating = row[:4]
            if min_difficulty is not None and int(rating) < min_difficulty:
                continue
            if len(q) != SEQ_LEN or len(a) != SEQ_LEN:
                raise ValueError(f"{path}: a row holds {len(q)} and {len(a)} cells, "
                                 f"{SEQ_LEN} expected")
            yield q, a


# '0' to '9' become 0 to 9 and '.' becomes 0; anything else becomes 255 and is refused.
_DIGITS = bytes((c - 48) if 48 <= c <= 57 else (0 if c == 46 else 255) for c in range(256))


def digits(cells: str) -> bytes:
    """The 81 cells as raw digits 0 to 9, a blank cell as 0."""
    out = cells.encode("ascii").translate(_DIGITS)
    if 255 in out:
        raise ValueError(f"not a Sudoku row: {cells}")
    return out


def read_split(path: str, subsample, min_difficulty, rng):
    """The puzzles of one split, as pairs of raw digit strings.

    Subsampling draws without replacement among the eligible rows, after the difficulty
    filter, as the reference does. It takes two passes over the file so that a draw of a
    thousand puzzles never holds the 3.8 million in memory.
    """
    if subsample:
        total = sum(1 for _ in eligible_rows(path, min_difficulty))
        if subsample < total:
            wanted = set(rng.sample(range(total), subsample))
            return [(digits(q), digits(a))
                    for k, (q, a) in enumerate(eligible_rows(path, min_difficulty))
                    if k in wanted]
    return [(digits(q), digits(a)) for q, a in eligible_rows(path, min_difficulty)]


# ----------------------------------------------------------------------------------
# Augmentation


def transposed(k: int) -> int:
    return (k % 9) * 9 + k // 9


def draw_symmetry(rng):
    """One symmetry of the game, given as where each cell is read from and how digits relabel.

    The order of operations is the reference's. The grid is transposed first, then the
    band and row permutation and the stack and column permutation apply, then the digit
    permutation. Cell (r, c) of the result is read from cell (row_perm[r], col_perm[c]) of
    the possibly transposed grid, and the two steps are folded into one list of source
    cells.
    """
    digit_map = [0] + rng.sample(range(1, 10), 9)
    transpose = rng.random() < 0.5
    bands = rng.sample(range(3), 3)
    row_perm = [b * 3 + r for b in bands for r in rng.sample(range(3), 3)]
    stacks = rng.sample(range(3), 3)
    col_perm = [s * 3 + c for s in stacks for c in rng.sample(range(3), 3)]
    source = [row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(SEQ_LEN)]
    if transpose:
        source = [transposed(k) for k in source]
    # The relabelling and the shift to tokens in one table for bytes.translate.
    table = bytes(digit_map[d] + DIGIT_OFFSET if d < 10 else 0 for d in range(256))
    return operator.itemgetter(*source), table


IDENTITY_TABLE = bytes(d + DIGIT_OFFSET if d < 10 else 0 for d in range(256))


def as_tokens(cells: bytes) -> bytes:
    return cells.translate(IDENTITY_TABLE)


def is_solution(tokens: bytes) -> bool:
    """Whether 81 tokens form a solved grid: every row, column and box holds 1 to 9."""
    full = set(range(DIGIT_OFFSET + 1, DIGIT_OFFSET + 10))
    for k in range(9):
        if set(tokens[k * 9:(k + 1) * 9]) != full:
            return False
        if set(tokens[k::9]) != full:
            return False
        r, c = (k // 3) * 3, (k % 3) * 3
        if {tokens[(r + i) * 9 + c + j] for i in range(3) for j in range(3)} != full:
            return False
    return True


def fits(puzzle: bytes, solution: bytes) -> bool:
    """Whether every clue of the puzzle is the digit the solution has there."""
    return all(p == BLANK_CELL_TOKEN or p == s for p, s in zip(puzzle, solution))


# ----------------------------------------------------------------------------------
# Building a split


class Split:
    """The arrays of one split, cells kept one byte each until they are written."""

    def __init__(self):
        self.inputs = bytearray()
        self.labels = bytearray()
        self.num_examples = 0
        self.group_indices = [0]

    def add(self, puzzle: bytes, solution: bytes) -> None:
        self.inputs += puzzle
        self.labels += solution
        self.num_examples += 1

    def close_group(self) -> None:
        self.group_indices.append(self.num_examples)


def build(pairs, num_aug: int, rng, checked: int) -> Split:
    """The original of each puzzle and num_aug variants of it, one group per puzzle.

    The first `checked` groups are verified cell by cell. Every variant has to be a solved
    grid that its puzzle's clues agree with, which is the property the symmetries
    guarantee, and a mistake in composing them would break it at once.
    """
    split = Split()
    for n, (q, a) in enumerate(pairs):
        puzzle, solution = as_tokens(q), as_tokens(a)
        if n < checked and not (is_solution(solution) and fits(puzzle, solution)):
            raise ValueError(f"puzzle {n} of the source is not a solved Sudoku")
        split.add(puzzle, solution)
        for _ in range(num_aug):
            read, table = draw_symmetry(rng)
            p = bytes(read(q)).translate(table)
            s = bytes(read(a)).translate(table)
            if n < checked and not (is_solution(s) and fits(p, s)):
                raise ValueError(f"an augmentation of puzzle {n} broke it")
            split.add(p, s)
        split.close_group()
    return split


# ----------------------------------------------------------------------------------
# Writing


def write_ints(f, data, chunk: int = 1 << 22) -> None:
    """Writes bytes, a list or a range as little-endian int32, a slice at a time.

    Bytes go through an unsigned-byte array first, since array() given bytes directly
    reads them as raw int32 values, four bytes to a value.
    """
    for start in range(0, len(data), chunk):
        part = data[start:start + chunk]
        if isinstance(part, (bytes, bytearray)):
            part = array("B", part)
        block = array("i", part)
        if sys.byteorder != "little":
            block.byteswap()
        block.tofile(f)


def write_bin(out_dir: str, name: str, split: Split) -> str:
    """The layout of build_arc_dataset.py, read by load_arc_dataset() unchanged.

    A header, then the inputs, the labels, the identifiers, the puzzle index and the group
    index, all int32. The identifier array opens with the blank puzzle, as the ARC files'
    does, and every puzzle after it carries the blank identifier too.
    """
    num_puzzles = split.num_examples
    path = os.path.join(out_dir, f"{name}.bin")
    with open(path, "wb") as f:
        f.write(b"ARCD")
        f.write(struct.pack("<7i", 1, SEQ_LEN, VOCAB_SIZE, split.num_examples,
                            num_puzzles + 1, num_puzzles + 1, len(split.group_indices)))
        write_ints(f, split.inputs)
        write_ints(f, split.labels)
        write_ints(f, [BLANK_PUZZLE_ID] * (num_puzzles + 1))
        write_ints(f, range(num_puzzles + 1))
        write_ints(f, split.group_indices)
    return path


def write_npy(out_dir: str, name: str, split: Split) -> None:
    """The reference layout, which its training code reads as it is."""
    if np is None:
        print("  numpy is not installed, so the reference layout is skipped", file=sys.stderr)
        return
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    n = split.num_examples
    np.save(os.path.join(d, "all__inputs.npy"),
            np.frombuffer(bytes(split.inputs), dtype=np.uint8).reshape(n, SEQ_LEN))
    np.save(os.path.join(d, "all__labels.npy"),
            np.frombuffer(bytes(split.labels), dtype=np.uint8).reshape(n, SEQ_LEN))
    np.save(os.path.join(d, "all__puzzle_identifiers.npy"), np.zeros(n, dtype=np.int32))
    np.save(os.path.join(d, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(d, "all__group_indices.npy"),
            np.array(split.group_indices, dtype=np.int32))
    meta = {
        "pad_id": PAD_TOKEN,
        "ignore_label_id": PAD_TOKEN,
        "blank_identifier_id": BLANK_PUZZLE_ID,
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": 1,
        "total_groups": len(split.group_indices) - 1,
        "mean_puzzle_examples": 1,
        "sets": ["all"],
    }
    with open(os.path.join(d, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(out_dir, "identifiers.json"), "w", encoding="utf-8") as f:
        json.dump(["<blank>"], f)


def write_metadata(out_dir: str, name: str, split: Split, source: str, num_aug: int,
                   args) -> dict:
    meta = {
        "split": name,
        "source": source,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "pad_token": PAD_TOKEN,
        "blank_cell_token": BLANK_CELL_TOKEN,
        "digit_offset": DIGIT_OFFSET,
        "blank_puzzle_id": BLANK_PUZZLE_ID,
        "num_examples": split.num_examples,
        "num_puzzles": split.num_examples,
        "num_groups": len(split.group_indices) - 1,
        "augmentations_per_puzzle": num_aug,
        "subsample_size": args.subsample_size if name == "train" else 0,
        "min_difficulty": args.min_difficulty,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ----------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Prepare Sudoku-Extreme for the hierarchical reasoning examples.")
    p.add_argument("--data-dir", default="sudoku-extreme",
                   help="directory holding train.csv and test.csv")
    p.add_argument("--out-dir", default="sudoku-data",
                   help="where the prepared arrays are written")
    p.add_argument("--download", action="store_true",
                   help="fetch train.csv and test.csv into --data-dir first")
    p.add_argument("--subsample-size", type=int, default=1000,
                   help="training puzzles drawn from the training split, 0 for all of "
                        "them (default: 1000, the reference's setting)")
    p.add_argument("--num-aug", type=int, default=1000,
                   help="augmentations of each training puzzle, 0 for none (default: "
                        "1000, the reference's setting)")
    p.add_argument("--min-difficulty", type=int, default=None,
                   help="keep only the puzzles rated at least this hard")
    p.add_argument("--test-limit", type=int, default=0,
                   help="keep only the first N test puzzles, 0 for all of them; the "
                        "reference reports on the whole split (default: 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=("npy", "bin", "both"), default="both")
    args = p.parse_args()

    if args.download:
        print(f"Fetching Sudoku-Extreme into {args.data_dir}")
        download(args.data_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for name in ("train", "test"):
        source = os.path.join(args.data_dir, f"{name}.csv")
        if not os.path.exists(source):
            print(f"{source} is missing; run with --download, or point --data-dir at the "
                  f"directory holding it")
            return 1

        subsample = args.subsample_size if name == "train" else 0
        pairs = read_split(source, subsample, args.min_difficulty, rng)
        if name == "test" and args.test_limit > 0:
            pairs = pairs[:args.test_limit]
        num_aug = args.num_aug if name == "train" else 0

        split = build(pairs, num_aug, rng, checked=min(len(pairs), 20))
        if args.format in ("npy", "both"):
            write_npy(args.out_dir, name, split)
        if args.format in ("bin", "both"):
            write_bin(args.out_dir, name, split)
        meta = write_metadata(args.out_dir, name, split, source, num_aug, args)
        print(f"{name}: {len(pairs)} puzzles, {meta['num_examples']} examples with their "
              f"augmentations, {meta['num_groups']} groups")

    print(f"\nWritten to {args.out_dir}")
    print(f"{SEQ_LEN} cells, vocabulary {VOCAB_SIZE}: padding {PAD_TOKEN}, blank cell "
          f"{BLANK_CELL_TOKEN}, digit d as d + {DIGIT_OFFSET}")


if __name__ == "__main__":
    sys.exit(main())
