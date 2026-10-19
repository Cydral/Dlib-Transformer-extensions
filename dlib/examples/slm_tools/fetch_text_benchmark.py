#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Fetches the reference corpora of the Large Text Compression Benchmark.
#
# enwik8 and enwik9 are the first 10^8 and 10^9 bytes of an English encyclopaedia dump,
# XML markup included, as Matt Mahoney published them for the benchmark that bears his
# name. They are what text compressors are usually measured on, which makes them the
# natural corpus for slm_tok_predictive_compressor_ex: its tokenizer and network can be
# trained on them and its results set against the published ones.
#
# WHAT IS CHECKED
#
# Each file is downloaded zipped, extracted, and accepted only if its size and MD5 match
# the published file, so that a truncated download or a different revision never reaches
# a training run unnoticed. A file already present and correct is left alone.
#
# OUTPUT
#
#   <out-dir>/enwik8     100,000,000 bytes
#   <out-dir>/enwik9     1,000,000,000 bytes
#
# Usage:
#   fetch_text_benchmark.py --out-dir corpus
#   fetch_text_benchmark.py --out-dir corpus --name enwik8
#   fetch_text_benchmark.py --out-dir corpus --name enwik8 --name enwik9 --keep-zip
#
# Only the standard library is used.

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

SOURCE_URL = "https://mattmahoney.net/dc"

CORPORA = {
    "enwik8": {"size": 100_000_000, "md5": "a1fa5ffddb56f4953e226637dabbb36a"},
    "enwik9": {"size": 1_000_000_000, "md5": "e206c3450ac99950df65bf70ef61a12d"},
}


def md5_of(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: str, name: str) -> bool:
    expected = CORPORA[name]
    return (os.path.isfile(path) and os.path.getsize(path) == expected["size"]
            and md5_of(path) == expected["md5"])


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {url}: {100.0 * done / total:5.1f}%", end="", flush=True)
    print()
    os.replace(tmp, dest)


def fetch(name: str, out_dir: str, keep_zip: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, name)
    if is_valid(dest, name):
        print(f"  {dest} is already here and matches the published file")
        return

    archive = dest + ".zip"
    if not os.path.isfile(archive):
        download(f"{SOURCE_URL}/{name}.zip", archive)

    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if os.path.basename(m) == name]
        if not members:
            sys.exit(f"error: {archive} holds no file named {name}")
        tmp = dest + ".part"
        with z.open(members[0]) as src, open(tmp, "wb") as out:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)

    if not is_valid(dest, name):
        sys.exit(f"error: {dest} does not match the published size and MD5; delete it and its zip and retry")
    if not keep_zip:
        os.remove(archive)
    print(f"  {dest}: {CORPORA[name]['size']:,} bytes, MD5 verified")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch the reference corpora of the Large Text Compression Benchmark.")
    p.add_argument("--out-dir", default=".", help="where the files go (default: current directory)")
    p.add_argument("--name", action="append", choices=sorted(CORPORA),
                   help="corpus to fetch, repeatable (default: enwik9)")
    p.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip next to the file")
    args = p.parse_args()

    for name in args.name or ["enwik9"]:
        fetch(name, args.out_dir, args.keep_zip)


if __name__ == "__main__":
    main()
