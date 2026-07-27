#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Knowledge-alignment corpus preparation, first stage of the fine-tuning pipeline.
#
# Streams the NIST cybersecurity corpus published on the Hugging Face hub, keeps the
# document text and discards the synthetic question-answer wrapper it is shipped in,
# then applies the cleaning that PDF-extracted technical material needs before it can
# be used as a causal language-model corpus.
#
# Why the wrapper is dropped: every record carries the same 298-character system
# paragraph and a template-generated question. Training on those teaches the model one
# paragraph it will never need and a question style it will never be asked. The value
# of this corpus is the standards prose in the answer field: the vocabulary, the
# phrasing and the subject matter of the domain. That is what the knowledge-alignment
# stage is for, and the task-alignment stage that follows is where question answering
# is actually taught (see cve_qa_prepare.py).
#
# The output is a sentinel-separated UTF-8 text file, one document per record, read by
# the C++ example and passed to dlib::build_causal_lm_dataset().
#
# Usage:
#   nist_corpus_prepare.py --out nist_corpus.txt --limit 20000
#   nist_corpus_prepare.py --out nist_corpus.txt --report-only --limit 5000

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

try:
    import requests
except ImportError:
    sys.exit("this script needs the requests package: pip install requests")

HF_BASE = ("https://huggingface.co/datasets/ethanolivertroy/"
           "nist-cybersecurity-training/resolve/main")

DOC_SENTINEL = "<<<doc>>>"

# Front matter and administrative boilerplate. These paragraphs are repeated across
# hundreds of NIST publications; kept, they would be memorized rather than learned.
BOILERPLATE_MARKERS = (
    "certain commercial entities, equipment, or materials may be identified",
    "this publication is available free of charge from",
    "national institute of standards and technology attribution would be appreciated",
    "there is no objection to the reproduction and use of this publication",
    "any mention of commercial products is for information only",
    "reports on computer systems technology",
    "comments on this publication may be submitted to",
)

RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_TABLE_RULE = re.compile(r"\|[\s\-:|]{4,}\|")
RE_PIPE_RUN = re.compile(r"(?:\s*\|\s*){2,}")
RE_LONG_RULE = re.compile(r"[-_=.]{4,}")
RE_SYNTH_PREFIX = re.compile(r"^According to .{0,220}?:\s*")
RE_PAGE_ARTEFACT = re.compile(
    r"\b(?:continued from previous page|this page intentionally left blank)\b\.?",
    re.IGNORECASE)
RE_WS = re.compile(r"[ \t\u00a0]+")
RE_BLANKS = re.compile(r"\n{3,}")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")
RE_SENTENCE_START = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9])")
# The vertical sidebar of NIST publications extracts one character at a time, which no
# ratio-based filter catches because the characters are letters and the spacing is
# regular. The run is matched explicitly and removed.
RE_SPACED_LETTERS = re.compile(r"(?:(?<!\S)\S ){10,}\S(?!\S)")
RE_TOKEN = re.compile(r"\S+")


def stream_records(split, limit, cache_dir):
    """Yield the parsed JSON objects of one split.

    The file is read as a stream so that a bounded run downloads only what it consumes,
    which is what makes a --limit 5000 dry run take seconds instead of the full 800 MB.
    A local copy is used when present, so repeated runs cost nothing.
    """
    name = {"train": "train.jsonl", "validation": "valid.jsonl"}[split]
    local = os.path.join(cache_dir, name) if cache_dir else None

    if local and os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as fin:
            for n, line in enumerate(fin):
                if limit and n >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with requests.get(f"{HF_BASE}/{name}", stream=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"
        for n, line in enumerate(resp.iter_lines(decode_unicode=True)):
            if limit and n >= limit:
                return
            if line:
                yield json.loads(line)


def extract_document(record):
    """Return the document text of a record, or None when the record carries none.

    The corpus is stored as a three-message conversation; the document lives in the
    assistant turn, the other two being the generated wrapper.
    """
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if message.get("role") == "assistant":
            return message.get("content") or None
    return None


def clean(text):
    """Normalize one PDF-extracted chunk.

    The transformations address what document conversion leaves behind rather than the
    prose itself: image placeholders, the skeleton of tables whose cells did not
    survive, page furniture, and the rules used as visual separators. Nothing here
    rewrites sentences, so the domain vocabulary is preserved exactly.
    """
    text = unicodedata.normalize("NFKC", text)
    text = RE_SYNTH_PREFIX.sub("", text)
    text = RE_HTML_COMMENT.sub(" ", text)
    text = RE_PAGE_ARTEFACT.sub(" ", text)
    text = RE_TABLE_RULE.sub(" ", text)
    text = RE_PIPE_RUN.sub(" ", text)
    text = RE_LONG_RULE.sub(" ", text)
    text = RE_SPACED_LETTERS.sub(" ", text)
    text = text.replace("|", " ")
    # Control characters other than newline and tab carry no meaning here.
    text = "".join(c for c in text if c == "\n" or c == "\t" or unicodedata.category(c)[0] != "C")
    text = RE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = RE_BLANKS.sub("\n\n", text)
    return text.strip(" ;:,\n\t")


def trim_partial_sentences(text):
    """Drop a leading and a trailing partial sentence.

    The corpus is chunked by character budget, so a record routinely opens in the
    middle of a word and closes in the middle of a clause. A causal objective takes
    those fragments at face value and learns to begin answers mid-word, so the head is
    advanced to the first sentence boundary and the tail cut back to the last one. The
    text is returned untouched when no boundary is found, rather than emptied.
    """
    head = RE_SENTENCE_START.search(text[:400])
    if head and head.end() < len(text):
        text = text[head.end():]
    tail = max(text.rfind(". "), text.rfind(".\n"), text.rfind("? "), text.rfind("! "))
    if tail == -1 and text.endswith("."):
        tail = len(text) - 2
    if tail > 0.5 * len(text):
        text = text[:tail + 1]
    return text.strip()


def quality(text):
    """Return (alpha_ratio, lexical_diversity, fragmentation) of a cleaned chunk.

    alpha_ratio separates prose from the residue of tables and figure captions, which
    survives cleaning as sparse punctuation and digits. lexical_diversity catches the
    chunks where extraction duplicated a header or a column dozens of times: those read
    as fluent English word by word yet teach the model nothing but repetition.
    fragmentation catches what neither sees, hex dumps and character-by-character
    extraction, where the characters are ordinary but almost no token is a word.
    """
    if not text:
        return 0.0, 0.0
    letters = sum(1 for c in text if c.isalpha() or c.isspace())
    words = RE_WORD.findall(text.lower())
    diversity = len(set(words)) / len(words) if words else 0.0
    tokens = RE_TOKEN.findall(text)
    short = sum(1 for t in tokens if len(t) <= 2)
    fragmented = short / len(tokens) if tokens else 1.0
    return letters / len(text), diversity, fragmented


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the NIST corpus for the knowledge-alignment stage.")
    parser.add_argument("--out", default="nist_corpus.txt",
                        help="output corpus file (default: nist_corpus.txt)")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation"],
                        help="source split (default: train)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many source records, 0 reads them all")
    parser.add_argument("--cache-dir", default="",
                        help="directory holding a local copy of the source jsonl files")
    parser.add_argument("--min-chars", type=int, default=300,
                        help="shortest kept document (default: 300)")
    parser.add_argument("--min-alpha-ratio", type=float, default=0.72,
                        help="lowest kept letter-and-space ratio (default: 0.72)")
    parser.add_argument("--min-diversity", type=float, default=0.32,
                        help="lowest kept unique-word ratio (default: 0.32)")
    parser.add_argument("--max-fragmentation", type=float, default=0.35,
                        help="highest kept ratio of one and two character tokens (default: 0.35)")
    parser.add_argument("--keep-partial-sentences", action="store_true",
                        help="keep the fragments the chunker leaves at both ends of a record")
    parser.add_argument("--keep-boilerplate", action="store_true",
                        help="keep the administrative front matter of the publications")
    parser.add_argument("--report-only", action="store_true",
                        help="run the filters and report, without writing the corpus")
    args = parser.parse_args()

    seen = set()
    counts = {"read": 0, "no_text": 0, "too_short": 0, "low_alpha": 0,
              "repetitive": 0, "fragmented": 0, "boilerplate": 0,
              "duplicate": 0, "kept": 0}
    kept_chars = 0
    sources = set()
    samples = []

    out = None
    if not args.report_only:
        out = open(args.out, "w", encoding="utf-8")

    try:
        for record in stream_records(args.split, args.limit, args.cache_dir):
            counts["read"] += 1
            raw = extract_document(record)
            if not raw:
                counts["no_text"] += 1
                continue

            text = clean(raw)
            if not args.keep_partial_sentences:
                text = trim_partial_sentences(text)
            if len(text) < args.min_chars:
                counts["too_short"] += 1
                continue

            lowered = text.lower()
            if not args.keep_boilerplate and any(m in lowered for m in BOILERPLATE_MARKERS):
                counts["boilerplate"] += 1
                continue

            alpha, diversity, fragmented = quality(text)
            if alpha < args.min_alpha_ratio:
                counts["low_alpha"] += 1
                continue
            if diversity < args.min_diversity:
                counts["repetitive"] += 1
                continue
            if fragmented > args.max_fragmentation:
                counts["fragmented"] += 1
                continue

            digest = hashlib.sha1(" ".join(lowered.split()).encode("utf-8")).hexdigest()
            if digest in seen:
                counts["duplicate"] += 1
                continue
            seen.add(digest)

            counts["kept"] += 1
            kept_chars += len(text)
            meta = record.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    meta = None
            if isinstance(meta, dict) and meta.get("source"):
                sources.add(meta["source"])
            if len(samples) < 3:
                samples.append(text)

            if out:
                out.write(DOC_SENTINEL + "\n")
                out.write(text + "\n")
    finally:
        if out:
            out.close()

    read = max(counts["read"], 1)
    print("Knowledge-alignment corpus")
    print(f"  source split    : {args.split}")
    print(f"  records read    : {counts['read']}")
    print(f"  documents kept  : {counts['kept']} "
          f"({100.0 * counts['kept'] / read:.1f}% of records)")
    print(f"  source documents: {len(sources)}")
    print(f"  characters kept : {kept_chars} "
          f"(about {kept_chars // 4} tokens on a byte-level BPE)")
    print("  rejected        : "
          f"{counts['too_short']} short, {counts['low_alpha']} non-prose, "
          f"{counts['repetitive']} repetitive, {counts['fragmented']} fragmented, "
          f"{counts['boilerplate']} boilerplate, "
          f"{counts['duplicate']} duplicate, {counts['no_text']} empty")
    if not args.report_only:
        print(f"  written to      : {args.out}")
    if samples:
        print("\nFirst kept document, truncated:")
        print("  " + samples[0][:300].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
