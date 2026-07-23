#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Task-alignment dataset preparation, second stage of the fine-tuning pipeline.
#
# Streams the CVE records dataset published on the Hugging Face hub and turns it into
# supervised question-and-answer records: a system block, a user question and the
# reference answer the model is scored on. The knowledge-alignment stage that precedes
# it (see nist_corpus_prepare.py) exposes the model to the domain's language; this one
# teaches it to answer.
#
# Two adjustments matter more than the volume of records.
#
# The source asks the same templated question 297441 times. A model fine-tuned on it
# learns that one phrasing and answers poorly to any other, so the questions are
# rewritten from a small set of natural forms, picked deterministically from the CVE
# identifier: the same record always yields the same question, and a rerun produces a
# byte-identical dataset.
#
# The source answers carry a reference list of bare URLs, often longer than the
# vulnerability description itself. Those positions are scored like any other, so they
# spend the training budget teaching the model to invent plausible URLs. They are
# dropped by default and kept only on request.
#
# The output is a sentinel-separated UTF-8 file read by the C++ example and passed to
# dlib::build_supervised_finetuning_dataset(), which masks the prompt positions so that
# only the answer contributes to the loss. The chat template is deliberately not
# applied here: the C++ side renders it with the model's own chat_template_formatter,
# which keeps one implementation of the turn markers for training and inference alike.
#
# Usage:
#   cve_qa_prepare.py --out cve_qa.txt --limit 20000 --year-min 2015
#   cve_qa_prepare.py --out cve_qa.txt --valid-out cve_qa_valid.txt --valid-fraction 0.05

import argparse
import hashlib
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    sys.exit("this script needs the requests package: pip install requests")

HF_URL = ("https://huggingface.co/datasets/AlicanKiraz0/"
          "All-CVE-Records-Training-Dataset/resolve/main/all_cve_database.jsonl")

RECORD_SENTINEL = "<<<record>>>"
SYSTEM_SENTINEL = "<<<system>>>"
USER_SENTINEL = "<<<user>>>"
ASSISTANT_SENTINEL = "<<<assistant>>>"

DEFAULT_SYSTEM = ("You are a cybersecurity analyst. Answer questions about published "
                  "vulnerabilities accurately and concisely, and say so when a record "
                  "carries no usable information.")

# Question forms covering the ways the same record is actually asked about: an open
# request, a scoped one, a severity question, a remediation question. The identifier is
# always present, since it is the key the answer is retrieved by.
QUESTION_FORMS = (
    "What is {cve}?",
    "Explain {cve}.",
    "Describe the vulnerability tracked as {cve}.",
    "Provide a technical analysis of {cve}, including affected products and impact.",
    "What software is affected by {cve}, and what does the vulnerability allow?",
    "Summarize {cve} and its remediation.",
    "I found a reference to {cve} in a scan report. What does it cover?",
    "How would you characterize the risk of {cve}?",
)

RE_CVE_ID = re.compile(r"CVE-\d{4}-\d{4,7}")
RE_SECTION = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
RE_WS = re.compile(r"[ \t\u00a0]+")
RE_BLANKS = re.compile(r"\n{3,}")

# States and placeholder texts that mark a record with nothing to teach.
EMPTY_MARKERS = (
    "no description available",
    "** rejected **",
    "** reserved **",
    "this candidate has been rejected",
    "do not use this candidate number",
)


def stream_records(limit, cache_dir):
    """Yield the parsed JSON objects of the source dataset.

    Read as a stream so that a bounded run downloads only what it consumes, which keeps
    a --limit 5000 dry run to a few seconds rather than the full 475 MB. A local copy is
    used when present.
    """
    local = os.path.join(cache_dir, "all_cve_database.jsonl") if cache_dir else None
    if local and os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as fin:
            for n, line in enumerate(fin):
                if limit and n >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with requests.get(HF_URL, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"
        for n, line in enumerate(resp.iter_lines(decode_unicode=True)):
            if limit and n >= limit:
                return
            if line:
                yield json.loads(line)


def split_sections(answer):
    """Split a markdown answer into (heading, body) pairs, preamble under an empty key.

    The source answers follow one layout: a title line, then level-three sections. The
    split is what lets a caller drop the reference list without touching the rest.
    """
    sections = []
    matches = list(RE_SECTION.finditer(answer))
    if not matches:
        return [("", answer)]
    if matches[0].start() > 0:
        sections.append(("", answer[:matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        sections.append((m.group(1), answer[m.end():end]))
    return sections


def rebuild_answer(answer, dropped):
    """Reassemble an answer without the named sections, comparison being case-free."""
    lowered = {d.strip().lower() for d in dropped if d.strip()}
    if not lowered:
        return answer
    out = []
    for heading, body in split_sections(answer):
        if heading.lower() in lowered:
            continue
        out.append(body if not heading else "### " + heading + "\n" + body)
    text = "".join(out)
    return RE_BLANKS.sub("\n\n", text).strip()


def pick_question(cve_id):
    """Deterministic question form for a record.

    Derived from the identifier rather than drawn at random, so two runs of this script
    produce the same dataset and a training curve stays comparable across reruns.
    """
    digest = hashlib.sha1(cve_id.encode("utf-8")).digest()
    return QUESTION_FORMS[digest[0] % len(QUESTION_FORMS)].format(cve=cve_id)


def normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_WS.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return RE_BLANKS.sub("\n\n", text).strip()


def write_record(out, system, user, assistant):
    out.write(RECORD_SENTINEL + "\n")
    if system:
        out.write(SYSTEM_SENTINEL + "\n" + system + "\n")
    out.write(USER_SENTINEL + "\n" + user + "\n")
    out.write(ASSISTANT_SENTINEL + "\n" + assistant + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the CVE question-and-answer set for the task-alignment stage.")
    parser.add_argument("--out", default="cve_qa.txt",
                        help="output training file (default: cve_qa.txt)")
    parser.add_argument("--valid-out", default="",
                        help="output validation file, empty writes no validation set")
    parser.add_argument("--valid-fraction", type=float, default=0.0,
                        help="fraction of kept records routed to the validation file")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many source records, 0 reads them all")
    parser.add_argument("--cache-dir", default="",
                        help="directory holding a local copy of all_cve_database.jsonl")
    parser.add_argument("--year-min", type=int, default=0,
                        help="drop CVE identifiers older than this year")
    parser.add_argument("--min-answer-chars", type=int, default=250,
                        help="shortest kept answer (default: 250)")
    parser.add_argument("--max-answer-chars", type=int, default=3000,
                        help="answers longer than this are dropped, 0 disables the bound")
    parser.add_argument("--drop-sections", default="References",
                        help="comma-separated answer sections to remove "
                             "(default: References; pass an empty string to keep all)")
    parser.add_argument("--keep-source-questions", action="store_true",
                        help="keep the templated source question instead of rewriting it")
    parser.add_argument("--system", default=DEFAULT_SYSTEM,
                        help="system block written to every record, empty writes none")
    parser.add_argument("--report-only", action="store_true",
                        help="run the filters and report, without writing the dataset")
    args = parser.parse_args()

    dropped_sections = args.drop_sections.split(",") if args.drop_sections else []
    counts = {"read": 0, "malformed": 0, "no_id": 0, "empty_record": 0,
              "too_old": 0, "too_short": 0, "too_long": 0, "duplicate": 0,
              "kept": 0, "validation": 0}
    seen = set()
    answer_chars = 0
    sample = None

    out = valid = None
    if not args.report_only:
        out = open(args.out, "w", encoding="utf-8")
        if args.valid_out and args.valid_fraction > 0.0:
            valid = open(args.valid_out, "w", encoding="utf-8")

    try:
        for record in stream_records(args.limit, args.cache_dir):
            counts["read"] += 1
            user_in = record.get("User") or ""
            answer = record.get("Assistant") or ""
            if not user_in or not answer:
                counts["malformed"] += 1
                continue

            found = RE_CVE_ID.search(user_in) or RE_CVE_ID.search(answer)
            if not found:
                counts["no_id"] += 1
                continue
            cve_id = found.group(0)

            if args.year_min and int(cve_id.split("-")[1]) < args.year_min:
                counts["too_old"] += 1
                continue

            lowered = answer.lower()
            if any(m in lowered for m in EMPTY_MARKERS):
                counts["empty_record"] += 1
                continue

            if cve_id in seen:
                counts["duplicate"] += 1
                continue
            seen.add(cve_id)

            answer = normalize(rebuild_answer(answer, dropped_sections))
            if len(answer) < args.min_answer_chars:
                counts["too_short"] += 1
                continue
            if args.max_answer_chars and len(answer) > args.max_answer_chars:
                counts["too_long"] += 1
                continue

            question = user_in.strip() if args.keep_source_questions else pick_question(cve_id)

            # Deterministic routing to the validation file, on a hash of the identifier
            # rather than on position: the source is ordered, and a positional cut would
            # put one range of years on one side of the split.
            to_valid = False
            if valid and args.valid_fraction > 0.0:
                bucket = hashlib.sha1(("split:" + cve_id).encode("utf-8")).digest()[0]
                to_valid = bucket < args.valid_fraction * 256

            counts["kept"] += 1
            answer_chars += len(answer)
            if to_valid:
                counts["validation"] += 1
            if sample is None:
                sample = (question, answer)

            target = valid if to_valid else out
            if target:
                write_record(target, args.system.strip(), question, answer)
    finally:
        for f in (out, valid):
            if f:
                f.close()

    read = max(counts["read"], 1)
    train_kept = counts["kept"] - counts["validation"]
    print("Task-alignment dataset")
    print(f"  records read    : {counts['read']}")
    print(f"  records kept    : {counts['kept']} "
          f"({100.0 * counts['kept'] / read:.1f}% of records)")
    print(f"  training        : {train_kept}")
    if counts["validation"]:
        print(f"  validation      : {counts['validation']}")
    if counts["kept"]:
        print(f"  mean answer     : {answer_chars // counts['kept']} characters "
              f"(about {answer_chars // counts['kept'] // 4} tokens)")
    print("  rejected        : "
          f"{counts['empty_record']} rejected or reserved, {counts['too_short']} short, "
          f"{counts['too_long']} long, {counts['too_old']} out of year range, "
          f"{counts['duplicate']} duplicate, {counts['no_id']} without identifier, "
          f"{counts['malformed']} malformed")
    if not args.report_only:
        print(f"  written to      : {args.out}"
              + (f" and {args.valid_out}" if valid else ""))
    if sample:
        print("\nFirst kept record:")
        print("  user      : " + sample[0])
        print("  assistant : " + sample[1][:280].replace("\n", "\n              "))


if __name__ == "__main__":
    main()
