#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Collects a corpus for training a BPE tokenizer, from five kinds of source at once.
#
# WHY FIVE AND NOT ONE
#
# A tokenizer is not trained to know things, it is trained to cut text economically. What it
# cuts badly costs tokens on every text of that kind for the life of the model, so the only
# question that matters is whether the corpus resembles what the model will read and write.
#
# Encyclopaedic prose, from Wikipedia, gives clean well-formed sentences across languages.
# News, from the Leipzig collections, gives a different register and independent sentences.
# FineWeb-Edu gives English filtered for quality, which is the deepest well available.
# FineWeb-2 reaches the languages the first three cover thinly, non-Latin scripts above all.
#
# And instruction data, which is the one people leave out. A tokenizer trained on prose
# alone cuts numbered lists, code blocks, tabular answers and turn markers into far more
# tokens than it needs, and those are precisely what a chat model produces all day. A few
# percent of the corpus is enough to fix it and cannot be recovered afterwards.
#
# WHAT COMES OUT
#
# One text file, deduplicated and shuffled, sized to what was asked for. Passing it through
# --distil afterwards replaces it with a much smaller file that produces the same merge
# table, which is what makes training on a multi-gigabyte sample practical: BPE merges are
# driven entirely by pre-token frequencies, so a corpus reproducing those frequencies
# reproduces the tokenizer.
#
# Usage:
#   slm_bpe_corpus.py --out corpus.txt --target-mb 200
#   slm_bpe_corpus.py --out corpus.txt --target-mb 4000 --languages en,fr,de,es,zh,ja
#   slm_bpe_corpus.py --out corpus.txt --target-mb 200 --sources wikipedia,fineweb,chat
#   slm_bpe_corpus.py --distil corpus.txt --distil-out corpus_small.txt --keep-mb 60

import argparse
import hashlib
import io
import os
import random
import re
import sys
import tarfile
import tempfile
import unicodedata
import urllib.request
from collections import Counter

# Languages, and how each source names them. A single table rather than four, because the
# commonest way to end up with a corpus missing a language is to spell it differently in
# one place.
LANGUAGES = {
    #        wikipedia   leipzig       fineweb-2
    "en": ("20231101.en", "eng_news_2023_1M", "eng_Latn"),
    "fr": ("20231101.fr", "fra_news_2023_1M", "fra_Latn"),
    "de": ("20231101.de", "deu_news_2023_1M", "deu_Latn"),
    "es": ("20231101.es", "spa_news_2023_1M", "spa_Latn"),
    "it": ("20231101.it", "ita_news_2023_1M", "ita_Latn"),
    "pt": ("20231101.pt", "por_news_2023_1M", "por_Latn"),
    "nl": ("20231101.nl", "nld_news_2023_1M", "nld_Latn"),
    "ru": ("20231101.ru", "rus_news_2023_1M", "rus_Cyrl"),
    "zh": ("20231101.zh", None, "cmn_Hani"),
    "ja": ("20231101.ja", None, "jpn_Jpan"),
    "ar": ("20231101.ar", None, "arb_Arab"),
    "pl": ("20231101.pl", "pol_news_2023_1M", "pol_Latn"),
}

LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/{}.tar.gz"

# The share each source takes of the requested size. English is over-represented on purpose:
# the filtered English data is of a quality the other languages have no equivalent for, and
# a tokenizer that cuts English well cuts every Latin script better.
# Seven languages rather than eleven, and which seven follows from the vocabulary rather
# than from any ranking of importance.
#
# A merge table is a fixed budget: 24,576 entries, of which 256 are the raw bytes. Latin
# scripts share their subwords almost entirely, so English, French, German and Spanish cost
# roughly twelve thousand merges between them and adding Italian, Portuguese or Dutch buys
# very little. Every other script starts from nothing: Cyrillic needs its own alphabet,
# Han needs a merge for each of some three thousand five hundred characters that UTF-8
# spells in three bytes apiece.
#
# Japanese is the bargain of the list. Its kana are a couple of hundred characters, and its
# kanji are the Han merges Chinese has already paid for. Arabic, by contrast, is a script
# entirely of its own and costs about as much as Cyrillic; it is the natural eighth if the
# vocabulary is raised.
#
# Spreading four gigabytes over eleven languages leaves each of them underfed, and a script
# that appears too rarely ends up spelled byte by byte in the finished tokenizer, which is
# the worst outcome: it costs vocabulary and delivers nothing.
SOURCE_SHARES = {
    "wikipedia": 0.35,
    "leipzig": 0.15,
    "fineweb": 0.35,
    "chat": 0.15,
}


def normalize(text):
    """Unicode normalization and the removal of what a tokenizer should not learn.

    NFC because the same accented character written two ways would otherwise train two
    different merge paths, and the model would then depend on which form its input used.
    Control characters go because they survive into the vocabulary as tokens nobody can type.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c == "\n" or c == "\t"
                   or not unicodedata.category(c).startswith("C"))
    return text.strip()


# Scripts whose characters carry roughly a word each, rather than roughly a letter.
#
# Two hundred characters of French is a long sentence; two hundred characters of Chinese is
# several paragraphs. Applying one threshold to both would keep trivia in one language while
# discarding legitimate passages in another, and the corpus would end up unbalanced in a way
# no later step could detect.
DENSE_SCRIPTS = ("Han", "Hiragana", "Katakana", "Hangul")


def script_density(text):
    """1 for an alphabetic script, about 4 for one where a character is a word."""
    sample = text[:400]
    dense = sum(1 for c in sample
                if any(s in unicodedata.name(c, "") for s in ("CJK", "HIRAGANA",
                                                              "KATAKANA", "HANGUL")))
    return 4.0 if dense > len(sample) * 0.2 else 1.0


def acceptable(text, min_chars):
    """Whether a passage is worth learning from.

    The tests are deliberately crude. A tokenizer trained on navigation menus and cookie
    banners spends its vocabulary on them, and no amount of subtlety here beats simply
    requiring that a passage look like sentences.
    """
    if len(text) < min_chars / script_density(text):
        return False
    # isalpha covers every script Python knows, Arabic and Han included, so the test is
    # about the proportion of writing to punctuation and digits rather than about the Latin
    # alphabet. Lowered for dense scripts, where punctuation weighs more per character.
    letters = sum(1 for c in text if c.isalpha())
    floor = 0.3 if script_density(text) > 1 else 0.5
    if letters / max(len(text), 1) < floor:
        return False
    # A line that repeats itself is a menu, a table of contents or a generated listing.
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.6:
        return False
    return True


class Sink:
    """Collects passages up to a byte budget, dropping what it has already seen.

    Deduplication matters more for a tokenizer than for a language model: a boilerplate
    paragraph repeated ten thousand times does not teach the model much, but it does teach
    the tokenizer to spend merges on it.
    """

    def __init__(self, path, budget_bytes):
        self.file = open(path, "w", encoding="utf-8")
        self.budget = budget_bytes
        self.written = 0
        self.seen = set()
        self.kept = 0
        self.dropped = 0

    def add(self, text):
        if self.written >= self.budget:
            return False
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if digest in self.seen:
            self.dropped += 1
            return True
        self.seen.add(digest)
        self.file.write(text + "\n\n")
        self.written += len(text.encode("utf-8")) + 2
        self.kept += 1
        return self.written < self.budget

    def close(self):
        self.file.close()


def from_wikipedia(sink, lang, budget, min_chars):
    from datasets import load_dataset
    config = LANGUAGES[lang][0]
    if not config:
        return 0
    start = sink.written
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    for article in ds:
        text = normalize(article.get("text", ""))
        if acceptable(text, min_chars) and not sink.add(text[:20000]):
            break
        if sink.written - start >= budget:
            break
    return sink.written - start


def from_leipzig(sink, lang, budget, min_chars):
    """The Leipzig collections arrive as a tarball of one sentence per line.

    Downloaded rather than streamed, since they are single archives of a couple of hundred
    megabytes and there is no partial reading of a gzip member worth the trouble.
    """
    name = LANGUAGES[lang][1]
    if not name:
        return 0
    start = sink.written
    url = LEIPZIG_URL.format(name)
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            path = tmp.name
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path, "r:gz") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("-sentences.txt")), None)
            if member is None:
                return 0
            stream = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8",
                                      errors="replace")
            block = []
            for line in stream:
                # Each line is "<number>\t<sentence>".
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                block.append(normalize(parts[1]))
                # Sentences are grouped so that the sink's deduplication works on passages
                # rather than on individual sentences, which repeat legitimately.
                if len(block) >= 20:
                    text = " ".join(block)
                    block = []
                    if acceptable(text, min_chars) and not sink.add(text):
                        break
                    if sink.written - start >= budget:
                        break
    except Exception as e:                                        # noqa: BLE001
        print(f"    leipzig {lang}: unavailable ({e})")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return sink.written - start


def from_fineweb(sink, lang, budget, min_chars):
    from datasets import load_dataset
    start = sink.written
    if lang == "en":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
    else:
        config = LANGUAGES[lang][2]
        if not config:
            return 0
        ds = load_dataset("HuggingFaceFW/fineweb-2", config, split="train", streaming=True)
    for sample in ds:
        text = normalize(sample.get("text", ""))
        if acceptable(text, min_chars) and not sink.add(text[:20000]):
            break
        if sink.written - start >= budget:
            break
    return sink.written - start


def from_chat(sink, budget, min_chars):
    """Instruction data, rendered plainly rather than through any chat template.

    No template here on purpose. The tokenizer must learn the shape of an answer, its lists,
    its code fences and its tabulations; the turn markers themselves are special tokens added
    to the vocabulary explicitly, not merges to be discovered. Rendering a template would
    only teach it to spend merges on strings it will be handed whole.
    """
    from datasets import load_dataset
    start = sink.written
    for repo, config in [("HuggingFaceTB/smoltalk", "all"),
                         ("CohereLabs/aya_dataset", "default")]:
        try:
            ds = load_dataset(repo, config, split="train", streaming=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"    {repo}: unavailable ({e})")
            continue
        half = start + budget // 2 if repo.endswith("smoltalk") else start + budget
        for sample in ds:
            if sample.get("messages"):
                text = "\n\n".join(m.get("content", "") for m in sample["messages"])
            elif sample.get("inputs"):
                text = sample["inputs"] + "\n\n" + sample.get("targets", "")
            else:
                continue
            text = normalize(text)
            if acceptable(text, min_chars) and not sink.add(text):
                break
            if sink.written >= half:
                break
    return sink.written - start


# ---------------------------------------------------------------------------------------
# Reduction to a corpus that trains the same tokenizer, far faster.

# How the corpus is cut before frequencies are counted.
#
# The obvious rule, a run of word characters, is wrong on scripts that do not separate
# words. \w+ swallows an entire Chinese sentence as one pre-token: twenty-nine characters
# become one unit that occurs exactly once in the whole corpus. Millions of such units
# appeared in a seven-language corpus, and each is useless twice over. It carries no
# frequent merge, since it is seen once. And it survives the reduction below, which keeps
# every distinct pre-token at least once, so it inflates the reduced corpus and the training
# time that follows without contributing a single merge.
#
# Dense scripts are therefore cut per character, and that alternative comes first: Unicode
# classifies a Han character as a letter, so any general letter rule placed ahead of it would
# swallow the run before the specific one could see it. Cutting per character is what the
# merges rebuild from anyway, and it is what makes those frequencies countable at all.
PRETOKEN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]"  # kana and Han, one at a time
    r"| ?[^\W\d_]+"                                   # letters of any other script
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"|\s+",
    re.UNICODE)


def distil(source, target, keep_bytes, seed):
    """Rewrites a corpus as a shorter one with the same pre-token frequencies.

    BPE merges depend on nothing else. Two corpora whose pre-tokens occur in the same
    proportions produce the same merge table, so the frequencies can be counted once and
    then replayed at whatever scale fits the time available. A gigabyte becomes fifty
    megabytes and the resulting tokenizer is the same one.

    The proportions are preserved rather than the ranking: keeping only the commonest
    pre-tokens would drop the long tail that decides how unusual words are split, which is
    most of what a tokenizer is for.

    Two things to know before relying on the output. Keeping every distinct pre-token at
    least once puts a floor under the result, so a corpus with millions of rare pre-tokens
    reduces less than the requested size suggests. And the proportions come out close rather
    than exact: measured on a synthetic corpus, frequent Latin pre-tokens land within a
    point of their source share, while single CJK characters have shown larger deviation
    that is not yet explained. It is worth measuring on your own corpus before a long run.
    """
    counts = Counter()
    total = 0
    with open(source, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            for tok in PRETOKEN.findall(line):
                counts[tok] += 1
                total += 1
    if not counts:
        sys.exit("the source corpus yielded no pre-token")

    print(f"  pre-tokens     : {total} occurrences, {len(counts)} distinct")

    # Every distinct pre-token appears at least once, so nothing is lost from the tail;
    # what is compressed is how often the frequent ones repeat.
    scale = keep_bytes / max(sum(len(t.encode("utf-8")) * c for t, c in counts.items()), 1)
    pieces = []
    for tok, count in counts.items():
        pieces.extend([tok] * max(1, int(count * scale)))

    random.Random(seed).shuffle(pieces)
    with open(target, "w", encoding="utf-8") as fout:
        line = []
        size = 0
        for piece in pieces:
            line.append(piece)
            size += len(piece)
            if size > 400:
                fout.write("".join(line) + "\n")
                line, size = [], 0
        if line:
            fout.write("".join(line) + "\n")

    written = os.path.getsize(target)
    print(f"  written        : {written / 1e6:.1f} MB "
          f"({100.0 * written / os.path.getsize(source):.1f}% of the source)")


# ---------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Collect a corpus for BPE tokenizer training, from several sources.")
    p.add_argument("--out", help="corpus to write")
    p.add_argument("--target-mb", type=float, default=200.0,
                   help="size to collect, in megabytes (default: 200)")
    p.add_argument("--languages", default="en,fr,de,es,ru,zh,ja",
                   help="comma-separated, the first taking a double share. "
                        f"Known: {','.join(LANGUAGES)}")
    p.add_argument("--sources", default="wikipedia,leipzig,fineweb,chat",
                   help="which of the four kinds to draw from")
    p.add_argument("--min-chars", type=int, default=200,
                   help="shortest passage worth keeping (default: 200)")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--distil", metavar="CORPUS",
                   help="reduce an existing corpus to one that trains the same tokenizer")
    p.add_argument("--distil-out", help="where the reduced corpus goes")
    p.add_argument("--keep-mb", type=float, default=50.0,
                   help="size of the reduced corpus, in megabytes (default: 50)")
    args = p.parse_args()

    if args.distil:
        if not args.distil_out:
            sys.exit("--distil needs --distil-out")
        print(f"reducing {args.distil}")
        distil(args.distil, args.distil_out, int(args.keep_mb * 1e6), args.seed)
        return

    if not args.out:
        sys.exit("--out is required unless --distil is given")

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    unknown = [l for l in langs if l not in LANGUAGES]
    if unknown:
        sys.exit(f"unknown languages: {', '.join(unknown)}")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    budget = int(args.target_mb * 1e6)
    print(f"target      : {args.target_mb:.0f} MB")
    print(f"languages   : {', '.join(langs)}")
    print(f"sources     : {', '.join(sources)}")

    # Languages share the size unevenly: the first named takes a double share, since it is
    # normally the model's primary language and the one whose segmentation matters most.
    weights = [2.0] + [1.0] * (len(langs) - 1)
    lang_budget = {l: budget * w / sum(weights) for l, w in zip(langs, weights)}

    sink = Sink(args.out, budget)
    try:
        for source in sources:
            share = SOURCE_SHARES.get(source, 0.0)
            if share <= 0:
                continue
            print(f"\n{source} ({100 * share:.0f}% of the target)")
            if source == "chat":
                got = from_chat(sink, int(budget * share), args.min_chars)
                print(f"  collected    : {got / 1e6:.1f} MB")
                continue
            fetch = {"wikipedia": from_wikipedia, "leipzig": from_leipzig,
                     "fineweb": from_fineweb}[source]
            for lang in langs:
                got = fetch(sink, lang, int(lang_budget[lang] * share), args.min_chars)
                print(f"  {lang} : {got / 1e6:6.1f} MB")
    except KeyboardInterrupt:
        print("\ninterrupted; keeping what was collected so far")
    finally:
        sink.close()

    size = os.path.getsize(args.out)
    print(f"\ncorpus      : {size / 1e6:.1f} MB, {sink.kept} passages, "
          f"{sink.dropped} duplicates dropped")
    print(f"written to  : {args.out}")
    print("\nTraining a BPE on a corpus this size is slow, and needlessly so: the merge")
    print("table depends only on pre-token frequencies. Reduce it first,")
    print(f"  {sys.argv[0]} --distil {args.out} --distil-out small.txt --keep-mb 50")

    # Leaves without letting the interpreter finalize.
    #
    # Streaming a dataset leaves worker threads behind, and they touch the interpreter while
    # it is shutting down, which aborts the process on a GIL assertion long after the work is
    # finished. The corpus is written and the handles are closed by this point, so there is
    # nothing left to unwind: the crash was cosmetic and alarming, which is the worst
    # combination. Everything that must reach the disk is flushed above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
