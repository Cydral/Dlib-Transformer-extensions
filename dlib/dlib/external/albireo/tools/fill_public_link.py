# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Fills the local copy of the public link of the published files of Albireo, the folder that the
storage service synchronizes and publishes (host/public_link.h), and checks the link once the
copy is synchronized.

    python3 fill_public_link.py dist <copy>/dist --online <albireo-setup-online.exe>
                                [--complete <albireo-setup.exe>] [--dist <dist/albireo>]
                                [--sign <albireo-sign>]
    python3 fill_public_link.py models <copy>/models [--models <id,id,...>] [--from <folder>]...
                                [--dist <dist/albireo>]
    python3 fill_public_link.py check [--link <public link>] [--dist <dist/albireo>]

dist copies the signed feed and the signed catalog of dist/albireo with their signatures, checked
with albireo-sign (--sign, else build-albireo/bin of the repository, else the PATH), the installer
of the updates under the name the feed gives it in the link, checked against the size and the
SHA-256 the feed gives, and the complete installer as albireo-setup-complete-<version>.exe, checked
against the version of the feed, with the file of its models when they lie beside it
(albireo-setup.models of a build, as albireo-setup-complete-<version>.models).  The feed must name
its installer in the link: a version published with publish_release.py --public-link.

models places each file of the catalog, decoders and projectors, each part of a file published in
parts as it is published, at <identifier>/<file name of its official address>: kept when it is
there at its size with its SHA-256, else copied from a file of the folders of --from of that size
and SHA-256, else downloaded from its official address, resumed after a cut, and checked.  A file
takes its name once checked, so that the link never shows a file cut short under its final name.
--models limits the copy to the models named.

check reads the listing of the link through its API, with the password of public_link.py, and
lists the files of the feed, of the catalog and of the models the link lacks or holds at another
size: a synchronization in progress shows its missing files, a complete one none.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402
import public_link  # noqa: E402
import publish_release  # noqa: E402

DIST = os.path.join(publish_release.ROOT, "dist", "albireo")


def load(dist, name):
    with open(os.path.join(dist, name), encoding="utf-8") as f:
        return json.load(f)


def parts_of(catalog, chosen=None):
    """(identifier, label, part) of each published file of the models of the catalog."""
    for entry in catalog["models"]:
        if chosen is not None and entry["id"] not in chosen:
            continue
        for part in entry["files"]:
            yield entry["id"], entry["id"], part
        for part in entry.get("projector", {}).get("files", []):
            yield entry["id"], "the projector of %s" % entry["id"], part


def raw_size(part):
    """The bytes of a published part: compressed, the compressed size."""
    return part["compressed_size"] if part.get("compression") == "gzip" else part["size"]


def matches(path, part):
    """Whether the file at path is the published part: its size, and the SHA-256 of its content,
    inflated for a compressed part."""
    if not os.path.isfile(path) or os.path.getsize(path) != raw_size(part):
        return False
    h = hashlib.sha256()
    produced = 0
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS) if part.get("compression") == "gzip" else None
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                if inflate:
                    block = inflate.decompress(block)
                produced += len(block)
                h.update(block)
            if inflate:
                tail = inflate.flush()
                produced += len(tail)
                h.update(tail)
    except zlib.error:
        return False
    return produced == part["size"] and h.hexdigest() == part["sha256"]


def download(url, partial, size, label):
    """Downloads url into partial, resuming the bytes partial holds already, up to size bytes;
    three attempts, the next ones resuming where the previous stopped."""
    for attempt in range(3):
        have = os.path.getsize(partial) if os.path.isfile(partial) else 0
        if have > size:
            os.remove(partial)
            have = 0
        if have == size:
            return
        headers = {"User-Agent": "albireo-tools"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                # A server that ignores the range sends the whole file again: the copy starts over.
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                shown = 0.0
                with open(partial, mode) as out:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        have += len(block)
                        if have > size:
                            raise RuntimeError("%s sends more than the %d bytes announced" % (url, size))
                        out.write(block)
                        if time.monotonic() - shown >= 2:
                            shown = time.monotonic()
                            print("\r  %s: %d of %d MB" % (label, have >> 20, size >> 20), end="", file=sys.stderr)
            print("\r  %s: %d of %d MB" % (label, have >> 20, size >> 20), file=sys.stderr)
            if have == size:
                return
        except (OSError, urllib.error.URLError) as e:
            print("\n  %s: attempt %d: %s" % (label, attempt + 1, e), file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("%s: the download stops before its end" % label)


def local_files(folders):
    """The files of the folders, by size."""
    by_size = {}
    for folder in folders:
        for dirpath, _, names in os.walk(folder):
            for n in names:
                path = os.path.join(dirpath, n)
                try:
                    by_size.setdefault(os.path.getsize(path), []).append(path)
                except OSError:
                    pass
    return by_size


def fill_models(a):
    catalog = load(a.dist, "catalog.json")
    chosen = None
    if a.models:
        chosen = {m.strip() for m in a.models.split(",") if m.strip()}
        unknown = chosen - {e["id"] for e in catalog["models"]}
        if unknown:
            raise RuntimeError("not in the catalog: " + ", ".join(sorted(unknown)))
    by_size = local_files(a.sources)
    total = done = 0
    for identifier, label, part in parts_of(catalog, chosen):
        path = public_link.model_path(identifier, part["url"])
        target = os.path.join(a.output, *path.split("/")[1:])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        total += raw_size(part)
        if matches(target, part):
            print("%s: %s in place" % (label, path))
            done += 1
            continue
        partial = target + ".partial"
        copied = False
        if part.get("compression") != "gzip":
            for candidate in by_size.get(part["size"], []):
                if make_feed.sha256_of(candidate) == part["sha256"]:
                    shutil.copyfile(candidate, partial)
                    print("%s: %s copied from %s" % (label, path, candidate))
                    copied = True
                    break
        if not copied:
            download(part["url"], partial, raw_size(part), label)
        if not matches(partial, part):
            os.remove(partial)
            raise RuntimeError("%s: %s does not match the catalog" % (label, part["url"]))
        os.replace(partial, target)
        if not copied:
            print("%s: %s downloaded" % (label, path))
        done += 1
    print("%d files, %.1f GB, in %s" % (done, total / 1e9, a.output))
    return 0


def fill_dist(a):
    signer = publish_release.find_signer(a.sign)
    for name in ("feed.json", "catalog.json"):
        subprocess.run([signer, "verify", os.path.join(a.dist, name)], check=True, stdout=subprocess.DEVNULL)
    stable = load(a.dist, "feed.json")["channels"]["stable"]
    installer = stable["installer"]
    version = stable["version"]
    if not installer.get("file"):
        raise RuntimeError("the feed of %s names no installer in the public link: publish the version with "
                           "publish_release.py --public-link" % a.dist)
    if os.path.getsize(a.online) != installer["size"] or make_feed.sha256_of(a.online) != installer["sha256"]:
        raise RuntimeError("%s is not the installer the feed announces" % a.online)
    if a.complete:
        make_feed.check_installer(a.complete, version)
    os.makedirs(a.output, exist_ok=True)
    copies = [(os.path.join(a.dist, n + s), n + s) for n in ("feed.json", "catalog.json") for s in ("", ".sig")]
    copies.append((a.online, installer["file"]))
    beside = "albireo-setup-complete-%s.models" % version
    if a.complete:
        copies.append((a.complete, "albireo-setup-complete-%s.exe" % version))
        # The models of a complete installer that would reach the 4 GiB Windows runs lie beside it.
        models = os.path.splitext(a.complete)[0] + ".models"
        if os.path.isfile(models):
            copies.append((models, beside))
        elif os.path.isfile(os.path.join(a.output, beside)):
            os.remove(os.path.join(a.output, beside))
    # The installers first and the feed last: the link never announces a file it does not hold yet.
    copies.sort(key=lambda c: c[1].startswith("feed.json"))
    for source, name in copies:
        target = os.path.join(a.output, name)
        partial = target + ".partial"
        shutil.copyfile(source, partial)
        os.replace(partial, target)
        print("%s: %d bytes" % (name, os.path.getsize(target)))
    print("version %s in %s" % (version, a.output))
    return 0


def check(a):
    link = public_link.Link(a.link)
    files = link.files()
    expected = [("dist/" + n, None) for n in ("feed.json", "feed.json.sig", "catalog.json", "catalog.json.sig")]
    installer = load(a.dist, "feed.json")["channels"]["stable"]["installer"]
    if installer.get("file"):
        expected.append(("dist/" + installer["file"], installer["size"]))
    for identifier, _, part in parts_of(load(a.dist, "catalog.json")):
        expected.append((public_link.model_path(identifier, part["url"]), raw_size(part)))
    missing = 0
    for path, size in expected:
        held = files.get(path)
        if held is None:
            print("missing: %s" % path)
            missing += 1
        elif size is not None and held[1] != size:
            print("other size: %s (%d bytes, %d expected)" % (path, held[1], size))
            missing += 1
    print("%d files expected, %d in place, %d missing or of another size" % (len(expected), len(expected) - missing,
                                                                             missing))
    return 1 if missing else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dist")
    d.add_argument("output")
    d.add_argument("--online", required=True)
    d.add_argument("--complete")
    d.add_argument("--sign")
    m = sub.add_parser("models")
    m.add_argument("output")
    m.add_argument("--models")
    m.add_argument("--from", dest="sources", action="append", default=[])
    c = sub.add_parser("check")
    c.add_argument("--link", default=public_link.DEFAULT_LINK)
    for s in (d, m, c):
        s.add_argument("--dist", default=DIST)
    a = p.parse_args()
    try:
        return {"dist": fill_dist, "models": fill_models, "check": check}[a.command](a)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
