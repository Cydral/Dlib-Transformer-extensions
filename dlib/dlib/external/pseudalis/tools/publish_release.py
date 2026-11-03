# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Pseudalis on GitHub: puts its installer in dist/pseudalis of the
repository, beside the update feed that announces it, written and signed.

    python3 publish_release.py <pseudalis-setup.exe> --key <private key>
                               [--notes <notes.json>] [--sign <pseudalis-sign>]
                               [--dist <folder>] [--dry-run]

The version is that of common/version.h, which must also be the version the installer carries
in its resources.  The installer is copied into <dist> (dist/pseudalis of the repository by
default) as pseudalis-setup-<version>.exe, and the installers of the other versions leave it:
the repository holds the installer its feed names, the earlier ones staying in its history.
GitHub refuses the files of 100 MB and more and warns from 50 MB: an installer of 95 MB or more
is refused, one of 50 MB or more announced.  The feed names the installer by its file beside it,
with its size and its SHA-256; the program writes it, signs it with pseudalis-sign, checks the
signature, puts the installer, then the signature and the feed in place of the published ones,
and prints the commands that publish them.  A failure before that leaves the published files as
they are.

notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}.  --dry-run checks the
installer, the version and the notes and prints what would be done, without writing anything.
pseudalis-sign is searched in build-pseudalis/bin and build-pseudalis-linux/bin of the
repository, then in the PATH.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
DIST = os.path.join(ROOT, "dist", "pseudalis")
# The public key of the feed, which the programs embed when it is there at their configuration.
PUBLIC_KEY = os.path.join(HERE, "..", "resources", "feed-public-key.pem")
# The limits of GitHub for a file of a repository, with a margin under the refusal.
REFUSED_BYTES = 95 << 20
WARNED_BYTES = 50 << 20


def verify_command(signer, path):
    """The command that checks the signature of path: with the public key of the sources when it is
    there, so that a pseudalis-sign built before the key was made checks it too."""
    return [signer, "verify", path] + ([PUBLIC_KEY] if os.path.isfile(PUBLIC_KEY) else [])


def source_version():
    with open(os.path.join(HERE, "..", "common", "version.h"), encoding="utf-8") as f:
        m = re.search(r'#define PSEUDALIS_VERSION "(\d+\.\d+\.\d+)"', f.read())
    if not m:
        raise RuntimeError("common/version.h names no version")
    return m.group(1)


def installer_name(version):
    return "pseudalis-setup-%s.exe" % version


def find_signer(given):
    if given:
        return given
    for tree in ("build-pseudalis", "build-pseudalis-linux"):
        bin_dir = os.path.join(ROOT, tree, "bin")
        for name in ("pseudalis-sign.exe", "pseudalis-sign"):
            for folder in (bin_dir, os.path.join(bin_dir, "Release")):
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    return path
    found = shutil.which("pseudalis-sign")
    if not found:
        raise RuntimeError("pseudalis-sign not found: build Pseudalis, or give its path with --sign")
    return found


def compare(a, b):
    x = [int(p) for p in a.split(".")]
    y = [int(p) for p in b.split(".")]
    return (x > y) - (x < y)


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def check_newer(feed_path, version):
    """Raises RuntimeError when the feed at feed_path already announces a later version."""
    try:
        with open(feed_path, encoding="utf-8") as f:
            published = json.load(f)["channels"]["stable"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    if compare(version, published) < 0:
        raise RuntimeError("the feed already announces %s, later than %s" % (published, version))


def check_size(size):
    if size >= REFUSED_BYTES:
        raise RuntimeError("the installer weighs %.1f MB: GitHub refuses the files of 100 MB and more" %
                           (size / 1048576.0))
    if size >= WARNED_BYTES:
        print("warning: the installer weighs %.1f MB: GitHub warns from 50 MB" % (size / 1048576.0))


def publish(installer, dist, name, sha, feed, signer, key):
    """Puts the installer and the signed feed in dist, the installers of the other versions leaving
    it; returns the names of those that left."""
    feed_path = os.path.join(dist, "feed.json")
    target = os.path.join(dist, name)
    partial = target + ".partial"
    new = feed_path + ".new"
    try:
        shutil.copyfile(installer, partial)
        if make_feed.sha256_of(partial) != sha:
            raise RuntimeError("the copy of the installer differs from it: %s" % partial)
        make_feed.write_feed(new, feed)
        subprocess.run([signer, "sign", key, new], check=True)
        subprocess.run(verify_command(signer, new), check=True)
        os.replace(partial, target)
        os.replace(new + ".sig", feed_path + ".sig")
        os.replace(new, feed_path)
    finally:
        remove_quietly(partial)
        remove_quietly(new)
        remove_quietly(new + ".sig")
    left = []
    for other in sorted(glob.glob(os.path.join(dist, "pseudalis-setup-*.exe"))):
        if os.path.basename(other) != name:
            os.remove(other)
            left.append(os.path.basename(other))
    return left


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--dist", default=DIST)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    try:
        version = source_version()
        feed_path = os.path.join(a.dist, "feed.json")
        notes = make_feed.read_notes(a.notes, feed_path, version)
        name = installer_name(version)
        size = os.path.getsize(a.installer)
        check_size(size)
        sha = make_feed.sha256_of(a.installer)
        feed = make_feed.make_feed(version, a.installer, None, notes, size, sha, file=name)
        check_newer(feed_path, version)
        rel = os.path.relpath(a.dist, ROOT).replace(os.sep, "/")
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("published in %s: %s, feed.json, feed.json.sig" % (rel, name))
        if a.dry_run:
            print("dry run: nothing written")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        os.makedirs(a.dist, exist_ok=True)
        left = publish(a.installer, a.dist, name, sha, feed, find_signer(a.sign), a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    for other in left:
        print("removed from %s: %s" % (rel, other))
    print()
    print("git add %s" % rel)
    print('git commit -m "Publish version %s of Pseudalis" -- %s' % (version, rel))
    print("git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
