# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Pseudalis: writes and signs the update feed that announces it, the
installer lying in the folder dist of the public link of the published files.

    python3 publish_release.py <pseudalis-setup.exe> --key <private key>
                               [--notes <notes.json>] [--sign <pseudalis-sign>]
                               [--dist <folder>] [--dry-run]

The version is that of common/version.h, which must also be the version the installer carries
in its resources.  The feed names the installer by its file in the folder dist of the public
link, pseudalis-setup-<version>.exe, with its size and its SHA-256; the program writes it beside
<dist>/feed.json (dist/pseudalis of the repository by default), signs it with pseudalis-sign,
checks the signature, puts the feed and its signature in place of the published ones, and prints
the commands that publish them.  fill_public_link.py dist then puts the feed, its signature and
the installer in the local copy of the link.

notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}.  --dry-run checks the
installer, the version and the notes and prints what would be done, without writing the feed.
pseudalis-sign is searched in build-pseudalis/bin and build-pseudalis-linux/bin of the
repository, then in the PATH.
"""

import argparse
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


def write_signed_feed(feed_path, feed, signer, key):
    """Writes the feed and its signature beside feed_path, checks the signature, then puts both
    in place of the published files.  A failure before that leaves the published files as they
    are."""
    new = feed_path + ".new"
    try:
        make_feed.write_feed(new, feed)
        subprocess.run([signer, "sign", key, new], check=True)
        subprocess.run(verify_command(signer, new), check=True)
        os.replace(new, feed_path)
        os.replace(new + ".sig", feed_path + ".sig")
    finally:
        remove_quietly(new)
        remove_quietly(new + ".sig")


def check_newer(feed_path, version):
    """Raises RuntimeError when the feed at feed_path already announces a later version."""
    try:
        with open(feed_path, encoding="utf-8") as f:
            published = json.load(f)["channels"]["stable"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    if compare(version, published) < 0:
        raise RuntimeError("the feed already announces %s, later than %s" % (published, version))


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
        sha = make_feed.sha256_of(a.installer)
        feed = make_feed.make_feed(version, a.installer, None, notes, size, sha, file=name)
        check_newer(feed_path, version)
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("in the public link: dist/%s, dist/pseudalis-feed.json, dist/pseudalis-feed.json.sig" % name)
        if a.dry_run:
            print("dry run: nothing written")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        os.makedirs(a.dist, exist_ok=True)
        write_signed_feed(feed_path, feed, find_signer(a.sign), a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    rel = os.path.relpath(feed_path, ROOT).replace(os.sep, "/")
    print()
    print("git add %s %s.sig" % (rel, rel))
    print('git commit -m "Publish version %s of Pseudalis in its update feed"' % version)
    print("git push")
    print("then fill_public_link.py dist <local copy of the link>/dist --installer <installer>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
