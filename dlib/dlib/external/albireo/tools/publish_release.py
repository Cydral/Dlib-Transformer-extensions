# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Albireo: writes and signs the update feed that announces it, whose
installers lie in the folder dist of the public link of the published files, the only source of
the downloads of the services (host/public_link.h).

    python3 publish_release.py <albireo-setup-online.exe> --key <private key>
                               [--complete <albireo-setup.exe>] [--notes <notes.json>]
                               [--sign <albireo-sign>] [--dist <folder>] [--dry-run]

The first installer is the one of the updates, without models (albireo-setup-online.exe of a
build): the feed names it by its file in the folder dist of the link,
albireo-setup-online-<version>.exe, with its size and its SHA-256, and the installed copies of
Albireo download it from there.  --complete checks the installer that carries the models that
come with Albireo, for the installations without network, which fill_public_link.py dist then
places in the link as albireo-setup-complete-<version>.exe.

The version is that of common/version.h, which must also be the version both installers carry
in their resources.  The program writes the feed beside <dist>/feed.json (dist/albireo of the
repository by default), signs it with albireo-sign, checks the signature, puts the feed and its
signature in place of the earlier ones, and prints the commands that keep them in the repository
and that place the feed, the catalog and both installers in the local copy of the link, the feed
last (fill_public_link.py dist).  notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}; without it, the notes the feed
already gives the same version are kept.

--dry-run checks the installer, the version and the notes and prints what would be done, without
writing the feed.  albireo-sign is searched in build-albireo/bin of the repository, then in the
PATH.
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


def source_version():
    with open(os.path.join(HERE, "..", "common", "version.h"), encoding="utf-8") as f:
        m = re.search(r'#define ALBIREO_VERSION "(\d+\.\d+\.\d+)"', f.read())
    if not m:
        raise RuntimeError("common/version.h names no version")
    return m.group(1)


def find_signer(given):
    if given:
        return given
    bin_dir = os.path.join(ROOT, "build-albireo", "bin")
    for name in ("albireo-sign.exe", "albireo-sign"):
        for folder in (bin_dir, os.path.join(bin_dir, "Release")):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    found = shutil.which("albireo-sign")
    if not found:
        raise RuntimeError("albireo-sign not found: build Albireo, or give its path with --sign")
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
        subprocess.run([signer, "verify", new], check=True)
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
    p.add_argument("--complete", help="the complete installer, with the models that come with Albireo")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--dist", default=os.path.join(ROOT, "dist", "albireo"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    try:
        version = source_version()
        feed_path = os.path.join(a.dist, "feed.json")
        notes = make_feed.read_notes(a.notes, feed_path, version)
        name = "albireo-setup-online-%s.exe" % version
        size = os.path.getsize(a.installer)
        sha = make_feed.sha256_of(a.installer)
        # The feed is checked before anything is written: version, name in the link, notes.
        feed = make_feed.make_feed(version, a.installer, name, notes, size, sha)
        check_newer(feed_path, version)
        if a.complete:
            make_feed.check_installer(a.complete, version)
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("in the public link: dist/%s%s" % (name, ", dist/albireo-setup-complete-%s.exe" % version
                                                  if a.complete else ""))
        if a.dry_run:
            print("dry run: nothing written")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        write_signed_feed(feed_path, feed, find_signer(a.sign), a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    rel = os.path.relpath(feed_path, ROOT).replace(os.sep, "/")
    print()
    print("git add %s %s.sig" % (rel, rel))
    print('git commit -m "Publish version %s of Albireo in its update feed"' % version)
    print("git push")
    print("python3 %s dist <local copy of the link>/dist --online %s%s"
          % (os.path.relpath(os.path.join(HERE, "fill_public_link.py"), ROOT).replace(os.sep, "/"), a.installer,
             " --complete " + a.complete if a.complete else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
