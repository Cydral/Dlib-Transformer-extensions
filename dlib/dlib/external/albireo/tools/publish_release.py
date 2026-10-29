# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Albireo: attaches its installer to the GitHub release of the version,
then writes and signs the update feed that announces it.

    python3 publish_release.py <albireo-setup.exe> --key <private key> [--notes <notes.json>]
                               [--sign <albireo-sign>] [--repo <owner/name>] [--dist <folder>]
                               [--dry-run]

The version is that of common/version.h, which must also be the version the installer
carries in its resources.  The release albireo-v<version> is created when it does not exist
yet, at the head of the default branch of the repository, which therefore holds the sources
of the version once they are pushed.  The installer is attached as albireo-setup-<version>.exe;
an earlier file of that name is replaced when its content differs.

The program then writes <dist>/feed.json (dist/albireo of the repository by default) with
the address, the size and the SHA-256 of the installer, signs it with albireo-sign, checks
the signature, reads the beginning of the installer back from its public address, and
prints the commands that publish the feed.  notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}, which also become the
description of the release.

--dry-run checks the installer, the version and the notes and prints what would be done,
without GitHub and without writing the feed.

The token of GitHub comes from GITHUB_TOKEN, GH_TOKEN or gh auth token (see
github_release.py); albireo-sign is searched in build-albireo/bin of the repository, then in
the PATH.
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

import github_release  # noqa: E402
import make_feed  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
LANGUAGES = (("en", "English"), ("fr", "Français"), ("de", "Deutsch"), ("it", "Italiano"), ("es", "Español"))


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


def release_notes(version, notes):
    lines = []
    for code, title in LANGUAGES:
        if code in notes:
            lines += ["## " + title, "", notes[code], ""]
    lines.append("Installer: albireo-setup-%s.exe. Installed copies of Albireo receive this version through "
                 "the signed feed of dist/albireo." % version)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--repo", default=github_release.DEFAULT_REPOSITORY)
    p.add_argument("--dist", default=os.path.join(ROOT, "dist", "albireo"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    try:
        version = source_version()
        feed_path = os.path.join(a.dist, "feed.json")
        notes = make_feed.read_notes(a.notes, feed_path, version)
        tag = "albireo-v" + version
        name = "albireo-setup-%s.exe" % version
        url = github_release.download_url(a.repo, tag, name)
        size = os.path.getsize(a.installer)
        if size > github_release.MAX_ASSET:
            raise RuntimeError("the installer holds %d bytes, more than the 2 GiB of a release file" % size)
        sha = make_feed.sha256_of(a.installer)
        # The feed is checked before anything is published: version, address, notes.
        feed = make_feed.make_feed(version, a.installer, url, notes, size, sha)
        try:
            with open(feed_path, encoding="utf-8") as f:
                published = json.load(f)["channels"]["stable"]["version"]
            if compare(version, published) < 0:
                raise RuntimeError("the feed already announces %s, later than %s" % (published, version))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("release %s of %s, file %s" % (tag, a.repo, name))
        if a.dry_run:
            print("dry run: nothing published")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        signer = find_signer(a.sign)

        repo = github_release.repository(a.repo)
        release = repo.ensure_release(tag, "Albireo " + version, release_notes(version, notes))
        asset = repo.attach(release, a.installer, name, sha)
        print("attached: %s" % asset.get("browser_download_url", url))

        make_feed.write_feed(feed_path, feed)
        subprocess.run([signer, "sign", a.key, feed_path], check=True)
        subprocess.run([signer, "verify", feed_path], check=True)
        try:
            ok = github_release.read_back(url, a.installer)
            print("public address checked" if ok else "WARNING: the public address serves another file: " + url)
        except OSError as e:
            print("WARNING: the public address cannot be read (%s); a private repository serves its files to "
                  "signed-in users only" % e)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1

    rel = os.path.relpath(feed_path, ROOT).replace(os.sep, "/")
    print()
    print("git add %s %s.sig" % (rel, rel))
    print('git commit -m "Albireo %s"' % version)
    print("git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
