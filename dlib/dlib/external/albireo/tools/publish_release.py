# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Albireo: attaches its installers to the GitHub release of the version,
then writes and signs the update feed that announces it.

    python3 publish_release.py <albireo-setup-online.exe> --key <private key>
                               [--complete <albireo-setup.exe>] [--notes <notes.json>]
                               [--sign <albireo-sign>] [--repo <owner/name>] [--dist <folder>]
                               [--dry-run]

The first installer is the one of the updates, without models (albireo-setup-online.exe of a
build): the feed announces it, and the installed copies of Albireo download it.  --complete
also attaches the installer that carries the models that come with Albireo, for the
installations without network, as albireo-setup-complete-<version>.exe, when it fits the
2 GiB of a release file.

The version is that of common/version.h, which must also be the version both installers
carry in their resources.  The release albireo-v<version> is created when it does not exist
yet, at the head of the default branch of the repository, which therefore holds the sources
of the version once they are pushed.  The installer is attached as albireo-setup-<version>.exe;
an earlier file of that name is replaced when its content differs.

The program then reads the whole installer back from its public address, and stops with an
error when the address serves another file, or cannot be read while the repository is public;
the files of a private repository need a sign-in, which a warning recalls.  It writes the
feed, with the address, the size and the SHA-256 of the installer, beside <dist>/feed.json
(dist/albireo of the repository by default), signs it with albireo-sign, checks the
signature, puts the feed and its signature in place of the published ones, and prints the
commands that publish them.  notes.json holds the release notes by language,
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


def release_notes(version, notes, complete):
    lines = []
    for code, title in LANGUAGES:
        if code in notes:
            lines += ["## " + title, "", notes[code], ""]
    lines.append("Installer: albireo-setup-%s.exe, which downloads the models. Installed copies of Albireo "
                 "receive this version through the signed feed of dist/albireo." % version)
    if complete:
        lines.append("")
        lines.append("Complete installer: albireo-setup-complete-%s.exe, with the models that come with Albireo, "
                     "for an installation without network." % version)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer")
    p.add_argument("--complete", help="the complete installer, with the models that come with Albireo")
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
        complete = None
        if a.complete:
            make_feed.check_installer(a.complete, version)
            complete_size = os.path.getsize(a.complete)
            if complete_size > github_release.MAX_ASSET:
                print("WARNING: the complete installer holds %d bytes, more than the 2 GiB of a release file: it "
                      "travels in the offline bundles only" % complete_size)
            else:
                complete = ("albireo-setup-complete-%s.exe" % version, make_feed.sha256_of(a.complete))
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("release %s of %s, file %s" % (tag, a.repo, name))
        if complete:
            print("complete installer: file %s, SHA-256 %s" % complete)
        if a.dry_run:
            print("dry run: nothing published")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        signer = find_signer(a.sign)

        repo = github_release.repository(a.repo)
        release = repo.ensure_release(tag, "Albireo " + version, release_notes(version, notes, complete))
        asset = repo.attach(release, a.installer, name, sha)
        print("attached: %s" % asset.get("browser_download_url", url))
        if complete:
            asset = repo.attach(release, a.complete, complete[0], complete[1])
            print("attached: %s" % asset.get("browser_download_url", complete[0]))

        # The feed changes only once the public address serves the installer it announces.
        warning = github_release.verify_public(repo, url, a.installer, sha)
        print("WARNING: " + warning if warning else "public address checked: " + url)
        write_signed_feed(feed_path, feed, signer, a.key)
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
