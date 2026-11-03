# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Publishes a version of Pseudalis on GitHub: puts its installer in dist/pseudalis of the
repository, beside the update feed that announces it, written and signed.

    python3 publish_release.py <pseudalis-setup.exe> --key <private key>
                               [--notes <notes.json>] [--sign <pseudalis-sign>]
                               [--dist <folder>] [--dry-run] [--push [--wait <seconds>]]
    python3 publish_release.py --check-key --key <private key> [--sign <pseudalis-sign>]
    python3 publish_release.py --create-key --key <private key> [--sign <pseudalis-sign>]
    python3 publish_release.py --check [--dist <folder>] [--wait <seconds>]

The version is that of common/version.h, which must also be the version the installer carries
in its resources.  The installer is copied into <dist> (dist/pseudalis of the repository by
default) as pseudalis-setup-<version>.exe, and the installers of the other versions leave it:
the repository holds the installer its feed names, the earlier ones staying in its history.
GitHub refuses the files of 100 MB and more and warns from 50 MB: an installer of 95 MB or more
is refused, one of 50 MB or more announced.  The feed names the installer by its file beside it,
with its size and its SHA-256; the program writes it, signs it with pseudalis-sign, checks the
signature with the public key of the sources, which the programs embed, and puts the installer,
then the signature and the feed in place of the published ones.  A failure before that leaves
the published files as they are.

--push then commits the folder and pushes it, after checking first that the branch follows its
remote one and lags behind it in nothing, and that the folder holds no change of its own; it ends
once GitHub serves the feed and its signature as they were written, and the installer by its size
and its SHA-256, which is what the installed copies read (within --wait seconds, 900 by default:
GitHub keeps its answers in cache for some minutes).  Without --push, the program prints the
commands that publish the folder.  --check makes that last check alone, for the folder as it is.

--check-key checks, before the installer is even built, that the private key signs what the
public key of the sources verifies: a feed signed with another key would be ignored by every
installed copy.  --create-key makes the pair once, on the system of the publisher: the private
key at --key, which must not exist yet, readable by its owner alone in a folder of its own, and
the public key of the sources (resources/feed-public-key.pem), which it replaces and which is
committed before the installer is built.  The copies installed before trust the former key
alone: they ignore the feeds of the new one until they are installed again.

notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}.  --dry-run checks the
installer, the version and the notes and prints what would be done, without writing anything.
pseudalis-sign is searched in build-pseudalis/bin and build-pseudalis-linux/bin of the
repository, then in the PATH.
"""
import argparse
import glob
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
DIST = os.path.join(ROOT, "dist", "pseudalis")
# The public key of the feed, which the programs embed when it is there at their configuration.
PUBLIC_KEY = os.path.join(HERE, "..", "resources", "feed-public-key.pem")
# The address of the feed that the programs read, in host/settings.cpp.
SETTINGS = os.path.join(HERE, "..", "host", "settings.cpp")
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


def check_key(signer, key):
    """Raises RuntimeError unless key signs what the public key of the sources verifies."""
    if not os.path.isfile(PUBLIC_KEY):
        raise RuntimeError("the sources hold no public key of the feed (resources/feed-public-key.pem)")
    if not os.path.isfile(key):
        raise RuntimeError("no private key at %s" % key)
    folder = tempfile.mkdtemp(prefix="pseudalis-key-")
    try:
        probe = os.path.join(folder, "probe.json")
        with open(probe, "wb") as f:
            f.write(os.urandom(64))
        subprocess.run([signer, "sign", key, probe], check=True, stdout=subprocess.DEVNULL)
        if subprocess.run(verify_command(signer, probe), stdout=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError("%s is not the key of the public key of the sources: the installed copies would "
                               "ignore the feeds it signs" % key)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def create_key(signer, key):
    """Makes the key pair of the feed: the private key at key, which must not exist, and the public
    key of the sources, which it replaces."""
    if os.path.lexists(key):
        raise RuntimeError("%s exists: --check-key checks it, and a new pair would make every installed copy "
                           "ignore the feeds it signs" % key)
    folder = os.path.dirname(os.path.abspath(key))
    os.makedirs(folder, mode=0o700, exist_ok=True)
    temp = tempfile.mkdtemp(prefix="pseudalis-key-")
    try:
        public = os.path.join(temp, "feed-public-key.pem")
        subprocess.run([signer, "keygen", key, public], check=True, stdout=subprocess.DEVNULL)
        shutil.copyfile(public, PUBLIC_KEY)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def git(*args):
    """The output of git run in the repository; raises RuntimeError with its message when it fails."""
    r = subprocess.run(["git", "-C", ROOT] + list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if r.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), r.stdout.strip()))
    return r.stdout.strip()


def check_repository(rel):
    """Raises RuntimeError unless a push can carry the publication: a branch that follows a remote
    one and lags behind it in nothing, and a folder of the publication without changes of its own."""
    if git("rev-parse", "--abbrev-ref", "HEAD") == "HEAD":
        raise RuntimeError("the repository is on no branch")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    git("fetch", "--quiet")
    behind = int(git("rev-list", "--count", "HEAD..@{u}"))
    if behind:
        raise RuntimeError("the branch lags %d commit(s) behind %s: pull first" % (behind, upstream))
    if git("status", "--porcelain", "--", rel):
        raise RuntimeError("%s holds changes of its own: commit or remove them first" % rel)


def published_address():
    """The address of the folder that the programs read, that of their feed."""
    with open(SETTINGS, encoding="utf-8") as f:
        m = re.search(r'default_feed_url\s*=\s*"(https://[^"]+)/feed\.json"', f.read())
    if not m:
        raise RuntimeError("host/settings.cpp names no address of the feed")
    return m.group(1)


def fetch(url, limit):
    """The body of url, of limit bytes at most, read as the programs read it."""
    request = urllib.request.Request(url, headers={"User-Agent": "pseudalis-publish"})
    with urllib.request.urlopen(request, timeout=60) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("%s is larger than expected" % os.path.basename(url))
    return data


def check_online(dist, wait):
    """Waits until GitHub serves the feed and its signature of dist as they are, then checks the
    installer it serves against the feed; raises RuntimeError past wait seconds."""
    if not os.path.isfile(os.path.join(dist, "feed.json")) or not os.path.isfile(os.path.join(dist, "feed.json.sig")):
        raise RuntimeError("%s holds no signed feed: no version is published there yet" % dist)
    with open(os.path.join(dist, "feed.json"), "rb") as f:
        feed = f.read()
    with open(os.path.join(dist, "feed.json.sig"), "rb") as f:
        sig = f.read()
    stable = json.loads(feed.decode("utf-8"))["channels"]["stable"]
    version, installer = stable["version"], stable["installer"]
    base = published_address()
    deadline = time.time() + wait
    seen = "nothing read yet"
    while True:
        try:
            served = fetch(base + "/feed.json", 1 << 20)
            if served == feed and fetch(base + "/feed.json.sig", 4096) == sig:
                break
            seen = "an earlier feed"
        except (OSError, http.client.HTTPException, RuntimeError) as e:
            seen = str(e)
        if time.time() >= deadline:
            raise RuntimeError("GitHub does not serve the feed of version %s after %d seconds (%s)" %
                               (version, wait, seen))
        time.sleep(20)
    h = hashlib.sha256()
    size = 0
    request = urllib.request.Request(base + "/" + installer["file"], headers={"User-Agent": "pseudalis-publish"})
    with urllib.request.urlopen(request, timeout=120) as r:
        for block in iter(lambda: r.read(1 << 20), b""):
            size += len(block)
            if size > installer["size"]:
                break
            h.update(block)
    if size != installer["size"] or h.hexdigest() != installer["sha256"]:
        raise RuntimeError("GitHub serves another installer than the one the feed of version %s names" % version)
    print("GitHub serves version %s: the feed, its signature and the installer of %d bytes" % (version, size))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer", nargs="?")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--dist", default=DIST)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-key", action="store_true")
    p.add_argument("--create-key", action="store_true")
    p.add_argument("--wait", type=int, default=900)
    a = p.parse_args()
    # The steps print in their order, before the messages of pseudalis-sign and of git.
    sys.stdout.reconfigure(line_buffering=True)
    if a.check_key or a.check or a.create_key:
        try:
            if a.create_key:
                if not a.key:
                    raise RuntimeError("--key names the private key to make")
                create_key(find_signer(a.sign), a.key)
                print("%s made; the public key of the sources, resources/feed-public-key.pem, is replaced: commit it "
                      "before the installer is built, and keep a copy of the private key apart, without which no "
                      "installed copy receives an update" % a.key)
            if a.check_key:
                if not a.key:
                    raise RuntimeError("--key names the private key that signs the feed")
                check_key(find_signer(a.sign), a.key)
                print("%s signs what the public key of the sources verifies" % a.key)
            if a.check:
                check_online(a.dist, a.wait)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError,
                http.client.HTTPException) as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        return 0
    if not a.installer:
        p.error("the installer to publish is missing")
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
        signer = find_signer(a.sign)
        # What can fail before anything is written, the key and the repository, is checked first.
        check_key(signer, a.key)
        if a.push:
            check_repository(rel)
        os.makedirs(a.dist, exist_ok=True)
        left = publish(a.installer, a.dist, name, sha, feed, signer, a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    for other in left:
        print("removed from %s: %s" % (rel, other))
    commit = 'git commit -m "Publish version %s of Pseudalis" -- %s' % (version, rel)
    check = "python3 %s --check" % os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")
    if not a.push:
        print()
        print("git add %s" % rel)
        print(commit)
        print("git push")
        return 0
    # What remains to run when a step fails, the files of the folder being written.
    remaining = "git add %s && %s && git push && %s" % (rel, commit, check)
    try:
        git("add", "--", rel)
        # The same version published again with the same notes on the same day gives the same files.
        if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet", "--", rel]).returncode != 0:
            git("commit", "-m", "Publish version %s of Pseudalis" % version, "--", rel)
        remaining = "git push && %s" % check
        git("push")
        print("version %s pushed" % version)
        remaining = check
        check_online(a.dist, a.wait)
    except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e, file=sys.stderr)
        print("the publication ends, once the cause is removed, with: %s" % remaining, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
