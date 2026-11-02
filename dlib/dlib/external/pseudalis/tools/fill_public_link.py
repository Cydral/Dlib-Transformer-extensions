# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Fills the local copy of the public link of the published files of Pseudalis, the folder that
the storage service synchronizes and publishes (host/public_link.h), and checks the link once
the copy is synchronized.

    python fill_public_link.py dist <copy>/dist --installer <pseudalis-setup.exe>
                               [--dist <dist/pseudalis>] [--sign <pseudalis-sign>]
    python fill_public_link.py check [--link <public link>] [--dist <dist/pseudalis>]

dist copies the signed feed of dist/pseudalis as pseudalis-feed.json, with its signature as
pseudalis-feed.json.sig, checked with pseudalis-sign (--sign, else build-pseudalis/bin of the
repository, else the PATH), and the installer under the name the feed gives it in the link,
pseudalis-setup-<version>.exe, checked against the size and the SHA-256 the feed gives.  The
installer goes first and the feed last, each under a temporary name until it is complete, so
that the link never announces a file it does not hold yet.  The files of Albireo in the same
folder stay as they are.

check reads the listing of the link through its API, with the password of public_link.py, and
lists the files of the feed the link lacks or holds at another size: a synchronization in
progress shows its missing files, a complete one none.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402
import public_link  # noqa: E402
import publish_release  # noqa: E402

# The names of the feed and of its signature in the folder dist of the link.
LINK_FEED = "pseudalis-feed.json"


def load_feed(dist):
    with open(os.path.join(dist, "feed.json"), encoding="utf-8") as f:
        return json.load(f)


def verify(signer, path):
    """Checks the signature of path, and names the ends of line a checkout converted when it fails."""
    if subprocess.run(publish_release.verify_command(signer, path), stdout=subprocess.DEVNULL).returncode == 0:
        return
    with open(path, "rb") as f:
        converted = b"\r\n" in f.read()
    raise RuntimeError("the signature of %s does not verify%s" % (path, (
        ": the file has CRLF ends of line, which a checkout with core.autocrlf gives it; delete it, then "
        "git checkout -- dist/pseudalis (dist/pseudalis/.gitattributes keeps the signed files as they are)")
        if converted else ""))


def fill_dist(a):
    signer = publish_release.find_signer(a.sign)
    verify(signer, os.path.join(a.dist, "feed.json"))
    stable = load_feed(a.dist)["channels"]["stable"]
    installer = stable["installer"]
    version = stable["version"]
    if not installer.get("file"):
        raise RuntimeError("the feed of %s names no installer in the public link: publish the version with "
                           "publish_release.py" % a.dist)
    make_feed.check_installer(a.installer, version)
    if os.path.getsize(a.installer) != installer["size"] or make_feed.sha256_of(a.installer) != installer["sha256"]:
        raise RuntimeError("%s is not the installer the feed announces" % a.installer)
    os.makedirs(a.output, exist_ok=True)
    copies = [(a.installer, installer["file"]),
              (os.path.join(a.dist, "feed.json.sig"), LINK_FEED + ".sig"),
              (os.path.join(a.dist, "feed.json"), LINK_FEED)]
    for source, name in copies:
        target = os.path.join(a.output, name)
        partial = target + ".partial"
        shutil.copyfile(source, partial)
        os.replace(partial, target)
        print("%s: %d bytes" % (name, os.path.getsize(target)))
    print("version %s in %s" % (version, a.output))
    return 0


def check(a):
    files = public_link.Link(a.link).files()
    installer = load_feed(a.dist)["channels"]["stable"]["installer"]
    expected = [("dist/" + LINK_FEED, None), ("dist/" + LINK_FEED + ".sig", None)]
    if installer.get("file"):
        expected.append(("dist/" + installer["file"], installer["size"]))
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
    d.add_argument("--installer", required=True)
    d.add_argument("--sign")
    c = sub.add_parser("check")
    c.add_argument("--link", default=public_link.DEFAULT_LINK)
    for s in (d, c):
        s.add_argument("--dist", default=publish_release.DIST)
    a = p.parse_args()
    try:
        return {"dist": fill_dist, "check": check}[a.command](a)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
