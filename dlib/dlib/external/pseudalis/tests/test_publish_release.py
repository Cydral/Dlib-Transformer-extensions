# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Tests of tools/publish_release.py on one computer.  The chain of the website is a script that
records what it is asked and copies the files of the publication into a folder, which a local
server plays as the hosting; the installer is a program made here, that carries nothing but its
version resource.

    python3 test_publish_release.py <pseudalis-sign>
"""
import functools
import http.server
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
sys.path.insert(0, TOOLS)

import make_feed  # noqa: E402
import publish_release  # noqa: E402

FAILED = []


def check(condition, what):
    if not condition:
        FAILED.append(what)
        print("FAILED: %s" % what)


def program_with_version(version):
    """The bytes of a Windows program that holds nothing but the version resource of version."""
    major, minor, patch = (int(part) for part in version.split("."))
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, (major << 16) | minor, patch << 16,
                        (major << 16) | minor, patch << 16, 0x3F, 0, 0x40004, 1, 0, 0, 0)
    info = struct.pack("<HHH", 92, len(fixed), 0) + "VS_VERSION_INFO\0".encode("utf-16-le") + b"\0\0" + fixed
    rva, raw = 0x1000, 0x200

    def directory(ident, target):
        return struct.pack("<IIHHHH", 0, 0, 0, 0, 0, 1) + struct.pack("<II", ident, target)

    # The type, its name and its language, 24 bytes each, then the entry of the data and the data.
    section = (directory(16, 0x80000000 | 24) + directory(1, 0x80000000 | 48) + directory(0x409, 72) +
               struct.pack("<IIII", rva + 88, len(info), 0, 0) + info)
    section += b"\0" * (-len(section) % 0x200)
    pe = 0x80
    dos = b"MZ" + b"\0" * 0x3A + struct.pack("<I", pe) + b"\0" * (pe - 0x40)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x22)
    optional = bytearray(240)
    struct.pack_into("<H", optional, 0, 0x20B)
    struct.pack_into("<II", optional, 112 + 16, rva, len(section))
    table = b".rsrc\0\0\0" + struct.pack("<IIII", len(section), rva, len(section), raw) + b"\0" * 16
    head = dos + coff + bytes(optional) + table
    return head + b"\0" * (raw - len(head)) + section


SITE_SCRIPT = r"""#!/usr/bin/env bash
# Plays the chain of the website: records the command and the folder it runs in, asks a word on
# the terminal before it publishes, then copies the files of the publication into the hosting.
set -eu
printf '%s %s\n' "$1" "$PWD" >> calls.log
if [ -f "fail-$1" ]; then exit 3; fi
if [ "$1" = publish ]; then
  repo=$(sed -n 's/^PRODUCTS_REPO=//p' deploy.conf | sed 's/[[:space:]]\{1,\}#.*$//; s/^"//; s/"$//')
  IFS= read -r word || word=""
  printf 'asked %s\n' "$word" >> calls.log
  echo "website: publishing"
  mkdir -p hosting/files/pseudalis
  cp "$repo"/dist/pseudalis/feed.json "$repo"/dist/pseudalis/feed.json.sig \
    "$repo"/dist/pseudalis/pseudalis-setup-*.exe hosting/files/pseudalis/
fi
"""

BOOT = """
import sys
sys.path.insert(0, %(tools)r)
import publish_release as p
p.PUBLIC_KEY = %(public)r
p.published_address = lambda: %(address)r
sys.argv = ["publish_release.py"] + %(args)r
sys.exit(p.main())
"""


class hosting(http.server.SimpleHTTPRequestHandler):
    """The files of a folder, the requests recorded; an address under /moved answers a redirection."""
    seen = []

    def answer(self, body):
        hosting.seen.append((self.command, self.path, self.headers.get("User-Agent", "")))
        if self.path.startswith("/moved/"):
            self.send_response(301)
            self.send_header("Location", self.path[len("/moved"):])
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif body:
            http.server.SimpleHTTPRequestHandler.do_GET(self)
        else:
            http.server.SimpleHTTPRequestHandler.do_HEAD(self)

    def do_GET(self):
        self.answer(True)

    def do_HEAD(self):
        self.answer(False)

    def log_message(self, *args):
        pass


def read(path):
    with open(path, "rb") as f:
        return f.read()


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    signer = os.path.abspath(sys.argv[1])
    version = publish_release.source_version()
    work = tempfile.mkdtemp(prefix="pseudalis-publication-")
    dist = os.path.join(work, "repo", "dist", "pseudalis")
    site = os.path.join(work, "site")
    hosted = os.path.join(site, "hosting", "files", "pseudalis")
    calls = os.path.join(site, "calls.log")
    os.makedirs(dist)
    os.makedirs(os.path.join(site, "tools"))
    os.makedirs(os.path.join(site, "hosting"))
    write(os.path.join(site, "tools", "site.sh"), SITE_SCRIPT.encode("utf-8"))
    conf = os.path.join(site, "deploy.conf")
    good_conf = ('# Settings of the website.\nSITE_URL=https://example.org\nPRODUCTS_REPO="%s"   # the repository\n'
                 % os.path.join(work, "repo")).encode("utf-8")
    write(conf, good_conf)
    key = os.path.join(work, "keys", "feed.key")
    public = os.path.join(work, "feed-public-key.pem")
    os.makedirs(os.path.dirname(key))
    subprocess.run([signer, "keygen", key, public], check=True, stdout=subprocess.DEVNULL)
    installer = os.path.join(work, "pseudalis-setup.exe")
    write(installer, program_with_version(version))
    notes = os.path.join(work, "notes.json")
    write(notes, json.dumps({"en": "Notes of the test", "fr": "Notes du test"}).encode("utf-8"))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        hosting, directory=os.path.join(site, "hosting")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = "http://127.0.0.1:%d/files/pseudalis" % server.server_address[1]
    # The requests to this computer go straight to it, whatever proxy the system names.
    env = dict(os.environ, no_proxy="127.0.0.1", NO_PROXY="127.0.0.1")

    def run(args, terminal="", base=address):
        code = BOOT % {"tools": TOOLS, "public": public, "address": base, "args": args}
        r = subprocess.run([sys.executable, "-c", code], input=terminal, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, env=env)
        return r.returncode, r.stdout

    def called():
        if not os.path.isfile(calls):
            return []
        with open(calls, encoding="utf-8") as f:
            return f.read().splitlines()

    common = [installer, "--any-folder", "--key", key, "--sign", signer, "--dist", dist, "--site", site,
              "--notes", notes, "--wait", "1"]
    try:
        # The program made here is read as the installer of its version.
        check(make_feed.installer_version(installer) == version, "the version resource of the test installer is read")

        # The settings of the website, read as its chain reads them.
        sample = os.path.join(work, "sample.conf")
        write(sample, b'#PRODUCTS_REPO=/commented\n PRODUCTS_REPO = "/first"\nMODELS=default\n'
                      b'PRODUCTS_REPO=/a folder/of the repo   # the last one counts\n')
        check(publish_release.site_setting(sample, "PRODUCTS_REPO") == "/a folder/of the repo",
              "a setting keeps its spaces, without its comment, the last line counting")
        check(publish_release.site_setting(sample, "MODELS") == "default" and
              publish_release.site_setting(sample, "ALBIREO_SPACE") == "", "a setting without quotes, a missing one")

        # A dry run writes nothing and publishes nothing.
        code, out = run(common + ["--dry-run"])
        check(code == 0 and "dry run: nothing written, nothing published" in out, "the dry run ends well: %s" % out)
        check(os.listdir(dist) == [] and called() == [], "the dry run writes nothing and runs no chain")

        # The website must take its files in the copy written.
        write(conf, b"PRODUCTS_REPO=" + os.path.join(work, "elsewhere").encode("utf-8") + b"\n")
        code, out = run(common)
        check(code == 1 and "PRODUCTS_REPO" in out and "tools/site.sh setup --repo" in out,
              "another copy of the repository stops the publication: %s" % out)
        write(conf, b"MODELS=default\n")
        code, out = run(common)
        check(code == 1 and "names no copy of the repository" in out, "no copy named stops the publication")
        check(os.listdir(dist) == [] and called() == [], "a publication stopped there writes nothing")
        write(conf, good_conf)
        code, out = run(common[:8] + ["--site", os.path.join(work, "nowhere")] + common[10:])
        check(code == 1 and "no chain of the website" in out, "a folder without the chain stops the publication")

        # An installer of another version is refused.
        other = os.path.join(work, "other.exe")
        write(other, program_with_version("0.0.1"))
        code, out = run([other] + common[1:])
        check(code == 1 and "installs version 0.0.1" in out, "the installer of another version is refused: %s" % out)
        check(os.listdir(dist) == [], "a refused installer writes nothing")

        # The publication: the files written, the chain run in its folder, the hosting checked.
        write(os.path.join(dist, "pseudalis-setup-0.9.0.exe"), b"an earlier installer")
        hosting.seen = []
        code, out = run(common, terminal="word-of-the-terminal\n")
        check(code == 0 and "the website serves version %s" % version in out, "the publication ends well: %s" % out)
        real_site = os.path.realpath(site)
        check(called() == ["status " + real_site, "publish " + real_site, "asked word-of-the-terminal"],
              "the chain runs status, then publish, in its folder, the terminal its own: %s" % called())
        check("website: publishing" in out, "the chain writes on the terminal of the publisher")
        check("word-of-the-terminal" not in out, "the program shows nothing of what the chain is told")
        name = "pseudalis-setup-%s.exe" % version
        check(sorted(os.listdir(dist)) == ["feed.json", "feed.json.sig", name], "the folder holds the three files, "
              "the earlier installer gone: %s" % sorted(os.listdir(dist)))
        feed = read(os.path.join(dist, "feed.json"))
        stable = json.loads(feed.decode("utf-8"))["channels"]["stable"]
        check(b"\r" not in feed and feed.endswith(b"}\n"), "the feed has LF ends of line")
        check(stable["version"] == version and stable["notes"]["fr"] == "Notes du test" and
              stable["installer"] == {"file": name, "size": os.path.getsize(installer),
                                      "sha256": make_feed.sha256_of(installer)}, "the feed describes the installer")
        check(read(os.path.join(dist, name)) == read(installer), "the installer is copied as it is")
        check(subprocess.run([signer, "verify", os.path.join(dist, "feed.json"), public],
                             stdout=subprocess.DEVNULL).returncode == 0, "the signature of the feed verifies")
        check(all(read(os.path.join(hosted, f)) == read(os.path.join(dist, f)) for f in os.listdir(dist)),
              "the hosting holds the files of the folder")
        check(hosting.seen and all(agent == "Pseudalis/%s" % version for _, _, agent in hosting.seen),
              "the requests carry the name of the programs: %s" % hosting.seen)
        check(("HEAD", "/files/pseudalis/" + name, "Pseudalis/%s" % version) in hosting.seen,
              "the size the hosting announces is asked before the installer")

        # The check alone, then against a hosting that serves something else.
        code, out = run(["--check", "--dist", dist, "--wait", "1"])
        check(code == 0 and "the website serves version" in out, "the check alone ends well: %s" % out)
        saved = read(os.path.join(hosted, "feed.json"))
        write(os.path.join(hosted, "feed.json"), saved + b" ")
        code, out = run(["--check", "--dist", dist, "--wait", "1"])
        check(code == 1 and "does not serve the feed" in out, "another feed on the hosting is an error: %s" % out)
        write(os.path.join(hosted, "feed.json"), saved)
        saved = read(os.path.join(hosted, name))
        write(os.path.join(hosted, name), saved[:-1])
        code, out = run(["--check", "--dist", dist, "--wait", "1"])
        check(code == 1 and "announces an installer" in out, "another size on the hosting is an error: %s" % out)
        write(os.path.join(hosted, name), saved[:-1] + bytes([saved[-1] ^ 1]))
        code, out = run(["--check", "--dist", dist, "--wait", "1"])
        check(code == 1 and "serves another installer" in out, "another installer of the same size is an error")
        write(os.path.join(hosted, name), saved)
        os.remove(os.path.join(hosted, name))
        code, out = run(["--check", "--dist", dist, "--wait", "1"])
        check(code == 1 and "does not serve %s" % name in out, "a missing installer is an error: %s" % out)
        write(os.path.join(hosted, name), saved)
        code, out = run(["--check", "--dist", dist, "--wait", "1"], base=address.replace("/files/", "/moved/files/"))
        check(code == 1 and "redirects" in out, "a redirection is an error: %s" % out)

        # A chain that fails stops the publication, which says how it ends.
        os.remove(calls)
        write(os.path.join(site, "fail-publish"), b"")
        code, out = run(common)
        check(code == 1 and "tools/site.sh publish ended with code 3" in out and
              "the publication ends, once the cause is removed, with:" in out and "--check" in out,
              "a failure of the chain stops the publication: %s" % out)
        os.remove(os.path.join(site, "fail-publish"))
        os.remove(calls)
        write(os.path.join(site, "fail-status"), b"")
        code, out = run(common)
        check(code == 1 and called() == ["status " + real_site], "a failure of status runs no publish: %s" % called())
        os.remove(os.path.join(site, "fail-status"))

        # A feed that announces a later version is not replaced.
        saved = read(os.path.join(dist, "feed.json"))
        later = json.loads(saved.decode("utf-8"))
        later["channels"]["stable"]["version"] = "99.0.0"
        write(os.path.join(dist, "feed.json"), json.dumps(later).encode("utf-8"))
        code, out = run(common)
        check(code == 1 and "already announces 99.0.0" in out, "a later version announced stops the publication")
        write(os.path.join(dist, "feed.json"), saved)

        # The program has no part in the transfer: nothing in it names a session or a password file.
        with open(os.path.join(TOOLS, "publish_release.py"), encoding="utf-8") as f:
            source = f.read().lower()
        for word in ("ftplib", "paramiko", "getpass", "sftp", "lftp", "cydral_ftp", "cydral-ftp"):
            check(word not in source, "publish_release.py does not name %s" % word)
    finally:
        server.shutdown()
        shutil.rmtree(work, ignore_errors=True)
    if FAILED:
        print("test_publish_release: %d test(s) failed" % len(FAILED))
        return 1
    print("test_publish_release: all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
