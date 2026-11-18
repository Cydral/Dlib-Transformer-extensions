# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Tests of tools/publish_release.py and of tools/make_offline_bundle.py on one computer.  The chain
of the website is a script that records what it is asked and copies into a folder, which a local
server plays as the hosting, what it publishes of its folder of the publication: the signed feed,
the signed catalog and the installers of the version of the feed, with the list of these files.
The installers are programs made here, that carry nothing but their version resource, and the
signatures are those of a key pair made for the test.

    python3 test_publish_release.py <albireo-sign>
"""
import functools
import hashlib
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

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


def refusal(code, out, text):
    """Whether a run ended as a refusal: the code 1, a line that starts with "error: " and holds text,
    and no exception that escaped."""
    return code == 1 and "Traceback" not in out and any(
        line.startswith("error: ") and text in line for line in out.splitlines())


def program_with_version(version, tail=b""):
    """The bytes of a Windows program that holds nothing but the version resource of version,
    followed by tail."""
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
    return head + b"\0" * (raw - len(head)) + section + tail


SITE_SCRIPT = r"""#!/usr/bin/env bash
# Plays the chain of the website: records the command and the folder it runs in, asks a word on
# the terminal before it publishes, then copies into the hosting the signed files of its folder
# and the installers of the version of its feed, with the list of these files.  A file named
# stale-hosting leaves the hosting as it is, as a hosting that does not serve yet what it received.
set -eu
printf '%s %s\n' "$1" "$PWD" >> calls.log
if [ -f "fail-$1" ]; then exit 3; fi
if [ "$1" = publish ]; then
  IFS= read -r word || word=""
  printf 'asked %s\n' "$word" >> calls.log
  echo "website: publishing"
  if [ -f stale-hosting ]; then exit 0; fi
  mkdir -p hosting/files/albireo/dist
  python3 - <<'COPY'
import glob, hashlib, json, os, shutil
own = "releases/files/albireo/dist"
target = "hosting/files/albireo/dist"
with open(os.path.join(own, "feed.json"), encoding="utf-8") as f:
    version = json.load(f)["channels"]["stable"]["version"]
for name in ("feed.json", "feed.json.sig", "catalog.json", "catalog.json.sig"):
    shutil.copyfile(os.path.join(own, name), os.path.join(target, name))
# The installers of the version of the feed, and no other file of the folder.
details = {}
for old in glob.glob(os.path.join(target, "albireo-setup-*")):
    os.remove(old)
for pattern in ("albireo-setup-online-%s.exe", "albireo-setup-complete-%s.exe", "albireo-setup-complete-%s.models"):
    path = os.path.join(own, pattern % version)
    if os.path.isfile(path):
        shutil.copyfile(path, os.path.join(target, os.path.basename(path)))
        with open(path, "rb") as f:
            details["albireo/dist/" + os.path.basename(path)] = {"size": os.path.getsize(path),
                                                                "sha256": hashlib.sha256(f.read()).hexdigest()}
with open("hosting/files/manifest.json", "w", encoding="utf-8") as f:
    json.dump({"files": sorted(details), "details": details}, f)
COPY
fi
"""

BOOT = """
import sys
sys.path.insert(0, %(tools)r)
import publish_release as p
p.PUBLIC_KEY = %(public)r
p.CATALOG = %(catalog)r
p.NOTES = %(notes)r
p.POLL = 0.2
p.published_address = lambda: %(address)r
p.chain_copy = lambda: %(chain)r
%(patch)s
sys.argv = ["publish_release.py"] + %(args)r
sys.exit(p.main())
"""

BUNDLE_BOOT = """
import sys
sys.path.insert(0, %(tools)r)
import publish_release as p
p.PUBLIC_KEY = %(public)r
p.published_address = lambda: %(address)r
import make_offline_bundle
sys.argv = ["make_offline_bundle.py"] + %(args)r
sys.exit(make_offline_bundle.main())
"""

# Makes one file fail to take its name, and writes the names asked for, in their order, to a file.
FAILING_NAME = """
import os
real = os.replace
def replace(source, target):
    with open(%(order)r, "a", encoding="utf-8") as f:
        f.write(os.path.basename(target) + "\\n")
    if os.path.basename(target) == %(name)r:
        raise OSError(5, "Input/output error", target)
    return real(source, target)
os.replace = replace
"""


def attempt(function, *args):
    """The text of the error function raises with args, or an empty text when it raises none."""
    try:
        function(*args)
    except (RuntimeError, OSError) as e:
        return str(e) or type(e).__name__
    return ""


class hosting(http.server.SimpleHTTPRequestHandler):
    """The files of a folder, the requests recorded; an address under /moved answers a redirection."""
    seen = []

    def answer(self, body):
        hosting.seen.append((self.command, self.path, self.headers.get("User-Agent", ""),
                             self.headers.get("Accept-Encoding", "")))
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


def closed_port():
    """A port of this computer that no program listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    signer = os.path.abspath(sys.argv[1])
    version = publish_release.source_version()
    work = os.path.realpath(tempfile.mkdtemp(prefix="albireo-publication-"))
    repo = os.path.join(work, "repo")
    dist = os.path.join(repo, "dist", "albireo")
    notes_dir = os.path.join(repo, "notes")
    site = os.path.join(work, "site")
    # The folder of the publication, and the folder the local server plays as the hosting.
    folder = os.path.join(site, "releases", "files", "albireo", "dist")
    hosted = os.path.join(site, "hosting", "files", "albireo", "dist")
    calls = os.path.join(site, "calls.log")
    # The copy of the production chain on Windows, with the installers of a build.
    chain = os.path.join(work, "windows", "work", "Dlib-Transformer-extensions")
    build = os.path.join(chain, "build-albireo", "bin")
    for made in (dist, notes_dir, os.path.join(site, "tools"), os.path.join(site, "hosting"), build,
                 os.path.join(work, "keys")):
        os.makedirs(made)
    write(os.path.join(site, "tools", "site.sh"), SITE_SCRIPT.encode("utf-8"))
    key = os.path.join(work, "keys", "feed.key")
    public = os.path.join(work, "feed-public-key.pem")
    subprocess.run([signer, "keygen", key, public], check=True, stdout=subprocess.DEVNULL)
    other_key = os.path.join(work, "keys", "other.key")
    subprocess.run([signer, "keygen", other_key, os.path.join(work, "other-public-key.pem")], check=True,
                   stdout=subprocess.DEVNULL)
    online = os.path.join(build, "albireo-setup-online.exe")
    complete = os.path.join(build, "albireo-setup.exe")
    models = os.path.join(build, "albireo-setup.models")
    write(online, program_with_version(version, b"online"))
    write(complete, program_with_version(version, b"complete, with its models beside it"))
    write(models, b"the models of the complete installer" * 64)
    notes = os.path.join(work, "notes.json")
    write(notes, json.dumps({"en": "Notes of the test", "fr": "Notes du test"}).encode("utf-8"))
    # The signed catalog of the copy of the repository.
    catalog = os.path.join(dist, "catalog.json")

    def write_catalog(updated, models_of="[]", sign=True):
        write(catalog, ('{"format": 1, "updated": "%s", "models": %s}\n' % (updated, models_of)).encode("utf-8"))
        if sign:
            subprocess.run([signer, "sign", key, catalog], check=True, stdout=subprocess.DEVNULL)

    write_catalog("2026-11-17T00:00:00Z")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        hosting, directory=os.path.join(site, "hosting")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = "http://127.0.0.1:%d/files/albireo/dist" % server.server_address[1]
    # The requests to this computer go straight to it, whatever proxy the system names.
    env = dict(os.environ, no_proxy="127.0.0.1", NO_PROXY="127.0.0.1")

    def run(args, terminal="", base=address, patch=""):
        code = BOOT % {"tools": TOOLS, "public": public, "catalog": catalog, "notes": notes_dir, "address": base,
                       "chain": chain, "patch": patch, "args": args}
        r = subprocess.run([sys.executable, "-c", code], input=terminal, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, env=env)
        return r.returncode, r.stdout

    def bundle(args):
        code = BUNDLE_BOOT % {"tools": TOOLS, "public": public, "address": address, "args": args}
        r = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=env)
        return r.returncode, r.stdout

    def called():
        if not os.path.isfile(calls):
            return []
        with open(calls, encoding="utf-8") as f:
            return f.read().splitlines()

    def forget_calls():
        if os.path.isfile(calls):
            os.remove(calls)

    def held():
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    def snapshot():
        return {name: read(os.path.join(folder, name)) for name in held()
                if os.path.isfile(os.path.join(folder, name))}

    names = ["albireo-setup-complete-%s.exe" % version, "albireo-setup-complete-%s.models" % version,
             "albireo-setup-online-%s.exe" % version]
    signed = ["catalog.json", "catalog.json.sig", "feed.json", "feed.json.sig"]
    bare = ["--key", key, "--sign", signer, "--site", site, "--wait", "1"]
    common = bare + ["--notes", notes]
    check_only = ["--check", "--sign", signer, "--site", site, "--wait", "1"]
    catalog_only = ["--catalog", "--sign", signer, "--site", site, "--wait", "1"]
    nowhere = "http://127.0.0.1:%d/files/albireo/dist" % closed_port()
    try:
        # The programs made here are read as installers of their version.
        check(make_feed.installer_version(online) == version, "the version resource of the test installer is read")

        # The profile folder of Windows, from what its console prints, whatever its code page.
        check(publish_release.profile_path("C:\\Users\\Zo\u00e9 Martin\r\n".encode("utf-16-le"))
              == "C:\\Users\\Zo\u00e9 Martin", "a profile folder printed in UTF-16 is read with its accents")
        check(publish_release.profile_path(b"C:\\Users\\user\r\n") == "C:\\Users\\user",
              "a profile folder printed in bytes of one character each is read")
        check(publish_release.profile_path("%USERPROFILE%\r\n".encode("utf-16-le")) is None and
              publish_release.profile_path(b"C:\\Users\\Zo\x82\r\n") is None and
              publish_release.profile_path(b"") is None, "an output that holds no path of Windows gives none")

        # The same folder through a console and a path translator played by scripts: a console that prints
        # in UTF-16 when asked, one that ignores the request and prints the bytes of its code page, and one
        # that prints nothing when asked for UTF-16 and is then asked again without it.
        fakes = os.path.join(work, "fakes")
        os.makedirs(fakes)
        write(os.path.join(fakes, "wslpath"), b'#!/bin/bash\np="${2//\\\\//}"\nprintf \'/mnt/c%s\\n\' "${p#C:}"\n')
        os.chmod(os.path.join(fakes, "wslpath"), 0o755)
        consoles = (
            ("import sys\nsys.stdout.buffer.write('C:\\\\Users\\\\Zo\\u00e9\\r\\n'.encode('utf-16-le') "
             "if sys.argv[1] == '/u' else b'')\n", "/mnt/c/Users/Zo\u00e9"),
            ("import sys\nsys.stdout.buffer.write(b'C:\\\\Users\\\\user\\r\\n')\n", "/mnt/c/Users/user"),
            ("import sys\nsys.stdout.buffer.write(b'' if sys.argv[1] == '/u' else b'C:\\\\Users\\\\other\\r\\n')\n",
             "/mnt/c/Users/other"),
        )
        path = os.environ["PATH"]
        os.environ["PATH"] = fakes + os.pathsep + path
        try:
            for program, expected in consoles:
                write(os.path.join(fakes, "console.py"), program.encode("utf-8"))
                write(os.path.join(fakes, "cmd.exe"), ('#!/bin/bash\nexec "%s" "%s" "$@"\n' % (
                    sys.executable, os.path.join(fakes, "console.py"))).encode("utf-8"))
                os.chmod(os.path.join(fakes, "cmd.exe"), 0o755)
                check(publish_release.windows_profile() == expected,
                      "the profile folder of Windows is read through its console: %s" % expected)
        finally:
            os.environ["PATH"] = path

        # The address the programs read, from the sources, and what the signed files tell.
        base = publish_release.published_address()
        check(base.startswith("https://") and base.endswith("/files/albireo/dist") and "//www." not in base,
              "the folder of the versions of the sources is an address of the website: %s" % base)
        check(publish_release.catalog_time('{"updated": "2026-11-17T10:15:10Z"}') == "2026-11-17T10:15:10Z" and
              publish_release.catalog_time('{"updated": "17/11/2026"}') is None and
              publish_release.catalog_time('{"models": []}') is None and publish_release.catalog_time("<html>") is None,
              "the time of publication of a catalog is read in its one form")
        described = {"channels": {"stable": {"version": "1.2.3", "installer": {"file": "a.exe", "size": 5,
                                                                              "sha256": "0"}}}}
        check(publish_release.feed_installer(json.dumps(described), "there") == ("1.2.3", "a.exe", 5, "0"),
              "the installer a feed names is read")
        described["channels"]["stable"]["installer"]["file"] = 5
        for text in ("<html>", '{"channels": {"stable": {"version": "1.2.3"}}}', "[]", json.dumps(described)):
            try:
                publish_release.feed_installer(text, "there")
                told = False
            except RuntimeError as e:
                told = "there is not an update feed" in str(e)
            check(told, "a text that is no feed is told as such: %s" % text)

        # The folder of the website: its chain, then the folder where the chain takes what it publishes.
        code, out = run(common + ["--dry-run"])
        check(refusal(code, out, "has no folder releases/files") and "tools/site.sh setup makes it" in out,
              "a chain without its folder of the files stops the publication: %s" % out)
        os.makedirs(os.path.join(site, "releases", "files"))
        code, out = run(common[:4] + ["--site", os.path.join(work, "nowhere")] + common[6:])
        check(refusal(code, out, "no chain of the website"), "a folder without the chain stops the publication")

        # A dry run writes nothing and publishes nothing; the installers are those of the production chain.
        code, out = run(common + ["--dry-run"])
        check(code == 0 and "dry run: nothing written, nothing published" in out, "the dry run ends well: %s" % out)
        check(all(n in out for n in names + signed), "the dry run names the seven files of the folder: %s" % out)
        check("catalog of 0 models, published 2026-11-17T00:00:00Z" in out, "the dry run tells the catalog: %s" % out)
        check("release notes of %s: en, fr" % notes in out, "the dry run tells the notes and their file: %s" % out)
        check(not os.path.exists(folder) and called() == [] and sorted(os.listdir(dist)) == signed[:2],
              "the dry run writes nothing and runs no chain")

        # The release notes: those of the sources for the version by default, and texts by language.
        code, out = run(bare + ["--dry-run"])
        check(code == 0 and "note: no release notes for version %s" % version in out and
              "Albireo %s" % version in out, "a version without notes is told: %s" % out)
        of_version = os.path.join(notes_dir, version + ".json")
        write(of_version, json.dumps({"en": "Notes of the sources", "it": "Note"}).encode("utf-8"))
        code, out = run(bare + ["--dry-run"])
        check(code == 0 and "release notes of %s: en, it" % of_version in out,
              "the notes of the sources are those of the version: %s" % out)
        code, out = run(common + ["--dry-run"])
        check(code == 0 and "release notes of %s: en, fr" % notes in out, "--notes names other notes: %s" % out)
        for content, told in (({"fr": "Sans anglais"}, "English text"),
                              ({"en": "Text", "fr": None}, 'no text for "fr"'),
                              ({"en": 5}, 'no text for "en"'), ({"en": "  "}, 'no text for "en"')):
            write(of_version, json.dumps(content).encode("utf-8"))
            code, out = run(bare + ["--dry-run"])
            check(refusal(code, out, told), "notes that are no texts by language are refused: %s" % out)
        os.remove(of_version)
        code, out = run(bare + ["--notes", os.path.join(work, "none.json"), "--dry-run"])
        check(refusal(code, out, "No such file"), "a notes file that is not there is told: %s" % out)

        # An installer of another version, an installer outside the production chain, a key of another pair.
        other = os.path.join(build, "other.exe")
        write(other, program_with_version("0.0.1"))
        code, out = run([other] + common)
        check(refusal(code, out, "installs version 0.0.1"), "the installer of another version is refused: %s" % out)
        code, out = run([online, "--complete", other] + common)
        check(refusal(code, out, "installs version 0.0.1"), "a complete installer of another version is refused")
        os.remove(other)
        outside = os.path.join(work, "albireo-setup-online.exe")
        write(outside, read(online))
        code, out = run([outside] + common)
        check(refusal(code, out, "is not in the copy of the production chain"),
              "an installer of another folder is refused: %s" % out)
        code, out = run([outside, "--any-folder", "--complete", complete, "--dry-run"] + common)
        check(code == 0, "--any-folder takes an installer of another folder: %s" % out)
        outside_complete = os.path.join(work, "albireo-setup.exe")
        write(outside_complete, read(complete))
        code, out = run(["--complete", outside_complete, "--dry-run"] + common)
        check(refusal(code, out, "is not in the copy of the production chain"),
              "a complete installer of another folder, given alone, is refused: %s" % out)
        code, out = run(common[:1] + [other_key] + common[2:])
        check(refusal(code, out, "is not the key of the public key built into the programs"),
              "a key of another pair is refused: %s" % out)
        code, out = run(common[:1] + [os.path.join(work, "keys", "none.key")] + common[2:])
        check(refusal(code, out, "no private key at"), "a missing key is told: %s" % out)
        # A catalog whose signature fails, a catalog a checkout converted, a catalog of another form.
        saved = read(catalog)
        write(catalog, saved + b" ")
        code, out = run(common)
        check(refusal(code, out, "does not verify"), "a catalog whose signature fails stops the publication: %s" % out)
        write(catalog, saved.replace(b"\n", b"\r\n"))
        code, out = run(common)
        check(refusal(code, out, "CRLF ends of line"), "a catalog a checkout converted is told as such: %s" % out)
        write_catalog("17 November 2026")
        code, out = run(common)
        check(refusal(code, out, "gives no time of publication"), "a catalog without its time is refused: %s" % out)
        write_catalog("2026-11-17T00:00:00Z", '{"one": 1}')
        code, out = run(common)
        check(refusal(code, out, "lists no models"), "a catalog without its list of models is refused: %s" % out)
        write_catalog("2026-11-17T00:00:00Z")
        check(not os.path.exists(folder) and called() == [], "a refused publication writes nothing")

        # The deposit: the seven files in the folder of the publication, the chain left to the publisher.  The
        # folder holds the files of an earlier version, a download of the chain, a file of someone else, links
        # at a name the program gives and at names it writes under, and a file a run cut short left.
        os.makedirs(folder)
        write(os.path.join(folder, "albireo-setup-online-0.9.0.exe"), b"an earlier installer")
        write(os.path.join(folder, "albireo-setup-complete-0.9.0.models"), b"earlier models")
        write(os.path.join(folder, "albireo-setup-complete-0.9.0.models.partial"), b"a download of the chain")
        write(os.path.join(folder, ".albireo-setup-online-0.9.0.exe.partial"), b"left by a run cut short")
        write(os.path.join(folder, "feed.json"), b"a feed left there")
        write(os.path.join(folder, "notes.txt"), b"a file the program does not manage")
        victims = {}
        for name in (names[2], "." + names[1] + ".partial", ".feed.json.new", ".catalog.json.sig.partial"):
            victims[name] = os.path.join(work, "victim-" + name.strip("."))
            write(victims[name], b"a file outside the folder")
            os.symlink(victims[name], os.path.join(folder, name))
        kept = ["albireo-setup-complete-0.9.0.models.partial", "notes.txt"]
        code, out = run(common + ["--deposit"])
        check(code == 0 and "the publication ends with:" in out and "tools/site.sh status && tools/site.sh publish"
              in out and "--check" in out, "the files are deposited, and the rest is told: %s" % out)
        check(called() == [], "--deposit runs no chain")
        check(held() == sorted(names + signed + kept), "the folder of the publication holds the seven files of the "
              "version, those of the other versions and of the run cut short gone, the files of others kept: %s"
              % held())
        check("installers that leave the folder once the version is in place: albireo-setup-complete-0.9.0.models, "
              "albireo-setup-online-0.9.0.exe" in out and "removed: albireo-setup-online-0.9.0.exe" in out,
              "the installers of the earlier version are told before and after they leave: %s" % out)
        check(read(os.path.join(folder, names[2])) == read(online) and read(os.path.join(folder, names[0]))
              == read(complete) and read(os.path.join(folder, names[1])) == read(models),
              "the installers are copied as they are")
        check(not os.path.islink(os.path.join(folder, names[2])) and
              all(read(victim) == b"a file outside the folder" for victim in victims.values()),
              "a link at the name of a file, hidden or not, is replaced, the file it led to unchanged")
        check(read(os.path.join(folder, "catalog.json")) == read(catalog) and
              read(os.path.join(folder, "catalog.json.sig")) == read(catalog + ".sig"),
              "the catalog of the repository and its signature are copied as they are")
        check(sorted(os.listdir(dist)) == signed[:2], "nothing is written in the copy of the repository: %s"
              % sorted(os.listdir(dist)))
        feed = read(os.path.join(folder, "feed.json"))
        stable = json.loads(feed.decode("utf-8"))["channels"]["stable"]
        check(b"\r" not in feed and feed.endswith(b"}\n"), "the feed has LF ends of line")
        check(stable["version"] == version and stable["notes"]["fr"] == "Notes du test" and
              stable["installer"] == {"file": names[2], "size": os.path.getsize(online),
                                      "sha256": make_feed.sha256_of(online)}, "the feed describes the installer")
        check(subprocess.run([signer, "verify", os.path.join(folder, "feed.json"), public],
                             stdout=subprocess.DEVNULL).returncode == 0, "the signature of the feed verifies")
        # Without --notes and without notes of the sources, the notes of the feed of the folder stay.
        code, out = run(bare + ["--dry-run"])
        check(code == 0 and "those the feed of the folder already gives version %s (en, fr)" % version in out,
              "the notes the feed already gives the version are kept: %s" % out)

        # The publication: the copies in place kept, the chain run in its folder, the hosting checked.
        hosting.seen = []
        code, out = run(common, terminal="word-of-the-terminal\n")
        check(code == 0 and "the website serves version %s" % version in out and "which verify" in out and
              "the hash of 3 of these 3 installers" in out, "the publication ends well: %s" % out)
        check(all("in place: %s" % n in out for n in names) and "in place: catalog.json, catalog.json.sig" in out,
              "the copies already there are kept: %s" % out)
        real_site = os.path.realpath(site)
        check(called() == ["status " + real_site, "publish " + real_site, "asked word-of-the-terminal"],
              "the chain runs status, then publish, in its folder, the terminal its own: %s" % called())
        check("website: publishing" in out, "the chain writes on the terminal of the publisher")
        check("word-of-the-terminal" not in out, "the program shows nothing of what the chain is told")
        check(all(read(os.path.join(hosted, f)) == read(os.path.join(folder, f)) for f in signed + names),
              "the hosting holds the files of the folder of the publication")
        check(hosting.seen and all(agent == "Albireo/%s" % version and encoding == "identity"
                                   for _, _, agent, encoding in hosting.seen),
              "the requests carry the name of the programs and ask for no encoding: %s" % hosting.seen[:3])
        for n in names:
            check(("HEAD", "/files/albireo/dist/" + n, "Albireo/%s" % version, "identity") in hosting.seen,
                  "the size the hosting announces for %s is asked" % n)
        check(not any(command == "GET" and path.endswith((".exe", ".models")) for command, path, _, _
                      in hosting.seen), "no installer is downloaded to be checked")

        # The check alone, then against a hosting that serves something else.
        code, out = run(check_only)
        check(code == 0 and "the website serves version" in out, "the check alone ends well: %s" % out)
        started = time.time()
        for name, told in (("feed.json", "another feed of version %s" % version),
                           ("catalog.json.sig", "another catalog.json.sig")):
            saved = read(os.path.join(hosted, name))
            write(os.path.join(hosted, name), saved + b" ")
            code, out = run(check_only)
            check(refusal(code, out, "does not serve the signed files") and told in out,
                  "another %s on the hosting is an error: %s" % (name, out))
            write(os.path.join(hosted, name), saved)
        check(time.time() - started < 8, "the wait for the website ends with the time it is given: %.1f s for two "
              "waits of one second" % (time.time() - started))
        for name in names:
            saved = read(os.path.join(hosted, name))
            write(os.path.join(hosted, name), saved[:-1])
            code, out = run(check_only)
            check(refusal(code, out, "announces %s at" % name), "another size of %s is an error: %s" % (name, out))
            write(os.path.join(hosted, name), saved)
        manifest = os.path.join(site, "hosting", "files", "manifest.json")
        saved = read(manifest)
        for source, name in ((online, names[2]), (complete, names[0]), (models, names[1])):
            write(manifest, saved.replace(hashlib.sha256(read(source)).hexdigest().encode("ascii"), b"0" * 64))
            code, out = run(check_only)
            check(refusal(code, out, "holds another %s than the one of the folder" % name),
                  "another %s of the same size is an error: %s" % (name, out))
        os.remove(manifest)
        code, out = run(check_only)
        check(code == 0 and "the hash of 0 of these 3 installers" in out,
              "a website without its list of files is checked by the sizes: %s" % out)
        write(manifest, saved)
        os.remove(os.path.join(hosted, names[2]))
        code, out = run(check_only)
        check(refusal(code, out, "does not serve %s" % names[2]), "a missing installer is an error: %s" % out)
        write(os.path.join(hosted, names[2]), read(online))
        code, out = run(check_only, base=address.replace("/files/", "/moved/files/"))
        check(refusal(code, out, "redirects"), "a redirection is an error: %s" % out)
        code, out = run(["--check", "--site", os.path.join(work, "nowhere")])
        check(refusal(code, out, "no chain of the website"), "the check alone needs the folder of the website")
        code, out = run(check_only + ["--deposit"])
        check(refusal(code, out, "--check takes no --deposit"), "the check alone takes no other mode: %s" % out)
        # A feed of a later version on the hosting is told at once, and a hosting that serves the feed after
        # a moment is waited for.
        hosted_feed = read(os.path.join(hosted, "feed.json"))
        later = json.loads(hosted_feed.decode("utf-8"))
        later["channels"]["stable"]["version"] = "99.0.0"
        write(os.path.join(hosted, "feed.json"), json.dumps(later).encode("utf-8"))
        started = time.time()
        code, out = run(check_only[:-1] + ["60"])
        check(refusal(code, out, "the website serves version 99.0.0, later than") and "take online" in out and
              time.time() - started < 30, "a later version on the hosting ends the check at once: %s" % out)
        os.remove(os.path.join(hosted, "feed.json"))
        back = threading.Timer(1.5, write, (os.path.join(hosted, "feed.json"), hosted_feed))
        back.start()
        code, out = run(check_only[:-1] + ["30"])
        back.join()
        check(code == 0 and "the website serves version" in out, "a feed the hosting serves after a moment is "
              "waited for: %s" % out)
        # A signature that signs nothing, in the folder and on the hosting alike.
        true_signature = read(os.path.join(folder, "feed.json.sig"))
        for where in (folder, hosted):
            write(os.path.join(where, "feed.json.sig"), read(os.path.join(folder, "catalog.json.sig")))
        code, out = run(check_only)
        check(refusal(code, out, "feed.json does not verify"), "a feed whose signature fails is told by the check "
              "alone: %s" % out)
        for where in (folder, hosted):
            write(os.path.join(where, "feed.json.sig"), true_signature)

        # A link at the name of an installer that leads to the same bytes is replaced by a copy: the folder
        # holds files, which a later build cannot change under their names.
        os.remove(os.path.join(folder, names[0]))
        os.symlink(complete, os.path.join(folder, names[0]))
        code, out = run(common + ["--deposit"])
        check(code == 0 and not os.path.islink(os.path.join(folder, names[0])) and
              read(os.path.join(folder, names[0])) == read(complete) and "placed: %s" % names[0] in out,
              "a link to the same installer is replaced by a copy: %s" % out)

        # A build without the file of the models: the one of an earlier publication leaves the folder.
        os.rename(models, models + ".aside")
        code, out = run(common + ["--deposit"])
        check(code == 0 and held() == sorted([names[0], names[2]] + signed + kept),
              "the file of the models leaves with a build that has none: %s" % held())
        os.rename(models + ".aside", models)

        # A chain that fails stops the publication, which says how it ends.
        forget_calls()
        write(os.path.join(site, "fail-publish"), b"")
        code, out = run(common)
        check(refusal(code, out, "tools/site.sh publish ended with code 3") and
              "the publication ends, once the cause is removed, with: (cd " in out and "tools/site.sh publish) && "
              in out and "--check" in out, "a failure of the chain stops the publication: %s" % out)
        os.remove(os.path.join(site, "fail-publish"))
        forget_calls()
        write(os.path.join(site, "fail-status"), b"")
        code, out = run(common)
        check(code == 1 and called() == ["status " + real_site], "a failure of status runs no publish: %s" % called())
        os.remove(os.path.join(site, "fail-status"))
        # A hosting that does not serve yet what the chain sent: the check alone is left to run.
        write(online, program_with_version(version, b"online, built again"))
        write(os.path.join(site, "stale-hosting"), b"")
        code, out = run(common, terminal="word\n")
        ends = [line for line in out.splitlines() if line.startswith("the publication ends")]
        check(refusal(code, out, "does not serve the signed files") and len(ends) == 1 and
              "tools/site.sh" not in ends[0] and "--check" in ends[0],
              "a publication the hosting does not serve yet ends with the check alone: %s" % out)
        os.remove(os.path.join(site, "stale-hosting"))
        forget_calls()
        code, out = run(common, terminal="word\n")
        check(code == 0 and read(os.path.join(hosted, names[2])) == read(online),
              "the publication ends once the chain runs again: %s" % out)

        # A feed that announces a later version is not replaced, in the folder or on the website.
        before = snapshot()
        write(os.path.join(folder, "feed.json"), json.dumps(later).encode("utf-8"))
        forget_calls()
        code, out = run(common)
        check(refusal(code, out, "already holds the feed of version 99.0.0, later than %s" % version),
              "a later version in the folder of the publication stops the publication: %s" % out)
        check(called() == [] and read(os.path.join(folder, "feed.json")) == json.dumps(later).encode("utf-8"),
              "a publication stopped there changes nothing")
        write(os.path.join(folder, "feed.json"), before["feed.json"])
        hosted_feed = read(os.path.join(hosted, "feed.json"))
        write(os.path.join(hosted, "feed.json"), json.dumps(later).encode("utf-8"))
        code, out = run(common)
        check(refusal(code, out, "the website already serves version 99.0.0"),
              "a later version on the website stops the publication: %s" % out)
        write(os.path.join(hosted, "feed.json"), hosted_feed)

        # A catalog published after the one of the repository, in the folder or on the website: the copy
        # of the repository is behind, for a version as for the catalog alone.
        newer = b'{"format": 1, "updated": "2027-01-01T00:00:00Z", "models": []}\n'
        hosted_catalog = read(os.path.join(hosted, "catalog.json"))
        for where, place, told in ((folder, before["catalog.json"], "%s holds a catalog" % folder),
                                   (hosted, hosted_catalog, "the website holds a catalog")):
            write(os.path.join(where, "catalog.json"), newer)
            for args in (common, catalog_only):
                code, out = run(args)
                check(refusal(code, out, told + " published 2027-01-01T00:00:00Z, after the one of this copy"),
                      "a later catalog stops the publication (%s): %s" % (" ".join(args[:1]), out))
            write(os.path.join(where, "catalog.json"), place)
        check(snapshot() == before and called() == [], "the publications stopped there change nothing")

        # An answer of the website that is no feed, as a portal or a filter of a network gives: the version it
        # serves is unknown.
        write(os.path.join(hosted, "feed.json"), b"<html><body>Sign in to the network</body></html>")
        code, out = run(common)
        check(refusal(code, out, "is answered with something else than an update feed") and "--deposit" in out,
              "a page in place of the feed stops the publication: %s" % out)
        code, out = run(catalog_only)
        check(refusal(code, out, "is answered with something else than an update feed"),
              "and the one of the catalog: %s" % out)
        write(os.path.join(hosted, "feed.json"), hosted_feed)
        write(os.path.join(hosted, "catalog.json"), b"<html><body>Sign in to the network</body></html>")
        code, out = run(common)
        check(refusal(code, out, "is answered with something else than a catalog"),
              "a page in place of the catalog stops the publication: %s" % out)
        write(os.path.join(hosted, "catalog.json"), hosted_catalog)
        check(snapshot() == before and called() == [], "nothing is written then")

        # A failure while the files are written leaves the folder as it was: the installers of another build
        # are staged, then the signature of the feed fails, or is the one of another key.
        failing = os.path.join(work, "failing-sign")
        write(failing, ('#!/bin/bash\nif [ "$1" = sign ] && [[ "$3" == */.feed.json.new ]]; then exit 5; fi\n'
                        'exec "%s" "$@"\n' % signer).encode("utf-8"))
        wrong = os.path.join(work, "wrong-sign")
        write(wrong, ('#!/bin/bash\nif [ "$1" = sign ] && [[ "$3" == */.feed.json.new ]]; then '
                      'exec "%s" sign "%s" "$3"; fi\nexec "%s" "$@"\n' % (signer, other_key, signer)).encode("utf-8"))
        for program in (failing, wrong):
            os.chmod(program, 0o755)
        write(online, program_with_version(version, b"online, built a third time"))
        for program in (failing, wrong):
            code, out = run(common[:2] + ["--sign", program] + common[4:])
            check(code == 1 and "Traceback" not in out and "error: " in out and snapshot() == before and
                  held() == sorted(before), "a failure before the names are taken leaves the folder as it was "
                  "(%s): %s, %s" % (os.path.basename(program), out, held()))
        check(called() == [], "and runs no chain")

        # A failure at the first name leaves the folder as it was, and is told as it is.
        order = os.path.join(work, "order.txt")
        code, out = run(common, patch=FAILING_NAME % {"order": order, "name": names[2]})
        check(refusal(code, out, "Input/output error") and "now holds files" not in out and snapshot() == before
              and held() == sorted(before) and called() == [],
              "a failure at the first name leaves the folder as it was: %s" % out)
        os.remove(order)

        # A failure once the first name is taken: the installers are named first, the feed last, the folder
        # then holds a part of the version, which the program tells, and the same command completes it.
        write(os.path.join(folder, "albireo-setup-online-0.9.0.exe"), b"an earlier installer")
        code, out = run(common, patch=FAILING_NAME % {"order": order, "name": "feed.json"})
        check(refusal(code, out, "now holds files of the version that its feed does not describe yet") and
              "the same command completes it" in out, "a failure while the names are taken is told with what the "
              "folder holds: %s" % out)
        taken = read(order).decode("utf-8").split()
        check(taken == [names[2], "feed.json.sig", "feed.json"], "the installer takes its name first, then the "
              "signature of the feed, then the feed: %s" % taken)
        check(called() == [] and "placed: %s" % names[2] in out and
              read(os.path.join(folder, names[2])) == read(online) and
              read(os.path.join(folder, "feed.json")) == before["feed.json"] and
              read(os.path.join(folder, "feed.json.sig")) == before["feed.json.sig"] and
              "albireo-setup-online-0.9.0.exe" in held() and not any(n.startswith(".") for n in held()),
              "the new installer lies under the former feed and its signature, the earlier installer still "
              "there, and no chain ran: %s" % held())
        code, out = run(common, terminal="word\n")
        check(code == 0 and "in place: %s" % names[2] in out and "removed: albireo-setup-online-0.9.0.exe" in out
              and read(os.path.join(hosted, names[2])) == read(online) and held() == sorted(names + signed + kept),
              "the same command completes the version: %s" % out)

        # Another build of the same size is another file: copied, and published.
        again = program_with_version(version, b"online, built a third TIME")
        check(len(again) == len(read(online)), "the two builds have one size")
        write(online, again)
        before = snapshot()
        code, out = run(common, terminal="word\n")
        check(code == 0 and "placed: %s" % names[2] in out and "in place: %s" % names[0] in out and
              read(os.path.join(folder, names[2])) == again and read(os.path.join(hosted, names[2])) == again and
              read(os.path.join(folder, "feed.json")) != before["feed.json"] and
              read(os.path.join(hosted, "feed.json")) == read(os.path.join(folder, "feed.json")),
              "another build of the same size is copied and published with its feed: %s" % out)

        # The catalog alone, between two versions.
        former = read(catalog)
        write_catalog("2026-11-18T00:00:00Z", '[{"id": "one"}]', sign=False)
        code, out = run(catalog_only)
        check(refusal(code, out, "does not verify"),
              "a catalog written again without its signature is not published: %s" % out)
        write_catalog("2026-11-18T00:00:00Z", '[{"id": "one"}]')
        forget_calls()
        code, out = run(catalog_only + ["--dry-run"])
        check(code == 0 and "dry run: nothing written" in out and called() == [] and
              read(os.path.join(folder, "catalog.json")) == former,
              "the dry run of the catalog tells what it would do, and does nothing: %s" % out)
        code, out = run(catalog_only + ["--deposit"])
        check(code == 0 and called() == [] and "the publication ends with:" in out and
              read(os.path.join(folder, "catalog.json")) == read(catalog) and
              read(os.path.join(folder, "catalog.json.sig")) == read(catalog + ".sig"),
              "--deposit puts the catalog in the folder and leaves the chain to the publisher: %s" % out)
        check(read(os.path.join(hosted, "catalog.json")) == former, "the hosting still serves the former catalog")
        feed_before = read(os.path.join(folder, "feed.json"))
        code, out = run(catalog_only, terminal="word\n")
        check(code == 0 and "catalog of 1 models, published 2026-11-18T00:00:00Z, for version %s" % version in out
              and "the website serves version %s" % version in out and
              "in place: catalog.json, catalog.json.sig" in out, "the catalog alone is published: %s" % out)
        check(called() == ["status " + real_site, "publish " + real_site, "asked word"],
              "the chain runs status, then publish: %s" % called())
        check(read(os.path.join(hosted, "catalog.json")) == read(catalog) and
              read(os.path.join(hosted, "catalog.json.sig")) == read(catalog + ".sig") and
              read(os.path.join(hosted, "feed.json")) == feed_before and
              read(os.path.join(folder, "feed.json")) == feed_before,
              "the hosting serves the new catalog and the feed of the version, left as it was")
        # What the publication of the catalog refuses: an installer, notes, a key, another version on the
        # website, no feed there.
        for extra, told in (([online], "installer"), (["--notes", notes], "--notes"), (["--key", key], "--key")):
            code, out = run(extra + catalog_only)
            check(refusal(code, out, "--catalog takes no %s" % told), "--catalog takes no %s: %s" % (told, out))
        hosted_feed = read(os.path.join(hosted, "feed.json"))
        write(os.path.join(hosted, "feed.json"), json.dumps(later).encode("utf-8"))
        code, out = run(catalog_only)
        check(refusal(code, out, "the website serves version 99.0.0, later than") and "take online" in out,
              "a later version on the website stops the publication of the catalog: %s" % out)
        earlier = json.loads(hosted_feed.decode("utf-8"))
        earlier["channels"]["stable"]["version"] = "0.9.0"
        write(os.path.join(hosted, "feed.json"), json.dumps(earlier).encode("utf-8"))
        code, out = run(catalog_only)
        check(refusal(code, out, "the website serves version 0.9.0 and") and "publish that version" in out,
              "a version the website does not serve yet is published first: %s" % out)
        os.remove(os.path.join(hosted, "feed.json"))
        code, out = run(catalog_only)
        check(refusal(code, out, "the website serves no feed"), "a website without feed has its version first: %s"
              % out)
        write(os.path.join(hosted, "feed.json"), hosted_feed)
        # A folder whose feed is not the one the website serves, another build of the version deposited there
        # for instance: the chain would publish that feed and its installers with the catalog.
        write_catalog("2026-11-18T12:00:00Z", '[{"id": "one"}]')
        write(online, program_with_version(version, b"online, built a fifth time"))
        code, out = run(common + ["--deposit"])
        check(code == 0, "another build is deposited: %s" % out)
        forget_calls()
        deposited = snapshot()
        code, out = run(catalog_only)
        check(refusal(code, out, "is not the one the website serves for version %s" % version) and
              "take online" in out and called() == [] and snapshot() == deposited and
              read(os.path.join(hosted, names[2])) == again,
              "a folder whose feed is another one than the website serves publishes no catalog: %s" % out)
        code, out = run(common, terminal="word\n")
        check(code == 0, "the version deposited is published: %s" % out)
        # A website that cannot be read stops the publication of the catalog, and a deposit goes on.
        write_catalog("2026-11-18T18:00:00Z", '[{"id": "one"}]')
        forget_calls()
        code, out = run(catalog_only, base=nowhere)
        check(refusal(code, out, "cannot be read") and "--deposit" in out and "serves no feed" not in out,
              "a website that does not answer stops the publication of the catalog with its cause: %s" % out)
        code, out = run(catalog_only + ["--deposit"], base=nowhere)
        check(code == 0 and "was not compared" in out and called() == [],
              "the deposit of the catalog goes on without the website, and says so: %s" % out)
        # A computer that did not publish the version lacks its installer: the chain would stop at its upload.
        os.rename(os.path.join(folder, names[2]), os.path.join(work, "aside.exe"))
        write_catalog("2026-11-18T19:00:00Z", '[{"id": "one"}]')
        before = snapshot()
        code, out = run(catalog_only)
        check(refusal(code, out, "holds no %s, which its feed names" % names[2]) and "take online" in out and
              called() == [] and snapshot() == before,
              "a folder of the publication without the installer of the version publishes no catalog: %s" % out)
        code, out = run(catalog_only + ["--dry-run"])
        check(code == 0 and "note: " in out and "holds no %s" % names[2] in out,
              "the dry run tells the missing installer in a note: %s" % out)
        os.rename(os.path.join(work, "aside.exe"), os.path.join(folder, names[2]))
        os.rename(os.path.join(folder, "feed.json.sig"), os.path.join(work, "aside.sig"))
        code, out = run(catalog_only)
        check(refusal(code, out, "holds no signed feed") and "tools/site.sh take online" in out,
              "a folder without signed feed publishes no catalog: %s" % out)
        write(os.path.join(folder, "feed.json.sig"), b"another signature")
        code, out = run(catalog_only)
        check(refusal(code, out, "feed.json does not verify"),
              "a feed of the folder whose signature fails publishes no catalog: %s" % out)
        os.remove(os.path.join(folder, "feed.json.sig"))
        os.rename(os.path.join(work, "aside.sig"), os.path.join(folder, "feed.json.sig"))
        code, out = run(catalog_only, terminal="word\n")
        check(code == 0, "the catalog is published once the folder is whole: %s" % out)

        # The catalog of the repository that Git does not hold yet is told, with the commands that keep it.
        if shutil.which("git"):
            quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            git = ["git", "-C", repo, "-c", "user.name=Test", "-c", "user.email=test@example.org",
                   "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
            subprocess.run(["git", "init", "-q", repo], check=True, **quiet)
            subprocess.run(git + ["add", "-A"], check=True, **quiet)
            subprocess.run(git + ["commit", "-q", "-m", "The catalog"], check=True, **quiet)
            code, out = run(catalog_only + ["--deposit"])
            check(code == 0 and "Git does not hold" not in out, "a catalog that Git holds is told nothing: %s" % out)
            write_catalog("2026-11-19T00:00:00Z", '[{"id": "one"}, {"id": "two"}]')
            code, out = run(catalog_only + ["--deposit"])
            last = out.strip().splitlines()[-1]
            check(code == 0 and "Git does not hold the catalog published yet" in last and
                  "add catalog.json catalog.json.sig" in last and "Revise the catalog of Albireo" in last,
                  "a catalog that Git does not hold yet is told last, with the commands that keep it: %s" % out)
        else:
            print("skip the note on a catalog that Git does not hold: no git here")
        code, out = run(catalog_only, terminal="word\n")
        check(code == 0, "the last catalog is published: %s" % out)

        # A website that cannot be read: the publication of a version stops before it writes, a deposit goes on.
        before = snapshot()
        forget_calls()
        code, out = run(common, base=nowhere)
        check(refusal(code, out, "the version the website serves is unknown") and "--deposit" in out and
              snapshot() == before and called() == [],
              "a website that cannot be read stops the publication of a version: %s" % out)
        code, out = run(common + ["--dry-run"], base=nowhere)
        check(code == 0 and "was not compared" in out, "the dry run tells a website it could not read: %s" % out)
        code, out = run(common + ["--deposit"], base=nowhere)
        check(code == 0 and "was not compared" in out and called() == [],
              "--deposit goes on without the website, and says so: %s" % out)

        # The offline bundle: the signed files and the installer the website serves, or those of a folder.
        made = os.path.join(work, "bundle")
        hosting.seen = []
        code, out = bundle([made, "--sign", signer])
        check(code == 0 and sorted(os.listdir(made)) == sorted(signed + [names[2], "README.txt", "models"]) and
              all(read(os.path.join(made, f)) == read(os.path.join(hosted, f)) for f in signed + [names[2]]),
              "the bundle holds the signed files and the installer the website serves: %s, %s"
              % (out, sorted(os.listdir(made)) if os.path.isdir(made) else None))
        shutil.rmtree(made, ignore_errors=True)
        hosting.seen = []
        code, out = bundle([made, "--sign", signer, "--dist", folder])
        check(code == 0 and all(read(os.path.join(made, f)) == read(os.path.join(folder, f))
                                for f in signed + [names[2]]) and hosting.seen == [],
              "the bundle of a folder takes its signed files and its installer there, the website not asked: %s, "
              "%s" % (out, hosting.seen))
        shutil.rmtree(made, ignore_errors=True)
        os.rename(os.path.join(hosted, names[2]), os.path.join(work, "aside.exe"))
        code, out = bundle([made, "--sign", signer])
        check(refusal(code, out, "/files/albireo/dist/%s cannot be downloaded" % names[2]) and "--installer" in out
              and not any(os.path.exists(os.path.join(made, f)) for f in signed),
              "a website that lacks the installer is told with its address, the bundle left without signed "
              "files: %s" % out)
        os.rename(os.path.join(work, "aside.exe"), os.path.join(hosted, names[2]))
        shutil.rmtree(made, ignore_errors=True)
        code, out = bundle([made, "--sign", signer, "--dist", dist])
        check(refusal(code, out, "holds no feed.json") and "--dist names a folder" in out,
              "a folder without the signed feed makes no bundle: %s" % out)

        # The folder of the publication under a name that holds the characters of a pattern: its files alone
        # leave.
        marked = os.path.join(work, "site[1]", "dist")
        sibling = os.path.join(work, "site1", "dist")
        for made in (marked, sibling):
            os.makedirs(made)
            write(os.path.join(made, "albireo-setup-online-0.9.0.exe"), b"an earlier installer")
        publish_release.PUBLIC_KEY = public
        publish_release.CATALOG = catalog
        try:
            feed = json.loads(read(os.path.join(folder, "feed.json")).decode("utf-8"))
            publish_release.write_publication(marked, [(online, names[2], None)], feed, signer, key)
            check(sorted(os.listdir(marked)) == sorted([names[2]] + signed) and
                  os.listdir(sibling) == ["albireo-setup-online-0.9.0.exe"],
                  "the earlier files leave the folder of the publication, and no other folder: %s, %s"
                  % (sorted(os.listdir(marked)), os.listdir(sibling)))

            # A folder at the name of a file stops the publication before anything is written, and a folder
            # under the name of an earlier installer is left aside.
            os.makedirs(os.path.join(marked, names[0]))
            os.makedirs(os.path.join(marked, "albireo-setup-online-0.8.0.exe"))
            listing = sorted(os.listdir(marked))
            told = attempt(publish_release.write_publication, marked,
                           [(online, names[2], None), (complete, names[0], None)], feed, signer, key)
            check("is a folder: the file of that name cannot take its place" in told and
                  sorted(os.listdir(marked)) == listing,
                  "a folder at the name of an installer stops the publication, nothing written: %s" % told)
            os.rmdir(os.path.join(marked, names[0]))
            told = attempt(publish_release.write_publication, marked, [(online, names[2], None)], feed, signer, key)
            check(not told and os.path.isdir(os.path.join(marked, "albireo-setup-online-0.8.0.exe")),
                  "a folder under the name of an earlier installer stays: %s" % told)

            # A copy that differs from its source is not named.
            copied = publish_release.copy_hashed

            def damaged(source, target):
                result = copied(source, target)
                with open(target, "r+b") as f:
                    f.write(b"X")
                return result

            publish_release.copy_hashed = damaged
            try:
                told = attempt(publish_release.stage, complete, marked, names[0])
            finally:
                publish_release.copy_hashed = copied
            check("differs from it" in told and not os.path.exists(os.path.join(marked, names[0])),
                  "a copy that differs from its source is refused: %s" % told)
            # The SHA-256 the caller gives is the one of the bytes read: a source that changed since is refused.
            told = attempt(publish_release.stage, complete, marked, names[0], "0" * 64)
            check("differs from it" in told, "a source that is no longer the file described is refused: %s" % told)

            # A signed file that cannot take its place leaves the former signature beside the former file.
            blocked = os.path.join(work, "blocked")
            os.makedirs(os.path.join(blocked, "feed.json"))
            write(os.path.join(blocked, "feed.json.sig"), b"the former signature")
            write(os.path.join(blocked, ".new"), b"a feed")
            write(os.path.join(blocked, ".new.sig"), b"its signature")
            told = attempt(publish_release.put_pair, os.path.join(blocked, ".new"),
                           os.path.join(blocked, ".new.sig"), os.path.join(blocked, "feed.json"))
            check(told and read(os.path.join(blocked, "feed.json.sig")) == b"the former signature",
                  "a signed file that cannot be put in place keeps the former signature: %s" % told)
        finally:
            publish_release.PUBLIC_KEY = None

        # The program has no part in the transfer: nothing in it names a session or a password file, and it
        # reads no setting of the website.
        with open(os.path.join(TOOLS, "publish_release.py"), encoding="utf-8") as f:
            source = f.read().lower()
        for word in ("ftplib", "paramiko", "getpass", "sftp", "lftp", "cydral_ftp", "cydral-ftp", "deploy.conf",
                     "products_repo"):
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
