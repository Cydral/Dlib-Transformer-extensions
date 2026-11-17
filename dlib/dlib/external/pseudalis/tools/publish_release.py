# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
r"""
Publishes a version of Pseudalis on the website: puts its installer in dist/pseudalis of this
copy of the repository, beside the update feed that announces it, written and signed, has the
chain of the website publish the three files, then checks what the website serves.

    python3 publish_release.py [<pseudalis-setup.exe>] --key <private key>
                               [--notes <notes.json>] [--sign <pseudalis-sign>]
                               [--site <folder>] [--dist <folder>] [--dry-run] [--wait <seconds>]
    python3 publish_release.py --check [--dist <folder>] [--wait <seconds>]
    python3 publish_release.py --check-key --key <private key> [--sign <pseudalis-sign>]
    python3 publish_release.py --create-key --key <private key> [--sign <pseudalis-sign>]

The installer is by default the one that tools/build_windows.ps1 makes in the copy of the
repository of the production chain on Windows, %USERPROFILE%\work\Dlib-Transformer-extensions,
read under WSL (/mnt/c/Users/<user>/work/Dlib-Transformer-extensions/build-pseudalis/bin); an
installer named elsewhere is refused unless --any-folder is given.

The version is that of common/version.h, which must also be the version the installer carries
in its resources.  The installer is copied into <dist> (dist/pseudalis of this copy by default)
as pseudalis-setup-<version>.exe, and the installers of the other versions leave it.  The feed
names the installer by its file beside it, with its size and its SHA-256; the program writes it
with LF ends of line, signs it with pseudalis-sign, checks the signature with the public key of
the sources, which the programs embed, and puts the installer, then the signature and the feed
in place of the former ones.  A failure before that leaves the folder as it is.  Git ignores
these three files (dist/pseudalis/.gitignore): the repository holds the sources alone.

The website alone carries the files to its hosting.  Its chain lies in <site>
(~/work/cydral-website by default) and takes the feed, its signature and the installer in the
copy of the repository that PRODUCTS_REPO of its deploy.conf names: the program stops, before
it writes anything, unless that copy is the one whose <dist> it fills.  It then runs
tools/site.sh status and tools/site.sh publish from <site>, on the terminal of the publisher,
where the chain may ask for the password of the hosting, and stops when one of the two ends
with an error.  The program opens no session with the hosting, and it neither reads, keeps nor
passes on its password.

The publication ends once the website serves the feed and its signature as they were written,
and the installer by its size and its SHA-256, at the address the installed copies read, that
of the feed in host/settings.cpp (within --wait seconds, 300 by default).  --check makes that
last check alone, for the folder as it is.

--check-key checks, before the installer is even built, that the private key signs what the
public key of the sources verifies: a feed signed with another key would be ignored by every
installed copy.  --create-key makes the pair once, on the system of the publisher: the private
key at --key, which must not exist yet, readable by its owner alone in a folder of its own, and
the public key of the sources (resources/feed-public-key.pem), which it replaces and which is
committed before the installer is built.  The copies installed before trust the former key
alone: they ignore the feeds of the new one until they are installed again.

notes.json holds the release notes by language,
{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}.  --dry-run checks the
installer, the version, the notes and the folder of the website, and prints what would be done:
it writes nothing and publishes nothing.  pseudalis-sign is searched in build-pseudalis/bin and
build-pseudalis-linux/bin of the repository, then in the PATH.
"""
import argparse
import glob
import hashlib
import http.client
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
DIST = os.path.join(ROOT, "dist", "pseudalis")
# The folder of the chain of the website on the system of the publisher.
SITE = os.path.join("~", "work", "cydral-website")
# The public key of the feed, which the programs embed when it is there at their configuration.
PUBLIC_KEY = os.path.join(HERE, "..", "resources", "feed-public-key.pem")
# The address of the feed that the programs read, in host/settings.cpp.
SETTINGS = os.path.join(HERE, "..", "host", "settings.cpp")


# The copy of the repository of the production chain on Windows, under its profile folder.
CHAIN_FOLDER = ("work", "Dlib-Transformer-extensions")
CHAIN_INSTALLER = ("build-pseudalis", "bin", "pseudalis-setup.exe")


def windows_profile():
    """The profile folder of the Windows user, as WSL reads it: /mnt/c/Users/<user>."""
    try:
        out = subprocess.run(["cmd.exe", "/c", "echo %USERPROFILE%"], capture_output=True, text=True,
                             cwd="/mnt/c", timeout=30).stdout.strip()
        if out and "%" not in out:
            return subprocess.run(["wslpath", "-u", out], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("the profile folder of Windows cannot be read from here (cmd.exe and wslpath of WSL): "
                       "name the installer, with --any-folder")


def chain_installer():
    return os.path.join(windows_profile(), *(CHAIN_FOLDER + CHAIN_INSTALLER))


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


def write_release(installer, dist, name, sha, feed, signer, key):
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


def site_setting(conf, name):
    """The value deploy.conf gives the setting name, read as the chain of the website reads it: lines
    KEY=value, a comment after a space and #, the double quotes around a value left out."""
    value = ""
    with open(conf, encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.rstrip("\r\n").partition("=")
            if "".join(key.split()) != name:
                continue
            rest = re.split(r"\s#", rest, 1)[0].strip()
            if rest.endswith('"'):
                rest = rest[:-1]
            if rest.startswith('"'):
                rest = rest[1:]
            value = rest
    return value


def check_site(site, dist):
    """The folder of the chain of the website, checked: raises RuntimeError unless it holds the chain
    and takes the files of the publication in the copy of the repository whose dist is the folder
    written here."""
    site = os.path.realpath(os.path.expanduser(site))
    if not os.path.isfile(os.path.join(site, "tools", "site.sh")):
        raise RuntimeError("no chain of the website in %s (tools/site.sh): give its folder with --site" % site)
    conf = os.path.join(site, "deploy.conf")
    if not os.path.isfile(conf):
        raise RuntimeError("%s is missing: tools/site.sh setup writes it" % conf)
    repo = site_setting(conf, "PRODUCTS_REPO")
    clone = os.path.dirname(os.path.dirname(os.path.realpath(dist)))
    setup = "run tools/site.sh setup --repo %s in %s" % (shlex.quote(clone), site)
    if not repo:
        raise RuntimeError("%s names no copy of the repository (PRODUCTS_REPO): %s" % (conf, setup))
    if os.path.realpath(os.path.join(site, repo, "dist", "pseudalis")) != os.path.realpath(dist):
        raise RuntimeError("the website takes its files in %s (PRODUCTS_REPO of %s), not in the copy written "
                           "here: %s" % (repo, conf, setup))
    return site


def site_chain(site, command):
    """Runs tools/site.sh command from the folder of the website, on the terminal of the publisher:
    nothing of it is read here, and the chain asks what it needs there.  Raises RuntimeError when
    it ends with an error."""
    print()
    print("== website: tools/site.sh %s (in %s)" % (command, site))
    code = subprocess.run(["bash", "tools/site.sh", command], cwd=site).returncode
    if code != 0:
        raise RuntimeError("tools/site.sh %s ended with code %d: nothing more is published" % (command, code))


def published_address():
    """The address of the folder that the programs read, that of their feed."""
    with open(SETTINGS, encoding="utf-8") as f:
        m = re.search(r'default_feed_url\s*=\s*"(https://[^"]+)/feed\.json"', f.read())
    if not m:
        raise RuntimeError("host/settings.cpp names no address of the feed")
    return m.group(1)


class no_redirection(urllib.request.HTTPRedirectHandler):
    """The files are read at the address of the programs itself: a redirection is an error."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_url(url, agent, method="GET", timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": agent}, method=method)
    return urllib.request.build_opener(no_redirection).open(request, timeout=timeout)


def fetch(url, limit, agent):
    """The body of url, of limit bytes at most, read as the programs read it."""
    with open_url(url, agent) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("%s is larger than expected" % os.path.basename(url))
    return data


def check_online(dist, wait):
    """Waits until the website serves the feed and its signature of dist as they are, then checks the
    installer it serves against the feed, its announced size first; raises RuntimeError past wait
    seconds."""
    if not os.path.isfile(os.path.join(dist, "feed.json")) or not os.path.isfile(os.path.join(dist, "feed.json.sig")):
        raise RuntimeError("%s holds no signed feed: no version is published from there yet" % dist)
    with open(os.path.join(dist, "feed.json"), "rb") as f:
        feed = f.read()
    with open(os.path.join(dist, "feed.json.sig"), "rb") as f:
        sig = f.read()
    stable = json.loads(feed.decode("utf-8"))["channels"]["stable"]
    version, installer = stable["version"], stable["installer"]
    base = published_address()
    # The requests carry the name the programs of the version give themselves.
    agent = "Pseudalis/%s" % version
    deadline = time.time() + wait
    seen = "nothing read yet"
    while True:
        try:
            served = fetch(base + "/feed.json", 1 << 20, agent)
            if served == feed and fetch(base + "/feed.json.sig", 4096, agent) == sig:
                break
            seen = "another feed or another signature"
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise RuntimeError("the website redirects the address of the feed (HTTP %d): the programs read "
                                   "%s/feed.json itself" % (e.code, base))
            seen = str(e)
        except (OSError, http.client.HTTPException, RuntimeError) as e:
            seen = str(e)
        if time.time() >= deadline:
            raise RuntimeError("the website does not serve the feed of version %s after %d seconds (%s)" %
                               (version, wait, seen))
        time.sleep(10)
    address = base + "/" + installer["file"]
    h = hashlib.sha256()
    size = 0
    try:
        with open_url(address, agent, method="HEAD") as r:
            announced = r.headers.get("Content-Length")
        if announced != str(installer["size"]):
            raise RuntimeError("the website announces an installer of %s bytes, where the feed of version %s names "
                               "%d" % (announced, version, installer["size"]))
        with open_url(address, agent, timeout=120) as r:
            for block in iter(lambda: r.read(1 << 20), b""):
                size += len(block)
                if size > installer["size"]:
                    break
                h.update(block)
    except urllib.error.HTTPError as e:
        raise RuntimeError("the website does not serve %s (HTTP %d)" % (installer["file"], e.code))
    if size != installer["size"] or h.hexdigest() != installer["sha256"]:
        raise RuntimeError("the website serves another installer than the one the feed of version %s names" % version)
    print("the website serves version %s: the feed, its signature and the installer of %d bytes" % (version, size))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer", nargs="?")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--site", default=SITE, help="folder of the chain of the website")
    p.add_argument("--dist", default=DIST)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-key", action="store_true")
    p.add_argument("--create-key", action="store_true")
    p.add_argument("--wait", type=int, default=300)
    p.add_argument("--any-folder", action="store_true", help="take an installer of another copy than the one of "
                   "the production chain")
    a = p.parse_args()
    # The steps print in their order, before the messages of pseudalis-sign and of the website.
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
    try:
        if not a.installer:
            a.installer = chain_installer()
        elif not a.any_folder:
            chain = os.path.realpath(os.path.join(windows_profile(), *CHAIN_FOLDER))
            if not os.path.realpath(a.installer).startswith(chain + os.sep):
                raise RuntimeError("%s is not in the copy of the production chain, %s: publish the installer of "
                                   "that copy, or give --any-folder" % (a.installer, chain))
        if not os.path.isfile(a.installer):
            raise RuntimeError("no installer at %s: build it on Windows first (tools/build_windows.ps1)" % a.installer)
        version = source_version()
        feed_path = os.path.join(a.dist, "feed.json")
        notes = make_feed.read_notes(a.notes, feed_path, version)
        name = installer_name(version)
        size = os.path.getsize(a.installer)
        sha = make_feed.sha256_of(a.installer)
        feed = make_feed.make_feed(version, a.installer, None, notes, size, sha, file=name)
        check_newer(feed_path, version)
        # What can fail before anything is written, the folder of the website and the key, is checked first.
        site = check_site(a.site, a.dist)
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("folder of the publication, %s: %s, feed.json, feed.json.sig" % (a.dist, name))
        print("chain of the website, %s: tools/site.sh status, then tools/site.sh publish" % site)
        if a.dry_run:
            print("dry run: nothing written, nothing published")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        signer = find_signer(a.sign)
        check_key(signer, a.key)
        os.makedirs(a.dist, exist_ok=True)
        left = write_release(a.installer, a.dist, name, sha, feed, signer, a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    for other in left:
        print("removed from %s: %s" % (a.dist, other))
    check = "python3 %s --check" % shlex.quote(os.path.abspath(__file__))
    if os.path.realpath(a.dist) != os.path.realpath(DIST):
        check += " --dist %s" % shlex.quote(a.dist)
    # What remains to run when a step fails, the files of the folder being written.
    remaining = "(cd %s && tools/site.sh publish) && %s" % (shlex.quote(site), check)
    try:
        site_chain(site, "status")
        site_chain(site, "publish")
        remaining = check
        print()
        check_online(a.dist, a.wait)
    except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e, file=sys.stderr)
        print("the publication ends, once the cause is removed, with: %s" % remaining, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
