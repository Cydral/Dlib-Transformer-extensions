# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
r"""
Publishes a version of Pseudalis on the website: puts its installer in the folder of the chain of
the website that holds the files of Pseudalis, beside the update feed that announces it, written
and signed, has the chain publish the three files, then checks what the website serves.

    python3 publish_release.py [<pseudalis-setup.exe>] --key <private key>
                               [--notes <notes.json>] [--sign <pseudalis-sign>]
                               [--site <folder>] [--deposit] [--dry-run] [--wait <seconds>]
    python3 publish_release.py --check [--site <folder>] [--wait <seconds>]
    python3 publish_release.py --check-key --key <private key> [--sign <pseudalis-sign>]
    python3 publish_release.py --create-key --key <private key> [--sign <pseudalis-sign>]

The installer is by default the one that tools/build_windows.ps1 makes in the copy of the
repository of the production chain on Windows, %USERPROFILE%\work\Dlib-Transformer-extensions,
read under WSL (/mnt/c/Users/<user>/work/Dlib-Transformer-extensions/build-pseudalis/bin); an
installer named elsewhere is refused unless --any-folder is given.

The chain of the website lies in <site> (~/work/cydral-website by default).  It publishes the
files of its folder releases/files under files/ of the website, which gives the files of each
product the same layout, files/<product>/dist/: the folder of the publication is
<site>/releases/files/pseudalis/dist, served as files/pseudalis/dist/.  The repository holds
the sources alone, and nothing of a publication is written in it.

The version is that of common/version.h, which must also be the version the installer carries
in its resources.  The installer is copied into the folder of the publication as
pseudalis-setup-<version>.exe, and the installers of the other versions leave it.  The feed
names the installer by its file beside it, with its size and its SHA-256; the program writes it
with LF ends of line, signs it with pseudalis-sign, checks the signature with the public key of
the sources, which the programs embed, and puts the installer, then the signature and the feed
in place of the former ones.  A failure before that leaves the folder as it is, and the files
being written bear names that start with a dot, which the chain does not publish.

The website alone carries the files to its hosting.  The program runs tools/site.sh status and
tools/site.sh publish from <site>, on the terminal of the publisher, where the chain may ask
for the password of the hosting, and stops when one of the two ends with an error.  The program
opens no session with the hosting, and it neither reads, keeps nor passes on its password.
--deposit stops once the three files are written: the chain of the website publishes them when
the publisher runs it (tools/site.sh publish from <site>), with the files of another product
for instance, and --check then ends the publication.

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
# The folder of the chain of the website on the system of the publisher and, in it, the folder of
# the publication: the chain serves releases/files/<path> as files/<path> of the website.
SITE = os.path.join("~", "work", "cydral-website")
PUBLICATION = ("releases", "files", "pseudalis", "dist")
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
    it; returns the names of those that left.  The files being written bear hidden names, which the
    chain of the website does not publish: an interruption leaves nothing that it would take."""
    feed_path = os.path.join(dist, "feed.json")
    target = os.path.join(dist, name)
    partial = os.path.join(dist, "." + name + ".partial")
    new = os.path.join(dist, ".feed.json.new")
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


def check_site(site):
    """The folder of the chain of the website and, in it, the folder of the publication; raises
    RuntimeError unless site holds the chain and the folder where it takes the files it publishes."""
    site = os.path.realpath(os.path.expanduser(site))
    if not os.path.isfile(os.path.join(site, "tools", "site.sh")):
        raise RuntimeError("no chain of the website in %s (tools/site.sh): give its folder with --site" % site)
    if not os.path.isdir(os.path.join(site, *PUBLICATION[:2])):
        raise RuntimeError("the chain of the website has no folder %s in %s: tools/site.sh setup makes it" %
                           ("/".join(PUBLICATION[:2]), site))
    return site, os.path.join(site, *PUBLICATION)


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
            raise RuntimeError("the website does not serve the feed of version %s at %s/feed.json after %d seconds "
                               "(%s)" % (version, base, wait, seen))
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
    print("the website serves version %s under %s: the feed, its signature and the installer of %d bytes" %
          (version, base, size))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer", nargs="?")
    p.add_argument("--key", help="private key of the feed")
    p.add_argument("--notes")
    p.add_argument("--sign")
    p.add_argument("--site", default=SITE, help="folder of the chain of the website")
    p.add_argument("--deposit", action="store_true", help="write the files of the publication and leave the "
                   "chain of the website to the publisher")
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
                check_online(check_site(a.site)[1], a.wait)
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
        # The folder of the website, then the key, are checked before anything is written.
        site, dist = check_site(a.site)
        feed_path = os.path.join(dist, "feed.json")
        notes = make_feed.read_notes(a.notes, feed_path, version)
        name = installer_name(version)
        size = os.path.getsize(a.installer)
        sha = make_feed.sha256_of(a.installer)
        feed = make_feed.make_feed(version, a.installer, None, notes, size, sha, file=name)
        check_newer(feed_path, version)
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("folder of the publication, %s: %s, feed.json, feed.json.sig" % (dist, name))
        print("chain of the website, %s: %s" % (site, "left to the publisher (--deposit)" if a.deposit else
                                                "tools/site.sh status, then tools/site.sh publish"))
        if a.dry_run:
            print("dry run: nothing written, nothing published")
            return 0
        if not a.key:
            raise RuntimeError("--key names the private key that signs the feed")
        signer = find_signer(a.sign)
        check_key(signer, a.key)
        os.makedirs(dist, exist_ok=True)
        left = write_release(a.installer, dist, name, sha, feed, signer, a.key)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    for other in left:
        print("removed from %s: %s" % (dist, other))
    check = "python3 %s --check" % shlex.quote(os.path.abspath(__file__))
    if site != os.path.realpath(os.path.expanduser(SITE)):
        check += " --site %s" % shlex.quote(site)
    # What remains to run once the files of the folder are written, after --deposit or a step that fails.
    remaining = "(cd %s && tools/site.sh publish) && %s" % (shlex.quote(site), check)
    if a.deposit:
        print("the chain of the website publishes these files when it is run; the publication ends with: %s" %
              remaining)
        return 0
    try:
        site_chain(site, "status")
        site_chain(site, "publish")
        remaining = check
        print()
        check_online(dist, a.wait)
    except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e, file=sys.stderr)
        print("the publication ends, once the cause is removed, with: %s" % remaining, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
