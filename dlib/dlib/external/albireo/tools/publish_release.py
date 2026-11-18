# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
r"""
Publishes a version of Albireo on the website: puts its installers in the folder of the chain of
the website that holds the files of Albireo, beside the update feed that announces the version,
written and signed, and the signed catalog of the models, has the chain publish the whole, then
checks what the website serves.

    python3 publish_release.py [<albireo-setup-online.exe>] [--complete <albireo-setup.exe>]
                               [--key <private key>] [--notes <notes.json>] [--sign <albireo-sign>]
                               [--site <folder>] [--deposit] [--dry-run] [--wait <seconds>]
                               [--any-folder]
    python3 publish_release.py --catalog [--sign <albireo-sign>] [--site <folder>] [--deposit]
                               [--dry-run] [--wait <seconds>]
    python3 publish_release.py --check [--sign <albireo-sign>] [--site <folder>] [--wait <seconds>]

The installers are by default those tools/build_windows.ps1 makes in the copy of the repository
of the production chain on Windows, %USERPROFILE%\work\Dlib-Transformer-extensions, read under WSL
(/mnt/c/Users/<user>/work/Dlib-Transformer-extensions/build-albireo/bin): albireo-setup-online.exe,
the installer of the updates, without models, and albireo-setup.exe, complete with the models that
come with Albireo, which lie beside it in albireo-setup.models when both would reach the 4 GiB
Windows runs.  The complete installer is by default the one beside the installer of the updates.
An installer named outside that copy is refused unless --any-folder is given.  The folders of the
user, of Windows and of Linux, are read from the system, so that the same command publishes from
each computer of the publisher.

The chain of the website lies in <site> (~/work/cydral-website by default).  It publishes the
files of its folder releases/files under files/ of the website, which gives the files of each
product the same layout, files/<product>/dist/: the folder of the publication is
<site>/releases/files/albireo/dist, served as files/albireo/dist/, the address the installed
copies read (ALBIREO_DISTRIBUTION_URL of common/version.h).  The program reads no setting of
the website.  The repository holds the sources, the release notes of each version in notes/ and,
in dist/albireo, the signed catalog of the models: nothing of a publication is written in it.

The version is that of common/version.h, which must also be the version both installers carry
in their resources.  The folder of the publication receives the installers, as
albireo-setup-online-<version>.exe, albireo-setup-complete-<version>.exe and
albireo-setup-complete-<version>.models; catalog.json and catalog.json.sig, copied from
dist/albireo of the repository once the signature verifies; and feed.json with feed.json.sig,
the feed that names the installer of the updates by its file beside it, with its size and its
SHA-256, written with LF ends of line and signed with albireo-sign.  The private key is by
default ~/keys/albireo-feed-signing.key, and the program checks first that it signs what the key
built into the programs verifies.  A file the folder already holds as it is stays.  Each other
installer, the catalog and its signature are first copied under a name that starts with a dot,
which no feed names and the chain therefore leaves aside, and each copy is compared with its
source; the feed is written under such a name, signed, and its signature checked.  All then take
their names, the installers first, the signature of the feed and the feed last, and the
installers of the other versions leave the folder.  A failure before the first name is taken
leaves the folder as it was; a failure after it is told with what the folder then holds, and the
same command completes the version.

The publication stops before it writes anything when the folder of the publication or the
website already holds the feed of a later version, or a catalog published after the one of the
repository: the copy of the repository is then behind.  It stops too when the disk of the folder
lacks the room for the files, and when the website cannot be read from the computer, or answers
with something else than a feed and a catalog, as a portal or a filter of a network does: the
version it serves is then unknown, and what it serves afterwards out of reach of the last check.
--deposit goes on without the website, with a note.

The website alone carries the files to its hosting.  The program runs tools/site.sh status and
tools/site.sh publish from <site>, on the terminal of the publisher, where the chain may ask
for the password of the hosting, and stops when one of the two ends with an error.  It opens no
session with the hosting, and it neither reads, keeps nor passes on its password.  --deposit
stops once the files are written: the chain of the website publishes them when the publisher
runs it (tools/site.sh publish from <site>), with the files of another product for instance,
and --check then ends the publication.

The publication ends once the website serves the feed, the catalog and their signatures as they
are in the folder of the publication, byte for byte, and the installers at their sizes, at the
address the installed copies read (within --wait seconds, 300 by default); the SHA-256 of each
installer is the one the list of the files of the website gives, for those it lists.  --check
makes that last check alone, for the folder as it is, with the signatures of its feed and of its
catalog when albireo-sign is found.

--catalog publishes the catalog of the repository between two versions, once
tools/make_catalog.py wrote it and albireo-sign signed it: a model that left the hub, a model
added to the list.  The feed stays the one of the folder of the publication, which must be the
feed the website serves, byte for byte, beside the installer it names; the program checks both
signatures, puts the catalog and its signature in the folder, runs the two commands of the chain
and checks what the website serves.  It reads no installer and no key.

The release notes are read by language, {"en": "...", "fr": "...", "de": "...", "it": "...",
"es": "..."}, in the file --notes names, else in notes/<version>.json of the sources; without
either, the notes the feed of the folder already gives the same version are kept, and the feed
of a version without notes reads "Albireo <version>".  Each note is a text, and the English one
is required.  --dry-run makes the checks and prints the files of the publication, the notes and
the installers that would leave the folder: it writes nothing and publishes nothing.
albireo-sign is searched in build-albireo/bin of the repository, then in the PATH.
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
# The signed catalog of the models, which the repository keeps with its signature beside it.
CATALOG = os.path.join(ROOT, "dist", "albireo", "catalog.json")
# The release notes of the versions, one file by version: notes/<version>.json of the sources.
NOTES = os.path.join(HERE, "..", "notes")
# The folder of the chain of the website and the private key of the feed, under the folder of the user.
SITE = os.path.join("~", "work", "cydral-website")
KEY = os.path.join("~", "keys", "albireo-feed-signing.key")
# In the folder of the chain, the folder of the publication: the chain serves releases/files/<path> as
# files/<path> of the website.
PUBLICATION = ("releases", "files", "albireo", "dist")
# The copy of the repository of the production chain on Windows, under its profile folder.
CHAIN_FOLDER = ("work", "Dlib-Transformer-extensions")
CHAIN_BIN = ("build-albireo", "bin")
# The public key verify_command() checks with, in place of the key built into albireo-sign: the tests only.
PUBLIC_KEY = None
# The seconds between two readings of the website that does not serve the files of the folder yet.
POLL = 10
# The signed files of the folder of the publication.
SIGNED = ("feed.json", "feed.json.sig", "catalog.json", "catalog.json.sig")
# The installers of every version, as the folder of the publication names them.
INSTALLERS = ("albireo-setup-online-*.exe", "albireo-setup-complete-*.exe", "albireo-setup-complete-*.models")
# The names of the files being written, which a run cut short may leave in the folder of the publication.
HIDDEN = (".albireo-setup-*.partial", ".catalog.json.partial", ".catalog.json.sig.partial", ".feed.json.new",
          ".feed.json.new.sig")


def source_version():
    with open(os.path.join(HERE, "..", "common", "version.h"), encoding="utf-8") as f:
        m = re.search(r'#define ALBIREO_VERSION "(\d+\.\d+\.\d+)"', f.read())
    if not m:
        raise RuntimeError("common/version.h names no version")
    return m.group(1)


def published_address():
    """The address of the folder of the versions that the programs read, without a final slash."""
    with open(os.path.join(HERE, "..", "common", "version.h"), encoding="utf-8") as f:
        m = re.search(r'#define ALBIREO_DISTRIBUTION_URL "(https://[^"]+)"', f.read())
    if not m:
        raise RuntimeError("common/version.h names no folder of the versions (ALBIREO_DISTRIBUTION_URL)")
    return m.group(1).rstrip("/")


def online_name(version):
    return "albireo-setup-online-%s.exe" % version


def complete_name(version):
    return "albireo-setup-complete-%s.exe" % version


def models_name(version):
    return "albireo-setup-complete-%s.models" % version


def profile_path(output):
    """The folder a console of Windows printed for %USERPROFILE%, from the bytes of its output, which
    cmd.exe /u writes in UTF-16; None when they hold no path of Windows."""
    for encoding in ("utf-16-le", "utf-8"):
        try:
            text = output.decode(encoding).strip().strip("\ufeff")
        except UnicodeDecodeError:
            continue
        if re.fullmatch(r"[A-Za-z]:\\[^%\r\n\x00]+", text):
            return text
    return None


def windows_profile():
    """The profile folder of the Windows user, as WSL reads it: /mnt/c/Users/<user>."""
    # The console of Windows by its name, then by its path when WSL does not add the folders of Windows to
    # the PATH; in UTF-16 first, which carries every name, then in the bytes of its code page.
    # The console starts in a folder of Windows, which it takes as its own without a word.
    start = "/mnt/c" if os.path.isdir("/mnt/c") else None
    for console in ("cmd.exe", "/mnt/c/Windows/System32/cmd.exe"):
        for options in (["/u", "/c"], ["/c"]):
            try:
                out = profile_path(subprocess.run([console] + options + ["echo %USERPROFILE%"], capture_output=True,
                                                  cwd=start, timeout=30).stdout)
                if not out:
                    continue
                folder = subprocess.run(["wslpath", "-u", out], capture_output=True, timeout=30).stdout
                folder = folder.decode("utf-8", "surrogateescape").strip()
                if folder.startswith("/"):
                    return folder
            except (OSError, subprocess.SubprocessError):
                break
    raise RuntimeError("the profile folder of Windows cannot be read from here (cmd.exe and wslpath of WSL): "
                       "name the installers, with --any-folder")


def chain_copy():
    """The copy of the repository of the production chain on Windows."""
    return os.path.realpath(os.path.join(windows_profile(), *CHAIN_FOLDER))


def find_signer(given):
    if given:
        return given
    bin_dir = os.path.join(ROOT, "build-albireo", "bin")
    # The program of the system this one runs on comes first when a folder holds both.
    names = ("albireo-sign", "albireo-sign.exe") if os.name != "nt" else ("albireo-sign.exe", "albireo-sign")
    for name in names:
        for folder in (bin_dir, os.path.join(bin_dir, "Release")):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    found = shutil.which("albireo-sign")
    if not found:
        raise RuntimeError("albireo-sign not found: build Albireo, or give its path with --sign")
    return found


def verify_command(signer, path):
    """The command that checks the signature of path, with the key built into albireo-sign."""
    return [signer, "verify", path] + ([PUBLIC_KEY] if PUBLIC_KEY else [])


def compare(a, b):
    x = [int(p) for p in a.split(".")]
    y = [int(p) for p in b.split(".")]
    return (x > y) - (x < y)


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_text(path):
    """The text of the file at path, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, ValueError):
        return None


def feed_version(text):
    """The version a feed announces, from its text, or None."""
    try:
        version = json.loads(text)["channels"]["stable"]["version"]
    except (ValueError, KeyError, TypeError):
        return None
    return version if isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version) else None


def feed_installer(text, where):
    """The version the feed read at where announces and the installer it names: (version, file, size,
    SHA-256), from its text.  Raises RuntimeError for a text of another form."""
    version = feed_version(text)
    try:
        installer = json.loads(text)["channels"]["stable"]["installer"]
        described = (version, installer["file"], int(installer["size"]), installer["sha256"])
    except (ValueError, KeyError, TypeError):
        described = None
    if not version or not described or not isinstance(described[1], str) or not isinstance(described[3], str):
        raise RuntimeError("%s is not an update feed: it names no version with its installer" % where)
    return described


def catalog_time(text):
    """The time of publication a catalog gives, from its text, or None."""
    try:
        updated = json.loads(text)["updated"]
    except (ValueError, KeyError, TypeError):
        return None
    form = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    return updated if isinstance(updated, str) and re.fullmatch(form, updated) else None


def release_notes(given, folder, version):
    """The release notes of version by language, and the file they come from: the file given, else
    notes/<version>.json of the sources, else, with no file to name, the notes the feed of folder already
    gives that version, or the text "Albireo <version>".  Raises ValueError for notes that are no texts
    by language or lack the English one."""
    path = given or os.path.join(NOTES, version + ".json")
    if given or os.path.isfile(path):
        notes, source = make_feed.read_notes(path, None, version), os.path.normpath(path)
    else:
        notes, source = make_feed.read_notes(None, os.path.join(folder, "feed.json"), version), None
    for language, text in notes.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError("the release notes give no text for \"%s\"" % language)
    return notes, source


def check_newer(folder, version):
    """Raises RuntimeError when the feed of folder, the folder of the publication, announces a later
    version than version: the chain of the website would hold the files of a version that left."""
    held = feed_version(read_text(os.path.join(folder, "feed.json")) or "")
    if held and compare(version, held) < 0:
        raise RuntimeError("%s already holds the feed of version %s, later than %s: bring this copy of the "
                           "repository up to date and publish its version" % (folder, held, version))


def check_key(signer, key):
    """Raises RuntimeError unless key signs what the key built into the programs verifies."""
    if not os.path.isfile(key):
        raise RuntimeError("no private key at %s: give it with --key" % key)
    folder = tempfile.mkdtemp(prefix="albireo-key-")
    try:
        probe = os.path.join(folder, "probe.json")
        with open(probe, "wb") as f:
            f.write(os.urandom(64))
        subprocess.run([signer, "sign", key, probe], check=True, stdout=subprocess.DEVNULL)
        if subprocess.run(verify_command(signer, probe), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError("%s is not the key of the public key built into the programs: the installed copies "
                               "would ignore the feed it signs" % key)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def verifies(signer, path):
    """Whether the signature beside the file at path is one the key of the programs verifies."""
    return subprocess.run(verify_command(signer, path), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def check_catalog(signer):
    """The time of publication and the number of models of the catalog of the repository, its signature
    checked.  Raises RuntimeError for a catalog that is missing, unsigned or of another form."""
    if not os.path.isfile(CATALOG) or not os.path.isfile(CATALOG + ".sig"):
        raise RuntimeError("%s holds no signed catalog (catalog.json and catalog.json.sig)" %
                           os.path.dirname(CATALOG))
    if not verifies(signer, CATALOG):
        converted = b"\r\n" in read_bytes(CATALOG)
        raise RuntimeError("the signature of %s does not verify%s" % (CATALOG, (
            ": the file has CRLF ends of line, which a checkout with core.autocrlf gives it; delete it, then "
            "git checkout -- dist/albireo (dist/albireo/.gitattributes keeps the signed files as they are)")
            if converted else ""))
    text = read_text(CATALOG) or ""
    updated = catalog_time(text)
    if not updated:
        raise RuntimeError("%s gives no time of publication (updated)" % CATALOG)
    models = json.loads(text).get("models")
    if not isinstance(models, list):
        raise RuntimeError("%s lists no models" % CATALOG)
    return updated, len(models)


def check_later_catalog(updated, text, where):
    """Raises RuntimeError when the catalog whose text was found at where was published after updated,
    the time of the catalog of the repository: the services never take a catalog older than theirs."""
    found = catalog_time(text or "")
    if found and found > updated:
        raise RuntimeError("%s holds a catalog published %s, after the one of this copy of the repository (%s): "
                           "bring this copy up to date first" % (where, found, updated))


def held_feed(signer, folder):
    """The version the feed of folder announces and the file of the installer it names, its signature
    checked.  Raises RuntimeError otherwise."""
    path = os.path.join(folder, "feed.json")
    if not os.path.isfile(path) or not os.path.isfile(path + ".sig"):
        raise RuntimeError("%s holds no signed feed (feed.json and feed.json.sig): tools/site.sh take online "
                           "brings to this computer what the hosting serves, and the publication of a version "
                           "writes them" % folder)
    if not verifies(signer, path):
        raise RuntimeError("the signature of %s does not verify" % path)
    return feed_installer(read_text(path) or "", path)[:2]


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


def same_file(source, target, sha=None):
    """Whether target is a copy of source: its size, and its SHA-256, the one of source being sha when given."""
    if not os.path.isfile(target) or os.path.getsize(target) != os.path.getsize(source):
        return False
    return make_feed.sha256_of(target) == (sha or make_feed.sha256_of(source))


def hidden_name(folder, name):
    """The name a file bears in folder while it is written: it starts with a dot, so that no feed names
    it and the chain of the website leaves it aside."""
    return os.path.join(folder, "." + name + ".partial")


def take_name(path):
    """Frees the name path: a file or a link left there is removed, never written through."""
    remove_quietly(path)
    if os.path.lexists(path):
        raise RuntimeError("%s cannot be removed" % path)


def clear_hidden(folder):
    """Removes from folder the files that a run cut short left under the names of the files being written."""
    for pattern in HIDDEN:
        for path in glob.glob(os.path.join(glob.escape(folder), pattern)):
            take_name(path)


def check_names(folder, names):
    """Raises RuntimeError when folder holds a folder under one of names, which no file can replace."""
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isdir(path) and not os.path.islink(path):
            raise RuntimeError("%s is a folder: the file of that name cannot take its place" % path)


def check_room(folder, files):
    """Raises RuntimeError when the disk of folder, or of the folder above it that exists, lacks the room
    for those of files, (source, name, SHA-256 or None), that folder does not hold at their sizes."""
    needed = 0
    for source, name, _ in files:
        target = os.path.join(folder, name)
        if not os.path.isfile(target) or os.path.getsize(target) != os.path.getsize(source):
            needed += os.path.getsize(source)
    there = folder
    while not os.path.isdir(there):
        there = os.path.dirname(there)
    free = shutil.disk_usage(there).free
    if needed > free:
        raise RuntimeError("%s needs %d bytes for the files of the version and its disk has %d free: the "
                           "installers of the earlier version leave it once this one is in place" %
                           (folder, needed, free))


def copy_hashed(source, target):
    """Copies source to target, a name that must be free, in one reading; returns the SHA-256 and the
    size of the bytes read."""
    h = hashlib.sha256()
    size = 0
    with open(source, "rb") as given, open(target, "xb") as copy:
        for block in iter(lambda: given.read(1 << 20), b""):
            copy.write(block)
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def stage(source, folder, name, sha=None, always=False):
    """Copies source into folder under the hidden name of name, then reads the copy again and compares
    it with the bytes read from source, whose SHA-256 is sha when the caller has it; returns the hidden
    path, or None when folder already holds that file under name, unless always is set."""
    target = os.path.join(folder, name)
    if not always and not os.path.islink(target) and same_file(source, target, sha):
        return None
    partial = hidden_name(folder, name)
    take_name(partial)
    print("copying %s, %d bytes" % (name, os.path.getsize(source)))
    read, size = copy_hashed(source, partial)
    if (sha and read != sha) or os.path.getsize(partial) != size or make_feed.sha256_of(partial) != read:
        raise RuntimeError("the copy of %s differs from it: %s" % (source, partial))
    return partial


def put_pair(new, new_signature, path):
    """Puts a signed file and its signature, written beside path as new and new_signature, in place of
    the former ones, the signature first; the former signature comes back when the file cannot take
    its place."""
    former = None
    if os.path.isfile(path + ".sig"):
        former = read_bytes(path + ".sig")
    os.replace(new_signature, path + ".sig")
    try:
        os.replace(new, path)
    except OSError:
        if former is None:
            remove_quietly(path + ".sig")
        else:
            with open(path + ".sig", "wb") as f:
                f.write(former)
        raise


def stage_catalog(folder):
    """Copies the signed catalog of the repository into folder under hidden names, each copy compared
    with its source; returns the hidden paths of the catalog and of its signature, or None when folder
    holds both as they are."""
    parts = [(CATALOG + suffix, "catalog.json" + suffix) for suffix in ("", ".sig")]
    if all(not os.path.islink(os.path.join(folder, name)) and same_file(source, os.path.join(folder, name))
           for source, name in parts):
        return None
    # The two take their names together: the one that is already there is copied with the other.
    return [stage(source, folder, name, always=True) for source, name in parts]


def write_catalog(folder):
    """Puts the signed catalog of the repository in folder, the folder of the publication; returns the
    line that tells what was done."""
    os.makedirs(folder, exist_ok=True)
    check_names(folder, ("catalog.json", "catalog.json.sig"))
    clear_hidden(folder)
    try:
        staged = stage_catalog(folder)
        if not staged:
            return "in place: catalog.json, catalog.json.sig"
        put_pair(staged[0], staged[1], os.path.join(folder, "catalog.json"))
    finally:
        for name in ("catalog.json", "catalog.json.sig"):
            remove_quietly(hidden_name(folder, name))
    return "placed: catalog.json, catalog.json.sig"


def leaving(folder, kept):
    """The installers of folder that kept does not name, those of the other versions, sorted."""
    found = []
    for pattern in INSTALLERS:
        for other in glob.glob(os.path.join(glob.escape(folder), pattern)):
            if os.path.basename(other) not in kept:
                found.append(other)
    return sorted(found)


def write_publication(folder, files, feed, signer, key):
    """Fills folder, the folder of the publication, with the installers of files, (source, name, SHA-256
    or None), the signed catalog of the repository, and feed, written and signed here, and prints each
    step once it is done.  The files that folder lacks are first written under hidden names and
    checked; all then take their names, the installers first, the signature of the feed and the feed
    last, and the installers of the other versions leave the folder, with the file of the models of
    this version when the build gives none.  A failure before the first name is taken leaves the
    folder as it was; a failure after it raises RuntimeError, which tells what the folder then holds."""
    os.makedirs(folder, exist_ok=True)
    names = [name for _, name, _ in files]
    check_names(folder, names + list(SIGNED))
    clear_hidden(folder)
    new = os.path.join(folder, ".feed.json.new")
    ready = []
    named = False
    try:
        for source, name, sha in files:
            partial = stage(source, folder, name, sha)
            if partial:
                ready.append((partial, name))
            else:
                print("in place: %s" % name)
        catalog = stage_catalog(folder)
        make_feed.write_feed(new, feed)
        subprocess.run([signer, "sign", key, new], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(verify_command(signer, new), check=True, stdout=subprocess.DEVNULL)
        try:
            for partial, name in ready:
                os.replace(partial, os.path.join(folder, name))
                named = True
                print("placed: %s, %d bytes" % (name, os.path.getsize(os.path.join(folder, name))))
            if catalog:
                put_pair(catalog[0], catalog[1], os.path.join(folder, "catalog.json"))
                named = True
            print("%s: catalog.json, catalog.json.sig" % ("placed" if catalog else "in place"))
            put_pair(new, new + ".sig", os.path.join(folder, "feed.json"))
            print("written: feed.json, feed.json.sig, the signature verified")
        except OSError as e:
            if not named:
                raise
            raise RuntimeError("%s; %s now holds files of the version that its feed does not describe yet: the "
                               "same command completes it once the cause is removed" % (e, folder))
    finally:
        # Whatever was staged leaves the folder, a copy cut short by a failure included.
        for name in names + ["catalog.json", "catalog.json.sig"]:
            remove_quietly(hidden_name(folder, name))
        remove_quietly(new)
        remove_quietly(new + ".sig")
    for other in leaving(folder, set(names)):
        if os.path.isdir(other) and not os.path.islink(other):
            print("left aside: %s, a folder" % os.path.basename(other))
            continue
        os.remove(other)
        print("removed: %s" % os.path.basename(other))


def uncommitted_catalog():
    """Whether Git tells that the catalog of the repository or its signature differs from what it holds;
    False when it cannot tell."""
    folder = os.path.dirname(CATALOG)
    try:
        r = subprocess.run(["git", "-C", folder, "status", "--porcelain", "--", "catalog.json", "catalog.json.sig"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def site_chain(site, command):
    """Runs tools/site.sh command from the folder of the website, on the terminal of the publisher:
    nothing of it is read here, and the chain asks what it needs there.  Raises RuntimeError when
    it ends with an error."""
    print()
    print("== website: tools/site.sh %s (in %s)" % (command, site))
    code = subprocess.run(["bash", "tools/site.sh", command], cwd=site).returncode
    if code != 0:
        raise RuntimeError("tools/site.sh %s ended with code %d: nothing more is published" % (command, code))


class no_redirection(urllib.request.HTTPRedirectHandler):
    """The files are read at the address of the programs itself: a redirection is an error."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_url(url, agent, method="GET", timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": agent, "Accept-Encoding": "identity"},
                                     method=method)
    return urllib.request.build_opener(no_redirection).open(request, timeout=timeout)


def fetch(url, limit, agent):
    """The body of url, of limit bytes at most, read as the programs read it."""
    with open_url(url, agent) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("%s is larger than expected" % os.path.basename(url))
    return data


def served_bytes(name, limit, agent):
    """The bytes of the file name that the website serves in the folder of the versions, or None when it
    holds no such file yet.  Raises RuntimeError when the website cannot be read."""
    address = published_address() + "/" + name
    try:
        return fetch(address, limit, agent)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError("%s cannot be read: HTTP %d" % (address, e.code))
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError("%s cannot be read: %s" % (address, e))


def served_state(agent):
    """What the website serves: the version its feed announces, the bytes of that feed and the text of
    its catalog, each None when the website holds none yet.  Raises RuntimeError when the website
    cannot be read, an answer that is no feed or no catalog included: a portal or a filter of a
    network answers a request with its own page."""
    other = "%s/%s is answered with something else than %s, as a portal or a filter of the network does"
    feed = served_bytes("feed.json", 1 << 20, agent)
    version = feed_version(feed.decode("utf-8", "replace")) if feed is not None else None
    if feed is not None and not version:
        raise RuntimeError(other % (published_address(), "feed.json", "an update feed"))
    catalog = served_bytes("catalog.json", 16 << 20, agent)
    text = catalog.decode("utf-8", "replace") if catalog is not None else None
    if text is not None and not catalog_time(text):
        raise RuntimeError(other % (published_address(), "catalog.json", "a catalog"))
    return version, feed, text


def read_website(version, lenient):
    """What the website serves, as served_state() gives it, and None with the reason when it cannot be
    read and lenient lets the publication go on without it.  Raises RuntimeError otherwise."""
    try:
        return served_state("Albireo/%s" % version), None
    except RuntimeError as e:
        if not lenient:
            raise RuntimeError("%s: the version the website serves is unknown, and what it serves could not be "
                               "checked from this computer; --deposit writes the files of the publication and "
                               "leaves the chain of the website to the publisher" % e)
        return (None, None, None), str(e)


def check_online(folder, wait, signer=None):
    """Waits until the website serves the signed files of folder, the folder of the publication, as they
    are, then checks the installers it serves: the size of the one the feed names and of the other
    installers of the version that folder holds, and the SHA-256 of each one the list of the files of
    the website gives.  With signer, the signatures of the feed and of the catalog of folder are checked
    first.  Raises RuntimeError past wait seconds, for a feed of a later version, or for an installer
    the website serves otherwise."""
    local = {}
    for name in SIGNED:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise RuntimeError("%s holds no %s: no version is published from this computer yet, and "
                               "tools/site.sh take online brings what the hosting serves" % (folder, name))
        local[name] = read_bytes(path)
    version, file, size, sha = feed_installer(local["feed.json"].decode("utf-8", "replace"),
                                              os.path.join(folder, "feed.json"))
    for name in ("feed.json", "catalog.json") if signer else ():
        if not verifies(signer, os.path.join(folder, name)):
            raise RuntimeError("the signature of %s does not verify: the installed copies would ignore the file" %
                               os.path.join(folder, name))
    base = published_address()
    # The requests carry the name the programs of the version give themselves.
    agent = "Albireo/%s" % version
    deadline = time.time() + wait
    while True:
        seen = None
        for name in SIGNED:
            try:
                served = fetch(base + "/" + name, 16 << 20, agent)
            except urllib.error.HTTPError as e:
                if 300 <= e.code < 400:
                    raise RuntimeError("the website redirects the folder of the versions (HTTP %d): the programs read "
                                       "%s itself" % (e.code, base))
                seen = "%s: HTTP %d" % (name, e.code)
            except (OSError, http.client.HTTPException, RuntimeError) as e:
                seen = "%s: %s" % (name, e)
            else:
                if served != local[name]:
                    seen = "another %s" % name
                    other = feed_version(served.decode("utf-8", "replace")) if name == "feed.json" else None
                    if other and compare(other, version) > 0:
                        raise RuntimeError("the website serves version %s, later than the %s of %s: tools/site.sh "
                                           "take online brings to this computer what the hosting serves" %
                                           (other, version, folder))
                    if other == version:
                        seen = ("another feed of version %s: another build or other notes than those of the "
                                "folder" % version)
            if seen:
                break
        if not seen:
            break
        if time.time() >= deadline:
            raise RuntimeError("the website does not serve the signed files of version %s after %d seconds (%s)" %
                               (version, wait, seen))
        time.sleep(max(0.0, min(POLL, deadline - time.time())))
    expected = [(file, size)]
    for name in (complete_name(version), models_name(version)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            expected.append((name, os.path.getsize(path)))
    for name, length in expected:
        try:
            with open_url(base + "/" + name, agent, method="HEAD") as r:
                announced = r.headers.get("Content-Length")
        except urllib.error.HTTPError as e:
            raise RuntimeError("the website does not serve %s (HTTP %d)" % (name, e.code))
        if announced != str(length):
            raise RuntimeError("the website announces %s at %s bytes, where version %s has %d" %
                               (name, announced, version, length))
    # The list of the files of the website, which gives the hash it measured for each installer.
    root = base[:base.index("/files/") + len("/files/")] if "/files/" in base else base + "/"
    try:
        details = json.loads(fetch(root + "manifest.json", 1 << 20, agent).decode("utf-8"))["details"]
    except (ValueError, KeyError, TypeError, OSError, http.client.HTTPException, RuntimeError):
        details = {}
    measured = 0
    for name, _ in expected:
        try:
            listed = details[base[len(root):] + "/" + name]["sha256"]
        except (KeyError, TypeError):
            continue
        # The installer of the updates by the hash of the feed, the others by the one of their files.
        if listed != (sha if name == file else make_feed.sha256_of(os.path.join(folder, name))):
            raise RuntimeError("the website holds another %s than the one of the folder of version %s" %
                               (name, version))
        measured += 1
    print("the website serves version %s: the feed, the catalog, their signatures%s, and %s; its list of files "
          "gives the hash of %d of these %d installers" % (
              version, ", which verify" if signer else "", ", ".join("%s of %d bytes" % e for e in expected),
              measured, len(expected)))


def end_commands(site):
    """The command that checks the publication, and the commands that end it once the files of the folder
    of the publication are written."""
    check = "python3 %s --check" % shlex.quote(os.path.abspath(__file__))
    if site != os.path.realpath(os.path.expanduser(SITE)):
        check += " --site %s" % shlex.quote(site)
    return check, "(cd %s && tools/site.sh status && tools/site.sh publish) && %s" % (shlex.quote(site), check)


def run_chain(site, folder, wait, signer):
    """Has the chain of the website publish the folder of the publication, then checks what the website
    serves; returns the exit code of the program."""
    check, remaining = end_commands(site)
    try:
        site_chain(site, "status")
        site_chain(site, "publish")
        remaining = check
        print()
        check_online(folder, wait, signer)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e, file=sys.stderr)
        print("the publication ends, once the cause is removed, with: %s" % remaining, file=sys.stderr)
        return 1
    return 0


def note_catalog():
    """Tells a catalog of the repository that Git does not hold yet, with the commands that keep it."""
    if uncommitted_catalog():
        print("note: Git does not hold the catalog published yet; the repository keeps it with: git -C %s add "
              "catalog.json catalog.json.sig && git -C %s commit -m \"Revise the catalog of Albireo\" && git -C %s "
              "push" % ((shlex.quote(os.path.dirname(CATALOG)),) * 3))


def refused(mode, options):
    """Prints the error of the first of options, (value, name), that mode does not take; returns whether
    one was given."""
    for given, what in options:
        if given:
            print("error: %s takes no %s" % (mode, what), file=sys.stderr)
            return True
    return False


def publish_catalog(a):
    """The publication of the catalog of the repository between two versions (--catalog)."""
    if refused("--catalog", ((a.installer, "installer"), (a.complete, "--complete"), (a.notes, "--notes"),
                             (a.key, "--key"), (a.any_folder, "--any-folder"))):
        return 1
    lenient = a.deposit or a.dry_run
    try:
        signer = find_signer(a.sign)
        updated, count = check_catalog(signer)
        site, folder = check_site(a.site)
        version, installer = held_feed(signer, folder)
        (served, served_feed, served_catalog), unread = read_website(version, lenient)
        if not unread:
            if not served:
                raise RuntimeError("the website serves no feed at %s: publish version %s first, which carries the "
                                   "catalog with it" % (published_address(), version))
            if compare(version, served) < 0:
                raise RuntimeError("the website serves version %s, later than the %s of %s: tools/site.sh take "
                                   "online brings to this computer what the hosting serves" %
                                   (served, version, folder))
            if compare(version, served) > 0:
                raise RuntimeError("the website serves version %s and %s holds %s: publish that version, which "
                                   "carries the catalog with it" % (served, folder, version))
            # The chain publishes the feed of the folder with the catalog: it must be the feed served.
            signature = served_bytes("feed.json.sig", 4096, "Albireo/%s" % version)
            if served_feed != read_bytes(os.path.join(folder, "feed.json")) or \
                    signature != read_bytes(os.path.join(folder, "feed.json.sig")):
                raise RuntimeError("the feed of %s is not the one the website serves for version %s, another build "
                                   "or other notes: the publication of a version publishes its feed, and, to keep "
                                   "the version the website serves, remove feed.json, feed.json.sig and the "
                                   "installers of that folder, then run tools/site.sh take online" %
                                   (folder, version))
        check_later_catalog(updated, read_text(os.path.join(folder, "catalog.json")), folder)
        check_later_catalog(updated, served_catalog, "the website")
        print("catalog of %d models, published %s, for version %s" % (count, updated, version))
        print("folder of the publication, %s: catalog.json, catalog.json.sig, beside feed.json and feed.json.sig"
              % folder)
        print("chain of the website, %s: %s" % (site, "left to the publisher (--deposit)" if a.deposit
                                                 else "tools/site.sh status, then tools/site.sh publish"))
        if unread:
            print("note: %s: what the website serves was not compared with the folder" % unread)
        if not os.path.isfile(os.path.join(folder, installer)):
            # The chain stops at its upload when its folder lacks an installer its feed names.
            missing = ("%s holds no %s, which its feed names: tools/site.sh take online brings to this computer "
                       "what the hosting serves" % (folder, installer))
            if not lenient:
                raise RuntimeError(missing)
            print("note: %s" % missing)
        if a.dry_run:
            print("dry run: nothing written, nothing published")
            return 0
        print(write_catalog(folder))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if a.deposit:
        print("the chain of the website publishes the catalog when it is run; the publication ends with: %s" %
              end_commands(site)[1])
        note_catalog()
        return 0
    code = run_chain(site, folder, a.wait, signer)
    note_catalog()
    return code


def publish_version(a):
    """The publication of the version of the sources."""
    key = os.path.expanduser(a.key or KEY)
    try:
        chain = None if a.any_folder and a.installer else chain_copy()
        if not a.any_folder:
            for given in (a.installer, a.complete):
                if given and not os.path.realpath(given).startswith(chain + os.sep):
                    raise RuntimeError("%s is not in the copy of the production chain, %s: publish the installers "
                                       "of that copy, or give --any-folder" % (given, chain))
        if not a.installer:
            a.installer = os.path.join(chain, *(CHAIN_BIN + ("albireo-setup-online.exe",)))
        if not a.complete:
            a.complete = os.path.join(os.path.dirname(a.installer), "albireo-setup.exe")
        for path in (a.installer, a.complete):
            if not os.path.isfile(path):
                raise RuntimeError("no installer at %s: build it on Windows first (tools/build_windows.ps1)" % path)
        version = source_version()
        # What can fail is checked before anything is written, the version of both installers first.
        for path in (a.installer, a.complete):
            make_feed.check_installer(path, version)
        site, folder = check_site(a.site)
        notes, notes_file = release_notes(a.notes, folder, version)
        check_newer(folder, version)
        (served, _, served_catalog), unread = read_website(version, a.deposit or a.dry_run)
        if served and compare(version, served) < 0:
            raise RuntimeError("the website already serves version %s, later than %s: bring this copy of the "
                               "repository up to date and publish its version" % (served, version))
        name = online_name(version)
        files = [(a.installer, name, None), (a.complete, complete_name(version), None)]
        # The models of a complete installer that would reach the 4 GiB Windows runs lie beside it.
        models = os.path.splitext(a.complete)[0] + ".models"
        if os.path.isfile(models):
            files.append((models, models_name(version), None))
        signer = find_signer(a.sign)
        check_key(signer, key)
        updated, count = check_catalog(signer)
        check_later_catalog(updated, read_text(os.path.join(folder, "catalog.json")), folder)
        check_later_catalog(updated, served_catalog, "the website")
        check_names(folder, [n for _, n, _ in files] + list(SIGNED))
        check_room(folder, files)
        size = os.path.getsize(a.installer)
        sha = make_feed.sha256_of(a.installer)
        feed = make_feed.make_feed(version, a.installer, name, notes, size, sha)
        files[0] = (a.installer, name, sha)
        print("version %s, installer of %d bytes, SHA-256 %s" % (version, size, sha))
        print("folder of the publication, %s: %s, catalog.json, catalog.json.sig, feed.json, feed.json.sig" %
              (folder, ", ".join(n for _, n, _ in files)))
        print("catalog of %d models, published %s" % (count, updated))
        if notes_file:
            print("release notes of %s: %s" % (notes_file, ", ".join(sorted(notes))))
        elif notes != {"en": "Albireo " + version}:
            print("release notes: those the feed of the folder already gives version %s (%s)" %
                  (version, ", ".join(sorted(notes))))
        else:
            print("note: no release notes for version %s (%s): the feed reads \"Albireo %s\"" %
                  (version, os.path.normpath(os.path.join(NOTES, version + ".json")), version))
        print("chain of the website, %s: %s" % (site, "left to the publisher (--deposit)" if a.deposit
                                                 else "tools/site.sh status, then tools/site.sh publish"))
        earlier = [os.path.basename(other) for other in leaving(folder, {n for _, n, _ in files})]
        if earlier:
            print("installers that leave the folder once the version is in place: %s" % ", ".join(earlier))
        if unread:
            print("note: %s: what the website serves was not compared with version %s" % (unread, version))
        if a.dry_run:
            print("dry run: nothing written, nothing published")
            return 0
        write_publication(folder, files, feed, signer, key)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if a.deposit:
        print("the chain of the website publishes these files when it is run; the publication ends with: %s" %
              end_commands(site)[1])
        note_catalog()
        return 0
    code = run_chain(site, folder, a.wait, signer)
    note_catalog()
    return code


def check_publication(a):
    """The check of what the website serves, for the folder of the publication as it is (--check)."""
    if refused("--check", ((a.installer, "installer"), (a.complete, "--complete"), (a.notes, "--notes"),
                           (a.key, "--key"), (a.catalog, "--catalog"), (a.deposit, "--deposit"),
                           (a.dry_run, "--dry-run"), (a.any_folder, "--any-folder"))):
        return 1
    try:
        folder = check_site(a.site)[1]
        try:
            signer = find_signer(a.sign)
        except RuntimeError as e:
            signer = None
            print("note: %s: the signatures of the folder are not checked" % e)
        check_online(folder, a.wait, signer)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, http.client.HTTPException) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("installer", nargs="?")
    p.add_argument("--complete", help="the complete installer, with the models that come with Albireo")
    p.add_argument("--key", help="private key of the feed, ~/keys/albireo-feed-signing.key by default")
    p.add_argument("--notes", help="release notes by language, notes/<version>.json of the sources by default")
    p.add_argument("--sign")
    p.add_argument("--site", default=SITE, help="folder of the chain of the website")
    p.add_argument("--deposit", action="store_true", help="write the files of the publication and leave the "
                   "chain of the website to the publisher")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--catalog", action="store_true", help="publish the catalog of the repository, the version as "
                   "it is")
    p.add_argument("--wait", type=int, default=300)
    p.add_argument("--any-folder", action="store_true", help="take installers of another copy than the one of the "
                   "production chain")
    a = p.parse_args()
    # The steps print in their order, before the messages of albireo-sign and of the website.
    sys.stdout.reconfigure(line_buffering=True)
    try:
        if a.check:
            return check_publication(a)
        if a.catalog:
            return publish_catalog(a)
        return publish_version(a)
    except KeyboardInterrupt:
        # The files being written left the folder with the exception.
        print("\ninterrupted: the same command starts again", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
