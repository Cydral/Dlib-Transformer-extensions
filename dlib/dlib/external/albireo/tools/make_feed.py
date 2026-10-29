# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Writes the update feed of Albireo for a new version.

    python3 make_feed.py <output feed.json> <version> <installer> <address of the installer> [<notes.json>]

The installer is the albireo-setup.exe of the version, the address the one it is published
at, a release asset of the repository for instance:

    https://github.com/<owner>/<repository>/releases/download/albireo-v<version>/albireo-setup-<version>.exe

notes.json holds the release notes by language, {"en": "...", "fr": "...", ...}; the
service shows the notes of the language of the user, else the English ones.  Without it,
the notes of the output feed are kept when that feed already describes the version, and
the notes otherwise read "Albireo <version>".

The feed is then signed with albireo-sign (sign <private key> feed.json), and both files
are published in dist/albireo of the repository.  The services that read it download the
installer in the background, check its size and its SHA-256, and install it at their next
start.  publish_release.py runs these steps after attaching the installer to its release.
"""

import hashlib
import json
import os
import re
import struct
import sys
from datetime import date


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def installer_version(path):
    """The file version of a Windows program, "1.2.3", read from its version resource, or
    None for a file without one.  A feed must announce the version its installer installs:
    the services would otherwise download it again after each installation."""
    with open(path, "rb") as f:
        head = f.read(4096)
        if len(head) < 64 or head[:2] != b"MZ":
            return None
        pe = struct.unpack_from("<I", head, 0x3C)[0]
        f.seek(pe)
        header = f.read(24 + 240)
        if len(header) < 24 or header[:4] != b"PE\0\0":
            return None
        sections, optional_size = struct.unpack_from("<H12xH", header, 6)
        optional = header[24:24 + optional_size]
        magic = struct.unpack_from("<H", optional, 0)[0]
        directories = 112 if magic == 0x20B else 96
        if len(optional) < directories + 24:
            return None
        resources_rva = struct.unpack_from("<I", optional, directories + 16)[0]
        if resources_rva == 0:
            return None
        f.seek(pe + 24 + optional_size)
        table = f.read(40 * sections)

        def offset_of(rva):
            for i in range(sections):
                size, address, raw_size, raw = struct.unpack_from("<4I", table, 40 * i + 8)
                if address <= rva < address + max(size, raw_size):
                    return rva - address + raw
            return None

        base = offset_of(resources_rva)
        if base is None:
            return None

        def entries(offset):
            f.seek(base + offset)
            named, ids = struct.unpack("<12xHH", f.read(16))
            return [struct.unpack("<II", f.read(8)) for _ in range(named + ids)]

        # The type RT_VERSION (16), then its first name and its first language.
        node = next((target for ident, target in entries(0) if ident == 16), None)
        for _ in range(2):
            if node is None or not node & 0x80000000:
                return None
            children = entries(node & 0x7FFFFFFF)
            node = children[0][1] if children else None
        if node is None or node & 0x80000000:
            return None
        f.seek(base + node)
        data_rva, data_size = struct.unpack("<II", f.read(8))
        start = offset_of(data_rva)
        if start is None:
            return None
        f.seek(start)
        data = f.read(min(data_size, 4096))
    at = data.find(struct.pack("<I", 0xFEEF04BD))
    if at < 0 or at + 16 > len(data):
        return None
    ms, ls = struct.unpack_from("<II", data, at + 8)
    return "%d.%d.%d" % (ms >> 16, ms & 0xFFFF, ls >> 16)


def read_notes(notes_path, output, version):
    """The notes of notes_path, else those the feed at output gives the same version, else
    a default text.  Raises ValueError for notes without an English text."""
    notes = None
    if notes_path:
        with open(notes_path, encoding="utf-8") as f:
            notes = json.load(f)
        if not isinstance(notes, dict) or "en" not in notes:
            raise ValueError("the notes need an English text (\"en\")")
    else:
        try:
            with open(output, encoding="utf-8") as f:
                current = json.load(f)["channels"]["stable"]
            if current.get("version") == version and "en" in current.get("notes", {}):
                notes = current["notes"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    return notes or {"en": "Albireo " + version}


def make_feed(version, installer, url, notes, size=None, sha256=None):
    """The feed announcing version, whose installer is the file installer published at url.
    size and sha256 spare reading the file again when the caller has them."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("the version must have the form 1.2.3")
    if not url.startswith("https://"):
        raise ValueError("the address of the installer must be an https address")
    built = installer_version(installer)
    if built is not None and built != version:
        raise ValueError("%s installs version %s, not %s" % (installer, built, version))
    return {
        "product": "Albireo",
        "channels": {
            "stable": {
                "version": version,
                "date": date.today().isoformat(),
                "notes": notes,
                "installer": {
                    "url": url,
                    "size": size if size is not None else os.path.getsize(installer),
                    "sha256": sha256 or sha256_of(installer),
                },
            }
        },
    }


def write_feed(output, feed):
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(feed, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    if len(sys.argv) not in (5, 6):
        print(__doc__)
        return 2
    output, version, installer, url = sys.argv[1:5]
    try:
        notes = read_notes(sys.argv[5] if len(sys.argv) == 6 else None, output, version)
        feed = make_feed(version, installer, url, notes)
    except (OSError, ValueError) as e:
        print(e)
        return 2
    write_feed(output, feed)
    print("%s: version %s, installer of %d bytes" % (output, version, feed["channels"]["stable"]["installer"]["size"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
