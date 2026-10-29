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
start.
"""

import hashlib
import json
import os
import re
import sys
from datetime import date


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) not in (5, 6):
        print(__doc__)
        return 2
    output, version, installer, url = sys.argv[1:5]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print("the version must have the form 1.2.3")
        return 2
    if not url.startswith("https://"):
        print("the address of the installer must be an https address")
        return 2
    notes = None
    if len(sys.argv) == 6:
        with open(sys.argv[5], encoding="utf-8") as f:
            notes = json.load(f)
        if "en" not in notes:
            print("the notes need an English text (\"en\")")
            return 2
    else:
        try:
            with open(output, encoding="utf-8") as f:
                current = json.load(f)["channels"]["stable"]
            if current.get("version") == version and "en" in current.get("notes", {}):
                notes = current["notes"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    if not notes:
        notes = {"en": "Albireo " + version}
    feed = {
        "product": "Albireo",
        "channels": {
            "stable": {
                "version": version,
                "date": date.today().isoformat(),
                "notes": notes,
                "installer": {
                    "url": url,
                    "size": os.path.getsize(installer),
                    "sha256": sha256_of(installer),
                },
            }
        },
    }
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(feed, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("%s: version %s, installer of %d bytes" % (output, version, feed["channels"]["stable"]["installer"]["size"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
