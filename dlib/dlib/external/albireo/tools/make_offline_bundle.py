# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Prepares the offline bundle of Albireo, which installs or updates a computer without network
from a disk or a memory card.

    python3 make_offline_bundle.py <output folder> [--models <id,id,...> | --all]
                                   [--installer <albireo-setup-online.exe>]
                                   [--complete <albireo-setup.exe>] [--from <folder>]...
                                   [--dist <folder>] [--sign <albireo-sign>]

The bundle holds the signed catalog and the signed update feed of dist/albireo, the installer
the feed announces, and in its models folder the models chosen: by default the default
language model and the default embedding model of the catalog, which the search in the
documents needs.  The installer is taken from --installer when given, else downloaded from
the address of the feed.  --complete adds the complete installer of the version, which
carries the models that come with Albireo: the default models are then left out, unless
--models or --all name models; each model is taken from the folders of --from when one of their
files has its size and its SHA-256, else downloaded from the addresses of the catalog.
Every file is checked against the hashes of the signed files before it enters the bundle,
and the signatures are checked with albireo-sign when it is found.

The bundle is used in one of three ways:
- albireo-setup-<version>.exe started from the bundle installs Albireo, and the service
  installs the catalog and the models at its first start;
- the Import button of the Models window, given the folder of the bundle, installs the
  catalog, the models and the update of an installed copy;
- the content of the bundle copied into the import folder of the data folder
  (C:\\ProgramData\\Albireo\\import) is installed by the service within a minute.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_feed  # noqa: E402
import publish_release  # noqa: E402

README = """Albireo {version}, offline bundle of {date}

Content
- catalog.json, catalog.json.sig: the signed catalog of the models;
- feed.json, feed.json.sig, {installer}: the signed update and its installer;
- models: {models}.{complete_en}

New installation: start {first} from this folder. The service installs the catalog
and the models at its first start.

Installed computer: in the Models window, Import, then choose this folder; or copy the
content of this folder into C:\\ProgramData\\Albireo\\import, which the service reads within
a minute. The update is installed at the next start of the service.

----

Albireo {version}, paquet hors ligne du {date}

Contenu
- catalog.json, catalog.json.sig : le catalogue signé des modèles ;
- feed.json, feed.json.sig, {installer} : la mise à jour signée et son installateur ;
- models : {models}.{complete_fr}

Nouvelle installation : lancer {first} depuis ce dossier. Le service installe le
catalogue et les modèles à son premier démarrage.

Ordinateur déjà équipé : dans la fenêtre Modèles, Importer, puis choisir ce dossier ; ou
copier le contenu de ce dossier dans C:\\ProgramData\\Albireo\\import, que le service lit dans
la minute. La mise à jour s'installe au prochain démarrage du service.
"""


def download(url, out, expected_size=None, gzip_part=False, label="", whole=None):
    """Writes the content of url to the open file out, inflated when gzip_part, and returns
    the SHA-256 and the size of what was written; whole also receives the content."""
    h = hashlib.sha256()
    written = 0
    shown = 0.0
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS) if gzip_part else None
    req = urllib.request.Request(url, headers={"User-Agent": "albireo-bundle"})
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            if inflate:
                block = inflate.decompress(block)
            out.write(block)
            h.update(block)
            if whole:
                whole.update(block)
            written += len(block)
            if expected_size and time.monotonic() - shown >= 2:
                shown = time.monotonic()
                print("\r  %s: %d of %d MB" % (label, written >> 20, expected_size >> 20), end="", file=sys.stderr)
        if inflate:
            tail = inflate.flush()
            out.write(tail)
            h.update(tail)
            if whole:
                whole.update(tail)
            written += len(tail)
    if expected_size:
        print("\r  %s: %d of %d MB" % (label, written >> 20, expected_size >> 20), file=sys.stderr)
    return h.hexdigest(), written


def local_files(folders):
    by_size = {}
    for folder in folders:
        for dirpath, _, names in os.walk(folder):
            for n in names:
                if n.endswith(".gguf"):
                    path = os.path.join(dirpath, n)
                    by_size.setdefault(os.path.getsize(path), []).append(path)
    return by_size


def bundle_model(entry, models_dir, by_size):
    files = entry["files"]
    name = os.path.basename(files[0]["url"]) if len(files) == 1 else entry["id"] + ".gguf"
    target = os.path.join(models_dir, name)
    if os.path.isfile(target) and os.path.getsize(target) == entry["size"] \
            and make_feed.sha256_of(target) == entry["sha256"]:
        print("%s: already in the bundle" % entry["id"])
        return target
    for path in by_size.get(entry["size"], []):
        if make_feed.sha256_of(path) == entry["sha256"]:
            shutil.copyfile(path, target + ".partial")
            os.replace(target + ".partial", target)
            print("%s: copied from %s" % (entry["id"], path))
            return target
    whole = hashlib.sha256()
    with open(target + ".partial", "wb") as out:
        for i, part in enumerate(files):
            label = "%s %d/%d" % (entry["id"], i + 1, len(files))
            sha, size = download(part["url"], out, part["size"], part.get("compression") == "gzip", label, whole)
            if sha != part["sha256"] or size != part["size"]:
                raise RuntimeError("%s: the part %s does not match the catalog" % (entry["id"], part["url"]))
    if whole.hexdigest() != entry["sha256"]:
        os.remove(target + ".partial")
        raise RuntimeError("%s: the model does not match the catalog" % entry["id"])
    os.replace(target + ".partial", target)
    print("%s: downloaded" % entry["id"])
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--models", help="identifiers of the catalog, separated by commas")
    which.add_argument("--all", action="store_true")
    p.add_argument("--installer")
    p.add_argument("--complete")
    p.add_argument("--from", dest="sources", action="append", default=[])
    p.add_argument("--dist", default=os.path.join(publish_release.ROOT, "dist", "albireo"))
    p.add_argument("--sign")
    a = p.parse_args()

    try:
        os.makedirs(os.path.join(a.output, "models"), exist_ok=True)
        try:
            signer = publish_release.find_signer(a.sign)
        except RuntimeError:
            signer = None
            print("albireo-sign not found: the signatures are checked by the service only")
        for name in ("catalog.json", "feed.json"):
            source = os.path.join(a.dist, name)
            if signer:
                subprocess.run([signer, "verify", source], check=True, stdout=subprocess.DEVNULL)
            for suffix in ("", ".sig"):
                shutil.copyfile(source + suffix, os.path.join(a.output, name + suffix))

        with open(os.path.join(a.dist, "catalog.json"), encoding="utf-8") as f:
            catalog = json.load(f)
        with open(os.path.join(a.dist, "feed.json"), encoding="utf-8") as f:
            stable = json.load(f)["channels"]["stable"]

        installer = stable["installer"]
        installer_name = os.path.basename(installer["url"])
        if not installer.get("size"):
            raise RuntimeError("the feed of %s announces no installer yet: publish the version first "
                               "(publish_release.py)" % a.dist)
        target = os.path.join(a.output, installer_name)
        if a.installer:
            if os.path.getsize(a.installer) != installer["size"] \
                    or make_feed.sha256_of(a.installer) != installer["sha256"]:
                raise RuntimeError("%s is not the installer the feed announces" % a.installer)
            shutil.copyfile(a.installer, target)
        elif not (os.path.isfile(target) and make_feed.sha256_of(target) == installer["sha256"]):
            with open(target + ".partial", "wb") as out:
                sha, size = download(installer["url"], out, installer["size"], label=installer_name)
            if sha != installer["sha256"] or size != installer["size"]:
                os.remove(target + ".partial")
                raise RuntimeError("the installer downloaded does not match the feed")
            os.replace(target + ".partial", target)
        print("%s: version %s" % (installer_name, stable["version"]))
        complete_name = None
        if a.complete:
            complete_name = "albireo-setup-complete-%s.exe" % stable["version"]
            shutil.copyfile(a.complete, os.path.join(a.output, complete_name))
            print("%s: the complete installer, with the models that come with Albireo" % complete_name)

        entries = catalog["models"]
        if a.all:
            chosen = entries
        elif a.models:
            wanted = [m.strip() for m in a.models.split(",") if m.strip()]
            unknown = [m for m in wanted if m not in {e["id"] for e in entries}]
            if unknown:
                raise RuntimeError("not in the catalog: " + ", ".join(unknown))
            chosen = [e for e in entries if e["id"] in wanted]
        elif complete_name:
            chosen = []
        else:
            chosen = [e for e in entries if e.get("default")]
        by_size = local_files(a.sources)
        total = 0
        for e in chosen:
            bundle_model(e, os.path.join(a.output, "models"), by_size)
            total += e["size"]

        with open(os.path.join(a.output, "README.txt"), "w", encoding="utf-8", newline="\r\n") as f:
            complete_en = ("\n- %s: the complete installer, with the models that come with Albireo."
                           % complete_name) if complete_name else ""
            complete_fr = ("\n- %s : l'installateur complet, avec les modèles livrés avec Albireo."
                           % complete_name) if complete_name else ""
            f.write(README.format(version=stable["version"], date=catalog["updated"][:10], installer=installer_name,
                                  models=", ".join(e["id"] for e in chosen) or "-",
                                  first=complete_name or installer_name, complete_en=complete_en,
                                  complete_fr=complete_fr))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    print("bundle %s: installer %s, %d models, %.2f GB of models"
          % (a.output, stable["version"], len(chosen), total / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
