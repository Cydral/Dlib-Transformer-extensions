# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Cuts a model into the parts of a catalog entry, for a host that limits the size of a file,
such as the releases of GitHub (2 GiB per file), and publishes them on a release.

    python3 publish_model.py <model.gguf> <identifier> <output folder>
                             (--release <tag> [--repo <owner/name>] | --base <address>)
                             [--gzip] [--part-mb N]

Each part is written to the output folder as <identifier>.gguf.partNN, compressed with gzip
when --gzip is given (<identifier>.gguf.partNN.gz).  With --release, the parts are attached
to that release of the repository (Cydral/Dlib-Transformer-extensions by default), created
when it does not exist; with --base, they are expected at <address>/<name of the part> once
published by other means.

The program prints the catalog entry of the model: its architecture, number of parameters
and trained context read from the file, its size and SHA-256, and the "files" of its parts.
The entry is completed with the name, the description, the roles and the other fields of
make_catalog.py, which takes an entry with its "files" as it is, then the catalog is
written and signed again.

The service downloads the parts in parallel, inflates the compressed ones as they arrive,
writes each at its place in a single file and checks every part, then the whole model,
against these hashes.
"""

import argparse
import gzip
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import github_release  # noqa: E402

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_info(path):
    """The metadata of a GGUF file, its arrays left out, and its number of parameters."""
    meta = {}
    parameters = 0
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        version, tensors, count = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            raise ValueError("%s is a GGUF file of version %d, too old" % (path, version))

        def text():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                item, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    value(item)
                return None
            fmt = SCALARS[kind]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(count):
            key = text()
            kind = struct.unpack("<I", f.read(4))[0]
            meta[key] = value(kind)
        for _ in range(tensors):
            text()
            dims = struct.unpack("<I", f.read(4))[0]
            n = 1
            for d in struct.unpack("<%dQ" % dims, f.read(8 * dims)):
                n *= d
            f.read(12)
            parameters += n
    return meta, parameters


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("identifier")
    p.add_argument("output")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--release")
    where.add_argument("--base")
    p.add_argument("--repo", default=github_release.DEFAULT_REPOSITORY)
    p.add_argument("--gzip", action="store_true")
    p.add_argument("--part-mb", type=int, default=1900)
    a = p.parse_args()
    if not 1 <= a.part_mb <= 2000:
        print("--part-mb lies between 1 and 2000")
        return 2

    meta, parameters = gguf_info(a.model)
    arch = meta.get("general.architecture", "")
    if a.release:
        base = "https://github.com/%s/releases/download/%s" % (a.repo, a.release)
    else:
        base = a.base.rstrip("/")
    os.makedirs(a.output, exist_ok=True)
    part_bytes = a.part_mb << 20
    whole = hashlib.sha256()
    files = []
    paths = []
    total = 0
    with open(a.model, "rb") as src:
        index = 1
        while True:
            data = src.read(part_bytes)
            if not data:
                break
            whole.update(data)
            name = "%s.gguf.part%02d" % (a.identifier, index)
            entry = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if a.gzip:
                name += ".gz"
                data = gzip.compress(data, compresslevel=9, mtime=0)
                entry["compression"] = "gzip"
                entry["compressed_size"] = len(data)
            with open(os.path.join(a.output, name), "wb") as f:
                f.write(data)
            files.append(dict({"url": base + "/" + name}, **entry))
            paths.append(os.path.join(a.output, name))
            total += entry["size"]
            print("%s: %d bytes%s" % (name, entry["size"],
                  " (%d compressed)" % entry["compressed_size"] if a.gzip else ""), file=sys.stderr)
            index += 1

    if a.release:
        try:
            repo = github_release.repository(a.repo)
            release = repo.ensure_release(a.release, "Albireo models",
                                          "Models of the catalog of Albireo published outside the Hugging Face hub.",
                                          latest=False)
            for path in paths:
                repo.attach(release, path)
        except (OSError, RuntimeError) as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        try:
            if not github_release.read_back(files[0]["url"], paths[0]):
                print("WARNING: %s serves another file" % files[0]["url"], file=sys.stderr)
        except OSError as e:
            print("WARNING: %s cannot be read (%s); a private repository serves its files to signed-in users "
                  "only" % (files[0]["url"], e), file=sys.stderr)

    entry = {
        "id": a.identifier,
        "arch": arch,
        "parameters": parameters,
        "context": meta.get(arch + ".context_length", 0),
        "source": base,
        "size": total,
        "sha256": whole.hexdigest(),
        "files": files,
    }
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
