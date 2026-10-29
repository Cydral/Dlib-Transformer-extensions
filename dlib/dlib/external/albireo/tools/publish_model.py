# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Cuts a model into the parts of a catalog entry, for a host that limits the size of a file,
such as the assets of a GitHub release (2 GB each).

    python3 publish_model.py <model.gguf> <identifier> <address of the folder> <output folder> [--gzip]
                             [--part-mb N]

Each part is written to the output folder as <identifier>.gguf.partNN, compressed with gzip
when --gzip is given (<identifier>.gguf.partNN.gz), and is expected at
<address of the folder>/<name of the part> once published.  The program prints the
"size", "sha256" and "files" of the catalog entry, to complete with the name, the
description, the roles and the other fields of make_catalog.py, which takes an entry with
its "files" as it is.

The service downloads the parts in parallel, inflates the compressed ones as they arrive,
writes each at its place in a single file and checks every part, then the whole model,
against these hashes.
"""

import gzip
import hashlib
import json
import os
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    use_gzip = "--gzip" in flags
    part_mb = 1900
    for f in flags:
        if f.startswith("--part-mb="):
            part_mb = int(f.split("=", 1)[1])
    if len(args) != 4 or any(f != "--gzip" and not f.startswith("--part-mb=") for f in flags):
        print(__doc__)
        return 2
    model, ident, base, out = args
    base = base.rstrip("/")
    os.makedirs(out, exist_ok=True)
    part_bytes = part_mb << 20
    whole = hashlib.sha256()
    files = []
    total = 0
    with open(model, "rb") as src:
        index = 1
        while True:
            data = src.read(part_bytes)
            if not data:
                break
            whole.update(data)
            name = "%s.gguf.part%02d" % (ident, index)
            entry = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if use_gzip:
                name += ".gz"
                packed = gzip.compress(data, compresslevel=9, mtime=0)
                with open(os.path.join(out, name), "wb") as f:
                    f.write(packed)
                entry["compression"] = "gzip"
                entry["compressed_size"] = len(packed)
            else:
                with open(os.path.join(out, name), "wb") as f:
                    f.write(data)
            entry = dict({"url": base + "/" + name}, **entry)
            files.append(entry)
            total += len(data)
            print("%s: %d bytes%s" % (name, len(data),
                  " (%d compressed)" % entry["compressed_size"] if use_gzip else ""), file=sys.stderr)
            index += 1
    print(json.dumps({"id": ident, "size": total, "sha256": whole.hexdigest(), "files": files}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
