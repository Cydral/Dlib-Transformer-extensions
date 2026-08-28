#!/usr/bin/env python3
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Synthetic test image for the vision encoder, written as an uncompressed 24-bit BMP.
#
# BMP because that is what the C++ side can read with no build option at all: dlib decodes
# JPEG and PNG only when it was configured against those libraries, and its loader knows
# nothing of PPM, but BMP and its own DNG format are always available. One less thing
# between a fresh checkout and a first result.
#
# The picture is structured on purpose rather than random. A vision tower fed noise still
# produces numbers, and there is then nothing to tell a working encoder from a broken one.
# A horizon, two solid shapes and a striped band give the patch grid something different in
# every quadrant, so a stage that has quietly stopped depending on its input shows up as a
# uniform output rather than as plausible noise.
#
# Usage:
#   make_test_image.py photo.bmp        512 by 512, the size SmolVLM expects
#   make_test_image.py small.bmp 256    any side; the encoder resizes anyway

import struct
import sys


def draw(width, height):
    """Rows of BGR bytes, top row first."""
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            # Sky fading into ground, so no two horizontal bands are alike.
            r = 60 + y * 120 // height
            g = 120 + y * 80 // height
            b = 220 - y * 150 // height
            # A solid red rectangle, upper left.
            if 60 <= x < 200 and 80 <= y < 220:
                r, g, b = 200, 40, 40
            # A yellow disc, upper right.
            if (x - 360) ** 2 + (y - 140) ** 2 < 70 ** 2:
                r, g, b = 245, 220, 60
            # A dark band along the bottom, cut by white diagonals: high spatial
            # frequency, which is where a patch embedding of the wrong stride shows.
            if y > 400:
                r, g, b = (230, 230, 230) if ((x + y) // 24) % 2 == 0 else (40, 90, 40)
            row += bytes((b & 255, g & 255, r & 255))    # BMP stores blue first
        row += b"\0" * ((4 - len(row) % 4) % 4)          # every row padded to 4 bytes
        rows.append(bytes(row))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "photo.bmp"
    side = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    rows = draw(side, side)
    pixels = b"".join(reversed(rows))                    # BMP stores rows bottom up
    offset = 14 + 40                                     # file header + info header

    header = b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
    info = struct.pack("<IiiHHIIiiII", 40, side, side, 1, 24, 0, len(pixels),
                       2835, 2835, 0, 0)
    with open(path, "wb") as f:
        f.write(header + info + pixels)
    print(f"wrote {side}x{side} BMP to {path} ({offset + len(pixels)} bytes)")


if __name__ == "__main__":
    main()
