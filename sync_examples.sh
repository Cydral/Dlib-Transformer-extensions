#!/usr/bin/env bash
#
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Keeps the two copies of the examples from drifting apart.
#
# WHY THERE ARE TWO
#
# examples/ is what a visitor reads: this project's files and the guide that explains them,
# with none of the upstream library's own examples in the way. dlib/examples/ is what CMake
# builds: upstream's tree with this project's files added to it.
#
# The arrangement is deliberate and the risk it carries is obvious. Nothing connects the two
# directories, so a file edited in one and forgotten in the other leaves the repository
# holding two versions of the same example, and whichever a reader opens is a coin toss. The
# divergence is silent by construction: both copies compile, both look complete, and the
# difference only surfaces when someone builds what they read and gets different behaviour.
#
# Run this before committing. The check costs a second and the alternative costs an
# afternoon of confusion.
#
# Usage:
#   ./sync_examples.sh              report what differs, and say nothing if nothing does
#   ./sync_examples.sh --to-build   copy examples/ over dlib/examples/
#   ./sync_examples.sh --to-showcase copy dlib/examples/ over examples/

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
showcase="$root/examples"
build="$root/dlib/examples"
mode="${1:---check}"

if [ ! -d "$showcase" ] || [ ! -d "$build" ]; then
    echo "Error: run this from the repository root; $showcase or $build is missing." >&2
    exit 2
fi

# Only this project's files are compared, and only those Git would keep.
#
# The upstream examples living beside ours in dlib/examples/ are not ours to mirror, and
# Readme.md belongs to the showcase alone. Beyond that, anything .gitignore excludes is
# excluded here too: a model archive or a generated header dropped into one directory is not
# a divergence to report, and flagging it would train the reader to ignore this script's
# output, which is the one thing it cannot afford.
ignored() {
    git -C "$root" check-ignore -q "$1" 2>/dev/null
}

# Files under a directory, relative to it, ours and not ignored.
list_ours() {
    local dir="$1" rel
    [ -d "$dir" ] || return 0
    ( cd "$dir" && find . -type f 2>/dev/null | sed 's|^\./||' ) | grep -E '^slm_' | sort |
    while IFS= read -r rel; do
        ignored "$dir/$rel" || printf '%s\n' "$rel"
    done
}

names=$( { list_ours "$showcase"; list_ours "$build"; } | sort -u )

differ=0
missing_build=0
missing_showcase=0

for name in $names; do
    a="$showcase/$name"
    b="$build/$name"

    if [ ! -f "$a" ]; then
        echo "  only in dlib/examples : $name"
        missing_showcase=$((missing_showcase + 1))
    elif [ ! -f "$b" ]; then
        echo "  only in examples      : $name"
        missing_build=$((missing_build + 1))
    elif ! cmp -s "$a" "$b"; then
        echo "  differs               : $name"
        differ=$((differ + 1))
    fi
done

total=$((differ + missing_build + missing_showcase))

case "$mode" in
    --check)
        if [ "$total" -eq 0 ]; then
            echo "The two copies agree."
            exit 0
        fi
        echo
        echo "$total item(s) out of step. Copy one way or the other:"
        echo "  ./sync_examples.sh --to-build      (examples/ wins)"
        echo "  ./sync_examples.sh --to-showcase   (dlib/examples/ wins)"
        exit 1
        ;;

    --to-build|--to-showcase)
        if [ "$mode" = "--to-build" ]; then
            from="$showcase"; to="$build"; label="examples -> dlib/examples"
        else
            from="$build"; to="$showcase"; label="dlib/examples -> examples"
        fi
        echo
        echo "Copying $label"
        for name in $names; do
            [ -e "$from/$name" ] || continue
            mkdir -p "$to/$(dirname "$name")"
            cp "$from/$name" "$to/$name"
            echo "  $name"
        done
        echo "Done. Verify with ./sync_examples.sh before committing."
        ;;

    *)
        echo "Unknown option: $mode" >&2
        echo "Use --check, --to-build or --to-showcase." >&2
        exit 2
        ;;
esac
