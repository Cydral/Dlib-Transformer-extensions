#!/bin/bash
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Checks the tools that build and test Albireo on Linux, and installs those that are missing with
# apt-get (Debian, Ubuntu) or dnf (Fedora), through sudo unless the user is root:
#
#   bash dlib/dlib/external/albireo/tools/prerequisites_linux.sh [--dry-run]
#
# The tools: CMake 3.21 or later, a C++14 compiler (CXX, else c++, g++ or clang++), make or
# ninja, Python 3.8 or later, git, and curl, which tools/test_linux.sh uses.  The script prints one
# line per tool, says which packages it installs and runs the command; --dry-run prints that
# command without running it.  The exit code is 0 when every tool is there, 1 otherwise.

set -u
DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac

# Whether the version $1 is $2 or later.
at_least() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# The first version number of a text.
version_of() { grep -o -m1 '[0-9][0-9]*\.[0-9][0-9.]*' | head -1; }

# The checks: each prints its line and returns 0 when the tool is there.
line() { printf '  %-20s %-8s %s\n' "$1" "$2" "$3"; }

check_cmake() {
  local v
  v=$(cmake --version 2> /dev/null | version_of)
  if [ -n "$v" ] && at_least "$v" 3.21; then line "CMake 3.21+" found "$v, $(command -v cmake)"; return 0; fi
  line "CMake 3.21+" missing "${v:+version $v, older than 3.21}"
  return 1
}

check_compiler() {
  local c
  for c in ${CXX:-} c++ g++ clang++; do
    command -v "$c" > /dev/null 2>&1 || continue
    if printf '#include <memory>\nint main() { return *std::make_unique<int>(0); }\n' |
        "$c" -std=c++14 -x c++ -o /dev/null - > /dev/null 2>&1; then
      line "C++14 compiler" found "$c $("$c" -dumpversion 2> /dev/null), $(command -v "$c")"
      return 0
    fi
  done
  line "C++14 compiler" missing ""
  return 1
}

check_make() {
  local m
  for m in make ninja; do
    if command -v "$m" > /dev/null 2>&1; then
      line "make or ninja" found "$m $("$m" --version 2> /dev/null | version_of), $(command -v "$m")"
      return 0
    fi
  done
  line "make or ninja" missing ""
  return 1
}

check_python() {
  local v
  v=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2> /dev/null)
  if [ -n "$v" ] && at_least "$v" 3.8; then line "Python 3.8+" found "$v, $(command -v python3)"; return 0; fi
  line "Python 3.8+" missing "${v:+version $v, older than 3.8}"
  return 1
}

check_tool() { # name of the line, program
  if command -v "$2" > /dev/null 2>&1; then
    line "$1" found "$("$2" --version 2> /dev/null | version_of), $(command -v "$2")"
    return 0
  fi
  line "$1" missing ""
  return 1
}

# The packages of the missing tools, named for the package manager of $MANAGER.
check_all() {
  PACKAGES=""
  check_cmake || PACKAGES="$PACKAGES cmake"
  if ! check_compiler; then
    if [ "$MANAGER" = dnf ]; then PACKAGES="$PACKAGES gcc-c++"; else PACKAGES="$PACKAGES g++"; fi
  fi
  check_make || PACKAGES="$PACKAGES make"
  check_python || PACKAGES="$PACKAGES python3"
  check_tool git git || PACKAGES="$PACKAGES git"
  check_tool curl curl || PACKAGES="$PACKAGES curl"
  PACKAGES=${PACKAGES# }
}

MANAGER=""
if command -v apt-get > /dev/null 2>&1; then MANAGER=apt-get
elif command -v dnf > /dev/null 2>&1; then MANAGER=dnf
fi
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo "

echo "== checking the prerequisites"
check_all
if [ -z "$PACKAGES" ]; then
  echo "== every tool is there"
  exit 0
fi
if [ -z "$MANAGER" ]; then
  echo "== missing packages: $PACKAGES; this system has neither apt-get nor dnf, install them with its package manager"
  exit 1
fi
if [ "$MANAGER" = apt-get ]; then
  COMMAND="${SUDO}apt-get update && ${SUDO}env DEBIAN_FRONTEND=noninteractive apt-get install -y $PACKAGES"
else
  COMMAND="${SUDO}dnf install -y $PACKAGES"
fi
if [ $DRY_RUN -eq 1 ]; then
  echo "== would install with $MANAGER: $PACKAGES"
  echo "   $COMMAND"
  exit 1
fi
if [ -n "$SUDO" ] && ! command -v sudo > /dev/null 2>&1; then
  echo "== the installation of $PACKAGES needs root, and this system has no sudo: run as root"
  echo "   ${COMMAND//sudo /}"
  exit 1
fi
echo "== installing with $MANAGER${SUDO:+, through sudo}: $PACKAGES"
echo "   $COMMAND"
if ! sh -c "$COMMAND"; then
  echo "== the installation failed: see the messages of $MANAGER above"
  exit 1
fi
echo "== checking again"
check_all
if [ -n "$PACKAGES" ]; then
  case " $PACKAGES " in
    *" cmake "*)
      echo "== the cmake of this distribution is older than 3.21: python3 -m pip install --user 'cmake>=3.21'"
      echo "   installs a recent one in ~/.local/bin, which the PATH must name before /usr/bin" ;;
  esac
  echo "== still missing: $PACKAGES"
  exit 1
fi
echo "== every tool is there"
exit 0
