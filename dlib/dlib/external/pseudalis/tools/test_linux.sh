#!/bin/bash
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Builds and tests Pseudalis on Linux, where the service carries its engine and its drives work
# as folders without letters, and prints a summary, one line per step.
#
#   bash dlib/dlib/external/pseudalis/tools/test_linux.sh [build folder]
#
# Default build folder: build-pseudalis-linux at the root of the repository.  The tests of the
# engine and of the host run first; the service then runs in a data folder of its own, on port
# 18444 (PSEUDALIS_TEST_PORT), anonymizes and restores a text and a CSV file through its
# interface, then files and a folder through its drives, and stops.  The exit code is the
# number of failed steps.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
BUILD=${1:-$ROOT/build-pseudalis-linux}
PORT=${PSEUDALIS_TEST_PORT:-18444}
WORK=$(mktemp -d)
RESULTS=()
FAILED=0

record() { # step result detail
  RESULTS+=("$(printf '%-26s %-5s %s' "$1" "$2" "$3")")
  [ "$2" = FAIL ] && FAILED=$((FAILED + 1))
}

# Runs a command, then checks that its output holds a pattern.
step() { # name pattern command...
  local name=$1 pattern=$2
  shift 2
  local out rc
  out=$("$@" 2>&1)
  rc=$?
  if [ $rc -eq 0 ] && printf '%s\n' "$out" | grep -E -q "$pattern"; then
    record "$name" PASS "$(printf '%s\n' "$out" | grep -E -o -m1 ".{0,40}$pattern.{0,30}" | head -1)"
  else
    printf '%s\n' "$out" | tail -40 > "$WORK/${name// /_}.log"
    record "$name" FAIL "exit $rc, output in $WORK/${name// /_}.log"
  fi
}

step "configuration" "Generating done|Build files" cmake -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
step "build" "." cmake --build "$BUILD" --parallel "$(nproc)"
step "engine tests" "all tests passed" "$BUILD/bin/test_engine"
step "host tests" "all tests passed" "$BUILD/bin/test_host"

HOME_DIR=$WORK/home
mkdir -p "$HOME_DIR"
cat > "$HOME_DIR/settings.json" <<EOF
{"port": $PORT, "drive_letter": "A", "restore_letter": "R", "check_updates": false, "albireo": "off",
 "public_link": ""}
EOF
PSEUDALIS_HOME=$HOME_DIR "$BUILD/bin/pseudalis-service" --console > "$WORK/service.log" 2>&1 &
SERVICE=$!
CA=$HOME_DIR/certificates/pseudalis-authority.crt
URL=https://localhost:$PORT
api() { curl -sS --cacert "$CA" "$@"; }
ready=0
for _ in $(seq 1 60); do
  if [ -f "$CA" ] && api "$URL/v1/health" 2>/dev/null | grep -q '"ok"'; then ready=1; break; fi
  sleep 1
done
if [ $ready -eq 1 ]; then
  record "service start" PASS "$URL"
  TEXT='Jean Dupont habite 12 rue de la Paix, 75002 Paris, jean.dupont@exemple.fr'
  step "text anonymized" '\[\[PERSON:' api -X POST -H "Content-Type: application/json" \
    --data "{\"text\": \"$TEXT\"}" "$URL/v1/text/anonymize"
  PROTECTED=$(api -X POST -H "Content-Type: application/json" --data "{\"text\": \"$TEXT\"}" "$URL/v1/text/anonymize" |
    python3 -c 'import json, sys; print(json.dumps({"text": json.load(sys.stdin)["text"]}))')
  step "text restored" "jean.dupont@exemple.fr" api -X POST -H "Content-Type: application/json" \
    --data "$PROTECTED" "$URL/v1/text/restore"
  printf 'nom;prenom;email;telephone\nDupont;Jean;jean.dupont@exemple.fr;06 12 34 56 78\n' > "$WORK/sample.csv"
  api -X POST --data-binary @"$WORK/sample.csv" -o "$WORK/sample.anon.csv" "$URL/v1/anonymize?name=sample.csv"
  step "file anonymized" '\[\[EMAIL:' cat "$WORK/sample.anon.csv"
  api -X POST --data-binary @"$WORK/sample.anon.csv" -o "$WORK/sample.back.csv" "$URL/v1/restore?name=sample.csv"
  if cmp -s "$WORK/sample.csv" "$WORK/sample.back.csv"; then record "file restored" PASS "identical to the original"
  else record "file restored" FAIL "the restored file differs, in $WORK"
  fi
  # The name of a file takes substitutes, without tags.
  api -X POST --data-binary @"$WORK/sample.csv" -D "$WORK/name.headers" -o /dev/null \
    "$URL/v1/anonymize?name=jean.dupont%40exemple.fr.csv"
  if grep -i '^x-pseudalis-name:' "$WORK/name.headers" | grep -q -v -i -e dupont -e '\[\['; then
    record "name anonymized" PASS "$(grep -i '^x-pseudalis-name:' "$WORK/name.headers" | tr -d '\r')"
  else record "name anonymized" FAIL "$(grep -i '^x-pseudalis-name:' "$WORK/name.headers" | tr -d '\r')"
  fi

  # The drives: a document anonymized under a name without personal data, a file of another kind
  # moved out, a folder renamed once its files rest; then the restoration of the file and the folder.
  DRIVE=$HOME_DIR/drive
  RESTORE=$HOME_DIR/restore
  wait_for() { # seconds command...
    local n=$1
    shift
    for _ in $(seq 1 "$n"); do "$@" && return 0; sleep 1; done
    return 1
  }
  printf 'Bonjour,\nJe suis Jean Dupont, joignable au 06 12 34 56 78.\n' > "$WORK/lettre.txt"
  printf 'Note de Marie Curie, 12 rue des Lilas, 75005 Paris.\n' > "$WORK/note.txt"
  cp "$WORK/lettre.txt" "$DRIVE/Lettre jean.dupont@exemple.fr.txt"
  mkdir -p "$DRIVE/Dossier jean.dupont@exemple.fr"
  cp "$WORK/note.txt" "$DRIVE/Dossier jean.dupont@exemple.fr/note.txt"
  printf 'binary\0' > "$DRIVE/outil.bin"
  drive_done() {
    [ -z "$(find "$DRIVE" -iname '*dupont*')" ] && [ -f "$HOME_DIR/refused/files/outil.bin" ] &&
      [ -n "$(find "$DRIVE" -mindepth 2 -name note.txt)" ]
  }
  if wait_for 60 drive_done; then record "drive anonymized" PASS "names, content, the file of another kind out"
  else record "drive anonymized" FAIL "$(find "$DRIVE" "$HOME_DIR/refused" | tr '\n' ' ')"
  fi
  ANON=$(find "$DRIVE" -maxdepth 1 -type f -name 'Lettre*' | head -1)
  step "drive content" '\[\[PERSON:' cat "$ANON"
  cp "$ANON" "$RESTORE/"
  cp -r "$DRIVE"/Dossier* "$RESTORE/"
  restore_done() {
    cmp -s "$WORK/lettre.txt" "$RESTORE/Lettre jean.dupont@exemple.fr.txt" &&
      cmp -s "$WORK/note.txt" "$RESTORE/Dossier jean.dupont@exemple.fr/note.txt"
  }
  if wait_for 60 restore_done; then record "drive restored" PASS "the original names and contents"
  else record "drive restored" FAIL "$(find "$RESTORE" | tr '\n' ' ')"
  fi
else
  tail -20 "$WORK/service.log" > "$WORK/service_start.log"
  record "service start" FAIL "no answer on $URL, log in $WORK/service_start.log"
fi
kill "$SERVICE" 2>/dev/null
wait "$SERVICE" 2>/dev/null

printf '%s\n' "${RESULTS[@]}"
echo "$FAILED failed, work folder $WORK"
exit $FAILED
