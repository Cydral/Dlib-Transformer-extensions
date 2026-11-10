#!/bin/bash
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Builds and tests Pseudalis on Linux, where the service carries its engine and its drive works
# as a folder without letter, and prints a summary, one line per step.
#
#   bash dlib/dlib/external/pseudalis/tools/test_linux.sh [build folder]
#
# Default build folder: build-pseudalis-linux at the root of the repository.  The tests of the
# engine and of the host run first; the service then runs in a data folder of its own, on port
# 18444 (PSEUDALIS_TEST_PORT), anonymizes and restores through its interface a text, a CSV file,
# a name and the archive of a folder as the web application sends it, then files and a folder
# through its drive, whose originals come back through the interface, an archive extracted on the
# drive, an archive whose original was forgotten coming back and the copies of a file, saves the
# settings as the web application does, imposes terms, makes room for newer originals, restores from
# their tags an archive whose picture was forgotten and a text written elsewhere from anonymized
# values, protects files, a text and the drive without tags, restores files protected twice,
# imposes columns on a CSV file, an IP column of both forms among them, keeps the encoding of a
# Windows-1252 file, and stops.
# The exit code is the number of failed steps.

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
{"port": $PORT, "drive_letter": "A", "check_updates": false, "albireo": "off"}
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
  # A text of no kind is protected as a text; a binary file, or a text with encoded parts, is refused.
  printf 'Journal de Jean Dupont, jean.dupont@exemple.fr\n' > "$WORK/journal.log2"
  step "text of no kind" '\[\[EMAIL:' api -X POST --data-binary @"$WORK/journal.log2" \
    "$URL/v1/anonymize?name=journal.log2"
  printf 'Jean Dupont\0\001\002' > "$WORK/donnees.bin"
  step "binary of no kind" '"code":"unsupported"' api -X POST --data-binary @"$WORK/donnees.bin" \
    "$URL/v1/anonymize?name=donnees.bin"
  printf 'From: Jean Dupont\r\nContent-Transfer-Encoding: base64\r\n\r\nSmVhbiBEdXBvbnQ=\r\n' > "$WORK/message.eml"
  step "encoded of no kind" '"code":"encoded_content"' api -X POST --data-binary @"$WORK/message.eml" \
    "$URL/v1/anonymize?name=message.eml"
  # A name alone, the one of a folder, and its restoration.
  name_call() { # route name
    api -X POST -H "Content-Type: application/json" --data "$(python3 -c 'import json, sys
print(json.dumps({"name": sys.argv[1]}))' "$2")" "$URL/v1/names/$1" |
      python3 -c 'import json, sys; print(json.load(sys.stdin).get("name", ""))'
  }
  NAMED=$(name_call anonymize "Dossier jean.dupont@exemple.fr")
  if [ -n "$NAMED" ] && ! printf '%s' "$NAMED" | grep -q -i dupont &&
     [ "$(name_call restore "$NAMED")" = "Dossier jean.dupont@exemple.fr" ]; then
    record "folder name" PASS "$NAMED"
  else record "folder name" FAIL "anonymized as '$NAMED'"
  fi

  # A folder as the web application sends it, a ZIP archive stored without compression: its
  # documents and its names protected, the file of another kind left out, the original given back.
  python3 - "$WORK/dossier.zip" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED)
z.writestr("Dossier Jean Dupont/", b"")
z.writestr("Dossier Jean Dupont/lettre.txt", "Je suis Jean Dupont, jean.dupont@exemple.fr.\n".encode())
z.writestr("Dossier Jean Dupont/vide/", b"")
z.writestr("Dossier Jean Dupont/outil.bin", b"MZ\x90\x00binary\x00")
z.close()
PY
  api -X POST --data-binary @"$WORK/dossier.zip" -D "$WORK/zip.headers" -o "$WORK/dossier.anon.zip" \
    "$URL/v1/anonymize?name=Dossier%20Jean%20Dupont.zip"
  ZIP_CHECK=$(python3 - "$WORK/dossier.anon.zip" "$WORK/zip.headers" <<'PY'
import json, sys, urllib.parse, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
headers = dict(l.split(":", 1) for l in open(sys.argv[2], encoding="latin-1").read().splitlines() if ":" in l)
headers = {k.strip().lower(): v.strip() for k, v in headers.items()}
report = json.loads(urllib.parse.unquote(headers.get("x-pseudalis-report", "{}")))
name = urllib.parse.unquote(headers.get("x-pseudalis-name", ""))
clean = all("dupont" not in n.lower() for n in names) and "dupont" not in name.lower()
print("ok" if clean and report.get("dropped") == 1 and len(names) == 3 else "bad %s %s %s" % (name, names, report))
PY
)
  if [ "$ZIP_CHECK" = ok ]; then record "folder archive" PASS "names, documents, 1 entry left out"
  else record "folder archive" FAIL "$ZIP_CHECK"
  fi
  api -X POST --data-binary @"$WORK/dossier.anon.zip" -o "$WORK/dossier.back.zip" \
    "$URL/v1/restore?name=dossier.zip"
  if cmp -s "$WORK/dossier.zip" "$WORK/dossier.back.zip"; then record "archive restored" PASS "the original archive"
  else record "archive restored" FAIL "the restored archive differs, in $WORK"
  fi

  # The drive: a document anonymized under a name without personal data, the files of another
  # kind moved out at their place in the tree, a folder renamed once its files rest; then the
  # originals of the file and of the name of the folder through the interface.
  DRIVE=$HOME_DIR/drive
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
  printf 'binary\0' > "$DRIVE/Dossier jean.dupont@exemple.fr/outil2.bin"
  cp "$WORK/lettre.txt" "$DRIVE/faux.docx"
  cp "$WORK/lettre.txt" "$DRIVE/journal jean.dupont@exemple.fr.log2"
  drive_done() {
    [ -z "$(find "$DRIVE" -iname '*dupont*')" ] && [ -f "$HOME_DIR/refused/files/outil.bin" ] &&
      [ -f "$HOME_DIR/refused/files/faux.docx" ] && [ -n "$(find "$DRIVE" -mindepth 2 -name note.txt)" ] &&
      [ -n "$(find "$HOME_DIR/refused/files" -mindepth 2 -name outil2.bin)" ] &&
      [ -n "$(find "$DRIVE" -maxdepth 1 -name '*.log2')" ]
  }
  if wait_for 60 drive_done; then record "drive anonymized" PASS "names, content, the files of another kind out"
  else record "drive anonymized" FAIL "$(find "$DRIVE" "$HOME_DIR/refused" | tr '\n' ' ')"
  fi
  # An archive over 64 KB: anonymized, then renamed with an extension of no kind, it stays a protected
  # file of the drive; without extension, it is taken by its signature.
  python3 - "$WORK/grosse.zip" <<'PY'
import random, sys, zipfile
random.seed(5)
words = ["".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(3, 9)))
         for _ in range(5000)]
text = "Note de Jean Dupont, " + " ".join(random.choice(words) for _ in range(60000)) + "\n"
z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED)
z.writestr("notes.txt", text)
z.close()
PY
  cp "$WORK/grosse.zip" "$DRIVE/grosse.zip"
  cp "$WORK/grosse.zip" "$DRIVE/archive_sans_extension"
  # The name without extension may take substitutes too: the file is found by its end.
  bare() { find "$DRIVE" -maxdepth 1 -type f -name '*sans_extension' | head -1; }
  archive_done() {
    [ -f "$DRIVE/grosse.zip" ] && [ -n "$(bare)" ] && ! cmp -s "$WORK/grosse.zip" "$DRIVE/grosse.zip" &&
      ! cmp -s "$WORK/grosse.zip" "$(bare)"
  }
  wait_for 90 archive_done
  BIG=$(stat -c %s "$DRIVE/grosse.zip" 2>/dev/null || echo 0)
  mv "$DRIVE/grosse.zip" "$DRIVE/grosse.zip.bak"
  sleep 8
  if [ -f "$DRIVE/grosse.zip.bak" ] && [ ! -e "$HOME_DIR/refused/files/grosse.zip.bak" ] &&
     [ "$BIG" -gt 65536 ] && [ -n "$(bare)" ] && ! cmp -s "$WORK/grosse.zip" "$(bare)"; then
    record "drive signatures" PASS "a protected archive renamed stays, an archive without extension is taken"
  else
    record "drive signatures" FAIL "size $BIG, $(find "$DRIVE" "$HOME_DIR/refused" -name '*grosse*' -o -name '*sans*' |
      tr '\n' ' ')"
  fi
  ANON=$(find "$DRIVE" -maxdepth 1 -type f -name 'Lettre*' | head -1)
  step "drive content" '\[\[PERSON:' cat "$ANON"
  api -X POST --data-binary @"$ANON" -D "$WORK/back.headers" -o "$WORK/lettre.back.txt" \
    "$URL/v1/restore?name=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' \
      "$(basename "$ANON")")"
  BACK_NAME=$(grep -i '^x-pseudalis-name:' "$WORK/back.headers" | cut -d: -f2- | tr -d '\r ' |
    python3 -c 'import sys, urllib.parse; print(urllib.parse.unquote(sys.stdin.read().strip()))')
  FOLDER=$(find "$DRIVE" -mindepth 1 -maxdepth 1 -type d | head -1)
  if cmp -s "$WORK/lettre.txt" "$WORK/lettre.back.txt" && [ "$BACK_NAME" = "Lettre jean.dupont@exemple.fr.txt" ] &&
     [ "$(name_call restore "$(basename "$FOLDER")")" = "Dossier jean.dupont@exemple.fr" ]; then
    record "drive restored" PASS "the original, its name and the name of the folder"
  else record "drive restored" FAIL "name '$BACK_NAME', folder $(basename "$FOLDER")"
  fi

  # The records of the originals, through the interface of the administrators.
  TOKEN=$(cat "$HOME_DIR/admin.token" 2>/dev/null)
  records() { api -H "X-Pseudalis-Admin: $TOKEN" "$URL/v1/records?limit=1000"; }
  # The record of a protected file, by the SHA-256 of $1 or by its identifier $2: "<id>|<removed>|<original name>".
  record_of() {
    records | python3 -c 'import json, sys
sha, ident = sys.argv[1], sys.argv[2]
for r in json.load(sys.stdin)["records"]:
    if r["output_sha256"] == sha or str(r["id"]) == ident:
        print("%s|%s|%s" % (r["id"], r["removed"] or "-", r["original_name"]))
        break' "$( [ -f "$1" ] && sha256sum "$1" | cut -c1-64)" "${2:-}"
  }
  total() { records | python3 -c 'import json, sys; print(json.load(sys.stdin)["total"])'; }
  quoted() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$1"; }
  restore_to() { # file output, the headers of the answer in output.headers
    api -X POST --data-binary @"$1" -D "$2.headers" -o "$2" "$URL/v1/restore?name=$(quoted "$(basename "$1")")"
  }
  report_of() { # headers: the report they give, decoded
    python3 -c 'import sys, urllib.parse
for line in open(sys.argv[1], encoding="latin-1"):
    if line.lower().startswith("x-pseudalis-report:"):
        print(urllib.parse.unquote(line.split(":", 1)[1].strip()))' "$1"
  }
  # An archive anonymized on the drive, then extracted on the drive: its files stay as they are, each
  # recorded under its original name with the original of its entry, and restored and forgotten alone.
  python3 - "$WORK/envoi.zip" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED)
z.writestr("Dossier Jean Dupont/lettre.txt", "Je suis Jean Dupont, joignable au 06 12 34 56 78.\n")
z.writestr("note Marie Curie.txt", "Courrier de Marie Curie, 14 avenue des Tilleuls, 69003 Lyon.\n")
z.writestr("reste.txt", "Rendez-vous avec Paul Martin le 3 mars.\n")
z.close()
PY
  cp "$WORK/envoi.zip" "$DRIVE/envoi.zip"
  envoi_done() { [ -f "$DRIVE/envoi.zip" ] && ! cmp -s "$WORK/envoi.zip" "$DRIVE/envoi.zip" &&
    [ -n "$(record_of "$DRIVE/envoi.zip")" ]; }
  wait_for 60 envoi_done
  cp "$DRIVE/envoi.zip" "$WORK/envoi.anon.zip"
  # Two files extracted on the drive, the third elsewhere.
  python3 - "$WORK/envoi.anon.zip" "$WORK/extrait" "$WORK/ailleurs" <<'PY'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if n.endswith("/"):
        continue
    path = os.path.join(sys.argv[3] if os.path.basename(n) == "reste.txt" else sys.argv[2], n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(z.read(n))
PY
  OUTSIDE=$WORK/ailleurs/reste.txt
  mkdir -p "$DRIVE/extrait"
  cp -r "$WORK/extrait/." "$DRIVE/extrait/"
  extracted_done() {
    local f n=0
    while IFS= read -r f; do
      [ -n "$(record_of "$f")" ] || return 1
      n=$((n + 1))
    done < <(find "$DRIVE/extrait" -type f)
    [ "$n" -eq 2 ]
  }
  wait_for 60 extracted_done
  SAME=0
  while IFS= read -r f; do
    cmp -s "$f" "$DRIVE/extrait/${f#"$WORK/extrait/"}" && SAME=$((SAME + 1))
  done < <(find "$WORK/extrait" -type f)
  NAMES=$(find "$DRIVE/extrait" -type f | while IFS= read -r f; do record_of "$f" | cut -d'|' -f3; done | sort |
    tr '\n' ',')
  if [ "$SAME" -eq 2 ] && [ "$NAMES" = "lettre.txt,note Marie Curie.txt," ]; then
    record "archive extracted" PASS "2 files kept as they are, recorded under their original names"
  else record "archive extracted" FAIL "unchanged $SAME, names $NAMES"
  fi
  LETTRE=$(find "$DRIVE/extrait" -type f -name 'lettre.txt' | head -1)
  restore_to "$LETTRE" "$WORK/lettre.extrait.txt"
  restore_to "$OUTSIDE" "$WORK/reste.back.txt"
  ZIP_ID=$(record_of "$WORK/envoi.anon.zip" | cut -d'|' -f1)
  api -X DELETE -H "X-Pseudalis-Admin: $TOKEN" -o /dev/null "$URL/v1/records/$ZIP_ID"
  restore_to "$LETTRE" "$WORK/lettre.apres.txt"
  restore_to "$OUTSIDE" "$WORK/reste.apres.txt"
  if grep -q 'Jean Dupont, joignable' "$WORK/lettre.extrait.txt" && grep -q 'Paul Martin' "$WORK/reste.back.txt" &&
     grep -q 'Jean Dupont, joignable' "$WORK/lettre.apres.txt" &&
     grep -q 'Paul Martin le 3 mars' "$WORK/reste.apres.txt" &&
     grep -qi '^x-pseudalis-restored: tags' "$WORK/reste.apres.txt.headers" &&
     report_of "$WORK/reste.apres.txt.headers" | grep -q '"original_forgotten"' 
  then
    record "extracted restored" PASS "from their originals once the archive forgotten, the file outside from its tags"
  else record "extracted restored" FAIL "see lettre.extrait.txt, reste.back.txt, lettre.apres.txt, reste.apres.txt"
  fi
  # The archive forgotten comes back to the drive: it stays as it is, without a new original.
  COUNT=$(total)
  cp "$WORK/envoi.anon.zip" "$DRIVE/revenu.zip"
  sleep 8
  if cmp -s "$WORK/envoi.anon.zip" "$DRIVE/revenu.zip" && [ "$(total)" = "$COUNT" ]; then
    record "forgotten comes back" PASS "kept as it is, no new original"
  else record "forgotten comes back" FAIL "$(ls "$DRIVE" | tr '\n' ' '), records $COUNT then $(total)"
  fi
  # A protected file copied on the drive: its record stays while a copy remains, and goes with the last.
  printf 'Copie pour Lucie Marchal, 3 place du Champ de Foire, 29000 Quimper.\n' > "$DRIVE/copies.txt"
  copies_done() { ! grep -q Marchal "$DRIVE/copies.txt" 2>/dev/null && [ -n "$(record_of "$DRIVE/copies.txt")" ]; }
  wait_for 60 copies_done
  COPIES_ID=$(record_of "$DRIVE/copies.txt" | cut -d'|' -f1)
  COUNT=$(total)
  cp "$DRIVE/copies.txt" "$DRIVE/copies (2).txt"
  # The copy rests, then its record is the one of the file: no new original.
  sleep 6
  rm -f "$DRIVE/copies.txt"
  sleep 4
  KEPT=$(record_of "" "$COPIES_ID" | cut -d'|' -f2)
  rm -f "$DRIVE/copies (2).txt"
  copies_gone() { [ "$(record_of "" "$COPIES_ID" | cut -d'|' -f2)" != "-" ]; }
  wait_for 30 copies_gone
  GONE=$(record_of "" "$COPIES_ID" | cut -d'|' -f2)
  if [ -n "$COPIES_ID" ] && [ "$(total)" = "$COUNT" ] && [ "$KEPT" = "-" ] && [ -n "$GONE" ] && [ "$GONE" != "-" ]
  then
    record "copies on the drive" PASS "one record, kept with a copy, removed with the last"
  else record "copies on the drive" FAIL "record '$COPIES_ID', with the copy '$KEPT', after it '$GONE'"
  fi
  # The settings as the web application saves them, without feed nor proxy; a value out of its range refused.
  admin_json() { # method route body
    api -X "$1" -H "X-Pseudalis-Admin: $TOKEN" -H "Content-Type: application/json" --data "$3" "$URL$2"
  }
  SAVED=$(admin_json PUT /v1/settings '{"language": "auto", "drive_letter": "A", "date_order": "dmy",
    "max_file_mb": 150, "removed_days": 0, "history_mb": 10240, "albireo": "off", "check_updates": false}')
  REFUSED=$(admin_json PUT /v1/settings '{"removed_days": -1}')
  if printf '%s' "$SAVED" | grep -q '"removed_days":0' && printf '%s' "$SAVED" | grep -q '"restart":false' &&
     printf '%s' "$REFUSED" | grep -q '"code":"settings_value"'; then
    record "settings saved" PASS "the body of the web application, 0 days, a value out of range refused"
  else record "settings saved" FAIL "$SAVED / $REFUSED"
  fi
  # The imposed terms: kept, replaced before anything else, and refused with their reason.
  SET=$(admin_json PUT /v1/terms '{"terms": [{"term": "Projet Alpha", "type": "ORG"},
    {"term": "KX-77", "type": "ID", "case_sensitive": true}]}')
  BAD=$(admin_json PUT /v1/terms '{"terms": [{"term": "Alpha", "type": "DATE"}]}')
  TERMS_TEXT=$(api -X POST -H "Content-Type: application/json" \
    --data '{"text": "Le PROJET-ALPHA suit la reference KX-77, pas kx-77."}' "$URL/v1/text/anonymize")
  LISTED=$(api -H "X-Pseudalis-Admin: $TOKEN" "$URL/v1/terms")
  if printf '%s' "$SET" | grep -q 'Projet Alpha' && printf '%s' "$BAD" | grep -q '"code":"terms_type"' &&
     printf '%s' "$TERMS_TEXT" | grep -q '\[\[ORG:' && printf '%s' "$TERMS_TEXT" | grep -q '\[\[ID:' &&
     ! printf '%s' "$TERMS_TEXT" | grep -q -e ALPHA -e KX-77 && printf '%s' "$TERMS_TEXT" | grep -q 'kx-77' &&
     printf '%s' "$LISTED" | grep -q '"max":100'; then
    record "imposed terms" PASS "replaced first, case kept for KX-77, a type refused"
  else record "imposed terms" FAIL "$SET / $BAD / $TERMS_TEXT"
  fi
  admin_json PUT /v1/terms '{"terms": []}' > /dev/null
  # The space of the originals: with 1 MB, the oldest originals make room for the newer ones; a space
  # smaller than the largest file is refused.
  SMALL=$(admin_json PUT /v1/settings '{"history_mb": 1}')
  admin_json PUT /v1/settings '{"max_file_mb": 1, "history_mb": 1}' > /dev/null
  for k in 1 2 3; do
    python3 -c 'import sys
sys.stdout.write(("Note %s de Jean Dupont, jean.dupont@exemple.fr\n" % sys.argv[1]) * 9000)' "$k" > "$WORK/gros$k.txt"
    api -X POST --data-binary @"$WORK/gros$k.txt" -o "$WORK/gros$k.anon.txt" "$URL/v1/anonymize?name=gros$k.txt"
  done
  api -X POST --data-binary @"$WORK/gros1.anon.txt" -D "$WORK/gros1.back.headers" -o "$WORK/gros1.back.txt" \
    "$URL/v1/restore?name=gros1.txt"
  api -X POST --data-binary @"$WORK/gros3.anon.txt" -o "$WORK/gros3.back.txt" "$URL/v1/restore?name=gros3.txt"
  SPACE=$(api "$URL/v1/info" | python3 -c 'import json, sys; h = json.load(sys.stdin)["history"]
print(h["bytes"] <= h["limit"])')
  admin_json PUT /v1/settings '{"history_mb": 10240, "max_file_mb": 150}' > /dev/null
  if cmp -s "$WORK/gros1.txt" "$WORK/gros1.back.txt" &&
     report_of "$WORK/gros1.back.headers" | grep -q '"reason":"space"' &&
     cmp -s "$WORK/gros3.txt" "$WORK/gros3.back.txt" && [ "$SPACE" = True ] &&
     printf '%s' "$SMALL" | grep -q '"field":"history_mb"'; then
    record "room for originals" PASS "the oldest from its tags (space), the newest from its original, 1 MB refused"
  else record "room for originals" FAIL "$(head -c 200 "$WORK/gros1.back.txt"), within the limit: $SPACE / $SMALL"
  fi
  # An archive with a picture, its original forgotten: restored from its tags, without its picture; the
  # text file of a picture alone, forgotten, is refused.
  python3 - "$WORK/album.zip" "$WORK/photo.png" <<'PY'
import base64, sys, zipfile
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAAB"
                       "JRU5ErkJggg==")
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("lettre.txt", "Album de Jeanne Marchand, 12 rue des Lilas, 29000 Quimper.\n")
    z.writestr("photo.png", png)
open(sys.argv[2], "wb").write(png)
PY
  for f in album.zip photo.png; do
    api -X POST --data-binary @"$WORK/$f" -D "$WORK/$f.anon.headers" -o "$WORK/$f.anon" \
      "$URL/v1/anonymize?name=$f&record=true"
    ID=$(grep -i '^x-pseudalis-record:' "$WORK/$f.anon.headers" | tr -dc '0-9')
    api -X DELETE -H "X-Pseudalis-Admin: $TOKEN" -o /dev/null "$URL/v1/records/$ID"
  done
  api -X POST --data-binary @"$WORK/album.zip.anon" -D "$WORK/album.back.headers" -o "$WORK/album.back.zip" \
    "$URL/v1/restore?name=album.zip"
  PICTURE=$(api -X POST --data-binary @"$WORK/photo.png.anon" -o /dev/null -w '%{http_code}' \
    "$URL/v1/restore?name=photo.png.txt")
  api -X POST --data-binary @"$WORK/photo.png.anon" -o "$WORK/photo.back.txt" "$URL/v1/restore?name=photo.png.txt"
  ALBUM=$(python3 - "$WORK/album.back.zip" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
print("%s|%s" % (",".join(z.namelist()), z.read("lettre.txt").decode() if "lettre.txt" in z.namelist() else ""))
PY
)
  if [ "$ALBUM" = "lettre.txt|Album de Jeanne Marchand, 12 rue des Lilas, 29000 Quimper." ] &&
     report_of "$WORK/album.back.headers" | grep -q '"pictures_removed":1' &&
     report_of "$WORK/album.back.headers" | grep -q '"original_forgotten"' && [ "$PICTURE" = 410 ] &&
     grep -q '"code":"picture_forgotten"' "$WORK/photo.back.txt"; then
    record "pictures forgotten" PASS "the archive from its tags without its picture, the picture alone refused (410)"
  else record "pictures forgotten" FAIL "$ALBUM / picture $PICTURE / $(head -c 200 "$WORK/photo.back.txt")"
  fi
  # A text written elsewhere from anonymized values, which quotes a picture: restored from its tags without
  # the marker; anonymized and recorded as an original, it comes back with its values and without it too.
  printf 'Lettre de Jeanne Marchand.\n' > "$WORK/source.txt"
  api -X POST --data-binary @"$WORK/source.txt" -o "$WORK/source.anon" \
    "$URL/v1/anonymize?name=source.txt&record=false"
  printf 'Réponse : %s [Image@0123456789abcdef : Photo]\n' "$(cat "$WORK/source.anon")" > "$WORK/reponse.txt"
  api -X POST --data-binary @"$WORK/reponse.txt" -D "$WORK/reponse.back.headers" -o "$WORK/reponse.back.txt" \
    "$URL/v1/restore?name=reponse.txt"
  api -X POST --data-binary @"$WORK/reponse.txt" -o "$WORK/reponse.anon" \
    "$URL/v1/anonymize?name=reponse.txt&record=true"
  api -X POST --data-binary @"$WORK/reponse.anon" -o "$WORK/reponse.again.txt" "$URL/v1/restore?name=reponse.txt"
  if [ "$(cat "$WORK/reponse.back.txt")" = "Réponse : Lettre de Jeanne Marchand." ] &&
     report_of "$WORK/reponse.back.headers" | grep -q '"pictures_removed":1' &&
     [ "$(cat "$WORK/reponse.again.txt")" = "Réponse : Lettre de Jeanne Marchand." ]; then
    record "written elsewhere" PASS "from its tags without the marker, recorded: from its original, values back"
  else record "written elsewhere" FAIL \
    "$(head -c 120 "$WORK/reponse.back.txt") / $(head -c 120 "$WORK/reponse.again.txt")"
  fi
  # Without tags: substitutes alone, the original from its record; once forgotten, nothing comes back (410).
  printf 'nom;prenom;email\nMarchand;Jeanne;jeanne.marchand@exemple.fr\n' > "$WORK/bare.csv"
  api -X POST --data-binary @"$WORK/bare.csv" -D "$WORK/bare.headers" -o "$WORK/bare.anon.csv" \
    "$URL/v1/anonymize?name=bare.csv&tags=false&record=true"
  BARE_ID=$(grep -i '^x-pseudalis-record:' "$WORK/bare.headers" | tr -dc '0-9')
  api -X POST --data-binary @"$WORK/bare.anon.csv" -o "$WORK/bare.back.csv" "$URL/v1/restore?name=bare.csv"
  BARE_TAGS=$(records | python3 -c 'import json, sys
print(next((r["tags"] for r in json.load(sys.stdin)["records"] if str(r["id"]) == sys.argv[1]), "none"))' "$BARE_ID")
  api -X DELETE -H "X-Pseudalis-Admin: $TOKEN" -o /dev/null "$URL/v1/records/$BARE_ID"
  BARE_GONE=$(api -X POST --data-binary @"$WORK/bare.anon.csv" -o "$WORK/bare.gone.json" -w '%{http_code}' \
    "$URL/v1/restore?name=bare.csv")
  if ! grep -q -e '\[\[' -e Marchand "$WORK/bare.anon.csv" && cmp -s "$WORK/bare.csv" "$WORK/bare.back.csv" &&
     [ "$BARE_TAGS" = False ] && [ "$BARE_GONE" = 410 ] && grep -q '"code":"tagless_forgotten"' "$WORK/bare.gone.json"
  then
    record "files without tags" PASS "from the record, the record says so, forgotten: refused (410)"
  else record "files without tags" FAIL "$(head -c 100 "$WORK/bare.anon.csv") / tags $BARE_TAGS / $BARE_GONE"
  fi
  # Without tags and without record: nothing can come back, which the report tells.
  printf 'nom;prenom;email\nLemoine;Paul;paul.lemoine@exemple.fr\n' > "$WORK/bare2.csv"
  api -X POST --data-binary @"$WORK/bare2.csv" -D "$WORK/bare2.headers" -o "$WORK/bare2.anon.csv" \
    "$URL/v1/anonymize?name=bare2.csv&tags=false&record=false"
  BARE2=$(api -X POST --data-binary @"$WORK/bare2.anon.csv" -o /dev/null -w '%{http_code}' \
    "$URL/v1/restore?name=bare2.csv")
  if [ "$BARE2" = 404 ] && report_of "$WORK/bare2.headers" | grep -q '"restorable":false'; then
    record "without tags nor record" PASS "nothing kept, not restorable (404)"
  else record "without tags nor record" FAIL "$BARE2 / $(report_of "$WORK/bare2.headers" | head -c 160)"
  fi
  # A text without tags: its substitutes listed, which give it back when they come with it.
  BARE_TEXT=$(api -X POST -H "Content-Type: application/json" --data "{\"text\": \"$TEXT\", \"tags\": false}" \
    "$URL/v1/text/anonymize")
  BARE_BACK=$(printf '%s' "$BARE_TEXT" | python3 -c 'import json, sys
j = json.load(sys.stdin)
print(json.dumps({"text": j["text"], "substitutions": j.get("substitutions", [])}))' |
    api -X POST -H "Content-Type: application/json" --data @- "$URL/v1/text/restore" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["text"])')
  BARE_WRITTEN=$(printf '%s' "$BARE_TEXT" | python3 -c 'import json, sys; print(json.load(sys.stdin)["text"])')
  if ! printf '%s' "$BARE_WRITTEN" | grep -q -e '\[\[' -e 'Dupont' && [ "$BARE_BACK" = "$TEXT" ]; then
    record "text without tags" PASS "substitutes alone, given back with their list"
  else record "text without tags" FAIL "$(printf '%s' "$BARE_WRITTEN" | head -c 160) / $BARE_BACK"
  fi
  # Protected twice, by a newer engine for instance: tagged with its record forgotten, then with or without
  # tags; without tags with its record kept, then with or without; the values of both come back without warning.
  printf 'Dossier de Paul Lemaire, paul.lemaire@exemple.fr.\n' > "$WORK/twice.txt"
  api -X POST --data-binary @"$WORK/twice.txt" -D "$WORK/twice1.headers" -o "$WORK/twice1.txt" \
    "$URL/v1/anonymize?name=twice.txt&tags=true"
  T1=$(grep -i '^x-pseudalis-record:' "$WORK/twice1.headers" | tr -dc '0-9')
  api -X DELETE -H "X-Pseudalis-Admin: $TOKEN" -o /dev/null "$URL/v1/records/$T1"
  printf 'Visa de Hélène Martin.\n' >> "$WORK/twice1.txt"
  api -X POST --data-binary @"$WORK/twice1.txt" -o "$WORK/twice2.txt" "$URL/v1/anonymize?name=twice.txt&tags=true"
  api -X POST --data-binary @"$WORK/twice2.txt" -D "$WORK/twice.back.headers" -o "$WORK/twice.back.txt" \
    "$URL/v1/restore?name=twice.txt"
  api -X POST --data-binary @"$WORK/twice1.txt" -o "$WORK/bare4.txt" "$URL/v1/anonymize?name=twice.txt&tags=false"
  api -X POST --data-binary @"$WORK/bare4.txt" -o "$WORK/bare4.back.txt" "$URL/v1/restore?name=twice.txt"
  api -X POST --data-binary @"$WORK/twice.txt" -o "$WORK/bare1.txt" "$URL/v1/anonymize?name=twice.txt&tags=false"
  api -X POST --data-binary @"$WORK/bare1.txt" -o "$WORK/bare2.txt" "$URL/v1/anonymize?name=twice.txt&tags=true"
  api -X POST --data-binary @"$WORK/bare1.txt" -o "$WORK/bare3.txt" "$URL/v1/anonymize?name=twice.txt&tags=false"
  api -X POST --data-binary @"$WORK/bare2.txt" -o "$WORK/bare2.back.txt" "$URL/v1/restore?name=twice.txt"
  api -X POST --data-binary @"$WORK/bare3.txt" -o "$WORK/bare3.back.txt" "$URL/v1/restore?name=twice.txt"
  TWICE=$(printf 'Dossier de Paul Lemaire, paul.lemaire@exemple.fr.\nVisa de Hélène Martin.')
  if [ "$(cat "$WORK/twice.back.txt")" = "$TWICE" ] &&
     cmp -s "$WORK/twice.txt" "$WORK/bare2.back.txt" && cmp -s "$WORK/twice.txt" "$WORK/bare3.back.txt" &&
     ! grep -q -e '\[\[' -e Lemaire -e Martin "$WORK/bare4.txt" && [ "$(cat "$WORK/bare4.back.txt")" = "$TWICE" ] &&
     ! grep -q '"code"' "$WORK/twice.back.txt"; then
    record "protected twice" PASS "tags then with or without, without tags then with or without: the values of both"
  else record "protected twice" FAIL "$(head -c 120 "$WORK/twice.back.txt") / $(head -c 80 "$WORK/bare2.back.txt")"
  fi
  # The drive without tags, as the settings say: its file holds no tag, its original comes back.
  admin_json PUT /v1/settings '{"tags": false}' > /dev/null
  printf 'Courrier de Paul Lemaire, 06 11 22 33 44.\n' > "$WORK/courrier.txt"
  cp "$WORK/courrier.txt" "$DRIVE/courrier.txt"
  courrier_done() {
    ! cmp -s "$WORK/courrier.txt" "$DRIVE/courrier.txt" && [ -n "$(record_of "$DRIVE/courrier.txt")" ]
  }
  wait_for 60 courrier_done
  cp "$DRIVE/courrier.txt" "$WORK/courrier.anon.txt"
  restore_to "$WORK/courrier.anon.txt" "$WORK/courrier.back.txt"
  DRIVE_TAGS=$(api "$URL/v1/info" | python3 -c 'import json, sys; print(json.load(sys.stdin)["tags"])')
  # Edited on the drive, as an editor saves it: protected again, its values protected before come back too.
  { cat "$WORK/courrier.anon.txt"; printf 'Visa de Hélène Martin.\n'; } > "$WORK/courrier.edit.txt"
  cp "$WORK/courrier.edit.txt" "$DRIVE/courrier.txt"
  edited_done() {
    ! cmp -s "$WORK/courrier.edit.txt" "$DRIVE/courrier.txt" && [ -n "$(record_of "$DRIVE/courrier.txt")" ]
  }
  wait_for 60 edited_done
  cp "$DRIVE/courrier.txt" "$WORK/courrier.anon2.txt"
  restore_to "$WORK/courrier.anon2.txt" "$WORK/courrier.back2.txt"
  admin_json PUT /v1/settings '{"tags": true}' > /dev/null
  if ! grep -q -e '\[\[' -e Lemaire "$WORK/courrier.anon.txt" &&
     cmp -s "$WORK/courrier.txt" "$WORK/courrier.back.txt" && [ "$DRIVE_TAGS" = False ] &&
     ! grep -q -e '\[\[' -e Martin "$WORK/courrier.anon2.txt" &&
     [ "$(cat "$WORK/courrier.back2.txt")" = "$(printf 'Courrier de Paul Lemaire, 06 11 22 33 44.\nVisa de %s.' \
       'Hélène Martin')" ]
  then
    record "drive without tags" PASS "the setting followed, the original back, edited: the values of both"
  else record "drive without tags" FAIL "$(head -c 100 "$WORK/courrier.anon.txt") / $DRIVE_TAGS / \
    $(head -c 100 "$WORK/courrier.back2.txt")"
  fi
  # The imposed columns: below their names, a title above the header, substitutes alone, the other values
  # tagged; the original comes back, and from a file edited on the drive, the values of both.
  COLS=$(admin_json PUT /v1/columns '{"columns": [{"column": "Nom", "type": "PERSON"},
    {"column": "Courriel", "type": "EMAIL"}]}')
  BAD_COLS=$(admin_json PUT /v1/columns '{"columns": [{"column": "Nom", "type": "NONE"}]}')
  LISTED_COLS=$(api -H "X-Pseudalis-Admin: $TOKEN" "$URL/v1/columns")
  printf 'Export du 12/03/2024\nNOM;Courriel;Ville\nLemoine;paul.lemoine@exemple.fr;Rennes\n' > "$WORK/export.csv"
  api -X POST --data-binary @"$WORK/export.csv" -D "$WORK/export.headers" -o "$WORK/export.anon.csv" \
    "$URL/v1/anonymize?name=export.csv&tags=true&record=true"
  restore_to "$WORK/export.anon.csv" "$WORK/export.back.csv"
  cp "$WORK/export.csv" "$DRIVE/export.csv"
  export_done() {
    ! cmp -s "$WORK/export.csv" "$DRIVE/export.csv" && [ -n "$(record_of "$DRIVE/export.csv")" ]
  }
  wait_for 60 export_done
  { cat "$DRIVE/export.csv"; printf 'Martin;h.martin@exemple.fr;Lyon\n'; } > "$WORK/export.edit.csv"
  cp "$WORK/export.edit.csv" "$DRIVE/export.csv"
  export_edited() {
    ! cmp -s "$WORK/export.edit.csv" "$DRIVE/export.csv" && [ -n "$(record_of "$DRIVE/export.csv")" ]
  }
  wait_for 60 export_edited
  cp "$DRIVE/export.csv" "$WORK/export.anon2.csv"
  restore_to "$WORK/export.anon2.csv" "$WORK/export.back2.csv"
  admin_json PUT /v1/columns '{"columns": []}' > /dev/null
  if printf '%s' "$COLS" | grep -q '"column":"Nom"' && printf '%s' "$BAD_COLS" | grep -q '"code":"columns_type"' &&
     printf '%s' "$LISTED_COLS" | grep -q '"max":100' &&
     sed -n 2p "$WORK/export.anon.csv" | grep -q '^NOM;Courriel;Ville$' &&
     ! grep -q -e Lemoine -e paul.lemoine "$WORK/export.anon.csv" &&
     ! sed -n 3p "$WORK/export.anon.csv" | grep -q '^\[\[' && grep -q '\[\[LOCATION:' "$WORK/export.anon.csv" &&
     report_of "$WORK/export.headers" | grep -q '"restorable":true' &&
     cmp -s "$WORK/export.csv" "$WORK/export.back.csv" && ! grep -q -e Lemoine -e Martin "$WORK/export.anon2.csv" &&
     [ "$(cat "$WORK/export.back2.csv")" = "$(cat "$WORK/export.csv"; printf 'Martin;h.martin@exemple.fr;Lyon')" ]
  then
    record "imposed columns" PASS "substitutes below their names, the original back, edited on the drive: both"
  else record "imposed columns" FAIL "$(sed -n 3p "$WORK/export.anon.csv" | head -c 100) / $BAD_COLS / \
    $(tail -c 120 "$WORK/export.back2.csv")"
  fi
  # A Windows-1252 file whose accented letters all stand in an imposed column, protected in plain ASCII:
  # edited on the drive, it comes back in Windows-1252, the encoding of its original.
  admin_json PUT /v1/columns '{"columns": [{"column": "Nom", "type": "PERSON"}]}' > /dev/null
  printf 'Nom;Ville\nH\351l\350ne;Rennes\n' > "$WORK/accents.csv"
  cp "$WORK/accents.csv" "$DRIVE/accents.csv"
  accents_done() {
    ! cmp -s "$WORK/accents.csv" "$DRIVE/accents.csv" && [ -n "$(record_of "$DRIVE/accents.csv")" ]
  }
  wait_for 60 accents_done
  { cat "$DRIVE/accents.csv"; printf 'Martin;Lyon\n'; } > "$WORK/accents.edit.csv"
  cp "$WORK/accents.edit.csv" "$DRIVE/accents.csv"
  accents_edited() {
    ! cmp -s "$WORK/accents.edit.csv" "$DRIVE/accents.csv" && [ -n "$(record_of "$DRIVE/accents.csv")" ]
  }
  wait_for 60 accents_edited
  cp "$DRIVE/accents.csv" "$WORK/accents.anon2.csv"
  restore_to "$WORK/accents.anon2.csv" "$WORK/accents.back2.csv"
  { cat "$WORK/accents.csv"; printf 'Martin;Lyon\n'; } > "$WORK/accents.expected.csv"
  admin_json PUT /v1/columns '{"columns": []}' > /dev/null
  if ! LC_ALL=C grep -q "$(printf '[\200-\377]')" "$WORK/accents.edit.csv" &&
     ! grep -q -e Martin "$WORK/accents.anon2.csv" && cmp -s "$WORK/accents.expected.csv" "$WORK/accents.back2.csv"
  then
    record "Windows-1252 kept" PASS "protected in plain ASCII, edited on the drive: back in Windows-1252"
  else record "Windows-1252 kept" FAIL "$(od -An -c "$WORK/accents.back2.csv" | head -c 160)"
  fi
  # An IP column, given as IPV4, takes both forms of address: each address takes the substitute of its form,
  # its prefix length and its port kept; the original comes back, and from a file edited on the drive, both.
  IP_COLS=$(admin_json PUT /v1/columns '{"columns": [{"column": "ip_address", "type": "IPV4"}]}')
  printf 'host;ip_address\nweb;192.168.1.5\ndb;2001:db8::1\nnet;10.0.0.0/8\napi;[2001:db8::1]:443\n' \
    > "$WORK/hosts.csv"
  api -X POST --data-binary @"$WORK/hosts.csv" -D "$WORK/hosts.headers" -o "$WORK/hosts.anon.csv" \
    "$URL/v1/anonymize?name=hosts.csv&tags=true&record=true"
  restore_to "$WORK/hosts.anon.csv" "$WORK/hosts.back.csv"
  cp "$WORK/hosts.csv" "$DRIVE/hosts.csv"
  hosts_done() {
    ! cmp -s "$WORK/hosts.csv" "$DRIVE/hosts.csv" && [ -n "$(record_of "$DRIVE/hosts.csv")" ]
  }
  wait_for 60 hosts_done
  { cat "$DRIVE/hosts.csv"; printf 'vpn;fe80::1%%eth0\n'; } > "$WORK/hosts.edit.csv"
  cp "$WORK/hosts.edit.csv" "$DRIVE/hosts.csv"
  hosts_edited() {
    ! cmp -s "$WORK/hosts.edit.csv" "$DRIVE/hosts.csv" && [ -n "$(record_of "$DRIVE/hosts.csv")" ]
  }
  wait_for 60 hosts_edited
  cp "$DRIVE/hosts.csv" "$WORK/hosts.anon2.csv"
  restore_to "$WORK/hosts.anon2.csv" "$WORK/hosts.back2.csv"
  { cat "$WORK/hosts.csv"; printf 'vpn;fe80::1%%eth0\n'; } > "$WORK/hosts.expected.csv"
  admin_json PUT /v1/columns '{"columns": []}' > /dev/null
  if printf '%s' "$IP_COLS" | grep -q '"type":"IP"' &&
     ! grep -q -F -e 192.168.1.5 -e 2001:db8::1 -e 10.0.0.0 "$WORK/hosts.anon.csv" &&
     grep -q '/8$' "$WORK/hosts.anon.csv" && grep -q ']:443$' "$WORK/hosts.anon.csv" &&
     ! grep -q -F '[[' "$WORK/hosts.anon.csv" && cmp -s "$WORK/hosts.csv" "$WORK/hosts.back.csv" &&
     ! grep -q -F 'fe80::1' "$WORK/hosts.anon2.csv" && grep -q '%eth0$' "$WORK/hosts.anon2.csv" &&
     cmp -s "$WORK/hosts.expected.csv" "$WORK/hosts.back2.csv"
  then
    record "IP column" PASS "IPv4 and IPv6 in one column, the original back, edited on the drive: both"
  else record "IP column" FAIL "$(head -c 160 "$WORK/hosts.anon.csv") / $IP_COLS / \
    $(tail -c 80 "$WORK/hosts.back2.csv")"
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
