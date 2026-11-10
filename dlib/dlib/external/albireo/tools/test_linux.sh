#!/bin/bash
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Tests a Linux build of Albireo and prints a summary, one line per step.
#
#   bash dlib/dlib/external/albireo/tools/test_linux.sh [build folder] [model.gguf] [embedding model.gguf]
#
# Defaults: build-albireo at the root of the repository, ~/models/Qwen3-0.6B-Q4_K_M.gguf and
# ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf.  The steps whose model is missing are skipped.
# The service runs offline in a data folder of its own, on port 18443 (ALBIREO_TEST_PORT, and
# the next port for the local server of test_host), receives the model as the tray sends it,
# refuses it by its path, answers a question through the OpenAI interface and stops.  The
# first step checks the tools of the build with tools/prerequisites_linux.sh, which installs those
# that are missing.  The exit code is the number of failed steps.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
BUILD=${1:-$ROOT/build-albireo}
MODEL=${2:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
EMBEDDING=${3:-$HOME/models/Qwen3-Embedding-0.6B-Q8_0.gguf}
PORT=${ALBIREO_TEST_PORT:-18443}
QUESTION="What is the capital of France? Answer in one sentence."
WORK=$(mktemp -d)
RESULTS=()
FAILED=0

now() { date +%s%N; }
elapsed() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", (b - a) / 1e9 }'; }

record() { # step result detail seconds
  RESULTS+=("$(printf '%-28s %-5s %6s s  %s' "$1" "$2" "$4" "$3")")
  [ "$2" = FAIL ] && FAILED=$((FAILED + 1))
}

# Runs a command, then checks that its output holds a pattern.
step() { # name pattern command...
  local name=$1 pattern=$2
  shift 2
  local start out rc seconds line
  start=$(now)
  out=$("$@" 2>&1)
  rc=$?
  seconds=$(elapsed "$start")
  line=$(printf '%s\n' "$out" | grep -E -o -m1 ".{0,40}$pattern.{0,30}" | head -1)
  if [ $rc -eq 0 ] && [ -n "$line" ]; then
    record "$name" PASS "$line" "$seconds"
  else
    printf '%s\n' "$out" | tail -20 > "$WORK/$name.log"
    record "$name" FAIL "exit $rc, output in $WORK/$name.log" "$seconds"
  fi
}

skip() { record "$1" SKIP "$2" 0.0; }

# The tools of the build and of these tests, installed when missing; the run shows the prompt of sudo.
START=$(now)
if bash "$HERE/prerequisites_linux.sh"; then
  record "prerequisites" PASS "every tool is there" "$(elapsed "$START")"
else
  record "prerequisites" FAIL "see the messages of tools/prerequisites_linux.sh above" "$(elapsed "$START")"
fi

ENGINES=$BUILD/bin/engines
CHAT=$BUILD/tests/engine_chat

# The kernels of each family of processors, skipped by the processors without their instructions.
for family in "" _avxvnni _vnni256; do
  test=$BUILD/tests/test_qkernels$family
  name="quantized kernels${family:+ ${family#_}}"
  if [ ! -x "$test" ]; then
    [ -z "$family" ] && skip "$name" "no $test (ALBIREO_TESTS=ON)"
    continue
  fi
  out=$("$test" 2>&1)
  if printf '%s\n' "$out" | grep -q "skipped, the processor lacks"; then
    skip "$name" "$(printf '%s\n' "$out" | grep -o 'lacks [^(]*' | head -1)"
  else
    step "$name" "all checks passed" "$test"
  fi
done

# The attention of the steps split among the threads, against one task per key/value head, with
# keys and values in half precision and in 8 bits, on a tiny model written in the work folder.
if [ -x "$BUILD/tests/test_runtime" ]; then
  step "attention" "all checks passed" "$BUILD/tests/test_runtime" synthetic "$WORK"
else skip "attention" "no $BUILD/tests/test_runtime"
fi

# The attention of a device on the same tiny model, through the host code of dlib, when the tree
# was configured with ALBIREO_ENGINE_DLIB=ON.
if [ -x "$BUILD/tests/test_offload" ]; then
  step "attention of the device" "all checks passed" "$BUILD/tests/test_offload" synthetic "$WORK"
else skip "attention of the device" "no $BUILD/tests/test_offload (ALBIREO_ENGINE_DLIB=ON)"
fi

# Damaged model files, the chat format of the agents and the C interface of an engine under
# concurrent requests, on tiny models the tests write in the work folder.
if [ -x "$BUILD/tests/test_model_file" ]; then
  step "damaged model files" "all checks passed" "$BUILD/tests/test_model_file" "$WORK"
else skip "damaged model files" "no $BUILD/tests/test_model_file"
fi
if [ -x "$BUILD/tests/test_chat_format" ]; then step "chat format" "all checks passed" "$BUILD/tests/test_chat_format"
else skip "chat format" "no $BUILD/tests/test_chat_format"
fi
# The draws of the sampler with the drafts of the answers, against the draws without them.
if [ -x "$BUILD/tests/test_sampler" ]; then step "sampler" "all checks passed" "$BUILD/tests/test_sampler"
else skip "sampler" "no $BUILD/tests/test_sampler"
fi
if [ -x "$BUILD/tests/test_engine_api" ] && [ -f "$ENGINES/libalbireo-engine-cpu.so" ]; then
  step "engine interface" "all checks passed" "$BUILD/tests/test_engine_api" "$ENGINES/libalbireo-engine-cpu.so" "$WORK"
else skip "engine interface" "no $BUILD/tests/test_engine_api or no portable engine"
fi
# The files the service writes in its models folder and reads in its imports.
if [ -x "$BUILD/tests/test_host" ]; then
  step "models folder and imports" "all checks passed" "$BUILD/tests/test_host" "$WORK" $((PORT + 1))
else skip "models folder and imports" "no $BUILD/tests/test_host"
fi

for variant in cpu-avxvnni cpu-avx2 cpu; do
  lib=$ENGINES/libalbireo-engine-$variant.so
  if [ ! -f "$lib" ]; then skip "engine $variant" "no $lib"
  elif [ ! -f "$MODEL" ]; then skip "engine $variant" "no $MODEL"
  elif ! "$CHAT" "$lib" --describe > /dev/null 2>&1; then
    skip "engine $variant" "$("$CHAT" "$lib" --describe 2>&1 | grep -o 'lacks [^(]*' | head -1)"
  else step "engine $variant" "Paris" "$CHAT" "$lib" "$MODEL" --reasoning off --max 40 "$QUESTION"
  fi
done

if [ -f "$EMBEDDING" ] && [ -f "$ENGINES/libalbireo-engine-cpu-avx2.so" ]; then
  step "embedding model" "cosine" "$CHAT" "$ENGINES/libalbireo-engine-cpu-avx2.so" "$EMBEDDING" --parallel \
    "The service runs on this computer." "Where does the service run?" "The quarterly revenue grew."
else
  skip "embedding model" "no $EMBEDDING"
fi

# The program of the speech synthesis alone: a sentence read aloud in the folder of the voices.
SPEAKER=$BUILD/bin/albireo-speech
if [ -x "$SPEAKER" ] && [ -d "$BUILD/bin/speech" ]; then
  step "speech program" "RIFF" sh -c "cd '$BUILD/bin/speech' && printf '%s' 'Guten Tag, Albireo liest vor.' \
    | '$SPEAKER' de-DE 1 > '$WORK/program.wav' && head -c 4 '$WORK/program.wav' && echo && wc -c < '$WORK/program.wav'"
else
  skip "speech program" "no $SPEAKER or no $BUILD/bin/speech"
fi

# The service, offline, in a data folder of its own.
SERVICE=$BUILD/bin/albireo-service
if [ -x "$SERVICE" ] && [ -f "$MODEL" ]; then
  HOME_DIR=$WORK/home
  mkdir -p "$HOME_DIR"
  echo "{\"port\": $PORT, \"offline\": true, \"check_updates\": false, \"default_model\": false}" \
    > "$HOME_DIR/settings.json"
  ALBIREO_HOME=$HOME_DIR "$SERVICE" --console > "$WORK/service.log" 2>&1 &
  PID=$!
  START=$(now)
  for i in $(seq 1 300); do
    curl -sk --max-time 1 https://localhost:$PORT/albireo/v1/status -o /dev/null && break
    sleep 0.1
  done
  if curl -sk --max-time 2 https://localhost:$PORT/albireo/v1/status | grep -q '"offline":true'; then
    record "service start" PASS "offline, https://localhost:$PORT" "$(elapsed "$START")"
    TOKEN=$(cat "$HOME_DIR/admin.token")
    # The license texts of the installation, none in a build folder, and a name outside their folder refused.
    step "service licenses" '^\[(\]|\{"file")' curl -sk --max-time 5 https://localhost:$PORT/albireo/v1/licenses
    step "service license refusal" '"code":"invalid_license"' curl -sk --max-time 5 --path-as-is \
      "https://localhost:$PORT/albireo/v1/licenses/../settings.json"
    # A model of the user is not read by its path: the tray sends it, read with the rights of the user.
    step "service import refusal" '"code":"import_use_upload"' curl -sk --max-time 30 -X POST \
      -H "X-Albireo-Admin: $TOKEN" -H "Content-Type: application/json" -d "{\"path\": \"$MODEL\"}" \
      https://localhost:$PORT/albireo/v1/import
    step "service import" '"imported":\["' curl -sk --max-time 600 -X POST -H "X-Albireo-Admin: $TOKEN" \
      -H "Content-Type: application/octet-stream" -H "Expect:" -T "$MODEL" \
      "https://localhost:$PORT/albireo/v1/models/upload?name=$(basename "$MODEL")"
    for i in $(seq 1 300); do
      curl -sk --max-time 1 https://localhost:$PORT/v1/models | grep -q albireo-general && break
      sleep 0.2
    done
    BODY="{\"model\": \"albireo-general\", \"reasoning\": false, \"max_tokens\": 40,"
    BODY="$BODY \"messages\": [{\"role\": \"user\", \"content\": \"$QUESTION\"}]}"
    step "service chat" "Paris" curl -sk --max-time 300 https://localhost:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" -d "$BODY"
    # The voices of the speech synthesis, then a sentence read aloud by the program the service starts.
    step "service speech voices" '"id":"fr-FR"' curl -sk --max-time 5 \
      https://localhost:$PORT/albireo/v1/speech/voices
    SPEECH="{\"input\": \"Bonjour, Albireo lit ses réponses à voix haute.\", \"voice\": \"fr-FR\"}"
    step "service speech" "RIFF" sh -c "curl -sk --max-time 60 -o '$WORK/speech.wav' \
      -H 'Content-Type: application/json' -d '$SPEECH' https://localhost:$PORT/v1/audio/speech \
      && head -c 4 '$WORK/speech.wav' && echo && wc -c < '$WORK/speech.wav'"
  else
    record "service start" FAIL "no answer, log in $WORK/service.log" 0.0
  fi
  # The service has 30 seconds to stop, then it is killed by its process number.
  START=$(now)
  kill -INT $PID 2> /dev/null
  for i in $(seq 1 300); do
    kill -0 $PID 2> /dev/null || break
    sleep 0.1
  done
  if kill -0 $PID 2> /dev/null; then
    kill -KILL $PID 2> /dev/null
    wait $PID 2> /dev/null
    record "service stop" FAIL "killed after 30 s, log in $WORK/service.log" "$(elapsed "$START")"
  else
    wait $PID
    RC=$?
    if [ $RC -eq 0 ]; then record "service stop" PASS "exit 0" "$(elapsed "$START")"
    else record "service stop" FAIL "exit $RC, log in $WORK/service.log" "$(elapsed "$START")"
    fi
  fi
else
  skip "service" "no $SERVICE or no $MODEL"
fi

echo
printf '%s\n' "${RESULTS[@]}"
echo
echo "failed steps: $FAILED"
[ $FAILED -eq 0 ] && rm -rf "$WORK"
exit $FAILED
