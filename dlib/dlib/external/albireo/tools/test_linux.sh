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
# The service runs offline in a data folder of its own, on port 18443, imports the model,
# answers a question through the OpenAI interface and stops.  The exit code is the number of
# failed steps.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)
BUILD=${1:-$ROOT/build-albireo}
MODEL=${2:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
EMBEDDING=${3:-$HOME/models/Qwen3-Embedding-0.6B-Q8_0.gguf}
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

ENGINES=$BUILD/bin/engines
CHAT=$BUILD/tests/engine_chat

# The kernels of each family of processors, skipped by the processors without their instructions.
for family in "" _avxvnni _avx512 _vnni256; do
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

for variant in cpu-avx512 cpu-avxvnni cpu-avx2 cpu; do
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

# The service, offline, in a data folder of its own.
SERVICE=$BUILD/bin/albireo-service
if [ -x "$SERVICE" ] && [ -f "$MODEL" ]; then
  HOME_DIR=$WORK/home
  mkdir -p "$HOME_DIR"
  echo '{"port": 18443, "offline": true, "check_updates": false, "default_model": false}' > "$HOME_DIR/settings.json"
  ALBIREO_HOME=$HOME_DIR "$SERVICE" --console > "$WORK/service.log" 2>&1 &
  PID=$!
  START=$(now)
  for i in $(seq 1 300); do
    curl -sk --max-time 1 https://localhost:18443/albireo/v1/status -o /dev/null && break
    sleep 0.1
  done
  if curl -sk --max-time 2 https://localhost:18443/albireo/v1/status | grep -q '"offline":true'; then
    record "service start" PASS "offline, https://localhost:18443" "$(elapsed "$START")"
    TOKEN=$(cat "$HOME_DIR/admin.token")
    step "service import" '"imported":\["' curl -sk --max-time 300 -X POST -H "X-Albireo-Admin: $TOKEN" \
      -H "Content-Type: application/json" -d "{\"path\": \"$MODEL\"}" https://localhost:18443/albireo/v1/import
    for i in $(seq 1 300); do
      curl -sk --max-time 1 https://localhost:18443/v1/models | grep -q albireo-general && break
      sleep 0.2
    done
    BODY="{\"model\": \"albireo-general\", \"reasoning\": false, \"max_tokens\": 40,"
    BODY="$BODY \"messages\": [{\"role\": \"user\", \"content\": \"$QUESTION\"}]}"
    step "service chat" "Paris" curl -sk --max-time 300 https://localhost:18443/v1/chat/completions \
      -H "Content-Type: application/json" -d "$BODY"
  else
    record "service start" FAIL "no answer, log in $WORK/service.log" 0.0
  fi
  START=$(now)
  kill -INT $PID 2> /dev/null
  wait $PID
  RC=$?
  if [ $RC -eq 0 ]; then record "service stop" PASS "exit 0" "$(elapsed "$START")"
  else record "service stop" FAIL "exit $RC, log in $WORK/service.log" "$(elapsed "$START")"
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
