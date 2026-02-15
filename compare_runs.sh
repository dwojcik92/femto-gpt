#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BASE="${ROOT_DIR}/run_results"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-${OUT_BASE}/${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found (python/python3)." >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"

run_model() {
  local name="$1"
  local file="$2"
  local log="${OUT_DIR}/${name}.log"
  local status_file="${OUT_DIR}/${name}.status"
  local seconds_file="${OUT_DIR}/${name}.seconds"
  local start end status seconds

  echo "Running ${name} (${file})..."
  start="$(date +%s)"
  set +e
  (cd "$ROOT_DIR" && "$PYTHON_BIN" -u "$file") >"$log" 2>&1
  status="$?"
  set -e
  end="$(date +%s)"
  seconds="$((end - start))"

  printf "%s\n" "$status" >"$status_file"
  printf "%s\n" "$seconds" >"$seconds_file"
  echo "Finished ${name}: status=${status}, elapsed=${seconds}s"
}

metric_value() {
  local pattern="$1"
  local file="$2"
  grep -m1 "^${pattern}" "$file" | cut -d':' -f2- | sed 's/^ *//'
}

last_field() {
  local pattern="$1"
  local file="$2"
  grep "^${pattern}" "$file" | tail -n1 | awk '{print $NF}'
}

run_model "microgpt" "microgpt.py"
run_model "femtogpt" "femtogpt.py"
run_model "picogpt" "picogpt.py"

MICRO_LOG="${OUT_DIR}/microgpt.log"
FEMTO_LOG="${OUT_DIR}/femtogpt.log"
PICO_LOG="${OUT_DIR}/picogpt.log"
MICRO_STATUS="$(cat "${OUT_DIR}/microgpt.status")"
FEMTO_STATUS="$(cat "${OUT_DIR}/femtogpt.status")"
PICO_STATUS="$(cat "${OUT_DIR}/picogpt.status")"
MICRO_SECONDS="$(cat "${OUT_DIR}/microgpt.seconds")"
FEMTO_SECONDS="$(cat "${OUT_DIR}/femtogpt.seconds")"
PICO_SECONDS="$(cat "${OUT_DIR}/picogpt.seconds")"

grep '^step ' "$MICRO_LOG" >"${OUT_DIR}/microgpt.losses" || true
grep '^step ' "$FEMTO_LOG" >"${OUT_DIR}/femtogpt.losses" || true
grep '^step ' "$PICO_LOG" >"${OUT_DIR}/picogpt.losses" || true
grep '^sample ' "$MICRO_LOG" >"${OUT_DIR}/microgpt.samples" || true
grep '^sample ' "$FEMTO_LOG" >"${OUT_DIR}/femtogpt.samples" || true
grep '^sample ' "$PICO_LOG" >"${OUT_DIR}/picogpt.samples" || true

if cmp -s "$MICRO_LOG" "$FEMTO_LOG"; then FULL_MATCH_FEMTO="yes"; else FULL_MATCH_FEMTO="no"; fi
if cmp -s "$MICRO_LOG" "$PICO_LOG"; then FULL_MATCH_PICO="yes"; else FULL_MATCH_PICO="no"; fi

if diff -u "${OUT_DIR}/microgpt.losses" "${OUT_DIR}/femtogpt.losses" >"${OUT_DIR}/losses.diff"; then LOSSES_MATCH_FEMTO="yes"; else LOSSES_MATCH_FEMTO="no"; fi
if diff -u "${OUT_DIR}/microgpt.losses" "${OUT_DIR}/picogpt.losses" >"${OUT_DIR}/pico_losses.diff"; then LOSSES_MATCH_PICO="yes"; else LOSSES_MATCH_PICO="no"; fi

if diff -u "${OUT_DIR}/microgpt.samples" "${OUT_DIR}/femtogpt.samples" >"${OUT_DIR}/samples.diff"; then SAMPLES_MATCH_FEMTO="yes"; else SAMPLES_MATCH_FEMTO="no"; fi
if diff -u "${OUT_DIR}/microgpt.samples" "${OUT_DIR}/picogpt.samples" >"${OUT_DIR}/pico_samples.diff"; then SAMPLES_MATCH_PICO="yes"; else SAMPLES_MATCH_PICO="no"; fi

MICRO_DOCS="$(metric_value 'num docs:' "$MICRO_LOG" || true)"
FEMTO_DOCS="$(metric_value 'num docs:' "$FEMTO_LOG" || true)"
PICO_DOCS="$(metric_value 'num docs:' "$PICO_LOG" || true)"
MICRO_VOCAB="$(metric_value 'vocab size:' "$MICRO_LOG" || true)"
FEMTO_VOCAB="$(metric_value 'vocab size:' "$FEMTO_LOG" || true)"
PICO_VOCAB="$(metric_value 'vocab size:' "$PICO_LOG" || true)"
MICRO_PARAMS="$(metric_value 'num params:' "$MICRO_LOG" || true)"
FEMTO_PARAMS="$(metric_value 'num params:' "$FEMTO_LOG" || true)"
PICO_PARAMS="$(metric_value 'num params:' "$PICO_LOG" || true)"
MICRO_STEP1="$(grep -m1 '^step ' "$MICRO_LOG" | awk '{print $NF}' || true)"
FEMTO_STEP1="$(grep -m1 '^step ' "$FEMTO_LOG" | awk '{print $NF}' || true)"
PICO_STEP1="$(grep -m1 '^step ' "$PICO_LOG" | awk '{print $NF}' || true)"
MICRO_LAST="$(last_field 'step ' "$MICRO_LOG" || true)"
FEMTO_LAST="$(last_field 'step ' "$FEMTO_LOG" || true)"
PICO_LAST="$(last_field 'step ' "$PICO_LOG" || true)"
MICRO_STEP_COUNT="$(grep -c '^step ' "$MICRO_LOG" || true)"
FEMTO_STEP_COUNT="$(grep -c '^step ' "$FEMTO_LOG" || true)"
PICO_STEP_COUNT="$(grep -c '^step ' "$PICO_LOG" || true)"
MICRO_SAMPLE_COUNT="$(grep -c '^sample ' "$MICRO_LOG" || true)"
FEMTO_SAMPLE_COUNT="$(grep -c '^sample ' "$FEMTO_LOG" || true)"
PICO_SAMPLE_COUNT="$(grep -c '^sample ' "$PICO_LOG" || true)"

SUMMARY_FILE="${OUT_DIR}/summary.txt"
cat >"$SUMMARY_FILE" <<EOF
Output directory: ${OUT_DIR}
Python interpreter: ${PYTHON_BIN}

microgpt status=${MICRO_STATUS} elapsed=${MICRO_SECONDS}s
femtogpt status=${FEMTO_STATUS} elapsed=${FEMTO_SECONDS}s
picogpt status=${PICO_STATUS} elapsed=${PICO_SECONDS}s

num docs:    micro=${MICRO_DOCS} femto=${FEMTO_DOCS} pico=${PICO_DOCS}
vocab size:  micro=${MICRO_VOCAB} femto=${FEMTO_VOCAB} pico=${PICO_VOCAB}
num params:  micro=${MICRO_PARAMS} femto=${FEMTO_PARAMS} pico=${PICO_PARAMS}

first loss:  micro=${MICRO_STEP1} femto=${FEMTO_STEP1} pico=${PICO_STEP1}
last loss:   micro=${MICRO_LAST} femto=${FEMTO_LAST} pico=${PICO_LAST}
step count:  micro=${MICRO_STEP_COUNT} femto=${FEMTO_STEP_COUNT} pico=${PICO_STEP_COUNT}
sample count: micro=${MICRO_SAMPLE_COUNT} femto=${FEMTO_SAMPLE_COUNT} pico=${PICO_SAMPLE_COUNT}

femtogpt vs microgpt:
- full log match: ${FULL_MATCH_FEMTO}
- loss trace match: ${LOSSES_MATCH_FEMTO}
- sample output match: ${SAMPLES_MATCH_FEMTO}

picogpt vs microgpt:
- full log match: ${FULL_MATCH_PICO}
- loss trace match: ${LOSSES_MATCH_PICO}
- sample output match: ${SAMPLES_MATCH_PICO}

Logs:
- ${MICRO_LOG}
- ${FEMTO_LOG}
- ${PICO_LOG}
Diffs:
- ${OUT_DIR}/losses.diff
- ${OUT_DIR}/samples.diff
- ${OUT_DIR}/pico_losses.diff
- ${OUT_DIR}/pico_samples.diff
EOF

cat "$SUMMARY_FILE"
