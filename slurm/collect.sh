#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/common.sh"
ensure_env || exit 1
cd "$ROOT"
missing=0
for study in main extra; do
  ids=$("$PY" -B slurm/pending.py ids "$study" "$CONFIG" "results/$OUT_NAME/$study/tasks") || exit 1
  [ -n "$ids" ] && { echo "$study: tasks not complete: $ids"; missing=1; }
done
[ $missing -eq 0 ] || { echo "run slurm/status.sh, then submit.sh again"; exit 1; }
"$PY" -B run.py report "$OUT_NAME"
