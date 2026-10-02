#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/common.sh"
ensure_env || exit 1
cd "$ROOT"
squeue -u "${USER:-$(id -un)}" -n prbsid_main,prbsid_extra -o "%.12i %.14j %.9T %.10M %R" 2>/dev/null
for study in main extra; do
  "$PY" -B slurm/pending.py status "$study" "$CONFIG" "results/$OUT_NAME/$study/tasks"
done
find "results/$OUT_NAME/logs" -name "*.err" -size +0 2>/dev/null | head -20
