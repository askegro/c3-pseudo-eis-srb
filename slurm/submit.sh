#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/common.sh"
studies=$(study_list "${1:-all}") || exit 1
ensure_env || exit 1
"$PY" -B "$ROOT/tests/test_theory.py" | tail -1
cd "$ROOT"
echo "account: ${SBATCH_ACCOUNT:-cluster default}  partition: ${SBATCH_PARTITION:-cluster default}  module: ${PYTHON_MODULE:-none}"
for study in $studies; do
  out="results/$OUT_NAME/$study/tasks"
  mkdir -p "$out" "results/$OUT_NAME/logs"
  queued=$(squeue -h -r -u "${USER:-$(id -un)}" -n "prbsid_$study" -o "%K" 2>/dev/null | tr '\n' ' ')
  ids=$("$PY" -B slurm/pending.py ids "$study" "$CONFIG" "$out" "$queued") || exit 1
  if [ -z "$ids" ]; then
    echo "$study: every task is complete or queued"
    continue
  fi
  limit=03:00:00
  [ "$study" = extra ] && limit=02:00:00
  job=$(sbatch --parsable --job-name="prbsid_$study" --time="$limit" --array="$ids" \
        --output="results/$OUT_NAME/logs/${study}_%A_%a.out" --error="results/$OUT_NAME/logs/${study}_%A_%a.err" \
        slurm/task.sbatch "$study") || exit 1
  echo "$study: submitted job $job, tasks $ids"
done
