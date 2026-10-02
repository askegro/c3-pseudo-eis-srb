#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
STUDY="$1"
if [ -n "${PYTHON_MODULE:-}" ]; then
  module purge
  module load $PYTHON_MODULE
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYBAMM_DISABLE_TELEMETRY=true
.venv/bin/python -B -m prbsid.execute "$STUDY" "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --out "results/$OUT_NAME/$STUDY/tasks"
