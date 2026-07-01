#!/bin/bash
# Activate the project conda env inside non-interactive sbatch jobs.
# Interactive shells auto-init conda; batch jobs do not — without the hook, `conda activate` fails instantly.
#
#   source "$(dirname "$0")/_activate_env.sh"

set -euo pipefail

module purge
if module load miniconda 2>/dev/null; then
    :
elif module load Anaconda3 2>/dev/null; then
    :
else
    echo "[activate] ERROR: could not load miniconda or Anaconda3 module" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-vjepa-physics-decoder}"

echo "[activate] python=$(which python) base=${CONDA_PREFIX:-}"
