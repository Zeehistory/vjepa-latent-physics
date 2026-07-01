#!/bin/bash
# Activate conda in non-interactive sbatch jobs (match Zahid's pattern).
#   source "${SLURM_SUBMIT_DIR}/scripts/ablation/_activate_env.sh"

set -euo pipefail

module purge
module load miniconda

if ! command -v conda >/dev/null 2>&1; then
    echo "[activate] ERROR: conda not on PATH after 'module load miniconda'" >&2
    exit 1
fi

conda activate "${CONDA_ENV:-vjepa-physics-decoder}"
echo "[activate] python=$(which python) CONDA_PREFIX=${CONDA_PREFIX:-}"
