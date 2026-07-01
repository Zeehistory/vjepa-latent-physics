#!/bin/bash
# Shared bootstrap for ablation sbatch jobs.
# Slurm copies the job script to /var/spool/slurmd/... so dirname "$0" is NOT the repo.
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (repo root).
#
#   source "${SLURM_SUBMIT_DIR}/scripts/ablation/_job_init.sh"

set -euo pipefail

REPO_ROOT="${BASE_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT/scripts/ablation" ]; then
    echo "[job_init] ERROR: set BASE_DIR or run sbatch from the repo root (need SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
export BASE_DIR="$REPO_ROOT"
ABLATION_DIR="$REPO_ROOT/scripts/ablation"

source "$ABLATION_DIR/_encoder_env.sh"
source "$ABLATION_DIR/_activate_env.sh"
cd "$BASE_DIR"
mkdir -p logs
