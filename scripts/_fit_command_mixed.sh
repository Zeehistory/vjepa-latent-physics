#!/bin/bash
#SBATCH --job-name=v2dmix_cmdop
#SBATCH --partition=bigmem
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2dmix_cmdop_%j.out
#SBATCH --error=logs/v2dmix_cmdop_%j.err
# Fit COMMAND-ONLY subspace-synthesis operators on the appearance-mixed cache (W_U: command -> U-coords;
# B_rich: rich command -> dH). KU defaults to 8; set KU=16 for the U16 fallback (needs save_k>=16 basis).
# ALWAYS submit via the queue-aware picker (user directive):
#   PART=$(scripts/_pick_partition.sh 256000 bigmem mpi week day); sbatch --partition="$PART" --mem=256G scripts/_fit_command_mixed.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=/home/zss8/project_pi_jks79/zss8/vjepa
python -u scripts/fit_command_operators.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_v2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_v2d_mixed/test/vjepa2_large \
    --layers ${LAYERS:-6,12,18,23} --ridge 1.0 --ku ${KU:-8} \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_v2d_mixed/subspace
echo "[v2dmix_cmdop] exit=$? (KU=${KU:-8})"
