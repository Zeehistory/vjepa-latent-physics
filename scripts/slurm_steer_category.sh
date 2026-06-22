#!/bin/bash
#SBATCH --job-name=vjepa_catsteer
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/catsteer_%j.out
#SBATCH --error=logs/catsteer_%j.err

# Step 3 (category steering): steer real fluid Physics-IQ clips toward "solid" along (w_solid - w_fluid)
# and decode the result. Requires:
#   * the category probe directions (slurm_probe_categories.sh -> category_directions.npz)
#   * Physics-IQ latents and a trained transformer decoder.
# DIRECTIONS defaults to the standardized probe's npz; point it at .../raw/ to use the un-normalized run.
TARGET_LATENT_DIR=${TARGET_LATENT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/latents/physics_iq/vjepa2_large"}
DIRECTIONS=${DIRECTIONS:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/analysis/physics_iq/category_probe/standardized/category_directions.npz"}
CHECKPOINT=${CHECKPOINT:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
FROM_CATEGORY=${FROM_CATEGORY:-"fluid_dynamics"}
TO_CATEGORY=${TO_CATEGORY:-"solid_mechanics"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/analysis/steer_category/${FROM_CATEGORY}_to_${TO_CATEGORY}"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/steer_category.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --target_latent_dir "$TARGET_LATENT_DIR" \
    --directions "$DIRECTIONS" \
    --checkpoint "$CHECKPOINT" \
    --from_category "$FROM_CATEGORY" \
    --to_category "$TO_CATEGORY" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
