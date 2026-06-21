#!/bin/bash
#SBATCH --job-name=vjepa_catprobe
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/catprobe_%j.out
#SBATCH --error=logs/catprobe_%j.err

# Step 1: category-subspace probing on Physics-IQ latents (CPU-bound sklearn; GPU not required but the
# latent cache is large, so we keep the node's memory). Set LATENT_DIR / OUTPUT_DIR or use defaults.
LATENT_DIR=${LATENT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/latents/physics_iq/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/analysis/physics_iq/category_probe"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/probe_categories.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all
