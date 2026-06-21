#!/bin/bash
#SBATCH --job-name=vjepa_extract_syn
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/extract_syn_%j.out
#SBATCH --error=logs/extract_syn_%j.err

# Step 2: extract VJEPA latents for a synthetic-physics (or robot_toy) dataset with exact GT labels.
# The dataset is generated on the fly, so no data.root is needed.
#   DATASET=synthetic_solid sbatch scripts/slurm_extract_synthetic.sh
#   DATASET=synthetic_fluid sbatch scripts/slurm_extract_synthetic.sh
#   DATASET=robot_toy       sbatch scripts/slurm_extract_synthetic.sh
DATASET=${DATASET:-"synthetic_solid"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/zss8/project_pi_jks79/zss8/vjepa/outputs/latents/$DATASET/vjepa2_large"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python scripts/extract_latents.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --dataset "$DATASET" \
    --encoder vjepa2_large \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --shard_size 128
