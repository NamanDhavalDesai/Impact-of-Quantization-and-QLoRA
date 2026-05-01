#!/bin/bash
#SBATCH --job-name=nlp_qlora
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -p long
#SBATCH -o ./slurm-output/%x-%A.out
#SBATCH -e ./slurm-output/%x-%A.err

# ============================================================================
# Phase 2: QLoRA Fine-tuning (Creating M4)
# Trains on 5,000 Swedish samples
# Estimated time: 1-2 hours
# Usage: sbatch -p long slurm_phase2_qlora.sh
# ============================================================================

echo "============================================"
echo "Phase 2: QLoRA Fine-tuning"
echo "Job running on $(hostname)"
echo "Start time: $(date)"
echo "============================================"
nvidia-smi

export HF_TOKEN="your_token"

mkdir -p slurm-output

# Activate course venv and add local packages
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# Setup Local Packages
LOCAL_PKG_DIR="$PWD/local_packages"

# Remove old local packages
echo "Cleaning old local packages..."
rm -rf "$LOCAL_PKG_DIR"
mkdir -p "$LOCAL_PKG_DIR"

export PYTHONPATH="$LOCAL_PKG_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# Force install dependecies
echo "Installing/Updating critical dependencies..."
pip install --target="$LOCAL_PKG_DIR" \
    --upgrade \
    --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.29.0" \
    "bitsandbytes>=0.43.1" \
    "peft>=0.10.0" \
    "datasets>=2.19.0" \
    "typer" \
    "docstring_parser" \
    "rich" 

echo "Installing TRL (Force Update)..."
pip install --target="$LOCAL_PKG_DIR" \
    --upgrade \
    --no-cache-dir \
    --no-deps \
    "trl>=0.8.6"

echo "Starting Training Script..."
python3 scripts/train_qlora.py --max_samples 5000 --epochs 1

echo ""
echo "Phase 2 COMPLETE - End time: $(date)"
echo "Adapters saved to: results/adapters/m4_swedish/"
echo "Next: Run slurm_phase3_m4_eval.sh"