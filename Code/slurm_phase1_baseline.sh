#!/bin/bash
#SBATCH --job-name=nlp_baseline
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -p long
#SBATCH -o ./slurm-output/%x-%A.out
#SBATCH -e ./slurm-output/%x-%A.err

# ============================================================================
# Phase 1: Baseline Evaluations (M1, M2, M3)
# Runs 12 experiments: 3 models × 4 tasks
# Estimated time: 2-3 hours
# Usage: sbatch -p long slurm_phase1_baseline.sh
# ============================================================================

echo "============================================"
echo "Phase 1: Baseline Evaluations"
echo "Job running on $(hostname)"
echo "Start time: $(date)"
echo "============================================"
nvidia-smi

export HF_TOKEN="your_token"

mkdir -p slurm-output

# Activate course venv
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# Setup Local Packages
# We define the local directory using Absolute Path to avoid import errors
LOCAL_PKG_DIR="$PWD/local_packages"
export PYTHONPATH="$LOCAL_PKG_DIR:$PYTHONPATH"

# Check if hydra-core is actually installed. If not, install everything.
# We use 'python3 -c' to test import, rather than just checking directory existence.
if ! python3 -c "import hydra" &> /dev/null; then
    echo "Hydra not found. Installing dependencies to $LOCAL_PKG_DIR..."
    
    mkdir -p "$LOCAL_PKG_DIR"
    
    # Install with --target and --upgrade to ensure we get the files
    pip install --target="$LOCAL_PKG_DIR" \
        hydra-core \
        omegaconf \
        pydantic \
        outlines \
        peft \
        trl \
        bitsandbytes \
        scikit-learn \
        pandas \
        seaborn \
        matplotlib \
        --upgrade
else
    echo "Dependencies verified in $LOCAL_PKG_DIR. Skipping install."
fi

echo "Environment ready!"
echo "PYTHONPATH is: $PYTHONPATH"

python3 run_eval.py -m model=16bit,8bit,4bit task=nli_en,nli_sv,wic_en,wic_sv sample_limit=500

echo ""
echo "Phase 1 COMPLETE - End time: $(date)"
echo "Next: Run slurm_phase2_qlora.sh"
