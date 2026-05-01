#!/bin/bash
#SBATCH --job-name=nlp_m4_eval
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -p long
#SBATCH -o ./slurm-output/%x-%A.out
#SBATCH -e ./slurm-output/%x-%A.err

# Stop the script if any command fails
set -e 

# ============================================================================
# Phase 3: M4 Evaluation + Results Aggregation
# Runs 4 experiments on fine-tuned model + generates summary
# Estimated time: 45 min - 1 hour
# Usage: sbatch -p long slurm_phase3_m4_eval.sh
# ============================================================================

echo "============================================"
echo "Phase 3: M4 Evaluation & Results"
echo "Job running on $(hostname)"
echo "Start time: $(date)"
echo "============================================"
nvidia-smi

mkdir -p slurm-output

# Activate course venv and add local packages
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
export PYTHONPATH="./local_packages:$PYTHONPATH"

# Install additional packages for figures/tables
echo "Installing inference dependencies..."
pip install --target=./local_packages \
    hydra-core \
    omegaconf \
    pydantic \
    outlines \
    tabulate \
    matplotlib \
    seaborn \
    scikit-learn \
    -q

# Evaluate M4 on all tasks
echo "Generating M4 evaluation results..."
python3 run_eval.py -m model=4bit_adapter task=nli_sv,wic_sv,nli_en,wic_en

# Aggregate all results
echo ""
echo "Aggregating all results..."
python3 scripts/aggregate_results.py

# Generate figures for report
echo ""
echo "Generating figures and tables..."
python3 scripts/generate_figures.py

echo ""
echo "============================================"
echo "ALL PHASES COMPLETE!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Output locations:"
echo "  - Raw results: results/<date>/<time>/"
echo "  - Figures: figures/"
echo "  - Tables: tables/"
