#!/bin/bash
#SBATCH --job-name=nlp_plot
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o ./slurm-output/plot-%j.out
#SBATCH -e ./slurm-output/plot-%j.err

# ============================================================================
# Plotting Job
# Runs manual-plotting.py to generate training graphs
# Usage: sbatch slurm_plotting.sh
# ============================================================================

echo "============================================"
echo "Job running on $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# 1. Setup Environment
# Activate the course virtual environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# 2. Setup Local Packages
# We point to the same directory used in the QLoRA step to find libraries
LOCAL_PKG_DIR="$PWD/local_packages"
export PYTHONPATH="$LOCAL_PKG_DIR:$PYTHONPATH"

echo "PYTHONPATH set to include: $LOCAL_PKG_DIR"

# 3. Install Plotting Dependencies (Safety Check)
# Ensures matplotlib/seaborn are installed in your local directory if missing
echo "Checking plotting dependencies..."
pip install --target="$LOCAL_PKG_DIR" \
    --no-cache-dir \
    pandas \
    matplotlib \
    seaborn

# 4. Run the Plotting Script
echo "Running manual-plotting.py..."
python3 manual-plotting.py

echo "============================================"
echo "Plotting COMPLETE - End time: $(date)"
echo "Check for 'training_metrics.png' in the current directory."
echo "============================================"