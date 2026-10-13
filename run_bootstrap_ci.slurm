#!/bin/bash
#SBATCH --job-name=bootstrap_ci
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/hassan2/shared/users/hassan2/grape/GLIM/bootstrap_ci_%j.out
#SBATCH --error=/home/hassan2/shared/users/hassan2/grape/GLIM/bootstrap_ci_%j.err

echo "=== Bootstrap CI Job Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Date:   $(date)"

# Activate conda environment
source /shared/miniforge3/etc/profile.d/conda.sh
conda activate physionet2026

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

cd /home/hassan2/shared/users/hassan2/grape/GLIM

python bootstrap_ci.py \
    --n_boot 1000 \
    --seed 42 \
    --eval_dir evaluations \
    --versions v1 v2

echo ""
echo "=== Job Finished: $(date) ==="
