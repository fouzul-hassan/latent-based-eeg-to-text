#!/bin/bash
#SBATCH --job-name=bootstrap_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=iit-MS-7E06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hassan2/shared/users/hassan2/grape/GLIM/bootstrap_v2_%j.out
#SBATCH --error=/home/hassan2/shared/users/hassan2/grape/GLIM/bootstrap_v2_%j.err

echo "=== Bootstrap CI v2 Starting ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Date: $(date)"

PYTHON=/shared/users/hassan2/envs/physionet2026/bin/python3
echo "Python: $PYTHON"
$PYTHON --version

cd /home/hassan2/shared/users/hassan2/grape/GLIM

$PYTHON bootstrap_ci.py \
    --n_boot 1000 \
    --seed 42 \
    --eval_dir evaluations \
    --versions v2

echo ""
echo "=== v2 Done: $(date) ==="
