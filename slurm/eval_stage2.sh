#!/bin/bash
#SBATCH --job-name=grape-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=/shared/users/hassan2/grape/GLIM/slurm/logs/%x_%j.out
#SBATCH --error=/shared/users/hassan2/grape/GLIM/slurm/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fouzul.h@iit.ac.lk

VENV=/shared/users/hassan2/envs/physionet2026
PYTHON=$VENV/bin/python3
WORKDIR=/shared/users/hassan2/grape/GLIM
CKPT="runs/dev-dist/wandb/run-20260809_114002-v1pkxohf/files-checkpoints/epoch=139-step=278320.ckpt"

echo "============================================================"
echo "  GRAPE Evaluation"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURM_NODELIST"
echo "  Checkpoint: $CKPT"
echo "  Started  : $(date)"
echo "============================================================"

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment before submitting}"
export WANDB_API_KEY=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

$VENV/bin/wandb login "$WANDB_API_KEY" 2>/dev/null || true

cd "$WORKDIR"

echo ""
echo ">>> Running test.py"
$PYTHON test.py

echo ""
echo ">>> Running predict.py --task all"
$PYTHON predict.py --checkpoint "$CKPT" --task all

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
