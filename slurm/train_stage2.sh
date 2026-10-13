#!/bin/bash
#SBATCH --job-name=grape-stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/shared/users/hassan2/grape/GLIM/slurm/logs/%x_%j.out
#SBATCH --error=/shared/users/hassan2/grape/GLIM/slurm/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fouzul.h@iit.ac.lk

# ── Paths ──────────────────────────────────────────────────────────────────
# Use bin/python3 directly — bin/activate is missing in this env layout
VENV=/shared/users/hassan2/envs/physionet2026
PYTHON=$VENV/bin/python3
PIP=$VENV/bin/pip
WORKDIR=/shared/users/hassan2/grape/GLIM

echo "============================================================"
echo "  GRAPE Stage-II Training"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURM_NODELIST"
echo "  Started  : $(date)"
echo "============================================================"

# ── Verify Python ──────────────────────────────────────────────────────────
echo "Python  : $($PYTHON --version 2>&1)"
echo "PyTorch : $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed yet')"
echo ""

# ── Install / update requirements ─────────────────────────────────────────
cd "$WORKDIR"
echo ">>> pip install -r requirements.txt"
$PIP install -r requirements.txt --quiet

echo "PyTorch (after install): $($PYTHON -c 'import torch; print(torch.__version__)')"
echo ""

# ── Auth ───────────────────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment before submitting}"
export WANDB_API_KEY=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

$VENV/bin/huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
$VENV/bin/wandb login "$WANDB_API_KEY" 2>/dev/null || true

# ── GPU info ───────────────────────────────────────────────────────────────
echo ">>> GPU info"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

# ── Run training ───────────────────────────────────────────────────────────
echo ">>> Starting train_with_jepa_encoder2.py"
echo ""

$PYTHON train_with_jepa_encoder2.py

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
