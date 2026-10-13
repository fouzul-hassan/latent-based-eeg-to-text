# Ablation Study Status

> Last updated: 2026-04-24

## Overview

| ID  | Ablation                        | Status      | Cost     | Notes                  |
|-----|---------------------------------|-------------|----------|------------------------|
| A1  | Random-init linear probe        | RUNNING     | Free     | CPU-only               |
| A2  | Noise-input variance (10 seeds) | QUEUED      | Free     | Inference only (v2)    |
| A3  | SSP vs random masking           | NOT STARTED | ~$14     | Needs approval         |
| A4  | VICReg vs MSE-only loss         | NOT STARTED | ~$14     | Needs approval         |
| A5  | Gated vs standard attention     | DONE        | -        | Already in checkpoints |
| A6  | Random-init fine-tuning         | NOT STARTED | ~$14     | Optional               |

## Configuration

- **Canonical baseline**: GLIM_Pretrain5 (8-block, emb=256, heads=16, patch=8, wd=0.05, mask_ratio=0.7)
- **Fine-tuned model**: runs/v2/epoch=199-step=397600.ckpt (GRAPE-GLIM-G)
- **Data**: zuco_eeg_label_8variants.df (text-disjoint 8:1:1 split)

## Results

### A1 — Random-Init Encoder Linear Probe
- **Status**: Running...
- **Logs**: `ablations/logs/a1/`

### A2 — Noise-Input Variance
- **Status**: Queued (after A1)
- **Logs**: `ablations/logs/a2/`

### A5 — Gated vs Standard Attention
- **Status**: Already available in existing checkpoints
- Pretrain5: standard attention (no gated)
- v2 fine-tuning: gated attention (elementwise)000
