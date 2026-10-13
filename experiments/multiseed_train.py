"""
Multi-Seed Stage-II Fine-Tuning.

Trains the full GLIM (Stage II) pipeline over N seeds and aggregates
test metrics (BLEU@MTV, retrieval accuracy, zero-shot classification) to
report mean ± std — addressing reviewer requests for confidence intervals.

Usage:
    python experiments/multiseed_train.py \
        --data      data/tmp/zuco_eeg_label_8variants.df \
        --pretrain  pretraining/Results/GLIM_Pretrain5/best_model.pth \
        --seeds     5 \
        --epochs    100 \
        --out       results/multiseed_v2.json

    # Quick smoke-test (2 seeds, 10 epochs):
    python experiments/multiseed_train.py \
        --data data/tmp/zuco_eeg_label_8variants.df \
        --seeds 2 --epochs 10 --out /tmp/multiseed_smoke.json
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',     required=True)
    p.add_argument('--pretrain', default=None,
                   help='Pretrained Stage-I encoder checkpoint (.pth). '
                        'If omitted, trains Stage II from random init.')
    p.add_argument('--seeds',  type=int, default=5)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--gpus',   default='0')
    p.add_argument('--freeze_encoder', action='store_true',
                   help='Freeze pretrained encoder during Stage II')
    p.add_argument('--out', required=True,
                   help='Output JSON path for aggregated results')
    p.add_argument('--bsz_train', type=int, default=72)
    p.add_argument('--lr',  type=float, default=1e-4)
    p.add_argument('--full_val_interval', type=int, default=10,
                   help='Run full validation every N epochs')
    return p.parse_args()


# ── Per-seed trainer ──────────────────────────────────────────────────────────

def train_one_seed(seed: int, args, tmp_dir: Path) -> dict | None:
    """Spawn a separate Python process so seed isolation is clean."""
    script = tmp_dir / f'seed_{seed:03d}_train.py'
    result_path = tmp_dir / f'seed_{seed:03d}_results.json'

    pretrain_line = (
        f"    model = load_pretrained_encoder(model, '{args.pretrain}')\n"
        if args.pretrain else ""
    )
    freeze_line = (
        "    model = freeze_pretrained_encoder(model, freeze_in_blocks=True)\n"
        if args.freeze_encoder else ""
    )

    code = f"""\
import os, sys, json, torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, '.')
from model.glim import GLIM
from data.datamodule import GLIMDataModule
from pretraining.load_pretrained import load_pretrained_encoder, freeze_pretrained_encoder

SEED = {seed}
L.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision('medium')

group_name = f'multiseed-s{{SEED}}'
log_dir = '{str(tmp_dir)}/runs/s{{SEED}}'
os.makedirs(log_dir, exist_ok=True)

logger = None  # no WandB for reproducibility runs; add if needed

full_val_interval = {args.full_val_interval}
callbacks = [ModelCheckpoint(monitor='epoch',
    dirpath=log_dir + '/ckpts',
    save_top_k=-1, every_n_epochs=full_val_interval)]

trainer = L.Trainer(
    accelerator='gpu',
    devices=[{args.gpus.split(',')[0]}],
    logger=logger,
    max_epochs={args.epochs},
    precision='bf16-mixed',
    enable_checkpointing=True,
    callbacks=callbacks,
    use_distributed_sampler=False,
    num_sanity_val_steps=0,
)

dm = GLIMDataModule(
    data_path='{args.data}',
    bsz_train={args.bsz_train},
    bsz_val=24,
    num_workers=4,
)

model = GLIM(
    input_eeg_len=1280, hidden_eeg_len=96,
    input_text_len=96, tgt_text_len=64,
    input_dim=128, hidden_dim=256, embed_dim=1024,
    text_model_id='google/flan-t5-large',
    prompt_nums=(3, 3, 31),
    prompt_dropout_probs=(0.0, 1.0, 1.0),
    evaluate_prompt_embed='src',
    n_in_blocks=6, n_out_blocks=6,
    in_temporal_modulate=True, out_is_causal=True,
    use_gated_attention=True, gating_type='elementwise',
    prompt_tuning_len=0, dropout=0,
    clip_loss_weight=0.5, commitment_loss_weight=0.0,
    commitment_loss_key='mse',
    bsz_train=dm.bsz_train, bsz_val=dm.bsz_val,
    lr={args.lr}, weight_decay=0,
    full_val_interval=full_val_interval, bs_retrieval=24,
)
{pretrain_line}{freeze_line}

trainer.fit(model, datamodule=dm)
results = trainer.test(model, datamodule=dm)
metrics = results[0] if results else {{}}
metrics['seed'] = SEED

with open('{str(result_path)}', 'w') as f:
    json.dump({{k: (float(v) if hasattr(v,'item') else v)
               for k, v in metrics.items()}}, f, indent=2)
print(f"Seed {{SEED}} done → {str(result_path)}")
"""
    script.write_text(code)
    ret = subprocess.run([sys.executable, str(script)], cwd='.',
                         capture_output=False)
    if ret.returncode != 0:
        print(f"  ⚠️ Seed {seed} failed (returncode={ret.returncode})")
        return None
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return None


# ── Aggregation ───────────────────────────────────────────────────────────────

KEY_METRICS = [
    'test/mean_BLEU1@MTV',
    'test/mean_BLEU2@MTV',
    'test/mean_retrieval_acc_top01',
    'test/mean_retrieval_acc_top05',
    'test/mean_retrieval_acc_top10',
    'test/mean_relation_cls_acc_top01',
    'test/mean_relation_cls_acc_top03',
    'test/mean_corpus_cls_acc',
    'test/mean_sentiment_cls_acc_top01',
]


def aggregate(seed_results: list[dict]) -> dict:
    out = {'n_seeds': len(seed_results), 'per_seed': seed_results, 'summary': {}}
    for k in KEY_METRICS:
        vals = [r[k] for r in seed_results if k in r]
        if not vals:
            continue
        arr = np.array(vals)
        out['summary'][k] = {
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'min':  float(arr.min()),
            'max':  float(arr.max()),
            'n':    len(arr),
        }
    return out


def print_summary(agg: dict):
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY  (N={agg['n_seeds']} seeds)")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for k, v in agg['summary'].items():
        short = k.replace('test/', '').replace('mean_', '')
        print(f"  {short:<40} {v['mean']:>8.4f} {v['std']:>8.4f}"
              f" {v['min']:>8.4f} {v['max']:>8.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='grape_multiseed_') as tmp:
        tmp_dir = Path(tmp)
        seed_results = []
        for seed in range(args.seeds):
            print(f"\n{'─'*70}")
            print(f"  Running seed {seed}/{args.seeds - 1} ...")
            print(f"{'─'*70}")
            res = train_one_seed(seed, args, tmp_dir)
            if res:
                seed_results.append(res)
                print(f"  Seed {seed}: OK | "
                      f"BLEU1={res.get('test/mean_BLEU1@MTV', float('nan')):.4f}")
            else:
                print(f"  Seed {seed}: FAILED — skipped")

    if not seed_results:
        print("No seeds completed successfully.")
        return

    agg = aggregate(seed_results)
    print_summary(agg)

    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results saved → {out_path}")


if __name__ == '__main__':
    main()
