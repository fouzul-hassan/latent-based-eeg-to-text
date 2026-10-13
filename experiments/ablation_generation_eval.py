"""
Generation-Level Ablation Evaluation (Stage II).

Addresses reviewer concern (EcLa W4): "The ablation study does not show whether
representation-level gains translate to generation-level gains. It only reports
frozen-encoder linear-probe results, without evaluating the ablated models in the
full Stage II text-generation pipeline."

For each ablated pretrained encoder (A1/A2/A3 and the GRAPE V1/V2 baselines):
  1. Load the pretrained encoder into the full GLIM model.
  2. Fine-tune Stage II for `--epochs` epochs.
  3. Run test evaluation and save generation metrics.

All configs share the same Stage-II hyperparameters for fair comparison.

Usage:
    python experiments/ablation_generation_eval.py \
        --data  data/tmp/zuco_eeg_label_8variants.df \
        --ablations \
            runs/ablation_a1/best_model.pth \
            runs/ablation_a2/best_model.pth \
            runs/ablation_a3/best_model.pth \
        --baselines \
            pretraining/Results/GLIM_Pretrain1/best_model.pth \
            pretraining/Results/GLIM_Pretrain5/best_model.pth \
        --names A1_RandomMask A2_NoVICReg A3_MAE GRAPE_V1 GRAPE_V2 \
        --epochs 100 \
        --out results/ablation_generation.json

    # Quick run (10 epochs per model):
    python experiments/ablation_generation_eval.py \
        --data  data/tmp/zuco_eeg_label_8variants.df \
        --ablations runs/ablation_a1/best_model.pth \
        --names A1_RandomMask \
        --epochs 10 --out /tmp/abl_gen_smoke.json
"""
import os
import sys
import json
import argparse
import copy
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.glim import GLIM
from data.datamodule import GLIMDataModule
from pretraining.load_pretrained import load_pretrained_encoder

SEED = 42

REPORT_KEYS = [
    'test/mean_BLEU1@MTV',
    'test/mean_BLEU2@MTV',
    'test/mean_ROUGE1@MTV',
    'test/mean_retrieval_acc_top01',
    'test/mean_retrieval_acc_top05',
    'test/mean_retrieval_acc_top10',
    'test/mean_relation_cls_acc_top01',
    'test/mean_relation_cls_acc_top03',
    'test/mean_corpus_cls_acc',
    'test/mean_sentiment_cls_acc_top01',
]


def build_model(dm: GLIMDataModule) -> GLIM:
    return GLIM(
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
        lr=1e-4, weight_decay=0,
        full_val_interval=10, bs_retrieval=24,
    )


def run_one(encoder_ckpt: str, name: str, dm: GLIMDataModule,
            epochs: int, gpu: int, out_dir: Path) -> dict:
    print(f"\n{'='*70}")
    print(f"  [{name}]  encoder: {encoder_ckpt}")
    print(f"{'='*70}")

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision('medium')

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        accelerator='gpu',
        devices=[gpu],
        logger=False,
        max_epochs=epochs,
        precision='bf16-mixed',
        enable_checkpointing=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )

    model = build_model(dm)

    if encoder_ckpt and os.path.exists(encoder_ckpt):
        print(f"  Loading pretrained encoder from {encoder_ckpt}")
        model = load_pretrained_encoder(model, encoder_ckpt)
    else:
        print(f"  ⚠️  Checkpoint not found: {encoder_ckpt!r} — training from scratch")

    trainer.fit(model, datamodule=dm)
    test_results = trainer.test(model, datamodule=dm)
    metrics = test_results[0] if test_results else {}

    report = {
        'name': name,
        'encoder_ckpt': encoder_ckpt,
        'epochs': epochs,
        'seed': SEED,
        'metrics': {k: (float(v) if hasattr(v, 'item') else v)
                    for k, v in metrics.items()},
    }

    result_path = run_dir / 'generation_results.json'
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved → {result_path}")
    return report


def print_comparison(all_results: list[dict]):
    print(f"\n{'='*70}")
    print("  GENERATION-LEVEL ABLATION COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Metric':<35}" + "".join(f"  {r['name'][:10]:>12}" for r in all_results)
    print(header)
    print(f"  {'─'*35}" + "  " + "─"*12 * len(all_results))
    for key in REPORT_KEYS:
        short = key.replace('test/', '').replace('mean_', '')
        row = f"  {short:<35}"
        for r in all_results:
            val = r['metrics'].get(key)
            row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12}"
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ablations', nargs='*', default=[],
                   help='Ablated encoder checkpoint paths (A1, A2, A3 …)')
    p.add_argument('--baselines', nargs='*', default=[],
                   help='Baseline encoder checkpoint paths (V1, V2 …)')
    p.add_argument('--names', nargs='+', required=True,
                   help='Short names — must match total number of ablations + baselines')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--bsz_train', type=int, default=72)
    p.add_argument('--out', required=True, help='Output JSON path')
    args = p.parse_args()

    all_ckpts = list(args.ablations) + list(args.baselines)
    assert len(all_ckpts) == len(args.names), \
        f"Got {len(all_ckpts)} checkpoints but {len(args.names)} names"

    out_path = Path(args.out)
    out_dir  = out_path.parent / 'ablation_gen_runs'
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = GLIMDataModule(
        data_path=args.data,
        bsz_train=args.bsz_train,
        bsz_val=24,
        num_workers=4,
    )

    all_results = []
    for ckpt, name in zip(all_ckpts, args.names):
        res = run_one(ckpt, name, dm, args.epochs, args.gpu, out_dir)
        all_results.append(res)

    print_comparison(all_results)

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == '__main__':
    main()
