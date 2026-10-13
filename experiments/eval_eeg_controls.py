"""
EEG Control Condition Evaluation.

Addresses reviewer concern (oD5V W3): "The noise-input experiment is useful,
but replacing EEG with random noise only shows that the model reacts to obviously
abnormal inputs. It does not fully demonstrate that the model relies on meaningful
EEG information."

This script evaluates the trained GLIM model under five progressively weaker
EEG perturbation conditions and compares them to the real EEG baseline:

  - none           : Real EEG (upper bound)
  - channel_shuffle: Electrode labels permuted; temporal dynamics preserved
  - temporal_shuffle: Time windows shuffled; spatial topology preserved
  - phase_randomize : Fourier phase randomised; power spectrum preserved
  - cross_subject   : EEG from a different subject for the same sentence
  - gaussian_noise  : i.i.d. Gaussian (most disruptive; original ablation)

If the model truly uses EEG semantics, performance should degrade monotonically
as the perturbation becomes more disruptive.

Usage:
    python experiments/eval_eeg_controls.py \
        --checkpoint runs/v2/epoch=199-step=397600.ckpt \
        --data       data/tmp/zuco_eeg_label_8variants.df \
        --out        results/eeg_controls_v2.json

    # Quick mode (skip slow full-val metrics):
    python experiments/eval_eeg_controls.py \
        --checkpoint runs/v2/epoch=199-step=397600.ckpt \
        --data data/tmp/zuco_eeg_label_8variants.df \
        --out /tmp/controls_smoke.json --fast
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Literal

import torch
import lightning as L

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.glim import GLIM
from data.datamodule import GLIMDataModule, EEG_CONTROL

CONTROLS: list[EEG_CONTROL] = [
    'none',
    'channel_shuffle',
    'temporal_shuffle',
    'phase_randomize',
    'cross_subject',
    'gaussian_noise',
]

REPORT_KEYS = [
    'test/mean_BLEU1@MTV',
    'test/mean_BLEU2@MTV',
    'test/mean_retrieval_acc_top01',
    'test/mean_retrieval_acc_top05',
    'test/mean_relation_cls_acc_top01',
    'test/mean_relation_cls_acc_top03',
    'test/mean_corpus_cls_acc',
    'test/mean_sentiment_cls_acc_top01',
]


def load_model(checkpoint_path: str, device_id: int) -> GLIM:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hp = ckpt.get('hyper_parameters', {})

    model = GLIM(**{k: v for k, v in hp.items() if k in GLIM.__init__.__code__.co_varnames})
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def run_control(ckpt_path: str, data_path: str, control: EEG_CONTROL,
                gpu: int, bsz_test: int) -> dict:
    print(f"\n  [{control:20s}] evaluating ...", end=' ', flush=True)

    dm = GLIMDataModule(
        data_path=data_path,
        eval_eeg_control=control,
        bsz_val=bsz_test,
        bsz_test=bsz_test,
        num_workers=2,
    )

    trainer = L.Trainer(
        accelerator='gpu',
        devices=[gpu],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    model = GLIM.load_from_checkpoint(ckpt_path, map_location=f'cuda:{gpu}')
    model.eval()

    results = trainer.test(model, datamodule=dm)
    metrics = results[0] if results else {}
    flat = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}
    print("done")
    return flat


def print_table(all_results: dict[str, dict]):
    controls = list(all_results.keys())
    print(f"\n{'='*80}")
    print("  EEG CONTROL CONDITION EVALUATION")
    print(f"{'='*80}")
    print(f"  Interpretation guide:")
    print(f"    none            → real EEG (upper bound)")
    print(f"    channel_shuffle → spatial topology destroyed, time intact")
    print(f"    temporal_shuffle→ temporal order destroyed, topology intact")
    print(f"    phase_randomize → power spectrum intact, temporal structure gone")
    print(f"    cross_subject   → sentence–brain binding removed")
    print(f"    gaussian_noise  → all signal destroyed (lower bound)\n")
    header = f"  {'Metric':<38}" + "".join(f"  {c[:10]:>12}" for c in controls)
    print(header)
    print(f"  {'─'*38}" + "  " + "─" * 12 * len(controls))
    for key in REPORT_KEYS:
        short = key.replace('test/', '').replace('mean_', '')
        row = f"  {short:<38}"
        for ctrl in controls:
            val = all_results[ctrl].get(key)
            row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12}"
        print(row)
    print(f"{'='*80}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data',       required=True)
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--bsz_test',   type=int, default=24)
    p.add_argument('--controls',   nargs='+', choices=list(CONTROLS), default=None,
                   help='Subset of controls to run (default: all)')
    p.add_argument('--out',        required=True)
    args = p.parse_args()

    controls_to_run = args.controls or CONTROLS

    all_results: dict[str, dict] = {}
    for ctrl in controls_to_run:
        metrics = run_control(args.checkpoint, args.data, ctrl, args.gpu, args.bsz_test)
        all_results[ctrl] = metrics

    print_table(all_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'controls': all_results,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
