"""
Multi-Seed Linear Probe Evaluation.

Runs the frozen-encoder linear probe over N random seeds and reports mean ± std
for each classification task. Addresses reviewer concern about single-seed results.

Each seed re-initialises the LogisticRegression solver's random state AND, when the
data has no fixed phase column, re-shuffles the train/test split. For ZuCo (which
already has a fixed 'phase' column) the only source of variance is the LR solver
warm-start and any stochastic solver iterations.

Usage:
    cd /path/to/GLIM
    python pretraining/multiseed_probe.py \
        --ckpt pretraining/Results/GLIM_Pretrain5/best_model.pth \
        --data data/tmp/zuco_eeg_label_8variants.df \
        --seeds 5 \
        --out pretraining/Results/GLIM_Pretrain5/multiseed_probe.json

    # Compare two checkpoints side-by-side:
    python pretraining/multiseed_probe.py \
        --ckpt pretraining/Results/GLIM_Pretrain1/best_model.pth \
                pretraining/Results/GLIM_Pretrain5/best_model.pth \
        --data data/tmp/zuco_eeg_label_8variants.df \
        --seeds 5
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretraining.evaluate_multitask_probe import (
    ZuCoMultiLabelDataset, load_data, extract_features, count_parameters
)
from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_probe_with_seed(train_feat, train_labels, test_feat, test_labels, seed: int):
    """Fit logistic regression with a given seed, return (train_acc, test_acc)."""
    le = LabelEncoder()
    valid = [(f, l) for f, l in zip(range(len(train_labels)), train_labels)
             if l not in ('nan', 'unknown', '')]
    if len(valid) < 10:
        return None, None, None
    t_idx, t_labs = zip(*valid)
    v_valid = [(f, l) for f, l in zip(range(len(test_labels)), test_labels)
               if l not in ('nan', 'unknown', '')]
    if not v_valid:
        return None, None, None
    v_idx, v_labs = zip(*v_valid)

    all_labs = list(t_labs) + list(v_labs)
    le.fit(all_labs)
    tr_y = le.transform(list(t_labs))
    te_y = le.transform(list(v_labs))
    tr_x = train_feat[list(t_idx)]
    te_x = test_feat[list(v_idx)]

    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(tr_x, tr_y)
    tr_acc = accuracy_score(tr_y, clf.predict(tr_x))
    te_acc = accuracy_score(te_y, clf.predict(te_x))
    chance = 1.0 / len(le.classes_)
    return tr_acc, te_acc, chance


def load_encoder(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})
    actual_patch_size = ckpt_args.get('patch_size', 10)
    model = GLIMEncoderPretrainer(
        input_dim=ckpt_args.get('input_dim', 128),
        seq_len=ckpt_args.get('seq_len', 1280),
        emb_size=ckpt_args.get('emb_size', 128),
        n_blocks=ckpt_args.get('n_blocks', 6),
        num_heads=ckpt_args.get('num_heads', 8),
        patch_size=actual_patch_size,
        mask_ratio=ckpt_args.get('mask_ratio', 0.5),
        momentum=ckpt_args.get('momentum', 0.99),
        use_gated_attention=ckpt_args.get('use_gated_attention', False),
        pretraining_mode=ckpt_args.get('pretraining_mode', 'jepa'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def evaluate_ckpt(ckpt_path: str, data_path: str, n_seeds: int,
                  batch_size: int, device: torch.device) -> dict:
    print(f"\n{'='*70}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Seeds: {n_seeds}")
    print(f"{'='*70}")

    model = load_encoder(ckpt_path, device)
    print(f"  Parameters: {count_parameters(model):,}")

    # Features are deterministic given the frozen encoder — extract once.
    train_ds, test_ds = load_data(data_path, seed=42)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    tr_feat, tr_subj, tr_sent, tr_rel, tr_para, tr_sst = extract_features(model, tr_loader, device)
    te_feat, te_subj, te_sent, te_rel, te_para, te_sst = extract_features(model, te_loader, device)

    tasks = {
        'subject_id':   (tr_subj, te_subj),
        'sentiment':    (tr_sent, te_sent),
        'relation':     (tr_rel,  te_rel),
        'paradigm_no_sst': (
            [('NR' if p == 'NR' else 'TSR') for p in tr_para],
            [('NR' if p == 'NR' else 'TSR') for p in te_para],
        ),
    }

    seed_list = list(range(n_seeds))
    per_task: dict[str, dict] = defaultdict(lambda: {'train': [], 'test': [], 'chance': None})

    for seed in seed_list:
        print(f"  Seed {seed:02d} ...", end=' ', flush=True)
        for tname, (tr_labs, te_labs) in tasks.items():
            tr_a, te_a, chance = run_probe_with_seed(tr_feat, list(tr_labs),
                                                      te_feat, list(te_labs), seed)
            if te_a is not None:
                per_task[tname]['train'].append(tr_a)
                per_task[tname]['test'].append(te_a)
                per_task[tname]['chance'] = chance
        print("done")

    summary = {}
    print(f"\n{'─'*70}")
    print(f"  {'Task':<25} {'Chance':>7} {'Mean Test':>10} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for tname, vals in per_task.items():
        te = np.array(vals['test'])
        tr = np.array(vals['train'])
        chance = vals['chance']
        summary[tname] = {
            'n_seeds': len(te),
            'chance': chance,
            'test_mean': float(te.mean()),
            'test_std':  float(te.std()),
            'test_min':  float(te.min()),
            'test_max':  float(te.max()),
            'train_mean': float(tr.mean()),
            'train_std':  float(tr.std()),
            'per_seed_test': te.tolist(),
        }
        print(f"  {tname:<25} {chance:>7.3f} {te.mean():>10.4f} {te.std():>8.4f}"
              f" {te.min():>8.4f} {te.max():>8.4f}")
    print(f"{'─'*70}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', nargs='+', required=True,
                   help='One or more checkpoint paths to compare')
    p.add_argument('--data', required=True,
                   help='Path to zuco_eeg_label_8variants.df')
    p.add_argument('--seeds', type=int, default=5,
                   help='Number of random seeds (default: 5)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out', default=None,
                   help='Save JSON results to this path (optional)')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}
    for ckpt_path in args.ckpt:
        name = os.path.splitext(os.path.basename(os.path.dirname(ckpt_path)))[0] \
               or os.path.splitext(os.path.basename(ckpt_path))[0]
        all_results[name] = evaluate_ckpt(ckpt_path, args.data, args.seeds,
                                          args.batch_size, device)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.out}")

    # Side-by-side comparison if multiple checkpoints
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  COMPARISON TABLE  (test accuracy, mean ± std)")
        print(f"{'='*70}")
        ckpt_names = list(all_results.keys())
        tasks = list(next(iter(all_results.values())).keys())
        header = f"  {'Task':<25}" + "".join(f"  {n[:12]:>14}" for n in ckpt_names)
        print(header)
        for t in tasks:
            row = f"  {t:<25}"
            for n in ckpt_names:
                v = all_results[n].get(t, {})
                if v:
                    row += f"  {v['test_mean']:.4f}±{v['test_std']:.4f}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)


if __name__ == '__main__':
    main()
