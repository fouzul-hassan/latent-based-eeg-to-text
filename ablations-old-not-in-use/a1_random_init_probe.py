"""
A1 — Random-Initialised Encoder Linear Probe
=============================================

Purpose:
    Establishes the lower bound for representation quality by evaluating
    linear probes on features from a RANDOMLY INITIALISED encoder (no
    pretraining). If pretrained probes (Subject ID ~99.6%) fall to chance
    level here, it confirms that Signal-JEPA pretraining is responsible
    for the learned representations, not encoder architecture alone.

Architecture:
    Uses the canonical Pretrain5 config:
        8 encoder blocks, emb_size=256, num_heads=16, patch_size=8

Outputs (saved to ablations/logs/a1/):
    - a1_random_init_probe_results.txt     Full probe summary
    - multitask_tsne.png                   t-SNE visualisation
    - multitask_probe_summary.png          Bar chart
    - a1_run.log                           Full console log

Usage:
    cd GLIM
    python ablations/a1_random_init_probe.py

    # Custom options:
    python ablations/a1_random_init_probe.py --data data/tmp/zuco_eeg_label_8variants.df --gpu 0
"""
import os
import sys
import json
import argparse
import logging
import datetime
import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer, count_parameters
from pretraining.evaluate_multitask_probe import (
    load_data, extract_features, run_linear_probe,
    visualize_multitask_tsne, plot_probe_summary,
)
from torch.utils.data import DataLoader


# ── Canonical Pretrain5 config ──────────────────────────────────────────────
PRETRAIN5_CONFIG = dict(
    in_len=1280,
    in_dim=128,
    emb_size=256,
    n_blocks=8,
    num_heads=16,
    patch_size=8,
    mask_ratio=0.7,
    momentum=0.99,
    use_gated_attention=False,
)


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure dual logging to file + console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'a1_run.log')

    logger = logging.getLogger('A1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description='A1: Random-Init Encoder Linear Probe')
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'tmp', 'zuco_eeg_label_8variants.df'),
                        help='Path to ZuCo dataset with labels')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── Directories ─────────────────────────────────────────────────────────
    log_dir = os.path.join(SCRIPT_DIR, 'logs', 'a1')
    logger = setup_logging(log_dir)

    logger.info('=' * 60)
    logger.info('  A1 — Random-Initialised Encoder Linear Probe')
    logger.info('=' * 60)
    logger.info(f'  Timestamp : {datetime.datetime.now().isoformat()}')
    logger.info(f'  Log dir   : {log_dir}')
    logger.info(f'  Data      : {args.data}')
    logger.info(f'  Seed      : {args.seed}')
    logger.info(f'  Config    : Pretrain5 (8-block, 256-dim, 16-head)')
    logger.info('=' * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f'  Device    : {device}')

    # ── Create model with RANDOM weights (no checkpoint loaded) ─────────────
    logger.info('\n  Creating GLIMEncoderPretrainer with RANDOM weights...')
    model = GLIMEncoderPretrainer(**PRETRAIN5_CONFIG).to(device)
    model.eval()
    n_params = count_parameters(model)
    logger.info(f'  Parameters: {n_params:,}')
    logger.info(f'  NOTE: No checkpoint loaded — weights are Xavier/Kaiming init.')

    # ── Load data ───────────────────────────────────────────────────────────
    logger.info(f'\n  Loading data from {args.data}')
    train_dataset, test_dataset = load_data(args.data, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    logger.info(f'  Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples')

    # ── Extract features ────────────────────────────────────────────────────
    logger.info('\n  Extracting features from RANDOM encoder...')
    train_feats, train_subj, train_sent, train_rel, train_para, train_sst = \
        extract_features(model, train_loader, device)
    test_feats, test_subj, test_sent, test_rel, test_para, test_sst = \
        extract_features(model, test_loader, device)
    logger.info(f'  Feature shape: {train_feats.shape}')

    # ── Feature stats ───────────────────────────────────────────────────────
    std = train_feats.std(axis=0)
    logger.info(f'\n  Feature Statistics:')
    logger.info(f'    Dim      : {train_feats.shape[1]}')
    logger.info(f'    Mean std : {std.mean():.4f}')
    logger.info(f'    Min std  : {std.min():.4f}')
    if std.min() < 0.01:
        logger.info(f'    WARNING: Some dims have near-zero variance (possible collapse)')
    else:
        logger.info(f'    OK: No feature collapse')

    # ── Run probes ──────────────────────────────────────────────────────────
    results = []

    # 1. Subject ID
    result_subj = run_linear_probe(
        train_feats, train_subj, test_feats, test_subj,
        'Subject ID Classification')
    results.append(result_subj)

    # 2. Sentiment (filter nan)
    train_sent_valid = [(i, l) for i, l in enumerate(train_sent) if l not in ('nan', 'None', '')]
    test_sent_valid = [(i, l) for i, l in enumerate(test_sent) if l not in ('nan', 'None', '')]
    if len(train_sent_valid) > 10 and len(test_sent_valid) > 5:
        t_idx, t_lab = zip(*train_sent_valid)
        e_idx, e_lab = zip(*test_sent_valid)
        result_sent = run_linear_probe(
            train_feats[list(t_idx)], list(t_lab),
            test_feats[list(e_idx)], list(e_lab),
            'Sentiment Classification')
        results.append(result_sent)

    # 3. Relation (filter nan)
    train_rel_valid = [(i, l) for i, l in enumerate(train_rel) if l not in ('nan', 'None', '')]
    test_rel_valid = [(i, l) for i, l in enumerate(test_rel) if l not in ('nan', 'None', '')]
    if len(train_rel_valid) > 10 and len(test_rel_valid) > 5:
        t_idx, t_lab = zip(*train_rel_valid)
        e_idx, e_lab = zip(*test_rel_valid)
        result_rel = run_linear_probe(
            train_feats[list(t_idx)], list(t_lab),
            test_feats[list(e_idx)], list(e_lab),
            'Relation Classification')
        results.append(result_rel)

    # ── Visualisations ──────────────────────────────────────────────────────
    logger.info('\n  Generating t-SNE and summary plots...')
    visualize_multitask_tsne(test_feats, test_subj, test_sent, test_rel,
                             test_para, test_sst, log_dir)
    plot_probe_summary(results, log_dir)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('  A1 RESULTS — Random-Init Encoder')
    logger.info('=' * 60)
    logger.info(f'  {"Task":<30} {"Accuracy":>10} {"Random":>10} {"vs Random":>10}')
    logger.info(f'  {"-"*30} {"-"*10} {"-"*10} {"-"*10}')
    for r in results:
        logger.info(f'  {r["task"]:<30} {100*r["test_acc"]:>9.1f}% '
                     f'{100*r["random_baseline"]:>9.1f}% '
                     f'{r["improvement_over_random"]:>9.1f}x')
    logger.info('=' * 60)

    # ── Save results ────────────────────────────────────────────────────────
    # Text summary
    txt_path = os.path.join(log_dir, 'a1_random_init_probe_results.txt')
    with open(txt_path, 'w') as f:
        f.write('A1 - Random-Init Encoder Linear Probe Results\n')
        f.write(f'Timestamp: {datetime.datetime.now().isoformat()}\n')
        f.write(f'Config: Pretrain5 (8-block, emb=256, heads=16, patch=8)\n')
        f.write(f'NOTE: No checkpoint loaded - RANDOM WEIGHTS\n')
        f.write('=' * 50 + '\n\n')
        for r in results:
            f.write(f'Task: {r["task"]}\n')
            f.write(f'  Classes:   {r["n_classes"]}\n')
            f.write(f'  Train acc: {100*r["train_acc"]:.1f}%\n')
            f.write(f'  Test acc:  {100*r["test_acc"]:.1f}%\n')
            f.write(f'  Random:    {100*r["random_baseline"]:.1f}%\n')
            f.write(f'  vs Random: {r["improvement_over_random"]:.1f}x\n\n')
    logger.info(f'\n  Saved: {txt_path}')

    # JSON (machine-readable)
    json_path = os.path.join(log_dir, 'a1_results.json')
    json_results = {
        'ablation': 'A1',
        'description': 'Random-init encoder linear probe (lower bound)',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': PRETRAIN5_CONFIG,
        'n_parameters': n_params,
        'seed': args.seed,
        'data': args.data,
        'results': [
            {
                'task': r['task'],
                'n_classes': r['n_classes'],
                'train_acc': float(r['train_acc']),
                'test_acc': float(r['test_acc']),
                'random_baseline': float(r['random_baseline']),
                'improvement_over_random': float(r['improvement_over_random']),
            }
            for r in results
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f'  Saved: {json_path}')

    logger.info('\n  A1 complete.')


if __name__ == '__main__':
    main()
