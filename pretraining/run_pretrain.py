"""
Training script for GLIM-compatible EEG encoder pretraining (GRAPE Stage I).

V2 baseline (all ablation flags at defaults):
    python pretraining/run_pretrain.py --data_path data/tmp/zuco_eeg_128ch_1280len.df
           --epochs 200 --batch_size 64 --lr 1e-4 --weight_decay 0.01
           --n_blocks 8 --emb_size 256 --num_heads 16
           --mask_ratio 0.5 --momentum 0.99 --use_gated_attention
           --output_dir runs/v2

A1 (random mask):
    ... --masking_strategy random --output_dir runs/ablation_a1

A2 (MSE-only, no VICReg):
    ... --std_coeff 0.0 --cov_coeff 0.0 --output_dir runs/ablation_a2

A3 (MAE-style reconstruction):
    ... --pretraining_mode mae --output_dir runs/ablation_a3
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import argparse
import logging
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Resolve paths relative to this file so the script works from any CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in [_SCRIPT_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Dataset.zuco_memmap import load_zuco_memmap
from pretrain_glim_encoder import GLIMEncoderPretrainer, count_parameters

# For linear probe evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='GLIM Encoder Pretraining')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='../data/tmp/zuco_eeg_128ch_1280len.df')
    parser.add_argument('--output_dir', type=str, default='Results/GLIM_Pretrain')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Model architecture
    parser.add_argument('--n_blocks', type=int, default=8)
    parser.add_argument('--emb_size', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--use_gated_attention', action='store_true')

    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear_probe_interval', type=int, default=10)

    # ── Ablation knobs ────────────────────────────────────────────────────
    parser.add_argument('--masking_strategy', type=str, default='ssp',
                        choices=['ssp', 'random'],
                        help='A1: ssp (default) or random masking')
    parser.add_argument('--std_coeff', type=float, default=1.0,
                        help='A2: VICReg variance loss coefficient (0.0 = disabled)')
    parser.add_argument('--cov_coeff', type=float, default=1.0,
                        help='A2: VICReg covariance loss coefficient (0.0 = disabled)')
    parser.add_argument('--pretraining_mode', type=str, default='jepa',
                        choices=['jepa', 'mae'],
                        help='A3: jepa (default latent target) or mae (raw reconstruction)')

    # Checkpoint milestones
    parser.add_argument('--save_epochs', type=str, default='120,150',
                        help='Comma-separated epochs to save extra milestone checkpoints')

    return parser.parse_args()


# ── Reproducibility helpers ───────────────────────────────────────────────────

def get_git_hash() -> str:
    """Return current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def save_config(args, output_dir: str):
    """Persist resolved config + git hash to config.json."""
    cfg = vars(args).copy()
    cfg['git_commit'] = get_git_hash()
    cfg['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    path = os.path.join(output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config saved → {path}")
    return cfg


# ── Structured logging ────────────────────────────────────────────────────────

def append_epoch_log(log_path: str, record: dict):
    """Append one JSON record (one line) to the epoch log file."""
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ── Linear probe ─────────────────────────────────────────────────────────────

def make_representation(model, loader, device):
    model.eval()
    all_repr, all_labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            repr_ = model(x)
            all_repr.append(repr_.cpu())
            all_labels.append(y)
    return torch.cat(all_repr, dim=0), torch.cat(all_labels, dim=0)


def run_linear_probe(model, train_loader, test_loader, device) -> float:
    train_repr, train_labels = make_representation(model, train_loader, device)
    test_repr, test_labels = make_representation(model, test_loader, device)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_repr.numpy(), train_labels.numpy())
    pred = clf.predict(test_repr.numpy())
    return accuracy_score(test_labels.numpy(), pred)


# ── Single epoch training ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, epoch,
                std_coeff: float, cov_coeff: float):
    """Run one epoch; return (avg_total_loss, avg_loss_components_dict)."""
    model.copy_weight()   # no-op in MAE mode
    model.train()

    total_loss = 0.0
    sum_components = {'align': 0.0, 'std': 0.0, 'cov': 0.0, 'target_per_dim_std': 0.0}
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for x, _, _ in pbar:
        x = x.to(device)

        rep_target, rep_pred, _, _ = model.pretrain_forward(x)
        loss, loss_dict = model.compute_loss(rep_target, rep_pred)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.momentum_update()  # no-op in MAE mode

        total_loss += loss_dict['total']
        for k in sum_components:
            sum_components[k] += loss_dict.get(k, 0.0)

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'align': f"{loss_dict['align']:.4f}",
        })

    avg_total = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in sum_components.items()}
    return avg_total, avg_components


# ── Checkpoint saving ─────────────────────────────────────────────────────────

def save_checkpoint(path: str, epoch: int, model, optimizer, extra: dict = None):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.get_encoder_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': extra or {},
    }
    torch.save(payload, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save resolved config + git hash
    cfg_dict = save_config(args, args.output_dir)

    # Parse milestone epochs
    milestone_epochs = set()
    if args.save_epochs:
        for s in args.save_epochs.split(','):
            s = s.strip()
            if s.isdigit():
                milestone_epochs.add(int(s))
    logger.info(f"Milestone checkpoints at epochs: {sorted(milestone_epochs)}")

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data_path}")
    datasets = load_zuco_memmap(args.data_path, seed=args.seed)

    # num_workers=0 on Windows: avoids pickling the large DataFrame across
    # multiprocessing workers which causes MemoryError with the spawn method.
    _num_workers = 0 if sys.platform.startswith('win') else 2

    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                              shuffle=True, num_workers=_num_workers, pin_memory=True)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size,
                             shuffle=False, num_workers=_num_workers, pin_memory=True)
    logger.info(f"Train: {len(datasets['train'])} | Test: {len(datasets['test'])}")

    # ── Fixed gate-probe batch (same 16 samples every epoch) ──────────────
    # Drawn ONCE before training; used to log gate stats reproducibly.
    _probe_iter = iter(DataLoader(datasets['train'], batch_size=16,
                                  shuffle=False, num_workers=0))
    gate_probe_x = next(_probe_iter)[0].to(device)   # (16, C, T) or (16, T, C)
    del _probe_iter

    # ── Model ─────────────────────────────────────────────────────────────
    model = GLIMEncoderPretrainer(
        in_len=1280,
        in_dim=128,
        emb_size=args.emb_size,
        n_blocks=args.n_blocks,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        momentum=args.momentum,
        use_gated_attention=args.use_gated_attention,
        # Ablation knobs
        masking_strategy=args.masking_strategy,
        std_coeff=args.std_coeff,
        cov_coeff=args.cov_coeff,
        pretraining_mode=args.pretraining_mode,
    ).to(device)

    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Mode: {args.pretraining_mode} | Masking: {args.masking_strategy} | "
                f"std_coeff={args.std_coeff} | cov_coeff={args.cov_coeff}")
    logger.info(f"Target encoder: {model.target_encoder is not None} | "
                f"MAE head: {model.mae_head is not None}")

    # Sanity checks
    if args.pretraining_mode == 'mae':
        assert model.target_encoder is None, "BUG: target_encoder should be None in MAE mode"
        assert model.mae_head is not None, "BUG: mae_head should exist in MAE mode"
    if args.std_coeff == 0.0 and args.cov_coeff == 0.0:
        logger.info("A2 (MSE-only): std and cov losses will be computed but NOT added to backward graph")

    # ── Optimiser ─────────────────────────────────────────────────────────
    params_to_optimize = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.patch_embed.parameters())
        + [model.mask_token]
    )
    if model.mae_head is not None:
        params_to_optimize += list(model.mae_head.parameters())
    # Note: target_encoder is excluded (no gradients / None in MAE)

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    # ── Tensorboard + log file ─────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    log_path = os.path.join(args.output_dir, 'epoch_logs.jsonl')
    logger.info(f"Epoch logs → {log_path}")

    # ── Training loop ──────────────────────────────────────────────────────
    best_acc = 0.0
    logger.info("Starting pretraining…")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train one epoch ──
        avg_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, epoch,
            std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)

        epoch_time = time.time() - t0

        # ── Gate stats on fixed probe batch ──
        model.eval()
        with torch.no_grad():
            model.pretrain_forward(gate_probe_x)   # populates _last_gate_* on each block
        gate_stats = model.get_encoder_gate_stats()

        # ── Tensorboard scalars ──
        writer.add_scalar('loss/total',  avg_loss, epoch)
        writer.add_scalar('loss/align',  loss_components['align'], epoch)
        writer.add_scalar('loss/std',    loss_components['std'], epoch)
        writer.add_scalar('loss/cov',    loss_components['cov'], epoch)
        writer.add_scalar('loss/target_per_dim_std', loss_components['target_per_dim_std'], epoch)
        writer.add_scalar('gates/mean',     gate_stats['gate_mean'], epoch)
        writer.add_scalar('gates/std',      gate_stats['gate_std'], epoch)
        writer.add_scalar('gates/sparsity', gate_stats['gate_sparsity'], epoch)

        # ── Linear probe (every N epochs) ──
        probe_acc = None
        if epoch % args.linear_probe_interval == 0:
            probe_acc = run_linear_probe(model, train_loader, test_loader, device)
            writer.add_scalar('accuracy/linear_probe', probe_acc, epoch)
            logger.info(f"  Linear probe accuracy: {probe_acc:.4f}")

            if probe_acc > best_acc:
                best_acc = probe_acc
                save_checkpoint(
                    os.path.join(args.output_dir, 'best_model.pth'),
                    epoch, model, optimizer, extra=cfg_dict)
                logger.info(f"  Saved best model (acc={probe_acc:.4f})")

        # ── Milestone checkpoints ──
        if epoch in milestone_epochs:
            ckpt_path = os.path.join(args.output_dir, f'model_epoch{epoch}.pth')
            save_checkpoint(ckpt_path, epoch, model, optimizer, extra=cfg_dict)
            logger.info(f"  Milestone checkpoint saved → {ckpt_path}")

        # ── Structured JSON log (one record per epoch) ──
        log_record = {
            'epoch': epoch,
            'loss_total':          round(avg_loss, 6),
            'loss_align':          round(loss_components['align'], 6),
            'loss_std':            round(loss_components['std'], 6),
            'loss_cov':            round(loss_components['cov'], 6),
            'target_per_dim_std':  round(loss_components['target_per_dim_std'], 6),
            'gate_mean':           round(gate_stats['gate_mean'], 6),
            'gate_std':            round(gate_stats['gate_std'], 6),
            'gate_sparsity':       round(gate_stats['gate_sparsity'], 6),
            'linear_probe_acc':    round(probe_acc, 6) if probe_acc is not None else None,
            'lr':                  args.lr,
            'time_s':              round(epoch_time, 2),
        }
        append_epoch_log(log_path, log_record)

        # ── Console summary ──
        logger.info(
            f"Epoch {epoch}: loss={avg_loss:.4f} "
            f"align={loss_components['align']:.4f} "
            f"std={loss_components['std']:.4f} "
            f"cov={loss_components['cov']:.4f} "
            f"tgt_std={loss_components['target_per_dim_std']:.4f} "
            f"gate_sp={gate_stats['gate_sparsity']:.3f} "
            f"time={epoch_time:.1f}s"
        )

    # ── Save final checkpoint ──
    save_checkpoint(
        os.path.join(args.output_dir, 'final_model.pth'),
        args.epochs, model, optimizer, extra=cfg_dict)
    logger.info(f"Training complete. Best linear probe accuracy: {best_acc:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
