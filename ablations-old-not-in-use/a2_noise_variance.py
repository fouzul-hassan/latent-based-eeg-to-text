"""
A2 — Noise-Input Realisation Variance
======================================

Purpose:
    Demonstrates that GRAPE-GLIM-G's noise-input test scores are robust
    across different random noise realisations (Jo et al., 2024 diagnostic).
    Runs 10 inference passes on the SAME test set but with DIFFERENT random
    noise seeds replacing the EEG signals. Reports mean +/- std for
    BLEU-1, ROUGE-1, retrieval ACC-1 under noise.

    Expected: All metrics should be at or near chance level with low variance,
    confirming the model genuinely uses EEG signal, not data leakage.

Model:
    GRAPE-GLIM-G (v2) — runs/v2/epoch=199-step=397600.ckpt

Outputs (saved to ablations/logs/a2/):
    - a2_noise_variance_results.txt       Full summary table
    - a2_results.json                     Machine-readable results
    - a2_run.log                          Full console log

Usage:
    cd GLIM
    python ablations/a2_noise_variance.py

    # Custom options:
    python ablations/a2_noise_variance.py --checkpoint runs/v2/epoch=199-step=397600.ckpt --n_seeds 10 --gpu 0
"""
import os
import sys
import json
import argparse
import logging
import datetime
import subprocess
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ════════════════════════════════════════════════════════════════════════════
#  WORKER  — runs a single seed in its own process
#  Invoked via:  python a2_noise_variance.py --_worker --seed N ...
# ════════════════════════════════════════════════════════════════════════════

def _worker_main(args):
    """
    Runs one noise seed and writes results to a JSON file.
    Each seed gets a fresh Python process → fresh CUDA context.
    """
    import torch
    import lightning as L
    import numpy as np

    sys.path.insert(0, PROJECT_ROOT)
    from model.glim import GLIM
    from data.datamodule import GLIMDataModule, ZuCoDataset
    from lightning.pytorch.loggers import CSVLogger

    seed = args.seed
    devices = [args.gpu]

    # ── Patch noise seed ─────────────────────────────────────────────────
    original_init = ZuCoDataset.__init__

    def patched_init(self, df, phase, eval_noise_input=False):
        original_init(self, df, phase, eval_noise_input=False)
        if eval_noise_input:
            n = len(self.data['eeg'])
            l, c = self.data['eeg'][0].shape
            self.data.pop('eeg')
            self.data.pop('mask')
            rng = np.random.default_rng(seed=seed)
            self.data['eeg'] = rng.standard_normal((n, l, c)).astype(np.float32)
            self.data['mask'] = np.ones((n, l), dtype=np.int8)

    ZuCoDataset.__init__ = patched_init

    # ── Setup ────────────────────────────────────────────────────────────
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    dm = GLIMDataModule(
        data_path=args.data,
        eval_noise_input=True,
        bsz_test=24,
        num_workers=0,
    )

    model = GLIM.load_from_checkpoint(
        args.checkpoint,
        map_location=f'cuda:{devices[0]}',
        strict=False,
        weights_only=False,
        use_etes_eval=False,
        log_xai=False,
    )

    tmp_log_dir = os.path.join(SCRIPT_DIR, 'logs', 'a2', '_tmp')
    csv_logger = CSVLogger(save_dir=tmp_log_dir, name='noise_run', version=str(seed))

    trainer = L.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=csv_logger,
        precision=args.precision,
        enable_progress_bar=True,
    )

    trainer.test(model, datamodule=dm)

    # ── Collect metrics and write to JSON ─────────────────────────────────
    metrics = trainer.callback_metrics
    result = {'seed': seed}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, (int, float)):
            result[k] = float(v)

    out_path = args.worker_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f)

    print(f'[Worker] Seed {seed} done. Written to {out_path}', flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR  — spawns one subprocess per seed, aggregates results
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure dual logging to file + console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'a2_run.log')

    logger = logging.getLogger('A2')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


def run_seed_subprocess(seed: int, args, tmp_dir: str) -> dict | None:
    """Spawn a fresh Python process for one seed. Returns metrics dict or None on failure."""
    out_path = os.path.join(tmp_dir, f'seed_{seed:02d}.json')

    cmd = [
        sys.executable, __file__,
        '--_worker',
        '--seed', str(seed),
        '--checkpoint', args.checkpoint,
        '--data', args.data,
        '--gpu', str(args.gpu),
        '--precision', args.precision,
        '--worker_out', out_path,
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f'  [!] Seed {seed} subprocess failed (exit {result.returncode})', flush=True)
        return None

    if not os.path.exists(out_path):
        print(f'  [!] Seed {seed}: output file not found at {out_path}', flush=True)
        return None

    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='A2: Noise-Input Realisation Variance')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'runs', 'v2', 'epoch=199-step=397600.ckpt'))
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'tmp', 'zuco_eeg_label_8variants.df'))
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    # Internal worker flags (not for end users)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--worker_out', type=str, default='', help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── Dispatch to worker if called internally ──────────────────────────
    if args._worker:
        _worker_main(args)
        return

    # ── Orchestrator ─────────────────────────────────────────────────────
    log_dir = os.path.join(SCRIPT_DIR, 'logs', 'a2')
    tmp_dir = os.path.join(log_dir, '_tmp', 'seed_results')
    logger = setup_logging(log_dir)

    logger.info('=' * 60)
    logger.info('  A2 -- Noise-Input Realisation Variance')
    logger.info('=' * 60)
    logger.info(f'  Timestamp  : {datetime.datetime.now().isoformat()}')
    logger.info(f'  Checkpoint : {args.checkpoint}')
    logger.info(f'  Data       : {args.data}')
    logger.info(f'  Seeds      : 0..{args.n_seeds - 1} ({args.n_seeds} runs)')
    logger.info(f'  GPU        : {args.gpu}')
    logger.info(f'  Strategy   : isolated subprocess per seed (avoids CUDA ctx corruption)')
    logger.info('=' * 60)

    os.makedirs(tmp_dir, exist_ok=True)
    all_seed_metrics = []

    for seed_idx in range(args.n_seeds):
        logger.info(f'\n  --- Seed {seed_idx}/{args.n_seeds - 1} ---')
        result = run_seed_subprocess(seed_idx, args, tmp_dir)
        if result is not None:
            all_seed_metrics.append(result)
            printable = {k: f'{v:.4f}' for k, v in result.items()
                         if k != 'seed' and isinstance(v, (int, float))}
            logger.info(f'    Metrics: {json.dumps(printable, indent=2)}')
        else:
            logger.info(f'    [!] Seed {seed_idx} failed — skipping.')

    # ── Aggregate ────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('  A2 RESULTS -- Noise-Input Variance')
    logger.info('=' * 60)

    summary = {}
    if all_seed_metrics:
        common_keys = set(all_seed_metrics[0].keys())
        for d in all_seed_metrics[1:]:
            common_keys &= set(d.keys())
        common_keys.discard('seed')

        logger.info(f'  {"Metric":<40} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10}')
        logger.info(f'  {"-"*40} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
        for key in sorted(common_keys):
            vals = [d[key] for d in all_seed_metrics if isinstance(d.get(key), (int, float))]
            if vals:
                mean_v, std_v, min_v, max_v = np.mean(vals), np.std(vals), np.min(vals), np.max(vals)
                summary[key] = {'mean': mean_v, 'std': std_v, 'min': min_v, 'max': max_v}
                logger.info(f'  {key:<40} {mean_v:>10.4f} {std_v:>10.4f} {min_v:>10.4f} {max_v:>10.4f}')

    logger.info('=' * 60)

    # ── Save results ─────────────────────────────────────────────────────
    txt_path = os.path.join(log_dir, 'a2_noise_variance_results.txt')
    with open(txt_path, 'w') as f:
        f.write('A2 - Noise-Input Realisation Variance Results\n')
        f.write(f'Timestamp: {datetime.datetime.now().isoformat()}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Seeds: 0..{args.n_seeds - 1}\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'{"Metric":<40} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10}\n')
        f.write(f'{"-"*40} {"-"*10} {"-"*10} {"-"*10} {"-"*10}\n')
        for key, vals in sorted(summary.items()):
            f.write(f'{key:<40} {vals["mean"]:>10.4f} {vals["std"]:>10.4f} '
                    f'{vals["min"]:>10.4f} {vals["max"]:>10.4f}\n')
        f.write('\n\nPer-Seed Raw Values:\n')
        f.write('=' * 70 + '\n')
        for d in all_seed_metrics:
            f.write(f'\nSeed {d["seed"]}:\n')
            for k, v in sorted(d.items()):
                if k != 'seed' and isinstance(v, (int, float)):
                    f.write(f'  {k}: {v:.6f}\n')
    logger.info(f'\n  Saved: {txt_path}')

    json_path = os.path.join(log_dir, 'a2_results.json')
    json_results = {
        'ablation': 'A2',
        'description': 'Noise-input realisation variance (10 seeds)',
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'n_seeds': args.n_seeds,
        'seeds_completed': len(all_seed_metrics),
        'summary': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()},
        'per_seed': [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in d.items()} for d in all_seed_metrics],
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f'  Saved: {json_path}')
    logger.info('\n  A2 complete.')


if __name__ == '__main__':
    main()
