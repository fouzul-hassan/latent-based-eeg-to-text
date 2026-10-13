"""
bootstrap_ci.py
---------------
Perform 1,000-iteration bootstrap resampling on saved per-sample prediction
pickles (evaluations/v1 and evaluations/v2) to compute 95% confidence
intervals for all metrics reported in Tables 4, 5, and 6.

Usage:
    conda run -n physionet2026 python bootstrap_ci.py [--n_boot 1000] [--seed 42] [--eval_dir evaluations]

Outputs:
    evaluations/v1/bootstrap_ci_results.json
    evaluations/v1/bootstrap_corpus_results.txt    (updated Table 4)
    evaluations/v1/bootstrap_sentiment_results.txt (updated Table 5)
    evaluations/v1/bootstrap_relation_results.txt  (updated Table 6)
    evaluations/v2/  (same files)
    evaluations/bootstrap_ci_combined.json
"""

import os
import sys
import json
import pickle
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# optional torchmetrics for BLEU / ROUGE
try:
    from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not available; BLEU/ROUGE will not be computed.")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# PER-SAMPLE METRIC EXTRACTION
# ---------------------------------------------------------------------------

def extract_corpus_sample_metrics(results):
    eeg_correct, raw_correct, gen_correct = [], [], []
    llm_gen_correct, llm_raw_correct = [], []
    for r in results:
        label   = r.get('label_idx', -1)
        valid   = (label >= 0)
        pred_eeg = r.get('pred', -1)
        pred_raw = r.get('pred_raw', pred_eeg)
        pred_gen = r.get('pred_gen', pred_eeg)
        llm_gen  = r.get('llm_pred_gen', -1)
        llm_raw  = r.get('llm_pred_raw', -1)
        eeg_correct.append(int(pred_eeg == label) if valid else np.nan)
        raw_correct.append(int(pred_raw == label) if valid else np.nan)
        gen_correct.append(int(pred_gen == label) if valid else np.nan)
        llm_gen_correct.append(int(llm_gen == label) if (valid and llm_gen >= 0) else np.nan)
        llm_raw_correct.append(int(llm_raw == label) if (valid and llm_raw >= 0) else np.nan)
    return {
        'eeg_correct':     np.array(eeg_correct, dtype=float),
        'raw_correct':     np.array(raw_correct, dtype=float),
        'gen_correct':     np.array(gen_correct, dtype=float),
        'llm_gen_correct': np.array(llm_gen_correct, dtype=float),
        'llm_raw_correct': np.array(llm_raw_correct, dtype=float),
    }


def extract_sentiment_sample_metrics(results):
    eeg_correct, raw_correct, gen_correct = [], [], []
    llm_gen_correct, llm_raw_correct = [], []
    for r in results:
        label   = r.get('label_idx', -1)
        valid   = (label >= 0)
        pred_eeg = r.get('pred', -1)
        pred_raw = r.get('pred_raw', pred_eeg)
        pred_gen = r.get('pred_gen', pred_eeg)
        llm_gen  = r.get('llm_pred_gen', -1)
        llm_raw  = r.get('llm_pred_raw', -1)
        eeg_correct.append(int(pred_eeg == label) if valid else np.nan)
        raw_correct.append(int(pred_raw == label) if valid else np.nan)
        gen_correct.append(int(pred_gen == label) if valid else np.nan)
        llm_gen_correct.append(int(llm_gen == label) if (valid and llm_gen >= 0) else np.nan)
        llm_raw_correct.append(int(llm_raw == label) if (valid and llm_raw >= 0) else np.nan)
    return {
        'eeg_correct':     np.array(eeg_correct, dtype=float),
        'raw_correct':     np.array(raw_correct, dtype=float),
        'gen_correct':     np.array(gen_correct, dtype=float),
        'llm_gen_correct': np.array(llm_gen_correct, dtype=float),
        'llm_raw_correct': np.array(llm_raw_correct, dtype=float),
    }


def extract_relation_sample_metrics(results):
    n = len(results)
    keys = ['eeg_hit1','eeg_hit3','eeg_hit5',
            'raw_hit1','raw_hit3','raw_hit5',
            'gen_hit1','gen_hit3','gen_hit5',
            'llm_gen1','llm_gen3','llm_gen5',
            'llm_raw1','llm_raw3','llm_raw5']
    arrs = {k: np.full(n, np.nan) for k in keys}

    for i, r in enumerate(results):
        label = r.get('label_idx', -1)
        if label < 0:
            continue

        pred_eeg = r.get('pred', -1)
        topk_eeg = r.get('topk_preds', r.get('top_preds', [pred_eeg] if pred_eeg >= 0 else []))

        pred_raw = r.get('pred_raw', -1)
        topk_raw = r.get('topk_raw', [pred_raw] if pred_raw >= 0 else [])

        pred_gen = r.get('pred_gen', -1)
        topk_gen = r.get('topk_gen', [pred_gen] if pred_gen >= 0 else [])

        llm_g = r.get('llm_topk_gen', r.get('llm_pred_topk_gen', None))
        llm_r = r.get('llm_topk_raw', r.get('llm_pred_topk_raw', None))

        if topk_eeg:
            arrs['eeg_hit1'][i] = int(label in topk_eeg[:1])
            arrs['eeg_hit3'][i] = int(label in topk_eeg[:3])
            arrs['eeg_hit5'][i] = int(label in topk_eeg[:5])
        if topk_raw:
            arrs['raw_hit1'][i] = int(label in topk_raw[:1])
            arrs['raw_hit3'][i] = int(label in topk_raw[:3])
            arrs['raw_hit5'][i] = int(label in topk_raw[:5])
        if topk_gen:
            arrs['gen_hit1'][i] = int(label in topk_gen[:1])
            arrs['gen_hit3'][i] = int(label in topk_gen[:3])
            arrs['gen_hit5'][i] = int(label in topk_gen[:5])
        if llm_g is not None:
            arrs['llm_gen1'][i] = int(label in llm_g[:1])
            arrs['llm_gen3'][i] = int(label in llm_g[:3])
            arrs['llm_gen5'][i] = int(label in llm_g[:5])
        if llm_r is not None:
            arrs['llm_raw1'][i] = int(label in llm_r[:1])
            arrs['llm_raw3'][i] = int(label in llm_r[:3])
            arrs['llm_raw5'][i] = int(label in llm_r[:5])
    return arrs


def safe_bleu(gen, refs, n):
    if not HAS_TORCHMETRICS: return float('nan')
    gen  = str(gen).strip()
    refs = [str(r).strip() for r in refs if str(r).strip()]
    if not gen or not refs: return 0.0
    try:
        return bleu_score([gen], [refs], n_gram=n).item()
    except Exception:
        return 0.0


def safe_rouge1_f(gen, refs):
    if not HAS_TORCHMETRICS: return float('nan')
    gen  = str(gen).strip()
    refs = [str(r).strip() for r in refs if str(r).strip()]
    if not gen or not refs: return 0.0
    try:
        d = rouge_score([gen], [refs], rouge_keys='rouge1')
        return d['rouge1_fmeasure'].item()
    except Exception:
        return 0.0


def extract_generation_metrics(results):
    """Compute per-sample BLEU/ROUGE if gen_text present; else return None."""
    b1,b2,b3,b4,r1 = [],[],[],[],[]
    found = 0
    for r in results:
        gen = r.get('gen_text', '')
        raw = r.get('raw_input_text', '')
        if not str(gen).strip():
            b1.append(np.nan); b2.append(np.nan)
            b3.append(np.nan); b4.append(np.nan)
            r1.append(np.nan)
            continue
        found += 1
        variants = r.get('target_texts', r.get('variants', [raw]))
        if isinstance(variants, str):
            variants = [s.strip() for s in variants.split(',') if s.strip()]
        if not variants: variants = [raw]
        b1.append(safe_bleu(gen, variants, 1))
        b2.append(safe_bleu(gen, variants, 2))
        b3.append(safe_bleu(gen, variants, 3))
        b4.append(safe_bleu(gen, variants, 4))
        r1.append(safe_rouge1_f(gen, variants))
    if found == 0: return None
    return {
        'bleu1':   np.array(b1, dtype=float),
        'bleu2':   np.array(b2, dtype=float),
        'bleu3':   np.array(b3, dtype=float),
        'bleu4':   np.array(b4, dtype=float),
        'rouge1_f':np.array(r1, dtype=float),
    }


def extract_reading_paradigm_metrics(results):
    n = len(results)
    correct = np.full(n, np.nan)
    for i, r in enumerate(results):
        label = r.get('label', r.get('label_idx', -1))
        pred  = r.get('pred', -1)
        if label < 0 or pred < 0: continue
        correct[i] = int(pred == label)
    return {'correct': correct}


# ---------------------------------------------------------------------------
# BOOTSTRAP ENGINE
# ---------------------------------------------------------------------------

def bootstrap_metric(arr, n_boot=1000, ci=0.95, rng=None):
    """Bootstrap percentile CI for nanmean of arr."""
    if rng is None: rng = np.random.default_rng(42)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return dict(point=np.nan, lo=np.nan, hi=np.nan, hw=np.nan,
                    ci_str='N/A', n=0)
    n     = len(valid)
    point = float(np.mean(valid))
    boot  = np.array([np.mean(valid[rng.integers(0,n,n)]) for _ in range(n_boot)])
    alpha = (1.0 - ci) / 2.0
    lo    = float(np.percentile(boot, 100*alpha))
    hi    = float(np.percentile(boot, 100*(1-alpha)))
    hw    = (hi - lo) / 2.0
    return dict(point=point, lo=lo, hi=hi, hw=hw,
                ci_str=f"{point:.4f} ± {hw:.4f}  [{lo:.4f}, {hi:.4f}]",
                n=int(n))


def bootstrap_dict(metric_arrays, n_boot=1000, ci=0.95, rng=None):
    return {k: bootstrap_metric(v, n_boot=n_boot, ci=ci, rng=rng)
            for k, v in metric_arrays.items()}


# ---------------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------------

def fmt_ci(e):
    if e is None or e.get('point') is None or (isinstance(e['point'], float) and np.isnan(e['point'])):
        return 'N/A'
    return e.get('ci_str', 'N/A')


# ---------------------------------------------------------------------------
# REPORT WRITERS
# ---------------------------------------------------------------------------
SEP  = '=' * 80
DASH = '-' * 40


def _header(f, title, timestamp, checkpoint, n_boot):
    f.write(f"{SEP}\n  {title}  (Bootstrap 95% CI)\n{SEP}\n\n")
    f.write(f"Timestamp:  {timestamp}\n")
    f.write(f"Checkpoint: {checkpoint}\n")
    f.write(f"Bootstrap:  {n_boot:,} iterations, 95% CI (percentile method)\n\n")


def write_corpus_report(ci, gen_ci, path, checkpoint, timestamp, n, n_boot):
    with open(path, 'w') as f:
        _header(f, "CORPUS CLASSIFICATION RESULTS", timestamp, checkpoint, n_boot)
        f.write(f"{DASH}\n  Summary Metrics  (point ± half-width  [lo, hi])\n{DASH}\n")
        f.write(f"  N samples:           {n}\n\n")
        f.write(f"  EEG Accuracy:        {fmt_ci(ci.get('eeg_correct'))}\n")
        f.write(f"  Text Acc (Raw):      {fmt_ci(ci.get('raw_correct'))}\n")
        f.write(f"  Text Acc (Gen):      {fmt_ci(ci.get('gen_correct'))}\n")
        if not _isnan(ci.get('llm_gen_correct')):
            f.write(f"  LLM-Pred Acc [gen]: {fmt_ci(ci.get('llm_gen_correct'))}\n")
        if not _isnan(ci.get('llm_raw_correct')):
            f.write(f"  LLM-Pred Acc [raw]: {fmt_ci(ci.get('llm_raw_correct'))}\n")
        _write_gen(f, gen_ci)
        f.write(f"\n{SEP}\n")
    print(f"  → {path}")


def write_sentiment_report(ci, gen_ci, path, checkpoint, timestamp, n, n_boot):
    with open(path, 'w') as f:
        _header(f, "SENTIMENT CLASSIFICATION RESULTS", timestamp, checkpoint, n_boot)
        f.write(f"{DASH}\n  Summary Metrics  (point ± half-width  [lo, hi])\n{DASH}\n")
        f.write(f"  N samples:           {n}\n\n")
        f.write(f"  EEG Accuracy:        {fmt_ci(ci.get('eeg_correct'))}\n")
        f.write(f"  Text Acc (Raw):      {fmt_ci(ci.get('raw_correct'))}\n")
        f.write(f"  Text Acc (Gen):      {fmt_ci(ci.get('gen_correct'))}\n")
        if not _isnan(ci.get('llm_gen_correct')):
            f.write(f"  LLM-Pred Acc [gen]: {fmt_ci(ci.get('llm_gen_correct'))}\n")
        if not _isnan(ci.get('llm_raw_correct')):
            f.write(f"  LLM-Pred Acc [raw]: {fmt_ci(ci.get('llm_raw_correct'))}\n")
        _write_gen(f, gen_ci)
        f.write(f"\n{SEP}\n")
    print(f"  → {path}")


def write_relation_report(ci, gen_ci, path, checkpoint, timestamp, n, n_boot):
    with open(path, 'w') as f:
        _header(f, "RELATION CLASSIFICATION RESULTS", timestamp, checkpoint, n_boot)
        f.write(f"{DASH}\n  Summary Metrics  (point ± half-width  [lo, hi])\n{DASH}\n")
        f.write(f"  N samples:           {n}\n\n")
        f.write(f"  EEG Acc (Top-1):     {fmt_ci(ci.get('eeg_hit1'))}\n")
        f.write(f"  EEG Acc (Top-3):     {fmt_ci(ci.get('eeg_hit3'))}\n")
        f.write(f"  EEG Acc (Top-5):     {fmt_ci(ci.get('eeg_hit5'))}\n")
        f.write(f"  Text Acc Raw (Top-1):{fmt_ci(ci.get('raw_hit1'))}\n")
        f.write(f"  Text Acc Raw (Top-3):{fmt_ci(ci.get('raw_hit3'))}\n")
        f.write(f"  Text Acc Raw (Top-5):{fmt_ci(ci.get('raw_hit5'))}\n")
        f.write(f"  Text Acc Gen (Top-1):{fmt_ci(ci.get('gen_hit1'))}\n")
        f.write(f"  Text Acc Gen (Top-3):{fmt_ci(ci.get('gen_hit3'))}\n")
        f.write(f"  Text Acc Gen (Top-5):{fmt_ci(ci.get('gen_hit5'))}\n")
        if not _isnan(ci.get('llm_gen1')):
            f.write(f"  LLM Acc-Top1 [gen]: {fmt_ci(ci.get('llm_gen1'))}\n")
            f.write(f"  LLM Acc-Top3 [gen]: {fmt_ci(ci.get('llm_gen3'))}\n")
            f.write(f"  LLM Acc-Top5 [gen]: {fmt_ci(ci.get('llm_gen5'))}\n")
        if not _isnan(ci.get('llm_raw1')):
            f.write(f"  LLM Acc-Top1 [raw]: {fmt_ci(ci.get('llm_raw1'))}\n")
            f.write(f"  LLM Acc-Top3 [raw]: {fmt_ci(ci.get('llm_raw3'))}\n")
            f.write(f"  LLM Acc-Top5 [raw]: {fmt_ci(ci.get('llm_raw5'))}\n")
        _write_gen(f, gen_ci)
        f.write(f"\n{SEP}\n")
    print(f"  → {path}")


def write_reading_paradigm_report(ci, path, checkpoint, timestamp, n,
                                   nr_count, tsr_count, n_boot):
    baseline = max(nr_count, tsr_count) / (nr_count + tsr_count) if (nr_count + tsr_count) > 0 else 0.5
    with open(path, 'w') as f:
        _header(f, "READING PARADIGM CLASSIFICATION RESULTS", timestamp, checkpoint, n_boot)
        f.write(f"{DASH}\n  Summary Metrics  (point ± half-width  [lo, hi])\n{DASH}\n")
        f.write(f"  N samples:           {n}\n")
        f.write(f"  NR samples:          {nr_count}\n")
        f.write(f"  TSR samples:         {tsr_count}\n")
        f.write(f"  Random baseline:     {baseline:.4f}\n\n")
        f.write(f"  ACC (NR vs TSR):     {fmt_ci(ci.get('correct'))}\n")
        f.write(f"\n{SEP}\n")
    print(f"  → {path}")


def _isnan(e):
    try:
        return e is None or (isinstance(e.get('point'), float) and np.isnan(e['point']))
    except Exception:
        return True


def _write_gen(f, gen_ci):
    if not gen_ci:
        return
    has = any(not _isnan(gen_ci.get(k)) for k in ['bleu1','bleu2','bleu3','bleu4','rouge1_f'])
    if not has:
        return
    f.write(f"\n{DASH}\n  Generation Quality Metrics  (point ± half-width  [lo, hi])\n{DASH}\n")
    for label, key in [('BLEU-1','bleu1'),('BLEU-2','bleu2'),
                        ('BLEU-3','bleu3'),('BLEU-4','bleu4'),
                        ('ROUGE-1 F','rouge1_f')]:
        if key in gen_ci and not _isnan(gen_ci[key]):
            f.write(f"  {label:<14}: {fmt_ci(gen_ci[key])}\n")


# ---------------------------------------------------------------------------
# PER-VERSION RUNNER
# ---------------------------------------------------------------------------

def process_version(eval_dir, version, n_boot, seed):
    vdir = eval_dir / version
    if not vdir.exists():
        print(f"  [!] {vdir} not found — skipping.")
        return {}

    rng       = np.random.default_rng(seed)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    checkpoint = f"runs/{version}/epoch=199-step=397600.ckpt"
    all_ci    = {}

    def find_pkl(name):
        for cand in [vdir / f"{name}_predictions.pkl",
                     vdir / f"{name}_predictions (1).pkl"]:
            if cand.exists():
                return cand
        return None

    tasks = [
        ('corpus',   extract_corpus_sample_metrics,   write_corpus_report),
        ('sentiment',extract_sentiment_sample_metrics, write_sentiment_report),
        ('relation', extract_relation_sample_metrics,  write_relation_report),
    ]
    for task, extract_fn, write_fn in tasks:
        pkl = find_pkl(task)
        if not pkl:
            print(f"  [!] {task}_predictions.pkl not found"); continue
        print(f"\n[{version}] {task}: {pkl.name}")
        results = pd.read_pickle(pkl)
        if not isinstance(results, list): results = list(results)
        n = len(results)
        print(f"  {n} samples")

        cls_arrs = extract_fn(results)
        cls_ci   = bootstrap_dict(cls_arrs, n_boot=n_boot, rng=rng)

        gen_arrs = extract_generation_metrics(results)
        gen_ci   = bootstrap_dict(gen_arrs, n_boot=n_boot, rng=rng) if gen_arrs else {}

        all_ci[task] = {'classification': cls_ci, 'generation': gen_ci}

        out = vdir / f'bootstrap_{task}_results.txt'
        write_fn(cls_ci, gen_ci, out, checkpoint, timestamp, n, n_boot)

    # Reading paradigm (optional)
    pkl = find_pkl('reading_paradigm_main')
    if pkl:
        print(f"\n[{version}] reading_paradigm_main: {pkl.name}")
        results = pd.read_pickle(pkl)
        if not isinstance(results, list): results = list(results)
        n         = len(results)
        nr_count  = sum(1 for r in results if r.get('label', r.get('label_idx', -1)) == 0)
        tsr_count = sum(1 for r in results if r.get('label', r.get('label_idx', -1)) == 1)
        cls_arrs  = extract_reading_paradigm_metrics(results)
        cls_ci    = bootstrap_dict(cls_arrs, n_boot=n_boot, rng=rng)
        all_ci['reading_paradigm'] = {'classification': cls_ci}
        out = vdir / 'bootstrap_reading_paradigm_results.txt'
        write_reading_paradigm_report(cls_ci, out, checkpoint, timestamp, n,
                                       nr_count, tsr_count, n_boot)

    # JSON dump
    json_path = vdir / 'bootstrap_ci_results.json'
    with open(json_path, 'w') as jf:
        json.dump(_clean(all_ci), jf, indent=2)
    print(f"\n  → JSON: {json_path}")
    return all_ci


# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------

def print_summary_table(all_versions_ci, versions):
    print("\n" + "=" * 106)
    print("  BOOTSTRAP CI SUMMARY  —  95% Confidence Intervals (1,000 iterations)")
    print("=" * 106)
    print(f"  {'Metric':<40}", end="")
    for v in versions:
        print(f"  {v:>30}", end="")
    print()
    print("-" * 106)

    def get(vci, task, sub, key):
        try:
            e = vci[task][sub][key]
            if _isnan(e): return 'N/A'
            return f"{e['point']:.4f} ± {e['hw']:.4f}"
        except Exception:
            return 'N/A'

    sections = [
        ("TABLE 4 — CORPUS", 'corpus', 'classification', [
            ("EEG Accuracy", 'eeg_correct'),
            ("Text Acc (Raw)", 'raw_correct'),
            ("Text Acc (Gen)", 'gen_correct'),
            ("LLM-Pred Acc [gen]", 'llm_gen_correct'),
            ("LLM-Pred Acc [raw]", 'llm_raw_correct'),
        ]),
        ("TABLE 5 — SENTIMENT", 'sentiment', 'classification', [
            ("EEG Accuracy", 'eeg_correct'),
            ("Text Acc (Raw)", 'raw_correct'),
            ("Text Acc (Gen)", 'gen_correct'),
            ("LLM-Pred Acc [gen]", 'llm_gen_correct'),
            ("LLM-Pred Acc [raw]", 'llm_raw_correct'),
        ]),
        ("TABLE 6 — RELATION", 'relation', 'classification', [
            ("EEG Acc (Top-1)", 'eeg_hit1'),
            ("EEG Acc (Top-3)", 'eeg_hit3'),
            ("EEG Acc (Top-5)", 'eeg_hit5'),
            ("Text Acc Raw (Top-1)", 'raw_hit1'),
            ("Text Acc Raw (Top-3)", 'raw_hit3'),
            ("Text Acc Raw (Top-5)", 'raw_hit5'),
            ("Text Acc Gen (Top-1)", 'gen_hit1'),
            ("Text Acc Gen (Top-3)", 'gen_hit3'),
            ("Text Acc Gen (Top-5)", 'gen_hit5'),
            ("LLM Acc-Top1 [gen]", 'llm_gen1'),
            ("LLM Acc-Top3 [gen]", 'llm_gen3'),
            ("LLM Acc-Top5 [gen]", 'llm_gen5'),
            ("LLM Acc-Top1 [raw]", 'llm_raw1'),
            ("LLM Acc-Top3 [raw]", 'llm_raw3'),
            ("LLM Acc-Top5 [raw]", 'llm_raw5'),
        ]),
    ]

    for title, task, sub, metrics in sections:
        print(f"\n  {title}")
        for label, key in metrics:
            vals = [get(all_versions_ci.get(v, {}), task, sub, key) for v in versions]
            if all(v == 'N/A' for v in vals): continue
            print(f"    {label:<38}", end="")
            for val in vals:
                print(f"  {val:>30}", end="")
            print()

    # Generation sections
    for title, task in [("TABLE 4 — CORPUS GEN", 'corpus'),
                         ("TABLE 5 — SENTIMENT GEN", 'sentiment'),
                         ("TABLE 6 — RELATION GEN", 'relation')]:
        shown_header = False
        for label, key in [('BLEU-1','bleu1'),('BLEU-2','bleu2'),
                            ('BLEU-3','bleu3'),('BLEU-4','bleu4'),
                            ('ROUGE-1 F','rouge1_f')]:
            vals = [get(all_versions_ci.get(v, {}), task, 'generation', key) for v in versions]
            if all(v == 'N/A' for v in vals): continue
            if not shown_header:
                print(f"\n  {title}")
                shown_header = True
            print(f"    {label:<38}", end="")
            for val in vals:
                print(f"  {val:>30}", end="")
            print()

    print("\n" + "=" * 106)


# ---------------------------------------------------------------------------
# JSON CLEANER
# ---------------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bootstrap 95% CI for EEG-to-text evaluation")
    parser.add_argument('--n_boot',   type=int,   default=1000)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--eval_dir', type=str,   default='evaluations')
    parser.add_argument('--versions', nargs='+',  default=['v1', 'v2'])
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    print(f"\n{'='*60}")
    print(f"  Bootstrap CI  ({args.n_boot} iterations, seed={args.seed})")
    print(f"  eval_dir: {eval_dir.resolve()}")
    print(f"  versions: {args.versions}")
    print(f"{'='*60}\n")

    all_ci = {}
    for ver in args.versions:
        print(f"\n{'─'*60}\n  Processing: {ver}\n{'─'*60}")
        all_ci[ver] = process_version(eval_dir, ver, args.n_boot, args.seed)

    print_summary_table(all_ci, args.versions)

    combined = eval_dir / 'bootstrap_ci_combined.json'
    with open(combined, 'w') as jf:
        json.dump(_clean(all_ci), jf, indent=2)
    print(f"\nCombined JSON: {combined}")


if __name__ == '__main__':
    main()
