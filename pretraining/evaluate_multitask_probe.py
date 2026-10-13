"""
Multi-Task Linear Probe Evaluation for Pretrained JEPA Encoder.

Evaluates the pretrained encoder on 3 classification tasks using frozen features:
  1. Subject ID      — whose brain produced this EEG?
  2. Sentiment        — what sentiment is conveyed?
  3. Relation Type    — what relation type is present?

Usage:
    cd pretraining
    python evaluate_multitask_probe.py

    # With custom paths:
    python evaluate_multitask_probe.py --ckpt Results/GLIM_Pretrain1/best_model.pth --data ../data/tmp/zuco_eeg_label_8variants.df
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from collections import Counter

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer, count_parameters


# ── Dataset ──────────────────────────────────────────────────────────────────
class ZuCoMultiLabelDataset(Dataset):
    """
    Loads ZuCo data with multiple label types (subject, sentiment, relation).
    Uses zuco_eeg_label_8variants.df which contains all annotations.
    """
    def __init__(self, df: pd.DataFrame, indices: np.ndarray):
        self.df = df
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        eeg = row['eeg']             # (128, 1280) or (1280, 128)
        if eeg.shape[0] == 128:
            eeg = eeg.T              # → (1280, 128)
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

        subject = str(row.get('subject', 'unknown'))
        sentiment = str(row.get('sentiment label', 'nan'))
        relation = str(row.get('relation label', 'nan'))
        task = str(row.get('task', ''))
        paradigm = 'NR' if ('task1' in task or 'task2' in task) else ('TSR' if 'task3' in task else 'nan')
        is_sst = ('task1' in task)

        return eeg_tensor, subject, sentiment, relation, paradigm, is_sst


def load_data(data_path: str, seed: int = 42, val_ratio=0.1, test_ratio=0.1):
    """Load dataset and split into train/test, returning DataFrames."""
    print(f"📂 Loading data from {data_path}")
    df = pd.read_pickle(data_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # If data already has phase column, use it
    if 'phase' in df.columns:
        train_df = df[df['phase'] == 'train'].reset_index(drop=True)
        test_df = df[df['phase'] == 'test'].reset_index(drop=True)
        if len(test_df) == 0:
            test_df = df[df['phase'] == 'val'].reset_index(drop=True)
        train_indices = np.arange(len(train_df))
        test_indices = np.arange(len(test_df))
        return (ZuCoMultiLabelDataset(train_df, train_indices),
                ZuCoMultiLabelDataset(test_df, test_indices))
    else:
        # Manual split
        np.random.seed(seed)
        n = len(df)
        indices = np.arange(n)
        np.random.shuffle(indices)
        test_size = int(n * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return (ZuCoMultiLabelDataset(df, train_indices),
                ZuCoMultiLabelDataset(df, test_indices))


# ── Feature Extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device):
    """Extract features + all label types from the frozen encoder."""
    model.eval()
    all_features = []
    all_subjects = []
    all_sentiments = []
    all_relations = []
    all_paradigms = []
    all_is_sst = []

    for batch in loader:
        eeg, subjects, sentiments, relations, paradigms, is_sst = batch
        eeg = eeg.to(device)
        features = model(eeg)  # (B, emb_size)
        all_features.append(features.cpu())
        all_subjects.extend(subjects)
        all_sentiments.extend(sentiments)
        all_relations.extend(relations)
        all_paradigms.extend(paradigms)
        all_is_sst.extend(is_sst.tolist())

    features = torch.cat(all_features, dim=0).numpy()
    return features, all_subjects, all_sentiments, all_relations, all_paradigms, all_is_sst


# ── Linear Probe ─────────────────────────────────────────────────────────────
def run_linear_probe(train_features, train_labels, test_features, test_labels, task_name):
    """Run logistic regression linear probe for a single task."""
    # Encode string labels to integers
    le = LabelEncoder()
    all_labels = train_labels + test_labels
    le.fit(all_labels)

    train_y = le.transform(train_labels)
    test_y = le.transform(test_labels)

    n_classes = len(le.classes_)
    random_baseline = 1.0 / n_classes

    print(f"\n{'─'*60}")
    print(f"📊 {task_name}")
    print(f"{'─'*60}")
    print(f"   Classes ({n_classes}): {list(le.classes_)}")
    print(f"   Train: {len(train_y)} | Test: {len(test_y)}")
    print(f"   Random baseline: {random_baseline:.4f} ({100*random_baseline:.1f}%)")

    # Check class distribution
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    print(f"   Train distribution: {dict(train_dist)}")
    print(f"   Test distribution: {dict(test_dist)}")

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(train_features, train_y)

    train_pred = clf.predict(train_features)
    test_pred = clf.predict(test_features)

    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)
    improvement = test_acc / random_baseline

    print(f"\n   ✅ Results:")
    print(f"   Train accuracy: {train_acc:.4f} ({100*train_acc:.1f}%)")
    print(f"   Test accuracy:  {test_acc:.4f} ({100*test_acc:.1f}%)")
    print(f"   vs Random:      {improvement:.2f}× better")

    if n_classes <= 20:
        print(f"\n   📋 Per-Class Report:")
        report = classification_report(test_y, test_pred,
                                        target_names=le.classes_, digits=3)
        for line in report.split('\n'):
            print(f"   {line}")

    return {
        'task': task_name,
        'n_classes': n_classes,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'random_baseline': random_baseline,
        'improvement_over_random': improvement,
        'label_encoder': le,
        'classifier': clf,
    }


# ── Visualization ────────────────────────────────────────────────────────────
def visualize_multitask_tsne(features, subjects, sentiments, relations, paradigms, is_sst_list, save_dir):
    """Create t-SNE plots colored by each label type."""
    print("\n🎨 Creating t-SNE visualizations...")

    n_samples = min(3000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    feat_sample = features[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    feat_2d = tsne.fit_transform(feat_sample)

    paradigms_no_sst = [p if not sst else 'nan' for p, sst in zip(paradigms, is_sst_list)]

    label_sets = {
        'Subject ID': [subjects[i] for i in indices],
        'Paradigm (No SST)': [paradigms_no_sst[i] for i in indices],
        'Paradigm (With SST)': [paradigms[i] for i in indices],
        'Sentiment': [sentiments[i] for i in indices],
        'Relation': [relations[i] for i in indices],
    }

    fig, axes = plt.subplots(1, 5, figsize=(35, 6))
    fig.suptitle('t-SNE of Pretrained EEG Features (colored by label type)', fontsize=14, fontweight='bold')

    for ax, (title, labels) in zip(axes, label_sets.items()):
        # Filter out 'nan' labels for cleaner visualization
        valid_mask = [l not in ('nan', 'None', '') for l in labels]
        valid_feat = feat_2d[valid_mask]
        valid_labels = [l for l, v in zip(labels, valid_mask) if v]

        if len(valid_labels) == 0:
            ax.text(0.5, 0.5, 'No valid labels', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        le = LabelEncoder()
        encoded = le.fit_transform(valid_labels)
        n_classes = len(le.classes_)

        cmap = plt.cm.tab20 if n_classes <= 20 else plt.cm.viridis
        scatter = ax.scatter(valid_feat[:, 0], valid_feat[:, 1],
                            c=encoded, cmap=cmap, alpha=0.5, s=8)

        ax.set_title(f'{title} ({n_classes} classes)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        # Add legend for small number of classes
        if n_classes <= 10:
            handles = []
            for i, cls in enumerate(le.classes_):
                h = ax.scatter([], [], c=[cmap(i / max(1, n_classes - 1))], s=40, label=cls)
                handles.append(h)
            ax.legend(handles=handles, loc='best', fontsize=7, ncol=1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'multitask_tsne.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved to {save_path}")


def plot_probe_summary(results, save_dir):
    """Bar chart comparing all probe tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = [r['task'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    baselines = [r['random_baseline'] for r in results]
    improvements = [r['improvement_over_random'] for r in results]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, test_accs, width, label='Linear Probe Accuracy',
                   color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, baselines, width, label='Random Baseline',
                   color='#bdc3c7', edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, acc, imp in zip(bars1, test_accs, improvements):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{100*acc:.1f}%\n({imp:.1f}× random)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, bl in zip(bars2, baselines):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{100*bl:.1f}%', ha='center', va='bottom', fontsize=9, color='gray')

    ax.set_ylabel('Accuracy')
    ax.set_title('Multi-Task Linear Probe: Pretrained JEPA Encoder', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_ylim(0, max(test_accs) * 1.4)
    ax.grid(axis='y', alpha=0.3)

    save_path = os.path.join(save_dir, 'multitask_probe_summary.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   📊 Summary chart saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Multi-Task Linear Probe Evaluation')
    parser.add_argument('--ckpt', type=str, default='Results/GLIM_Pretrain1/best_model.pth',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--data', type=str, default='../data/tmp/zuco_eeg_label_8variants.df',
                        help='Path to ZuCo dataset with labels')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots (default: same as checkpoint)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    save_dir = args.save_dir or os.path.dirname(args.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  MULTI-TASK LINEAR PROBE EVALUATION")
    print("  Pretrained JEPA Encoder")
    print("=" * 60)

    # ── Load checkpoint ──
    print(f"\n📦 Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt_args = ckpt.get('args', {})

    # Infer patch_size from checkpoint
    pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape
    n_patches = pos_embed_shape[1]
    actual_patch_size = 1280 // n_patches

    print(f"   Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"   Config: emb_size={ckpt_args.get('emb_size', 128)}, "
          f"n_blocks={ckpt_args.get('n_blocks', 6)}, "
          f"patch_size={actual_patch_size}, "
          f"num_heads={ckpt_args.get('num_heads', 8)}")

    model = GLIMEncoderPretrainer(
        in_len=1280,
        in_dim=128,
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
    print(f"   ✅ Model loaded ({count_parameters(model):,} parameters)")

    # ── Load data ──
    train_dataset, test_dataset = load_data(args.data, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    print(f"   Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

    # ── Extract features ──
    print("\n🔍 Extracting features from frozen encoder...")
    train_features, train_subj, train_sent, train_rel, train_para, train_sst = extract_features(model, train_loader, device)
    test_features, test_subj, test_sent, test_rel, test_para, test_sst = extract_features(model, test_loader, device)
    print(f"   Feature shape: {train_features.shape}")

    # ── Feature statistics ──
    print("\n📈 Feature Statistics:")
    std = train_features.std(axis=0)
    print(f"   Dim: {train_features.shape[1]}")
    print(f"   Mean std: {std.mean():.4f} | Min std: {std.min():.4f}")
    if std.min() < 0.01:
        print(f"   ⚠️ Some dimensions have near-zero variance!")
    else:
        print(f"   ✅ No feature collapse")

    # ── Run all probes ──
    results = []

    # 1. Subject ID
    result_subj = run_linear_probe(train_features, train_subj,
                                    test_features, test_subj,
                                    "Subject ID Classification")
    results.append(result_subj)

    # 2. Sentiment (filter out 'nan')
    train_sent_valid = [(f, l) for f, l in zip(range(len(train_sent)), train_sent)
                        if l not in ('nan', 'None', '')]
    test_sent_valid = [(f, l) for f, l in zip(range(len(test_sent)), test_sent)
                       if l not in ('nan', 'None', '')]

    if len(train_sent_valid) > 10 and len(test_sent_valid) > 5:
        train_sent_idx, train_sent_labels = zip(*train_sent_valid)
        test_sent_idx, test_sent_labels = zip(*test_sent_valid)
        result_sent = run_linear_probe(
            train_features[list(train_sent_idx)], list(train_sent_labels),
            test_features[list(test_sent_idx)], list(test_sent_labels),
            "Sentiment Classification"
        )
        results.append(result_sent)
    else:
        print("\n⚠️ Insufficient sentiment labels for probe")

    # 3. Relation (filter out 'nan')
    train_rel_valid = [(f, l) for f, l in zip(range(len(train_rel)), train_rel)
                       if l not in ('nan', 'None', '')]
    test_rel_valid = [(f, l) for f, l in zip(range(len(test_rel)), test_rel)
                      if l not in ('nan', 'None', '')]

    if len(train_rel_valid) > 10 and len(test_rel_valid) > 5:
        train_rel_idx, train_rel_labels = zip(*train_rel_valid)
        test_rel_idx, test_rel_labels = zip(*test_rel_valid)
        result_rel = run_linear_probe(
            train_features[list(train_rel_idx)], list(train_rel_labels),
            test_features[list(test_rel_idx)], list(test_rel_labels),
            "Relation Classification"
        )
        results.append(result_rel)
    else:
        print("\n⚠️ Insufficient relation labels for probe")

    # 4. Reading Paradigm (NR vs TSR) - Without SST
    train_para_no_sst = [(f, l) for f, l, sst in zip(range(len(train_para)), train_para, train_sst)
                       if l not in ('nan', 'None', '') and not sst]
    test_para_no_sst = [(f, l, sst) for f, l, sst in zip(range(len(test_para)), test_para, test_sst)
                      if l not in ('nan', 'None', '') and not sst]

    if len(train_para_no_sst) > 10 and len(test_para_no_sst) > 5:
        train_para_idx, train_para_labels = zip(*train_para_no_sst)
        test_para_idx, test_para_labels, _ = zip(*test_para_no_sst)
        result_para_no_sst = run_linear_probe(
            train_features[list(train_para_idx)], list(train_para_labels),
            test_features[list(test_para_idx)], list(test_para_labels),
            "Paradigm (NR vs TSR) - No SST"
        )
        results.append(result_para_no_sst)
    else:
        print("\n⚠️ Insufficient paradigm labels for No-SST probe")

    # 5. Reading Paradigm (NR vs TSR) - With SST
    train_para_sst = [(f, l) for f, l in zip(range(len(train_para)), train_para)
                       if l not in ('nan', 'None', '')]
    test_para_sst = [(f, l) for f, l in zip(range(len(test_para)), test_para)
                      if l not in ('nan', 'None', '')]

    if len(train_para_sst) > 10 and len(test_para_sst) > 5:
        train_para_idx, train_para_labels = zip(*train_para_sst)
        test_para_idx, test_para_labels = zip(*test_para_sst)
        result_para_sst = run_linear_probe(
            train_features[list(train_para_idx)], list(train_para_labels),
            test_features[list(test_para_idx)], list(test_para_labels),
            "Paradigm (NR vs TSR) - WIth SST"
        )
        results.append(result_para_sst)
    else:
        print("\n⚠️ Insufficient paradigm labels for With-SST probe")

    # ── Visualizations ──
    visualize_multitask_tsne(test_features, test_subj, test_sent, test_rel, test_para, test_sst, save_dir)

    plot_probe_summary(results, save_dir)

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  📋 MULTI-TASK LINEAR PROBE SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<30} {'Accuracy':>10} {'Random':>10} {'vs Random':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        print(f"  {r['task']:<30} {100*r['test_acc']:>9.1f}% {100*r['random_baseline']:>9.1f}% {r['improvement_over_random']:>9.1f}×")

    # Verdict
    # print(f"\n  🎯 Verdict:")
    # avg_improvement = np.mean([r['improvement_over_random'] for r in results])
    # if avg_improvement > 2.0:
    #     print(f"  ✅ GOOD: Encoder learned meaningful, multi-aspect EEG features (avg {avg_improvement:.1f}× random)")
    # elif avg_improvement > 1.5:
    #     print(f"  ✅ DECENT: Encoder captures some discriminative patterns (avg {avg_improvement:.1f}× random)")
    # else:
    #     print(f"  ⚠️ MODEST: Features are marginally better than random (avg {avg_improvement:.1f}× random)")

    # semantic_tasks = [r for r in results if r['task'] != 'Subject ID Classification']
    # if semantic_tasks:
    #     avg_semantic = np.mean([r['improvement_over_random'] for r in semantic_tasks])
    #     if avg_semantic > 1.5:
    #         print(f"  ✅ Semantic probes ({avg_semantic:.1f}× random) show the encoder captures MEANING from EEG!")
    #         print(f"     This directly supports the world model hypothesis.")
    #     else:
    #         print(f"  ⚠️ Semantic probes ({avg_semantic:.1f}× random) show limited semantic capture.")

    print("=" * 60)

    # Save results to text file
    results_path = os.path.join(save_dir, 'multitask_probe_results.txt')
    with open(results_path, 'w') as f:
        f.write("Multi-Task Linear Probe Results\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"Task: {r['task']}\n")
            f.write(f"  Classes: {r['n_classes']}\n")
            f.write(f"  Train acc: {100*r['train_acc']:.1f}%\n")
            f.write(f"  Test acc:  {100*r['test_acc']:.1f}%\n")
            f.write(f"  Random:    {100*r['random_baseline']:.1f}%\n")
            f.write(f"  vs Random: {r['improvement_over_random']:.1f}×\n\n")
    print(f"\n  💾 Results saved to {results_path}")


if __name__ == '__main__':
    main()
