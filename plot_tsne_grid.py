import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from pretraining.pretrain_glim_encoder import GLIMEncoderPretrainer
from pretraining.evaluate_multitask_probe import load_data, extract_features

# Set font family to serif and adjust sizing for two-column paper layout
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'figure.autolayout': False
})

def main():
    ckpt_path = 'pretraining/Results/GLIM_Pretrain5/best_model.pth'
    data_path = 'data/tmp/zuco_eeg_label_8variants.df'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_args = ckpt.get('args', {})

    pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape
    n_patches = pos_embed_shape[1]
    actual_patch_size = 1280 // n_patches

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
    print("✅ Model loaded successfully.")

    # Load data
    train_dataset, test_dataset = load_data(data_path, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Extract test features and labels
    print("🔍 Extracting test features...")
    test_features, test_subj, test_sent, test_rel, test_para, test_sst = extract_features(model, test_loader, device)
    
    # Extract train relation labels only for complete vocabulary mapping
    print("🔍 Extracting train relation labels for vocabulary mapping...")
    _, _, _, train_rel, _, _ = extract_features(model, train_loader, device)

    # Compute a single standard t-SNE coordinate set for the entire test dataset (perplexity=30, learning_rate=200)
    print("🧠 Computing t-SNE projection for the entire test set (N=2227)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42,
        init='pca'
    )
    test_coords = tsne.fit_transform(test_features)
    print("   t-SNE coordinates computed.")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Subplot (a): Subject ID (30 classes)
    # ──────────────────────────────────────────────────────────────────────────
    ax_a = axes[0, 0]
    le_subj = LabelEncoder()
    subj_encoded = le_subj.fit_transform(test_subj)
    
    # Qualitative palette with 30 distinct colors by combining tab20 and tab20b
    cmap_30 = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)[:10]
    
    for i in range(30):
        mask = (subj_encoded == i)
        ax_a.scatter(
            test_coords[mask, 0], test_coords[mask, 1],
            color=cmap_30[i],
            alpha=0.7,
            s=2.5,
            edgecolors='none'
        )
    ax_a.set_title("(a) Subject ID (30 classes)", fontweight='bold')
    ax_a.set_xticklabels([])
    ax_a.set_yticklabels([])
    
    # ──────────────────────────────────────────────────────────────────────────
    # Subplot (b): Relation Type (9 classes)
    # ──────────────────────────────────────────────────────────────────────────
    ax_b = axes[0, 1]
    full_relations = sorted(list(set(
        [r for r in (train_rel + test_rel) if r not in ('nan', 'None', '', 'nan')]
    )))
    le_rel = LabelEncoder()
    le_rel.fit(full_relations)
    
    # Filter and plot only the points with valid relation labels
    rel_valid_mask = np.array([r in full_relations for r in test_rel])
    rel_coords = test_coords[rel_valid_mask]
    rel_labels_valid = [test_rel[i] for i in range(len(test_rel)) if rel_valid_mask[i]]
    rel_encoded = le_rel.transform(rel_labels_valid)
    
    cmap_9 = plt.get_cmap('tab10')
    legend_rel_handles = []
    
    for i, label in enumerate(le_rel.classes_):
        mask = (rel_encoded == i)
        if np.sum(mask) > 0:
            scatter = ax_b.scatter(
                rel_coords[mask, 0], rel_coords[mask, 1],
                color=cmap_9(i),
                alpha=0.75,
                s=3.0,
                edgecolors='none',
                label=label
            )
            legend_rel_handles.append(scatter)
        else:
            # Proxy handle for empty classes to ensure complete legend
            proxy = ax_b.scatter(
                [], [],
                color=cmap_9(i),
                alpha=0.75,
                s=3.0,
                edgecolors='none',
                label=label
            )
            legend_rel_handles.append(proxy)
            
    ax_b.set_title("(b) Relation type (9 classes)", fontweight='bold')
    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.legend(
        handles=legend_rel_handles,
        loc='lower right',
        fontsize=6,
        ncol=2,
        frameon=True,
        edgecolor='#e0e0e0',
        facecolor='#fefefe',
        borderpad=0.3,
        labelspacing=0.2
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Subplot (c): Sentiment (3 classes)
    # ──────────────────────────────────────────────────────────────────────────
    ax_c = axes[1, 0]
    valid_sentiments = ['negative', 'neutral', 'positive']
    
    # Filter and plot only the points with valid sentiment labels
    sent_valid_mask = np.array([s in valid_sentiments for s in test_sent])
    sent_coords = test_coords[sent_valid_mask]
    sent_labels_valid = [test_sent[i] for i in range(len(test_sent)) if sent_valid_mask[i]]
    
    sentiment_colors = {
        'negative': '#d62728', # Red
        'neutral': '#888888',  # Gray
        'positive': '#2ca02c'  # Green
    }
    
    legend_sent_handles = []
    for cls in valid_sentiments:
        mask = np.array([l == cls for l in sent_labels_valid])
        scatter = ax_c.scatter(
            sent_coords[mask, 0], sent_coords[mask, 1],
            color=sentiment_colors[cls],
            alpha=0.8,
            s=3.5,
            edgecolors='none',
            label=cls.capitalize()
        )
        legend_sent_handles.append(scatter)
        
    ax_c.set_title("(c) Sentiment (3 classes)", fontweight='bold')
    ax_c.set_xticklabels([])
    ax_c.set_yticklabels([])
    ax_c.legend(
        handles=legend_sent_handles,
        loc='lower right',
        fontsize=6.5,
        frameon=True,
        edgecolor='#e0e0e0',
        facecolor='#fefefe',
        borderpad=0.3
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Subplot (d): Reading Paradigm (2 classes)
    # ──────────────────────────────────────────────────────────────────────────
    ax_d = axes[1, 1]
    
    paradigm_colors = {
        'NR': '#1f77b4',  # Blue
        'TSR': '#ff7f0e'  # Orange
    }
    
    legend_para_handles = []
    for cls in ['NR', 'TSR']:
        mask = np.array([p == cls for p in test_para])
        label_map = {'NR': 'Natural Reading (NR)', 'TSR': 'Task-Specific Reading (TSR)'}
        scatter = ax_d.scatter(
            test_coords[mask, 0], test_coords[mask, 1],
            color=paradigm_colors[cls],
            alpha=0.75,
            s=2.5,
            edgecolors='none',
            label=label_map[cls]
        )
        legend_para_handles.append(scatter)
        
    ax_d.set_title("(d) Reading paradigm (2 classes)", fontweight='bold')
    ax_d.set_xticklabels([])
    ax_d.set_yticklabels([])
    ax_d.legend(
        handles=legend_para_handles,
        loc='lower right',
        fontsize=6.5,
        frameon=True,
        edgecolor='#e0e0e0',
        facecolor='#fefefe',
        borderpad=0.3
    )

    # Format layout with minimal spacing and borders
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)

    plt.tight_layout()
    
    # Save as PDF
    pdf_path = 'figures/tsne_v2.pdf'
    fig.savefig(pdf_path, dpi=200, bbox_inches='tight')
    print(f"🎉 PDF successfully saved to: {os.path.abspath(pdf_path)}")
    
    # Save as PNG
    png_path = 'figures/tsne_v2.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"🎉 PNG successfully saved to: {os.path.abspath(png_path)}")
    
    plt.close()

if __name__ == '__main__':
    main()
