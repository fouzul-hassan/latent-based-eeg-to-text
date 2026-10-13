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

# Set font family to serif and adjust sizing for appendix layout
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.grid': True,
    'grid.alpha': 0.2,
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

    # Extract all features and labels (we use union of train and test labels to fit LabelEncoder)
    print("🔍 Extracting test features...")
    test_features, _, _, test_rel, _, _ = extract_features(model, test_loader, device)
    
    # Extract train labels only for encoding reference to ensure all 9 classes are registered
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    print("🔍 Extracting train relation labels for complete vocabulary...")
    _, _, _, train_rel, _, _ = extract_features(model, train_loader, device)

    # Gather full 9-class vocab (excluding 'nan', 'None', '', 'nan')
    full_relations = sorted(list(set(
        [r for r in (train_rel + test_rel) if r not in ('nan', 'None', '', 'nan')]
    )))
    print(f"📚 Full Relation Classes ({len(full_relations)}): {full_relations}")

    # Initialize LabelEncoder on the full vocabulary
    le = LabelEncoder()
    le.fit(full_relations)

    # Filter test dataset for valid relation classes
    valid_indices = [i for i, r in enumerate(test_rel) if r in full_relations]
    feat_valid = test_features[valid_indices]
    labels_valid = [test_rel[i] for i in valid_indices]

    print(f"📊 Evaluated samples: {len(feat_valid)} test instances")

    # Encode test labels
    labels_encoded = le.transform(labels_valid)

    # Run standard t-SNE (perplexity=30, learning_rate=200)
    print("🧠 Computing t-SNE manifold projection...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42,
        init='pca'
    )
    feat_2d = tsne.fit_transform(feat_valid)

    # Plot creation with exact specifications
    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    # Qualitative colormap 'tab10' for distinct visual clusters
    cmap = plt.get_cmap('tab10')

    # Plot scatter points and save handles for all 9 classes (even if 0 points exist in test set)
    legend_handles = []
    for i, label in enumerate(le.classes_):
        mask = (labels_encoded == i)
        if np.sum(mask) > 0:
            scatter = ax.scatter(
                feat_2d[mask, 0], feat_2d[mask, 1],
                color=cmap(i),
                alpha=0.85,
                s=3.5,
                edgecolors='none',
                label=label
            )
            legend_handles.append(scatter)
        else:
            # Proxy handle for empty classes to ensure complete legend
            proxy = ax.scatter(
                [], [],
                color=cmap(i),
                alpha=0.85,
                s=3.5,
                edgecolors='none',
                label=label
            )
            legend_handles.append(proxy)

    ax.set_title("t-SNE projection of frozen GRAPE-V2 embeddings", fontweight='bold', pad=6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Place legend below the plot to fit single-column limits gracefully
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=True,
        edgecolor='#e0e0e0',
        facecolor='#fafafa',
        title="Relation Type",
        title_fontsize=7.5
    )

    # Ensure directories exist and save
    os.makedirs('figures', exist_ok=True)
    pdf_path = 'figures/tsne_v2.pdf'
    fig.savefig(pdf_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🎉 Plot successfully saved to: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    main()
