"""Detailed check: which keys match for Pretrain1 vs GLIM-G (gated attention)."""
import torch, sys
sys.path.insert(0, '.')

ckpt1 = torch.load('pretraining/Results/GLIM_Pretrain1/best_model.pth', map_location='cpu')
enc1 = ckpt1['encoder_state_dict']

from model.glim import GLIM

# v2 uses gated attention
glim_gated = GLIM(input_eeg_len=1280, hidden_eeg_len=96, input_text_len=96,
    input_dim=128, hidden_dim=256, embed_dim=512, n_in_blocks=6, n_out_blocks=6,
    use_gated_attention=True, gating_type='elementwise')
gs_gated = glim_gated.eeg_encoder.state_dict()

# v1 uses standard attention
glim_std = GLIM(input_eeg_len=1280, hidden_eeg_len=96, input_text_len=96,
    input_dim=128, hidden_dim=256, embed_dim=512, n_in_blocks=6, n_out_blocks=6,
    use_gated_attention=False)
gs_std = glim_std.eeg_encoder.state_dict()

print("=== Pretrain1 key names (first 20) ===")
for i, k in enumerate(enc1.keys()):
    if i >= 20: break
    print(f"  {k}")

print("\n=== GLIM-G (gated) in_blocks key names ===")
gated_in = sorted([k for k in gs_gated if k.startswith('in_blocks')])
for k in gated_in:
    print(f"  {k}: {gs_gated[k].shape}")

print("\n=== GLIM (standard) in_blocks key names ===")
std_in = sorted([k for k in gs_std if k.startswith('in_blocks')])
for k in std_in:
    print(f"  {k}: {gs_std[k].shape}")

print(f"\nGated in_blocks keys: {len(gated_in)}")
print(f"Standard in_blocks keys: {len(std_in)}")
print(f"Pretrain1 encoder keys: {len(enc1)}")

# Matched keys for gated vs pretrain1
matched_gated = [k for k in enc1 if k in gs_gated and gs_gated[k].shape == enc1[k].shape]
matched_std = [k for k in enc1 if k in gs_std and gs_std[k].shape == enc1[k].shape]
print(f"\nPretrain1 → GLIM-G (gated): {len(matched_gated)}/{len(enc1)} matched")
print(f"Pretrain1 → GLIM (standard): {len(matched_std)}/{len(enc1)} matched")

print("\nMatched keys (gated):")
for k in matched_gated:
    print(f"  {k}")

print("\nMissing from gated (in pretrain1 but not in GLIM-G):")
for k in enc1:
    if k not in gs_gated:
        print(f"  {k} (pretrain) -> NOT IN GLIM-G")
    elif gs_gated[k].shape != enc1[k].shape:
        print(f"  {k}: pretrain={enc1[k].shape} vs glim-g={gs_gated[k].shape}")
