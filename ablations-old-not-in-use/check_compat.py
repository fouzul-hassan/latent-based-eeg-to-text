"""Check weight compatibility between pretraining checkpoints and GLIM model."""
import torch
import sys
sys.path.insert(0, '.')

# Load checkpoints
ckpt5 = torch.load('pretraining/Results/GLIM_Pretrain5/best_model.pth', map_location='cpu')
enc5 = ckpt5['encoder_state_dict']
print(f"Pretrain5 args: n_blocks={ckpt5['args']['n_blocks']}, emb_size={ckpt5['args']['emb_size']}, num_heads={ckpt5['args']['num_heads']}")

ckpt1 = torch.load('pretraining/Results/GLIM_Pretrain1/best_model.pth', map_location='cpu')
enc1 = ckpt1['encoder_state_dict']
print(f"Pretrain1 args: n_blocks={ckpt1['args']['n_blocks']}, emb_size={ckpt1['args']['emb_size']}, num_heads={ckpt1['args']['num_heads']}")

# Create GLIM with same config as fine-tuned v2
from model.glim import GLIM
glim = GLIM(
    input_eeg_len=1280, hidden_eeg_len=96, input_text_len=96,
    input_dim=128, hidden_dim=256, embed_dim=512,
    n_in_blocks=6, n_out_blocks=6,
    use_gated_attention=True, gating_type='elementwise',
)
gs = glim.eeg_encoder.state_dict()

print("\n=== PRETRAIN5 (8-block/256-dim) vs GLIM (6-block/128-dim) ===")
match5, miss5, mismatch5 = 0, 0, 0
for k, v in enc5.items():
    if k not in gs:
        miss5 += 1
        print(f"  MISSING: {k} ({v.shape})")
    elif gs[k].shape != v.shape:
        mismatch5 += 1
        print(f"  MISMATCH: {k}: ckpt={v.shape} vs glim={gs[k].shape}")
    else:
        match5 += 1
print(f"  Summary: {match5} matched, {mismatch5} shape-mismatch, {miss5} missing")
print(f"  --> {match5}/{len(enc5)} weights would actually transfer!")

print("\n=== PRETRAIN1 (6-block/128-dim) vs GLIM (6-block/128-dim) ===")
match1, miss1, mismatch1 = 0, 0, 0
for k, v in enc1.items():
    if k not in gs:
        miss1 += 1
        print(f"  MISSING: {k} ({v.shape})")
    elif gs[k].shape != v.shape:
        mismatch1 += 1
        print(f"  MISMATCH: {k}: ckpt={v.shape} vs glim={gs[k].shape}")
    else:
        match1 += 1
print(f"  Summary: {match1} matched, {mismatch1} shape-mismatch, {miss1} missing")
print(f"  --> {match1}/{len(enc1)} weights would actually transfer!")

# Also check: how many in_blocks keys does GLIM have total?
in_block_keys = [k for k in gs if k.startswith('in_blocks')]
print(f"\nGLIM eeg_encoder in_blocks total keys: {len(in_block_keys)}")
print(f"GLIM eeg_encoder total keys: {len(gs)}")
