"""
GLIM-Compatible EEG Encoder Pretraining Module.

Uses GLIM's EncoderBlock architecture with EEG2Rep's self-supervised objective:
- Semantic Subsequence Preserving (SSP) masking  [V2 default]
- VICReg loss (alignment + variance + covariance) [V2 default]
- EMA target encoder                              [V2 default]

Ablation switches (all default to V2 behaviour):
  masking_strategy : 'ssp' (default) | 'random'   -- A1
  std_coeff        : float  default 1.0            -- A2 (set 0.0 to ablate)
  cov_coeff        : float  default 1.0            -- A2 (set 0.0 to ablate)
  pretraining_mode : 'jepa' (default) | 'mae'      -- A3
"""
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

# Import GLIM's EncoderBlock — resolve project root relative to this file
import sys, os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from model.modules import EncoderBlock, get_1d_sincos_pos_embed_from_grid


class GLIMEncoderPretrainer(nn.Module):
    """
    Wraps GLIM's EncoderBlocks for self-supervised pretraining.

    Architecture (JEPA mode – default / V2):
    - Input embedding (patchify EEG)
    - Context encoder (N x EncoderBlock from GLIM, optionally gated)
    - Target encoder (EMA copy, no gradients)
    - Cross-attention predictor → predicts latent targets
    Loss: VICReg (alignment + variance + covariance)

    Architecture (MAE mode – A3):
    - Input embedding, context encoder, predictor (identical)
    - NO target encoder
    - Predictor output projected to raw patch values
    Loss: MSE on raw patch pixels
    """

    def __init__(self,
                 in_len: int = 1280,
                 in_dim: int = 128,
                 emb_size: int = 128,
                 n_blocks: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 patch_size: int = 8,
                 mask_ratio: float = 0.5,
                 momentum: float = 0.99,
                 use_gated_attention: bool = False,
                 predictor_layers: int = 2,
                 # ── Ablation knobs ──────────────────────────────────────────
                 masking_strategy: str = 'ssp',   # 'ssp' | 'random'  (A1)
                 std_coeff: float = 1.0,           # VICReg std weight   (A2)
                 cov_coeff: float = 1.0,           # VICReg cov weight   (A2)
                 pretraining_mode: str = 'jepa',  # 'jepa' | 'mae'      (A3)
                 ):
        super().__init__()

        assert masking_strategy in ('ssp', 'random'), \
            f"masking_strategy must be 'ssp' or 'random', got '{masking_strategy}'"
        assert pretraining_mode in ('jepa', 'mae'), \
            f"pretraining_mode must be 'jepa' or 'mae', got '{pretraining_mode}'"

        self.in_len = in_len
        self.in_dim = in_dim
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.momentum = momentum
        self.masking_strategy = masking_strategy
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.pretraining_mode = pretraining_mode

        # Number of patches after patchifying
        self.num_patches = in_len // patch_size  # e.g. 1280 / 8 = 160

        # ── Input embedding (patchify + project) ─────────────────────────
        self.patch_embed = nn.Sequential(
            nn.Linear(in_dim * patch_size, emb_size),
            nn.LayerNorm(emb_size),
            nn.GELU(),
        )

        # ── Positional encoding (frozen sinusoidal) ───────────────────────
        pos_embed = get_1d_sincos_pos_embed_from_grid(emb_size, np.arange(self.num_patches))
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            requires_grad=False)

        # ── Context encoder ───────────────────────────────────────────────
        self.context_encoder = nn.ModuleList([
            EncoderBlock(emb_size, self.num_patches,
                         inject_prompt=False,
                         temporal_modulate=False,
                         is_causal=False,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         dropout=dropout,
                         use_gated_attention=use_gated_attention)
            for _ in range(n_blocks)
        ])

        # ── Target encoder (EMA copy) — JEPA only ────────────────────────
        if pretraining_mode == 'jepa':
            self.target_encoder = copy.deepcopy(self.context_encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            # A3: no target encoder; assertion guard in training loop
            self.target_encoder = None

        # ── Learnable mask token ──────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.randn(emb_size))

        # ── Predictor (cross-attention: mask tokens → context output) ─────
        self.predictor = nn.ModuleList([
            CrossAttnPredictorBlock(emb_size, num_heads, dropout)
            for _ in range(predictor_layers)
        ])
        self.predictor_norm = nn.LayerNorm(emb_size)

        # ── MAE reconstruction head (A3 only) ────────────────────────────
        if pretraining_mode == 'mae':
            # Projects from embedding dim → raw patch values
            self.mae_head = nn.Linear(emb_size, patch_size * in_dim)
        else:
            self.mae_head = None

        # ── Global average pooling (used by VICReg std/cov) ───────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

    # ── EMA helpers ───────────────────────────────────────────────────────────

    def copy_weight(self):
        """Copy context encoder weights to target encoder (JEPA only)."""
        if self.target_encoder is None:
            return  # A3: no-op
        with torch.no_grad():
            for param_ctx, param_tgt in zip(self.context_encoder.parameters(),
                                            self.target_encoder.parameters()):
                param_tgt.data = param_ctx.data

    def momentum_update(self):
        """EMA update of target encoder (JEPA only)."""
        if self.target_encoder is None:
            return  # A3: no-op
        with torch.no_grad():
            for param_ctx, param_tgt in zip(self.context_encoder.parameters(),
                                            self.target_encoder.parameters()):
                param_tgt.data = (self.momentum * param_tgt.data
                                  + (1 - self.momentum) * param_ctx.data)

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def patchify_raw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw (un-embedded) patch values for MAE targets.

        Args:
            x: (B, T, C) EEG signal
        Returns:
            patches_raw: (B, num_patches, patch_size * C)
        """
        B, T, C = x.shape
        return x.reshape(B, self.num_patches, self.patch_size * C)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify + project to embedding dim."""
        B, T, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        return self.patch_embed(x)  # (B, num_patches, emb_size)

    def encode_context(self, patches: torch.Tensor) -> torch.Tensor:
        """Forward through context encoder."""
        x = patches
        for block in self.context_encoder:
            x = block(x, mask=None, p=None)
        return x

    def encode_target(self, patches: torch.Tensor) -> torch.Tensor:
        """Forward through target encoder (no gradients, JEPA only)."""
        assert self.target_encoder is not None, \
            "encode_target() called in MAE mode — no target encoder exists."
        with torch.no_grad():
            x = patches
            for block in self.target_encoder:
                x = block(x, mask=None, p=None)
        return x

    # ── Masking strategies ────────────────────────────────────────────────────

    def _apply_mask(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select visible / masked indices according to masking_strategy.

        Returns:
            v_index: sorted visible patch indices
            m_index: sorted masked patch indices
        """
        if self.masking_strategy == 'ssp':
            # V2 default: contiguous semantic chunks
            visible_chunks = semantic_subsequence_preserving(
                indices, chunk_count=2, target_percentage=self.mask_ratio)
            v_index = np.unique(np.ravel(visible_chunks))
            m_index = np.setdiff1d(indices, v_index)
        else:
            # A1: i.i.d. random masking at same ratio
            v_index, m_index = random_masking(indices, mask_ratio=self.mask_ratio)
            # Sanity check: mask must be non-contiguous on average
            # (not guaranteed per sample, but checked in unit tests)
        return v_index, m_index

    # ── Main pretraining forward ──────────────────────────────────────────────

    def pretrain_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for pretraining (both JEPA and MAE modes).

        Args:
            x: (B, T, C) or (B, C, T) EEG input

        Returns:
            rep_target:     (B, num_masked, D)  — latent targets (JEPA) OR
                            (B, num_masked, patch_size*C)  — raw targets (MAE)
            rep_prediction: (B, num_masked, D)  — predictor output
            rep_context:    (B, num_visible, emb_size) — context enc output
            rep_target_full:(B, num_patches, emb_size) — full target (JEPA) / None (MAE)
        """
        # Normalise input shape to (B, T, C)
        if x.shape[1] == self.in_dim and x.shape[2] == self.in_len:
            x = x.transpose(1, 2)

        B = x.shape[0]

        # Raw patches (needed for MAE target before embedding)
        patches_raw = self.patchify_raw(x)          # (B, num_patches, patch_size*C)

        # Embedded patches + positional encoding
        patches = self.patch_embed(patches_raw)     # (B, num_patches, emb_size)
        patches = patches + self.pos_embed

        # ── Masking ──────────────────────────────────────────────────────
        indices = np.arange(self.num_patches)
        v_index, m_index = self._apply_mask(indices)

        # Visible patches → context encoder
        visible = patches[:, v_index, :]             # (B, num_visible, emb_size)
        rep_context = self.encode_context(visible)   # (B, num_visible, emb_size)

        # ── Target ───────────────────────────────────────────────────────
        if self.pretraining_mode == 'jepa':
            # JEPA: latent target from EMA encoder
            rep_target_full = self.encode_target(patches)    # (B, num_patches, emb_size)
            rep_target = rep_target_full[:, m_index, :]      # (B, num_masked, emb_size)
        else:
            # MAE (A3): raw patch pixels as target
            rep_target_full = None
            rep_target = patches_raw[:, m_index, :]          # (B, num_masked, patch_size*C)

        # ── Predictor ────────────────────────────────────────────────────
        # Mask tokens with positional encoding
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, len(m_index), -1)
        mask_tokens = mask_tokens + self.pos_embed[:, m_index, :]

        # Cross-attend to context
        rep_prediction = mask_tokens
        for pred_block in self.predictor:
            rep_prediction = pred_block(rep_prediction, rep_context)
        rep_prediction = self.predictor_norm(rep_prediction)  # (B, num_masked, emb_size)

        # Project to raw space in MAE mode
        if self.pretraining_mode == 'mae':
            rep_prediction = self.mae_head(rep_prediction)    # (B, num_masked, patch_size*C)

        return rep_target, rep_prediction, rep_context, rep_target_full

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(self, rep_target: torch.Tensor,
                     rep_prediction: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.

        JEPA mode (default / V2): VICReg  =  align + std_coeff*std + cov_coeff*cov
        MAE  mode (A3):           pure MSE on raw patches, std/cov logged as 0.0
        A2 (MSE-only):            std_coeff=cov_coeff=0 → align only

        Returns:
            total_loss: scalar
            loss_dict:  {'align', 'std', 'cov', 'total', 'target_per_dim_std'}
        """
        # Alignment / reconstruction loss (always present)
        align_loss = F.mse_loss(rep_prediction, rep_target)

        if self.pretraining_mode == 'mae':
            # A3: no VICReg terms
            std_loss_val = 0.0
            cov_loss_val = 0.0
            total_loss = align_loss
        else:
            # JEPA: VICReg on pooled predictions
            y = self.gap(rep_prediction.transpose(1, 2)).squeeze(-1)  # (B, emb_size)
            y = y - y.mean(dim=0)

            # Variance loss (penalise dims with std < 1)
            std_y = torch.sqrt(y.var(dim=0) + 1e-4)
            std_loss = torch.mean(F.relu(1 - std_y))

            # Covariance loss (decorrelate features)
            batch_size = y.shape[0]
            cov_y = (y.T @ y) / (batch_size - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum() / y.shape[-1]

            # Store raw tensors for logging even when coefficients are 0
            std_loss_val = std_loss.item()
            cov_loss_val = cov_loss.item()

            total_loss = align_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        # Per-dimension std of TARGET representations (collapse diagnostic for A2)
        with torch.no_grad():
            if self.pretraining_mode == 'jepa':
                # rep_target: (B, num_masked, emb_size) — pool over masked positions
                tgt_pooled = rep_target.mean(dim=1)   # (B, emb_size)
                target_per_dim_std = tgt_pooled.std(dim=0).mean().item()
            else:
                target_per_dim_std = 0.0   # not meaningful in MAE mode

        loss_dict = {
            'align': align_loss.item(),
            'std': std_loss_val,
            'cov': cov_loss_val,
            'total': total_loss.item(),
            'target_per_dim_std': target_per_dim_std,
        }

        return total_loss, loss_dict

    # ── Inference forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward: encode ALL patches, return pooled representation.

        Args:
            x: (B, T, C) or (B, C, T)
        Returns:
            (B, emb_size) pooled features
        """
        if x.shape[1] == self.in_dim and x.shape[2] == self.in_len:
            x = x.transpose(1, 2).contiguous()
        patches = self.patchify(x)
        patches = patches + self.pos_embed
        out = self.encode_context(patches)
        out = self.gap(out.transpose(1, 2)).squeeze(-1)  # (B, emb_size)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_encoder_state_dict(self) -> dict:
        """State dict for context encoder (for transfer to GLIM downstream)."""
        return {f'in_blocks.{i}.{key}': val
                for i, block in enumerate(self.context_encoder)
                for key, val in block.state_dict().items()}

    def get_encoder_gate_stats(self) -> dict:
        """
        Collect gate statistics from all gated context encoder blocks.

        Returns aggregated stats over all blocks (mean across layers).
        Returns zeros when gated attention is not used.
        """
        means, stds, sparsities = [], [], []
        for block in self.context_encoder:
            if hasattr(block, 'attn') and hasattr(block.attn, 'get_gate_stats'):
                s = block.attn.get_gate_stats()
                means.append(s.get('gate_mean', 0.0))
                stds.append(s.get('gate_std', 0.0))
                sparsities.append(s.get('gate_sparsity', 0.0))
        if means:
            return {
                'gate_mean':     float(np.mean(means)),
                'gate_std':      float(np.mean(stds)),
                'gate_sparsity': float(np.mean(sparsities)),
            }
        return {'gate_mean': 0.0, 'gate_std': 0.0, 'gate_sparsity': 0.0}


# ── Masking functions ─────────────────────────────────────────────────────────

def semantic_subsequence_preserving(indices: np.ndarray,
                                    chunk_count: int = 2,
                                    target_percentage: float = 0.5) -> List[np.ndarray]:
    """
    Semantic Subsequence Preserving (SSP) masking from EEG2Rep.
    Selects `chunk_count` contiguous chunks totalling ~`target_percentage` of patches
    as *visible*; the remainder is masked.

    Args:
        indices:          array of all patch indices
        chunk_count:      number of contiguous chunks to keep visible
        target_percentage: fraction of patches kept visible (i.e. 1 - mask_ratio)
    Returns:
        List of chunk arrays (visible indices)
    """
    total = len(indices)
    target_total = int(total * target_percentage)
    chunk_size = target_total // chunk_count

    start_points = [random.randint(0, total - chunk_size) for _ in range(chunk_count)]
    chunks = [indices[start:start + chunk_size] for start in start_points]
    return chunks


def random_masking(indices: np.ndarray,
                   mask_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    A1: I.I.D. random masking at the same ratio as SSP.

    Each patch is independently included in the visible set with probability
    (1 - mask_ratio).  The resulting mask is intentionally non-contiguous.

    Args:
        indices:    array of all patch indices
        mask_ratio: fraction to mask (same as SSP's mask_ratio)
    Returns:
        v_index: sorted visible patch indices
        m_index: sorted masked patch indices
    """
    n = len(indices)
    n_visible = max(1, int(n * (1.0 - mask_ratio)))
    perm = np.random.permutation(n)
    v_index = np.sort(perm[:n_visible])
    m_index = np.sort(perm[n_visible:])
    return v_index, m_index


# ── Predictor block ───────────────────────────────────────────────────────────

class CrossAttnPredictorBlock(nn.Module):
    """Cross-attention block for predicting masked representations."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  (B, num_masked, dim)   — mask tokens
            kv: (B, num_visible, dim)  — context encoder output
        """
        q = q + self.cross_attn(self.norm1(q), kv, kv, need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


# ── Misc utilities ────────────────────────────────────────────────────────────

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extract off-diagonal elements from a square matrix."""
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='jepa', choices=['jepa', 'mae'])
    parser.add_argument('--masking', default='ssp', choices=['ssp', 'random'])
    parser.add_argument('--std_coeff', type=float, default=1.0)
    parser.add_argument('--cov_coeff', type=float, default=1.0)
    args = parser.parse_args()

    model = GLIMEncoderPretrainer(
        masking_strategy=args.masking,
        std_coeff=args.std_coeff,
        cov_coeff=args.cov_coeff,
        pretraining_mode=args.mode,
    )
    print(f"Mode: {args.mode} | Masking: {args.masking} | "
          f"std_coeff={args.std_coeff} | cov_coeff={args.cov_coeff}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Target encoder present: {model.target_encoder is not None}")
    print(f"MAE head present: {model.mae_head is not None}")

    x = torch.randn(4, 1280, 128)
    rep_tgt, rep_pred, rep_ctx, rep_tgt_full = model.pretrain_forward(x)
    print(f"rep_target shape:     {rep_tgt.shape}")
    print(f"rep_prediction shape: {rep_pred.shape}")

    loss, loss_dict = model.compute_loss(rep_tgt, rep_pred)
    print(f"Losses: {loss_dict}")

    gate_stats = model.get_encoder_gate_stats()
    print(f"Gate stats: {gate_stats}")
