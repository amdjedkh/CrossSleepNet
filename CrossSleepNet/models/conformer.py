"""
Conformer Encoder — CrossSleepNet v10
======================================
Implements the ConformerBlock and ConformerEncoder used as the per-epoch
signal encoder for both EEG and EOG streams.

The Conformer architecture (Gulati et al., 2020) combines:
  - Depthwise separable convolution (local temporal patterns)
  - Multi-head self-attention (global sequence dependencies)
in a Macaron-style Feed-Forward → Attention → Conv → Feed-Forward ordering
that consistently outperforms standard Transformers on temporal biomedical
signals.

Each encoder tokenises a raw 3000-sample epoch into patches and produces a
sequence of contextualised patch embeddings (B, n_patches, d_model).
"""

import torch
import torch.nn as nn

from config import CFG, SAMPLES_EP


class ConformerBlock(nn.Module):
    """Single Conformer block: Feed-Forward → Self-Attention → Conv → Feed-Forward.

    This Macaron-style ordering (FF half-step, MHSA, depthwise conv, FF half-step)
    is from the original Conformer paper (Gulati 2020) and consistently
    outperforms standard Transformer blocks on temporal signals such as EEG.

    Args:
        d_model    (int): Token embedding dimension.
        nhead      (int): Number of attention heads.
        conv_kernel(int): Depthwise conv kernel size. Should be odd.
                          A kernel of 31 at 100 Hz covers ~300 ms patterns.
        dropout   (float): Dropout probability applied throughout.
    """

    def __init__(self, d_model: int, nhead: int, conv_kernel: int, dropout: float):
        super().__init__()

        # Feed-forward 1 (half-step residual weight = 0.5)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        # Depthwise Conv module
        self.conv_norm   = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),                          # pointwise expand
            nn.GLU(dim=1),                                                 # gating
            nn.Conv1d(d_model, d_model, conv_kernel,                      # depthwise conv
                      padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),                              # pointwise project
            nn.Dropout(dropout),
        )

        # Feed-forward 2 (half-step residual weight = 0.5)
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — sequence of token embeddings.
        Returns:
            (B, T, d_model) — contextualised token embeddings.
        """
        # FF1 half-step
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # Multi-head self-attention
        xn = self.attn_norm(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + self.attn_drop(attn_out)

        # Depthwise conv
        xn = self.conv_norm(x).transpose(1, 2)       # (B, d, T)
        x  = x + self.conv_module(xn).transpose(1, 2)

        # FF2 half-step
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of ConformerBlocks with multi-scale patch tokenisation.

    Replaces the pure TransformerEncoder from CrossSleepNet v1-v9.

    Tokenisation pipeline:
      1. Two parallel CNN branches (kernel=7 and kernel=25) extract short-range
         and long-range temporal features from the raw epoch signal.
      2. Outputs are concatenated along the channel dimension to form d_model
         features per time step.
      3. Contiguous non-overlapping patches of length `patch_len` are flattened
         and projected to d_model via a linear layer.
      4. Learned positional embeddings are added.
      5. The resulting patch sequence is passed through `n_layers` ConformerBlocks.

    Args:
        in_ch      (int): Number of input channels (e.g. 2 for EEG, 1 for EOG).
        d_model    (int): Embedding dimension throughout the encoder.
        n_layers   (int): Number of ConformerBlocks.
        nhead      (int): Attention heads per ConformerBlock.
        conv_kernel(int): Depthwise conv kernel size in each ConformerBlock.
        patch_len  (int): Samples per patch for tokenisation.
        dropout   (float): Dropout probability.
    """

    def __init__(self, in_ch: int, d_model: int, n_layers: int, nhead: int,
                 conv_kernel: int, patch_len: int, dropout: float):
        super().__init__()
        mid = d_model // 2
        self.patch_len = patch_len

        # Multi-scale CNN tokeniser
        self.conv_short = nn.Sequential(
            nn.Conv1d(in_ch, mid,  7, padding=3),  nn.BatchNorm1d(mid), nn.GELU())
        self.conv_long  = nn.Sequential(
            nn.Conv1d(in_ch, mid, 25, padding=12), nn.BatchNorm1d(mid), nn.GELU())

        self.patch_proj = nn.Linear(d_model * patch_len, d_model)
        self.proj_norm  = nn.LayerNorm(d_model)

        n_patches    = SAMPLES_EP // patch_len
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, 3000) — raw epoch signal(s).
        Returns:
            (B, n_patches, d_model) — contextualised patch embeddings.
        """
        # Multi-scale feature extraction
        f = torch.cat([self.conv_short(x), self.conv_long(x)], dim=1)   # (B, d_model, T)
        B, D, T = f.shape
        P = T // self.patch_len
        f = f[:, :, :P * self.patch_len]

        # Reshape into patches
        f = f.reshape(B, D, P, self.patch_len).permute(0, 2, 1, 3)      # (B, P, D, pl)
        f = f.reshape(B, P, D * self.patch_len)                          # (B, P, D*pl)

        # Project patches + positional embedding
        h = self.proj_norm(self.patch_proj(f)) + self.pos_emb            # (B, P, d_model)

        # Conformer blocks
        for block in self.blocks:
            h = block(h)

        return h   # (B, P, d_model)
