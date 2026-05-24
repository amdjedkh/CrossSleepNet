"""
TF Image Encoder — CrossSleepNet v10
======================================
Encodes the log-amplitude STFT time-frequency image of a sleep epoch
into a fixed-size embedding vector.

Architecture
------------
  1. A three-layer 2-D CNN (Conv2d → BN → GELU) with frequency-axis kernels
     extracts local spectral features from the (3, T_freq, F_freq) STFT image.
  2. An AdaptiveAvgPool collapses the frequency axis to a fixed length.
  3. A [CLS] token is prepended and a Transformer encoder with pre-norm layers
     (norm_first=True) integrates temporal context across frequency bands.
  4. The [CLS] output is projected to a 2×d_tf embedding via an MLP.

The three input channels correspond to the log-amplitude STFTs of:
  channel 0 — EEG Fpz-Cz
  channel 1 — EEG Pz-Oz
  channel 2 — Horizontal EOG
"""

import torch
import torch.nn as nn


class TFImageEncoder(nn.Module):
    """STFT time-frequency image encoder using CNN + Transformer.

    Args:
        d_tf    (int): Internal and output embedding dimension.
        nhead   (int): Attention heads in the Transformer encoder.
        n_layers(int): Number of Transformer encoder layers.
        dropout (float): Dropout probability.

    Input:
        tf_img : (B, 3, T_time, F_freq) — log-amplitude STFT image
                 with shape (B, 3, 29, 128) for a 30-s epoch at 100 Hz.

    Output:
        (B, d_tf * 2) — epoch-level TF embedding (CLS token after projection).
    """

    def __init__(self, d_tf: int, nhead: int, n_layers: int, dropout: float):
        super().__init__()

        # Multi-scale 2-D CNN — kernel operates along frequency axis only
        self.conv = nn.Sequential(
            nn.Conv2d(3,    32,   (1, 7), padding=(0, 3)), nn.BatchNorm2d(32),   nn.GELU(),
            nn.Conv2d(32,   64,   (1, 5), padding=(0, 2)), nn.BatchNorm2d(64),   nn.GELU(),
            nn.Conv2d(64,   d_tf, (1, 3), padding=(0, 1)), nn.BatchNorm2d(d_tf), nn.GELU(),
            nn.AdaptiveAvgPool2d((29, 1)),   # collapse frequency axis → (B, d_tf, 29, 1)
        )

        # Learnable [CLS] token and positional embeddings (30 = 1 CLS + 29 time frames)
        self.cls = nn.Parameter(torch.randn(1, 1, d_tf) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 30, d_tf) * 0.02)

        # Transformer encoder with pre-norm (more stable for short sequences)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tf, nhead=nhead, dim_feedforward=d_tf * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_tf)

        # Project CLS output to 2×d_tf with LayerNorm
        self.proj = nn.Sequential(
            nn.Linear(d_tf, d_tf * 2),
            nn.LayerNorm(d_tf * 2),
        )

    def forward(self, tf_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf_img : (B, 3, T_time, F_freq) — normalised log-STFT image.
        Returns:
            (B, d_tf * 2) — TF epoch embedding.
        """
        # CNN feature extraction: (B, d_tf, 29, 1) → squeeze → (B, 29, d_tf)
        x = self.conv(tf_img).squeeze(-1).permute(0, 2, 1)

        # Prepend [CLS] token
        B = x.size(0)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)  # (B, 30, d_tf)
        x = x + self.pos

        # Transformer + CLS extraction
        return self.proj(self.norm(self.enc(x))[:, 0])           # (B, d_tf*2)
