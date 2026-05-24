"""
CrossSleepNet v10 — Full Model and Ablations
=============================================
Four-level architecture for automated sleep staging from EEG + EOG signals.

Architecture overview
----------------------

  ┌──────────────────────────────────────────────────────────────────────┐
  │  INPUT: sequence of L=20 consecutive 30-second sleep epochs          │
  │                                                                       │
  │  Per epoch:                                                           │
  │    EEG  (2, 3000)  ─── Level 1a ──→  ConformerEncoder ──→ He         │
  │    EOG  (1, 3000)  ─── Level 1b ──→  ConformerEncoder ──→ Hg         │
  │    STFT (3,29,128) ─── Level 1c ──→  TFImageEncoder  ──→ tf_emb      │
  │                                                                       │
  │    Level 2: Bidirectional EEG-EOG Cross-Attention                    │
  │             He, Hg = CrossAttn(He, Hg)                               │
  │                                                                       │
  │    Level 3: MLP fusion                                                │
  │             epoch_emb = MLP([mean(He); mean(Hg); tf_emb])            │
  │                                                                       │
  │  Level 4: Sequence Transformer over L epoch embeddings               │
  │           logits = SeqTransformer(epoch_emb_1, ..., epoch_emb_L)     │
  │           → classification of centre epoch (epoch L//2)              │
  └──────────────────────────────────────────────────────────────────────┘

Ablation variants
-----------------
  CrossSleepNetV10_NoCross : removes Level 2 (no EEG-EOG cross-attention)
  CrossSleepNetV10_NoTF    : removes Level 1c (zeroed TF branch)
  SeqTransEEGBaseline      : EEG-only, no EOG, no STFT — Conformer + SeqTransformer
"""

import torch
import torch.nn as nn

from config import CFG, SEQ_LEN, NUM_CLASSES, EEG_IN_CH, EOG_IN_CH
from models.conformer import ConformerEncoder
from models.cross_attention import EEGEOGCrossAttention
from models.tf_encoder import TFImageEncoder


# ── Sequence Transformer ──────────────────────────────────────────────────────

class SeqTransformer(nn.Module):
    """Sequence-level Transformer operating over L epoch embeddings.

    Classifies the centre epoch (index seq_len // 2) using bidirectional
    context from all L epochs in the sequence.

    Args:
        d_in      (int): Dimension of input epoch embeddings.
        d_seq     (int): Internal dimension of the sequence Transformer.
        nhead     (int): Attention heads.
        n_layers  (int): Transformer encoder layers.
        seq_len   (int): Number of epochs per sequence (window length).
        dropout  (float): Dropout probability.
        num_classes(int): Number of output sleep stage classes.
    """

    def __init__(self, d_in: int, d_seq: int, nhead: int, n_layers: int,
                 seq_len: int, dropout: float, num_classes: int):
        super().__init__()
        self.centre_idx = seq_len // 2
        self.input_proj = nn.Sequential(nn.Linear(d_in, d_seq), nn.LayerNorm(d_seq))
        self.pos_emb    = nn.Parameter(torch.randn(1, seq_len, d_seq) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_seq, nhead=nhead, dim_feedforward=d_seq * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm       = nn.LayerNorm(d_seq)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_seq, d_seq // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_seq // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L, d_in) — sequence of L epoch embeddings.
        Returns:
            (B, num_classes) — logits for the centre epoch.
        """
        x = self.input_proj(x) + self.pos_emb
        x = self.norm(self.encoder(x))
        return self.classifier(x[:, self.centre_idx])


# ── Full CrossSleepNet v10 ─────────────────────────────────────────────────────

class CrossSleepNetV10(nn.Module):
    """CrossSleepNet v10: Conformer EEG-EOG cross-attention + STFT + SeqTransformer.

    The full model with all four levels active. See module docstring for the
    complete architecture diagram.
    """

    def __init__(self):
        super().__init__()
        d   = CFG["d_model"]
        nh  = CFG["nhead"]
        nl  = CFG["n_layers"]
        ck  = CFG["conv_kernel"]
        pl  = CFG["patch_len"]
        do  = CFG["dropout"]
        dtf = CFG["d_tf"]
        nhtf= CFG["nhead_tf"]
        nltf= CFG["n_layers_tf"]
        ds  = CFG["d_seq"]
        nhs = CFG["nhead_seq"]
        nls = CFG["n_layers_seq"]
        dos = CFG["dropout_seq"]

        # Level 1a/1b — Conformer encoders for EEG and EOG
        self.eeg_enc   = ConformerEncoder(EEG_IN_CH, d, nl, nh, ck, pl, do)
        self.eog_enc   = ConformerEncoder(EOG_IN_CH, d, nl, nh, ck, pl, do)

        # Level 2 — Bidirectional EEG-EOG cross-attention
        self.cross_attn = EEGEOGCrossAttention(d, nh, do)

        # Level 1c — STFT TF image encoder
        self.tf_enc = TFImageEncoder(dtf, nhtf, nltf, do)

        # Level 3 — MLP fusion: [EEG_mean; EOG_mean; TF] → epoch embedding
        d_sig = d * 2            # EEG mean-pool (d) + EOG mean-pool (d)
        d_in_fusion = d_sig + dtf * 2
        self.fusion = nn.Sequential(
            nn.Linear(d_in_fusion, ds * 2), nn.LayerNorm(ds * 2), nn.GELU(),
            nn.Dropout(dos),
            nn.Linear(ds * 2, ds), nn.LayerNorm(ds), nn.GELU(),
        )

        # Level 4 — Sequence Transformer
        self.seq_enc = SeqTransformer(ds, ds, nhs, nls, SEQ_LEN, dos, NUM_CLASSES)

    def forward(
        self,
        eeg_seq: torch.Tensor,
        eog_seq: torch.Tensor,
        tf_seq:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_seq : (B, L, 2, 3000) — EEG sequences (Fpz-Cz, Pz-Oz).
            eog_seq : (B, L, 1, 3000) — EOG sequences.
            tf_seq  : (B, L, 3, T_t, F_t) — STFT images.
        Returns:
            (B, num_classes) — logits for the centre epoch.
        """
        B, L, Ce, T = eeg_seq.shape
        _, _, Cg, _ = eog_seq.shape
        _, _, Ct, Tt, Ft = tf_seq.shape

        eeg_flat = eeg_seq.reshape(B * L, Ce, T)
        eog_flat = eog_seq.reshape(B * L, Cg, T)
        tf_flat  = tf_seq.reshape(B * L, Ct, Tt, Ft)

        # Conformer encoding
        He = self.eeg_enc(eeg_flat)   # (B*L, P, d)
        Hg = self.eog_enc(eog_flat)   # (B*L, P, d)

        # Bidirectional cross-attention
        He, Hg = self.cross_attn(He, Hg)

        # Mean-pool patch sequences
        eeg_emb = He.mean(1)                                   # (B*L, d)
        eog_emb = Hg.mean(1)                                   # (B*L, d)
        sig_emb = torch.cat([eeg_emb, eog_emb], dim=-1)       # (B*L, d*2)

        # TF encoding
        tf_emb = self.tf_enc(tf_flat)                          # (B*L, dtf*2)

        # MLP fusion → epoch embedding
        ep_emb = self.fusion(torch.cat([sig_emb, tf_emb], dim=-1))   # (B*L, ds)
        ep_emb = ep_emb.view(B, L, -1)                                # (B, L, ds)

        # Sequence Transformer → classification
        return self.seq_enc(ep_emb)


# ── Ablation 1: No EEG-EOG Cross-Attention ────────────────────────────────────

class CrossSleepNetV10_NoCross(CrossSleepNetV10):
    """Ablation: remove bidirectional EEG-EOG cross-attention (Level 2).

    EEG and EOG patch sequences are mean-pooled independently and concatenated
    without any cross-modal interaction. Used to quantify the contribution of
    the cross-attention module to the full model's performance.
    """

    def forward(
        self,
        eeg_seq: torch.Tensor,
        eog_seq: torch.Tensor,
        tf_seq:  torch.Tensor,
    ) -> torch.Tensor:
        B, L, Ce, T = eeg_seq.shape
        _, _, Cg, _ = eog_seq.shape
        _, _, Ct, Tt, Ft = tf_seq.shape

        eeg_flat = eeg_seq.reshape(B * L, Ce, T)
        eog_flat = eog_seq.reshape(B * L, Cg, T)
        tf_flat  = tf_seq.reshape(B * L, Ct, Tt, Ft)

        # Conformer encoding WITHOUT cross-attention
        He = self.eeg_enc(eeg_flat)
        Hg = self.eog_enc(eog_flat)

        # Direct mean-pool — no cross-attention
        eeg_emb = He.mean(1)
        eog_emb = Hg.mean(1)
        sig_emb = torch.cat([eeg_emb, eog_emb], dim=-1)

        tf_emb  = self.tf_enc(tf_flat)
        ep_emb  = self.fusion(torch.cat([sig_emb, tf_emb], dim=-1))
        ep_emb  = ep_emb.view(B, L, -1)

        return self.seq_enc(ep_emb)


# ── Ablation 2: No STFT Branch ────────────────────────────────────────────────

class CrossSleepNetV10_NoTF(CrossSleepNetV10):
    """Ablation: remove STFT time-frequency branch (Level 1c).

    The TF embedding is replaced with a zero vector of matching dimension.
    Isolates the contribution of spectral features from the raw signal path.
    """

    def forward(
        self,
        eeg_seq: torch.Tensor,
        eog_seq: torch.Tensor,
        tf_seq:  torch.Tensor,
    ) -> torch.Tensor:
        B, L, Ce, T = eeg_seq.shape
        _, _, Cg, _ = eog_seq.shape
        _, _, Ct, Tt, Ft = tf_seq.shape

        eeg_flat = eeg_seq.reshape(B * L, Ce, T)
        eog_flat = eog_seq.reshape(B * L, Cg, T)
        tf_flat  = tf_seq.reshape(B * L, Ct, Tt, Ft)

        He = self.eeg_enc(eeg_flat)
        Hg = self.eog_enc(eog_flat)
        He, Hg = self.cross_attn(He, Hg)

        eeg_emb = He.mean(1)
        eog_emb = Hg.mean(1)
        sig_emb = torch.cat([eeg_emb, eog_emb], dim=-1)

        # Zero TF embedding — isolates STFT contribution
        tf_emb = torch.zeros(
            B * L, self.tf_enc.proj[0].out_features,
            device=sig_emb.device,
        )
        ep_emb = self.fusion(torch.cat([sig_emb, tf_emb], dim=-1))
        ep_emb = ep_emb.view(B, L, -1)

        return self.seq_enc(ep_emb)


# ── Ablation 3: EEG-Only Baseline ────────────────────────────────────────────

class SeqTransEEGBaseline(nn.Module):
    """Baseline: single-modality EEG only, no EOG, no STFT.

    Minimal architecture: ConformerEncoder + mean-pool + SeqTransformer.
    Establishes the lower-bound performance from EEG signal alone.
    """

    def __init__(self):
        super().__init__()
        d   = CFG["d_model"]
        nh  = CFG["nhead"]
        nl  = CFG["n_layers"]
        ck  = CFG["conv_kernel"]
        pl  = CFG["patch_len"]
        do  = CFG["dropout"]
        ds  = CFG["d_seq"]
        nhs = CFG["nhead_seq"]
        nls = CFG["n_layers_seq"]
        dos = CFG["dropout_seq"]

        self.eeg_enc = ConformerEncoder(EEG_IN_CH, d, nl, nh, ck, pl, do)
        self.proj    = nn.Sequential(
            nn.Linear(d, ds), nn.LayerNorm(ds), nn.GELU()
        )
        self.seq_enc = SeqTransformer(ds, ds, nhs, nls, SEQ_LEN, dos, NUM_CLASSES)

    def forward(
        self,
        eeg_seq: torch.Tensor,
        eog_seq: torch.Tensor,
        tf_seq:  torch.Tensor,
    ) -> torch.Tensor:
        """EOG and TF inputs are accepted but ignored (interface compatibility)."""
        B, L, Ce, T = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * L, Ce, T)
        He       = self.eeg_enc(eeg_flat)
        ep_emb   = self.proj(He.mean(1)).view(B, L, -1)
        return self.seq_enc(ep_emb)


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(name: str) -> nn.Module:
    """Instantiate a model by name string.

    Args:
        name (str): One of 'CrossSleepNetV10', 'CrossSleepNetV10_NoCross',
                    'CrossSleepNetV10_NoTF', 'SeqTrans-EEG'.
    Returns:
        Instantiated nn.Module.
    Raises:
        ValueError: If name is not recognised.
    """
    if name == "CrossSleepNetV10":         return CrossSleepNetV10()
    if name == "CrossSleepNetV10_NoCross": return CrossSleepNetV10_NoCross()
    if name == "CrossSleepNetV10_NoTF":    return CrossSleepNetV10_NoTF()
    if name == "SeqTrans-EEG":             return SeqTransEEGBaseline()
    raise ValueError(f"Unknown model name: '{name}'")


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
