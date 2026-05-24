"""
EEG-EOG Bidirectional Cross-Attention — CrossSleepNet v10
===========================================================
Our core architectural contribution: bidirectional cross-attention between
the Conformer-encoded EEG patch sequence and the Conformer-encoded EOG patch
sequence.

Physiological motivation
------------------------
EEG and EOG signals are not independent during sleep:
  - During REM sleep, rapid eye movements (EOG bursts) co-occur with
    sawtooth EEG waves and ponto-geniculo-occipital (PGO) spikes.
  - K-complexes and sleep spindles (N2) are preceded and followed by
    correlated EOG micro-movements.
  - Wake episodes show alpha EEG suppression synchronised with gaze shifts
    detected by EOG.

Standard fusion approaches (concatenation, summation) ignore these
interdependencies. Cross-attention allows each modality to selectively
query the other, learning which EOG features are diagnostic given the
current EEG context, and vice versa.

Design
------
  He_out = LayerNorm(He + CrossAttn(Q=He, K=Hg, V=Hg))   # EEG attends to EOG
  Hg_out = LayerNorm(Hg + CrossAttn(Q=Hg, K=He, V=He))   # EOG attends to EEG

Both streams are updated in parallel (not sequentially), preserving symmetry.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EEGEOGCrossAttention(nn.Module):
    """Bidirectional cross-attention between Conformer-encoded EEG and EOG.

    EEG patch embeddings attend to EOG patch embeddings, and EOG attends
    to EEG — simultaneously. Both residual streams are updated.

    This layer captures the physiological neural-ocular coupling that
    distinguishes sleep stages (particularly REM vs. N2 vs. Wake).

    Args:
        d_model (int): Embedding dimension (must match ConformerEncoder output).
        nhead   (int): Number of attention heads.
        dropout (float): Dropout on attention weights.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()

        # EEG attends to EOG: Q=EEG, K/V=EOG
        self.cross_eeg = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        # EOG attends to EEG: Q=EOG, K/V=EEG
        self.cross_eog = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_eog = nn.LayerNorm(d_model)

        # Last attention weight tensors (detached CPU), for interpretability
        self.attn_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self,
        He: torch.Tensor,
        Hg: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            He         : (B, P, d_model) — EEG patch embeddings from ConformerEncoder.
            Hg         : (B, P, d_model) — EOG patch embeddings from ConformerEncoder.
            return_attn: If True, stores attention weight matrices in
                         self.attn_weights = (ae, ag) for post-hoc analysis.
        Returns:
            He_out : (B, P, d_model) — EEG embeddings enriched with EOG context.
            Hg_out : (B, P, d_model) — EOG embeddings enriched with EEG context.
        """
        He_x, ae = self.cross_eeg(
            He, Hg, Hg,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        Hg_x, ag = self.cross_eog(
            Hg, He, He,
            need_weights=return_attn,
            average_attn_weights=True,
        )

        He = self.norm_eeg(He + He_x)
        Hg = self.norm_eog(Hg + Hg_x)

        if return_attn:
            self.attn_weights = (
                ae.detach().cpu() if ae is not None else None,
                ag.detach().cpu() if ag is not None else None,
            )

        return He, Hg
