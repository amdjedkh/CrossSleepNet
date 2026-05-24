"""
SequenceDataset — CrossSleepNet v10
=====================================
PyTorch Dataset that constructs sliding-window sequences of L consecutive
sleep epochs and returns them as float16-stored tensors (converted to float32
on access).

Design notes (v6)
-----------------
  - float16 tensor storage halves RAM usage relative to float32.
  - Conversion back to float32 happens lazily in __getitem__, not upfront.
  - This design prevented the inter-fold RAM crash that killed v5 when running
    10-fold CV on Sleep-EDF-78 with 78 subjects.
  - Per-epoch z-score normalisation is applied during construction, NOT in the
    model forward pass, reducing GPU workload.

Sequence construction
---------------------
For each subject with N valid epochs:
  - Iterate over centre indices from SEQ_LEN//2 to N-SEQ_LEN//2, step SEQ_STEP.
  - Extract a contiguous window [centre-SEQ_LEN//2, centre+SEQ_LEN//2).
  - Apply per-epoch per-channel z-score to EEG and EOG.
  - Store the label of the centre epoch as the sequence target.

Training-time augmentation
--------------------------
  If training=True, augment_epoch_2ch is called for each epoch in the sequence
  inside __getitem__ (after float16 → float32 conversion).
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from config import NUM_CLASSES, SEQ_LEN, SEQ_STEP
from data.preprocessing import augment_epoch_2ch


class SequenceDataset(Dataset):
    """Dataset of sliding-window epoch sequences for sleep staging.

    Args:
        subjects (list[dict]): Each element is a subject dict with keys:
            'id'    : str — subject identifier
            'eeg'   : np.ndarray (N, 2, 3000) float32
            'eog'   : np.ndarray (N, 3000)    float32
            'tf'    : np.ndarray (N, 3, 29, 128) float32
            'label' : np.ndarray (N,)          int64
        seq_len  (int):  Epochs per sequence window (default SEQ_LEN=20).
        training (bool): If True, apply data augmentation in __getitem__.
    """

    def __init__(self, subjects: List[dict], seq_len: int = SEQ_LEN, training: bool = False):
        self.training = training
        self.seq_len  = seq_len

        eeg_list, eog_list, tf_list, lab_list = [], [], [], []

        for subj in subjects:
            eeg = subj["eeg"]    # (N, 2, 3000)
            eog = subj["eog"]    # (N, 3000)
            tf  = subj["tf"]     # (N, 3, 29, 128)
            lab = subj["label"]  # (N,)
            N   = len(lab)
            if N < seq_len:
                continue

            half = seq_len // 2
            for centre in range(half, N - half, SEQ_STEP):
                start = centre - half
                end   = start + seq_len

                eeg_s = eeg[start:end].copy()    # (L, 2, 3000)
                eog_s = eog[start:end].copy()    # (L, 3000)
                tf_s  = tf[start:end].copy()     # (L, 3, 29, 128)

                # Per-epoch per-channel z-score for EEG
                mu_e  = eeg_s.mean(axis=2, keepdims=True)
                sg_e  = eeg_s.std(axis=2,  keepdims=True) + 1e-8
                eeg_s = ((eeg_s - mu_e) / sg_e).astype(np.float32)

                # Per-epoch z-score for EOG
                mu_g  = eog_s.mean(axis=1, keepdims=True)
                sg_g  = eog_s.std(axis=1,  keepdims=True) + 1e-8
                eog_s = ((eog_s - mu_g) / sg_g).astype(np.float32)

                eeg_list.append(eeg_s)
                eog_list.append(eog_s)
                tf_list.append(tf_s)
                lab_list.append(int(lab[centre]))

        # float16 storage: halves RAM usage (conversion to float32 is in __getitem__)
        self.eeg_t = torch.from_numpy(np.stack(eeg_list)).to(torch.float16)
        self.eog_t = (
            torch.from_numpy(np.stack(eog_list))
            .unsqueeze(2)             # (N, L, 1, 3000)
            .to(torch.float16)
        )
        self.tf_t  = torch.from_numpy(np.stack(tf_list)).to(torch.float16)
        self.lab_t = torch.tensor(lab_list, dtype=torch.long)

        n_gb = (self.eeg_t.nbytes + self.eog_t.nbytes + self.tf_t.nbytes) / 1e9
        print(f"    Dataset built: {len(self.lab_t):,} sequences  RAM≈{n_gb:.1f} GB")

    def __len__(self) -> int:
        return len(self.lab_t)

    def __getitem__(self, idx: int):
        """Return (eeg_seq, eog_seq, tf_seq, label) for one sequence.

        Returns:
            eeg_seq : (L, 2, 3000)   float32
            eog_seq : (L, 1, 3000)   float32
            tf_seq  : (L, 3, 29, 128) float32
            label   : scalar int64 — sleep stage of the centre epoch
        """
        eeg_s = self.eeg_t[idx].clone().float()    # (L, 2, 3000)
        eog_s = self.eog_t[idx].clone().float()    # (L, 1, 3000)
        tf_s  = self.tf_t[idx].clone().float()     # (L, 3, 29, 128)

        if self.training:
            for i in range(self.seq_len):
                e_arr = eeg_s[i].numpy()
                g_arr = eog_s[i, 0].numpy()
                e_arr, g_arr = augment_epoch_2ch(e_arr, g_arr, training=True)
                eeg_s[i]    = torch.from_numpy(e_arr)
                eog_s[i, 0] = torch.from_numpy(g_arr)

        return eeg_s, eog_s, tf_s, self.lab_t[idx]


def class_weights_focal(subjects: List[dict]) -> torch.Tensor:
    """Compute inverse-sqrt class weights with N1 up-weighting.

    N1 sleep (the most under-represented class in Sleep-EDF) receives
    an additional 2× multiplier to counteract heavy class imbalance.

    Args:
        subjects (list[dict]): Subject dicts with 'label' arrays.

    Returns:
        (NUM_CLASSES,) float32 tensor of class weights.
    """
    labels = np.concatenate([s["label"] for s in subjects])
    cnt    = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    cnt    = np.maximum(cnt, 1)
    w      = 1.0 / np.sqrt(cnt)
    w      = w / w.sum() * NUM_CLASSES
    w[1]  *= 2.0    # extra weight for N1 (class index 1)
    return torch.tensor(w, dtype=torch.float32)
