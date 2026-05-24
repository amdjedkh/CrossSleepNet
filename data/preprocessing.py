"""
Preprocessing Pipeline — CrossSleepNet v10
===========================================
Functions for loading, filtering, resampling, and segmenting Sleep-EDF EDF files
into per-epoch numpy arrays ready for SequenceDataset.

All Kaggle-specific paths have been replaced with function parameters.
No hardcoded constants: all values are imported from config.py.

Pipeline summary
----------------
  1. parse_hypnogram   — read MNE annotations → integer label array
  2. select_channels   — locate Fpz-Cz, Pz-Oz, and EOG channels by name
  3. bandpass          — zero-phase Butterworth bandpass filter
  4. safe_resample     — rational-factor resampling (preserves phase)
  5. compute_stft_epoch — log1p STFT amplitude for one channel × one epoch
  6. compute_stft_all  — STFT for all 3 channels → (3, T_time, F_freq) array
  7. augment_epoch_2ch — training-time noise + amplitude jitter
  8. process_subject   — full pipeline for one PSG/Hypnogram pair → arrays
  9. match_files       — pair PSG files with their Hypnogram files
"""

import glob
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, stft

from config import (
    CFG,
    EDF20_SUBJECTS,
    NUM_CLASSES,
    SAMPLES_EP,
    SLEEPEDF_STAGE_MAP,
    STAGE_NAMES,
    STFT_N_FREQ,
    STFT_N_TIME,
    STFT_NFFT,
    STFT_NOVERLAP,
    STFT_NPERSEG,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# ── Annotation parsing ─────────────────────────────────────────────────────────

def parse_hypnogram(hyp_path: str) -> np.ndarray:
    """Parse a Sleep-EDF hypnogram EDF file into an integer label array.

    Each annotation with duration D is expanded to round(D/30) epochs,
    allowing for variable-duration annotations in older files.

    Args:
        hyp_path (str): Path to the *-Hypnogram.edf file.

    Returns:
        np.ndarray of shape (N_epochs,) with dtype int64.
        Unknown/movement epochs are labelled -1.
    """
    ann = mne.read_annotations(hyp_path)
    labels = []
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        stage    = SLEEPEDF_STAGE_MAP.get(desc, -1)
        n_epochs = max(1, int(round(dur / 30)))
        labels.extend([stage] * n_epochs)
    return np.array(labels, dtype=np.int64)


# ── Channel selection ──────────────────────────────────────────────────────────

def select_channels(
    ch_names: List[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Locate the indices of Fpz-Cz, Pz-Oz, and horizontal EOG channels.

    Uses case-insensitive substring matching. All three channels are required
    for CrossSleepNet; returns None for any that are missing.

    Args:
        ch_names (list[str]): Channel names from mne.io.read_raw_edf.

    Returns:
        (fpz_idx, pzoz_idx, eog_idx) — integer indices or None if not found.
    """
    ch_lower = [c.lower().strip() for c in ch_names]
    fpz_idx = pzoz_idx = eog_idx = None
    for i, ch in enumerate(ch_lower):
        if "fpz" in ch:
            fpz_idx = i
        elif "pz" in ch and "oz" in ch:
            pzoz_idx = i
        elif "eog" in ch or "horizontal" in ch:
            eog_idx = i
    return fpz_idx, pzoz_idx, eog_idx


# ── Signal filtering and resampling ───────────────────────────────────────────

def bandpass(
    signal: np.ndarray,
    lo: float,
    hi: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        signal (np.ndarray): 1-D or N-D signal; filtered along last axis.
        lo     (float): Low-cut frequency in Hz.
        hi     (float): High-cut frequency in Hz.
        fs     (float): Sampling frequency in Hz.
        order  (int):   Filter order (default 4).

    Returns:
        Filtered signal with the same shape as input.
    """
    nyq  = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def safe_resample(
    signal: np.ndarray,
    fs_orig: float,
    fs_target: int,
) -> np.ndarray:
    """Rational-factor polyphase resampling using GCD reduction.

    Avoids floating-point rounding by computing the exact up/down factors
    from the greatest common divisor.

    Args:
        signal    (np.ndarray): Input signal; resampled along last axis.
        fs_orig   (float): Original sampling frequency in Hz.
        fs_target (int):   Target sampling frequency in Hz.

    Returns:
        Resampled signal.
    """
    fs_orig = int(round(fs_orig))
    g       = gcd(fs_orig, fs_target)
    return resample_poly(signal, fs_target // g, fs_orig // g, axis=-1)


# ── STFT computation ──────────────────────────────────────────────────────────

def compute_stft_epoch(epoch_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """Compute log1p-amplitude STFT for one epoch of one channel.

    The output is z-score normalised per epoch, matching the preprocessing
    used in CrossFusionSleepNet (BSPC 2025) exactly.

    Args:
        epoch_signal (np.ndarray): 1-D array of length SAMPLES_EP (3000).
        fs           (int):        Sampling frequency in Hz (default 100).

    Returns:
        np.ndarray of shape (STFT_N_TIME, STFT_N_FREQ) = (29, 128), float32.
    """
    _, _, Zxx = stft(
        epoch_signal,
        fs=fs,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
    )
    amp     = np.abs(Zxx)[:STFT_N_FREQ, :STFT_N_TIME]
    log_amp = np.log1p(amp).T.astype(np.float32)      # (T_time, F_freq)
    mu      = log_amp.mean()
    sig     = log_amp.std() + 1e-8
    return ((log_amp - mu) / sig).astype(np.float32)


def compute_stft_all(
    fpz: np.ndarray,
    pz: np.ndarray,
    eog: np.ndarray,
    fs: int = 100,
) -> np.ndarray:
    """Compute STFT for all three channels and stack along channel axis.

    Args:
        fpz (np.ndarray): EEG Fpz-Cz epoch, shape (SAMPLES_EP,).
        pz  (np.ndarray): EEG Pz-Oz epoch,  shape (SAMPLES_EP,).
        eog (np.ndarray): EOG epoch,          shape (SAMPLES_EP,).
        fs  (int):        Sampling frequency in Hz (default 100).

    Returns:
        np.ndarray of shape (3, STFT_N_TIME, STFT_N_FREQ) = (3, 29, 128), float32.
    """
    return np.stack([
        compute_stft_epoch(fpz, fs),
        compute_stft_epoch(pz,  fs),
        compute_stft_epoch(eog, fs),
    ], axis=0)


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_epoch_2ch(
    eeg_ep: np.ndarray,
    eog_ep: np.ndarray,
    training: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply training-time Gaussian noise and random amplitude scaling.

    Augmentation parameters are read from CFG:
      aug_noise_std  — noise amplitude as a fraction of the signal std
      aug_amp_range  — (lo, hi) bounds for uniform amplitude multiplier

    Args:
        eeg_ep   (np.ndarray): EEG epoch, shape (2, SAMPLES_EP), float32.
        eog_ep   (np.ndarray): EOG epoch, shape (SAMPLES_EP,),   float32.
        training (bool):       If False, returns inputs unmodified.

    Returns:
        (eeg_ep_aug, eog_ep_aug) — augmented float32 arrays.
    """
    if not training:
        return eeg_ep, eog_ep

    noise_std        = CFG["aug_noise_std"]
    amp_lo, amp_hi   = CFG["aug_amp_range"]

    for c in range(eeg_ep.shape[0]):
        eeg_ep[c] = (
            eeg_ep[c]
            + np.random.randn(*eeg_ep[c].shape).astype(np.float32)
            * (eeg_ep[c].std() * noise_std)
        )
        eeg_ep[c] *= np.random.uniform(amp_lo, amp_hi)

    eog_ep = (
        eog_ep
        + np.random.randn(*eog_ep.shape).astype(np.float32)
        * (eog_ep.std() * noise_std)
    )
    eog_ep *= np.random.uniform(amp_lo, amp_hi)

    return eeg_ep.astype(np.float32), eog_ep.astype(np.float32)


# ── Full subject pipeline ──────────────────────────────────────────────────────

def process_subject(
    psg_path: str,
    hyp_path: str,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Full preprocessing pipeline for one PSG/Hypnogram pair.

    Steps:
      1. Read EDF, extract Fpz-Cz, Pz-Oz, and EOG channels.
      2. Resample to CFG['fs_target'] if necessary.
      3. Bandpass filter each channel.
      4. Parse hypnogram, crop to ±wake_pad_epochs around the sleep period.
      5. Segment into 30-second epochs, compute per-epoch z-score + STFT.
      6. Reject epochs with NaN values or near-zero variance.

    Wake cropping (v7): retains 60 epochs (30 minutes) of Wake before and after
    the sleep onset/offset, matching the distribution reported in CrossFusionSleepNet.

    Args:
        psg_path (str): Path to the *-PSG.edf file.
        hyp_path (str): Path to the corresponding *-Hypnogram.edf file.

    Returns:
        (eeg, eog, tf, labels) — float32/int64 arrays of shape:
          eeg    : (N, 2, 3000)
          eog    : (N, 3000)
          tf     : (N, 3, 29, 128)
          labels : (N,)
        Returns (None, None, None, None) on failure.
    """
    fs_t    = CFG["fs_target"]
    samp_ep = SAMPLES_EP
    pad     = CFG["wake_pad_epochs"]

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    except Exception:
        return None, None, None, None

    fs_orig                     = raw.info["sfreq"]
    fpz_idx, pzoz_idx, eog_idx = select_channels(raw.ch_names)
    if fpz_idx is None or pzoz_idx is None or eog_idx is None:
        del raw
        return None, None, None, None

    eeg_fpz = raw.get_data(picks=[fpz_idx])[0]
    eeg_pz  = raw.get_data(picks=[pzoz_idx])[0]
    eog_raw = raw.get_data(picks=[eog_idx])[0]
    del raw

    # Resample if needed
    if abs(fs_orig - fs_t) > 0.5:
        eeg_fpz = safe_resample(eeg_fpz, fs_orig, fs_t)
        eeg_pz  = safe_resample(eeg_pz,  fs_orig, fs_t)
        eog_raw = safe_resample(eog_raw, fs_orig, fs_t)

    # Bandpass filter
    lo_eeg, hi_eeg = CFG.get("eeg_band", (0.5, 45.0))
    lo_eog, hi_eog = CFG.get("eog_band", (0.3, 35.0))
    try:    fpz_f = bandpass(eeg_fpz, 0.5, 45.0, fs_t)
    except: fpz_f = eeg_fpz
    try:    pz_f  = bandpass(eeg_pz,  0.5, 45.0, fs_t)
    except: pz_f  = eeg_pz
    try:    eog_f = bandpass(eog_raw, 0.3, 35.0, fs_t)
    except: eog_f = eog_raw

    # Parse labels and crop to sleep-centred window
    labels = parse_hypnogram(hyp_path)
    if len(labels) == 0:
        return None, None, None, None

    non_wake = np.where((labels != 0) & (labels != -1))[0]
    if len(non_wake) == 0:
        return None, None, None, None

    crop_start = max(0, non_wake[0] - pad)
    crop_end   = min(len(labels), non_wake[-1] + pad + 1)
    labels     = labels[crop_start:crop_end]
    max_signal = len(fpz_f) // samp_ep

    eeg_eps, eog_eps, tf_eps, valid_labels = [], [], [], []
    for idx in range(len(labels)):
        sig_idx = crop_start + idx
        if sig_idx >= max_signal:
            break
        lbl = labels[idx]
        if lbl == -1:
            continue
        s = sig_idx * samp_ep
        e = s + samp_ep
        ef = fpz_f[s:e].astype(np.float32)
        ep = pz_f[s:e].astype(np.float32)
        og = eog_f[s:e].astype(np.float32)
        if len(ef) < samp_ep or len(ep) < samp_ep or len(og) < samp_ep:
            continue
        if np.isnan(ef).any() or np.isnan(ep).any() or np.isnan(og).any():
            continue
        if ef.std() < 1e-8 or ep.std() < 1e-8 or og.std() < 1e-8:
            continue
        eeg_eps.append(np.stack([ef, ep], axis=0))
        eog_eps.append(og)
        tf_eps.append(compute_stft_all(ef, ep, og))
        valid_labels.append(lbl)

    if len(valid_labels) < 10:
        return None, None, None, None

    return (
        np.array(eeg_eps,      dtype=np.float32),
        np.array(eog_eps,      dtype=np.float32),
        np.array(tf_eps,       dtype=np.float32),
        np.array(valid_labels, dtype=np.int64),
    )


# ── File matching ──────────────────────────────────────────────────────────────

def match_files(
    psg_list: List[str],
    hyp_list: List[str],
    dataset_mode: str = "edf78",
) -> List[Tuple[str, str]]:
    """Pair PSG files with their corresponding Hypnogram files.

    Matching is performed by the first 6 characters of the filename (subject ID).

    Args:
        psg_list     (list[str]): Paths to *PSG.edf files.
        hyp_list     (list[str]): Paths to *Hypnogram.edf files.
        dataset_mode (str):       "edf78" uses all subjects;
                                  "edf20" restricts to the 20 SC40xx subjects
                                  and excludes second recordings (index [6]=='1').

    Returns:
        List of (psg_path, hyp_path) tuples.
    """
    hyp_map = {}
    for h in hyp_list:
        key = Path(h).name[:6]
        hyp_map[key] = h

    pairs = []
    for psg in psg_list:
        key = Path(psg).name[:6]
        if key not in hyp_map:
            continue
        if dataset_mode == "edf20":
            if key not in EDF20_SUBJECTS:
                continue
            if Path(psg).name[6] == "1":
                continue
        pairs.append((psg, hyp_map[key]))

    return pairs
