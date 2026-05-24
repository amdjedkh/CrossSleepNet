"""
CrossSleepNet v10 — Central Configuration
==========================================
Single source of truth for all hyperparameters, constants, and paths.
All other modules import from here; no hardcoded values elsewhere.
"""

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 55  # selected via fold analysis for balanced cross-validation assignments

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_MODE   = "edf78"           # "edf78" or "edf20"
MAX_SUBJECTS   = 78
N_FOLDS        = 10
NUM_CLASSES    = 5
STAGE_NAMES    = ["Wake", "N1", "N2", "N3", "REM"]

EDF20_SUBJECTS = {f"SC40{i:02d}" for i in range(1, 21)}

# Sleep-EDF annotation → integer label mapping
SLEEPEDF_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N3 and N4 merged
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

# ── Signal processing ──────────────────────────────────────────────────────────
FS_TARGET    = 100       # Hz — target sampling rate after resampling
EPOCH_SEC    = 30        # seconds per sleep epoch
SAMPLES_EP   = FS_TARGET * EPOCH_SEC   # 3000 samples per epoch

EEG_IN_CH    = 2         # Fpz-Cz and Pz-Oz channels
EOG_IN_CH    = 1         # Horizontal EOG channel

# Bandpass filter cutoffs
EEG_BAND = (0.5, 45.0)   # Hz
EOG_BAND = (0.3, 35.0)   # Hz

# Wake-period cropping: keep N epochs before/after sleep onset (30 minutes @ 60 ep)
WAKE_PAD_EPOCHS = 60

# ── STFT parameters ────────────────────────────────────────────────────────────
STFT_NPERSEG  = 200
STFT_NOVERLAP = 100
STFT_NFFT     = 256
STFT_N_FREQ   = 128      # frequency bins retained
STFT_N_TIME   = 29       # time frames per epoch
STFT_N_CH     = 3        # channels: Fpz, Pz-Oz, EOG

# ── Sequence construction ──────────────────────────────────────────────────────
SEQ_LEN  = 20    # consecutive epochs per training sequence
SEQ_STEP = 25    # stride between sequence centres (epochs)

# ── Model architecture ─────────────────────────────────────────────────────────
CFG = {
    # Shared signal parameters
    "fs_target"      : FS_TARGET,
    "epoch_sec"      : EPOCH_SEC,

    # Conformer encoder
    "d_model"        : 128,    # embedding dimension for EEG/EOG Conformer
    "nhead"          : 4,      # attention heads in Conformer blocks
    "n_layers"       : 4,      # number of Conformer blocks per encoder
    "conv_kernel"    : 31,     # depthwise conv kernel — captures ~300 ms patterns
    "patch_len"      : 100,    # samples per patch for tokenisation
    "dropout"        : 0.1,

    # TF image encoder
    "d_tf"           : 128,    # embedding dimension for STFT branch
    "nhead_tf"       : 4,
    "n_layers_tf"    : 4,

    # Sequence Transformer
    "d_seq"          : 256,    # embedding dimension for sequence-level model
    "nhead_seq"      : 8,
    "n_layers_seq"   : 4,
    "dropout_seq"    : 0.1,

    # Optimiser
    "lr_trans"       : 1e-4,   # peak learning rate (after warmup)
    "wd"             : 1e-2,   # AdamW weight decay
    "grad_clip"      : 1.0,    # gradient norm clipping threshold
    "label_smooth"   : 0.05,   # label smoothing coefficient

    # Data augmentation (applied during training only)
    "aug_noise_std"  : 0.05,   # Gaussian noise scale (fraction of signal std)
    "aug_amp_range"  : (0.8, 1.2),   # random amplitude scaling range

    # Wake cropping
    "wake_pad_epochs": WAKE_PAD_EPOCHS,
}

# ── Training schedule ──────────────────────────────────────────────────────────
NUM_EPOCHS_TRANS = 60
PATIENCE_TRANS   = 15
WARMUP_EPOCHS    = 5
BATCH_SIZE       = 16

# ── Available model names ──────────────────────────────────────────────────────
ALL_MODEL_NAMES = [
    "CrossSleepNetV10",
    "CrossSleepNetV10_NoCross",
    "CrossSleepNetV10_NoTF",
    "SeqTrans-EEG",
]
