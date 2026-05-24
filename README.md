# CrossSleepNet

> Conformer-based Bidirectional EEG-EOG Cross-Attention for Automated Sleep Staging

## Architecture

CrossSleepNet v10 is a multimodal deep learning framework that jointly analyses
EEG (Fpz-Cz, Pz-Oz) and EOG signals for automated sleep staging according to
the AASM 5-class standard (Wake, N1, N2, N3, REM).

```
INPUT: sequence of L=20 consecutive 30-second epochs

Per epoch
─────────
EEG  (2 × 3000)   ─── Level 1a ──▶  ConformerEncoder ──▶ He  (P × 128)
EOG  (1 × 3000)   ─── Level 1b ──▶  ConformerEncoder ──▶ Hg  (P × 128)
STFT (3 × 29×128) ─── Level 1c ──▶  TFImageEncoder   ──▶ tf  (256,)

Level 2  ──▶  Bidirectional EEG-EOG Cross-Attention
              He, Hg = CrossAttn(He, Hg)

Level 3  ──▶  MLP fusion
              epoch_emb = MLP( [mean(He); mean(Hg); tf] )   → (256,)

Level 4  ──▶  Sequence Transformer (over L epoch embeddings)
              logits = SeqTransformer(epoch_emb_1 … epoch_emb_L)
              → classification of centre epoch
```

**ConformerEncoder** (Level 1a/1b) — Macaron-style blocks combining
depthwise conv (local ~300 ms patterns) and multi-head self-attention (global
dependencies), with multi-scale patch tokenisation.

**EEGEOGCrossAttention** (Level 2) — Our core contribution. Bidirectional
cross-attention (EEG attends to EOG, EOG attends to EEG) captures the
physiological neural-ocular coupling that distinguishes sleep stages,
particularly REM vs. N2.

**TFImageEncoder** (Level 1c) — log₁p STFT amplitude images processed by a
2-D CNN followed by a Transformer with a [CLS] token.

**SeqTransformer** (Level 4) — Sequence-level Transformer classifying the
centre epoch using bidirectional context from all 20 epochs.

## Results

### Sleep-EDF-78 (10-fold subject-wise CV)

| Model                     |    κ   |  ±std  |  MF1   | Accuracy |
|---------------------------|--------|--------|--------|----------|
| CrossSleepNetV10 (full)   | 0.7527 | 0.0339 | 0.7803 | 0.8199   |
| w/o EEG-EOG cross-attn    | 0.7371 | 0.0381 | 0.7687 | 0.8066   |
| w/o STFT branch           | 0.7375 | 0.0278 | 0.7655 | 0.8080   |
| EEG only (baseline)       | 0.6967 | 0.0232 | 0.7299 | 0.7783   |

### Sleep-EDF-20 (10-fold subject-wise CV)

| Model              |    κ   |  ±std  |  MF1   | Accuracy |
|--------------------|--------|--------|--------|----------|
| CrossSleepNetV10   | 0.7974 | 0.0561 | 0.8256 | 0.8519   |

### Literature context (Sleep-EDF-78, 78 subjects)

| Method                         | Modality | ACC   | MF1   |   κ   |
|--------------------------------|----------|-------|-------|-------|
| AttnSleep (TNSRE 2021)         | EEG      | 82.9% | 0.778 | 0.774 |
| XSleepNet2 (TPAMI 2021)        | EEG      | 84.0% | 0.781 | 0.778 |
| SleepTransformer (TBME 2022)   | EEG      | 84.9% | 0.788 | 0.789 |
| L-SeqSleepNet (JBHI 2023)      | EEG      | ~85.4%| ~0.800| ~0.800|
| CrossFusionSleepNet (BSPC 2025)| EEG+EOG  | 87.5% | 0.830 | ~0.830|

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download Sleep-EDF Expanded (cassette study) from PhysioNet:

```
https://physionet.org/content/sleep-edfx/1.0.0/
```

The expected file naming convention is `*PSG.edf` and `*Hypnogram.edf` in a
flat directory. The preprocessing pipeline handles the full Sleep-EDF-78 (SC*)
and Sleep-EDF-20 (SC40* first recordings) subsets automatically.

## Training

```bash
# Full 10-fold CV on Sleep-EDF-78
python train.py --data_dir /path/to/sleep-edf --dataset edf78

# Sleep-EDF-20
python train.py --data_dir /path/to/sleep-edf --dataset edf20

# Custom output directory
python train.py --data_dir /path/to/sleep-edf --output_dir ./my_outputs

# Ablation study (all four models)
python train.py --data_dir /path/to/sleep-edf \
    --models CrossSleepNetV10 CrossSleepNetV10_NoCross CrossSleepNetV10_NoTF SeqTrans-EEG
```

Training automatically resumes from an existing checkpoint in `--output_dir`.
The checkpoint JSON is saved after every fold.

## Evaluation

```bash
python evaluate.py --checkpoint results/v10_checkpoint.json
```

Outputs:
- Per-fold metrics table (κ, MF1, accuracy)
- Mean ± std summary
- Per-class F1 breakdown
- `confusion_matrices.png` in the same directory

## Repository structure

```
CrossSleepNet/
├── config.py              # All hyperparameters and constants
├── train.py               # Training script (10-fold CV)
├── evaluate.py            # Evaluation from checkpoint JSON
├── requirements.txt
├── models/
│   ├── conformer.py       # ConformerBlock, ConformerEncoder
│   ├── cross_attention.py # EEGEOGCrossAttention
│   ├── tf_encoder.py      # TFImageEncoder
│   └── crosssleepnet.py   # Full model + ablations + factory
├── data/
│   ├── preprocessing.py   # EDF loading, filtering, STFT
│   └── dataset.py         # SequenceDataset (float16 storage)
├── utils/
│   ├── metrics.py         # κ, MF1, accuracy, per-class F1
│   └── training.py        # LabelSmoothCE, train_fold, evaluate
└── results/               # Place checkpoint JSON files here
```

## Citation

*To be updated after publication.*

## License

MIT
