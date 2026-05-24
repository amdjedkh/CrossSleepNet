"""
train.py — CrossSleepNet v10 Training Script
=============================================
Runs 10-fold subject-wise cross-validation on Sleep-EDF-78 or Sleep-EDF-20.

Usage
-----
  python train.py --data_dir /path/to/sleep-edf --dataset edf78

  # Resume from a previous checkpoint:
  python train.py --data_dir /path/to/sleep-edf --output_dir ./outputs --dataset edf78

  # Ablation run (single model, fewer folds):
  python train.py --data_dir /path/to/sleep-edf --models CrossSleepNetV10_NoCross --n_folds 5

Arguments
---------
  --data_dir   : (required) path to the directory containing *PSG.edf and *Hypnogram.edf
  --output_dir : output directory for checkpoints and logs (default: ./outputs)
  --n_folds    : number of CV folds (default: 10)
  --seed       : random seed (default: 55)
  --dataset    : 'edf78' or 'edf20' (default: edf78)
  --models     : space-separated list of model names to train (default: CrossSleepNetV10)
  --max_subjects : cap on subjects to load (default: 78)
"""

import argparse
import gc
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# Make imports work when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALL_MODEL_NAMES,
    BATCH_SIZE,
    N_FOLDS,
    NUM_CLASSES,
    SEED,
    STAGE_NAMES,
)
from data.preprocessing import match_files, process_subject
from models.crosssleepnet import count_params, make_model
from utils.training import train_fold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrossSleepNet v10 — 10-fold subject-wise CV training"
    )
    p.add_argument("--data_dir",     required=True,
                   help="Directory containing *PSG.edf and *Hypnogram.edf files")
    p.add_argument("--output_dir",   default="./outputs",
                   help="Directory for checkpoint JSON and logs (default: ./outputs)")
    p.add_argument("--n_folds",      type=int, default=N_FOLDS,
                   help=f"Number of CV folds (default: {N_FOLDS})")
    p.add_argument("--seed",         type=int, default=SEED,
                   help=f"Random seed (default: {SEED})")
    p.add_argument("--dataset",      default="edf78", choices=["edf78", "edf20"],
                   help="Dataset variant (default: edf78)")
    p.add_argument("--models",       nargs="+", default=["CrossSleepNetV10"],
                   choices=ALL_MODEL_NAMES,
                   help="Model(s) to train (default: CrossSleepNetV10)")
    p.add_argument("--max_subjects", type=int, default=78,
                   help="Maximum subjects to load (default: 78)")
    return p.parse_args()


def load_cohort(
    data_dir: str,
    dataset_mode: str,
    max_subjects: int,
) -> list:
    """Load and preprocess all subjects from data_dir."""
    data_path = Path(data_dir)
    psg_all   = sorted(glob.glob(str(data_path / "*PSG.edf")))
    hyp_all   = sorted(glob.glob(str(data_path / "*Hypnogram.edf")))

    if not psg_all:
        sys.exit(
            f"ERROR: No *PSG.edf files found in {data_dir}.\n"
            "Download Sleep-EDF from: https://physionet.org/content/sleep-edfx/1.0.0/"
        )

    pairs = match_files(psg_all, hyp_all, dataset_mode=dataset_mode)
    pairs = pairs[:max_subjects]
    print(f"Dataset mode  : {dataset_mode}")
    print(f"Matched pairs : {len(pairs)} (using up to {max_subjects})")

    subject_data = []
    failed       = []

    for psg_path, hyp_path in tqdm(pairs, desc="Preprocessing"):
        from pathlib import Path as _P
        subj_id = _P(psg_path).name[:8]
        eeg, eog, tf, lbl = process_subject(psg_path, hyp_path)
        if eeg is None:
            failed.append(subj_id)
            continue
        subject_data.append({"id": subj_id, "eeg": eeg, "eog": eog, "tf": tf, "label": lbl})

    total_epochs = sum(len(s["label"]) for s in subject_data)
    all_labels   = np.concatenate([s["label"] for s in subject_data])
    counts       = Counter(all_labels.tolist())

    print(f"\nLoaded  : {len(subject_data)} subjects")
    if failed:
        print(f"Failed  : {len(failed)} — {failed}")
    print(f"Epochs  : {total_epochs:,}")
    print("\nClass distribution:")
    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        print(f"  {STAGE_NAMES[c]:6s}: {n:6,}  ({100*n/total_epochs:5.1f}%)")

    return subject_data


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus  = torch.cuda.device_count()
    print(f"Device : {device}")
    if torch.cuda.is_available():
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f} GB)")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "v10_checkpoint.json"

    # ── Load data ─────────────────────────────────────────────────────────────
    subject_data = load_cohort(args.data_dir, args.dataset, args.max_subjects)
    assert len(subject_data) >= args.n_folds, (
        f"Not enough subjects ({len(subject_data)}) for {args.n_folds}-fold CV"
    )

    # ── Resume or start fresh ─────────────────────────────────────────────────
    model_names = args.models

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            loaded = json.load(f)
        fold_results    = {n: loaded["fold_results"].get(n, []) for n in model_names}
        all_preds       = {n: loaded["all_preds"].get(n, [])    for n in model_names}
        all_labs_list   = loaded.get("all_labs", [])
        completed_folds = set(loaded["completed_folds"])
        best_kappa      = loaded.get("best_kappa", -1.0)
        print(f"Resumed checkpoint: {ckpt_path}")
        print(f"Completed folds: {sorted(completed_folds)}")
    else:
        fold_results    = {n: [] for n in model_names}
        all_preds       = {n: [] for n in model_names}
        all_labs_list   = []
        completed_folds = set()
        best_kappa      = -1.0
        print("Starting fresh.")

    # ── Cross-validation loop ─────────────────────────────────────────────────
    kf   = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    idxs = list(range(len(subject_data)))

    print(f"\n{'+'*60}")
    print(f"CrossSleepNet v10  —  {args.n_folds}-fold CV  SEED={args.seed}")
    print(f"Models: {model_names}")
    print(f"{'+'*60}")

    labs = None   # will hold test labels from the last fold of the first model

    for fold, (tr_idx, te_idx) in enumerate(kf.split(idxs)):
        fold_num = fold + 1
        if fold_num in completed_folds:
            print(f"FOLD {fold_num} — already done, skipping.")
            continue

        print(f"\nFOLD {fold_num}/{args.n_folds}  "
              f"(train={len(tr_idx)} subj, test={len(te_idx)} subj)")

        train_s = [subject_data[i] for i in tr_idx]
        test_s  = [subject_data[i] for i in te_idx]

        for name in model_names:
            gc.collect()
            torch.cuda.empty_cache()

            model = make_model(name).to(device)
            if n_gpus > 1:
                model = torch.nn.DataParallel(model)
                print(f"  DataParallel: {n_gpus} GPUs")

            m, preds, labs, trained = train_fold(model, name, train_s, test_s)
            fold_results[name].append(m)
            all_preds[name].extend(preds.tolist())

            if m["Kappa"] > best_kappa:
                best_kappa = m["Kappa"]

            del model, trained
            gc.collect()
            torch.cuda.empty_cache()

        all_labs_list.extend(labs.tolist())
        completed_folds.add(fold_num)

        kappas = [r["Kappa"] for r in fold_results[model_names[0]]]
        state  = {
            "completed_folds": sorted(completed_folds),
            "fold_results"   : fold_results,
            "all_preds"      : all_preds,
            "all_labs"       : all_labs_list,
            "best_kappa"     : best_kappa,
        }
        with open(ckpt_path, "w") as f:
            json.dump(state, f)
        print(f"Checkpoint → {ckpt_path}")
        print(f"Running mean κ = {sum(kappas)/len(kappas):.4f} ({len(kappas)} folds)")
        print("*** DOWNLOAD checkpoint before session ends ***")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'+'*60}")
    for name in model_names:
        kappas = [r["Kappa"] for r in fold_results[name]]
        f1s    = [r["F1_macro"] for r in fold_results[name]]
        accs   = [r["Accuracy"] for r in fold_results[name]]
        print(f"{name}:")
        print(f"  κ     = {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        print(f"  MF1   = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  ACC   = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'+'*60}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
