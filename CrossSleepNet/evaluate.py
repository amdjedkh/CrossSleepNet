"""
evaluate.py — CrossSleepNet v10 Evaluation Script
==================================================
Load a checkpoint JSON produced by train.py and print a complete results table,
per-class F1 breakdown, and confusion matrix PNG.

Usage
-----
  python evaluate.py --checkpoint results/v10_checkpoint.json
  python evaluate.py --checkpoint results/v10_checkpoint.json --output_dir results/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from config import NUM_CLASSES, STAGE_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrossSleepNet v10 — evaluate a saved checkpoint"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to the v10_checkpoint.json file produced by train.py")
    p.add_argument("--output_dir", default=None,
                   help="Directory to save confusion matrix PNG (default: same as checkpoint)")
    return p.parse_args()


def print_results_table(fold_results: dict, model_names: list) -> None:
    """Print per-fold and aggregate metrics in a formatted table."""
    col_w = max(len(n) for n in model_names) + 2

    for name in model_names:
        folds = fold_results[name]
        if not folds:
            print(f"  {name}: no results")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {name}")
        print(f"{'='*70}")
        header = f"  {'Fold':>5}  {'Acc':>8}  {'MF1':>8}  {'F1_wtd':>8}  {'κ':>8}"
        print(header)
        print(f"  {'-'*55}")
        for i, m in enumerate(folds):
            print(
                f"  {i+1:>5}  "
                f"{m['Accuracy']:>8.4f}  "
                f"{m['F1_macro']:>8.4f}  "
                f"{m['F1_wtd']:>8.4f}  "
                f"{m['Kappa']:>8.4f}"
            )
        print(f"  {'-'*55}")
        kappas = [m["Kappa"]    for m in folds]
        mf1s   = [m["F1_macro"] for m in folds]
        accs   = [m["Accuracy"] for m in folds]
        print(
            f"  {'MEAN':>5}  "
            f"{np.mean(accs):>8.4f}  "
            f"{np.mean(mf1s):>8.4f}  "
            f"{'---':>8}  "
            f"{np.mean(kappas):>8.4f}"
        )
        print(
            f"  {'±STD':>5}  "
            f"{np.std(accs):>8.4f}  "
            f"{np.std(mf1s):>8.4f}  "
            f"{'---':>8}  "
            f"{np.std(kappas):>8.4f}"
        )

    print(f"\n{'='*70}")
    print("Summary (mean κ ± std):")
    for name in model_names:
        folds  = fold_results.get(name, [])
        kappas = [m["Kappa"] for m in folds]
        mf1s   = [m["F1_macro"] for m in folds]
        if kappas:
            print(
                f"  {name:<40}  "
                f"κ={np.mean(kappas):.4f}±{np.std(kappas):.4f}  "
                f"MF1={np.mean(mf1s):.4f}"
            )


def print_per_class_f1(
    all_preds: dict,
    all_labs: list,
    model_names: list,
) -> None:
    """Print per-class F1 table using concatenated predictions."""
    labs_np = np.array(all_labs)
    if len(labs_np) == 0:
        print("No aggregated predictions in checkpoint — skip per-class F1.")
        return

    print(f"\n{'='*70}")
    print("Per-class F1 (aggregated across all folds):")
    header = f"  {'Model':<35}  " + "  ".join(f"{s:>7}" for s in STAGE_NAMES)
    print(header)
    print(f"  {'-'*65}")
    for name in model_names:
        preds = all_preds.get(name, [])
        if not preds:
            continue
        pc = f1_score(
            labs_np, np.array(preds),
            average=None, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        row = "  ".join(f"{v:>7.4f}" for v in pc)
        print(f"  {name:<35}  {row}")


def save_confusion_matrices(
    all_preds: dict,
    all_labs: list,
    model_names: list,
    output_dir: Path,
) -> None:
    """Save normalised confusion matrix PNG for each model."""
    labs_np = np.array(all_labs)
    if len(labs_np) == 0:
        print("No aggregated labels — skip confusion matrix.")
        return

    ncols = len(model_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        preds = all_preds.get(name, [])
        if not preds:
            ax.set_visible(False)
            continue
        p  = np.array(preds)
        cm = confusion_matrix(labs_np, p, labels=list(range(NUM_CLASSES)))
        cm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            ax=ax, vmin=0, vmax=1, cbar=False,
        )
        kappas = []
        # Compute kappa from aggregated predictions if available
        from sklearn.metrics import cohen_kappa_score
        try:
            k = cohen_kappa_score(labs_np, p)
        except Exception:
            k = float("nan")
        ax.set_title(f"{name}\nκ={k:.3f}", fontweight="bold", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Normalised confusion matrices (aggregated folds)", fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "confusion_matrices.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        sys.exit(f"ERROR: Checkpoint not found: {ckpt_path}")

    with open(ckpt_path) as f:
        state = json.load(f)

    model_names     = state.get("model_names", list(state["fold_results"].keys()))
    fold_results    = state["fold_results"]
    all_preds       = state.get("all_preds", {})
    all_labs        = state.get("all_labs", [])
    completed_folds = state.get("completed_folds", [])

    print(f"Checkpoint     : {ckpt_path}")
    print(f"Completed folds: {completed_folds}")
    print(f"Models         : {model_names}")
    if "n_subjects" in state:
        print(f"Subjects       : {state['n_subjects']}")
    if "dataset_mode" in state:
        print(f"Dataset        : {state['dataset_mode']}")

    print_results_table(fold_results, model_names)
    print_per_class_f1(all_preds, all_labs, model_names)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrices(all_preds, all_labs, model_names, output_dir)


if __name__ == "__main__":
    main()
