"""
Evaluation Metrics — CrossSleepNet v10
========================================
Standard sleep staging evaluation metrics following the AASM convention.
All functions wrap scikit-learn with consistent zero_division handling
and return float scalars or dicts.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from config import NUM_CLASSES, STAGE_NAMES


def cohen_kappa(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute Cohen's κ (quadratic weighting not used in sleep staging).

    Args:
        labels (np.ndarray): Ground-truth integer labels.
        preds  (np.ndarray): Predicted integer labels.

    Returns:
        float — Cohen's κ in [-1, 1].
    """
    return float(cohen_kappa_score(labels, preds))


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """Unweighted macro-average F1 score across all sleep stages.

    Args:
        labels (np.ndarray): Ground-truth integer labels.
        preds  (np.ndarray): Predicted integer labels.

    Returns:
        float — Macro-F1 in [0, 1].
    """
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def weighted_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """Support-weighted F1 score.

    Args:
        labels (np.ndarray): Ground-truth integer labels.
        preds  (np.ndarray): Predicted integer labels.

    Returns:
        float — Weighted F1 in [0, 1].
    """
    return float(f1_score(labels, preds, average="weighted", zero_division=0))


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Overall classification accuracy.

    Args:
        labels (np.ndarray): Ground-truth integer labels.
        preds  (np.ndarray): Predicted integer labels.

    Returns:
        float — Accuracy in [0, 1].
    """
    return float(accuracy_score(labels, preds))


def per_class_f1(
    labels: np.ndarray,
    preds: np.ndarray,
) -> np.ndarray:
    """Per-class F1 scores in stage order [Wake, N1, N2, N3, REM].

    Args:
        labels (np.ndarray): Ground-truth integer labels.
        preds  (np.ndarray): Predicted integer labels.

    Returns:
        np.ndarray of shape (NUM_CLASSES,) — per-class F1 scores.
    """
    return f1_score(
        labels, preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute the full set of standard sleep staging metrics.

    Args:
        preds  (np.ndarray): Predicted integer labels.
        labels (np.ndarray): Ground-truth integer labels.

    Returns:
        dict with keys: 'Accuracy', 'F1_macro', 'F1_wtd', 'Precision',
                        'Recall', 'Kappa'.
    """
    return {
        "Accuracy" : accuracy_score(labels, preds),
        "F1_macro" : f1_score(labels, preds, average="macro",    zero_division=0),
        "F1_wtd"   : f1_score(labels, preds, average="weighted", zero_division=0),
        "Precision": precision_score(labels, preds, average="macro", zero_division=0),
        "Recall"   : recall_score(labels, preds, average="macro",    zero_division=0),
        "Kappa"    : cohen_kappa_score(labels, preds),
    }


def metrics_summary(
    fold_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-fold metric dicts into mean ± std.

    Args:
        fold_results (list[dict]): One metrics dict per fold.

    Returns:
        dict mapping metric_name → {'mean': float, 'std': float}.
    """
    keys = list(fold_results[0].keys())
    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_results])),
            "std":  float(np.std( [m[k] for m in fold_results])),
        }
        for k in keys
    }
