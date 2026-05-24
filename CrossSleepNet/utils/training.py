"""
Training Infrastructure — CrossSleepNet v10
=============================================
Training loop, evaluation loop, and fold-level training orchestration.

Implements:
  - LabelSmoothCE  : label-smoothing cross-entropy with optional class weights
  - train_one_epoch: single-epoch training pass with gradient clipping
  - evaluate       : inference loop returning predictions and labels
  - train_fold     : complete training run for one CV fold with warmup +
                     cosine-annealing LR schedule and early stopping
"""

import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CFG,
    NUM_EPOCHS_TRANS,
    PATIENCE_TRANS,
    SEQ_LEN,
    WARMUP_EPOCHS,
)
from data.dataset import SequenceDataset, class_weights_focal
from utils.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()


# ── Loss function ──────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    """Label-smoothing cross-entropy loss with optional class weighting.

    Distributes a fraction `smoothing` of the probability mass uniformly
    across all classes, reducing over-confidence on noisy sleep-stage labels.

    Args:
        smoothing (float): Smoothing factor in [0, 1) (default 0.05).
        weight (torch.Tensor | None): Per-class weight tensor (NUM_CLASSES,).
    """

    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor = None):
        super().__init__()
        self.s = smoothing
        self.w = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, num_classes) — raw logits.
            target : (B,)             — integer class indices.
        Returns:
            Scalar loss tensor.
        """
        n = pred.size(-1)
        with torch.no_grad():
            soft = torch.full_like(pred, self.s / (n - 1))
            soft.scatter_(-1, target.unsqueeze(-1), 1.0 - self.s)
        lp   = F.log_softmax(pred, dim=-1)
        loss = -(soft * lp).sum(-1)
        if self.w is not None:
            loss = loss * self.w[target]
        return loss.mean()


# ── Single-epoch training pass ─────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    crit: nn.Module,
) -> Tuple[float, float]:
    """Run one training epoch over the DataLoader.

    Args:
        model  : PyTorch model (may be DataParallel-wrapped).
        loader : Training DataLoader.
        opt    : Optimiser.
        crit   : Loss criterion (LabelSmoothCE).

    Returns:
        (mean_loss, accuracy) — floats over the epoch.
    """
    model.train()
    tot_loss = tot_correct = tot = 0

    for eeg_seq, eog_seq, tf_seq, lab in loader:
        eeg_seq = eeg_seq.to(DEVICE, non_blocking=True)
        eog_seq = eog_seq.to(DEVICE, non_blocking=True)
        tf_seq  = tf_seq.to(DEVICE,  non_blocking=True)
        lab     = lab.to(DEVICE,     non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out  = model(eeg_seq, eog_seq, tf_seq)
        loss = crit(out, lab)
        loss.backward()

        # Gradient clipping on the underlying (non-DataParallel) model
        core = model.module if hasattr(model, "module") else model
        nn.utils.clip_grad_norm_(core.parameters(), CFG["grad_clip"])
        opt.step()

        bs           = lab.size(0)
        tot_loss    += loss.item() * bs
        tot_correct += (out.argmax(1) == lab).sum().item()
        tot         += bs

    return tot_loss / tot, tot_correct / tot


# ── Evaluation pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run inference over the DataLoader.

    Args:
        model  : PyTorch model (may be DataParallel-wrapped).
        loader : Validation/test DataLoader.
        crit   : Loss criterion.

    Returns:
        (mean_loss, preds, labels) — float and two 1-D numpy arrays.
    """
    model.eval()
    tot_loss = tot = 0
    preds, labs = [], []

    for eeg_seq, eog_seq, tf_seq, lab in loader:
        eeg_seq = eeg_seq.to(DEVICE, non_blocking=True)
        eog_seq = eog_seq.to(DEVICE, non_blocking=True)
        tf_seq  = tf_seq.to(DEVICE,  non_blocking=True)
        lab     = lab.to(DEVICE,     non_blocking=True)

        out       = model(eeg_seq, eog_seq, tf_seq)
        tot_loss += crit(out, lab).item() * lab.size(0)
        tot      += lab.size(0)
        preds.append(out.argmax(1).cpu().numpy())
        labs.append(lab.cpu().numpy())

    return tot_loss / tot, np.concatenate(preds), np.concatenate(labs)


# ── Fold-level training ────────────────────────────────────────────────────────

def train_fold(
    model: nn.Module,
    name: str,
    train_subjects: List[dict],
    test_subjects: List[dict],
    num_epochs: int = NUM_EPOCHS_TRANS,
    patience: int = PATIENCE_TRANS,
    base_lr: float = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, nn.Module]:
    """Full training pipeline for one cross-validation fold.

    Schedule: linear warmup for WARMUP_EPOCHS, then cosine annealing to 1%
    of peak LR. Early stopping on validation macro-F1.

    Args:
        model           : Freshly instantiated model (already on DEVICE).
        name            : Model name string (for logging).
        train_subjects  : Subject dicts for the training split.
        test_subjects   : Subject dicts for the test split.
        num_epochs      : Maximum training epochs.
        patience        : Early stopping patience (epochs without improvement).
        base_lr         : Peak learning rate; defaults to CFG['lr_trans'].

    Returns:
        (metrics_dict, preds, labels, trained_core_model)
          metrics_dict  : dict from compute_metrics
          preds         : (N_test,) predicted labels
          labels        : (N_test,) ground-truth labels
          trained_core  : the (non-DataParallel) model with best weights loaded
    """
    if base_lr is None:
        base_lr = CFG["lr_trans"]

    cw     = class_weights_focal(train_subjects).to(DEVICE)
    tr_ds  = SequenceDataset(train_subjects, seq_len=SEQ_LEN, training=True)
    te_ds  = SequenceDataset(test_subjects,  seq_len=SEQ_LEN, training=False)
    tr_ldr = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  drop_last=True,
                        num_workers=2, pin_memory=True)
    te_ldr = DataLoader(te_ds, BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

    crit = LabelSmoothCE(CFG["label_smooth"], cw)
    core = model.module if hasattr(model, "module") else model
    opt  = torch.optim.AdamW(core.parameters(), lr=1.0, weight_decay=CFG["wd"])

    def lr_schedule(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return base_lr * (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, num_epochs - WARMUP_EPOCHS)
        return base_lr * max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    best_f1    = -1.0
    patience_c = 0
    best_state = None

    print(
        f"  {name} | train={len(tr_ds):,}  test={len(te_ds):,}  "
        f"ep={num_epochs}  pat={patience}  GPUs={N_GPUS}"
    )

    for ep in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, tr_ldr, opt, crit)
        va_loss, vp, vl = evaluate(model, te_ldr, crit)
        va_f1  = f1_score(vl, vp, average="macro", zero_division=0)
        va_acc = accuracy_score(vl, vp)
        sch.step()
        cur_lr = sch.get_last_lr()[0]

        print(
            f"    ep {ep+1:2d}/{num_epochs}  "
            f"tr={tr_loss:.4f}/{tr_acc:.3f} | "
            f"va={va_loss:.4f}/{va_acc:.3f}/F1={va_f1:.3f} "
            f"lr={cur_lr:.2e} [{time.time()-t0:.1f}s]",
            flush=True,
        )

        if va_f1 > best_f1:
            best_f1    = va_f1
            patience_c = 0
            core = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f"    early stop ep {ep+1} (best F1={best_f1:.4f})")
                break

    # Restore best weights and compute final metrics
    core = model.module if hasattr(model, "module") else model
    core.load_state_dict(best_state)
    _, preds, labs = evaluate(model, te_ldr, crit)
    m = compute_metrics(preds, labs)
    print(f"  -> acc={m['Accuracy']:.4f}  F1={m['F1_macro']:.4f}  κ={m['Kappa']:.4f}")

    del tr_ldr, te_ldr, tr_ds, te_ds, opt, sch, crit, cw, best_state
    gc.collect()
    torch.cuda.empty_cache()

    return m, preds, labs, core
