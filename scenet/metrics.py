from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    f1: float
    auc_roc: float | None


def select_threshold_max_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_thresholds: int = 101,
) -> float:
    """Select a probability threshold that maximizes F1 on a validation set.

    Notes:
    - Uses a simple grid search over [0,1].
    - Tie-breaks by higher accuracy, then threshold closest to 0.5.
    """

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1 or y_proba.ndim != 1:
        raise ValueError("y_true and y_proba must be 1D arrays")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length")

    n_thresholds = int(n_thresholds)
    if n_thresholds < 2:
        raise ValueError("n_thresholds must be >= 2")

    thresholds = np.linspace(0.0, 1.0, num=n_thresholds, dtype=float)
    preds = (y_proba[:, None] >= thresholds[None, :]).astype(int)

    y_col = y_true[:, None]
    tp = ((preds == 1) & (y_col == 1)).sum(axis=0)
    fp = ((preds == 1) & (y_col == 0)).sum(axis=0)
    fn = ((preds == 0) & (y_col == 1)).sum(axis=0)

    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    acc = (preds == y_col).mean(axis=0)

    best_f1 = float(f1.max())
    best_idx = np.flatnonzero(f1 == best_f1)

    if best_idx.size > 1:
        best_acc = float(acc[best_idx].max())
        best_idx = best_idx[acc[best_idx] == best_acc]

    if best_idx.size > 1:
        # Prefer the threshold closest to 0.5 for stability.
        best_idx = np.asarray([
            int(best_idx[np.argmin(np.abs(thresholds[best_idx] - 0.5))])
        ])

    return float(thresholds[int(best_idx[0])])


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to hard labels using a threshold."""

    return (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> ClassificationMetrics:
    """Compute standard binary classification metrics.

    - y_pred should be hard labels in {0,1}
    - y_proba should be probability for class 1 if available
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    auc = None
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, np.asarray(y_proba)))
        except ValueError:
            auc = None

    return ClassificationMetrics(accuracy=acc, f1=f1, auc_roc=auc)
