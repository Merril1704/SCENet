from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .metrics import ClassificationMetrics, compute_binary_metrics
from .models.scenet import SCENet, SCENetOutputs


@dataclass(frozen=True)
class InterpretabilityConfig:
    top_k: int = 10
    gate_threshold: float = 0.5
    threshold: float = 0.5
    noise_std: float = 0.05
    max_samples: int = 2000
    batch_size: int = 512
    seed: int = 42


def _iter_batches(X: np.ndarray, batch_size: int) -> Iterator[tuple[int, np.ndarray]]:
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield start, X[start:end]


def _predict_proba(model: SCENet, X: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for _, xb in _iter_batches(X, batch_size):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            out: SCENetOutputs = model(xt)
            p = torch.sigmoid(out.logits).detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0)


def _topk_feature_sets(model: SCENet, X: np.ndarray, *, k: int, device: torch.device, batch_size: int) -> list[set[int]]:
    model.eval()
    sets: list[set[int]] = []

    with torch.no_grad():
        for _, xb in _iter_batches(X, batch_size):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            out: SCENetOutputs = model(xt)
            imp = out.selected.abs()
            kk = max(1, min(int(k), imp.shape[1]))
            idx = torch.topk(imp, k=kk, dim=1).indices.detach().cpu().numpy()
            for row in idx:
                sets.append(set(int(i) for i in row.tolist()))

    return sets


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _summarize_counts(counts: list[int]) -> dict[str, float]:
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def evaluate_interpretability(
    *,
    model: SCENet,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    cfg: InterpretabilityConfig,
) -> dict[str, Any]:
    """Compute SCENet interpretability metrics on a dataset split.

    Metrics (from project spec):
    - Sparsity: number of active features per prediction
    - Stability: Jaccard similarity of top-k features under small perturbations
    - Faithfulness: performance drop when top-k features are removed/zeroed
    - Consistency: Jaccard similarity of top-k features between nearest neighbors

    Notes:
    - X is assumed to be the *preprocessed* numeric matrix (output of preprocessing).
    - Feature removal is implemented as setting selected features to 0.0 (mean after scaling).
    """

    rng = np.random.default_rng(int(cfg.seed))

    n = X.shape[0]
    if n == 0:
        raise ValueError("X is empty")

    if cfg.max_samples and n > cfg.max_samples:
        idx = rng.choice(n, size=int(cfg.max_samples), replace=False)
        idx.sort()
        X_eval = X[idx]
        y_eval = y[idx]
    else:
        X_eval = X
        y_eval = y

    # 1) Sparsity: count active gates per sample.
    active_counts: list[int] = []
    model.eval()
    with torch.no_grad():
        for _, xb in _iter_batches(X_eval, cfg.batch_size):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            out: SCENetOutputs = model(xt)
            counts = (out.gates > float(cfg.gate_threshold)).sum(dim=1).detach().cpu().numpy()
            active_counts.extend(int(c) for c in counts.tolist())

    sparsity = {
        "gate_threshold": float(cfg.gate_threshold),
        "active_features": _summarize_counts(active_counts),
    }

    # Pre-compute top-k sets for X_eval (used by 2/3/4)
    topk_sets = _topk_feature_sets(
        model,
        X_eval,
        k=cfg.top_k,
        device=device,
        batch_size=cfg.batch_size,
    )

    # 2) Stability: perturb inputs and compare top-k sets.
    noise = rng.normal(loc=0.0, scale=float(cfg.noise_std), size=X_eval.shape).astype(np.float32)
    X_pert = (X_eval + noise).astype(np.float32)
    topk_sets_pert = _topk_feature_sets(
        model,
        X_pert,
        k=cfg.top_k,
        device=device,
        batch_size=cfg.batch_size,
    )

    stability_scores = [
        _jaccard(a, b) for a, b in zip(topk_sets, topk_sets_pert, strict=False)
    ]
    stability = {
        "top_k": int(cfg.top_k),
        "noise_std": float(cfg.noise_std),
        "jaccard_mean": float(np.mean(stability_scores)),
        "jaccard_median": float(np.median(stability_scores)),
    }

    # 3) Faithfulness: remove top-k features and measure performance drop.
    p_base = _predict_proba(model, X_eval, device=device, batch_size=cfg.batch_size)

    X_mask = X_eval.copy()
    for i, s in enumerate(topk_sets):
        if s:
            X_mask[i, list(s)] = 0.0

    p_mask = _predict_proba(model, X_mask, device=device, batch_size=cfg.batch_size)

    thr = float(cfg.threshold)
    base = compute_binary_metrics(y_eval, (p_base >= thr).astype(int), p_base)
    masked = compute_binary_metrics(y_eval, (p_mask >= thr).astype(int), p_mask)

    faithfulness = {
        "top_k": int(cfg.top_k),
        "baseline": base.__dict__,
        "masked": masked.__dict__,
        "delta": {
            "accuracy": float(base.accuracy - masked.accuracy),
            "f1": float(base.f1 - masked.f1),
            "auc_roc": None
            if base.auc_roc is None or masked.auc_roc is None
            else float(base.auc_roc - masked.auc_roc),
        },
        "mean_abs_proba_change": float(np.mean(np.abs(p_base - p_mask))),
    }

    # 4) Consistency: nearest neighbors should have similar explanations.
    if X_eval.shape[0] < 2:
        consistency = {"top_k": int(cfg.top_k), "jaccard_mean": None, "reason": "not enough samples"}
    else:
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(X_eval)
        neigh_idx = nn.kneighbors(return_distance=False)

        # For each point, compare with its nearest *other* neighbor.
        scores: list[float] = []
        for i in range(X_eval.shape[0]):
            j = int(neigh_idx[i, 1])
            if i == j:
                continue
            scores.append(_jaccard(topk_sets[i], topk_sets[j]))

        consistency = {
            "top_k": int(cfg.top_k),
            "jaccard_mean": None if not scores else float(np.mean(scores)),
            "jaccard_median": None if not scores else float(np.median(scores)),
        }

    return {
        "n_samples": int(X_eval.shape[0]),
        "sparsity": sparsity,
        "stability": stability,
        "faithfulness": faithfulness,
        "consistency": consistency,
    }
