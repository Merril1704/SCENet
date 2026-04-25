from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .models.scenet import SCENet, SCENetOutputs


def feature_group(feature_name: str) -> str:
    """Map a transformed feature name to an original feature group.

    Our OneHotEncoder uses "{feature}={category}" naming, so grouping by the
    prefix improves readability (e.g., "sex=0" -> "sex").
    """

    s = str(feature_name)
    if "=" in s:
        return s.split("=", 1)[0]
    return s


def group_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        g = feature_group(name)
        groups.setdefault(g, []).append(int(i))
    return groups


def _iter_batches(X: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    n = int(X.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        yield X[start:end]


@dataclass(frozen=True)
class ConceptSummaryConfig:
    top_k_groups: int = 10
    top_k_features: int = 12
    batch_size: int = 512
    max_samples: int = 2000
    seed: int = 42


def _subsample(X: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    if not max_samples or X.shape[0] <= int(max_samples):
        return X

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(max_samples), replace=False)
    idx.sort()
    return X[idx]


def mean_abs_selected(
    *,
    model: SCENet,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Compute mean(|z|) over samples where z = x * g(x)."""

    X_eval = _subsample(X, max_samples=max_samples, seed=seed)

    model.eval()
    total = None
    n_seen = 0

    with torch.no_grad():
        for xb in _iter_batches(X_eval, batch_size=batch_size):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            out: SCENetOutputs = model(xt)
            abs_z = out.selected.abs().detach().cpu().numpy()

            if total is None:
                total = abs_z.sum(axis=0)
            else:
                total += abs_z.sum(axis=0)
            n_seen += abs_z.shape[0]

    if total is None or n_seen == 0:
        raise ValueError("No samples available for concept summary")

    return (total / float(n_seen)).astype(np.float32)


def summarize_concepts(
    *,
    model: SCENet,
    X: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    cfg: ConceptSummaryConfig,
) -> dict[str, Any]:
    """Dataset-level concept summary.

    For each concept, ranks features by |W2| * mean(|z|) and reports:
    - top transformed features
    - top grouped (original) features (aggregating one-hot columns)
    """

    mean_abs_z = mean_abs_selected(
        model=model,
        X=X,
        device=device,
        batch_size=cfg.batch_size,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )

    W2 = model.concept_weight.detach().cpu().numpy()  # (n_concepts, n_features)
    if W2.shape[1] != mean_abs_z.shape[0]:
        raise ValueError(
            f"Shape mismatch: W2 has n_features={W2.shape[1]}, but X produced n_features={mean_abs_z.shape[0]}"
        )

    importance = np.abs(W2) * mean_abs_z[None, :]

    # Concept -> output weights (for context)
    w_out = model.out.weight.detach().cpu().numpy().reshape(-1)

    groups = group_feature_indices(feature_names)

    concepts: list[dict[str, Any]] = []
    for j in range(int(importance.shape[0])):
        row = importance[j]

        kf = max(1, min(int(cfg.top_k_features), row.size))
        top_idx = np.argpartition(-row, kth=kf - 1)[:kf]
        top_idx = top_idx[np.argsort(-row[top_idx])]

        top_features = []
        for i in top_idx.tolist():
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            top_features.append(
                {
                    "feature": name,
                    "group": feature_group(name),
                    "index": int(i),
                    "importance": float(row[i]),
                    "abs_weight": float(abs(W2[j, i])),
                    "mean_abs_selected": float(mean_abs_z[i]),
                }
            )

        group_scores: list[tuple[str, float]] = []
        for g, idxs in groups.items():
            s = float(row[np.asarray(idxs, dtype=int)].sum())
            group_scores.append((g, s))
        group_scores.sort(key=lambda t: t[1], reverse=True)

        kg = max(1, min(int(cfg.top_k_groups), len(group_scores)))
        top_groups = [
            {"group": g, "importance": float(s)} for g, s in group_scores[:kg]
        ]

        concepts.append(
            {
                "concept_index": int(j),
                "weight_to_output": float(w_out[j]) if j < w_out.size else None,
                "top_feature_groups": top_groups,
                "top_features": top_features,
            }
        )

    # Global top groups across all concepts (for plotting convenience)
    group_global: dict[str, float] = {}
    for g, idxs in groups.items():
        idx_arr = np.asarray(idxs, dtype=int)
        group_global[g] = float(importance[:, idx_arr].sum())

    global_groups = sorted(group_global.items(), key=lambda t: t[1], reverse=True)

    return {
        "n_samples": int(min(X.shape[0], int(cfg.max_samples) if cfg.max_samples else X.shape[0])),
        "n_features": int(mean_abs_z.shape[0]),
        "n_concepts": int(importance.shape[0]),
        "top_groups_global": [
            {"group": g, "importance": float(s)} for g, s in global_groups[: max(10, cfg.top_k_groups)]
        ],
        "concepts": concepts,
    }


def plot_concept_heatmap(
    *,
    summary: dict[str, Any],
    out_path: str | Path,
    max_groups: int = 24,
) -> None:
    """Write a concept-vs-feature-group heatmap plot."""

    # Local import to avoid requiring plotting deps for non-plot workflows.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    concepts = summary.get("concepts", [])
    if not concepts:
        raise ValueError("No concepts found in summary")

    top_groups = summary.get("top_groups_global", [])
    groups = [g["group"] for g in top_groups[: int(max_groups)]]
    if not groups:
        raise ValueError("No groups found for heatmap")

    # Build matrix: concepts x groups
    mat = np.zeros((len(concepts), len(groups)), dtype=float)
    group_to_col = {g: i for i, g in enumerate(groups)}

    for r, c in enumerate(concepts):
        for item in c.get("top_feature_groups", []):
            g = item.get("group")
            if g in group_to_col:
                mat[r, group_to_col[g]] = float(item.get("importance", 0.0))

    plt.figure(figsize=(max(8.0, 0.35 * len(groups)), max(4.0, 0.28 * len(concepts))))
    sns.heatmap(mat, xticklabels=groups, yticklabels=[f"c{c['concept_index']}" for c in concepts])
    plt.title("Concept vs Feature-Group Importance")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
