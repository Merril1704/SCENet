from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .models.scenet import SCENet, SCENetOutputs
from .concepts import group_feature_indices


@dataclass(frozen=True)
class ExplainConfig:
    top_k_features: int = 10
    top_k_concepts: int = 8
    top_k_features_per_concept: int = 6


def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), values.numel()))
    return torch.topk(values, k=k).indices


def explain_single(
    *,
    model: SCENet,
    x: np.ndarray,
    feature_names: list[str],
    cfg: ExplainConfig,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Explain a single transformed input vector (post-preprocessing).

    Returns a JSON-serializable dict with:
    - prediction probability + label
    - top transformed features by |selected|
    - concept activations + contributions to output
    - per-concept top contributing features (linear part)
    """

    model.eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out: SCENetOutputs = model(xb)
        prob = torch.sigmoid(out.logits).squeeze(0)

    p = float(prob.detach().cpu().item())
    pred = int(p >= float(threshold))

    gates = out.gates.squeeze(0).detach().cpu()
    selected = out.selected.squeeze(0).detach().cpu()
    concepts = out.concepts.squeeze(0).detach().cpu()

    groups = group_feature_indices(feature_names)

    importance = selected.abs()
    top_feat_idx = _topk_indices(importance, cfg.top_k_features)

    top_features = []
    for i in top_feat_idx.tolist():
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        top_features.append(
            {
                "feature": name,
                "index": int(i),
                "x": float(x[i]) if i < len(x) else None,
                "gate": float(gates[i].item()),
                "selected": float(selected[i].item()),
                "importance": float(importance[i].item()),
            }
        )

    # Aggregate importance by original feature group (helps interpret one-hot expansions).
    group_scores = []
    for g, idxs in groups.items():
        idx = torch.as_tensor(idxs, dtype=torch.long)
        s = float(importance.index_select(0, idx).sum().item())
        group_scores.append((g, s))
    group_scores.sort(key=lambda t: t[1], reverse=True)

    top_feature_groups = [
        {"group": g, "importance": float(s)}
        for g, s in group_scores[: max(1, min(cfg.top_k_features, len(group_scores)))]
    ]

    # Concept -> output contributions
    w_out = model.out.weight.detach().cpu().squeeze(0)  # (n_concepts,)
    contrib = w_out * concepts
    top_concept_idx = _topk_indices(contrib.abs(), cfg.top_k_concepts)

    # Feature -> concept contributions (linear part): W2[j,i] * selected[i]
    W2 = model.concept_weight.detach().cpu()  # (n_concepts, n_features)

    concept_details = []
    for j in top_concept_idx.tolist():
        per_feat = W2[j] * selected
        top_local_idx = _topk_indices(per_feat.abs(), cfg.top_k_features_per_concept)

        local_features = []
        for i in top_local_idx.tolist():
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            local_features.append(
                {
                    "feature": name,
                    "index": int(i),
                    "weight": float(W2[j, i].item()),
                    "selected": float(selected[i].item()),
                    "contribution": float(per_feat[i].item()),
                    "abs_contribution": float(per_feat[i].abs().item()),
                }
            )

        # Grouped per-concept contributions (sum of abs contributions within a group)
        group_contrib = []
        abs_per_feat = per_feat.abs()
        for g, idxs in groups.items():
            idx = torch.as_tensor(idxs, dtype=torch.long)
            s = float(abs_per_feat.index_select(0, idx).sum().item())
            group_contrib.append((g, s))
        group_contrib.sort(key=lambda t: t[1], reverse=True)
        top_groups = [
            {"group": g, "abs_contribution": float(s)}
            for g, s in group_contrib[: max(1, min(cfg.top_k_features_per_concept, len(group_contrib)))]
        ]

        concept_details.append(
            {
                "concept_index": int(j),
                "activation": float(concepts[j].item()),
                "weight_to_output": float(w_out[j].item()),
                "contribution_to_logit": float(contrib[j].item()),
                "top_feature_groups": top_groups,
                "top_features": local_features,
            }
        )

    return {
        "threshold": float(threshold),
        "probability": p,
        "predicted_label": pred,
        "top_feature_groups": top_feature_groups,
        "top_features": top_features,
        "concepts": concept_details,
    }
