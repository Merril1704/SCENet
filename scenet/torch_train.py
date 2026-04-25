from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam

from .metrics import ClassificationMetrics, apply_threshold, compute_binary_metrics, select_threshold_max_f1
from .models.scenet import SCENet, SCENetOutputs


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 8
    lambda_g: float = 1e-3
    lambda_w2: float = 1e-4
    lambda_z: float = 0.0
    lambda_gate_binary: float = 0.0


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    val: ClassificationMetrics
    test: ClassificationMetrics
    threshold: float
    val_thresholded: ClassificationMetrics
    test_thresholded: ClassificationMetrics


def _eval_probs(model: SCENet, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out: SCENetOutputs = model(xb)
            prob = torch.sigmoid(out.logits)

            ys.append(yb.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    y = np.concatenate(ys, axis=0).astype(int)
    p = np.concatenate(ps, axis=0)
    return y, p


def train_scenet(
    *,
    model: SCENet,
    loaders,
    device: torch.device,
    cfg: TrainConfig,
) -> TrainResult:
    model.to(device)
    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_state = None
    best_auc = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loaders.train:
            xb = xb.to(device)
            yb = yb.to(device)

            out: SCENetOutputs = model(xb)
            loss = bce(out.logits, yb)

            # Sparsity on gates.
            # - sigmoid gates: L1 on gate value
            # - hard-concrete gates: expected L0 (probability of being non-zero)
            if cfg.lambda_g:
                if getattr(model, "gate_type", "sigmoid") == "hard_concrete":
                    gate_logits = (xb * model.gate_scale + model.gate_bias) / float(model.gate_temperature)
                    beta = float(model.hard_concrete_beta)
                    gamma = float(model.hard_concrete_gamma)
                    zeta = float(model.hard_concrete_zeta)
                    l0_prob = torch.sigmoid(gate_logits - beta * math.log(-gamma / zeta))
                    loss = loss + cfg.lambda_g * l0_prob.mean()
                else:
                    loss = loss + cfg.lambda_g * out.gates.mean()

            # Sparsity on selected features (encourages explanations to use fewer effective inputs)
            if cfg.lambda_z:
                loss = loss + cfg.lambda_z * out.selected.abs().mean()

            # Encourage gates to be near {0,1} (reduces "many ~0.5" gates)
            if cfg.lambda_gate_binary:
                loss = loss + cfg.lambda_gate_binary * (out.gates * (1.0 - out.gates)).mean()

            # Sparsity on concept weights (feature -> concept connections)
            loss = loss + cfg.lambda_w2 * model.concept_weight.abs().mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        # Validate by AUC (stable for imbalance)
        yv, pv = _eval_probs(model, loaders.val, device)
        try:
            val_auc = float(roc_auc_score(yv, pv))
        except ValueError:
            val_auc = -1.0

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    yv, pv = _eval_probs(model, loaders.val, device)
    yt, pt = _eval_probs(model, loaders.test, device)

    # Default 0.5 threshold metrics (for comparability)
    val_pred_05 = apply_threshold(pv, 0.5)
    test_pred_05 = apply_threshold(pt, 0.5)
    val_05 = compute_binary_metrics(yv, val_pred_05, pv)
    test_05 = compute_binary_metrics(yt, test_pred_05, pt)

    # Choose threshold on validation set to maximize F1, then report thresholded test metrics.
    threshold = select_threshold_max_f1(yv, pv)
    val_pred_t = apply_threshold(pv, threshold)
    test_pred_t = apply_threshold(pt, threshold)
    val_t = compute_binary_metrics(yv, val_pred_t, pv)
    test_t = compute_binary_metrics(yt, test_pred_t, pt)

    return TrainResult(
        best_epoch=best_epoch,
        val=val_05,
        test=test_05,
        threshold=float(threshold),
        val_thresholded=val_t,
        test_thresholded=test_t,
    )
