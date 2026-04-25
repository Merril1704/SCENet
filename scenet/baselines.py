from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .metrics import ClassificationMetrics, apply_threshold, compute_binary_metrics, select_threshold_max_f1


@dataclass(frozen=True)
class BaselineResult:
    name: str
    val: ClassificationMetrics
    test: ClassificationMetrics
    threshold: float
    val_thresholded: ClassificationMetrics
    test_thresholded: ClassificationMetrics


def _predict_binary(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = apply_threshold(proba, 0.5)
        return pred, proba

    pred = model.predict(X)
    return pred.astype(int), None


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=10,
    )
    model.fit(X_train, y_train)
    return model


def train_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        random_state=seed,
        learning_rate=0.05,
        max_depth=None,
        max_iter=400,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
):
    """Optional baseline (requires lightgbm)."""

    import importlib

    try:
        lgbm = importlib.import_module("lightgbm")
        LGBMClassifier = getattr(lgbm, "LGBMClassifier")
    except Exception as e:
        raise ImportError("lightgbm is not installed") from e

    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
):
    """Optional baseline (requires catboost)."""

    import importlib

    try:
        cat = importlib.import_module("catboost")
        CatBoostClassifier = getattr(cat, "CatBoostClassifier")
    except Exception as e:
        raise ImportError("catboost is not installed") from e

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        loss_function="Logloss",
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(
    name: str,
    model,
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> BaselineResult:
    val_pred, val_proba = _predict_binary(model, X_val)
    test_pred, test_proba = _predict_binary(model, X_test)

    val = compute_binary_metrics(y_val, val_pred, val_proba)
    test = compute_binary_metrics(y_test, test_pred, test_proba)

    # If probabilities are available, choose threshold on val for max F1.
    if val_proba is not None and test_proba is not None:
        threshold = select_threshold_max_f1(y_val, val_proba)
        val_pred_t = apply_threshold(val_proba, threshold)
        test_pred_t = apply_threshold(test_proba, threshold)
        val_t = compute_binary_metrics(y_val, val_pred_t, val_proba)
        test_t = compute_binary_metrics(y_test, test_pred_t, test_proba)
        return BaselineResult(
            name=name,
            val=val,
            test=test,
            threshold=float(threshold),
            val_thresholded=val_t,
            test_thresholded=test_t,
        )

    # Models without predict_proba keep thresholding semantics unchanged.
    return BaselineResult(
        name=name,
        val=val,
        test=test,
        threshold=0.5,
        val_thresholded=val,
        test_thresholded=test,
    )
