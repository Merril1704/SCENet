from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .baselines import (
    evaluate_baseline,
    train_catboost,
    train_gbdt,
    train_lightgbm,
    train_logreg,
    train_mlp,
)
from .concepts import ConceptSummaryConfig, plot_concept_heatmap, summarize_concepts
from .datasets.registry import load_dataset
from .interpretability import InterpretabilityConfig, evaluate_interpretability
from .models import SCENet
from .preprocessing import build_preprocessor, transform_X
from .results_table import ResultsTableConfig, write_results_table
from .torch_data import make_loaders
from .torch_train import TrainConfig, train_scenet
from .utils import ensure_dir, save_json, set_seed


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class SCENetRunConfig:
    concepts: int = 16
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    lambda_g: float = 1e-3
    lambda_w2: float = 1e-4
    lambda_z: float = 0.0
    lambda_gate_binary: float = 0.0
    gate_type: str = "sigmoid"
    gate_temperature: float = 1.0


@dataclass(frozen=True)
class BaselinesRunConfig:
    enabled: bool = True
    extra_baselines: bool = False


@dataclass(frozen=True)
class ExtrasConfig:
    with_interpretability: bool = False
    with_concepts: bool = False
    interpretability: InterpretabilityConfig = InterpretabilityConfig()
    concepts: ConceptSummaryConfig = ConceptSummaryConfig()


@dataclass(frozen=True)
class RunAllConfig:
    credit_path: str
    heart_path: str
    credit_seeds: list[int]
    heart_kfold: int = 5
    seed: int = 42
    scenet: SCENetRunConfig = SCENetRunConfig()
    extras: ExtrasConfig = ExtrasConfig()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_seed(
    *,
    dataset: str,
    path: str | Path,
    seed: int,
    out_dir: str | Path,
    scenet_cfg: SCENetRunConfig,
    baselines_cfg: BaselinesRunConfig,
    extras: ExtrasConfig,
) -> dict[str, Any]:
    """Train baselines + SCENet for one seed and write artifacts."""

    set_seed(int(seed))

    from .pipeline import load_and_prepare_numpy

    ds, nd = load_and_prepare_numpy(dataset=dataset, path=path, seed=int(seed))

    out_dir = ensure_dir(out_dir)
    out_baselines = ensure_dir(out_dir / "baselines")
    out_scenet = ensure_dir(out_dir / "scenet")
    out_interp = ensure_dir(out_dir / "interpretability")
    out_concepts = ensure_dir(out_dir / "concepts")

    results: dict[str, Any] = {"dataset": dataset, "seed": int(seed)}

    if baselines_cfg.enabled:
        models = {
            "logreg": train_logreg(nd.X_train, nd.y_train, seed=int(seed)),
            "mlp": train_mlp(nd.X_train, nd.y_train, seed=int(seed)),
            "gbdt": train_gbdt(nd.X_train, nd.y_train, seed=int(seed)),
        }

        if baselines_cfg.extra_baselines:
            for name, fn in [
                ("lightgbm", train_lightgbm),
                ("catboost", train_catboost),
            ]:
                try:
                    models[name] = fn(nd.X_train, nd.y_train, seed=int(seed))
                except ImportError:
                    pass

        baseline_results: dict[str, Any] = {}
        for name, model in models.items():
            r = evaluate_baseline(
                name,
                model,
                X_val=nd.X_val,
                y_val=nd.y_val,
                X_test=nd.X_test,
                y_test=nd.y_test,
            )
            baseline_results[name] = {
                "val": r.val.__dict__,
                "test": r.test.__dict__,
                "threshold": r.threshold,
                "val_thresholded": r.val_thresholded.__dict__,
                "test_thresholded": r.test_thresholded.__dict__,
            }

        save_json(out_baselines / f"{dataset}_seed{seed}.metrics.json", baseline_results)
        save_json(
            out_baselines / f"{dataset}_seed{seed}.features.json",
            {"n_features": len(nd.prepared.feature_names), "feature_names": nd.prepared.feature_names},
        )
        results["baselines"] = list(models.keys())

    device = _device()

    model = SCENet(
        n_features=int(nd.X_train.shape[1]),
        n_concepts=int(scenet_cfg.concepts),
        gate_type=scenet_cfg.gate_type,
        gate_temperature=float(scenet_cfg.gate_temperature),
    )

    train_cfg = TrainConfig(
        lr=float(scenet_cfg.lr),
        weight_decay=float(scenet_cfg.weight_decay),
        epochs=int(scenet_cfg.epochs),
        patience=int(scenet_cfg.patience),
        lambda_g=float(scenet_cfg.lambda_g),
        lambda_w2=float(scenet_cfg.lambda_w2),
        lambda_z=float(scenet_cfg.lambda_z),
        lambda_gate_binary=float(scenet_cfg.lambda_gate_binary),
    )

    loaders = make_loaders(
        X_train=nd.X_train,
        y_train=nd.y_train,
        X_val=nd.X_val,
        y_val=nd.y_val,
        X_test=nd.X_test,
        y_test=nd.y_test,
        batch_size=int(scenet_cfg.batch_size),
    )

    train_res = train_scenet(model=model, loaders=loaders, device=device, cfg=train_cfg)

    metrics_payload = {
        "val": train_res.val.__dict__,
        "test": train_res.test.__dict__,
        "threshold": train_res.threshold,
        "val_thresholded": train_res.val_thresholded.__dict__,
        "test_thresholded": train_res.test_thresholded.__dict__,
        "best_epoch": train_res.best_epoch,
        "device": str(device),
        "seed": int(seed),
        **asdict(scenet_cfg),
    }

    save_json(out_scenet / f"{dataset}_seed{seed}.metrics.json", metrics_payload)

    ckpt_path = out_scenet / f"{dataset}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": int(nd.X_train.shape[1]),
            "n_concepts": int(scenet_cfg.concepts),
            "feature_names": nd.prepared.feature_names,
            "threshold": float(train_res.threshold),
            "gate_type": str(scenet_cfg.gate_type),
            "gate_temperature": float(scenet_cfg.gate_temperature),
        },
        ckpt_path,
    )
    results["scenet_checkpoint"] = str(ckpt_path)

    if extras.with_interpretability:
        cfg_i = extras.interpretability
        cfg_i = InterpretabilityConfig(
            top_k=cfg_i.top_k,
            gate_threshold=cfg_i.gate_threshold,
            threshold=float(cfg_i.threshold if cfg_i.threshold != 0.5 else train_res.threshold),
            noise_std=cfg_i.noise_std,
            max_samples=cfg_i.max_samples,
            batch_size=cfg_i.batch_size,
            seed=int(seed),
        )
        report = evaluate_interpretability(model=model, X=nd.X_test, y=nd.y_test, device=device, cfg=cfg_i)
        report.update(
            {
                "dataset": dataset,
                "seed": int(seed),
                "split": "test",
                "threshold": float(cfg_i.threshold),
                "checkpoint": str(ckpt_path),
            }
        )
        save_json(out_interp / f"{dataset}_seed{seed}_test.interpretability.json", report)

    if extras.with_concepts:
        cfg_c = extras.concepts
        summary = summarize_concepts(
            model=model,
            X=nd.X_train,
            feature_names=list(nd.prepared.feature_names),
            device=device,
            cfg=cfg_c,
        )
        summary.update({"dataset": dataset, "seed": int(seed), "split": "train"})
        out_json = out_concepts / f"{dataset}_seed{seed}_train.concepts.json"
        save_json(out_json, summary)
        plot_concept_heatmap(summary=summary, out_path=out_concepts / f"{dataset}_seed{seed}_train.heatmap.png")

    return results


def run_multiseed(
    *,
    dataset: str,
    path: str | Path,
    seeds: list[int],
    out_dir: str | Path,
    scenet_cfg: SCENetRunConfig,
    baselines_cfg: BaselinesRunConfig,
    extras: ExtrasConfig,
    tag: str | None = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    run_dir = ensure_dir(out_dir / f"{dataset}_multiseed_{tag or _now_tag()}")

    save_json(
        run_dir / "config.json",
        {
            "dataset": dataset,
            "path": str(path),
            "seeds": [int(s) for s in seeds],
            "scenet": asdict(scenet_cfg),
            "baselines": asdict(baselines_cfg),
            "extras": {
                "with_interpretability": bool(extras.with_interpretability),
                "with_concepts": bool(extras.with_concepts),
                "interpretability": asdict(extras.interpretability),
                "concepts": asdict(extras.concepts),
            },
        },
    )

    for s in seeds:
        run_single_seed(
            dataset=dataset,
            path=path,
            seed=int(s),
            out_dir=run_dir,
            scenet_cfg=scenet_cfg,
            baselines_cfg=baselines_cfg,
            extras=extras,
        )

    # Write a combined table for easy reporting
    write_results_table(
        ResultsTableConfig(
            root=run_dir,
            out_csv=run_dir / "results.csv",
            out_md=run_dir / "results.md",
        )
    )

    return run_dir


def run_kfold(
    *,
    dataset: str,
    path: str | Path,
    n_splits: int,
    seed: int,
    out_dir: str | Path,
    scenet_cfg: SCENetRunConfig,
    baselines_cfg: BaselinesRunConfig,
    extras: ExtrasConfig,
    tag: str | None = None,
) -> Path:
    """Run stratified K-fold CV for a dataset (intended for Heart)."""

    set_seed(int(seed))

    out_dir = ensure_dir(out_dir)
    run_dir = ensure_dir(out_dir / f"{dataset}_kfold{int(n_splits)}_{tag or _now_tag()}")

    save_json(
        run_dir / "config.json",
        {
            "dataset": dataset,
            "path": str(path),
            "seed": int(seed),
            "n_splits": int(n_splits),
            "scenet": asdict(scenet_cfg),
            "baselines": asdict(baselines_cfg),
            "extras": {
                "with_interpretability": bool(extras.with_interpretability),
                "with_concepts": bool(extras.with_concepts),
                "interpretability": asdict(extras.interpretability),
                "concepts": asdict(extras.concepts),
            },
        },
    )

    ds = load_dataset(dataset, path)
    X_df = ds.X
    y = ds.y.astype(int)

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))

    out_baselines = ensure_dir(run_dir / "baselines")
    out_scenet = ensure_dir(run_dir / "scenet")
    out_interp = ensure_dir(run_dir / "interpretability")
    out_concepts = ensure_dir(run_dir / "concepts")

    device = _device()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        # Split train fold further into train/val for early stopping + threshold selection.
        X_trainval = X_df.iloc[train_idx]
        y_trainval = y.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_test = y.iloc[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.2,
            random_state=int(seed) + int(fold),
            stratify=y_trainval,
        )

        pre = build_preprocessor(ds)
        pre.fit(X_train)
        feature_names = list(pre.get_feature_names_out())

        X_train_np = transform_X(pre, X_train)
        X_val_np = transform_X(pre, X_val)
        X_test_np = transform_X(pre, X_test)

        y_train_np = y_train.to_numpy(dtype=np.int64)
        y_val_np = y_val.to_numpy(dtype=np.int64)
        y_test_np = y_test.to_numpy(dtype=np.int64)

        fold_tag = f"{dataset}_seed{seed}_fold{fold}"

        if baselines_cfg.enabled:
            models = {
                "logreg": train_logreg(X_train_np, y_train_np, seed=int(seed) + int(fold)),
                "mlp": train_mlp(X_train_np, y_train_np, seed=int(seed) + int(fold)),
                "gbdt": train_gbdt(X_train_np, y_train_np, seed=int(seed) + int(fold)),
            }

            if baselines_cfg.extra_baselines:
                for name, fn in [
                    ("lightgbm", train_lightgbm),
                    ("catboost", train_catboost),
                ]:
                    try:
                        models[name] = fn(X_train_np, y_train_np, seed=int(seed) + int(fold))
                    except ImportError:
                        pass

            baseline_results: dict[str, Any] = {}
            for name, model in models.items():
                r = evaluate_baseline(
                    name,
                    model,
                    X_val=X_val_np,
                    y_val=y_val_np,
                    X_test=X_test_np,
                    y_test=y_test_np,
                )
                baseline_results[name] = {
                    "val": r.val.__dict__,
                    "test": r.test.__dict__,
                    "threshold": r.threshold,
                    "val_thresholded": r.val_thresholded.__dict__,
                    "test_thresholded": r.test_thresholded.__dict__,
                }

            save_json(out_baselines / f"{fold_tag}.metrics.json", baseline_results)
            save_json(
                out_baselines / f"{fold_tag}.features.json",
                {"n_features": len(feature_names), "feature_names": feature_names},
            )

        model = SCENet(
            n_features=int(X_train_np.shape[1]),
            n_concepts=int(scenet_cfg.concepts),
            gate_type=scenet_cfg.gate_type,
            gate_temperature=float(scenet_cfg.gate_temperature),
        )

        train_cfg = TrainConfig(
            lr=float(scenet_cfg.lr),
            weight_decay=float(scenet_cfg.weight_decay),
            epochs=int(scenet_cfg.epochs),
            patience=int(scenet_cfg.patience),
            lambda_g=float(scenet_cfg.lambda_g),
            lambda_w2=float(scenet_cfg.lambda_w2),
            lambda_z=float(scenet_cfg.lambda_z),
            lambda_gate_binary=float(scenet_cfg.lambda_gate_binary),
        )

        loaders = make_loaders(
            X_train=X_train_np,
            y_train=y_train_np,
            X_val=X_val_np,
            y_val=y_val_np,
            X_test=X_test_np,
            y_test=y_test_np,
            batch_size=int(scenet_cfg.batch_size),
        )

        train_res = train_scenet(model=model, loaders=loaders, device=device, cfg=train_cfg)

        metrics_payload = {
            "val": train_res.val.__dict__,
            "test": train_res.test.__dict__,
            "threshold": train_res.threshold,
            "val_thresholded": train_res.val_thresholded.__dict__,
            "test_thresholded": train_res.test_thresholded.__dict__,
            "best_epoch": train_res.best_epoch,
            "device": str(device),
            "seed": int(seed),
            "fold": int(fold),
            **asdict(scenet_cfg),
        }

        save_json(out_scenet / f"{fold_tag}.metrics.json", metrics_payload)

        ckpt_path = out_scenet / f"{fold_tag}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": int(X_train_np.shape[1]),
                "n_concepts": int(scenet_cfg.concepts),
                "feature_names": feature_names,
                "threshold": float(train_res.threshold),
                "gate_type": str(scenet_cfg.gate_type),
                "gate_temperature": float(scenet_cfg.gate_temperature),
            },
            ckpt_path,
        )

        if extras.with_interpretability:
            cfg_i = extras.interpretability
            cfg_i = InterpretabilityConfig(
                top_k=cfg_i.top_k,
                gate_threshold=cfg_i.gate_threshold,
                threshold=float(cfg_i.threshold if cfg_i.threshold != 0.5 else train_res.threshold),
                noise_std=cfg_i.noise_std,
                max_samples=cfg_i.max_samples,
                batch_size=cfg_i.batch_size,
                seed=int(seed) + int(fold),
            )
            report = evaluate_interpretability(model=model, X=X_test_np, y=y_test_np, device=device, cfg=cfg_i)
            report.update(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "fold": int(fold),
                    "split": "test",
                    "threshold": float(cfg_i.threshold),
                    "checkpoint": str(ckpt_path),
                }
            )
            save_json(out_interp / f"{fold_tag}_test.interpretability.json", report)

        if extras.with_concepts:
            cfg_c = extras.concepts
            summary = summarize_concepts(
                model=model,
                X=X_train_np,
                feature_names=list(feature_names),
                device=device,
                cfg=cfg_c,
            )
            summary.update({"dataset": dataset, "seed": int(seed), "fold": int(fold), "split": "train"})
            save_json(out_concepts / f"{fold_tag}_train.concepts.json", summary)
            plot_concept_heatmap(summary=summary, out_path=out_concepts / f"{fold_tag}_train.heatmap.png")

    write_results_table(
        ResultsTableConfig(
            root=run_dir,
            out_csv=run_dir / "results.csv",
            out_md=run_dir / "results.md",
        )
    )

    return run_dir


def run_all(
    *,
    credit_path: str,
    heart_path: str,
    credit_seeds: list[int],
    heart_kfold: int,
    seed: int,
    out_dir: str | Path,
    scenet_cfg: SCENetRunConfig,
    baselines_cfg: BaselinesRunConfig,
    extras: ExtrasConfig,
    tag: str | None = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    run_dir = ensure_dir(out_dir / f"run_all_{tag or _now_tag()}")

    cfg = RunAllConfig(
        credit_path=str(credit_path),
        heart_path=str(heart_path),
        credit_seeds=[int(s) for s in credit_seeds],
        heart_kfold=int(heart_kfold),
        seed=int(seed),
        scenet=scenet_cfg,
        extras=extras,
    )
    save_json(run_dir / "config.json", asdict(cfg))

    # Credit: multi-seed
    run_multiseed(
        dataset="credit_default",
        path=str(credit_path),
        seeds=[int(s) for s in credit_seeds],
        out_dir=run_dir,
        scenet_cfg=scenet_cfg,
        baselines_cfg=baselines_cfg,
        extras=extras,
        tag="credit",
    )

    # Heart: k-fold
    run_kfold(
        dataset="heart_disease",
        path=str(heart_path),
        n_splits=int(heart_kfold),
        seed=int(seed),
        out_dir=run_dir,
        scenet_cfg=scenet_cfg,
        baselines_cfg=baselines_cfg,
        extras=extras,
        tag="heart",
    )

    # Global combined table
    write_results_table(
        ResultsTableConfig(
            root=run_dir,
            out_csv=run_dir / "results.csv",
            out_md=run_dir / "results.md",
        )
    )

    return run_dir
