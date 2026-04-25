import argparse
from pathlib import Path

from .datasets import DatasetId
from .datasets.registry import load_dataset
from .pipeline import load_and_prepare_numpy
from .baselines import (
    evaluate_baseline,
    train_catboost,
    train_gbdt,
    train_lightgbm,
    train_logreg,
    train_mlp,
)
from .models import SCENet
from .torch_data import make_loaders
from .torch_train import TrainConfig, train_scenet
from .explain import ExplainConfig, explain_single
from .interpretability import InterpretabilityConfig, evaluate_interpretability
from .utils import ensure_dir, env_summary, save_json, set_seed

from .concepts import ConceptSummaryConfig, plot_concept_heatmap, summarize_concepts
from .experiments import (
    BaselinesRunConfig,
    ExtrasConfig,
    SCENetRunConfig,
    run_all,
    run_kfold,
    run_multiseed,
)
from .results_table import ResultsTableConfig, write_results_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenet",
        description="SCENet: intrinsic explainability for tabular models (PyTorch)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Print environment + data path hints")
    doctor.add_argument(
        "--data-root",
        default="data/raw",
        help="Default raw data folder (can contain dataset subfolders/files)",
    )

    peek = sub.add_parser("peek-data", help="Try loading a dataset and print schema")
    peek.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    peek.add_argument(
        "--path",
        required=True,
        help="Path to raw dataset file or a folder containing it",
    )

    train_b = sub.add_parser("train-baselines", help="Train baselines and write metrics")
    train_b.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    train_b.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    train_b.add_argument("--seed", type=int, default=42)
    train_b.add_argument("--out", default="outputs/baselines", help="Output directory")
    train_b.add_argument(
        "--extra-baselines",
        action="store_true",
        help="If available, also run LightGBM/CatBoost baselines",
    )

    train_s = sub.add_parser("train-scenet", help="Train SCENet and write metrics + checkpoint")
    train_s.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    train_s.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    train_s.add_argument("--seed", type=int, default=42)
    train_s.add_argument("--out", default="outputs/scenet", help="Output directory")
    train_s.add_argument("--concepts", type=int, default=16)
    train_s.add_argument("--epochs", type=int, default=50)
    train_s.add_argument("--patience", type=int, default=8)
    train_s.add_argument("--lr", type=float, default=1e-3)
    train_s.add_argument("--lambda-g", type=float, default=1e-3)
    train_s.add_argument("--lambda-w2", type=float, default=1e-4)
    train_s.add_argument(
        "--lambda-z",
        type=float,
        default=0.0,
        help="L1 penalty on selected features z=x*g (often improves practical sparsity)",
    )
    train_s.add_argument(
        "--lambda-gate-binary",
        type=float,
        default=0.0,
        help="Penalty on g*(1-g) to encourage near-binary gates",
    )
    train_s.add_argument(
        "--gate-type",
        choices=["sigmoid", "hard_concrete"],
        default="sigmoid",
        help="Gate mechanism (hard_concrete is an L0-style gate)",
    )
    train_s.add_argument(
        "--gate-temperature",
        type=float,
        default=1.0,
        help="Gate temperature (smaller => sharper / more binary)",
    )
    train_s.add_argument("--batch-size", type=int, default=256)

    explain = sub.add_parser("explain-scenet", help="Explain one sample using a trained SCENet")
    explain.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    explain.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    explain.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    explain.add_argument("--seed", type=int, default=42, help="Seed used for splitting/preprocessing")
    explain.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to draw the sample from",
    )
    explain.add_argument("--index", type=int, default=0, help="Row index within the chosen split")
    explain.add_argument("--top-features", type=int, default=10)
    explain.add_argument("--top-concepts", type=int, default=8)
    explain.add_argument("--top-features-per-concept", type=int, default=6)
    explain.add_argument("--out", default="", help="Optional path to write explanation JSON")

    interp = sub.add_parser(
        "eval-interpretability",
        help="Compute interpretability metrics (sparsity/stability/faithfulness/consistency)",
    )
    interp.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    interp.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    interp.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    interp.add_argument("--seed", type=int, default=42, help="Seed used for splitting/preprocessing")
    interp.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate",
    )
    interp.add_argument("--top-k", type=int, default=10)
    interp.add_argument("--gate-threshold", type=float, default=0.5)
    interp.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for metrics (defaults to checkpoint threshold, else 0.5)",
    )
    interp.add_argument("--noise-std", type=float, default=0.05)
    interp.add_argument("--max-samples", type=int, default=2000)
    interp.add_argument("--batch-size", type=int, default=512)
    interp.add_argument("--out", default="outputs/interpretability", help="Output directory")

    concepts = sub.add_parser("summarize-concepts", help="Summarize learned concepts and write plots")
    concepts.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    concepts.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    concepts.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    concepts.add_argument("--seed", type=int, default=42, help="Seed used for splitting/preprocessing")
    concepts.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to summarize",
    )
    concepts.add_argument("--top-groups", type=int, default=10)
    concepts.add_argument("--top-features", type=int, default=12)
    concepts.add_argument("--batch-size", type=int, default=512)
    concepts.add_argument("--max-samples", type=int, default=2000)
    concepts.add_argument("--out", default="outputs/concepts", help="Output directory")

    table = sub.add_parser("make-results-table", help="Generate a CSV + Markdown results table from outputs")
    table.add_argument("--root", default="outputs", help="Root directory to scan (recursive)")
    table.add_argument("--out", default="", help="Output directory (defaults to --root)")

    ms = sub.add_parser("run-multiseed", help="Run baselines + SCENet across multiple seeds")
    ms.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    ms.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    ms.add_argument("--seeds", nargs="+", type=int, required=True)
    ms.add_argument("--out", default="outputs/experiments", help="Output directory")
    ms.add_argument("--tag", default="", help="Optional tag for the run directory")
    ms.add_argument("--skip-baselines", action="store_true")
    ms.add_argument("--extra-baselines", action="store_true", help="Also run LightGBM/CatBoost if installed")
    ms.add_argument("--with-interpretability", action="store_true")
    ms.add_argument("--with-concepts", action="store_true")
    ms.add_argument("--concepts", type=int, default=16)
    ms.add_argument("--epochs", type=int, default=50)
    ms.add_argument("--patience", type=int, default=8)
    ms.add_argument("--lr", type=float, default=1e-3)
    ms.add_argument("--weight-decay", type=float, default=1e-4)
    ms.add_argument("--batch-size", type=int, default=256)
    ms.add_argument("--lambda-g", type=float, default=1e-3)
    ms.add_argument("--lambda-w2", type=float, default=1e-4)
    ms.add_argument("--lambda-z", type=float, default=0.0)
    ms.add_argument("--lambda-gate-binary", type=float, default=0.0)
    ms.add_argument("--gate-type", choices=["sigmoid", "hard_concrete"], default="sigmoid")
    ms.add_argument("--gate-temperature", type=float, default=1.0)
    ms.add_argument("--top-k", type=int, default=10)
    ms.add_argument("--gate-threshold", type=float, default=0.5)
    ms.add_argument("--noise-std", type=float, default=0.05)
    ms.add_argument("--max-samples", type=int, default=2000)
    ms.add_argument("--interp-batch-size", type=int, default=512)

    kf = sub.add_parser("run-kfold", help="Run stratified K-fold CV (recommended for Heart)")
    kf.add_argument("--dataset", choices=[d.value for d in DatasetId], required=True)
    kf.add_argument("--path", required=True, help="Path to raw dataset file or folder")
    kf.add_argument("--k", type=int, default=5)
    kf.add_argument("--seed", type=int, default=42)
    kf.add_argument("--out", default="outputs/experiments", help="Output directory")
    kf.add_argument("--tag", default="", help="Optional tag for the run directory")
    kf.add_argument("--skip-baselines", action="store_true")
    kf.add_argument("--extra-baselines", action="store_true", help="Also run LightGBM/CatBoost if installed")
    kf.add_argument("--with-interpretability", action="store_true")
    kf.add_argument("--with-concepts", action="store_true")
    kf.add_argument("--concepts", type=int, default=16)
    kf.add_argument("--epochs", type=int, default=50)
    kf.add_argument("--patience", type=int, default=8)
    kf.add_argument("--lr", type=float, default=1e-3)
    kf.add_argument("--weight-decay", type=float, default=1e-4)
    kf.add_argument("--batch-size", type=int, default=256)
    kf.add_argument("--lambda-g", type=float, default=1e-3)
    kf.add_argument("--lambda-w2", type=float, default=1e-4)
    kf.add_argument("--lambda-z", type=float, default=0.0)
    kf.add_argument("--lambda-gate-binary", type=float, default=0.0)
    kf.add_argument("--gate-type", choices=["sigmoid", "hard_concrete"], default="sigmoid")
    kf.add_argument("--gate-temperature", type=float, default=1.0)
    kf.add_argument("--top-k", type=int, default=10)
    kf.add_argument("--gate-threshold", type=float, default=0.5)
    kf.add_argument("--noise-std", type=float, default=0.05)
    kf.add_argument("--max-samples", type=int, default=2000)
    kf.add_argument("--interp-batch-size", type=int, default=512)

    ra = sub.add_parser("run-all", help="Run credit multi-seed + heart k-fold in one command")
    ra.add_argument("--credit-path", required=True)
    ra.add_argument("--heart-path", required=True)
    ra.add_argument("--credit-seeds", nargs="+", type=int, required=True)
    ra.add_argument("--heart-kfold", type=int, default=5)
    ra.add_argument("--seed", type=int, default=42)
    ra.add_argument("--out", default="outputs/experiments", help="Output directory")
    ra.add_argument("--tag", default="", help="Optional tag for the run directory")
    ra.add_argument("--skip-baselines", action="store_true")
    ra.add_argument("--extra-baselines", action="store_true", help="Also run LightGBM/CatBoost if installed")
    ra.add_argument("--with-interpretability", action="store_true")
    ra.add_argument("--with-concepts", action="store_true")
    ra.add_argument("--concepts", type=int, default=16)
    ra.add_argument("--epochs", type=int, default=50)
    ra.add_argument("--patience", type=int, default=8)
    ra.add_argument("--lr", type=float, default=1e-3)
    ra.add_argument("--weight-decay", type=float, default=1e-4)
    ra.add_argument("--batch-size", type=int, default=256)
    ra.add_argument("--lambda-g", type=float, default=1e-3)
    ra.add_argument("--lambda-w2", type=float, default=1e-4)
    ra.add_argument("--lambda-z", type=float, default=0.0)
    ra.add_argument("--lambda-gate-binary", type=float, default=0.0)
    ra.add_argument("--gate-type", choices=["sigmoid", "hard_concrete"], default="sigmoid")
    ra.add_argument("--gate-temperature", type=float, default=1.0)
    ra.add_argument("--top-k", type=int, default=10)
    ra.add_argument("--gate-threshold", type=float, default=0.5)
    ra.add_argument("--noise-std", type=float, default=0.05)
    ra.add_argument("--max-samples", type=int, default=2000)
    ra.add_argument("--interp-batch-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print("SCENet repo looks runnable.")
        print("Environment:")
        for k, v in env_summary().items():
            print(f"- {k}: {v}")
        print(f"Default raw data root: {args.data_root}")
        print("Next: place datasets under data/raw/ or pass --path to commands.")
        return 0

    if args.command == "peek-data":
        ds = load_dataset(args.dataset, args.path)
        print(f"dataset: {ds.dataset_id.value}")
        print(f"X shape: {ds.X.shape}")
        print(f"y distribution: {ds.y.value_counts(dropna=False).to_dict()}")
        print(f"numeric cols: {len(ds.numeric_cols)}")
        print(f"categorical cols: {len(ds.categorical_cols)}")
        return 0

    if args.command == "train-baselines":
        set_seed(args.seed)
        _, nd = load_and_prepare_numpy(dataset=args.dataset, path=args.path, seed=args.seed)

        out_dir = ensure_dir(args.out)

        models = {
            "logreg": train_logreg(nd.X_train, nd.y_train, seed=args.seed),
            "mlp": train_mlp(nd.X_train, nd.y_train, seed=args.seed),
            "gbdt": train_gbdt(nd.X_train, nd.y_train, seed=args.seed),
        }

        if args.extra_baselines:
            for name, fn in [
                ("lightgbm", train_lightgbm),
                ("catboost", train_catboost),
            ]:
                try:
                    models[name] = fn(nd.X_train, nd.y_train, seed=args.seed)
                except ImportError:
                    # Optional dependency not installed.
                    pass

        results = {}
        for name, model in models.items():
            r = evaluate_baseline(
                name,
                model,
                X_val=nd.X_val,
                y_val=nd.y_val,
                X_test=nd.X_test,
                y_test=nd.y_test,
            )
            results[name] = {
                "val": r.val.__dict__,
                "test": r.test.__dict__,
                "threshold": r.threshold,
                "val_thresholded": r.val_thresholded.__dict__,
                "test_thresholded": r.test_thresholded.__dict__,
            }

        save_json(out_dir / f"{args.dataset}_seed{args.seed}.metrics.json", results)
        save_json(out_dir / f"{args.dataset}_seed{args.seed}.features.json", {
            "n_features": len(nd.prepared.feature_names),
            "feature_names": nd.prepared.feature_names,
        })

        print(f"Wrote metrics to: {out_dir}")
        return 0

    if args.command == "train-scenet":
        import torch

        set_seed(args.seed)
        _, nd = load_and_prepare_numpy(dataset=args.dataset, path=args.path, seed=args.seed)
        loaders = make_loaders(
            X_train=nd.X_train,
            y_train=nd.y_train,
            X_val=nd.X_val,
            y_val=nd.y_val,
            X_test=nd.X_test,
            y_test=nd.y_test,
            batch_size=args.batch_size,
        )

        model = SCENet(
            n_features=nd.X_train.shape[1],
            n_concepts=args.concepts,
            gate_type=args.gate_type,
            gate_temperature=args.gate_temperature,
        )
        cfg = TrainConfig(
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            lambda_g=args.lambda_g,
            lambda_w2=args.lambda_w2,
            lambda_z=args.lambda_z,
            lambda_gate_binary=args.lambda_gate_binary,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = train_scenet(model=model, loaders=loaders, device=device, cfg=cfg)

        out_dir = ensure_dir(args.out)
        save_json(out_dir / f"{args.dataset}_seed{args.seed}.metrics.json", {
            "val": result.val.__dict__,
            "test": result.test.__dict__,
            "threshold": result.threshold,
            "val_thresholded": result.val_thresholded.__dict__,
            "test_thresholded": result.test_thresholded.__dict__,
            "best_epoch": result.best_epoch,
            "device": str(device),
            "concepts": args.concepts,
            "seed": args.seed,
            "lambda_g": args.lambda_g,
            "lambda_w2": args.lambda_w2,
            "lambda_z": args.lambda_z,
            "lambda_gate_binary": args.lambda_gate_binary,
            "lr": args.lr,
            "gate_type": args.gate_type,
            "gate_temperature": args.gate_temperature,
        })

        # Save checkpoint + feature names for explanations.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": int(nd.X_train.shape[1]),
                "n_concepts": int(args.concepts),
                "feature_names": nd.prepared.feature_names,
                "threshold": float(result.threshold),
                "gate_type": str(args.gate_type),
                "gate_temperature": float(args.gate_temperature),
            },
            out_dir / f"{args.dataset}_seed{args.seed}.pt",
        )

        print(f"Wrote SCENet outputs to: {out_dir}")
        return 0

    if args.command == "explain-scenet":
        import torch

        _, nd = load_and_prepare_numpy(dataset=args.dataset, path=args.path, seed=args.seed)

        split_map = {
            "train": nd.X_train,
            "val": nd.X_val,
            "test": nd.X_test,
        }
        X = split_map[args.split]
        if not (0 <= args.index < X.shape[0]):
            raise SystemExit(f"index out of range for split '{args.split}': {args.index}")

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        n_features = int(ckpt.get("n_features", X.shape[1]))
        n_concepts = int(ckpt.get("n_concepts", 16))
        threshold = float(ckpt.get("threshold", 0.5))
        gate_type = str(ckpt.get("gate_type", "sigmoid"))
        gate_temperature = float(ckpt.get("gate_temperature", 1.0))
        feature_names = ckpt.get("feature_names", nd.prepared.feature_names)

        if n_features != X.shape[1]:
            raise SystemExit(
                f"Checkpoint expects n_features={n_features}, but preprocessed data has {X.shape[1]}. "
                "Make sure you used the same dataset + preprocessing seed."
            )

        model = SCENet(
            n_features=n_features,
            n_concepts=n_concepts,
            gate_type=gate_type,
            gate_temperature=gate_temperature,
        )
        model.load_state_dict(ckpt["state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        cfg = ExplainConfig(
            top_k_features=args.top_features,
            top_k_concepts=args.top_concepts,
            top_k_features_per_concept=args.top_features_per_concept,
        )

        explanation = explain_single(
            model=model,
            x=X[args.index],
            feature_names=list(feature_names),
            cfg=cfg,
            device=device,
            threshold=threshold,
        )
        explanation["dataset"] = args.dataset
        explanation["split"] = args.split
        explanation["index"] = args.index

        if args.out:
            save_json(args.out, explanation)
            print(f"Wrote explanation to: {args.out}")
        else:
            # Print as JSON-ish text (human + machine friendly)
            import json

            print(json.dumps(explanation, indent=2))
        return 0

    if args.command == "eval-interpretability":
        import torch

        _, nd = load_and_prepare_numpy(dataset=args.dataset, path=args.path, seed=args.seed)

        split_map_X = {
            "train": nd.X_train,
            "val": nd.X_val,
            "test": nd.X_test,
        }
        split_map_y = {
            "train": nd.y_train,
            "val": nd.y_val,
            "test": nd.y_test,
        }

        X = split_map_X[args.split]
        y = split_map_y[args.split]

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        n_features = int(ckpt.get("n_features", X.shape[1]))
        n_concepts = int(ckpt.get("n_concepts", 16))
        ckpt_threshold = float(ckpt.get("threshold", 0.5))
        gate_type = str(ckpt.get("gate_type", "sigmoid"))
        gate_temperature = float(ckpt.get("gate_temperature", 1.0))
        if n_features != X.shape[1]:
            raise SystemExit(
                f"Checkpoint expects n_features={n_features}, but preprocessed data has {X.shape[1]}. "
                "Make sure you used the same dataset + preprocessing seed."
            )

        model = SCENet(
            n_features=n_features,
            n_concepts=n_concepts,
            gate_type=gate_type,
            gate_temperature=gate_temperature,
        )
        model.load_state_dict(ckpt["state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        cfg = InterpretabilityConfig(
            top_k=args.top_k,
            gate_threshold=args.gate_threshold,
            threshold=float(ckpt_threshold if args.threshold is None else args.threshold),
            noise_std=args.noise_std,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        report = evaluate_interpretability(model=model, X=X, y=y, device=device, cfg=cfg)
        report.update(
            {
                "dataset": args.dataset,
                "split": args.split,
                "seed": args.seed,
                "checkpoint": args.checkpoint,
                "device": str(device),
                "n_features": int(n_features),
                "n_concepts": int(n_concepts),
            }
        )

        out_dir = ensure_dir(args.out)
        out_path = out_dir / f"{args.dataset}_seed{args.seed}_{args.split}.interpretability.json"
        save_json(out_path, report)
        print(f"Wrote interpretability report to: {out_path}")
        return 0

    if args.command == "summarize-concepts":
        import torch

        _, nd = load_and_prepare_numpy(dataset=args.dataset, path=args.path, seed=args.seed)

        split_map = {
            "train": nd.X_train,
            "val": nd.X_val,
            "test": nd.X_test,
        }
        X = split_map[args.split]

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        n_features = int(ckpt.get("n_features", X.shape[1]))
        n_concepts = int(ckpt.get("n_concepts", 16))
        gate_type = str(ckpt.get("gate_type", "sigmoid"))
        gate_temperature = float(ckpt.get("gate_temperature", 1.0))
        feature_names = ckpt.get("feature_names", nd.prepared.feature_names)

        if n_features != X.shape[1]:
            raise SystemExit(
                f"Checkpoint expects n_features={n_features}, but preprocessed data has {X.shape[1]}. "
                "Make sure you used the same dataset + preprocessing seed."
            )

        model = SCENet(
            n_features=n_features,
            n_concepts=n_concepts,
            gate_type=gate_type,
            gate_temperature=gate_temperature,
        )
        model.load_state_dict(ckpt["state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        cfg = ConceptSummaryConfig(
            top_k_groups=args.top_groups,
            top_k_features=args.top_features,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            seed=args.seed,
        )

        summary = summarize_concepts(
            model=model,
            X=X,
            feature_names=list(feature_names),
            device=device,
            cfg=cfg,
        )
        summary.update(
            {
                "dataset": args.dataset,
                "split": args.split,
                "seed": args.seed,
                "checkpoint": args.checkpoint,
                "gate_type": gate_type,
                "gate_temperature": gate_temperature,
            }
        )

        out_dir = ensure_dir(args.out)
        out_json = out_dir / f"{args.dataset}_seed{args.seed}_{args.split}.concepts.json"
        save_json(out_json, summary)
        plot_concept_heatmap(summary=summary, out_path=out_dir / f"{args.dataset}_seed{args.seed}_{args.split}.heatmap.png")
        print(f"Wrote concept summary to: {out_json}")
        return 0

    if args.command == "make-results-table":
        root = args.root
        out_dir = args.out or root
        df = write_results_table(
            ResultsTableConfig(
                root=Path(root),
                out_csv=Path(out_dir) / "results.csv",
                out_md=Path(out_dir) / "results.md",
            )
        )
        print(f"Wrote results table with {len(df)} rows to: {Path(out_dir) / 'results.csv'}")
        return 0

    if args.command == "run-multiseed":
        scenet_cfg = SCENetRunConfig(
            concepts=args.concepts,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            lambda_g=args.lambda_g,
            lambda_w2=args.lambda_w2,
            lambda_z=args.lambda_z,
            lambda_gate_binary=args.lambda_gate_binary,
            gate_type=args.gate_type,
            gate_temperature=args.gate_temperature,
        )
        baselines_cfg = BaselinesRunConfig(enabled=not args.skip_baselines, extra_baselines=bool(args.extra_baselines))
        extras = ExtrasConfig(
            with_interpretability=bool(args.with_interpretability),
            with_concepts=bool(args.with_concepts),
            interpretability=InterpretabilityConfig(
                top_k=args.top_k,
                gate_threshold=args.gate_threshold,
                threshold=0.5,
                noise_std=args.noise_std,
                max_samples=args.max_samples,
                batch_size=args.interp_batch_size,
                seed=42,
            ),
            concepts=ConceptSummaryConfig(
                top_k_groups=10,
                top_k_features=12,
                batch_size=512,
                max_samples=2000,
                seed=42,
            ),
        )
        run_dir = run_multiseed(
            dataset=args.dataset,
            path=args.path,
            seeds=list(args.seeds),
            out_dir=args.out,
            scenet_cfg=scenet_cfg,
            baselines_cfg=baselines_cfg,
            extras=extras,
            tag=args.tag or None,
        )
        print(f"Wrote multiseed run to: {run_dir}")
        return 0

    if args.command == "run-kfold":
        scenet_cfg = SCENetRunConfig(
            concepts=args.concepts,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            lambda_g=args.lambda_g,
            lambda_w2=args.lambda_w2,
            lambda_z=args.lambda_z,
            lambda_gate_binary=args.lambda_gate_binary,
            gate_type=args.gate_type,
            gate_temperature=args.gate_temperature,
        )
        baselines_cfg = BaselinesRunConfig(enabled=not args.skip_baselines, extra_baselines=bool(args.extra_baselines))
        extras = ExtrasConfig(
            with_interpretability=bool(args.with_interpretability),
            with_concepts=bool(args.with_concepts),
            interpretability=InterpretabilityConfig(
                top_k=args.top_k,
                gate_threshold=args.gate_threshold,
                threshold=0.5,
                noise_std=args.noise_std,
                max_samples=args.max_samples,
                batch_size=args.interp_batch_size,
                seed=args.seed,
            ),
            concepts=ConceptSummaryConfig(
                top_k_groups=10,
                top_k_features=12,
                batch_size=512,
                max_samples=2000,
                seed=args.seed,
            ),
        )
        run_dir = run_kfold(
            dataset=args.dataset,
            path=args.path,
            n_splits=args.k,
            seed=args.seed,
            out_dir=args.out,
            scenet_cfg=scenet_cfg,
            baselines_cfg=baselines_cfg,
            extras=extras,
            tag=args.tag or None,
        )
        print(f"Wrote k-fold run to: {run_dir}")
        return 0

    if args.command == "run-all":
        scenet_cfg = SCENetRunConfig(
            concepts=args.concepts,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            lambda_g=args.lambda_g,
            lambda_w2=args.lambda_w2,
            lambda_z=args.lambda_z,
            lambda_gate_binary=args.lambda_gate_binary,
            gate_type=args.gate_type,
            gate_temperature=args.gate_temperature,
        )
        baselines_cfg = BaselinesRunConfig(enabled=not args.skip_baselines, extra_baselines=bool(args.extra_baselines))
        extras = ExtrasConfig(
            with_interpretability=bool(args.with_interpretability),
            with_concepts=bool(args.with_concepts),
            interpretability=InterpretabilityConfig(
                top_k=args.top_k,
                gate_threshold=args.gate_threshold,
                threshold=0.5,
                noise_std=args.noise_std,
                max_samples=args.max_samples,
                batch_size=args.interp_batch_size,
                seed=args.seed,
            ),
            concepts=ConceptSummaryConfig(
                top_k_groups=10,
                top_k_features=12,
                batch_size=512,
                max_samples=2000,
                seed=args.seed,
            ),
        )
        run_dir = run_all(
            credit_path=args.credit_path,
            heart_path=args.heart_path,
            credit_seeds=list(args.credit_seeds),
            heart_kfold=args.heart_kfold,
            seed=args.seed,
            out_dir=args.out,
            scenet_cfg=scenet_cfg,
            baselines_cfg=baselines_cfg,
            extras=extras,
            tag=args.tag or None,
        )
        print(f"Wrote run-all outputs to: {run_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
