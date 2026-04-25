from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ResultsTableConfig:
    root: Path
    out_csv: Path
    out_md: Path
    include_thresholded: bool = True


def _escape_md(val: Any) -> str:
    if val is None:
        return ""
    s = str(val)
    return s.replace("|", "\\|")


def _to_markdown_simple(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(_escape_md(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_escape_md(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_ID_RE = re.compile(
    r"^(?P<dataset>.+)_seed(?P<seed>\d+)(?:_fold(?P<fold>\d+))?(?:_(?P<split>train|val|test))?$",
    re.IGNORECASE,
)


def _parse_ids_from_name(name: str) -> dict[str, Any]:
    base = name
    for suffix in [".metrics.json", ".interpretability.json", ".concepts.json"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    m = _ID_RE.match(base)
    if not m:
        return {"dataset": base, "seed": None, "fold": None, "split": None}

    d = m.groupdict()
    return {
        "dataset": d.get("dataset"),
        "seed": int(d["seed"]) if d.get("seed") else None,
        "fold": int(d["fold"]) if d.get("fold") else None,
        "split": d.get("split"),
    }


def _flatten_metrics(
    *,
    dataset: str,
    model: str,
    seed: int | None,
    fold: int | None,
    split: str,
    threshold_type: str,
    threshold: float | None,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "fold": fold,
        "split": split,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "auc_roc": metrics.get("auc_roc"),
    }
    if extra:
        row.update(extra)
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Metrics files
    for path in root.rglob("*.metrics.json"):
        data = _load_json(path)
        ids = _parse_ids_from_name(path.name)
        dataset = ids["dataset"]
        seed = ids["seed"]
        fold = ids["fold"]

        # Baselines format: {model_name: {...}}
        if isinstance(data, dict) and "val" not in data and "test" not in data:
            for model_name, payload in data.items():
                if not isinstance(payload, dict) or "test" not in payload:
                    continue
                thr = payload.get("threshold")
                rows.append(
                    _flatten_metrics(
                        dataset=dataset,
                        model=str(model_name),
                        seed=seed,
                        fold=fold,
                        split="test",
                        threshold_type="default",
                        threshold=0.5,
                        metrics=payload.get("test", {}),
                    )
                )

                if "test_thresholded" in payload:
                    rows.append(
                        _flatten_metrics(
                            dataset=dataset,
                            model=str(model_name),
                            seed=seed,
                            fold=fold,
                            split="test",
                            threshold_type="val_max_f1",
                            threshold=thr,
                            metrics=payload.get("test_thresholded", {}),
                        )
                    )
            continue

        # SCENet format
        if isinstance(data, dict) and "test" in data:
            thr = data.get("threshold")
            extra = {
                "concepts": data.get("concepts"),
                "lambda_g": data.get("lambda_g"),
                "lambda_w2": data.get("lambda_w2"),
                "lambda_z": data.get("lambda_z"),
                "lambda_gate_binary": data.get("lambda_gate_binary"),
                "gate_type": data.get("gate_type"),
                "gate_temperature": data.get("gate_temperature"),
            }

            rows.append(
                _flatten_metrics(
                    dataset=dataset,
                    model="scenet",
                    seed=seed,
                    fold=fold,
                    split="test",
                    threshold_type="default",
                    threshold=0.5,
                    metrics=data.get("test", {}),
                    extra=extra,
                )
            )

            if "test_thresholded" in data:
                rows.append(
                    _flatten_metrics(
                        dataset=dataset,
                        model="scenet",
                        seed=seed,
                        fold=fold,
                        split="test",
                        threshold_type="val_max_f1",
                        threshold=thr,
                        metrics=data.get("test_thresholded", {}),
                        extra=extra,
                    )
                )

    # Interpretability files (join onto SCENet rows by dataset+seed+fold+split)
    interp_map: dict[tuple[str, int | None, int | None, str], dict[str, Any]] = {}
    for path in root.rglob("*.interpretability.json"):
        data = _load_json(path)
        ids = _parse_ids_from_name(path.name)
        dataset = ids["dataset"]
        seed = ids["seed"]
        fold = ids["fold"]
        split = ids.get("split") or "test"

        sparsity = data.get("sparsity", {})
        active = (sparsity.get("active_features") or {})

        interp_map[(dataset, seed, fold, split)] = {
            "sparsity_mean_active": active.get("mean"),
            "sparsity_median_active": active.get("median"),
            "stability_jaccard_mean": (data.get("stability", {}) or {}).get("jaccard_mean"),
            "consistency_jaccard_mean": (data.get("consistency", {}) or {}).get("jaccard_mean"),
        }

    for r in rows:
        if r.get("model") != "scenet":
            continue
        key = (r.get("dataset"), r.get("seed"), r.get("fold"), r.get("split"))
        extra = interp_map.get(key)
        if extra:
            r.update(extra)

    return rows


def write_results_table(cfg: ResultsTableConfig) -> pd.DataFrame:
    root = Path(cfg.root)
    rows = collect_rows(root)
    df = pd.DataFrame(rows)

    # Deterministic sort for stable tables.
    sort_cols = [c for c in ["dataset", "model", "seed", "fold", "threshold_type"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out_csv, index=False)

    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_md.open("w", encoding="utf-8") as f:
        try:
            f.write(df.to_markdown(index=False))
            f.write("\n")
        except Exception:
            f.write(_to_markdown_simple(df))

    return df
