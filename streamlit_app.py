from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Workaround for a common Windows error when both PyTorch and scikit-learn
# load OpenMP runtimes in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class AppDefaults:
    dataset: str
    data_path: str
    checkpoint_path: str
    seed: int = 42


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_paths() -> dict[str, AppDefaults]:
    root = _repo_root()
    return {
        "heart_disease": AppDefaults(
            dataset="heart_disease",
            data_path=str((root / "datasets" / "heart_cleveland_upload.csv").as_posix()),
            checkpoint_path=str((root / "outputs" / "scenet" / "heart_disease_seed42.pt").as_posix()),
            seed=42,
        ),
        "credit_default": AppDefaults(
            dataset="credit_default",
            data_path=str((root / "datasets" / "UCI_Credit_Card.csv").as_posix()),
            checkpoint_path=str((root / "outputs" / "scenet" / "credit_default_seed42.pt").as_posix()),
            seed=42,
        ),
        "german_credit": AppDefaults(
            dataset="german_credit",
            data_path=str((root / "data" / "raw").as_posix()),
            checkpoint_path=str((root / "outputs" / "scenet" / "german_credit_seed42.pt").as_posix()),
            seed=42,
        ),
    }


def _to_abs_path(p: str) -> Path:
    root = _repo_root()
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@st.cache_resource(show_spinner=False)
def _load_numpy_data(dataset: str, data_path: str, seed: int):
    from scenet.pipeline import load_and_prepare_numpy

    ds, nd = load_and_prepare_numpy(dataset=dataset, path=data_path, seed=int(seed))
    return ds, nd


@st.cache_resource(show_spinner=False)
def _load_model_and_ckpt(checkpoint_path: str):
    import torch

    from scenet.models.scenet import SCENet

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    n_features = int(ckpt["n_features"])
    n_concepts = int(ckpt["n_concepts"])
    gate_type = str(ckpt.get("gate_type", "sigmoid"))
    gate_temperature = float(ckpt.get("gate_temperature", 1.0))

    model = SCENet(
        n_features=n_features,
        n_concepts=n_concepts,
        gate_type=gate_type,
        gate_temperature=gate_temperature,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt


def _available_checkpoints() -> list[str]:
    root = _repo_root()
    candidates = sorted((root / "outputs").rglob("*.pt"))
    return [c.as_posix() for c in candidates]


def _split_arrays(nd: Any, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    split = str(split)
    if split == "train":
        return nd.X_train, nd.y_train, nd.prepared.splits.X_train, nd.prepared.splits.y_train
    if split == "val":
        return nd.X_val, nd.y_val, nd.prepared.splits.X_val, nd.prepared.splits.y_val
    if split == "test":
        return nd.X_test, nd.y_test, nd.prepared.splits.X_test, nd.prepared.splits.y_test
    raise ValueError(f"Unknown split: {split}")


def _df_from_top_features(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    for col in ["importance", "gate", "selected", "x"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _df_from_concepts(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    for col in ["activation", "weight_to_output", "contribution_to_logit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    st.set_page_config(page_title="SCENet Demo", layout="wide")

    st.title("SCENet demo: features → concepts → contributions")

    st.markdown("""
    **SCENet (Sparse Concept Explainer Network)** is an interpretable neural network architecture that makes its decision-making process transparent. 
    It bridges the gap between complex raw data and human-understandable reasoning by introducing an intermediate "concept" layer.
    
    In a standard neural network, features are transformed into abstract representations that are hard to interpret. 
    **SCENet instead:**
    1. Selects the most important input features.
    2. Maps these features into high-level, human-understandable **concepts**. 
    3. Combines evidence from these concepts to make its final prediction.
    
    This demo lets you explore exactly how SCENet forms these concepts from raw features and how much each concept contributes to the final prediction for each sample.
    """)

    defaults = _default_paths()

    with st.sidebar:
        st.header("Inputs")

        dataset = st.selectbox(
            "Dataset",
            options=["heart_disease", "credit_default", "german_credit"],
            index=0,
        )

        seed = st.number_input("Preprocessing seed", min_value=0, max_value=10_000, value=int(defaults[dataset].seed))

        data_path = st.text_input("Dataset path (file or folder)", value=defaults[dataset].data_path)

        ckpt_candidates = [defaults[dataset].checkpoint_path]
        ckpt_candidates.extend([p for p in _available_checkpoints() if p not in ckpt_candidates])
        checkpoint_path = st.selectbox("Checkpoint (.pt)", options=ckpt_candidates)

        split = st.selectbox("Split", options=["test", "val", "train"], index=0)

    # Load dataset + preprocessor
    try:
        with st.spinner("Loading dataset + preprocessing..."):
            ds, nd = _load_numpy_data(dataset, data_path, int(seed))
    except Exception as e:
        st.error(f"Failed to load/prepare dataset: {e}")
        st.stop()

    # Load checkpoint + model
    ckpt_path_abs = _to_abs_path(checkpoint_path)
    if not ckpt_path_abs.exists():
        st.error(f"Checkpoint not found: {ckpt_path_abs.as_posix()}")
        st.stop()

    try:
        with st.spinner("Loading checkpoint..."):
            model, ckpt = _load_model_and_ckpt(str(ckpt_path_abs))
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    X_split, y_split, X_raw_split, y_raw_split = _split_arrays(nd, split)
    if int(ckpt.get("n_features", X_split.shape[1])) != int(X_split.shape[1]):
        st.error(
            "Checkpoint feature dimension does not match the preprocessed data. "
            "Make sure `dataset`, `data_path`, and `seed` match what was used in training."
        )
        st.stop()

    threshold = float(ckpt.get("threshold", 0.5))

    with st.sidebar:
        max_index = int(X_split.shape[0]) - 1
        index = st.slider("Sample index", min_value=0, max_value=max_index, value=0)
        threshold = st.number_input("Decision threshold", min_value=0.0, max_value=1.0, value=float(threshold))

    # Compute explanation
    import torch

    from scenet.explain import ExplainConfig, explain_single

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    explain_cfg = ExplainConfig(
        top_k_features=12,
        top_k_concepts=int(ckpt.get("n_concepts", 16)),
        top_k_features_per_concept=8,
    )

    with st.spinner("Computing explanation for the selected sample..."):
        explanation = explain_single(
            model=model,
            x=X_split[int(index)],
            feature_names=list(ckpt.get("feature_names", nd.prepared.feature_names)),
            cfg=explain_cfg,
            device=device,
            threshold=float(threshold),
        )

    # Compute dataset-level concept summary
    from scenet.concepts import ConceptSummaryConfig, summarize_concepts

    summary_cfg = ConceptSummaryConfig(top_k_groups=10, top_k_features=12, batch_size=512, max_samples=2000, seed=int(seed))

    with st.spinner("Computing dataset-level concept definitions..."):
        concept_summary = summarize_concepts(
            model=model,
            X=nd.X_train,
            feature_names=list(ckpt.get("feature_names", nd.prepared.feature_names)),
            device=device,
            cfg=summary_cfg,
        )

    # --- Layout ---
    col_left, col_mid, col_right = st.columns(3)

    # Left: raw features
    with col_left:
        st.subheader("1) Raw features")
        raw_row = X_raw_split.iloc[[int(index)]].copy()
        raw_row.insert(0, "target", [int(y_raw_split.iloc[int(index)])])
        st.dataframe(raw_row, width="stretch")

        st.caption("These are the original tabular features (pre one-hot / scaling).")

    # Middle: transformed feature importance for this sample
    with col_mid:
        st.subheader("2) Features used by the model")
        st.metric("Predicted probability", f"{explanation['probability']:.4f}")
        st.metric("Predicted label", int(explanation["predicted_label"]))

        tf = _df_from_top_features(explanation.get("top_features", []))
        st.write("Top transformed features (post-preprocessing):")
        st.dataframe(tf, width="stretch")

        tg = pd.DataFrame(explanation.get("top_feature_groups", []))
        st.write("Grouped by original feature (one-hot collapsed):")
        st.dataframe(tg, width="stretch")

    # Right: concepts + contributions for this sample
    with col_right:
        st.subheader("3) Concepts and their contributions")

        cdf = _df_from_concepts(explanation.get("concepts", []))
        if not cdf.empty:
            st.write("Per-concept activation and contribution to the output logit:")
            plot_df = cdf[["concept_index", "contribution_to_logit"]].set_index("concept_index")
            st.bar_chart(plot_df)
            st.dataframe(cdf, width="stretch")

            concept_ids = cdf["concept_index"].astype(int).tolist()
            selected_concept = st.selectbox("Inspect concept", options=concept_ids)

            # Find per-sample concept details
            sample_concept = None
            for c in explanation.get("concepts", []):
                if int(c.get("concept_index")) == int(selected_concept):
                    sample_concept = c
                    break

            if sample_concept is not None:
                st.write("Top contributors to this concept (linear part):")
                st.dataframe(pd.DataFrame(sample_concept.get("top_features", [])), width="stretch")
                st.write("Grouped contributors to this concept (sum of abs contributions):")
                st.dataframe(pd.DataFrame(sample_concept.get("top_feature_groups", [])), width="stretch")
        else:
            st.info("No concept details found in explanation output.")

    st.divider()

    # Concept definitions (dataset-level)
    st.subheader("Concept definitions: what each concept consists of")
    concepts = concept_summary.get("concepts", [])
    if not concepts:
        st.info("No concept summary available.")
        return

    concepts_df = pd.DataFrame(
        [{
            "concept_index": int(c.get("concept_index")),
            "weight_to_output": c.get("weight_to_output"),
        } for c in concepts]
    ).sort_values("concept_index")

    left2, right2 = st.columns(2)

    with left2:
        st.write("All concepts (click a row to inspect in the dropdown on the right):")
        st.dataframe(concepts_df, width="stretch")

    with right2:
        concept_idx = st.selectbox(
            "Concept to inspect",
            options=[int(c.get("concept_index")) for c in concepts],
            index=0,
            key="concept_definition_select",
        )

        chosen = None
        for c in concepts:
            if int(c.get("concept_index")) == int(concept_idx):
                chosen = c
                break

        if chosen is None:
            st.info("Concept not found in summary.")
        else:
            st.write("Top feature groups inside this concept:")
            st.dataframe(pd.DataFrame(chosen.get("top_feature_groups", [])), width="stretch")
            st.write("Top transformed features inside this concept:")
            st.dataframe(pd.DataFrame(chosen.get("top_features", [])), width="stretch")


if __name__ == "__main__":
    main()
