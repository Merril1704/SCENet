<p align="center">
  <h1 align="center">🧠 SCENet</h1>
  <p align="center">
    <strong>Structured Concept-Embedded Network for Intrinsic Explainability in High-Stakes Tabular Decision Systems</strong>
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#cli-reference">CLI Reference</a> •
    <a href="#interactive-demo">Interactive Demo</a> •
    <a href="#results">Results</a>
  </p>
</p>

---

## Overview

Deep learning models deliver state-of-the-art accuracy but remain **black boxes** — a critical limitation in regulated domains like healthcare, finance, and criminal justice. Post-hoc explanation methods (SHAP, LIME) are approximate, unstable, and fragile under perturbation.

**SCENet** solves this by making explainability **intrinsic**. Every prediction comes with an exact, mathematically verifiable explanation — not an after-the-fact approximation.

### Key Contributions

| Contribution | Description |
|---|---|
| **Per-Feature Local Gating** | Sigmoid or Hard-Concrete $L_0$ gates that dynamically silence irrelevant features per-instance |
| **Concept Bottleneck** | Compresses gated features into a small set of interpretable high-level abstractions |
| **Linear Output Layer** | Enables exact decomposition of every prediction into concept contributions |
| **4-Metric Evaluation** | Rigorous quantitative interpretability: Sparsity, Stability, Faithfulness, Consistency |

---

## Architecture

```
Input x ∈ ℝᵈ
    │
    ▼
┌──────────────────────┐
│  Per-Feature Gates   │   g_i = σ((a_i · x_i + b_i) / τ)
│  g ∈ [0,1]ᵈ         │   Learns which features matter per-instance
└──────────┬───────────┘
           │ z = x ⊙ g  (element-wise masking)
           ▼
┌──────────────────────┐
│  Concept Bottleneck  │   c = φ(W_c · z + b_c) ∈ ℝᴷ
│  K concepts (K ≪ d)  │   Groups features into interpretable abstractions
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Linear Output       │   ŷ = σ(w_o^T · c + b_o)
│  Exact decomposition │   Contribution of concept k = w_k · c_k
└──────────────────────┘
```

The three-stage pipeline produces **hierarchical explanations**:
1. **Feature level** — Which input features are active? (gates)
2. **Concept level** — How do features group into abstractions? (bottleneck weights)
3. **Decision level** — How much does each concept push the prediction? (output weights)

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.1+

### Installation

```bash
# Clone the repository
git clone https://github.com/Merril1704/SCENet.git
cd SCENet

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Verify Setup

```bash
python -m scenet doctor
```

### Dataset Setup

Place datasets in the `datasets/` folder:

| Dataset | Source | File |
|---|---|---|
| **Credit Default (Taiwan)** | [UCI / Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) | `UCI_Credit_Card.csv` |
| **Cleveland Heart Disease** | [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) | `heart_cleveland_upload.csv` |

---

## CLI Reference

SCENet ships with a comprehensive CLI. All commands are invoked via `python -m scenet <command>`.

### Core Workflow

```bash
# 1. Peek at a dataset
python -m scenet peek-data --dataset heart_disease --path datasets/heart_cleveland_upload.csv

# 2. Train baseline models (Logistic Regression, MLP, GBDT)
python -m scenet train-baselines --dataset heart_disease --path datasets/heart_cleveland_upload.csv

# 3. Train SCENet
python -m scenet train-scenet --dataset heart_disease --path datasets/heart_cleveland_upload.csv \
    --concepts 16 --epochs 50 --gate-type sigmoid

# 4. Explain a single prediction
python -m scenet explain-scenet --dataset heart_disease --path datasets/heart_cleveland_upload.csv \
    --checkpoint outputs/scenet/heart_disease_seed42.pt --index 0

# 5. Evaluate interpretability metrics
python -m scenet eval-interpretability --dataset heart_disease --path datasets/heart_cleveland_upload.csv \
    --checkpoint outputs/scenet/heart_disease_seed42.pt

# 6. Summarize learned concepts
python -m scenet summarize-concepts --dataset heart_disease --path datasets/heart_cleveland_upload.csv \
    --checkpoint outputs/scenet/heart_disease_seed42.pt
```

### Experiment Runners

```bash
# Multi-seed evaluation (recommended for Credit Default)
python -m scenet run-multiseed --dataset credit_default \
    --path datasets/UCI_Credit_Card.csv \
    --seeds 0 1 2 3 4 --with-interpretability --with-concepts

# K-Fold cross-validation (recommended for Heart Disease)
python -m scenet run-kfold --dataset heart_disease \
    --path datasets/heart_cleveland_upload.csv \
    --k 5 --with-interpretability --with-concepts

# Run everything at once
python -m scenet run-all \
    --credit-path datasets/UCI_Credit_Card.csv \
    --heart-path datasets/heart_cleveland_upload.csv \
    --credit-seeds 0 1 2 3 4 --heart-kfold 5 \
    --with-interpretability --with-concepts

# Generate results table
python -m scenet make-results-table --root outputs
```

### All Commands

| Command | Description |
|---|---|
| `doctor` | Check environment and print data path hints |
| `peek-data` | Load a dataset and print schema |
| `train-baselines` | Train LogReg, MLP, GBDT baselines |
| `train-scenet` | Train SCENet and save checkpoint |
| `explain-scenet` | Generate per-sample explanation |
| `eval-interpretability` | Compute sparsity, stability, faithfulness, consistency |
| `summarize-concepts` | Summarize learned concepts with heatmaps |
| `make-results-table` | Generate CSV/Markdown results table |
| `run-multiseed` | Run baselines + SCENet across multiple seeds |
| `run-kfold` | Run stratified K-fold cross-validation |
| `run-all` | Run full benchmark (credit multi-seed + heart k-fold) |

---

## Interactive Demo

SCENet includes a Streamlit web app for interactive exploration of predictions, feature gates, and concept contributions.

```bash
streamlit run streamlit_app.py
```

The demo provides:
- **Raw feature inspection** — View original tabular features for any sample
- **Feature importance** — See which transformed features the model uses and their gate values
- **Concept explorer** — Drill into each learned concept to see its activation, contribution to the output, and constituent features
- **Dataset-level concept definitions** — Understand what each concept represents across the full training set

---

## Results

### Predictive Performance (AUC-ROC)

| Model | Credit Default | Heart Disease |
|---|:---:|:---:|
| GBDT (HistGradient) | 0.781 | 0.881 |
| MLP (128-64) | 0.772 | 0.899 |
| Logistic Regression | 0.764 | 0.913 |
| **SCENet (Ours)** | **0.776** | **0.868** |

> SCENet trails the best black-box model by at most **0.005–0.045 AUC-ROC** — a minimal cost for exact, stable explanations.

### Interpretability Metrics

| Metric | Credit Default | Heart Disease |
|---|:---:|:---:|
| Stability (Jaccard, k=10) | **0.93** | **0.95** |
| Faithfulness (ΔAUC) | **−0.165** | **−0.294** |
| Sparsity (mean active) | ~68 / 91 | ~16 / 28 |
| Consistency (Jaccard, k=10) | 0.76 | 0.55 |

- **Stability**: Adding noise (σ=0.05) changes fewer than 1 in 10 top features — far more robust than SHAP/LIME
- **Faithfulness**: Masking top-10 features devastates predictions, proving explanations reflect true model reliance
- **Sparsity**: Up to **43% feature reduction** natively within the forward pass

---

## Project Structure

```
SCENet/
├── scenet/                     # Core Python package
│   ├── models/
│   │   └── scenet.py           # SCENet architecture (114 lines)
│   ├── cli.py                  # Full CLI interface
│   ├── preprocessing.py        # Imputation, scaling, one-hot encoding
│   ├── torch_train.py          # Training loop with early stopping
│   ├── explain.py              # Per-sample explanation generation
│   ├── interpretability.py     # 4-metric interpretability evaluation
│   ├── concepts.py             # Concept summarization & heatmaps
│   ├── baselines.py            # LogReg, MLP, GBDT baselines
│   ├── experiments.py          # Multi-seed & K-fold runners
│   └── results_table.py        # Results aggregation
├── streamlit_app.py            # Interactive web demo
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Hyperparameters

| Parameter | Default | Description |
|---|:---:|---|
| `--concepts` | 16 | Number of concept bottleneck dimensions |
| `--gate-type` | sigmoid | Gate mechanism (`sigmoid` or `hard_concrete`) |
| `--gate-temperature` | 1.0 | Gate sharpness (lower → more binary) |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--lambda-g` | 1e-3 | Gate sparsity penalty |
| `--lambda-w2` | 1e-4 | Concept weight L₁ penalty |
| `--lambda-z` | 0.0 | Selected feature L₁ penalty |
| `--lambda-gate-binary` | 0.0 | Binary gate encouragement penalty |
| `--epochs` | 50 | Max training epochs |
| `--patience` | 8 | Early stopping patience |

---

## License

This project is for academic and research purposes.

---

<p align="center">
  <sub>Built by <a href="https://github.com/Merril1704">Merril Baiju</a></sub>
</p>
