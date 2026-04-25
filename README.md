# SCENet
Intrinsic Explainability in Neural Networks for High-Stakes Tabular Decision Systems

This repo implements SCENet (Structured Concept-Embedded Network) for tabular prediction with intrinsic explanations.

## Setup (Conda)
```bash
conda create -n scenet python=3.11
conda activate scenet
pip install -r requirements.txt
```

## Datasets
This project expects you to keep raw datasets locally (not committed to git).

Recommended datasets:
- Credit Default (UCI “Default of Credit Card Clients (Taiwan)”) — primary
- German Credit (UCI) — secondary
- Cleveland Heart Disease (UCI) — secondary

You can pass a file path directly via `--path` (recommended), or place datasets under `data/raw/`.

## Commands

Sanity check:
```bash
python -m scenet doctor
```

Peek dataset schema:
```bash
python -m scenet peek-data --dataset credit_default --path "PATH_TO_UCI_EXCEL_FILE_OR_FOLDER"
python -m scenet peek-data --dataset german_credit --path "PATH_TO_GERMAN_DATA_FILE_OR_FOLDER"
python -m scenet peek-data --dataset heart_disease --path "PATH_TO_HEART_CSV_OR_FOLDER"
```

Train baselines (writes to `outputs/baselines/`):
```bash
python -m scenet train-baselines --dataset credit_default --path "PATH" --seed 42
```

Optional modern baselines (runs LightGBM/CatBoost if installed):
```bash
python -m scenet train-baselines --dataset credit_default --path "PATH" --seed 42 --extra-baselines
```

Train SCENet (writes metrics + checkpoint to `outputs/scenet/`):
```bash
python -m scenet train-scenet --dataset credit_default --path "PATH" --seed 42 --concepts 16
```

Notes:
- Metrics are reported twice: default threshold (0.5) and a validation-chosen threshold that maximizes F1.
- For more sparse explanations, try increasing `--lambda-g`, adding `--lambda-z`, lowering `--gate-temperature`,
  or using `--gate-type hard_concrete`.

Explain a single sample (prints JSON to stdout unless `--out` is provided):
```bash
python -m scenet explain-scenet \
	--dataset credit_default \
	--path "PATH" \
	--checkpoint "outputs/scenet/credit_default_seed42.pt" \
	--split test \
	--index 0
```

Summarize learned concepts + write a heatmap plot (writes to `outputs/concepts/`):
```bash
python -m scenet summarize-concepts \
	--dataset credit_default \
	--path "PATH" \
	--checkpoint "outputs/scenet/credit_default_seed42.pt" \
	--seed 42 \
	--split train
```

Interpretability metrics report (writes to `outputs/interpretability/`):
```bash
python -m scenet eval-interpretability \
	--dataset credit_default \
	--path "PATH" \
	--checkpoint "outputs/scenet/credit_default_seed42.pt" \
	--split test \
	--seed 42
```

## Experiment runners

Multi-seed evaluation (writes a self-contained run folder under `outputs/experiments/`):
```bash
python -m scenet run-multiseed \
	--dataset credit_default \
	--path "PATH" \
	--seeds 0 1 2 3 4 \
	--with-interpretability \
	--with-concepts
```

K-fold CV (recommended for Heart):
```bash
python -m scenet run-kfold \
	--dataset heart_disease \
	--path "PATH" \
	--k 5 \
	--seed 42
```

Run the default protocol in one command (credit multi-seed + heart k-fold):
```bash
python -m scenet run-all \
	--credit-path "datasets/UCI_Credit_Card.csv" \
	--heart-path "datasets/heart_cleveland_upload.csv" \
	--credit-seeds 0 1 2 3 4 \
	--heart-kfold 5 \
	--seed 42
```

Generate a results table (CSV + Markdown) from any outputs folder:
```bash
python -m scenet make-results-table --root outputs --out outputs
```

## Streamlit demo app

Run an interactive demo (features → concepts → contributions):
```bash
conda activate scenet
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Defaults:
- Datasets load from `datasets/` (you can change the path in the sidebar).
- Checkpoints load from `outputs/scenet/` if present.
