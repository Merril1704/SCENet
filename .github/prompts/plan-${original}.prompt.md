## Plan: Implement SCENet in staged milestones

SCENet is an intrinsically-interpretable neural network for high-stakes tabular prediction: it forces predictions to flow through a small “concept bottleneck” so each decision can be explained via (1) selected features, (2) concept activations, and (3) how concepts drive the final output. The repo currently contains only the spec docs, so the plan starts by scaffolding a minimal, reproducible ML codebase, then iterates from baselines → SCENet core → explanation outputs → interpretability evaluation.

**Steps**
1. Stage 0 — Scope + success criteria (0.5–1 day)
   - Lock the first target task as supervised tabular classification (binary by default) and define: target label, positive class meaning, and “high-stakes” framing.
   - Choose 1 primary dataset for development (fast iteration) and 1 secondary dataset for generalization testing.
   - Define what “done” means for v1: competitive baseline performance + stable structured explanations + a short report.

2. Stage 1 — Dataset + preprocessing pipeline (1–2 days)
   - Implement a single, versioned data ingestion path that outputs:
     - X (numeric matrix), y (labels), feature_names, and metadata about categorical/numeric columns.
   - Preprocessing requirements:
     - Missing values (impute), categorical encoding (one-hot or ordinal), numeric scaling (standardization).
     - Reproducible split strategy (train/val/test or stratified k-fold).
     - Leakage prevention (fit preprocessing on train only; apply to val/test).
   - Deliverable: a “dataset card” for the chosen dataset(s): source, task type, target definition, features, preprocessing decisions.

3. Stage 2 — Baselines (1–2 days)
   - Train and evaluate baselines to establish an anchor:
     - Logistic Regression (interpretable linear baseline)
     - MLP (black-box NN baseline)
     - Gradient boosting baseline (sklearn GBDT or XGBoost/LightGBM if you’re ok adding deps)
   - Report predictive metrics: Accuracy, F1, AUC-ROC (as listed in the spec).
   - Deliverable: baseline results table + saved model artifacts + reproducible run command.

4. Stage 3 — SCENet v0 (core architecture) (2–4 days)
   - Implement SCENet as: Input → Feature Selector → Concept Layer → Output Layer.
   - Recommended concrete design choices (simple, faithful to the doc, and explanation-friendly):
     - Feature selector as per-feature gate (sample-dependent but local):
       - g_i = sigmoid(a_i * x_i + b_i)
       - z_i = x_i * g_i
       - Regularize sparsity via L1 on g (encourages few active features per prediction).
     - Concept layer as small hidden layer (m concepts, e.g., 8–32):
       - c = activation(W2 z + b2)
       - Add L1 on W2 to encourage sparse feature→concept connections.
     - Output layer shallow/linear:
       - binary: y_hat = sigmoid(w3^T c + b3)
       - multi-class: softmax(W3 c + b3)
   - Train with standard loss (cross-entropy) + regularizers:
     - L = CE(y_hat, y) + λ_g ||g||_1 + λ_2 ||W2||_1 (+ optional λ_w ||W||_2^2)
   - Deliverable: SCENet trains end-to-end and matches/approaches MLP on the primary dataset.

5. Stage 4 — Explanation module (structured, intrinsic) (1–2 days)
   - Add an “explain(x)” routine that returns (per sample):
     - Prediction (class/probability)
     - Top feature contributions
     - Concept activations
     - Concept→output contributions (“decision pathway”)
   - Practical computation (fast and faithful to the architecture):
     - Feature importance (per sample): rank by |z_i| or |g_i * x_i|.
     - Concept activation: c_j values.
     - Concept contribution to output:
       - binary: contribution_j = w3_j * c_j
       - multi-class: per-class contributions from W3.
     - Feature→concept contributions: contrib_{j,i} = W2_{j,i} * z_i (use absolute value for ranking).
   - Deliverable: a consistent JSON-like explanation schema and a small CLI/notebook demo producing the example output format in the doc.

6. Stage 5 — Interpretability evaluation (2–4 days)
   - Implement the spec’s interpretability metrics and compare against baselines + post-hoc (optional):
     - Sparsity: average number of “active” features per prediction (e.g., count(g_i > τ)).
     - Stability: perturb inputs slightly (Gaussian noise / small % feature jitter) and measure explanation similarity (e.g., Jaccard@k for top-k features, rank correlation).
     - Faithfulness: remove/zero the top-k important features and measure performance drop.
     - Consistency: for similar inputs (nearest neighbors), explanations should be similar (same similarity metrics as stability).
   - Deliverable: a metrics report + plots showing tradeoff curves (performance vs sparsity).

7. Stage 6 — Results packaging (1–3 days)
   - Produce:
     - Experimental results table across datasets
     - Visualization: feature importance distribution, concept activation heatmaps, per-concept top features
     - Short “paper-style” report tying methodology to outcomes
   - Deliverable: final report + demo script that runs one sample end-to-end.

**Dataset needed (recommended)**
- Minimum dataset characteristics for SCENet:
  - Tabular supervised dataset with a clear label (classification recommended for v1)
  - Feature names available (critical for explanations)
  - Enough samples to validate stability (hundreds+; thousands ideal)
  - Mix of numerical and categorical features is fine (categoricals will be encoded)
- Recommended primary dataset (best single choice for a strong, realistic evaluation):
  - UCI “Default of Credit Card Clients (Taiwan)” / “Credit Card Default”
  - Why: ~30k samples (supports stability/faithfulness testing), credit-risk framing (high-stakes), and straightforward tabular preprocessing.
  - Important preprocessing detail: several columns are integer-coded categories (don’t treat them as continuous).
- Secondary dataset (optional but improves the report):
  - German Credit (small/fast iteration) or Heart Disease (healthcare domain shift).

**Requirements**
- Software
  - Python 3.10+ (recommended)
  - Core libs: numpy, pandas, scikit-learn, PyTorch (or TensorFlow/Keras), matplotlib/seaborn
  - Optional: xgboost/lightgbm (if you want strong gradient boosting baselines)
- Hardware
  - CPU-only is sufficient for the suggested datasets; a GPU is optional.
- Reproducibility
  - Fixed random seeds, saved configs per run, and a single “run experiment” entrypoint.

**Methodology (and why)**
- Engineering methodology: incremental milestones with verifiable outputs.
  - Why: the repo is currently docs-only; staged deliverables prevent getting stuck in model tuning without a reliable data/eval pipeline.
- Modeling methodology: concept bottleneck + sparsity regularization.
  - Why: forcing prediction to flow through a small concept vector makes explanations intrinsic; sparsity makes “top features” and “decision pathways” stable and auditable.
- Evaluation methodology: measure both predictive performance and interpretability properties.
  - Why: in high-stakes tabular systems, accuracy alone is insufficient; stability and faithfulness help detect brittle or misleading explanations.

**Relevant files**
- docs/context/project_explanation.md — source of architecture, metrics, and deliverables
- README.md — project title

**Verification**
1. Data pipeline verification
   - Preprocessing fit only on train; same transform applied to val/test; feature_names preserved.
2. Baseline verification
   - Baselines produce metrics on the same splits with consistent preprocessing.
3. SCENet verification
   - Training converges; explanations are produced for arbitrary inputs; explanation schema stable.
4. Interpretability verification
   - Sparsity improves as λ_g increases; stability doesn’t collapse under tiny perturbations; faithfulness shows measurable performance drop when removing top features.

**Decisions (defaults unless you choose otherwise)**
- Start with binary classification and PyTorch.
- Use a small number of concepts (8–32) to keep explanations readable.
- Use L1-driven sparsity to enable “active features per prediction.”

**Further considerations**
1. Concept semantics
   - Option A: unsupervised concepts (name concepts post-hoc by their top contributing features; simplest)
   - Option B: weak supervision (if you can define concept labels; better alignment but needs extra data)
2. Feature selector design
   - Option A: per-feature local gates (recommended for simplicity and auditability)
   - Option B: full linear selector W1 x (matches the doc equation but less directly per-sample interpretable unless constrained)
