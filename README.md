# CRISPR-Cas9 sgRNA Activity Prediction

In-silico pipeline for predicting CRISPR-Cas9 single-guide RNA (sgRNA) activity
from 30-mer DNA target sequences. Developed as an undergraduate research
project (skripsi) with a strong emphasis on methodological rigor and
reproducibility.

## Overview

The pipeline trains three regression models and evaluates them against an
independent external test set:

- **CNN (primary model)** — CRISPRpred-style parallel-convolution network over
  the one-hot encoded 30-mer.
- **XGBoost (baseline)** — gradient boosting on 197 engineered features.
- **Random Forest (baseline)** — ensemble on the same 197 engineered features.

## Sequence geometry (fixed)

    5' context  |   guide (20 bp)  | PAM  | 3' context
    [0:4]       |   [4:24]         | [24:27] NGG | [27:30]

## Methodology

- **Datasets** — DeepSpCas9 (train/validation; 10,117 valid sequences after a
  homopolymer filter) and Moreno-Mateos (independent external test set; 810
  valid). Moreno-Mateos is **never** used for training, tuning, early stopping
  or model selection.
- **Split** — DeepSpCas9 85/15 train/validation, fixed seed 42. Validation is
  held out from gradient updates; the **CNN** uses it for early stopping /
  model selection (D-006), the baselines only for evaluation (D-005).
- **Features** — 197 tabular features: GC content, nucleotide composition,
  k-mer counts (k=2,3) with entropy/complexity, and per-position guide
  one-hot. CNN input is a (n, 30, 4) one-hot matrix.
- **Primary metrics** — MAE, RMSE, R², Pearson, Spearman. MAPE is documented
  as unstable near zero and is not reported as a primary metric (D-007).
- **Reproducibility** — fixed seeds (42); CNN training is deterministic run to
  run; all experiments, decisions and metrics are recorded under
  `docs/` and `results/experiments/`.
- Decision log: `docs/decisions.md` (D-001 … D-007).

## Repository layout

    config.yaml                    # single source of truth (data, split, models)
    src/
      bioinformatics/              # sequence validation + feature extraction
      data/                        # data loading, validation, preprocessing
      models/                      # RandomForestModel, XGBoostModel, CNNModel
      evaluation/                  # metrics, comparison (paired tests, bootstrap CI)
      ablation/                    # sequence-region ablation (Phase 7)
      interpretability/            # CNN attribution + tabular importance (Phase 8)
    scripts/                       # one entry-point per pipeline phase
    notebooks/                     # exploratory notebooks
    tests/                         # unit tests (169 passing)
    docs/                          # phase reports + decisions log
    results/experiments/           # reproducible experiment JSONs
    models/                        # saved model artifacts (gitignored)

## Pipeline phases

| Phase | Deliverable | Key module / artifact |
|-------|-------------|-----------------------|
| 1 | Data preparation & validation | `scripts/prepare_data.py`, `docs/dataset_report.md` |
| 2 | Feature extraction | `src/bioinformatics/` |
| 3 | Random Forest baseline | `scripts/train_random_forest.py` |
| 4 | XGBoost baseline | `scripts/train_xgboost.py` |
| 5 | CNN (primary model) | `scripts/train_cnn.py`, `src/models/cnn.py` |
| 6 | Evaluation & statistical comparison | `scripts/evaluate_models_phase6.py` |
| 7 | Sequence-region ablation | `scripts/run_ablation_region_phase7.py` |
| 8 | Interpretability | `scripts/run_interpretability_phase8.py` |

## Canonical results (Moreno-Mateos external test, n = 810)

| Metric | CNN | XGBoost | RandomForest |
|--------|-------|---------|--------------|
| MAE | 0.2518 | 0.2540 | **0.2485** |
| RMSE | 0.2958 | 0.2977 | **0.2874** |
| R² | -0.0111 | -0.0240 | **0.0456** |
| Pearson r | 0.1838 | 0.1833 | **0.2352** |
| Spearman ρ | 0.1602 | 0.1772 | **0.2263** |

Paired significance tests (per-sample squared error) on the test set:
Random Forest significantly outperforms XGBoost (t p ≈ 3e-5) and the CNN
(t p ≈ 4e-3, borderline after Bonferroni); no significant XGBoost-vs-CNN
difference. Bootstrap 95% CIs place Random Forest's R² entirely above zero;
all other intervals straddle zero. Cross-dataset generalization is weak for
every model (top-5 precision 0 out-of-domain) — a central finding discussed in
`docs/phase6_evaluation_report.md`.

**Dominant signal location:** CNN attribution and the Phase 7 region ablation
agree the 20 bp guide `[4:24]` carries ~70% of the predictive signal;
`guide_pos_19` (adjacent to the PAM) is the top feature in both tabular models.

## Usage

```bash
pip install -r requirements.txt

python -m pytest                  # run the test suite (169 tests)

python scripts/train_cnn.py       # train / reproduce the primary CNN
python scripts/evaluate_models_phase6.py     # consolidated evaluation
python scripts/run_ablation_region_phase7.py # region ablation
python scripts/run_interpretability_phase8.py # attribution analysis
```

Model artifacts are gitignored (regenerable); every experiment JSON is
committed for reproducibility.

## Reports

- `docs/phase3_rf_report.md`, `docs/phase4_xgboost_report.md`,
  `docs/phase5_cnn_report.md` — per-model training and evaluation
- `docs/phase6_evaluation_report.md` — cross-model statistical comparison
- `docs/phase7_region_ablation_report.md` — sequence-region importance
- `docs/phase8_interpretability_report.md` — attribution and feature importance
- `docs/decisions.md` — methodology decisions D-001 … D-007