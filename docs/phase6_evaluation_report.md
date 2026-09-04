# Phase 6 Report - Cross-Model Evaluation and Comparison

## Overview

This phase consolidates the three trained models — Random Forest and XGBoost
(baselines, `config.yaml` fixed hyperparameters) and the CNN (primary model) —
into a single rigorous evaluation. No retraining and no tuning are performed:
the saved canonical artifacts are loaded and their predictions are compared on
the same fixed split (seed 42) and on the same independent external test set.
MAPE is deliberately not used as a primary metric (see D-007).

## Methodology

- **Models / artifacts**
  - Random Forest: `models/rf_baseline_fixed_20260905_001107.pkl`
    (n=200, depth=20, min_split=5, min_leaf=2, sqrt, seed 42)
  - XGBoost: `models/xgboost_baseline_20260905_002839.pkl`
    (n=100, depth=6, lr=0.1, seed 42, no early stopping per D-005)
  - CNN: `models/cnn_baseline_20260905_011720.pt`
    (kernels 5/7/9, 64 filters, dense 64, dropout 0.3, lr 1e-3, batch 32,
    adam, mse, early stopping patience 10)
- **Validation split:** DeepSpCas9 15% (n = 1,518), seed 42 — held-out from
  gradient fitting for all models. **Caveat (D-006):** the CNN used this split
  for early stopping / model selection; in-domain significance tests are
  therefore NOT a fair cross-model comparison and are reported only
  descriptively.
- **Test set:** Moreno-Mateos (n = 810) — completely untouched by all models;
  this is the **fair basis for cross-model significance tests**.
- Sequence geometry and the 197 tabular features / one-hot encoding are
  identical to the training pipelines (guide `[4:24]`, PAM `[24:27]`).

## Scalar Results

### Validation (in-domain, n = 1,518) — descriptive only (D-006 caveat)

| Metric | CNN | XGBoost | RandomForest |
|--------|---------|---------|--------------|
| MAE | 0.1385 | **0.1263** | 0.1499 |
| RMSE | 0.1709 | **0.1564** | 0.1787 |
| R² | 0.4164 | **0.5112** | 0.3619 |
| Pearson r | 0.6468 | **0.7173** | 0.6375 |
| Spearman ρ | 0.6313 | **0.7051** | 0.6262 |
| Kendall τ | 0.4519 | **0.5144** | 0.4477 |

XGBoost had the highest in-domain performance; these numbers describe relative
performance but must not be treated as a fair test because the CNN used this
split for model selection.

### Test (Moreno-Mateos, n = 810) — fair for all models

| Metric | CNN | XGBoost | RandomForest |
|--------|---------|---------|--------------|
| MAE | 0.2518 | 0.2540 | **0.2485** |
| RMSE | 0.2958 | 0.2977 | **0.2874** |
| R² | -0.0111 | -0.0240 | **0.0456** |
| Pearson r | 0.1838 | 0.1833 | **0.2352** |
| Spearman ρ | 0.1602 | 0.1772 | **0.2263** |
| Kendall τ | 0.1092 | 0.1192 | **0.1516** |

Random Forest had the highest measured performance on the external test set.

## Paired Significance Tests (test set)

Paired tests comparing per-sample errors of two models on the same samples.
`mean_diff` is `error_A − error_B` (squared error); a negative value means
model A had the lower error. Raw p-values are reported; with 3 pairwise
comparisons a Bonferroni threshold of 0.0167 applies for strong control.

| Pair (A vs B) | mean sq-err diff | paired t p | Wilcoxon p |
|---------------|------------------|------------|------------|
| RandomForest vs XGBoost | −6.02e-3 | **3.3e-5** | 2.7e-2 |
| RandomForest vs CNN | −4.90e-3 | **4.1e-3** | 3.8e-2 |
| XGBoost vs CNN | +1.12e-3 | 5.7e-1 | 7.8e-1 |

- Random Forest had a **statistically significantly lower** per-sample squared
  error than XGBoost (paired-t p < Bonferroni threshold) and than the CNN
  (paired-t p = 4.1e-3 < 0.0167). The Wilcoxon results point the same
  direction but are not significant after the conservative Bonferroni
  adjustment (p = 0.027 and 0.038).
- No statistically significant difference was detected between XGBoost and the
  CNN.

## Bootstrap 95% Confidence Intervals (test set, n_boot = 1000, seed 42)

| Metric | RandomForest | XGBoost | CNN |
|--------|--------------|---------|-----|
| R² | 0.046 [0.012, 0.077] | -0.024 [-0.084, 0.027] | -0.011 [-0.069, 0.035] |
| Pearson r | 0.235 [0.170, 0.298] | 0.183 [0.115, 0.249] | 0.184 [0.112, 0.246] |
| Spearman ρ | 0.226 [0.160, 0.289] | 0.177 [0.105, 0.245] | 0.160 [0.087, 0.225] |
| MAE | 0.249 [0.239, 0.258] | 0.254 [0.244, 0.264] | 0.252 [0.242, 0.262] |

- The R² confidence interval of Random Forest lies **entirely above zero**,
  whereas the XGBoost and CNN intervals straddle zero. Random Forest is the
  only model whose external-test R² is credibly above the mean-prediction
  baseline.
- All Pearson/Spearman intervals are strictly positive, indicating weak but
  real monotonic signal for all three models out-of-domain.

## Per-Activity-Bin Analysis (test set)

Bins are five equal-width activity intervals of the measured Moreno-Mateos
targets. Within-bin R² is strongly negative for every model in every bin, and
MAE is markedly higher in the extreme bins (bin_1 low activity, bin_5 high
activity) than in the middle:

| Model | bin_1 MAE | bin_3 MAE | bin_5 MAE | bin R² range |
|-------|-----------|-----------|-----------|--------------|
| RandomForest | 0.350 | 0.082 | 0.400 | [-50.2, -1.9] |
| XGBoost | 0.357 | 0.113 | 0.381 | [-47.8, -5.1] |
| CNN | 0.370 | 0.104 | 0.378 | [-46.8, -3.8] |

Interpretation: the poor overall out-of-domain R² is not restricted to any one
activity range; errors concentrate at the extremes. Models fare relatively best
in mid-range activity bins.

## Ranking Quality (test set)

| Metric | RandomForest | XGBoost | CNN |
|--------|--------------|---------|-----|
| Precision@5 | 0.00 | 0.00 | 0.00 |
| Precision@10 | 0.10 | 0.00 | 0.10 |
| NDCG@5 | 0.75 | 0.54 | 0.63 |
| NDCG@10 | 0.66 | 0.47 | 0.59 |

None of the models recover the measured top-5 guides on the external dataset
(top-5 precision 0 for all), and Precision@10 is at most 0.1. NDCG is
non-trivial but modest. Out-of-domain **absolute ranking at the very top is
largely lost**, even though correlations are significant; RF shows the
strongest relative ranking signal.

## Interpretation

- **In-domain vs cross-dataset.** All models perform best in-domain
  (validation) and substantially worse on the independent Moreno-Mateos set.
  Under the present feature and model configurations, cross-dataset
  generalization is limited (D-007 terminology). This separation is reported
  without asserting an overfitting mechanism.
- **Model ranking.** On validation, XGBoost had the highest measured
  performance. On the external test set, Random Forest showed the strongest
  measured performance, and a paired significance test indicates its lower
  per-sample squared error vs the other two models was unlikely to be due to
  chance (raw t p ≈ 3e-5 vs XGBoost, 4e-3 vs CNN; borderline under the
  conservative Bonferroni adjustment). No significant XGBoost-vs-CNN
  difference was detected.
- **Uncertainty.** Bootstrap R² intervals show RF is the only model whose
  external R² is fully above zero; all correlation intervals are strictly
  positive across models.
- **Ranking caveat.** Significant low-level correlations do not translate into
  reliable identification of the very best guides out-of-domain (P@5 = 0 for
  all models).

## Conclusion for Thesis

- XGBoost: best in-domain performance (validation).
- Random Forest: strongest and most reliable external-test performance
  (highest R², significant squared-error advantage, R² CI above zero).
- CNN (primary model): learns in-domain signal from raw sequence (validation
  R² 0.42), comparable to XGBoost on the external set (no significant
  difference), but below RF out-of-domain.
- All three models show limited cross-dataset generalization — a central
  finding to discuss in the thesis, not a claim of superiority of any single
  model.

## Files

- Results: `results/experiments/evaluation_phase6_20260905_013405.json`
  (scalars, paired tests, bootstrap CIs, bin metrics, ranking).
- Module: `src/evaluation/comparison.py` (`paired_error_tests`,
  `bootstrap_metric_ci`); exported from `src/evaluation/__init__.py`.
- Script: `scripts/evaluate_models_phase6.py`.
- Tests: `tests/test_evaluation_comparison.py` (9 tests, passing).