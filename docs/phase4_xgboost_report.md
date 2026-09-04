# Phase 4 Report - XGBoost Baseline

## Overview

Implementation and evaluation of the XGBoost regressor for sgRNA activity
prediction, using an identical methodology and data pipeline to the corrected
Phase 3 Random Forest baseline for a fair comparison.

## Methodology

Identical to Phase 3 (see `docs/decisions.md` D-001, D-002, D-003) and, after
correction D-005, uses a **fully fair protocol** matching Random Forest:

- **Training set:** DeepSpCas9 85% split (n = 8,599)
- **Validation set:** DeepSpCas9 15% split (n = 1,518), a held-out validation
  set used **only for evaluation** — not for early stopping or model selection
- **Independent test set:** Moreno-Mateos (n = 810), held out; used only for
  final evaluation, never for tuning or feature engineering
- Fixed seed split (`random_seed = 42`)
- Consistent sequence geometry: guide `[4:24]`, PAM `[24:27]`
- Same 197 bioinformatics features across all models

### XGBoost Configuration (`config.yaml`)

```yaml
n_estimators: 100
max_depth: 6
learning_rate: 0.1
objective: reg:squarederror
random_seed: 42
```

The baseline is trained with a **fixed** number of boosting rounds
(`n_estimators = 100`) and **no early stopping**. Validation is not used to
influence training in any way, exactly matching the Random Forest baseline.

## Results

### Validation set (held-out, used only for evaluation, n = 1,518)

| Metric | Value |
|--------|-------|
| MAE | 0.1263 |
| RMSE | 0.1564 |
| R² | 0.5112 |
| Pearson r | 0.7173 (p = 3.9e-240) |
| Spearman ρ | 0.7051 (p = 1.4e-228) |
| Kendall τ | 0.5144 (p = 4.1e-198) |

### Independent test set (Moreno-Mateos, n = 810)

| Metric | Value |
|--------|-------|
| MAE | 0.2540 |
| RMSE | 0.2977 |
| R² | -0.0240 |
| Pearson r | 0.1833 (p = 1.5e-07) |
| Spearman ρ | 0.1772 (p = 3.9e-07) |
| Kendall τ | 0.1192 (p = 3.8e-07) |

## RF vs XGBoost Comparison

### Validation (in-domain)

| Metric | XGBoost | RandomForest |
|--------|---------|--------------|
| MAE | **0.1263** | 0.1499 |
| RMSE | **0.1564** | 0.1787 |
| R² | **0.5112** | 0.3619 |
| Pearson r | **0.7173** | 0.6375 |
| Spearman ρ | **0.7051** | 0.6262 |

### Test (cross-domain, Moreno-Mateos)

| Metric | XGBoost | RandomForest |
|--------|---------|--------------|
| MAE | 0.2540 | **0.2485** |
| RMSE | 0.2977 | **0.2874** |
| R² | -0.0240 | **0.0456** |
| Pearson r | 0.1833 | **0.2352** |
| Spearman ρ | 0.1772 | **0.2263** |

## Interpretation

**In-domain (validation):** XGBoost outperforms Random Forest
(R² 0.5112 vs 0.3619; Pearson 0.7173 vs 0.6375). The measured difference
reflects the models' behavior on this split under the fixed configurations
used; no causal mechanism is claimed here.

**Cross-domain (Moreno-Mateos):** Random Forest achieved the stronger measured
performance on the independent test set (R² 0.0456 vs -0.0240; Pearson 0.2352
vs 0.1833). Both models showed substantially reduced performance on this
independent dataset relative to the validation split, indicating limited
cross-dataset generalization under the present feature and model
configurations. Both models' test correlations remain statistically
significant (p < 1e-7), i.e. a weak but detectable relationship with measured
activity persists, while absolute accuracy (R²) is poor out-of-domain. No
claim of overfitting as a cause is made without a dedicated analysis.

**Top features:** consistent with Phase 3 — guide positions 17-19 (PAM-proximal
seed region), TT/AT-rich features, and GC content dominate.

## Conclusion for Thesis

- XGBoost had the higher in-domain performance on the DeepSpCas9 validation
  split.
- Random Forest had the higher measured performance on the external
  Moreno-Mateos test set.
- Both serve as strong baselines against which the CNN (Phase 5) is compared.

## Files

- Model: `models/xgboost_baseline_*.pkl`
- Results: `results/experiments/xgboost_baseline_*.json` (includes RF comparison)
- Module: `src/models/xgboost_model.py`
- Tests: `tests/test_xgboost.py` (9 tests, passing)