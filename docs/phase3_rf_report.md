# Phase 3 Report - Random Forest Baseline (corrected)

> **IMPORTANT:** This report supersedes the earlier Phase 3 baseline results.
> The previous results (MAE 0.0745 / RMSE 0.0905 / R² 0.8242 on validation)
> were **invalid** for two reasons and must NOT be used as final thesis results:
> 1. **Data leakage:** the model was fitted on the *full* DeepSpCas9 set
>    (including the samples later used for validation), so validation was
>    **not** truly unseen.
> 2. **Wrong sequence geometry:** `guide_start` was erroneously derived as
>    `(30-20)//2 = 5`, so the guide spanned positions 5-24 (including the first
>    PAM base) instead of the correct positions 4-23. PAM/context GC slices were
>    also shifted by one base.

## Corrected Geometry (30-mer)

| Region | Slice | Length |
|--------|-------|--------|
| 5' context | [0:4] | 4 bp |
| Guide | [4:24] | 20 bp |
| PAM (NGG) | [24:27] | 3 bp |
| 3' context | [27:30] | 3 bp |

This matches the standard DeepSpCas9 30-mer format and the slicing already
used in `scripts/prepare_data.py`.

## Methodology (corrected)

- **Training set:** DeepSpCas9, 85% split (n = 8,599) — model fitted **only**
  on this split.
- **Validation set:** DeepSpCas9, 15% split (n = 1,518) — completely unseen
  during training; used for model selection only.
- **Independent test set:** Moreno-Mateos (n = 810) — held out entirely; used
  **only** for final evaluation, never for tuning or feature engineering.
- Fixed seed random split (`random_seed = 42`).

## Data Summary (post-filter)

| Dataset | Total | Valid | Filtered | Reason |
|---------|-------|-------|----------|--------|
| DeepSpCas9 | 12,832 | 10,117 | 2,715 | Homopolymer runs (≥4 identical nucleotides) |
| Moreno-Mateos | 1,020 | 810 | 210 | Homopolymer runs (≥4 identical nucleotides) |

## Feature Engineering

Total features: **197** (identical columns across train/test)

| Category | Count |
|----------|-------|
| GC Content | 10 |
| Composition | 17 |
| k-mer (k=2) | 18 |
| k-mer (k=3) | 66 |
| Positional (guide one-hot) | 80 |

Model hyperparameters (from `config.yaml`): `n_estimators=200, max_depth=20,
min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42`.

## Results (corrected)

### Validation set (truly unseen, n = 1,518)

| Metric | Value |
|--------|-------|
| MAE | 0.1499 |
| RMSE | 0.1787 |
| R² | 0.3619 |
| Pearson r | 0.6375 (p = 6.7e-174) |
| Spearman ρ | 0.6262 (p = 4.2e-166) |
| Kendall τ | 0.4477 (p = 1.6e-150) |

### Independent test set (Moreno-Mateos, n = 810)

| Metric | Value |
|--------|-------|
| MAE | 0.2485 |
| RMSE | 0.2874 |
| R² | 0.0456 |
| Pearson r | 0.2352 (p = 1.2e-11) |
| Spearman ρ | 0.2263 (p = 7.2e-11) |
| Kendall τ | 0.1516 (p = 1.1e-10) |

## Interpretation

With leakage and geometry bugs fixed, the honestly measured in-domain
performance (R² = 0.36, Pearson r = 0.64) is substantially lower than the
invalid 0.82 R² previously reported. These corrected numbers are the ones to
use in the thesis.

The cross-dataset drop (r ≈ 0.64 → 0.24) is expected and consistent with the
CRISPR prediction literature: different laboratories, cell lines, and activity
measurement protocols introduce large domain shifts. The independent-test
correlations remain statistically significant (p < 1e-10), indicating that the
learned features carry real biological signal.

The top features (guide position 18/19 near the PAM-proximal end, GC content,
TT/AT-rich seed-region features) are consistent with known determinants of
sgRNA activity.

## Files

- Model: `models/rf_baseline_fixed_*.pkl`
- Results: `results/experiments/rf_baseline_fixed_*.json`

## Decision Record

See `docs/decisions.md` for the full record of methodology corrections.