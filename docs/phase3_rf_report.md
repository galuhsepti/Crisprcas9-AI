# Phase 3 Report - Random Forest Baseline

## Overview

This report documents the implementation and evaluation of the Random Forest baseline model for sgRNA activity prediction as part of the CRISPR-Cas9 prediction pipeline.

## Data Summary

| Dataset | Total | Valid | Filtered | Filter Reason |
|---------|-------|-------|----------|---------------|
| DeepSpCas9 (train) | 12,832 | 10,094 | 2,738 | Homopolymer runs (≥4 identical nucleotides) |
| Moreno-Mateos (test) | 1,020 | 836 | 184 | Homopolymer runs (≥4 identical nucleotides) |

### Filter Details (DeepSpCas9)
- 891 sequences with C homopolymer
- 703 sequences with G homopolymer
- 657 sequences with A homopolymer
- 487 sequences with T homopolymer

## Feature Engineering

Total features: **197**

| Category | Count | Description |
|----------|-------|-------------|
| GC Content | 10 | Overall GC, regional GC, skew, optimal GC |
| Composition | 17 | Nucleotide frequencies, di-nucleotides, heterogeneity |
| k-mer (k=2) | 18 | 16 dinucleotides + entropy + complexity |
| k-mer (k=3) | 66 | 64 trinucleotides + entropy + complexity |
| Positional | 80 | One-hot binary features for 20 guide positions × 4 nucleotides |

## Model Configuration

```yaml
n_estimators: 200
max_depth: 20
min_samples_split: 5
min_samples_leaf: 2
max_features: 'sqrt'
random_state: 42
```

## Results

### Validation Set (held-out 15% of DeepSpCas9, n=1,515)
Evaluation on held-out DeepSpCas9 sequences (same distribution as training):

| Metric | Value |
|--------|-------|
| MAE | 0.0745 |
| RMSE | 0.0905 |
| R² | 0.8242 |
| Pearson r | 0.9585 (p < 0.001) |
| Spearman ρ | 0.9622 (p < 0.001) |
| Kendall τ | 0.8295 (p < 0.001) |

### Independent Test Set (Moreno-Mateos, n=836)
Evaluation on completely independent dataset (different laboratory, methods, cells):

| Metric | Value |
|--------|-------|
| MAE | 0.2507 |
| RMSE | 0.2901 |
| R² | 0.0364 |
| Pearson r | 0.2288 (p=2.16e-11) |
| Spearman ρ | 0.2224 (p=7.91e-11) |
| Kendall τ | 0.1496 (p=9.52e-11) |

## Interpretation

### Within-dataset performance (validation)
The Random Forest model performs excellently on held-out DeepSpCas9 sequences:
- Pearson correlation of 0.96 indicates strong sequence-activity relationship modeling
- R² of 0.82 shows the model explains most variance in DeepSpCas9 data

### Cross-dataset performance (Moreno-Mateos test set)
Significant performance drop on the independent test set:
- Correlation drops from 0.96 to 0.23
- R² drops from 0.82 to 0.04

This domain shift is **well-known in CRISPR prediction literature** and is caused by:
1. Different experimental conditions (cell lines, delivery methods, assay types)
2. Different activity measurement methodologies (modification frequency vs. other readouts)
3. Batch effects between laboratories
4. Potentially different sequence context distributions

Despite the drop, the Pearson/Spearman correlations remain **statistically significant** (p < 1e-10), demonstrating that the learned sequence features carry real biological signal that generalizes across datasets.

## Top Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | guide_pos_18_G | 0.0491 |
| 2 | dinuc_TT | 0.0235 |
| 3 | k2_TT | 0.0208 |
| 4 | guide_pos_16_C | 0.0204 |
| 5 | gc_guide | 0.0200 |
| 6 | guide_pos_18_C | 0.0186 |
| 7 | freq_T | 0.0183 |
| 8 | at_skew | 0.0179 |
| 9 | gc_full | 0.0160 |
| 10 | k2_entropy | 0.0154 |

This aligns with known biology:
- **Position 18 of the guide** (adjacent to PAM) is critical for Cas9 binding specificity
- **GC content** is a well-established predictor of sgRNA efficiency
- **TT dinucleotide** in seed region affects off-target specificity

## Files Generated

- `models/rf_baseline_*.pkl` - Trained Random Forest model
- `results/experiments/rf_baseline_*.json` - Complete results with metrics
- `src/models/random_forest.py` - RandomForestModel class
- `src/evaluation/metrics.py` - Comprehensive evaluation metrics
- `tests/test_evaluation.py` - 13 evaluation tests (pass)
- `tests/test_models.py` - 10 model tests (pass)

## Next Steps

1. Proceed to Phase 4: XGBoost model for comparison
2. Compare XGBoost vs Random Forest performance
3. Consider reducing homopolymer filtering strictness to retain more training data
4. Explore feature selection to improve cross-dataset generalization