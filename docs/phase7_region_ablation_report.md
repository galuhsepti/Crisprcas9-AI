# Phase 7 Report - Sequence-Region Ablation Study (CNN)

## Overview

To isolate which parts of the 30-mer window drive CNN predictive performance,
CNN variants were trained on restricted sequence regions with an **identical
training recipe** to the canonical Phase 5 CNN (kernels [5,7,9], 64 filters,
dense 64, dropout 0.3, lr 1e-3, batch 32, patience 10, seed 42). The canonical
model and its artifacts were **not modified**: the `full` region reuses the
canonical saved model as a uniform comparison basis and pipeline-consistency
check.

## Regions (fixed geometry: `[0:4]` 5' ctx | `[4:24]` guide | `[24:27]` PAM | `[27:30]` 3' ctx)

| Region | Positions | Length (bp) |
|--------|-----------|-------------|
| full | [0:30] | 30 |
| guide | [4:24] | 20 |
| guide_pam | [4:27] | 23 |
| context_pam | [0:4] + [24:30] (everything except guide) | 10 |

## Methodology

- Same validated sequences, same DeepSpCas9 15% validation split (seed 42),
  same Moreno-Mateos test set (n = 810, untouched by all variants).
- Validation used for early stopping / model selection (D-006); in-domain
  comparisons are **descriptive only**. The test set is the fair comparison
  basis.
- Bootstrap 95% CIs (n_boot = 1000, seed 42) on test-set metrics.

## Results - Validation (in-domain, descriptive)

| Region | len | MAE | RMSE | R² | Pearson | Spearman |
|--------|-----|-------|-------|--------|---------|----------|
| full | 30 | 0.1385 | 0.1709 | 0.4164 | 0.6468 | 0.6313 |
| guide | 20 | **0.1254** | **0.1572** | **0.5061** | **0.7114** | **0.6999** |
| guide_pam | 23 | 0.1311 | 0.1617 | 0.4770 | 0.6922 | 0.6804 |
| context_pam | 10 | 0.1841 | 0.2165 | 0.0630 | 0.2573 | 0.2525 |

## Results - Test (Moreno-Mateos, fair for all variants)

| Region | len | MAE | RMSE | R² | Pearson | Spearman |
|--------|-----|-------|-------|--------|---------|----------|
| full | 30 | **0.2518** | **0.2958** | **-0.0111** | 0.1838 | 0.1602 |
| guide | 20 | 0.2520 | 0.2995 | -0.0365 | **0.1968** | **0.1811** |
| guide_pam | 23 | 0.2544 | 0.2995 | -0.0362 | 0.1743 | 0.1571 |
| context_pam | 10 | 0.2590 | 0.3005 | -0.0434 | 0.1140 | 0.1210 |

## Bootstrap 95% CI on test R²

| Region | R² | 95% CI |
|--------|------|-----------------|
| full | -0.0111 | [-0.0690, 0.0353] |
| guide | -0.0365 | [-0.1049, 0.0250] |
| guide_pam | -0.0362 | [-0.0996, 0.0194] |
| context_pam | -0.0434 | [-0.0837, -0.0103] |

## Findings

1. **The guide sequence carries essentially all the predictive signal.**
   In-domain, the variant trained on the 20 bp guide alone reaches R² = 0.51,
   the highest of the four variants, and the variant trained **without** the
   guide (context_pam) collapses to R² = 0.06.
2. **Flanking/PAM context adds little in-domain.** Full (30 bp) in-domain
   performance is below guide-only, and removing PAM from the guide (guide vs
   guide_pam) slightly *improves* it; added context does not help under the
   present architecture.
3. **Out-of-domain all variants are weak, and differences are statistical
   noise.** On Moreno-Mateos, every R² CI overlaps zero (except context_pam,
   which is marginally below zero). The guide-only variant has a slightly
   higher Pearson/Spearman than the full model, but this is not
   distinguishable from chance given overlapping intervals and is not claimed
   as a finding.

## Interpretation

- The CNN learns almost exclusively from the 20 bp guide `[4:24]`; the PAM and
  flanking context do not measurably improve in-domain prediction, and the
  guide-free variant loses nearly all predictive ability. This is consistent
  with the guide being the functional determinant of sgRNA activity.
- The finding ranks as descriptive: it reflects this pipeline's architecture
  and data, and only the in-domain signal decomposition is asserted. No claim
  is made that guide-only is superior overall, since the external-test
  differences are not statistically significant.
- Implication for the thesis: the 30-mer window could potentially be narrowed
  to the guide region without loss of in-domain accuracy, but cross-dataset
  generalization remains limited for all variants.

## Files

- Results: `results/experiments/ablation_region_phase7_20260905_015158.json`
- Module: `src/ablation/sequence_regions.py` (region definition + one-hot
  extraction), exported via `src/ablation/__init__.py`.
- Script: `scripts/run_ablation_region_phase7.py`
- Variant models (gitignored): `models/ablation/cnn_guide.pt`,
  `cnn_guide_pam.pt`, `cnn_context_pam.pt`
- Tests: `tests/test_sequence_regions.py` (10 tests, passing).