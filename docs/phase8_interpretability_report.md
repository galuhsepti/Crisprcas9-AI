# Phase 8 Report - Model Interpretability

## Overview

Attribution analysis of the three fitted models on the in-domain DeepSpCas9
validation split (n = 1,518). Canonical artifacts are used unchanged; no
Moreno-Mateos data is used in this phase.

| Model | Method |
|-------|--------|
| CNN (primary) | Gradient saliency (`\x7c∂ŷ/∂one-hot\x7c`) and Integrated Gradients (zero baseline, 50 steps) |
| Random Forest | Impurity-decrease feature importances |
| XGBoost | Split-gain feature importances |

## CNN Attribution

Per-position profiles are normalized per sample (unit L1) and averaged. Both
methods agree strongly (Spearman rho = 0.859, p = 1.2e-9).

**Attribution mass by region:**

| Region | Positions | Saliency | IG |
|--------|-----------|----------|-----|
| guide | [4:24] | **0.704** | **0.677** |
| PAM | [24:27] | 0.113 | 0.120 |
| 5' flank | [0:4] | 0.104 | 0.102 |
| 3' flank | [27:30] | 0.079 | 0.100 |

## Tabular Feature Importance (group-aggregated shares)

| Group | RF (impurity) | XGBoost (gain) |
|-------|---------------|----------------|
| positional guide one-hot | 0.323 | **0.570** |
| kmer_3 (trinucleotides) | 0.247 | 0.245 |
| kmer_2 (dinucleotides) | 0.244 | 0.114 |
| gc_composition | 0.186 | 0.072 |

**Top features (both models):** `guide_pos_19_G` (RF) / `guide_pos_19_G`
(XGBoost) — the guide position closest to the PAM — followed by dinucleotide
features (`dinuc_TT`, `k2_TT`) and nucleotide composition (`freq_T`). The two
models share 4 of their top-15 features: `guide_pos_19_G`, `guide_pos_17_C`,
`dinuc_TT`, `freq_T`.

## Cross-Method Agreement

| Model | Guide-region attribution |
|-------|---------------------------|
| CNN saliency | 0.704 |
| CNN Integrated Gradients | 0.677 |
| XGBoost (guide one-hot group) | 0.570 |
| Random Forest (guide one-hot group) | 0.323 |

## Findings

1. **Consistent across methods.** The CNN places roughly 68–70% of its input
   sensitivity inside the 20 bp guide `[4:24]`, and both saliency and IG
   produce near-identical region fractions. The guide-dominant pattern is
   consistent with the Phase 7 region ablation (guide-only variant: R² 0.51;
   guide-free variant: R² 0.06).
2. **Guide 3' end is the most influential.** The most important single
   position in both tabular models is `guide_pos_19` (the guide base adjacent
   to the PAM), consistent with literature on seed-region effects.
3. **The baselines complement CNN behavior.** XGBoost also concentrates its
   gain on the guide one-hot features (0.57); Random Forest leans more on
   k-mer statistics (0.49 combined) than on position identity.

## Interpretation

- All four attribution estimates point to the guide as the dominant
  information source, with the PAM and flanking context as minor
  contributors.
- The internal consistency (saliency vs IG; CNN vs Phase 7 ablation; RF vs
  XGB top features) strengthens confidence in a qualitative, not biological,
  conclusion: the fitted models rely predominantly on the guide sequence,
  particularly its PAM-proximal 3' positions.
- These are model-attribution findings, not causal claims about sgRNA
  activity.

## Files

- Results: `results/experiments/interpretability_phase8_20260905_142321.json`
- Modules:
  - `src/interpretability/cnn_attribution.py`
    (`position_saliency`, `integrated_gradients`, `attribution_by_region`)
  - `src/interpretability/tabular_importance.py`
    (`classify_feature`, `importance_frame`,
    `aggregate_importance_by_group`, `top_features`)
- Script: `scripts/run_interpretability_phase8.py`
- Tests: `tests/test_cnn_attribution.py`, `tests/test_tabular_importance.py`
  (36 passing)
- `src/interpretability/__init__.py` exports the public API.