# Phase 18D Post-Experiment Interpretation Report

## Status and scope

Phase 18D is complete as an **EXPLORATORY POST-HOC** analysis of only the official Phase 18C campaign. The immutable source classification remains **NO_CLEAR_BENEFIT**. No Phase 18C classifier was rerun, no predictions were generated or changed, and no model was fitted, tuned, calibrated, or updated.

Domains retain native label scales throughout; raw activities are never pooled across domains.

## Deterministic policy

- Residual is prediction minus observed activity.
- Activity bins are balanced within-domain rank quartiles fixed by true activity, with SHA-256 sequence identity as the tie-breaker.
- GC bins are fixed at `<0.40`, `0.40-0.60`, and `>0.60`.
- Substantial model disagreement is an absolute-error difference of at least `0.10 x within-domain target IQR`, set before examples are extracted.
- Hard cases are the top 10% within each domain by mean absolute error across four model families and five seeds, normalized by target IQR.
- Position-specific rows require `n >= 30`; stratum Spearman requires `n >= 10` and nonconstant values.
- Performance metrics are computed separately for each registered seed and then arithmetically averaged; seed-averaged predictions are not evaluated as an ensemble.

## Residual results

| domain | model | n | mean_residual | median_residual | mae | rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepspcas9 | single_cnn | 1526 | 0.0066391 | 0.004726 | 0.13412 | 0.16903 | 0.63112 |
| deepspcas9 | cnn_d | 1526 | -0.0017329 | -0.0021128 | 0.13231 | 0.16601 | 0.64944 |
| deepspcas9 | random_forest | 1526 | 0.0011403 | 0.0014926 | 0.14637 | 0.17666 | 0.64177 |
| deepspcas9 | xgboost | 1526 | 0.0019952 | -0.00020451 | 0.12277 | 0.15501 | 0.70805 |
| crispron_xiang_luo | single_cnn | 1559 | -0.62388 | 0.017053 | 13.797 | 17.252 | 0.67782 |
| crispron_xiang_luo | cnn_d | 1559 | -1.2057 | -0.23171 | 13.737 | 17.144 | 0.68496 |
| crispron_xiang_luo | random_forest | 1559 | -1.2443 | 0.3726 | 14.81 | 17.88 | 0.68188 |
| crispron_xiang_luo | xgboost | 1559 | -0.95851 | -0.18679 | 12.154 | 15.519 | 0.75578 |

The activity-, GC-, PAM-, and complexity-stratified estimates are associative and exploratory. Sample sizes are reported in the tables. Tiny PAM groups are not interpreted strongly.

## CNN_D error signature

| domain | stratifier | stratum | n | mean_delta_absolute_error | fraction_improved |
| --- | --- | --- | --- | --- | --- |
| deepspcas9 | activity | Q1 | 382 | 0.0040907 | 0.53927 |
| deepspcas9 | activity | Q2 | 381 | 0.011986 | 0.64304 |
| deepspcas9 | activity | Q3 | 382 | -0.0023342 | 0.48429 |
| deepspcas9 | activity | Q4 | 381 | -0.0065247 | 0.43307 |
| deepspcas9 | spacer_gc | high_gt_0.60 | 324 | 0.00094208 | 0.52778 |
| deepspcas9 | spacer_gc | low_lt_0.40 | 148 | -0.002563 | 0.49324 |
| deepspcas9 | spacer_gc | mid_0.40_to_0.60 | 1054 | 0.0026809 | 0.52846 |
| deepspcas9 | full_gc | high_gt_0.60 | 421 | 0.0019829 | 0.52732 |
| deepspcas9 | full_gc | low_lt_0.40 | 98 | 0.00012468 | 0.5 |
| deepspcas9 | full_gc | mid_0.40_to_0.60 | 1007 | 0.0018913 | 0.52632 |
| deepspcas9 | pam | AGG | 545 | 0.0015189 | 0.52661 |
| deepspcas9 | pam | CGG | 225 | 0.0017119 | 0.52444 |
| deepspcas9 | pam | GGG | 133 | 0.0054545 | 0.58647 |
| deepspcas9 | pam | TGG | 623 | 0.0013052 | 0.51043 |
| deepspcas9 | sequence_complexity | high | 508 | 0.0017137 | 0.52165 |
| deepspcas9 | sequence_complexity | low | 509 | 0.00073984 | 0.51081 |
| deepspcas9 | sequence_complexity | middle | 509 | 0.0029557 | 0.54224 |
| crispron_xiang_luo | activity | Q1 | 390 | 0.46118 | 0.58462 |
| crispron_xiang_luo | activity | Q2 | 390 | 0.37478 | 0.54872 |
| crispron_xiang_luo | activity | Q3 | 390 | 0.37973 | 0.54615 |
| crispron_xiang_luo | activity | Q4 | 389 | -0.97646 | 0.3856 |
| crispron_xiang_luo | spacer_gc | high_gt_0.60 | 616 | -0.25719 | 0.48864 |
| crispron_xiang_luo | spacer_gc | low_lt_0.40 | 75 | 0.67325 | 0.62667 |
| crispron_xiang_luo | spacer_gc | mid_0.40_to_0.60 | 868 | 0.23296 | 0.5265 |
| crispron_xiang_luo | full_gc | high_gt_0.60 | 712 | -0.058799 | 0.51685 |
| crispron_xiang_luo | full_gc | low_lt_0.40 | 31 | 0.92222 | 0.77419 |
| crispron_xiang_luo | full_gc | mid_0.40_to_0.60 | 816 | 0.1318 | 0.50613 |
| crispron_xiang_luo | pam | AGG | 384 | 0.27633 | 0.55469 |
| crispron_xiang_luo | pam | CGG | 251 | -0.20638 | 0.49402 |
| crispron_xiang_luo | pam | GGG | 389 | -0.28881 | 0.46272 |
| crispron_xiang_luo | pam | TGG | 535 | 0.28469 | 0.53832 |
| crispron_xiang_luo | sequence_complexity | high | 519 | 0.37793 | 0.52408 |
| crispron_xiang_luo | sequence_complexity | low | 520 | -0.16483 | 0.50192 |
| crispron_xiang_luo | sequence_complexity | middle | 520 | -0.031085 | 0.52308 |

Positive delta absolute error means CNN_D improved over the corresponding single-domain CNN. These patterns do not alter the Phase 18C outcome.

## Position-specific associations

912 nucleotide-position/model strata met the `n >= 30` threshold. The reported associations are not causal sequence determinants.

## XGBoost representation and importance

The registered XGBoost models received exactly 197 deterministic features: full/regional GC and skew features; global nucleotide composition and dinucleotide frequencies; nucleotide heterogeneity; global 2-mer and 3-mer frequencies plus entropy/complexity; and one-hot nucleotide identity at each of the 20 spacer positions. The exact ordered names are archived.

| domain | rank | feature | feature_class | mean_normalized_gain | sd_normalized_gain |
| --- | --- | --- | --- | --- | --- |
| crispron_xiang_luo | 1 | guide_pos_19_G | spacer_position_one_hot | 0.12684 | 0 |
| crispron_xiang_luo | 2 | dinuc_TT | global_dinucleotide_frequency | 0.068758 | 0 |
| crispron_xiang_luo | 3 | guide_pos_18_T | spacer_position_one_hot | 0.030838 | 0 |
| crispron_xiang_luo | 4 | guide_pos_19_A | spacer_position_one_hot | 0.023931 | 0 |
| crispron_xiang_luo | 5 | guide_pos_16_T | spacer_position_one_hot | 0.022001 | 0 |
| crispron_xiang_luo | 6 | guide_pos_18_G | spacer_position_one_hot | 0.01675 | 0 |
| crispron_xiang_luo | 7 | guide_pos_0_C | spacer_position_one_hot | 0.015303 | 0 |
| crispron_xiang_luo | 8 | guide_pos_13_G | spacer_position_one_hot | 0.014831 | 0 |
| crispron_xiang_luo | 9 | guide_pos_17_T | spacer_position_one_hot | 0.014741 | 0 |
| crispron_xiang_luo | 10 | guide_pos_17_C | spacer_position_one_hot | 0.012554 | 0 |
| deepspcas9 | 1 | guide_pos_19_G | spacer_position_one_hot | 0.0706 | 0 |
| deepspcas9 | 2 | dinuc_TT | global_dinucleotide_frequency | 0.048086 | 0 |
| deepspcas9 | 3 | guide_pos_18_T | spacer_position_one_hot | 0.040999 | 0 |
| deepspcas9 | 4 | guide_pos_17_C | spacer_position_one_hot | 0.025975 | 0 |
| deepspcas9 | 5 | guide_pos_18_A | spacer_position_one_hot | 0.020489 | 0 |
| deepspcas9 | 6 | is_optimal_gc | gc_regional_and_skew | 0.019092 | 0 |
| deepspcas9 | 7 | k3_complexity | global_kmer_entropy_complexity | 0.018633 | 0 |
| deepspcas9 | 8 | guide_pos_19_A | spacer_position_one_hot | 0.014985 | 0 |
| deepspcas9 | 9 | guide_pos_16_A | spacer_position_one_hot | 0.014217 | 0 |
| deepspcas9 | 10 | guide_pos_17_G | spacer_position_one_hot | 0.01277 | 0 |

Built-in gain importance describes model split utility, not biological causality. SHAP was not used, avoiding a new dependency and reproducibility risk.

## Bridge interpretation

**Limitation: the bridge contains only 41 exact pairs.** It remains a diagnostic and cannot alter **NO_CLEAR_BENEFIT**.

| seed | single_head_spearman | cnn_d_head_spearman | delta_spearman | single_mean_rank_disagreement | cnn_d_mean_rank_disagreement | pairs_reduced_disagreement |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.75557 | 0.89286 | 0.13728 | 0.15854 | 0.086585 | 31 |
| 43 | 0.68171 | 0.94704 | 0.26533 | 0.17561 | 0.067073 | 32 |
| 44 | 0.68223 | 0.94721 | 0.26498 | 0.17805 | 0.063415 | 30 |
| 45 | 0.73815 | 0.94861 | 0.21045 | 0.1561 | 0.068293 | 29 |
| 46 | 0.74268 | 0.94286 | 0.20017 | 0.1561 | 0.070732 | 30 |

Leave-one-out and pairwise rank-disagreement summaries are in `summary.json` and `bridge_pairs.csv`; they assess whether the consistency shift is broad or outlier-dominated.

## Scientific interpretation

### Observed results

- Domain A (DeepSpCas9): xgboost had the lowest exploratory mean per-seed RMSE (0.155012 native units).
- Domain A (DeepSpCas9): mean residuals shifted from 0.147691 in Q1 to -0.154045 in Q4 for XGBoost.
- Domain A (DeepSpCas9): CNN_D reduced absolute error for 52.5% of observations when comparing each observation's mean error across the five registered seeds.
- Domain A (DeepSpCas9): the algorithmic hard-case set contained 153 observations.
- Domain A (DeepSpCas9): XGBoost/CNN_D absolute-error Spearman correlation averaged 0.5898 across seeds.
- Domain B (CRISPRon Xiang/Luo): xgboost had the lowest exploratory mean per-seed RMSE (15.5193 native units).
- Domain B (CRISPRon Xiang/Luo): mean residuals shifted from 12.0362 in Q1 to -17.021 in Q4 for XGBoost.
- Domain B (CRISPRon Xiang/Luo): CNN_D reduced absolute error for 51.6% of observations when comparing each observation's mean error across the five registered seeds.
- Domain B (CRISPRon Xiang/Luo): the algorithmic hard-case set contained 156 observations.
- Domain B (CRISPRon Xiang/Luo): XGBoost/CNN_D absolute-error Spearman correlation averaged 0.5266 across seeds.
- Across the 41-pair bridge, CNN_D reduced mean rank disagreement for 37/41 pairs on average across seeds.

### Plausible interpretations

- The registered 197-feature representation exposes regional GC, composition, k-mer, entropy/complexity, and spacer-position information directly; its superior internal metrics are consistent with efficient tabular representation for short 30-mers.
- With 7,040 Domain A and 7,465 Domain B training observations, the fixed tree ensemble may use the engineered representation more statistically efficiently than the 17,985-parameter single-domain CNNs; Phase 18D does not isolate dataset size as the cause.
- The increased bridge head agreement is consistent with CNN_D learning a more shared cross-domain rank representation, while separate heads retain native assay scales.

### Untested hypotheses

- Engineered features causally explain XGBoost's advantage.
- A larger dataset or different CNN architecture would reverse the ranking.
- XGBoost and CNN predictions would improve performance in an ensemble.
- Any reported sequence association is a causal biological determinant.
- Internal findings generalize to an external dataset or production use.

### Recommended next questions

- Preregister a controlled representation-ablation study to separate positional, k-mer, composition, and GC contributions.
- Preregister an ensemble experiment only if the observed residual correlations justify testing complementarity.
- Test whether bridge consistency predicts cross-domain utility in a larger independent paired-sequence set.
- Evaluate learning-curve and architecture questions in a later phase with fixed hypotheses and no post-hoc model selection.

No ensemble was built or evaluated. Any ensemble, alternative feature set, architecture, or external evaluation requires a later preregistered phase.

## Outputs

- Figure: `results/phase18d/figures/model_performance_comparison.png`
- Figure: `results/phase18d/figures/residual_distributions.png`
- Figure: `results/phase18d/figures/activity_vs_absolute_error.png`
- Figure: `results/phase18d/figures/error_by_spacer_gc.png`
- Figure: `results/phase18d/figures/error_by_pam.png`
- Figure: `results/phase18d/figures/cnn_d_improvement_distribution.png`
- Figure: `results/phase18d/figures/xgboost_feature_importance.png`
- Figure: `results/phase18d/figures/bridge_rank_consistency.png`
- Table: `results/phase18d/tables/analysis_rows_domain_A.csv`
- Table: `results/phase18d/tables/analysis_rows_domain_B.csv`
- Table: `results/phase18d/tables/observation_summary_domain_A.csv`
- Table: `results/phase18d/tables/observation_summary_domain_B.csv`
- Table: `results/phase18d/tables/model_error_summary.csv`
- Table: `results/phase18d/tables/strata_summary.csv`
- Table: `results/phase18d/tables/model_disagreement_assignments.csv`
- Table: `results/phase18d/tables/model_disagreement_summary.csv`
- Table: `results/phase18d/tables/cnn_d_help_patterns.csv`
- Table: `results/phase18d/tables/cnn_d_seed_consistency.csv`
- Table: `results/phase18d/tables/position_specific_error_associations.csv`
- Table: `results/phase18d/tables/hard_cases.csv`
- Table: `results/phase18d/tables/error_complementarity.csv`
- Table: `results/phase18d/tables/xgboost_feature_importance.csv`
- Table: `results/phase18d/tables/xgboost_feature_names.csv`
- Table: `results/phase18d/tables/bridge_pairs.csv`
- Table: `results/phase18d/tables/bridge_seed_summary.csv`

## Integrity

All 70 Phase 18C prediction/model artifacts matched the official manifest before analysis and all Phase 18C source identities remained unchanged after analysis. The locked external dataset remained prohibited and untouched.
