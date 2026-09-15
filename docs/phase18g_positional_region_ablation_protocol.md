# Phase 18G - Preregistered Positional-Region Ablation Protocol

## Final Decision

**READY_FOR_PHASE18H**

Phase 18G is protocol and preregistration only. No model was fitted, no
prediction was generated, and no positional ablation was executed. The
authorized future execution is **Phase 18H - Locked Positional-Region Ablation
Experiment**. Phase 18H has not started.

Machine-readable protocol:
`results/phase18g_positional_region_protocol_20260915_093504.json`.
Protocol SHA-256:
`05bee9cd3719824f7314366c1a47d1b9b0b3e1a1a8e651982da42a4943715e6c`.

## Scientific Context

Phase 18F commit
`0fc7273eba47e4a7ee242a945db66de4fe183100` established that removing all 80
`POSITION_SPECIFIC_NUCLEOTIDE` features caused strong predictive degradation in
both development domains. The family was classified
`ESSENTIAL_OR_STRONG_CONTRIBUTOR` in DeepSpCas9 and CRISPRon Xiang/Luo and
`CROSS_DOMAIN_CONSISTENT` across domains.

The five other feature families were `WEAK_IN_BOTH`. This means little unique
predictive contribution conditional on retained representations; it does not
mean those families contain no useful information.

Phase 18D exploratory importance highlighted positions 17-19, including
`guide_pos_19_G`. That post-hoc observation does not determine any Phase 18G
boundary and does not authorize individual-position or individual-feature
experiments.

## Primary Question

Is unique predictive information in the 20-mer spacer broadly distributed
across fixed positional regions, or concentrated in a particular region such
as the PAM-proximal segment?

This experiment measures predictive contribution under the locked model and
feature set. It does not establish molecular causality, Cas9 binding mechanism,
seed-region causality, or PAM-proximal mechanistic necessity.

## Feature Identity

Implementation truth remains
`SequenceFeatureExtractor().get_feature_names()`, consumed by
`src.multidomain.data.materialize_tabular`. The canonical feature space has 197
ordered features with SHA-256
`9adf5c5e285338ac3cf4bfd3987b1245ae61f7a18d2cbf78f6b96d6079c84dab`.

The positional family has exactly 80 ordered features with SHA-256
`73e23152a66ec80f6e52a9d37324b4cf0bc705123d82720bcd5e3ee8ce88f939`.
For every spacer position 0-19, the order is A, C, G, T. Every feature belongs
to exactly one position.

## Orientation Audit

The canonical extractor defines a forward 30-mer in 5-prime to 3-prime order:

```text
30-mer indices:  0----3  4----------------23  24--26  27--29
geometry:        5' ctx  20-nt spacer         PAM    3' ctx
spacer indices:          0-----------------19
```

`extract_position_specific_features` takes `sequence[4:24]` and maps its first
base to `guide_pos_0_*` and its last base to `guide_pos_19_*`. Therefore:

- Spacer position 0 is 30-mer index 4, the 5-prime spacer end, and PAM-distal.
- Spacer position 19 is 30-mer index 23, the 3-prime spacer end immediately
  before the PAM, and PAM-proximal.
- Positions 0-4 are designated PAM-distal.
- Positions 15-19 are designated PAM-proximal.

These conclusions come from the implementation geometry, not feature names or
biological expectation alone.

## Locked Regions

The primary partition is simple, contiguous, symmetric, complete, and fixed
before Phase 18H. Each region contains five positions and 20 nucleotide
indicators. Every positional feature occurs in exactly one region.

| Region | Spacer positions | Orientation | Removed features | Remaining canonical features |
|---|---|---|---:|---:|
| `REGION_1_PAM_DISTAL` | 0-4 | PAM-distal | 20 | 177 |
| `REGION_2_MID_DISTAL` | 5-9 | middle, distal half | 20 | 177 |
| `REGION_3_MID_PROXIMAL` | 10-14 | middle, proximal half | 20 | 177 |
| `REGION_4_PAM_PROXIMAL` | 15-19 | PAM-proximal | 20 | 177 |

The boundaries were not adjusted to isolate or favor positions 17-19.

### REGION_1_PAM_DISTAL

Positions: `0, 1, 2, 3, 4`

Features:
`guide_pos_0_A`, `guide_pos_0_C`, `guide_pos_0_G`, `guide_pos_0_T`,
`guide_pos_1_A`, `guide_pos_1_C`, `guide_pos_1_G`, `guide_pos_1_T`,
`guide_pos_2_A`, `guide_pos_2_C`, `guide_pos_2_G`, `guide_pos_2_T`,
`guide_pos_3_A`, `guide_pos_3_C`, `guide_pos_3_G`, `guide_pos_3_T`,
`guide_pos_4_A`, `guide_pos_4_C`, `guide_pos_4_G`, `guide_pos_4_T`.

Feature-list SHA-256:
`5a255dbe0eda288cbcde6f80f3cce232a555194b5d446b79af96641ad8d0504e`.

### REGION_2_MID_DISTAL

Positions: `5, 6, 7, 8, 9`

Features:
`guide_pos_5_A`, `guide_pos_5_C`, `guide_pos_5_G`, `guide_pos_5_T`,
`guide_pos_6_A`, `guide_pos_6_C`, `guide_pos_6_G`, `guide_pos_6_T`,
`guide_pos_7_A`, `guide_pos_7_C`, `guide_pos_7_G`, `guide_pos_7_T`,
`guide_pos_8_A`, `guide_pos_8_C`, `guide_pos_8_G`, `guide_pos_8_T`,
`guide_pos_9_A`, `guide_pos_9_C`, `guide_pos_9_G`, `guide_pos_9_T`.

Feature-list SHA-256:
`1f8db099357909e2874a3a79f40d3d228bf81e1bbff183adb858d2906437e5d4`.

### REGION_3_MID_PROXIMAL

Positions: `10, 11, 12, 13, 14`

Features:
`guide_pos_10_A`, `guide_pos_10_C`, `guide_pos_10_G`, `guide_pos_10_T`,
`guide_pos_11_A`, `guide_pos_11_C`, `guide_pos_11_G`, `guide_pos_11_T`,
`guide_pos_12_A`, `guide_pos_12_C`, `guide_pos_12_G`, `guide_pos_12_T`,
`guide_pos_13_A`, `guide_pos_13_C`, `guide_pos_13_G`, `guide_pos_13_T`,
`guide_pos_14_A`, `guide_pos_14_C`, `guide_pos_14_G`, `guide_pos_14_T`.

Feature-list SHA-256:
`1cf04d6394dfd6bcfeab3c1f796e8005ec72a3fd649956c4706d04aa71e1eb0e`.

### REGION_4_PAM_PROXIMAL

Positions: `15, 16, 17, 18, 19`

Features:
`guide_pos_15_A`, `guide_pos_15_C`, `guide_pos_15_G`, `guide_pos_15_T`,
`guide_pos_16_A`, `guide_pos_16_C`, `guide_pos_16_G`, `guide_pos_16_T`,
`guide_pos_17_A`, `guide_pos_17_C`, `guide_pos_17_G`, `guide_pos_17_T`,
`guide_pos_18_A`, `guide_pos_18_C`, `guide_pos_18_G`, `guide_pos_18_T`,
`guide_pos_19_A`, `guide_pos_19_C`, `guide_pos_19_G`, `guide_pos_19_T`.

Feature-list SHA-256:
`a00bfdff1ec3297b6ecb8b23368e94d84ff77a2e19ff55b0dbffec73490397eb`.

## Phase 18H Arms

The primary comparison in each domain is reused `FULL_XGBOOST` versus four new
drop-region fits:

- `DROP_REGION_1_PAM_DISTAL`
- `DROP_REGION_2_MID_DISTAL`
- `DROP_REGION_3_MID_PROXIMAL`
- `DROP_REGION_4_PAM_PROXIMAL`

Only the 20 indicators in the named region are removed. All other canonical
features remain in their original order. This yields 177 features per new arm.
Labels remain separate and in native units for DeepSpCas9 and CRISPRon
Xiang/Luo.

## Reused Phase 18F References

The two Phase 18F FULL models and predictions are compatible with the locked
feature hash, split, XGBoost configuration, seed, and internal-test observations.
Their artifact hashes were verified. They are reused as the two primary FULL
references, so no FULL refit is registered.

The two Phase 18F `DROP_ALL_POSITION_SPECIFIC` results and artifacts are also
compatible and are reused as secondary whole-family context. No new whole-family
fit is registered. Regional effects are not assumed to add to the whole-family
effect.

The matrix therefore contains 12 records: eight new confirmatory drop-region
arms, two reused FULL references, and two reused whole-positional-family
references.

## Region-Only Decision

`REGION_1_ONLY` through `REGION_4_ONLY` are **not registered**. They answer the
distinct question of whether a 20-feature local region is sufficient, whereas
the primary experiment tests unique regional contribution conditional on all
retained features. Adding them would require eight extra fits and is not needed
to answer the primary question. A later study requires a separate
preregistration.

## Split, Model, and Seed Locks

The immutable Phase 18C/18F forward-spacer group split is reused without
regeneration:

`b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8`

Bridge and quarantine remain excluded.

Future Phase 18H uses the exact deterministic Phase 18C/18F XGBoost
configuration: 100 estimators, depth 6, learning rate 0.1,
`min_child_weight=1`, full row and feature sampling, no L1 penalty, L2 penalty
1.0, squared-error regression, histogram trees on CPU, four jobs, no sample
weighting, no tuning, no early stopping, and no validation refit.

Only seed 42 is registered. Phase 18C produced identical XGBoost outcomes at
seeds 42-46 with full sampling, and Phase 18F retained seed 42. Redundant repeats
would not estimate stochastic uncertainty.

## Metrics and Effects

- Primary: Spearman rho.
- Regression guardrail: RMSE.
- Secondary: Pearson, R-squared, and MAE.
- `delta_spearman = Spearman_drop_region - Spearman_full`.
- `delta_rmse = RMSE_drop_region - RMSE_full`.
- Relative RMSE degradation is
  `(RMSE_drop_region - RMSE_full) / RMSE_full`.

Negative delta Spearman and positive delta RMSE indicate degradation. Every
effect is calculated separately by domain. Raw RMSE magnitudes are not compared
between domain label scales.

## Bootstrap and Multiplicity

Each drop-region arm is compared with the reused FULL prediction on identical
internal-test observations. Phase 18H uses 10,000 paired percentile bootstrap
replicates over sorted unique exact forward spacer groups, NumPy PCG64 seed 42,
and linear percentile quantiles. The same sampled groups index both vectors.

The confirmatory family contains 16 intervals: four regions by two domains by
two endpoints. Bonferroni controls familywise alpha at 0.05:

- Per-interval alpha: `0.003125`.
- Simultaneous confidence: `0.996875` or `99.6875%`.
- Two-sided quantiles: `0.0015625` and `0.9984375`.

Secondary metrics receive nominal 95% descriptive intervals. Cross-region
rankings are descriptive; no pairwise region-superiority tests are registered.

## Regional Contribution Rules

- `STRONG_REGIONAL_CONTRIBUTOR`: adjusted upper CI for delta Spearman is below
  zero and adjusted lower CI for delta RMSE is above zero.
- `MODERATE_REGIONAL_CONTRIBUTOR`: exactly one adjusted interval supports
  degradation, the other point estimate is directionally concordant, and
  neither endpoint supports improvement.
- `LITTLE_UNIQUE_REGIONAL_CONTRIBUTION`: neither adjusted endpoint supports
  degradation or improvement. This is not equivalence or absence of
  information.
- `POTENTIALLY_REDUNDANT_REGION`: at least one adjusted endpoint supports
  improvement after removal and neither supports degradation.
- `UNSTABLE_OR_INCONCLUSIVE`: metrics are undefined/nonfinite, a bootstrap
  replicate is invalid, or endpoint directions conflict with statistical
  support.

No post-result minimum-effect threshold is introduced.

## Cross-Domain Rules

- `CROSS_DOMAIN_CONSISTENT`: both domains receive strong or moderate regional
  contributor classes with concordant degradation directions.
- `DOMAIN_A_ENRICHED`: only DeepSpCas9 receives a contributor class.
- `DOMAIN_B_ENRICHED`: only Xiang/Luo receives a contributor class.
- `WEAK_IN_BOTH`: both domains receive little-unique or potentially-redundant
  classes.
- `UNSTABLE_OR_INCONCLUSIVE`: either domain is inconclusive or endpoint
  directions conflict.

Enriched classifications describe evidence asymmetry, not a formal comparison
of native-scale effect magnitudes.

## Preregistered Questions

1. Does removal of the fixed PAM-proximal region have the largest descriptive
   degradation?
2. Does removal of the fixed PAM-distal region have weaker descriptive impact?
3. Do at least two regions receive contributor classifications, supporting
   broadly distributed predictive information?
4. Are regional contribution classes consistent across the two domains?
5. Does the fixed region containing positions 17-19 retain confirmatory
   group-level predictive contribution?

Questions 1 and 2 use descriptive point-effect ordering because no pairwise
region-superiority tests are registered. Questions 3-5 use the preregistered
region-versus-FULL classifications. Question 5 remains regional and does not
authorize a `guide_pos_19_G` test.

## Prohibited Searches

Phase 18H is not authorized to perform 20 leave-one-position-out experiments,
80 individual indicator removals, `guide_pos_19_G`-specific experiments,
importance-guided position selection, recursive positional elimination,
post-result boundary changes, or unregistered region-only models.

## Compute Budget

- Required new fits: 8.
- Reused FULL references: 2.
- Reused whole-family contextual references: 2.
- Optional region-only fits: 0.
- Bootstrap comparisons: 8.
- Comparison-replicates: 80,000.

Phase 18F recorded approximately 18 seconds for 18 comparable fits. Eight new
fits are expected to require roughly 8-15 seconds on similar hardware. The
10,000-replicate bootstrap may dominate runtime. This is a prospective estimate,
not an execution result.

## Isolation and Stop Conditions

The locked external dataset is prohibited from opening, reading, searching,
hashing, statistics, training, evaluation, arm selection, and threshold design.
The process-wide Phase 18C access guard remains mandatory. Tests use synthetic
paths only.

Phase 18G fails if feature identity, orientation, region membership, split,
configuration, Phase 18F reference metadata/artifact hashes, bridge/quarantine
exclusion, no-training enforcement, or locked-data isolation changes.

`READY_FOR_PHASE18H` means only that the protocol is complete. It does not
authorize Phase 18H execution without explicit approval.
