# Phase 18E - Preregistered XGBoost Feature-Ablation Protocol

## Final Decision

**READY_FOR_PHASE18F**

Phase 18E is design and preregistration only. No model was fitted, no
prediction was generated, and no feature-ablation result was calculated. The
authorized future execution name is **Phase 18F - Locked XGBoost
Feature-Ablation Experiment**. Phase 18F has not started.

## Scientific Question and Scope

The primary question is which engineered feature families contribute unique
predictive information to the strong within-domain XGBoost performance observed
in Phase 18C. This controlled experiment can test representation contribution;
it cannot by itself distinguish every algorithmic reason why XGBoost
outperformed the current CNNs.

Feature ablation measures **predictive information contribution**. It does not
establish biological causality, molecular necessity, or mechanistic Cas9
dependence. Permitted wording is "removing this feature family reduced
predictive performance," not "this feature family causes CRISPR activity."

## Canonical Feature Registry

Implementation truth is
`SequenceFeatureExtractor().get_feature_names()`, consumed without fitting by
`src.multidomain.data.materialize_tabular`. It yields exactly 197 unique ordered
features. The canonical ordered-list SHA-256 is
`9adf5c5e285338ac3cf4bfd3987b1245ae61f7a18d2cbf78f6b96d6079c84dab`.
The generated Phase 18E JSON contains every name in canonical order and every
family membership in full.

| Family | Count | Exact membership rule in canonical order |
|---|---:|---|
| `GC_AND_SKEW` | 10 | `gc_content`, `gc_5prime_context`, `gc_guide`, `gc_pam`, `gc_3prime_context`, `gc_full`, `gc_guide_skew`, `gc_skew`, `at_skew`, `is_optimal_gc` |
| `NUCLEOTIDE_COMPOSITION` | 7 | `freq_A`, `freq_C`, `freq_G`, `freq_T`, `purine_content`, `pyrimidine_content`, `heterogeneity` |
| `DINUCLEOTIDE_FREQUENCY` | 16 | `dinuc_AA` through `dinuc_TT`, Cartesian order over alphabet A/C/G/T |
| `GLOBAL_KMER` | 80 | all 16 `k2_*` frequencies followed by all 64 `k3_*` frequencies, Cartesian order over A/C/G/T; entropy/complexity excluded |
| `ENTROPY_COMPLEXITY` | 4 | `k2_entropy`, `k2_complexity`, `k3_entropy`, `k3_complexity` |
| `POSITION_SPECIFIC_NUCLEOTIDE` | 80 | `guide_pos_0_A/C/G/T` through `guide_pos_19_A/C/G/T`, position-major and A/C/G/T order |

The registry is a complete disjoint partition: its union equals the canonical
197 names and every pairwise family intersection is empty. Any identity,
ordering, count, overlap, or completeness change is a stop condition.

## Registered Arms

The primary design is `FULL` versus one family removed at a time, independently
within each domain:

- `FULL`
- `FULL - GC_AND_SKEW` (187 features)
- `FULL - NUCLEOTIDE_COMPOSITION` (190 features)
- `FULL - DINUCLEOTIDE_FREQUENCY` (181 features)
- `FULL - GLOBAL_KMER` (117 features)
- `FULL - ENTROPY_COMPLEXITY` (193 features)
- `FULL - POSITION_SPECIFIC_NUCLEOTIDE` (117 features)

Two secondary family-only arms are retained: `GLOBAL_KMER` only (80 features)
and `POSITION_SPECIFIC_NUCLEOTIDE` only (80 features). They answer the distinct
sufficiency question of order-agnostic motif spectrum versus spacer-position
identity. They are descriptive, not part of the primary contribution
classification. Smaller family-only models, individual-feature removals,
recursive elimination, SHAP/importance-guided selection, all family
combinations, and post-ablation rescue tuning are excluded.

The Phase 18D ranking of `guide_pos_19_G` does not authorize an individual
position ablation. The entire position-specific family is removed in the
registered primary arm.

## Domains and Split

Future fits are independent within DeepSpCas9 and CRISPRon Xiang/Luo. Labels
remain in native units and domain labels are never pooled. Raw performance on
the two label scales is not compared.

The exact Phase 18C forward-spacer group split is reused without modification:

`b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8`

It retains the locked 70/15/15 train/validation/internal-test allocation.
Bridge and quarantine observations remain excluded from development and from
the ablation evaluation. No split is regenerated.

## XGBoost and Seed Policy

The locked configuration is 100 estimators, depth 6, learning rate 0.1,
`min_child_weight=1`, `subsample=1.0`, `colsample_bytree=1.0`, no L1 penalty,
L2 penalty 1.0, squared-error regression, histogram trees on CPU, four jobs,
and project seed 42. There is no early stopping, validation-based selection,
sample weighting, tuning, or refit on train plus validation.

Only one fit is registered for each domain/feature arm. With full row and
feature sampling, Phase 18C observed identical XGBoost outcomes at seeds 42-46;
five deterministic refits do not estimate stochastic uncertainty. Seed 42 is
retained rather than silently changing the project convention. Statistical
uncertainty is estimated by paired resampling of test groups, not redundant
model seeds.

## Metrics and Effects

- Primary: Spearman rho.
- Regression guardrail: RMSE.
- Secondary: Pearson, R-squared, and MAE.
- `delta_Spearman = Spearman_ablation - Spearman_full`.
- `delta_RMSE = RMSE_ablation - RMSE_full`.
- Relative RMSE degradation is `(RMSE_ablation - RMSE_full) / RMSE_full`.

Negative delta Spearman and positive delta RMSE indicate degradation. All
metrics and effects are calculated separately by domain.

## Paired Bootstrap and Multiplicity

Each drop-one arm is compared only with its corresponding full XGBoost control
on identical internal-test observations. The analysis uses 10,000 paired
percentile bootstrap replicates over sorted unique exact forward spacer groups,
with NumPy PCG64 seed 42. Groups are sampled with replacement; every observation
of a sampled group is retained, and the same draw indexes full and ablated
predictions. Nominal confidence is 95%.

Multiplicity is locked before results. Bonferroni simultaneous percentile
intervals cover 24 confirmatory intervals: six families by two domains by two
endpoints (Spearman and RMSE). Familywise alpha is 0.05, per-interval alpha is
0.0020833333, interval confidence is 99.7916667%, and two-sided quantiles are
0.0010416667 and 0.9989583333. Secondary metrics and family-only arms receive
nominal 95% descriptive intervals only.

## Contribution Decisions

No minimum effect-size or equivalence threshold is claimed because a
scientifically justified prospective margin is not available. Categories use
the locked simultaneous intervals:

- `ESSENTIAL_OR_STRONG_CONTRIBUTOR`: adjusted upper CI for delta Spearman is
  below zero and adjusted lower CI for delta RMSE is above zero. "Essential"
  refers only to predictive information under this model and feature set.
- `MODERATE_CONTRIBUTOR`: exactly one adjusted interval supports degradation,
  the other point estimate is directionally concordant, and neither endpoint
  supports improvement.
- `LITTLE_UNIQUE_CONTRIBUTION`: neither adjusted endpoint supports degradation
  or improvement. This means no clear unique contribution was detected, not
  equivalence or absence of information.
- `POTENTIALLY_REDUNDANT`: at least one adjusted endpoint supports improvement
  after removal and neither supports degradation.
- `UNSTABLE_OR_INCONCLUSIVE`: undefined/nonfinite metrics, invalid bootstrap
  replicates, or supported conflict between endpoint directions.

These same rules answer the preregistered questions for position-specific,
global k-mer, GC/skew, dinucleotide, nucleotide-composition, and
entropy/complexity families. No family receives a post-result threshold.

## Domain Consistency

- `CROSS_DOMAIN_CONSISTENT`: both domains receive a strong or moderate
  contributor classification with concordant degradation directions.
- `DOMAIN_A_ENRICHED`: only DeepSpCas9 receives a contributor classification.
- `DOMAIN_B_ENRICHED`: only Xiang/Luo receives a contributor classification.
- `WEAK_IN_BOTH`: both domains receive little-unique-contribution or
  potentially-redundant classifications.
- `UNSTABLE_OR_INCONCLUSIVE`: either domain is inconclusive or directions
  conflict.

`DOMAIN_A_ENRICHED` and `DOMAIN_B_ENRICHED` describe asymmetry of evidence; they
are not formal proof that native-scale effect magnitudes differ between assays.

## Redundancy Rule

GC, nucleotide composition, k-mer frequencies, and position indicators encode
overlapping sequence information. A small drop-one impact may mean retained
families compensate for the removed family. It does not show that the removed
family lacks useful biological information. Likewise, feature importance and
ablation effects are not biological causality.

## Matrix and Compute

The locked matrix contains 18 future fits: two domains, two full controls, 12
drop-one-family ablations, and four optional family-only fits. Every JSON matrix
record includes experiment ID, domain, removed family, remaining feature count,
full canonical XGBoost configuration, split hash, metric policy, bootstrap
policy, and hypothesis.

Phase 18C recorded approximately 6.75 seconds for 10 XGBoost fits. On comparable
hardware, 18 fits should require on the order of tens of seconds; the 10,000
bootstrap replicates may dominate analysis time. This is an estimate, not an
execution result.

## Isolation and Boundary

The locked external dataset is prohibited from opening, reading, searching,
hashing, statistics, training, evaluation, arm selection, and thresholds. The
Phase 18E runtime guard fails before locked-name file access and rejects model
training or prediction calls. Protocol construction reads only the approved
Phase 18B protocol artifact and allowlisted source files needed to verify the
feature, split, and configuration locks.

Phase 18F must not begin without explicit approval of this protocol artifact.
