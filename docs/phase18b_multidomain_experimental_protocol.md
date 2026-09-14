# Phase 18B — Multi-domain experimental protocol and preregistration

**Decision: READY_FOR_PHASE18C**, conditional on the accompanying deterministic
audit passing and supervisor approval before implementation. This is a locked
design, not evidence of model benefit. Phase 18C is not started.

## 1. Evidence and formal question (A)

Authoritative sources inspected:

- `docs/phase17g_r1_data_sufficiency_reassessment_report.md` and both retained
  final reassessment JSONs (`20260911_233322`, `20260911_233749`). The later JSON
  adds the official-code `Quant_norm_efficiency` provenance caveat. The accepted
  candidate, counts and GO decision agree.
- `docs/phase18a_cross_dataset_compatibility_report.md` and
  `results/phase18a_cross_dataset_compatibility_20260912_160313.json`.
- `README.md`, `config.yaml`, `docs/PROJECT_MASTER_HANDOVER.md`, and
  `docs/decisions.md`, especially D-004 through D-008.

No conflict was found between the supplied Phase 18A scientific summary and its
authoritative report/JSON. The supplied historical test totals are not embedded
in those artifacts; this phase reports its own checks separately.

Domain A is DeepSpCas9, canonical eligible n=10,117. Domain B is CRISPRon
Xiang/Luo, accepted n=10,592 before overlap holding, with 10,544 new observations
relative to raw DeepSpCas9. Kim contributes zero new data. Corsi remains
AUXILIARY_ONLY and is outside this matrix.

Let `X ∈ {A,C,G,T}^30`, with positions `[0:4]` upstream context, `[4:24]` spacer,
`[24:27]` PAM and `[27:30]` downstream context. For domain `d`, observations
are `(X_i, y_di, d)`. The target functions are distinct conditional expectations
`f_A(X)=E[y_A|X,A]` and `f_B(X)=E[y_B|X,B]`; neither the target random variables
nor their numerical values are assumed equal.

- **Shared biological concept:** sequence-linked SpCas9 on-target indel
  efficiency in HEK293T integrated-surrogate assays.
- **A label:** local `activity`, an uncorrected day-2.9 indel fraction derived
  from `modFreq`. The publication also describes a corrected endpoint; that is
  not silently substituted for the repository label.
- **B label:** workbook-native `HEK293T_indel_freq_avg_d8_d10`, a Dox-free mean
  day-8/day-10 indel percentage after background-variant removal.
- Timing, selection, Cas9 delivery/expression, library composition, background
  handling and experimental response mapping remain domain-specific. The
  absent `Quant_norm_efficiency` field is not reconstructed or inferred.

**Primary hypothesis:** a shared sequence encoder may learn complementary
sequence determinants of SpCas9 indel activity while separate prediction heads
preserve each assay's label mapping, improving ranking in at least one domain
without meaningful ranking or regression degradation in either domain.

Phase 18A supports testing, not accepting, this hypothesis: same broad concept,
usable diverse sequences and a positive canonical bridge association (rho
approximately 0.747) coexist with six major operational differences. Shared
representation can cause negative transfer. The small bridge, already examined
in Phase 18A, cannot establish common numeric targets or prove generalization.

## 2. Minimum arm set and alternative assessment (B, C, Q)

| Arm | Training | Architecture / domain handling | Hypothesis and benefit | Risk and interpretation |
|---|---|---|---|---|
| A | A train only | Canonical-capacity CNN encoder + A linear head | Establish single-domain reference and no-sharing ablation | Domain-specific overfitting; appropriate comparator for D on A |
| B | B train only | Identical encoder + B linear head | Independent development-domain comparator; no label equivalence needed | Domain-specific overfitting; appropriate comparator for D on B |
| D, primary | A and B train | Shared encoder + separate A/B linear heads, explicit domain routing | Complementary representation with preserved endpoints | Negative transfer or oversharing; test against both A and B |
| RF_A, RF_B | Respective domain only | Independent fixed Random Forest on 197 features | Established tabular reference within each domain | Different representation; descriptive comparison, not neural multi-domain evidence |
| XGB_A, XGB_B | Respective domain only | Independent fixed XGBoost on 197 features | Established boosted-tree reference within each domain | Different representation; descriptive comparison only |

All inputs, domain targets, architectures, losses, weights, hypotheses, expected
benefits, risks and interpretation fields are repeated per run in the JSON
matrix. CNN losses are specified below; trees use native squared-error raw
regression. No model is warm-started from a canonical binary.

**C, naive pooling: excluded.** A raw single-head mixture would confound sharing
with a roughly 100-fold unit difference and contradictory endpoint semantics.
Failure would be unsurprising and uninformative about useful shared sequence
information. It is not needed to identify D versus independent A/B learning.

**E, encoder domain conditioning: deferred.** It asks whether assay identity
must alter intermediate sequence features rather than only the response
mapping. This is distinct but adds interaction/capacity choices before the
simpler design has been tested.

**F, transfer learning in either direction: deferred.** Sequential adaptation
tests initialization and forgetting, not simultaneous complementary learning.
It adds direction, freezing and sequential-budget confounders.

Paired A/B models already supply the separate-encoder comparator. No extra
separate-encoder arm, loss-weight grid, auxiliary rank loss, architecture grid
or post-result rescue experiment is allowed. Any expansion requires a new
approved preregistration, not an amendment after looking at internal tests.

## 3. Primary architecture and output semantics (C, F, G)

Reuse the architecture actually present in `src/models/cnn.py`, including its
MaxPool layer (not just the abbreviated configuration comment):

1. A/C/G/T one-hot `(n,30,4)`, transposed to `(n,4,30)`.
2. Three `Conv1d(4,64,k)` branches, k=5,7,9, padding `k//2`, each followed by
   ReLU, MaxPool1d(2, stride=2), and AdaptiveAvgPool1d(1).
3. Concatenate to 192 dimensions; Linear(192,64), ReLU, Dropout(0.3).
4. A single Linear(64,1) for each represented domain. D shares steps 1–3;
   A and B each have their own encoder and their own head.

For a domain d, define future raw output
`yhat_d = mu_train_d + sd_train_d * h_d(g_theta(X))`.
The affine output constants are computed in Phase 18C **only from that domain's
locked training labels**, in float64 with population variance (`ddof=0`), and
stored as fixed buffers. Targets remain unmodified raw labels. This is numerical
output preconditioning, algebraically equivalent to optimizing a domain-local
standardized residual; it is disclosed explicitly, not presented as absence of
all scaling. It asserts no between-domain measurement equivalence. It prevents
the B head requiring 100-fold larger trainable weights merely because of units.

The observed domain ID is required. A mixed forward pass routes each row to its
own head, returning a raw-scale scalar per row in original order. Unknown or
missing domain IDs fail. Only the corresponding head receives that row's loss;
the encoder receives both domain gradients. No inferred-domain classifier,
cross-head target loss, bridge-alignment regularizer, sigmoid or clipping.

Analytic parameter counts (verify actual counts in Phase 18C): convolutions
5,568; shared dense 12,352; encoder total **17,920**. Each head adds **65**.
A/B each have **17,985** trainable parameters; D has **18,050**, just 65 extra
(approximately 0.36%). A+B jointly have 35,970. Output buffers add no trainable
parameters. Report counts, initialization, runtime and peak memory in 18C.

### Label-policy alternatives (F)

| Policy | Assumption | Advantages | Information loss / reversibility | Interpretation risk / Phase 18A compatibility |
|---|---|---|---|---|
| **Raw domain regression, selected** | Separate continuous endpoints; squared residuals meaningful within domain | Native-unit outputs and unchanged source labels | No target information loss; identity target mapping | Compatible with explicit scale-balanced loss and disclosed affine output; no cross-domain raw-error interpretation |
| Domain-specific standardized targets | Train-local location/scale is numerical conditioning, not biology | Easier optimization and balanced units | Invertible with saved train mean/SD; no information loss in exact arithmetic | Compatible with separate heads; not equivalent endpoints. Selected output/loss preconditioning is algebraically equivalent, but target arrays are kept raw |
| Within-domain rank/percentile target | Relative ordering is the sole target and training library defines reference | Scale-invariant prioritization | Loses interval and absolute response information; generally not invertible | Changes prediction target; only possible with caution under Phase 18A; not selected |
| Auxiliary rank loss plus raw regression | Pairwise order should receive additional optimization weight | May align ranking objective with primary metric | Raw component retains values; rank component discards distances; not a target conversion | Compatible in principle, but pair sampling and loss strength add tuning and tradeoffs; not selected |

No standardization, rank conversion, train-statistic computation, calibration,
or loss execution occurs in Phase 18B.

### Loss and domain weights (G)

`L_d = mean_i[(yhat_di - y_di)^2] / var_train_d`.

For D, `L_total = 0.5 L_A + 0.5 L_B`. Single-domain CNNs use `L_d` with weight
1.0 and the same output preconditioning. A zero/nonfinite SD or SD <=1e-12 in
native units stops; no arbitrary epsilon floor replaces a degenerate domain.

Equal-domain weighting is selected. Sample-proportional weighting would change
domain influence with dataset size; unnecessary here. Balanced batches implement
equal-domain exposure but **do not by themselves solve label-scale dominance**.
Domain-local train variance and affine output remove unit/initial-scale
dominance; equal per-domain means remove sample-count dominance. Residual
differences in task difficulty and gradient conflict remain scientific risks,
not grounds for post-hoc reweighting. Log per-domain losses; no adaptive weights.

## 4. Deterministic split and bridge policy (D, E, N)

The historical canonical 85/15 split and trained artifacts remain historical.
They cannot be reused as supposedly clean references on a new test partition:
their original training set would contain many new internal-test rows. All
Phase 18C references are fresh experimental fits.

**Final grouping unit: exact forward 20-nt spacer `[4:24]`, globally across
both domains.** Exact 30-mer grouping is necessary but insufficient: different
context/PAM rows can reuse an identical guide. Spacer grouping protects that
stronger dependency without inventing a similarity threshold. It does not
guarantee gene-, locus-, or near-homology independence. Reverse complement
grouping and approximate matching remain unapproved and are not performed.

Before splitting, hold all spacer groups belonging to the **48 raw exact
overlaps** out of train, validation and internal test in both domains:

- The **41 canonical exact pairs** are assigned `bridge` (41 A and 41 B rows).
- The **7 remaining B rows** overlapping canonical-ineligible raw A sequences
  are assigned `quarantine`; they are not novel observations and are not used
  for model development or the paired canonical bridge analysis.
- All other rows sharing any of these 48 spacers would also be quarantined.
  Unexpected additional relatives stop this locked version for review.
- No cross-domain labels are averaged or deduplicated into a common target.

Each remaining group receives an independent deterministic hash allocation:
ASCII `phase18b-multidomain-v1|42|<forward spacer>`, SHA-256 interpreted as a
big-endian unsigned integer H. Assign train if `100H < 70*2^256`; validation
if `70*2^256 <= 100H < 85*2^256`; otherwise internal test. This is a fixed
**70/15/15 expected allocation**, not a promise of exact counts. No label
stratification, input-order dependence, search over split seeds or count repair.
The single split seed is **42**, distinct in role from optimization repetitions.

A uses the existing Phase 18A canonical mask (including its established
homopolymer handling), n=10,117. B uses the accepted Phase 17 population with
valid 30-mer geometry and no newly imposed homopolymer exclusion. Applying the
A filter anew to B would silently reduce the accepted development domain;
that population decision is explicitly not part of this experiment. All arms
within each domain use identical eligibility and preprocessing. Do not rerun a
generic preprocessor that changes these populations, reconstructs contexts or
silently drops rows. Enforce the recorded counts and hashes instead.

The accompanying artifact locks every row with domain, SHA-256 of the exact
sequence, SHA-256 of its spacer and partition. It includes counts, group counts,
the manifest hash and sorted sequence/label fingerprints. No labels, training
arrays or model predictions are written to the manifest. Phase 18C must
reconstruct it exactly before fitting; no regeneration into a different split.

Verified membership counts:

| Domain | Train | Validation | Internal test | Bridge | Quarantine | Total |
|---|---:|---:|---:|---:|---:|---:|
| A: DeepSpCas9 | 7040 | 1510 | 1526 | 41 | 0 | 10117 |
| B: Xiang/Luo | 7465 | 1520 | 1559 | 41 | 7 | 10592 |

A contains 10,116 unique spacers across its 10,117 distinct 30-mers, confirming
one within-domain spacer repetition; B has 10,592 unique spacers. The main B
partitions sum to 10,544. These counts are now explicit fail-fast constants.

## 5. Batching, fairness, seeds and stopping (H, I, M–P)

**Optimization seeds: 42, 43, 44, 45, 46 (five).** Seed 42 reuses the canonical
convention; the four additional consecutive seeds are an explicit new fixed
replication set. No best-seed reporting, dropped bad seeds or replacement seeds.

Every CNN arm uses batch size 32 and the same maximum update budget. For D,
each batch has 16 A and 16 B rows. Accumulate both weighted losses before one
Adam update. A/B each use 32 rows of their own domain. An experimental epoch
contains `ceil(max(n_train_A,n_train_B)/16)` optimizer steps for **every** CNN
arm, rather than an arm-dependent pass over a dataset.

Sample sorted IDs within a domain by without-replacement permutations, cycling
and re-permuting upon exhaustion, including across full-batch boundaries. Use
NumPy PCG64 initialized with `SeedSequence([run_seed, domain_index, epoch, cycle])`,
A index 0, B index 1; epochs start at 1 and cycles at 0, resetting each epoch.
No weighted guide sampling. This defines unequal-size handling and full batches
without nondeterministic data-loader behavior.

All CNNs see the same maximum total examples and optimizer steps. Single-domain
models see twice D's exposure to their own domain per epoch, a conservative
comparison for D; this tradeoff is reported. D's additional domain information
is the treatment. Early stopping can shorten a run, but the maximum budget and
stopping framework are identical. Record realized exposures, updates and time.

Reuse Adam, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0; max epochs
100; patience 10. No scheduler, augmentation, weight search, architecture search,
calibration, rank auxiliary, pretrained checkpoint, or train+validation refit.

Initialize fresh PyTorch-default Conv/Linear encoder parameters with run seed.
Initialize each head on an independent RNG stream seeded
`run_seed + 1000 + domain_index`; this pairs A's head with D's A head, and B's
head with D's B head. Reset the training/dropout RNG to run seed after
construction. Use CPU float32 model computation, four threads, deterministic
algorithms, zero loader workers, and seed Python/NumPy/PyTorch. Log environment
versions and stop on unavailable deterministic operations.

After each epoch evaluate the **whole validation partitions**, in inference
mode without dropout, using train-only variance denominators. A/B use their
domain objective; D uses the equal-domain mean. Never average raw A/B MSE.

- Patience resets only when `val_loss < significant_best - 0.0001`; update
  significant_best on reset. Initialize it to +infinity.
- Independently retain the checkpoint with the lowest unrounded validation
  objective; earliest epoch wins exact ties. Stop after 10 non-reset epochs or
  epoch 100, whichever comes first. Restore the retained checkpoint.
- Internal tests and bridge are evaluated only after **all** checkpoints and
  analysis code are frozen. No test-informed model selection or early stopping.
- A nonfinite training/validation objective halts the run and campaign for
  diagnosis. Preserve partial logs; no silent replacement. Numeric instability
  yields UNSTABLE_RESULT unless a protocol breach caused it.

Phase 18C may implement the required domain-aware trainer, but Phase 18B does
not. Architecture and optimizer are reused; the new grouped three-way split,
output/loss preconditioning and step-normalized epochs are explicit experimental
changes applied consistently to all CNN comparators.

## 6. Random Forest and XGBoost (J)

Both remain independent domain reference regressors on the existing 197
engineered sequence features. Reuse feature extraction without fitted scaling,
feature selection or label transformation. Train only on the locked train IDs.

- RF: 200 trees, max_depth=20, min_samples_split=5, min_samples_leaf=2,
  max_features=sqrt; squared_error, bootstrap=True and remaining existing
  wrapper defaults, random_state=run seed.
- XGB: 100 estimators, max_depth=6, learning_rate=0.1,
  objective=reg:squarederror; remaining existing wrapper defaults,
  random_state=run seed.
- Lock four CPU threads for both tree families (a resource setting rather than
  a model search), and CPU `tree_method=hist` for XGBoost. XGB remaining explicit
  values are min_child_weight=1, subsample=1, colsample_bytree=1, reg_alpha=0,
  reg_lambda=1. The machine-readable policy records these values.
- Both use all five seeds, no validation early stopping or model selection
  (D-005). Validation is descriptive. Record complete resolved parameters and
  installed library versions; exact wrapper source hashes are locked.

The CNN supplies the primary representation-sharing experiment. RF/XGB cannot
be forced into domain-specific neural heads, and historical canonical metrics
are not substitutes for clean internal-test baseline fits.

## 7. Metrics, statistics and outcome definitions (K–M, R)

Compute each domain independently on its own internal test, in native units:

- **Primary endpoint:** Spearman rho (higher better), chosen for sequence
  prioritization and invariance to the distinct label units.
- **Regression preservation guardrail:** relative RMSE gain, defined per seed
  as `(RMSE_single - RMSE_D)/RMSE_single`; higher better. Baseline RMSE=0 makes
  this comparison undefined and must be reported as unstable, not epsilon-fixed.
- **Secondary metrics:** Pearson, R², MAE, RMSE. RMSE is also the guardrail above.
  Report raw RMSE/MAE separately for A and B; do not average their raw values.

Use existing `src/evaluation/metrics.py` definitions where applicable, but check
for constant labels/predictions and nonfinite metrics explicitly. Undefined
correlations must not become a silent zero. Report per-seed metrics plus the
arithmetic mean and SD, never metrics from a post-hoc seed ensemble.

Primary comparisons are **D versus A on A** and **D versus B on B**, paired by
seed and identical held-out observations. Define per-seed delta rho as D minus
the relevant baseline and relative RMSE gain as above.

Future inferential plan (not executed in Phase 18B):

1. 10,000 hierarchical paired bootstrap replicates, NumPy PCG64 seed **42**.
2. Each replicate first samples five paired seed indices with replacement,
   using the same seed draw across domains. Then, in A followed by B order,
   sample that domain's sorted unique internal-test spacer groups with
   replacement, drawing as many groups as originally observed.
3. Keep all rows belonging to each sampled group, including multiplicities.
   Use the same row/group resample for both models and every seed within a
   domain. Recompute each seed's correlations and RMSEs on that resample;
   average the five paired deltas/gains using the sampled seed indices.
4. Use percentile intervals with NumPy linear quantile interpolation. Report
   nominal 95% descriptive intervals. For the four decision endpoints (two
   domains × rho and RMSE gain), use **98.75% two-sided intervals**, quantiles
   0.00625 and 0.99375: Bonferroni familywise nominal 95% coverage.
5. Any undefined primary replicate is counted, not silently resampled or
   discarded; the decision becomes UNSTABLE_RESULT. Report distributions and
   invalid count. No additional hypothesis search based on these results.

Reuse the repository metric utilities, not its unpaired single-metric bootstrap
as a substitute for the paired grouped hierarchical comparison. Five seeds and
one fixed split give manageable approximate uncertainty, not a prospective
power guarantee or proof of gene-level generalization. Secondary tree
comparisons may receive descriptive 95% paired intervals under the same group
scheme; they do not alter the primary outcome or create confirmatory claims.

### Locked practical margins

These are **prospective project tolerances**, not biological constants or
thresholds estimated from Moreno, bridge concordance or Phase 18C performance:

- rho preservation margin: **0.01** absolute correlation units.
- minimum ranking benefit: **0.01** mean delta rho, with decision CI lower >0.
- regression harm margin: **2%** relative RMSE increase (gain >=−0.02 preserved).
- instability: per-domain delta-rho seed range >=**0.10**, or relative-RMSE-gain
  seed range >=**0.20**. Missing/undefined seed results also imply instability.

Apply outcome categories in this exact precedence:

1. **PROTOCOL_VIOLATION:** any integrity, access, leakage, mutation, geometry,
   overwrite, count, unregistered-arm/seed/tuning or split-determinism breach.
   Results cannot support scientific benefit; stop immediately.
2. **UNSTABLE_RESULT:** missing/nonfinite required values, invalid decision CI or
   bootstrap replicate, numerical training failure, or either seed-range limit
   met. Do not replace seeds or hide the failed run.
3. **NEGATIVE_TRANSFER:** any domain has delta-rho decision CI **upper <−0.01**
   or RMSE-gain decision CI **upper <−0.02**. This includes both-domain harm and
   one-domain harm despite improvement in the other. Also report mean-level
   directional harm flags (mean delta rho <−0.01 or mean RMSE gain <−0.02),
   even if uncertainty prevents this categorical finding.
4. **MULTI_DOMAIN_BENEFIT:** in **both** domains, delta-rho decision CI lower
   >=−0.01 and RMSE-gain decision CI lower >=−0.02; at least **4/5 paired seeds
   per domain preserve both margins**; and at least one domain has mean delta
   rho >=0.01 with decision CI lower >0.
5. **NO_CLEAR_BENEFIT:** every remaining stable, valid result. This includes
   preservation without demonstrated improvement and uncertainty about harm.

These conjunctive gates prevent a large gain in one domain masking a large
loss in the other. No pooled metric score or tree result can rescue a failed
domain. `classify_outcome` validates synthetic summaries only in Phase 18B.

## 8. Canonical bridge evaluation (S)

After all checkpoints and primary analyses are frozen, evaluate both D heads
and the paired A/B reference models on the same 41 canonical sequences. Bridge
membership never participates in fitting statistics, early stopping, threshold
selection, calibration or architecture selection.

Report head-to-head Spearman and Kendall, each head's Spearman and Pearson
against its **own** observed labels, and mean absolute percentile-rank
disagreement between heads. Rank ties use average ranks; percentile is
`(rank−1)/(41−1)`. Compare D head-pair coherence with the corresponding A/B
model pair and show per-seed values/ranges. Constant-output coherence is
undefined, not evidence of agreement. No cross-domain raw MAE or fitted label
mapping, no bridge-based success gate, no follow-up tuning.

This is a small diagnostic set already label-audited in Phase 18A. Increased
head consistency alone may reflect forced agreement or common error. It must
be interpreted alongside own-label fidelity and the main internal tests.

## 9. Matrix and compute budget (T, U)

Seven arms × five seeds = **35 planned independent fits**:

| Family | Arms | Seeds per arm | Runs |
|---|---|---:|---:|
| CNN | A, B, D | 5 | 15 |
| Random Forest | RF_A, RF_B | 5 | 10 |
| XGBoost | XGB_A, XGB_B | 5 | 10 |

No transfer pretraining, final refit or tuning repetitions are hidden in this
count. CNN cost is bounded by 15 equal-capacity, equal-step, at-most-100-epoch
runs, plus 20 fixed tree fits. The artifact derives the exact step budget from
the locked train counts. No empirical wall-time estimate is claimed because
Phase 18B does not benchmark training. Bridge inference needs no additional fit.
The locked count is **467 optimizer steps per CNN epoch**, at most **46,700**
per CNN run and **700,500** over all 15 CNN runs before early stopping.

## 10. Integrity, execution boundaries and fail-fast plan (V)

Reuse Phase 18A's loader, canonical mask, corrected RF hash and integrity
utilities. Phase 10 pooling is unsuitable because it concerns a different
dataset/label policy and historical external-evaluation protocol. A separate
small `src/experiment_protocols/` module avoids changing that architecture.

The Phase 18B builder checks all five Phase 18A pinned safe inputs/artifacts
before reading allowed datasets and after construction. It verifies upstream
decisions, source components, counts, overlap and diversity; it records source
code/config/report hashes, label fingerprints and the exact split manifest.
The JSON includes a deterministic protocol hash computed from the whole payload
excluding its `protocol_sha256` field. Timestamp is in the filename only; repeat
builds under the same sources are byte-identical. Exclusive creation prevents
accidental replacement of a previous result.

Runtime guards reject Moreno/Mateos path names before open, including resolved
aliases; all data/model opens are restricted to the five pinned allowlisted
paths, and data/model writes are rejected. A Python audit hook intercepts ordinary
dependency file opens; a profile guard rejects training/calibration entry points.
These are process-level controls, not an operating-system sandbox. Unknown
aliases inside data directories are rejected by the allowlist, not merely by
their names. Tests use nonexistent synthetic locked paths to verify denial;
the real locked external file is never opened.

Before Phase 18C fitting, the approved implementation must:

1. Validate the protocol hash against the approved recorded digest, not just a
   self-consistent rehash; verify all pinned data, binary, source/config and
   protocol hashes. Version any approved implementation separately.
2. Reconstruct the split and compare every ID, count and fingerprint; rerun
   both exact-30mer and spacer leakage checks, including bridge relatives.
3. Verify sorted sequence/raw-label fingerprints before tensor conversion and
   after preprocessing. Preserve raw float64 labels separately from float32
   model tensors; no silent mutation, clipping, filtering or geometry change.
4. Instantiate data loaders from explicit A/B train/validation allowlists;
   internal-test and bridge loaders remain closed until checkpoint freeze.
5. Keep Moreno inaccessible for the **entire** Phase 18C campaign. No distribution
   comparison, split design, tuning, calibration or evaluation involving it.
6. Create exclusive run directories only under `results/phase18c/<protocol_hash>/`;
   all new model weights and predictions go there. No writes under canonical
   `models/`, `results/predictions/`, `results/experiments/`, raw data or config.
7. Fail before the offending read/write/update on any stop condition; retain a
   separate violation log, halt the campaign, require review, and never silently
   repair counts, reseed, regenerate canonical artifacts or promote models.
8. Check protected hashes again at campaign completion. Freeze environment,
   analysis code, resolved hyperparameters and output destinations before fits.

The Phase 18C trainer and its output guards are future implementation work.
Phase 18B has implemented protocol validation and audit controls only.

## 11. Verification and stop

Deterministic focused tests cover canonical integrity, locked access denial,
exact and spacer grouping, cross-domain leakage, bridge exclusion, deterministic
splits, matrix/seeds, metric/loss/success policies, negative transfer and
serialization. The accompanying verification record gives exact commands and
results, including historical regressions and compile/lint/diff checks.

The unrestricted existing full suite contains actual CNN/RF/XGB fitting and
calibration; running those tests would violate this phase. Use an explicitly
reported no-training subset instead, and do not claim the historical 570-pass
full-suite result as this phase's result.

**STOP after Phase 18B. Supervisor approval is required before Phase 18C.**
