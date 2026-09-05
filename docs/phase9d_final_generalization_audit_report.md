# Phase 9D — Final Generalization Audit

**Status: AUDIT — NOT COMMITTED**
**Date:** 2026-09-05
**Run:** `scripts/run_phase9d_audit.py`
**Result JSON:** `results/experiments/phase9d_final_generalization_audit_20260905_235023.json`
**git HEAD at run:** `7a9af4a` (clean canonical lineage, no canonical artifact modified)

---

## 1. Objective

Answer: **"After Phase 3–9C, what is the strongest scientifically defensible
conclusion about the generalization behavior of the current CRISPR activity
prediction models?"**

Phase 9D is a read/audit/synthesis layer. It performs no fitting, no tuning,
no calibration, no adaptation, and no external iteration. Predictions are
recomputed from the frozen canonical artifacts and **verified exactly** against
the metrics stored in the canonical result JSONs before any audit statistic is
reported.

## 2. Canonical Models and Frozen Evaluation Policy

| Aspect | Value |
|---|---|
| Models | Random Forest (200 trees, max_depth 20, sqrt) · XGBoost (100, depth 6) · 1D CNN (64 filters, [5,7,9], dense 64, dropout 0.3) |
| Training | DeepSpCas9, 85% split (seed 42) |
| Validation | DeepSpCas9 15% (seed 42) — model selection / CNN early stopping only |
| External test | Moreno-Mateos (810 valid, LOCKED) |
| External use | Final evaluation **only**; never for fit/tune/select/calibrate/adapt |

**Metric reproducibility (Audit H / integrity):** recomputed vs stored
canonical metrics on validation and external, for all three models:
**max absolute delta = 0.00** for MAE, RMSE, R², Pearson, Spearman.
The audit therefore operates on provably identical predictions to those that
produced Phases 3–8 artifacts.

## 3. Internal vs External Performance

Scale-sensitive errors roughly double externally while R² collapses:

| model | set | MAE | RMSE | R² | Pearson | Spearman | Kendall | pred-SD |
|---|---|---|---|---|---|---|---|---|
| RF | validation | 0.150 | 0.179 | 0.362 | 0.638 | 0.626 | 0.448 | 0.095 |
| RF | external | 0.249 | 0.287 | 0.046 | 0.235 | 0.226 | 0.152 | 0.075 |
| XGB | validation | 0.126 | 0.156 | 0.511 | 0.717 | 0.705 | 0.514 | 0.148 |
| XGB | external | 0.254 | 0.298 | −0.024 | 0.183 | 0.177 | 0.119 | 0.122 |
| CNN | validation | 0.139 | 0.171 | 0.416 | 0.647 | 0.631 | 0.452 | 0.135 |
| CNN | external | 0.252 | 0.296 | −0.011 | 0.184 | 0.160 | 0.109 | 0.115 |

**Qualitative pattern (Audit E).** Internal: moderate absolute performance and
moderate association. External: near-zero or negative R², weak but nonzero
correlation, strong prediction compression, larger errors, and a broader,
shifted activity distribution. This is **consistent with domain shift** rather
than with a specific equipment/measurement failure of the external dataset.

## 4. Activity-Range Error Analysis

Common true-activity grid `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` for both datasets
and all models (identical bins → internal/external directly comparable).
Within-bin R² is deliberately not reported: within-bin true SD ≈ 0.06 in every
bin, so bin-level R² would be variance-dominated and uninformative.

**Validation (internal):**

| model | bin | n | mean true | mean pred | bias | MAE | RMSE | true SD | pred SD |
|---|---|---|---|---|---|---|---|---|---|
| RF | low [0,0.2) | 314 | 0.107 | 0.329 | +0.223 | 0.223 | 0.240 | 0.057 | 0.089 |
| RF | low-mid | 372 | 0.298 | 0.396 | +0.098 | 0.111 | 0.132 | 0.056 | 0.081 |
| RF | mid | 439 | 0.502 | 0.450 | −0.052 | 0.079 | 0.098 | 0.058 | 0.066 |
| RF | high-mid | 355 | 0.685 | 0.489 | −0.196 | 0.197 | 0.211 | 0.056 | 0.067 |
| RF | high [0.8,1] | 38 | 0.843 | 0.529 | −0.314 | 0.314 | 0.318 | 0.035 | 0.051 |
| XGB | low | 314 | 0.107 | 0.263 | +0.157 | 0.162 | 0.197 | 0.057 | 0.127 |
| XGB | mid | 439 | 0.502 | 0.465 | −0.037 | 0.089 | 0.108 | 0.058 | 0.093 |
| XGB | high | 38 | 0.843 | 0.616 | −0.227 | 0.227 | 0.243 | 0.035 | 0.088 |
| CNN | low | 314 | 0.107 | 0.285 | +0.179 | 0.181 | 0.215 | 0.057 | 0.120 |
| CNN | mid | 439 | 0.502 | 0.458 | −0.044 | 0.093 | 0.117 | 0.058 | 0.097 |
| CNN | high | 38 | 0.843 | 0.573 | −0.270 | 0.270 | 0.284 | 0.035 | 0.085 |

**External (Moreno-Mateos):**

| model | bin | n | mean true | mean pred | bias | MAE | RMSE | true SD | pred SD |
|---|---|---|---|---|---|---|---|---|---|
| RF | low | 177 | 0.098 | 0.448 | **+0.350** | 0.350 | 0.365 | 0.058 | 0.080 |
| RF | mid | 156 | 0.503 | 0.474 | −0.029 | 0.082 | 0.097 | 0.057 | 0.069 |
| RF | high | 164 | 0.899 | 0.498 | **−0.401** | 0.401 | 0.409 | 0.058 | 0.060 |
| XGB | low | 177 | 0.098 | 0.454 | +0.356 | 0.357 | 0.383 | 0.058 | 0.127 |
| XGB | mid | 156 | 0.503 | 0.482 | −0.021 | 0.113 | 0.141 | 0.057 | 0.124 |
| XGB | high | 164 | 0.899 | 0.517 | −0.383 | 0.383 | 0.400 | 0.058 | 0.103 |
| CNN | low | 177 | 0.098 | 0.468 | +0.370 | 0.370 | 0.393 | 0.058 | 0.118 |
| CNN | mid | 156 | 0.503 | 0.490 | −0.013 | 0.104 | 0.125 | 0.057 | 0.110 |
| CNN | high | 164 | 0.899 | 0.520 | −0.379 | 0.379 | 0.396 | 0.058 | 0.105 |

**Findings (Audit A).**
- **Systematic overprediction of low activity** and **underprediction of high
  activity** in *both* datasets and *all* models — the classic
  regression-to-the-mean signature of target compression. The mid bin sits at
  ~zero bias because predictions collapse toward the central mean.
- The biases **amplify externally**: low-bin bias +0.35…+0.37 (vs +0.16…+0.22
  internally) and high-bin bias −0.38…−0.40 (vs −0.23…−0.31 internally).
- External high-bin mean prediction (0.50–0.52) is almost indistinguishable
  from external mid-bin prediction (0.47–0.49): the models essentially cannot
  separate "0.5" from "0.9" activity samples.
- Crucial coverage point: external has **164 high-bin samples (n=164, 20.3% of
  the test set)** vs only 38 internally — the external set is concentrated in
  exactly the regime where the models are weakest. Internal appraisal
  under-sampled the high-activity tail and therefore under-estimated the
  external failure mode.

## 5. Prediction Dispersion (Audit B)

| set | model | true SD | pred SD | pred/true SD ratio | Pearson | Spearman |
|---|---|---|---|---|---|---|
| validation | RF | 0.224 | 0.095 | **0.427** | 0.637 | 0.626 |
| validation | XGB | 0.224 | 0.148 | **0.660** | 0.717 | 0.705 |
| validation | CNN | 0.224 | 0.135 | **0.604** | 0.647 | 0.631 |
| external | RF | 0.294 | 0.075 | **0.253** | 0.235 | 0.226 |
| external | XGB | 0.294 | 0.122 | **0.414** | 0.183 | 0.177 |
| external | CNN | 0.294 | 0.115 | **0.390** | 0.184 | 0.160 |

OLS resolution (prediction ~ true, slope): validation 0.27 (RF), 0.47 (XGB),
0.39 (CNN) → external 0.06, 0.076, 0.072. On the external set the predictions
are nearly flat with respect to true activity.

**Interpretation.** All models show **prediction under-dispersion**: they
preserve some ordering (weak-to-moderate Pearson/Spearman) but fail to
reproduce the target dynamic range. This is *association without calibration*.
A high correlation would NOT establish accurate absolute prediction; here the
correlations are themselves weak-to-moderate, so both absolute and ranking
quality are degraded externally.

## 6. Ranking Behavior (Audit C)

| model | external Pearson | external Spearman | external Kendall |
|---|---|---|---|
| RF | 0.235 | 0.226 | 0.152 |
| XGB | 0.183 | 0.177 | 0.119 |
| CNN | 0.184 | 0.160 | 0.109 |

**Interpretation.** A weak but **nonzero** ranking signal survives externally.
The models retain *some* ability to order sgRNAs by activity, but the signallers
are 0.16–0.23 Spearman — too weak to support a claim of reliable sgRNA
selection utility on Moreno-Mateos. "Some ranking signal" and "accurate
absolute prediction" are different claims; the evidence supports the former
(weakly) and does NOT support the latter.

## 7. Model Agreement (Audit D)

Pairwise agreement on external Moreno-Mateos predictions (Pearson / Spearman):

| pair | Pearson | Spearman |
|---|---|---|
| RF vs XGB | 0.858 | 0.844 |
| RF vs CNN | 0.674 | 0.651 |
| XGB vs CNN | 0.672 | 0.652 |

**Interpretation.** RF and XGBoost agree strongly; the CNN agrees moderately
with both. High agreement does **not** prove correctness; it is consistent
with (a) shared sequence signal, (b) a shared bias toward the DeepSpCas9
mean, and (c) common prediction compression. Causality cannot be inferred.

## 8. Domain Shift Evidence (Audit E/F evidence base)

Recomputed from frozen data (consistent with Phase 9A):

| indicator | internal (validation) | external (Moreno-Mateos) |
|---|---|---|
| true activity mean / SD | 0.422 / 0.224 | 0.497 / 0.294 |
| proportion activity > 0.8 | 2.5% | 20.3% |
| proportion ≤ 0.05 | 4.3% | 5.8% |
| label KS test | — | D≈0.30, p=1.2e-17 (Cohen's d = −0.30) |
| GC mean / SD | 0.550 / 0.116 | 0.584 / 0.079 |
| GC KS test | — | p=9.6e-19 (Cohen's d = −0.33) |
| exact sequence overlap | — | **0 shared** (validation & all-DeepSpCas9; Jaccard 0) |
| k-mer profile (Pearson across k-mers) | — | 2-mer r=0.861, 3-mer r=0.835 |

The external test is **not** a random sample of the same distribution: labels
are shifted toward high activity and sequences toward high GC, with no exact
sequence overlap. The observed internal→external degradation is **associated
with** this covariate/label shift; this is a distributional association, not a
causal claim about biology.

## 9. Synthesis of Phase 9A–9C (Audit F — evidence table)

| # | Hypothesis | Evidence tested | Result | Interpretation | Status |
|---|---|---|---|---|---|
| 1 | Prediction-scale mismatch is the primary cause of external degradation | 9B: OLS-linear and variance calibration fit on internal data, OOF-evaluated, applied once externally | Calibration did not improve scale metrics (all Δ≈0 in ranking; MAE/R² flat to worse); variance correction worsened R² | Monotonic transforms cannot fix the cause; under-dispersion is a symptom, not the root | **REFUTED** |
| 2 | Domain weighting can substantially recover external performance | 9C: inverse-density reweighting (L/G/LG) retrained from scratch on DeepSpCas9 train, locked, external once | LG (primary) worse externally on all models (R²: RF 0.046→0.036, XGB −0.024→−0.071, CNN −0.011→−0.087); sensitivity arms ≤ +0.011 R², nonsignificant | Reweighting toward the target regime does not recover the gap | **REFUTED** |
| 3 | External degradation is associated with sequence/covariate distribution shift | 9A: GC/K-mer/positional + overlap analyses | Large, significant GC/covariate gaps; zero sequence overlap | Domain shift and degradation co-occur across datasets | **SUPPORTED** |
| 4 | External activity distribution differs substantially from internal data | 9A: label KS/Welch, tails; 9D: high-bin coverage 2.5%→20.3% | Highly significant label shift, d≈−0.30, broadened SD | External set is not from the internal distribution | **SUPPORTED** |
| 5 | Current models exhibit prediction under-dispersion | 9A dispersion; 9D sd-ratio 0.25–0.66, external resolution slope 0.06–0.08 | Confirmed across all models/sets | Models compress toward the internal mean | **SUPPORTED** |
| 6 | Current models retain some ranking/association signal externally | 9A/9D Pearson/Spearman | Pearson 0.18–0.24, Spearman 0.16–0.23 externally | Weak but real ranking signal | **SUPPORTED, but weak** |
| 7 | Current representation suffices for strong cross-dataset absolute prediction | Canonical R²/MAE on external | R² ≈ −0.02…0.05, MAE ≈ 0.25 | Evidence does not support this claim | **NOT SUPPORTED** |
| 8 | Representation/capacity limitation is the primary cause | Indirect only (9B/9C reject post-hoc fixes; 9D dispersion) | Not directly tested | Motivated but **PLAUSIBLE / UNTESTED** | **PLAUSIBLE / UNTESTED** |

> The refutation of calibration and weighting does **not** convert into
> "therefore architecture is the cause". Row 8 remains an untested hypothesis.

## 10. Experimental Integrity Audit (Audit G)

Static + runtime checks (no methodology was rewritten):

| Check | Finding | Severity |
|---|---|---|
| Moreno-Mateos use in training paths | Test data loaded in `train_*.py` **only** for post-fit prediction (grep-verified; `model.fit` precedes any `predict(X_test)`); never in fit/tune | No issue |
| Calibration fitting (9B) | Fit on internal validation; 5-fold OOF internally; external applied once | No issue |
| Adaptation weights (9C) | Computed on DeepSpCas9 train only; external once | No issue |
| Split reproducibility | Every phase reproduces seed-42 85/15 from raw DeepSpCas9; 9D reproduced n=1518/8599 | No issue |
| Preprocessing / geometry | Consistent 30-mer encoding, guide [4:24], PAM [24:27]; runtime feature-count guard (RF feature_names == extractor count) passes | No issue |
| External test reuse | No iterative or selection-based evaluation of Moreno-Mateos in 9A–9D | No issue |
| Canonical artifact modification | Models loaded read-only; 9D recorded SHA-256 + sizes; 9C never wrote to `models/` | No issue |
| Metric consistency | Same `calculate_all_metrics` used across phases; 9D recomputation matches stored metrics exactly (Δ=0.0) | No issue |
| CNN validation use | CNN used the internal validation split for early stopping/best-epoch selection in Phase 5 (documented then; re-flagged in 9B). Internal-data reuse only — not external contamination | Documented reuse, not leakage |
| Random seed reproducibility | Seeds recorded (split 42; CNN set-seed; 9C OOF 4242) | No issue |

**Verdict:** no critical integrity issue found. The Moreno-Mateos lockdown has
been respected across Phases 3–9D.

## 11. Reproducibility Audit (Audit H)

- Canonical artifacts exist and produce documented metrics exactly (Δ=0.0,
  see §2/§10).
- Phase 9C reproduced canonical results to float precision via its `B` arm
  (RF RMSE 5.6e-17, XGB/CNN 0.0) — the retraining pipeline is faithful and
  did not disturb the canonical baselines.
- No canonical artifact was overwritten by 9A–9C; `git log` shows additive
  commits.
- Full test suite at Phase 9D: **258 passed, 0 failed, 0 skipped** (248 prior
  + 10 new Phase 9D audit tests).

## 12. What Has Been Demonstrated

1. The canonical models are **reproducible**, and their external metrics are
   independently confirmed (Δ=0.0).
2. External degradation **exists and is severe** in absolute terms (MAE ≈ 0.25,
   R² ≈ −0.02…0.05).
3. The degradation is **associated with substantial domain shift** — label,
   GC, and k-mer distribution gaps with zero exact-sequence overlap.
4. All three models **under-disperse** predictions; external resolution slope
   is 0.06–0.08.
5. The models **overpredict low and underpredict high activity** in both
   datasets, worse externally; the high-activity tail (where the external set
   concentrates) is the weakest region.
6. A **weak but nonzero external ranking signal** survives (Spearman
   0.16–0.23; Kendall 0.11–0.15).
7. Simple post-hoc fixes — prediction-scale calibration (9B) and
   domain-weighted retraining (9C) — **do not** recover external performance.

## 13. What Has Not Been Demonstrated

- That Moreno-Mateos results are "bad data" or biologically wrong.
- That the models preserve a practically usable ranking for sgRNA selection.
- That prediction under-dispersion is the *cause* (it is a symptom).
- That recentering/reweighting failure implies **architecture is the cause**.
- That training on more data would fix the gap.
- Any causal (biological) claim from model behaviour.

## 14. Remaining Plausible Hypotheses

All **plausible and untested** — none proven by current evidence:
- **H-Augmentation (data diversity):** insufficient training-domain diversity;
  the single-source DeepSpCas9 corpus does not cover the external covariate/
  label regime (directly consistent with 9A's demonstrated shift; architectureindependent failure pattern across three model families).
- **H-Representation:** limited feature/representation capacity for
  cross-dataset activity rules.
- **H-Measurement/assay (null-ish):** irreconcilable genotype-to-phenotype
  protocols between datasets (untestable with current data alone).
- **H-Signal ceiling:** the true shared signal between datasets is weak and the
  residual is dataset-specific noise.

These are hypotheses about the *source* of the gap; §13 forbids elevating any
one of them to a conclusion.

## 15. Final Decision Gate

### Decision: **B. NEED MORE DATA / DATASET DIVERSIFICATION FIRST**

Rationale, evidence-based:
- The one **demonstrated** root-associated condition (9A) is domain shift
  between the training corpus and the target distribution.
- Post-hoc remedies on a fixed corpus (calibration 9B, domain weighting 9C)
  are **refuted** — so the remaining actionable uncertainty sits in the
  training-*data* domain, not in cheap loss/transform adjustments.
- The failure pattern is **architecture-independent** (RF, XGB, CNN all show
  the same near-flat external resolution), which weakens an architecture-only
  explanation and strengthens the diversity/coverage hypothesis.
- The external high-activity regime is mishandled precisely where the target
  distribution concentrates (20.3% > 0.8 vs 2.5% internal).

**Recommended next experiment (high level only; NOT implemented in 9D):**
obtain one or more additional labeled sgRNA activity datasets compatible with
the existing 30-mer encoding, and evaluate whether *training-domain
diversification* (multi-dataset training under the SAME frozen-evaluation
policy) changes the external outcome. Such a study simultaneously (a) tests
H-Augmentation directly and (b) provides a fair basis for any later
architecture/representation comparison. Architecture and representation
experiments (candidate A) should only follow once either a diversified corpus
is available or a positive representation signal emerges — they cannot be
concluded from the current negative results.

## 16. Limitations

- External sample size (n=810) gives wide confidence intervals on the already
  weak metrics; observed external R² point estimates are within ~±0.06 of the
  null for most models (bootstrap ranges from 9C) — the "weak ranking signal"
  claim should be read as low-confidence.
- Activity bins on the common [0,1] grid produce uneven n (internal high bin
  n=38); per-bin numbers are descriptive, not inferential.
- Within-bin true SD ≈ 0.06 makes any bin-level variance/R² interpretation
  meaningless by construction; none is reported.
- Domain-shift statistics use large-n tests (huge n → tiny p); effect sizes
  (Cohen's d, ratio, slopes) are the primary evidence, p-values secondary.
- 9D does not audit beyond the stored artifacts and files inspected; it cannot
  exclude latent environmental/software drift not reflected in stored metrics.

## 17. Conclusion

The strongest defensible conclusion after Phases 3–9C is:

> The canonical CRISPR activity regression models are reproducible, perform
> moderately on in-domain data, and degrade sharply on the locked Moreno-Mateos
> test. The degradation is robust across three architecturally distinct model
> families and is associated with a substantial distribution shift (labels,
> GC, sequence content, zero sequence overlap) and with strong prediction
> under-dispersion. Simple post-hoc remedies — prediction-scale calibration and
> domain-weighted retraining — are refuted as solutions. The models retain
> weak, nonzero ranking signal externally but do **not** achieve useful
> absolute prediction. The cause of the gap is not established: representation/
> capacity limitation is plausible but untested, and calibration/weighting
> failure must not be read as evidence for an architectural explanation.

The recommended, conservative next step is **dataset diversification**
(decision B): expanding the training corpus across sgRNA activity domains,
under the unchanged frozen-evaluation policy, to test the one association that
has been demonstrated (domain shift) before any further architecture-driven
experiments are proposed.

---

**Outputs of Phase 9D**
- `docs/phase9d_final_generalization_audit_report.md` (this report)
- `results/experiments/phase9d_final_generalization_audit_20260905_235023.json`
- `results/figures/phase9d_{pred_vs_true,bias_mae_vs_activity_bin,prediction_sd_vs_true_sd,ranking_scatter}_*.png`
- `scripts/run_phase9d_audit.py`, `src/audit/audit.py`, `src/audit/__init__.py`, `tests/test_phase9d_audit.py`

No canonical artifact, dataset, split, Phase 3–8 result, or Phase 9A–9C result
was modified. **Nothing is committed; Phase 10 must not start without explicit
approval.**