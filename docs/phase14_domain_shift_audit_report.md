# Phase 14 — Domain-Shift Attribution / Robustness Audit Report

**Experiment:** `phase14_domain_shift_audit_20260907_115601`
**Status:** PASS (state machine reached `COMPLETE`)
**Decision gate:** **B/E**
**Date:** 2026-09-07
**Canonical artifacts:** `rf_baseline_fixed_20260905_001107.pkl`, `xgboost_baseline_20260905_002839.pkl`, `cnn_baseline_20260905_011720.pt` (SHA-256 verified identical before and after; see [Integrity](#integrity-and-leakage-checks))

---

## 1. Purpose and protocol

This phase is a strictly **read-only, associational** audit of why frozen canonical
models, trained on the internal DeepSpCas9 corpus, degrade on the external
Moreno–Mateos corpus. Following the scope and interpretation decisions D-1…D-10
and the C1/C2 statistical clarification (independent-domains bootstrap, not
paired), it characterises observable domain shift (sequence composition,
GC, activity distribution) and assesses which pre-registered shift dimensions
are most consistently **associated** with the external error increase.

No model was trained, tuned, calibrated, or adapted; no data were pooled; no
canonical artifact was modified or re-fitted. The locked external test file was
read **once, after FREEZE** (os.stat-only before FREEZE).

State machine log (recorded in the final JSON):

```
start:DESIGN -> FEASIBILITY -> ANALYSIS -> STATISTICAL_EVALUATION
-> FREEZE (frozen:True) -> FINAL_EXTERNAL_EVALUATION -> AUDIT -> COMPLETE
```

Frozen bin edges (fixed before any external access, from the internal TRAIN
distribution only):

- GC, full 30-mer: `[0.47, 0.53, 0.57, 0.63]` (internal-train quantile) → 5 bins
- GC, guide [4:24]: `[0.45, 0.50, 0.55, 0.65]` (internal-train quantile) → 5 bins
- Activity: fixed equal-width grid `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (bin 5 closed) → 5 bins

Datasets: internal n=10,117 valid (train 8,599 / validation 1,518); external
(Moreno–Mateos) n=810 valid after canonical validation — exactly the
pre-registered counts.

## 2. Observed domain shift (descriptive)

| Quantity | Internal train | External (Moreno–Mateos) |
|---|---|---|
| GC mean ± SD (30-mer) | 0.551 ± 0.115 | 0.584 ± 0.079 |
| Cohen's d (train − external) | — | **−0.296** |
| JSD k-mer, 2-mer | — | 0.107 |
|  internal train null band (2.5–97.5 pct) | — | (0.0035, 0.0085) |
| JSD k-mer, 3-mer | — | 0.140 |
|  internal train null band (2.5–97.5 pct) | — | (0.0093, 0.0147) |

The external corpus sits 0.30 SDs higher in GC than the internal train and
~0.11–0.14 bits of Jensen–Shannon divergence in k-mer composition — an order of
magnitude above the internal sampling null band. The external true-activity
distribution is narrower and shifted, and **external predictions are strongly
under-dispersed** relative to their targets (Table 2). These are real, observable
differences in the input/prior distributions; the audit is *not* testing their
causal role.

## 3. External accuracy collapse

| Model | Internal val MAE | External MAE | ΔMAE | Internal R² | External R² |
|---|---|---|---|---|---|
| Random forest | 0.150 | **0.248** | +0.099 | 0.362 | 0.046 |
| XGBoost | 0.126 | 0.254 | +0.128 | 0.511 | −0.024 |
| CNN | 0.138 | 0.252 | +0.113 | 0.416 | −0.011 |
| Phase 13 CNN (sensitivity) | 0.138 | 0.251 | +0.113 | 0.421 | −0.012 |

All three primary models degrade by ~0.10–0.13 MAE and effectively lose
predictive rank-order fit on the external corpus. RF validation MAE was
recomputed at AUDIT and is bit-identical (reproducible internal statistics = True).

**Table 2 — external dispersion (SD of pred / SD of true):**

| Model | SD ratio |
|---|---|
| Random forest | 0.253 |
| XGBoost | 0.414 |
| CNN | 0.390 |

All are below the pre-registered under-dispersion trigger 0.6 for at least one
primary model (in fact all three), satisfying the **E**-gate condition.

## 4. Confirmatory tests (C1–C7)

All tests use the pre-registered thresholds: FDR q=0.05, α=0.05,
|ρ| ≥ 0.15, |ΔMAE| ≥ 0.02, C6 cell ΔMAE ≥ 0.02, C7 mean pairwise ρ ≥ 0.6.
A test is *notable* only when the adjusted statistic AND the effect threshold
are both satisfied. C1/C2 are between-domain, independent-groups bootstraps
(external − internal); they are **not paired**.

| ID | Test | Point | 95% CI | p | BH-adj p | Notable? |
|---|---|---|---|---|---|---|
| C1 | ΔMAE, activity bin [0,0.2) | +0.127 | (0.109, 0.145) | 0.001 | 0.0018 | ✅ |
| C2 | ΔMAE, activity bin [0.8,1.0] | +0.088 | (0.066, 0.110) | 0.001 | 0.0018 | ✅ |
| C3 | ρ(abs err RF, GC) external | 0.087 | (0.018, 0.155) | — | 1.000 | ❌ |
| C4 | ρ(abs err RF, activity) external | 0.136 | (0.049, 0.223) | — | 1.000 | ❌ |
| C5 | ρ(abs err RF, GC_train_z) external | 0.087 | (0.018, 0.155) | — | 1.000 | ❌ |
| C6 | median ΔMAE, common-support GC×activity cells (14 cells, n≥20 both) | +0.042 | (0.017, 0.096) | 0.001 | 0.0018 | ✅ |
| C7 | cross-model per-bin MAE consistency, external (mean pairwise ρ) | 0.867 | (0.800, 1.000) | 0.001 | 0.0018 | ✅ |

Notes:
- C6 is restricted to the 14 common-support GC×activity cells with n ≥ 20 in
  **both** domains. It is explicitly **not** a statement about the full datasets.
- C3/C4/C5 CI bounds exclude the effect threshold 0.15, so these hypotheses
  (interaction with GC/composition, as rank associations) are **not supported**
  even though their CIs exclude 0. BH-adjusted p-values are 1.00 by design
  (the p-value rule was pre-registered as the CI rule for correlations).

## 5. Decision gates

Support counts per hypothesis:
- H1 (GC/composition association with error): **not supported** (C3/C5 ❌)
- H2 (activity-extreme subgroup ΔMAE): **supported** (C1/C2 ✅)
- H3 (joint common-support shift): **supported** (C6 ✅)
- H4 (cross-model error-pattern consistency): **supported** (C7 ✅)

`support_count = 3` → gate **B** (≥2 hypotheses supported, not all four).
`underdispersed_external = True` (all three primary models SD ratio < 0.6) and
H1∪H2∪H3 supported → gate **E** appended.

The **D** gate (joint-shift independently of single factors) was not triggered:
the joint-cell median ΔMAE (0.042) does not exceed the single-factor median
bin ΔMAE (GC 0.093, activity 0.048) by the margin of 0.02. i.e. the joint
stratification is consistent with the single-factor marginals; there is no
evidence of an *additional* joint-specific error.

**Final grade: `B/E` — multiple, mutually consistent observables are associated
with the external error increase; the pattern is strongly stable across models;
and external predictions are markedly under-dispersed.**

## 6. Interpretation (strictly associational — D-10)

The most defensible, strictly associational reading of the frozen evidence:

1. The **activity extremes** are the strongest single discriminator of the
   external error increase: subgroup MAE is 0.09–0.13 higher externally in the
   lowest and highest activity bins, with CIs comfortably above the effect
   threshold (C1, C2).
2. The **GC×activity joint stratification** is consistent with a broad, diffuse
   external error increase across common-support cells (median ΔMAE +0.042,
   C6), but it does **not** add beyond the marginal shifts (D not triggered).
3. The three primary models record **highly consistent** per-bin error patterns
   on the external corpus (mean pairwise ρ = 0.87, C7), so the external increase
   is a stable, model-independent phenomenon.
4. **External predictions are under-dispersed** (SD ratio 0.25–0.41), consistent
   with the observed squeezing of the external target distribution.
5. **GC/composition rank-associations with error are not supported** at the
   pre-registered effect size (C3–C5): the composition shift (GC +0.30 SD, JSD
   far above the internal null) is real, but the audit finds no evidence that
   these shifts correlate strongly with error magnitude.

These statements describe associations between frozen covariates and errors.
They do **not** assert causality, and they do not generalise to settings,
models, or shifting dimensions that were not pre-registered.

## 7. Limitations (pre-registered caveats)

- **Internal evaluation context (D-006):** the internal (validation) comparisons
  of C3–C7 were computed as descriptive context only; they do not upgrade the
  external, post-freeze confirmatory family.
- **C6 scope:** restricted to common-support cells; does not represent the full
  datasets. The empty cells of the joint grid and the imbalance of n across
  domains are recorded in the final JSON.
- **C1/C2 independence:** subgroups are compared across independent domains;
  this is not a paired comparison and does not control for between-corpus
  composition differences beyond the frozen covariates.
- **Bootstrap p-values** are percentile-based companions to the CIs (not BCa).
- **Phase 13 CNN** is a secondary sensitivity model only (D-5); it does not
  enter the confirmatory family or the gates.

## 8. Integrity and leakage checks

| Check | Value |
|---|---|
| External file content read only after FREEZE | True |
| External read performed exactly once | True |
| No training / tuning / calibration / adaptation | True |
| No pooled training data | True |
| Canonical hashes unchanged (before ≡ after) | True (RF, XGB, CNN) |
| Freeze config hash matches final config | True |
| RF validation MAE recomputation | bit-identical |
| Moreno–Mateos SHA-256 recorded at read time | `…` (full hash in final JSON) |

Full verification bundle: `results/experiments/phase14_domain_shift_audit_20260907_115601_final.json`
(contains `state_log`, `canonical_artifact_hashes`, `leakage_checks`, `audit`).

## 9. Artifacts

- Final report (this doc): `docs/phase14_domain_shift_audit_report.md`
- Internal snapshot: `results/experiments/phase14_domain_shift_audit_20260907_115601.json`
- Freeze record: `results/experiments/phase14_domain_shift_audit_20260907_115601_freeze.json`
- Final result bundle: `results/experiments/phase14_domain_shift_audit_20260907_115601_final.json`
- Figures (8): `results/figures/phase14_domain_shift_audit_20260907_115601_fig{1..8}_*.png`
- Predictions: `results/predictions/phase14_domain_shift_audit_20260907_115601_{internal,external}_predictions.csv`
- Source: `scripts/run_phase14_domain_shift_audit.py`, `src/domain_shift/*.py`,
  tests `tests/test_phase14_domain_shift.py` (409 tests pass, incl. integration schema tests)