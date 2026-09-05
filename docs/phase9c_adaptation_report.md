# Phase 9C — Domain-Weighted Retraining (Adaptation) Audit Report

**Status: FOR AUDIT — NOT COMMITTED**
**Date:** 2026-09-05
**Experiment:** `results/experiments/phase9c_adaptation_20260905_231823.json`
**Pre-registered primary arm: `LG` (label × GC combined reweighting)**

---

## 1. Objective

Phase 9B refuted the prediction-scale/calibration hypothesis: post-hoc
monotonic transforms learned on internal data did **not** improve the locked
Moreno-Mateos test. Phase 9C keeps the canonical *training data* boundary
strict but changes the *training recipe*: DeepSpCas9 samples are
importance-reweighted toward the covariate/label regime that dominates
Moreno-Mateos (high activity, high GC) — never touching external data during
weight fitting, arm selection, or tuning.

## 2. Methodology (diff vs canonical)

| Aspect | Canonical (Phase 3–8) | Phase 9C |
|---|---|---|
| Training data | DeepSpCas9 85% train | **Same**, unchanged |
| Validation | DeepSpCas9 15% (seed 42) | Same (selection only) |
| External test | Moreno-Mateos 810, locked | **Same, locked** |
| Recipe | Uniform per-sample loss | Sample-weighted by arm |
| New code | — | `src/adaptation/`, optional `sample_weight` in CNN/RF/XGB |
| Hyperparameters | Canonical | **Read at runtime** from canonical artifacts (identical) |

**Arms** (all computed on DeepSpCas9 train only):
- `B` uniform (1.0) — canonical-reproduction arm
- `L` inverse-density on activity label
- `G` inverse-density on full-sequence GC
- `LG` = product of L×G (clipped, renormalised) — **pre-registered primary**
- Recipe: `w_i ∝ 1/p̂(z_i)`, 20 equal-width bins, clip [0.1, 10.0], mean-1 scale.

**Reproducibility guarantee:** the `B` arm reproduces the canonical models
almost exactly on validation (RF RMSE diff `5.6e-17`; XGB and CNN diff `0.0`),
confirming the retraining pipeline itself is faithful.

**Weight effectiveness** (ESS ratio before clipping applied on mean-1 scale):
- `L`: ESS/n = 0.68, label density range 0.136 → 0.040 (flattened 3.4×)
- `G`: ESS/n = 0.23, GC density range 0.341 → 0.120 (flattened 2.8×)
- `LG`: ESS/n = 0.27 (combined)

> **Implementation note (fixed bug):** an initial version clipped on the raw
> `1/p̂` scale (~13–833), which wiped all variation out (all weights→10→1.0).
> The clip is now applied on the **relative** (mean-1) scale so all arms carry
> meaningful weight. A regression test guards against the all-1.0 collapse.

## 3. Internal (validation) results — selection step

| model | arm | MAE | R² | Pearson | pred-SD |
|---|---|---|---|---|---|
| RF | B | 0.1499 | 0.3619 | 0.6375 | 0.095 |
| RF | **LG** | 0.1483 | **0.3664** | 0.6228 | 0.107 |
| XGB | B | 0.1263 | 0.5112 | 0.7173 | 0.148 |
| XGB | **LG** | 0.1297 | 0.4816 | 0.6955 | 0.165 |
| CNN | B | 0.1385 | 0.4164 | 0.6468 | 0.135 |
| CNN | **LG** | 0.1456 | 0.3359 | 0.5999 | 0.168 |

Internal OOF-in-train sanity (5-fold): RF LG ≈ B (R² +0.0012); XGB LG < B
(R² −0.0235). **Reweighting did not help internal fit** — consistent with it
deliberately sacrificing in-domain accuracy for external-domain emphasis.

## 4. External (locked Moreno-Mateos) — evaluated ONCE

| model | arm | MAE | R² | Pearson | Spearman | pred-SD |
|---|---|---|---|---|---|---|
| RF | B | 0.2485 | **0.0456** | 0.235 | 0.226 | 0.075 |
| RF | **LG** | 0.2489 | 0.0361 | 0.214 | 0.210 | 0.083 |
| XGB | B | 0.2540 | −0.0240 | 0.183 | 0.177 | 0.122 |
| XGB | **LG** | 0.2560 | −0.0712 | 0.166 | 0.165 | 0.141 |
| CNN | B | 0.2518 | **−0.0111** | 0.184 | 0.160 | 0.115 |
| CNN | **LG** | 0.2547 | −0.0866 | 0.186 | 0.179 | 0.156 |

**Primary-arm verdict: the domain-weighting hypothesis is REFUTED.** The
pre-registered `LG` arm does not beat the canonical baseline `B` on any
model. R² degrades for all three (most clearly for CNN: −0.011 → −0.087 and
XGB: −0.024 → −0.071). Paired tests on absolute error show no significant
difference (all p ≥ 0.16), with slightly worse R² and higher pred-SD for LG.

**Sensitivity arms (`L`, `G`) confirm the direction uniformly:**

| model | best external R² (arm) | vs B |
|---|---|---|
| RF | 0.0522 (L) | +0.007 (marginal, only gain) |
| XGB | −0.0132 (G) | +0.011 (marginal) |
| CNN | −0.0111 (B) | baseline best |

No sensitivity arm materially improves on `B`; the only wins are tiny
(+0.007/+0.011 R², non-significant, not reproduced across models).

**Tercile bias (low/high):** LG raises low-activity bias and reduces
high-activity bias slightly, but both biases remain large (+0.30/−0.33).
**GC-quintile analysis (RF):** LG marginally helps the highest-GC quintile
(MAE 0.2819 → 0.2800) and q4 (0.2558 → 0.2542) but is ~flat to worse
elsewhere — no consistent GC-regime benefit.

## 5. Why it failed (analysis)

1. Reweighting the *training loss* cannot fix what is primarily a
   **representation / resolution** failure: canonical predictors barely
   spread predictions (external pred-SD 0.075–0.122 vs true 0.294). The
   underlying function still cannot resolve high-activity from low-activity
   samples; reweighting just reallocates a fixed, insufficient resolution.
2. More aggressively reweighting toward the high-activity/high-GC tail
   **inflated prediction variance** (pred-SD rose 0.075→0.083, 0.122→0.141,
   0.115→0.156) without a corresponding accuracy gain — i.e. it traded
   bias for added, unhelpful noise in the target regime.
3. The gain was largest on the covariate most weakly predictive (GC),
   and internal metrics (XGB/CNN) actually *dropped* under LG — the weights
   pushed fits away from the bulk of DeepSpCas9 data with no downstream
   payoff on the external tail.

## 6. Conclusion & recommendation

**Domain-weighted retraining (Phase 9C) does not improve the locked
Moreno-Mateos test.** Reweighting toward the target regime is dominated by
the canonical baseline; the external gap is not a *weighting*/covariate-mass
artifact but a genuine out-of-distribution prediction-quality ceiling for
these supervised architectures.

Recommended next steps (ranked):
1. **Prioritise representation/capacity** — the strongest lever for the
   low-resolution external issue is architectural: sequence-embedding /
   pretrained-encoder CNNs (CNN-based sgRNA embeddings), or ensembling the
   three families, rather than reweighting a fixed-capacity model.
2. **Report 9C as a documented negative** alongside 9A/9B; retain canonical
   Phase 3–8 models as the deployed baseline.
3. If retraining is revisited, prefer **covariate-shift via resampling +
   architecture change** (not loss weighting) and pre-register external
   evaluation exactly as done here.

## 7. Integrity / lock discipline

- Weights computed from **train split only**; Moreno-Mateos never used for
  fitting/selection/tuning.
- All arms applied to external **exactly once**; `LG` pre-registered as
  primary, no iteration.
- `B` arm reproduces canonical artifacts to float precision
  (RMSE ≤ 5.6e-17) — pipeline faithful.
- Full test suite: **248 passed** (229 canonical + 19 new weighting tests).
- Canonical models/artifacts untouched; new code is additive/backward-compatible.
- **Nothing committed; waiting for PASS before any commit.**

**Files added:**
- `src/adaptation/weighting.py`, `src/adaptation/__init__.py`
- `scripts/run_phase9c_adaptation.py`
- `tests/test_domain_weighting.py`
- `results/experiments/phase9c_adaptation_20260905_231823.json`
- `results/figures/phase9c_*.png`

**Files modified (backward-compatible `sample_weight` support only):**
- `src/models/cnn.py`, `src/models/random_forest.py`, `src/models/xgboost_model.py`
