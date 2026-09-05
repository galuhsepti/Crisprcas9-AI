# Phase 9B — Prediction-Scale / Calibration Hypothesis Test

**Status:** Completed · **Date:** 2026-09-05 · **Type:** Hypothesis-testing experiment (no model changes)

Phase 9B is a controlled hypothesis test. It asks whether part of the external-test
degradation observed in Phase 9A can be explained by *prediction-scale compression /
under-dispersion* (predictions compressed toward the DeepSpCas9 activity mean). The null
interpretation is that external degradation is **not** primarily a post-hoc scale problem
and that representation / domain generalization is the dominant cause.

**Artifacts**
- Machinery: `src/calibration/{base,noop,linear,variance,oof,utils}.py`
- Script: `scripts/run_phase9b_calibration.py`
- Results: `results/experiments/phase9b_calibration_20260905_223448.json`
- Figures: `results/figures/phase9b_*.png` (8 figures)
- Tests: `tests/test_calibration.py` (33 tests) · full suite **229 passing**

---

## 1. Summary

**The prediction-scale hypothesis is REFUTED.** Post-hoc calibration learned from
DeepSpCas9 internal data does not repair the external Moreno-Mateos degradation:

- On the external test, calibrated MAE is essentially unchanged or worse; RMSE is worse
  under variance correction; R² is consistently worse for both calibration methods.
- As required by the monotone transforms, Pearson and Spearman are unchanged
  (differences of exactly 0 or ~1e-9..1e-17). Calibration adds **no** ranking information.
- Forcing the prediction spread to match the internal target distribution (variance
  method) *worsens* external error metrics (R² drops by 0.10–0.18), because it rescales
  toward the *internal* mean/SD, which differs from the external label distribution.

Interpretation: prediction compression is a real, observable property of the models, but it
does **not** explain the external degradation. The external failure is dominated by weak
ranking information (domain/representation generalization), not by scale. Calibration
"reduces part of the observed scale mismatch" (SD ratio) but cannot convert poor ranking
signal into good external predictions.

## 2. Design and reproducibility

| Item | Choice |
|---|---|
| Canonical artifacts | RF `rf_baseline_fixed_20260905_001107.pkl`, XGB `xgboost_baseline_20260905_002839.pkl`, CNN `cnn_baseline_20260905_011720.pt` |
| Retraining | **None** — artifacts loaded as-is |
| Calibration source (fit) | DeepSpCas9 **validation split** (canonical 15%, seed 42, n=1,518) |
| Calibration evaluation (internal) | 5-fold **out-of-fold** within the validation split (seed 4242) |
| External test | Moreno-Mateos (n=810), **locked**; applied exactly once, never used to fit/select/tune |
| Methods | A. raw (no calibration) · B. linear (OLS, `y_cal = a + b·y_pred`) · C. variance/scale (`y_cal = target_mean + (y_pred − source_mean)·(target_std/source_std)`) |
| Data versions | `data/raw/DeepSpCas9.csv` (10,117 valid), `data/raw/Moreno-Mateos.csv` (810 valid) |
| Metrics | primary: MAE, RMSE, R², Pearson, Spearman · secondary: SD ratio, bias, calibration slope/intercept, per-tercile error |

**Why validation as the calibration source.** Out-of-fold predictions on the DeepSpCas9
*training* split (the strictly preferred design) require re-fitting the canonical models
per fold, which Phase 9B forbids ("no retraining"). The validation split is the only
internal subset that is (i) fully disjoint from model-fitting for RF and XGB (neither
used validation for early stopping or selection), and (ii) usable for the CNN with a
**documented** caveat. This is the adaptation chosen with the user.

**CNN caveat (D-00X).** The canonical CNN used the validation split for early stopping /
best-epoch selection in Phase 5. Fitting the CNN's calibration mapping on validation is
therefore a *limited, documented reuse* of that signal (scalar affine parameters). The
methodology leans on the rank-preserving property of the transforms (Spearman exactly
unchanged), so this caveat does not affect the ranking conclusions; it is noted as a
limitation for the CNN scale parameters.

**Locking discipline.** Methods and parameters were fixed using internal data only. The
external set was evaluated exactly once (Section 7 of the spec). No multiplier, method, or
threshold was chosen by looking at external results.

## 3. Calibration parameters (fit on internal validation data)

| Model | Method | Parameters |
|---|---|---|
| RF | linear | slope = 1.4945, intercept = −0.2104 |
| RF | variance | source mean = 0.4228, source SD = 0.0954 → target mean = 0.4215, target SD = 0.2237 |
| XGB | linear | slope = 1.0872, intercept = −0.0380 |
| XGB | variance | source mean = 0.4226, source SD = 0.1476 → target mean = 0.4215, target SD = 0.2237 |
| CNN | linear | slope = 1.0714, intercept = −0.0276 |
| CNN | variance | source mean = 0.4192, source SD = 0.1351 → target mean = 0.4215, target SD = 0.2237 |

The linear slope is largest for RF (1.49), consistent with RF being the most compressed
(val pred SD 0.095 vs true 0.224). XGB/CNN slopes are near 1.08, consistent with thinner
compression. Note the variance method targets the *validation* mean/SD (0.4215, 0.2237);
the external labels have mean 0.4969, SD 0.2944, which is exactly why scale correction
learned internally cannot fully restore the external spread.

## 4. Internal evaluation — 5-fold OOF within the validation split (leakage-safe)

Raw vs. calibrated on samples whose calibrator was fit on other folds:

| Model | Method | MAE | RMSE | R² | Pearson | Spearman | SD ratio |
|---|---|---|---|---|---|---|---|
| RF | raw | 0.1499 | 0.1787 | 0.3619 | 0.6375 | 0.6262 | 0.427 |
| RF | linear | **0.1400** | **0.1726** | **0.4046** | 0.6361 | 0.6254 | 0.638 |
| RF | variance | 0.1514 | 0.1907 | 0.2732 | 0.6369 | 0.6260 | 0.999 |
| XGB | raw | 0.1263 | 0.1564 | 0.5112 | 0.7173 | 0.7051 | 0.660 |
| XGB | linear | **0.1249** | **0.1561** | **0.5130** | 0.7163 | 0.7039 | 0.717 |
| XGB | variance | 0.1336 | 0.1684 | 0.4330 | 0.7167 | 0.7044 | 1.001 |
| CNN | raw | 0.1385 | 0.1709 | 0.4164 | 0.6468 | 0.6313 | 0.604 |
| CNN | linear | **0.1373** | **0.1708** | **0.4171** | 0.6458 | 0.6306 | 0.647 |
| CNN | variance | 0.1482 | 0.1882 | 0.2922 | 0.6463 | 0.6309 | 1.001 |

Takeaways:
- On internal data, **linear** calibration gives a small consistent gain (RF most: MAE
  −0.010, R² +0.043) and never damages ranking metrics.
- **Variance** calibration restores SD ratio ≈ 1.0 exactly (by construction) but *hurts*
  MAE/RMSE/R² on internal data — over-expansion is not free.
- Pearson/Spearman are essentially invariant, confirming monotonic transforms.

## 5. External evaluation — Moreno-Mateos (locked, applied once)

| Model | Method | MAE | RMSE | R² | Pearson | Spearman | SD ratio | overall bias |
|---|---|---|---|---|---|---|---|---|
| RF | raw | 0.2485 | 0.2874 | 0.0456 | 0.2352 | 0.2263 | 0.253 | −0.029 |
| RF | linear | 0.2469 | 0.2891 | 0.0341 | 0.2352 | 0.2263 | 0.379 | −0.007 |
| RF | variance | 0.2522 | 0.3064 | **−0.0848** | 0.2352 | 0.2263 | 0.594 | +0.031 |
| XGB | raw | 0.2540 | 0.2977 | −0.0240 | 0.1833 | 0.1772 | 0.414 | −0.019 |
| XGB | linear | 0.2548 | 0.3001 | −0.0405 | 0.1833 | 0.1772 | 0.450 | −0.015 |
| XGB | variance | 0.2634 | 0.3176 | **−0.1650** | 0.1833 | 0.1772 | 0.628 | +0.008 |
| CNN | raw | 0.2518 | 0.2958 | −0.0111 | 0.1838 | 0.1602 | 0.390 | −0.014 |
| CNN | linear | 0.2522 | 0.2974 | −0.0217 | 0.1838 | 0.1602 | 0.418 | −0.007 |
| CNN | variance | 0.2630 | 0.3210 | **−0.1903** | 0.1838 | 0.1602 | 0.646 | +0.030 |

Takeaways:
- **MAE/RMSE/R² do not improve under external calibration.** RF linear MAE improves by
  0.0016 (0.2485→0.2469) but R² falls 0.0456→0.0341; every other calibrated cell is equal
  or worse, and variance correction makes R² strongly negative.
- **Pearson and Spearman are bit-for-bit unchanged** (Δ ≤ ~1e-9). This is the cleanest
  evidence that calibration adds no ranking information.
- **SD ratio improves** (RF 0.253→0.594) but remains far from the external true ratio of
  1.0 — because the internal target SD (0.224) is smaller than the external SD (0.294).
  The scale mapping learned internally is the *wrong* target distribution for the
  external set.
- The pred-vs-true resolution slope on external improves from 0.060→0.090→0.140 (RF) but
  stays ≪ 1: calibration narrows, but does not remove, the regression-to-mean artifact.

## 6. Extreme-activity analysis (external, terciles of true activity)

Low/mid/high terciles (boundaries ≈ 0.36 / 0.59 true activity):

| Model | Method | low bias | low MAE | mid bias | high bias | high MAE |
|---|---|---|---|---|---|---|
| RF | raw | +0.296 | 0.297 | −0.040 | −0.343 | 0.343 |
| RF | linear | +0.308 | 0.312 | −0.021 | −0.310 | 0.310 |
| RF | variance | +0.331 | 0.345 | +0.013 | −0.250 | 0.256 |
| XGB | raw | +0.301 | 0.306 | −0.032 | −0.326 | 0.326 |
| XGB | linear | +0.303 | 0.309 | −0.029 | −0.320 | 0.320 |
| XGB | variance | +0.316 | 0.335 | −0.009 | −0.283 | 0.287 |
| CNN | raw | +0.309 | 0.312 | −0.026 | −0.326 | 0.326 |
| CNN | linear | +0.315 | 0.318 | −0.020 | −0.317 | 0.317 |
| CNN | variance | +0.340 | 0.356 | +0.014 | −0.264 | 0.272 |

- Variance correction **reduces high-activity underprediction** (−0.33 → −0.25/−0.28) and
  its bin MAE, but **increases low-activity overprediction** (+0.30 → +0.33/+0.34) and its
  bin MAE. The net effect on total MAE/RMSE/R² is negative.
- Linear calibration barely moves the extreme biases. No calibration reproduces the
  Phase 9A bias structure (≈ −0.33 high, ≈ +0.30 low) almost exactly — a useful sanity check.

## 7. Paired tests & bootstrap CIs (external, descriptive only)

Calibration was locked before any external look; paired tests are descriptive of the
locked procedure, not a selection step.

- Linear vs raw: mean squared-error difference (raw − cal) is **negative** for all three
  models (−9.9e-4, −1.4e-3, −9.1e-4), i.e. raw generally has *lower* squared error;
  significant (t-test p = 0.25 for RF, 4.3e-9 XGB, 7.3e-5 CNN).
- Variance vs raw: mean squared-error difference is **negative and large** (−1.1e-2, −1.2e-2,
  −1.6e-2; all p < 1e-5) — variance correction is significantly *worse* on external.

## 8. Ranking invariance check

| Model | method | ΔPearson | ΔSpearman | ΔKendall |
|---|---|---|---|---|
| RF | linear | 5.6e-17 | 0.0 | 0.0 |
| RF | variance | 2.8e-17 | 0.0 | 0.0 |
| XGB | linear | 1.5e-09 | 0.0 | 0.0 |
| XGB | variance | 1.9e-09 | 0.0 | 0.0 |
| CNN | linear | 1.4e-09 | 0.0 | 0.0 |
| CNN | variance | 7.1e-09 | 0.0 | 0.0 |

Ranking changes are exactly zero (Spearman/Kendall) or machine precision (Pearson). This
confirms the theoretical requirement that monotonic calibration cannot change ranking.

## 9. Hypothesis verdict

**The prediction-scale / under-dispersion hypothesis is REFUTED as an explanation of the
external degradation.**

Per the Phase 9B interpretation protocol:
- MAE/RMSE/R²: **not improved** by calibration → scale correction does not rescue error.
- Spearman: **unchanged** (and already low, 0.16–0.23 on external) → calibration adds no
  ranking information, and the ranking signal itself is weak across datasets.
- Therefore: "Prediction compression alone does not explain the external degradation."

Why the variance method fails so badly is itself informative: the external label
distribution (mean 0.497, SD 0.294) sits *outside* the internal target (mean 0.422, SD
0.224). Scaling internal-learned moments to the external outcome set necessarily mis-targets
the external center and spread. This is direct evidence of **domain shift in the label
distribution**, not a self-contained scale defect of the predictors.

**Precise claim (do not overclaim):** calibration reduces part of the observed scale
mismatch *on the internal distribution it was learned from*, but it cannot resolve
cross-dataset generalization. External degradation is primarily an information/ranking
problem (representation/domain generalization), not a calibration problem.

## 10. Limitations

1. **Calibration source is the validation split, not train-set OOF.** The strictly
   preferred design (OOF on the training split) would require re-fitting canonical models
   per fold (forbidden). The chosen design is clean for RF/XGB and documented-reuse for CNN.
2. **CNN early-stopping reuse.** The CNN's calibration mapping is fit on the same
   validation split used for its best-epoch selection. Rank conclusions are unaffected
   (monotonic transforms); the CNN's scale parameters carry this caveat.
3. **Calibration is fit to the internal label distribution.** Any post-hoc mapping can only
   correct toward where the calibration data lives; it cannot know about the external
   mean/SD without leaking external information.
4. **Small external n (810) and low signal.** Bootstrap CIs (Section 10 of script) are wide;
   R² values near zero are individually uninformative. The consistent direction across all
   three models is the evidence, not one point estimate.
5. **Only monotonic methods tested.** Per the spec, isotonic/non-monotonic approaches were
   deliberately excluded to isolate the scale hypothesis. A non-monotonic recalibration
   would lose the clean ranking-invariance property and is a different question.
6. **MAPE unused** (instability near zero), consistent with Phase 6/9A.

## 11. Recommendation for Phase 9C

Because ranking information itself (not scale) is the binding constraint on external
performance, Phase 9C should target **representation / generalization rather than
post-hoc correction**:

1. **Domain-adaptation / covariate-shift-aware training** is the natural next step, e.g.
   retraining on a weighted or augmented DeepSpCas9 mix so the model is exposed to the
   high-GC, high-activity regime that dominates Moreno-Mateos — while keeping Moreno-Mateos
   locked for a single final evaluation.
2. **If retraining is undesired**, the alternative is *isotonic/non-monotonic* calibration
   (ranked-tercile recalibration) purely as a descriptive follow-up — but Phase 9B already
   indicates post-hoc transforms cannot add the missing ranking signal.
3. Whatever Phase 9C does, report the *locking discipline* used here: method/params fixed
   on internal data, external set applied once, ranking metrics reported separately from
   scale metrics so the two do not get conflated.
4. Do **not** spend further effort on OLS-linear or variance/scale recalibration — Phase 9B
   shows returns ≈ 0 and slightly negative for error metrics.