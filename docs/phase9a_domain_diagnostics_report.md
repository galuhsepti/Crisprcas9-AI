# Phase 9A — Model Diagnostics & Domain-Shift Analysis

**Status:** Completed · **Date:** 2026-09-05 · **Type:** Diagnostic only (no model changes)

This phase is strictly diagnostic. It characterizes the external Moreno-Mateos test set
relative to the DeepSpCas9 internal validation set and — using the canonical,
untrained models — analyzes prediction errors on the external set. No model was
retrained, no hyperparameter was changed, no data was modified, and no model-selection
decision was made on the basis of the external set.

**Artifacts**
- Machinery: `src/diagnostics/{stats,sequence_analysis,model_diagnostics}.py`
- Script: `scripts/run_phase9a_diagnostics.py` (reproduces the canonical split, seed 42)
- Results: `results/experiments/phase9a_domain_diagnostics_20260905_161859.json`
- Figures: `results/figures/phase9a_*.png` (8 figures)
- Tests: `tests/test_diagnostics.py` (27 tests) · suite total 196 passing

---

## 1. Summary

The canonical models degrade on the external Moreno-Mateos test set (R² ≈ 0; Spearman
0.16–0.23) while holding respectable internal-validation performance (Pearson 0.64–0.72,
MAE 0.126–0.150). The diagnostics attribute this primarily to **covariate (sequence)
shift** — the external set is composed of different, higher-GC, higher-activity guides
with no overlap in sequences — combined with **label-distribution shift** (a large mass
of high-activity targets the models were not calibrated on) and a structural property of
the predictors themselves: **predictions are strongly under-dispersed (regression toward
the activity mean)**, which by itself produces sign-opposite bias at the activity extremes
(+0.30 overprediction on low-activity guides, −0.33 underprediction on high-activity
guides). This is a compatibility/precondition-analysis diagnostic, not a verdict: it
identifies where the shift lies and does not itself justify switching to the external
dataset as the evaluation target.

## 2. Dataset and label-distribution comparison

Internal reference = canonical validation split of DeepSpCas9 (n = 1,518); external =
Moreno-Mateos (n = 810). Both subsets are internally duplicate-free (0 duplicates) and
well-formed (30-mer, no ambiguous characters in valid rows).

| Quantity | DeepSpCas9 validation | Moreno-Mateos test | Shift evidence |
|---|---|---|---|
| n | 1,518 | 810 | — |
| Activity mean / median | 0.4215 / 0.4352 | 0.4969 / 0.5079 | +0.075 mean |
| Activity SD | 0.2237 | 0.2944 | +32% spread |
| Activity range | 0.0004–0.934 | 0.0000–1.000 | wider tails |
| Q25 / Q75 | 0.233 / 0.607 | 0.233 / 0.755 | upper tail shifted |
| Fraction ≤ 0.05 | 4.3% | 5.8% | mild |
| Fraction > 0.8 | 2.5% | **20.3%** | ~8× more high-activity |
| KS on activity (D, p) | — | — | 0.193, **p = 1.2e-17** |
| Cohen's d (val − test) | — | — | −0.30 (small) |

The test distribution is shifted upward and strongly stretched at the top: one in five
external guides exceeds 0.8 activity, versus 1 in 40 internal.

### Sequence composition
- **GC content**: val mean 0.550 vs test mean 0.584 (Cohen's d = −0.33; KS D = 0.199,
  p = 9.6e-19). The external set is more GC-rich.
- **Nucleotide frequencies**: consistent with GC — test is G/C-enriched (G: 0.249→0.268,
  C: 0.261→0.284) at the expense of A/T.
- **3-mer profiles**: correlated (Pearson r = 0.835, p = 1.1e-17) but with systematic
  differences; the largest shifts are G-rich motifs that are roughly 2× more frequent in
  the external set: TGG (+0.024), GGA (+0.023), GGG (+0.023), GAG (+0.019), AGG (+0.018),
  GGT (+0.014), GTG (+0.013). A/T-rich motifs (CTT, TTC, GCC) are depleted. This is the
  k-mer signature of high-GC target selection (e.g., GC-rich SpCas9 target genes).
- **Positional composition**: per-position frequencies differ modestly; mean per-position
  Shannon entropy is ≈ 0.93 (validation) and ≈ 0.86 (test). Neither set is
  positionally degenerate, but the test set is marginally more compositionally biased,
  consistent with its higher GC content.
- **Exact-sequence overlap**: **zero** at both levels — the 1,518 validation sequences vs
  the 810 test sequences share nothing, and, more conclusively, **all 10,117 DeepSpCas9
  sequences vs all 810 Moreno-Mateos sequences share nothing** (unique-in-A: 10,117,
  unique-in-B: 810, Jaccard = 0). The external set is a genuinely disjoint sequence
  population, not a subset or near-duplicate.

## 3. Prediction profiles and bias (canonical models)

| Model | Set | Pred mean | Pred SD | True SD | Bias (pred−true) | MAE | RMSE | Pearson |
|---|---|---|---|---|---|---|---|---|
| RF | val | 0.4228 | 0.095 | 0.224 | +0.0013 | 0.150 | 0.179 | 0.638 |
| RF | test | 0.4684 | 0.075 | 0.294 | **−0.0286** | 0.2485 | 0.287 | 0.235 |
| XGB | val | 0.4226 | 0.148 | 0.224 | +0.0011 | 0.126 | 0.156 | 0.717 |
| XGB | test | 0.4779 | 0.122 | 0.294 | **−0.0191** | 0.2540 | 0.298 | 0.183 |
| CNN | val | 0.4192 | 0.135 | 0.224 | −0.0023 | 0.139 | 0.171 | 0.647 |
| CNN | test | 0.4827 | 0.115 | 0.294 | **−0.0142** | 0.2518 | 0.296 | 0.184 |

**Key observation — under-dispersion.** All three models regress predictions toward the
training-set activity mean: on validation their predicted SD (0.095–0.148) is already
only ~45–65% of the true SD, and on the test set the predicted SD (0.075–0.122) is only
~25–42% of the true SD (0.294). Predicted ranges on the test set are 0.14–0.66 (RF),
0.05–0.80 (XGB), 0.15–0.82 (CNN) against a true range of 0–1. Overall bias stays near
zero because over- and under-prediction cancel; the bias is entirely explained by
conditioning on the true activity level (below).

## 4. Error analysis (external test set)

### By true-activity quintile (equal-width bins)

| True-activity bin | n | MAE | Bias (pred−true) |
|---|---|---|---|
| ~0.00–0.13 | 177 | 0.35–0.37 | **+0.35 / +0.36 / +0.37** |
| ~0.13–0.30 | 148 | 0.16–0.18 | +0.15 |
| ~0.30–0.60 | 156 | 0.08–0.11 | −0.01 … −0.03 |
| ~0.60–0.80 | 165 | 0.21–0.23 | −0.21 … −0.23 |
| ~0.80–1.00 | 164 | 0.38–0.40 | **−0.38 / −0.40** |

(RF / XGB / CNN respectively; bias values listed in that order where they differ.)

### By true-activity tercile (summary)

| Model | Overall bias | Low tercile | Mid tercile | High tercile |
|---|---|---|---|---|
| RF | −0.029 | **+0.296** | −0.040 | **−0.343** |
| XGB | −0.019 | **+0.301** | −0.032 | **−0.326** |
| CNN | −0.014 | **+0.309** | −0.026 | **−0.326** |

MAE is minimized in the middle of the activity scale and roughly doubles at both extremes
(0.08–0.11 middle vs 0.35–0.40 extremes). The error structure is the canonical signature
of a model mapping inputs to a compressed, training-conditional activity level: it
overpredicts truly low-activity guides and underpredicts truly high-activity guides. The
effect is exaggerated on the test set simply because the test set contains far more
extremes (especially 20% above 0.8), i.e., the discrepancy is concentrated exactly where
the internal distribution was thinnest.

### By sequence GC and by predicted activity
- **GC bins**: MAE increases mildly with GC (0.125 at low GC → 0.318 at GC > 0.71 for RF,
  n = 37 in the extreme bin), with a slight overprediction at low GC and underprediction
  at high GC. This is secondary to the activity-effect and consistent with the
  activity–GC association.
- **Predicted-activity bins**: mean true and mean predicted track each other within bins
  (e.g., RF bin means: (true, pred) = (0.18, 0.19), (0.37, 0.32), (0.44, 0.41), (0.52, 0.50),
  (0.62, 0.58)) — again the compressed-range pattern; even at the top predicted bin the mean
  prediction (0.58) stays far below the true high-activity mean.

## 5. Model agreement on the external set

| Pair | Pearson | Spearman | Mean abs pred difference |
|---|---|---|---|
| RF vs XGB | 0.858 | 0.844 | 0.055 |
| RF vs CNN | 0.674 | 0.652 | 0.068 |
| XGB vs CNN | 0.672 | 0.652 | 0.076 |

The tree models agree strongly with each other and each agrees only moderately with the
CNN. Note that agreement does not imply correctness: all three models share the same
under-dispersion and the same few clusters of systematic error, and all three fail to
recover the top-activity fraction of the external set (Spearman 0.16–0.23, R² ≈ 0).

## 6. Relative contribution of the candidate causes (assessment)

Based only on the diagnostics above, the observed degradation is attributable to a
**combination** of several factors, ordered by their likely weight:

1. **Under-dispersion of predictions by construction.** The dominant structural error
   driver. The models' outputs are compressed toward the (low-mean, narrow) training
   activity distribution. This alone explains the sign-opposite extreme-bin bias and is
   not a domain-shift artifact — it exists on the internal validation set too (pred SD
   ≈ half of true SD) but only becomes gross on the wider external scale. This is a
   property of the trained predictors + training-label distribution, in the same family
   as the already-documented shrinkage toward the guide-only signal.
2. **Label/activity distribution shift.** The external set is 8× richer in high-activity
   targets (>0.8) than the internal distribution; the models were never calibrated in
   that regime and thus fall farthest there. The noise in that regime is also the largest
   part of RMSE/MAE.
3. **Covariate (sequence) shift with zero overlap.** Distinct (entirely non-overlapping,
   GC-enriched, G-motif-enriched) composition. Sequence shift feeds the label shift —
   GC-rich targeting selects for G-rich, often high-activity guides — so effects 1–3
   compound rather than stack independently.
4. **Representation signal reaches the right genes but the wrong scale.** Combined with
   Phases 7–8 (guide-region carries the signal; guide one-hot is the dominant feature
   group) the diagnostics are consistent with a model that has learned *where* the
   activity lives (guide imprint) but not the *scale* of activity in other target
   populations. Nothing here suggests the features or the architectures extract the wrong
   information qualitatively; the failure is in distributional calibration across
   populations.
5. **Label-definition/label-noise difference between datasets** cannot be excluded but is
   not evidenced by these diagnostics (both labels are 0–1 continuous activities measured
   by the same readout modality; no label artifact is apparent). It remains an untested
   hypothesis requiring a controlled protocol.

## 7. Limitations

- Diagnostic-only; **no causal claim** is made for any factor. "Contributing weight" above
  is an ordering of evidence strength, not an effect-size estimate, and the factors are
  confounded with each other (sequence shift ⟷ label shift).
- The external set was used for characterization; no canonical model was selected or tuned
  against it.
- Some statistical tests (KS on activity, KS on GC, k-mer profile correlations) are
  reported on correlated observations; they are directional evidence, and multiple-testing
  is not adjusted because the tests are descriptive rather than confirmatory. Exact p-values
  appear in the results JSON for transparency.
- The by-GC high-bin (n = 37) and the internal-vs-external comparisons cover different
  sample sizes (1,518 vs 810); paired statistics were not used across datasets.
- Bootstrap uncertainty of the error-by-bin and dispersion statistics was not propagated;
  point estimates only.

## 8. Prerequisite for Phase 9B

Per the research plan, Phase 9B must start from a decision, jointly with the supervisor,
about which single experiment is most rational given these findings. The diagnostics
constrain the space: they do **not** point to an architecture "bug" or a feature gap, and
do **not** justify switching the evaluation target to Moreno-Mateos as-is; they point
toward a test of whether better-calibrated prediction scales (e.g., a per-dataset scale
matching) can be demonstrated, or a controlled check of the across-population
generalization hypothesis, without altering the canonical pipeline or the canonical
evaluation target.