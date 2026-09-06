# Phase 11 — Targeted Dataset Analysis (Post Phase 10 Diversification)

**Status:** completed · **Type:** analysis-only (no model training, no external evaluation)
**Result JSON:** `results/experiments/phase11_targeted_dataset_analysis_20260906_141922.json`
**Composition CSV:** `results/experiments/phase11_pool_composition_20260906_141922.csv`
**Figures:** `results/figures/phase11_*.png` (6)
**Environment:** system `python3 3.14.4` · `git HEAD d5cf9fd67b3002ce905b0e357beff30c4cd78f50` (committed Phase 9D state; Phase 10 artifacts uncommitted)

---

## 1. Overview & Objective

Phase 10 added a second experimental training domain (DeepHF, 48,295 labelled
rows) to the canonical DeepSpCas9 training split (8,599) and measured
generalisability on the *locked* Moreno-Mateos external test. The Phase 10
verdict was **PARTIAL_SUPPORT** (primary CNN `d = +0.0022`, CI
`[−0.0029, +0.0075]` crosses zero; below the effect threshold
`EFFECT_THRESHOLD = 0.01`).

Phase 11 does **not** run any new experiment. It explains why the gain was only
partial and whether any mechanism can be attributed at all, using only:

1. the **training domain** (DeepSpCas9 train + DeepHF) as it was actually used
   by arm D1,
2. the Phase 10 **provenance artifacts** (result JSON + pool CSV), and
3. already-generated Phase 9A/Phase 10 **diagnostics**, quoted verbatim.

No model was trained, no hyperparameter was changed, and no cached/external
data was re-evaluated. Moreno-Mateos was **never opened** in this phase.

## 2. Constraints & Stop Conditions (all verified)

| Condition | Status |
|---|---|
| Domain provenance reconstructible and row-level identical to stored Phase 10 pool | ✅ |
| Pool counts match the locked Phase 10 JSON (8,599 / 1,518 / 56,894) | ✅ |
| No new model training / no hyperparameter change | ✅ |
| Moreno-Mateos not read or re-evaluated (quoted only from locked JSONs) | ✅ |
| Canonical artifacts untouched | ✅ |
| No commit, no push (per project policy) | ✅ |

Had any of these failed the script would have halted before producing output.

## 3. Methodology

- DeepSpCas9 canonical split reproduced (seed 42, 85/15) → **8,599 / 1,518**.
- The diversified pool was **rebuilt** from the same inputs and compared
  **row-for-row** against `pool_d1_20260906_131729.csv` → **exact match**.
- DeepHF provenance reconstructed: 59,852 raw → 51,965 geometry-mapped+valid →
  **48,295** labelled (3,670 NaN `Wt_Efficiency` rows dropped **before** any
  statistics; no imputation).
- All activity binning uses the canonical fixed grid
  `DEFAULT_ACTIVITY_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.0]`
  (`src/audit/audit.py`); **never** re-fit or adapted to anything.
- All "external" numbers below originate in the locked Phase 9A / Phase 10
  result JSONs and are quoted verbatim.

## 4. Data & Provenance Reconstruction

| Quantity | Reconstructed | Phase 10 JSON | Match |
|---|---|---|---|
| DeepSpCas9 valid | 10,117 | — | — |
| Training split | 8,599 | 8,599 | ✅ |
| Validation split | 1,518 | 1,518 | ✅ |
| DeepHF mapped+valid | 51,965 | — | — |
| DeepHF NaN-label dropped | 3,670 | — | — |
| DeepHF retained (incl. tr% of pool) | 48,295 (84.9%) | 48,295 | ✅ |
| Pool total (arm D1) | 56,894 | 56,894 | ✅ |
| Pool row-level equality vs stored CSV | exact | — | ✅ |

`DeepHF label stats`: mean 0.7325, SD 0.2127, median 0.8057, min 0.0296,
max 1.0. `DeepSpCas9 train label stats`: mean 0.4264, SD 0.2237.

## 5. Pool Contribution (Analysis A)

| Source | n | fraction of pool | label mean | label SD | fraction > 0.8 |
|---|---|---|---|---|---|
| DeepSpCas9 (train) | 8,599 | 15.1% | 0.426 | 0.224 | 3.3% |
| DeepHF | 48,295 | 84.9% | 0.733 | 0.213 | 51.2% |

The added domain is the **increasing-mass** component (≈85% of the pool) and is
skewed toward **high activity**. `results/experiments/phase11_pool_composition_*.csv`.

## 6. Activity-Range Coverage (Analysis B)

Fraction of rows per canonical bin (`activity_distribution_by_source`,
`activity_ecdf`, `coverage_by_bin` figures):

| bin | DeepSpCas9 | DeepHF | combined |
|---|---|---|---|
| [0.00, 0.20) | 0.190 | 0.026 | 0.050 |
| [0.20, 0.40) | 0.261 | 0.076 | 0.103 |
| [0.40, 0.60) | 0.297 | 0.131 | 0.156 |
| [0.60, 0.80) | 0.219 | 0.255 | 0.249 |
| [0.80, 1.00] | 0.033 | **0.512** | 0.439 |

The only bin reaching a **material** coverage expansion (≥ 2 percentage-point
delta **and** ≥ 25% relative gain, both thresholds descriptive and pre-set) is
the **high-activity tail [0.80, 1.00]** (0.033 → 0.512). This is the same
region Phase 9A identified as under-represented *and* the region where the
Phase 10 external improvement was concentrated (see §13).

## 7. Sequence Diversity (Analysis C)

| metric | DeepSpCas9 | DeepHF | difference |
|---|---|---|---|
| GC mean (30-mer) | 0.551 | 0.427 | Cohen's d **+1.53** |
| k-mer 2 JSD | — | — | 0.063 |
| k-mer 3 JSD | — | — | 0.080 |
| nucleotide mean abs Δ | — | — | 0.100 |
| guide-level overlap (20-mer) | 8,599 | 48,295 | 330 shared (Jaccard 0.006) |

- Exact 30-mer overlap between the training domains is **0** (DeepHF source is
  flankless and mapped via constant `AAAA`/`AAA`).
- Only 330 of the 56,894 pool rows are a **cross-source same-guide** guide; the
  compositional distributions are clearly distinct (large GC and k-mer gap).
- **Caveat:** the top enriched 3-mers in DeepHF (`AAA`, `GAA`, `AAG`)
  substantially reflect the constant flanking `AAAA`/`AAA` of the geometry
  transformation — a transformation artifact, not biological novelty. GC is the
  cleaner signal, and it points **down** (0.427) relative to DeepSpCas9 (0.551).

## 8. Domain Diversity vs Sample Size (Analysis D)

Phase 10 changed **three things simultaneously** by adding DeepHF: sample count
(5.6×), activity coverage (high tail), and sequence composition. There is **no
size-matched arm and no single-factor arm** in the Phase 10 design, so these
dimensions are **fully confounded** at the design level. The evidence matrix
in the result JSON therefore reports each dimension descriptively and does
**not** fabricate a single "diversity score". This confound is itself a primary
reason a mechanistic attribution is not identifiable.

## 9. High-Activity Coverage (Analysis E)

| threshold | base fraction | added fraction | combined |
|---|---|---|---|
| > 0.6 | 0.252 | 0.767 | 0.687 |
| > 0.7 | 0.129 | 0.625 | 0.550 |
| > 0.8 | 0.033 | 0.512 | 0.439 |
| > 0.9 | 0.005 | 0.234 | 0.199 |

The added domain massively increases the training **density** of high-activity
guides. This is *consistent with* (not proof of) the external high-activity
MAE improvement seen in Phase 10 (CNN high-tercile MAE 0.326 → 0.309).

## 10. Phase 10 Arm Attribution (Analysis F, descriptive)

Internal validation (a conservative proxy; no conclusions drawn):

| model | arm B R² | arm D1 R² |
|---|---|---|
| RandomForest | 0.362 | 0.342 |
| XGBoost | 0.511 | 0.504 |
| CNN | 0.416 | 0.410 |

Locked external results (quoted verbatim from Phase 10 JSON `20260906_131729`):

| model | MAE B | MAE D1 | ΔMAE |
|---|---|---|---|
| RandomForest | 0.2485 | 0.2485 | ≈ 0.0000 |
| XGBoost | 0.2540 | 0.2536 | −0.0004 |
| CNN | 0.2518 | 0.2496 | **−0.0022** |

Primary CNN paired statistic: `d = +0.0022`, 95% bootstrap CI
`[−0.0029, +0.0075]`, `dz = 0.030`, paired t p = 0.400, Wilcoxon p = 0.444.
Spearman 0.160 → 0.197; pred-SD std-ratio 0.39 → 0.42. **These values were not
recomputed; they are only re-quoted.**

## 11. Label Quality Audit

- DeepHF targetless rows: **3,670** dropped (of 51,965 mapped/valid) at load,
  before any training/statistics. **No imputation anywhere.**
- Removed rows are *not in* the retained pool or any result (0 NaN labels in
  the 48,295 retained rows; 0 NaN in the 56,894-row pool).
- DeepSpCas9 labels untouched and NaN-free (10,117 validated rows).
- This is a documented data-quality exclusion, not a design goal (consistent
  with Phase 10's methodology record).

## 12. Duplicate & Conflict Policy Verification

| policy check | result |
|---|---|
| within-source duplicate rows in pool | **0** |
| cross-source exact 30-mer rows in pool | **0** |
| cross-source shared guides (20-mer) | **330 pairs / 660 rows**, kept as independent experiments |
| cross-source label conflicts | none (absent 30-mers) |

Verified: exactly what the pre-registered Phase 10 duplicate policy specified —
within-source exact duplicates excluded at load, cross-source same-guide
co-occurrences retained with provenance and **no averaging**.

## 13. Phase 9A → Phase 10 Consistency

Phase 9A (`phase9a_domain_diagnostics_20260905_161859.json`) characterised the
validation→external shift as: higher mean activity, a large high-activity mass
(> 0.8: 2.5% → 20.2%), higher GC (0.55 → 0.58), 3-mer enrichment of
TGG/GGA/GGG/GAG, and **zero** sequence overlap.

| shift (quoted from 9A) | base (validation) | DeepHF | did DeepHF close it? |
|---|---|---|---|
| high-activity tail > 0.8 | 2.5% | 51.2% | ✅ potentially addressed (overshoot) |
| label distribution mean | 0.4215 | 0.7325 | ⚠️ correct direction, overshoots test mean 0.497 |
| GC content mean | 0.5496 | 0.4274 | ❌ **not addressed (moves away)** |
| nucleotide G fraction | 0.2956 | 0.2540 | ❌ not addressed (moves away) |
| k-mer profile similarity | (9A: test-vs-val corr 0.835) | pool-vs-base JSD 0.080 | ⚠️ changes, direction not classifiable from stored scalars |

**Key interpretation:** the added domain closed the **label/activity** gap
(especially the high tail) — the dimension that plausibly *did* help the
external high-activity tercile — but it did **not** close the **sequence/GC**
gap and 9A's compositional signal moved the *other* way. The external gain is
therefore *consistent with* added high-activity coverage but is confounded with
simultaneous changes in composition that a size-matched or composition-matched
arm would be required to separate.

## 14. Hypothesis Assessment (H1–H6)

| hypothesis | verdict | basis (training-domain evidence) |
|---|---|---|
| H1 sample size is the dominant lever | **PARTIALLY SUPPORTED** | 5.6× n, 85% mass, but composition changed a lot too |
| H2 activity-range coverage materially increased | **SUPPORTED** | only the [0.8, 1.0] bin is materially expanded |
| H3 sequence diversity materially increased | **SUPPORTED** | GC d = 1.53, k-mer JSD ≈ 0.06–0.08, zero 30-mer overlap |
| H4 experimentally distinct domains | **SUPPORTED** | 2 datasets / 2 assay contexts, distinct label scales un-hooked |
| H5 high-activity gain consistent with coverage | **PARTIALLY SUPPORTED** | coverage rose 0.033→0.512, but the external effect is below threshold & CI crosses zero |
| H6 evidence suffices to attribute a mechanism | **NOT IDENTIFIABLE** | effect too small/uncertain; dimensions confounded at design level |

## 15. Decision Gate (A–D)

| gate | meaning | outcome |
|---|---|---|
| A | dataset diversity is the stronger lever | — |
| B | activity coverage is the stronger lever | — |
| C | heterogeneity the main uncertainty (effect meaningful, levers confounded) | — |
| **D** | **evidence too weak to attribute — freeze dataset experiments** | ✅ **selected** |

**Rationale (evidence-based, not "because it's small"):** the primary paired
bootstrap CI `[−0.0029, +0.0075]` **includes zero**, so the Phase 10 effect is
not reliably distinguishable from no effect; the point estimate is also below
the pre-registered threshold. Because there is no reliable effect to attribute,
declaring a "stronger lever" (A/B/C) would be unsupported. D is the appropriate
decision: **no further dataset/domain experiments are warranted on the current
evidence; the canonical Phases 3–8 production model is retained.**

## 16. Caveats, Limitations & Conclusion

**Limitations**
- Phase 10's design confounds n / coverage / composition; no single-factor or
  size-matched arm exists to separate them.
- The DeepHF k-mer composition signal is partly a flank-transformation
  artifact (`AAAA/AAA`); GC is the cleaner molecular signal and moved **away**
  from the 9A external regime.
- 9A stored only scalar k-mer summaries, so the pool→external k-mer
  *direction* cannot be classified post-hoc without re-reading external data
  (prohibited); the pool-side numbers are reported instead.
- Internal validation R² dropped slightly for every family under D1 — the
  added domain did not help DeepSpCas9-distribution guides (expected, given
  the domain shift), reinforcing that any D1 gain is external-specific.

**Conclusion**

The Phase 10 gain was partial and *cannot* currently be attributed to a
mechanism: the effect is small and its CI crosses zero, and the single added
dataset simultaneously changed sample size, activity coverage and sequence
composition. Training-domain evidence is **consistent with** (H2/H5) the
high-activity tail being the coverage dimension that changed, and with (H3)
substantial sequence diversity, but gate **D** requires freezing dataset
experiments rather than extending them. If attribution is ever needed, the
protocol would need a *composition-matched / size-matched* control arm — a new
pre-registered phase, not a reanalysis of the current evidence.

**Next steps (only if authorised as a new phase):** no dataset
additions/removals for the production recipe; canonical Phases 3–8 model
remains production; any future diversification must add orthogonal controls
that the Phase 11 evidence matrix identified as missing.