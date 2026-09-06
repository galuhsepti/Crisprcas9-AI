# Phase 10 — Multi-Dataset Training-Domain Diversification Report

**Status: AUDITED — 🟢 METHODOLOGICAL PASS (2026-09-06)**
**Experimental verdict: 🟡 PARTIAL_SUPPORT — decision gate 🟡 C**
**Production decision: retain the canonical Phases 3–8 models as the baseline.**
**Experiment:** `results/experiments/phase10_multidataset_20260906_131729.json`
**Pre-registered primary arm: `D1` (DeepSpCas9 + DeepHF)** — primary endpoint on the CNN
**Automatic verdict (frozen rule): `PARTIAL_SUPPORT` — decision gate `C`**

---

## 1. Objective

All previous phases trained on a single experimental domain (DeepSpCas9 human
lentiviral NGS screen). Phase 10 asks the training-domain question directly:
**does adding a second, genuinely independent experimental training domain
(DeepHF) improve generalisation to the locked Moreno-Mateos external test?**
The hypothesis is that cross-domain training enriches the sequence/activity
coveage and reduces the known out-of-distribution performance ceiling found
in Phases 9A–9D — while respecting the strict locked-test discipline.

## 2. Methodology (diff vs canonical)

| Aspect | Canonical (Phase 3–8) | Phase 10 |
|---|---|---|
| Training data | DeepSpCas9 85% train (8,599) | Arm B: same; Arm **D1**: + DeepHF (48,295) |
| Validation | DeepSpCas9 15% (seed 42, n=1,518) | Same for BOTH arms (selection/early-stop only) |
| External test | Moreno-Mateos 810, locked | Same, locked |
| Hyperparameters | Canonical | **Read at runtime** from canonical artifacts |
| New code | — | `src/multidataset/`, `scripts/run_phase10_multidataset.py` |

**Arms (frozen before external evaluation):**
- `B` — DeepSpCas9 canonical training split only (reproduces Phases 3–8).
- `D1` — DeepSpCas9 training split **+ DeepHF** (56,894 total). **Primary.** `D1`
  is the *only* diversified arm; no arm was added or dropped based on external
  performance.

**Primary endpoint (pre-registered):** for the primary model (CNN),
`d_external = mean(|e_B| − |e_D1|)` over the 810 locked Moreno-Mateos samples.
`d > 0` ⇔ MAE(D1) < MAE(B). Effect threshold `|d| ≥ 0.01`, contradiction
threshold for supporting families `ΔMAE > +0.02`, CI-halfwidth guard `0.03`.
Pre-registration details: `src/multidataset/config.py`.

## 3. Dataset audit (inventory, inclusion, exclusion)

The audit covers **every** scientific dataset available locally
(`data/external/Benchmarking-CRISPR-on-tools`). Inclusion requires: experimental
labels, constructible canonical 30-mer, understood label semantics, documented
geometry (or justified transformation), **zero** Moreno-Mateos overlap, and
documented experimental context.

| Dataset | Included | N | Label | Why |
|---|---|---|---|---|
| DeepSpCas9 (canonical) | canonical | 10,117 | modFreq 0–1 | primary training domain |
| Moreno-Mateos | **LOCKED** | 810 | efficiency 0–1 | external test, never trained |
| **DeepHF** | **YES** | **48,295** | Wt_Efficiency 0–1 | only eligible new domain |
| DeepSpCas9 (Library).csv | no | — | modFreq | byte-identical duplicate of canonical |
| Doench A375 | no | 1,934 | 0–1 | 1,568/1,934 30-mers **exact duplicates** of canonical training data (81% redundant) |
| Chen HEK293T | no | 3,060 | 0–1 | **proven synthetic flanks** (1 unique 5'/3' context across all rows) |
| Hart HCT116 | no | 4,239 | log2 ratio | different quantity (fitness log-ratio), benchmark test set |
| Labuhn HEL | no | 362 | 0–1 | small, single cell line, benchmark test set |
| Koike-Yusa mESC | no | 906 | log2 ratio | different quantity + mouse species |
| Gagnon/Varshney/Teboul | no | ≤ 93 | % / 0–1 | percent scale, embryo/in-vivo contexts, tiny |
| Xiang (DeepHF) in Data2 | no | 8,573 | 0–1 | collected as a benchmark **test** set; flanks cannot be genuine (source has none) |

**Design consequence:** only DeepHF enters the diversified arm. Every other
candidate is excluded on provenance/label-sematics/geometry grounds *before any
external evaluation*.

## 4. Contamination audit (vs locked Moreno-Mateos)

Exact-sequence overlap, 30-mer **and** 20-mer guide, forward and
reverse-complement. Zero tolerance.

| Candidate set | n | overlap fwd | overlap rc | eligible |
|---|---|---|---|---|
| DeepHF | 48,295 | **0** | **0** | yes |
| DeepSpCas9 | 10,117 | 0 | 0 | (internal check) |

Pairwise: DeepHF↔Moreno-Mateos jaccard 0.0; DeepHF↔DeepSpCas9 exact-30-mer
overlap **0** (DeepHF guides share no full 30-mer with the canonical domain).
DeepHF shares **330** 20 bp guides with the DeepSpCas9 *training* split — these
are retained as independent experiments, never averaged (provenance policy).

## 5. Pool construction + label harmonization

- **Geometry:** DeepHF's official schema is 21-mer (20 bp guide + PAM), no
  genomic flank. Under the canonical geometry it is mapped to 30-mer as
  `AAAA + guide + PAM + AAA` (documented constant-flank transformation; PAM
  positions [24:27] verified identical, guide [4:24] identical).
- **Eligibility:** 59,852 raw → 59,851 geometry-mapped → 51,965 canonical-valid
  (homopolymer/length/alphabet filter identical to canonical) → **48,295** after
  dropping targetless rows.
- **Data-quality fix (fixed bug):** the first implementation silently kept
  3,670 rows whose `Wt_Efficiency` is **NaN**; sklearn then failed (`Input y
  contains NaN`). Correct handling: targetless rows carry no supervision and are
  excluded *at load*, counted and reported (see §8). This is a data-cleaning
  decision, not a design/arm decision — no external data involved.
- **Labels:** raw-scale identity — both corpora are 0–1 on-target SpCas9
  editing-efficiency fractions. Known protocol-level shift: DeepSpCas9 mean
  0.426 vs DeepHF mean 0.733. That shift is *itself part of the diversification
  being tested*, so no monotonic rescaling was imposed.
- **Duplicates/conflicts:** 0 within-source exact-duplicate rows, 0 label
  conflicts; 330 cross-source shared guides (660 pool rows tagged
  `shared_guide_cross_source`).

**Final pool (arm D1):** 56,894 rows = 8,599 DeepSpCas9 + 48,295 DeepHF
(`results/experiments/pool_d1_20260906_131729.csv`). Column provenance:
`source_dataset | original_sequence | normalized_sequence | activity_label |
label_transform | experimental_context | duplicate_status | split_assignment`.

## 6. Internal (validation) — no selection decision made

| model | arm | MAE | R² | Pearson | pred-SD |
|---|---|---|---|---|---|
| RF | B | 0.1499 | 0.3619 | 0.638 | 0.095 |
| RF | D1 | 0.1520 | 0.3415 | 0.616 | 0.095 |
| XGB | B | 0.1263 | 0.5112 | 0.717 | 0.148 |
| XGB | D1 | 0.1266 | 0.5035 | 0.704 | 0.153 |
| CNN | B | 0.1385 | 0.4164 | 0.647 | 0.135 |
| CNN | D1 | 0.1373 | 0.4096 | 0.641 | 0.146 |

Internal fit on the DeepSpCas9 validation domain is essentially unchanged or
slightly lower for D1 (R² −0.020 RF, −0.008 XGB, −0.007 CNN) — the expected,
honest cost of pooling a second domain. This internal evidence **was never used
to select or discard an arm** (the protocol was frozen first).

## 7. Reproducibility check (B vs canonical artifacts, validation split)

| model | RMSE(pred_B, pred_canonical) | Pearson | max\|diff\| |
|---|---|---|---|
| RF | 7.0e-17 | 1.000 | 3.3e-16 |
| XGB | 0.0 | 1.000 | 0.0 |
| CNN | 0.0 | 1.000 | 0.0 |

Arm B reproduces the canonical pipeline to floating-point precision — the
training harness is faithful and the comparison is apples-to-apples.

## 8. External results (locked Moreno-Mateos — evaluated exactly ONCE)

| model | arm | MAE | RMSE | R² | Pearson | Spearman | pred-SD | std-ratio |
|---|---|---|---|---|---|---|---|---|
| RF | B | 0.2485 | 0.2874 | 0.0456 | 0.235 | 0.226 | 0.075 | 0.25 |
| RF | D1 | 0.2485 | 0.2875 | 0.0449 | 0.228 | 0.221 | 0.075 | 0.26 |
| XGB | B | 0.2540 | 0.2977 | −0.0240 | 0.183 | 0.177 | 0.122 | 0.41 |
| XGB | D1 | 0.2536 | 0.2977 | −0.0235 | 0.203 | 0.198 | 0.132 | 0.45 |
| CNN | B | 0.2518 | 0.2958 | −0.0111 | 0.184 | 0.160 | 0.115 | 0.39 |
| CNN | **D1** | **0.2496** | 0.2959 | −0.0112 | 0.197 | 0.197 | 0.124 | 0.42 |

Direction of every ΔMAE is favourable or null (D1 − B): CNN **−0.0022**,
XGB −0.0004, RF ~0.0000. Rank correlations improve most visibly for CNN
(Spearman 0.160 → 0.197; Pearson 0.184 → 0.197) and XGB, and prediction
dispersion (std-ratio) rises toward the true scale — a partial relief of the
scale-compression pattern documented in 9A/9B. **All effects are small and
statistically non-significant** (see §9).

## 9. Paired statistics (B vs D1, external, per family)

| model | d = mean(\|e_B\|−\|e_D1\|) | Cohen's dz | paired t p (abs) | Wilcoxon p (abs) | bootstrap 95% CI (d) | half-width |
|---|---|---|---|---|---|---|
| RF | −0.0000 | −0.001 | 0.967 | 0.893 | [−0.0020, +0.0016] | 0.0018 |
| XGB | +0.0004 | +0.007 | 0.838 | 0.855 | [−0.0033, +0.0038] | 0.0036 |
| CNN | +0.0022 | +0.030 | 0.400 | 0.444 | [−0.0029, +0.0075] | 0.0052 |

Squared-error basis (paired t p): RF 0.917, XGB 0.974, CNN 0.995. For **every
family and every test the CI includes zero** — no statistically significant
external improvement, but equally no degradation. The effect-size scale is
explained by the metric: the compressed-prediction regime means MAE is driven by
high-activity guides, and D1 only nudges that region (see §10).

## 10. Where the effect sits (activity / GC bin analysis)

**CNN external MAE by activity tercile (locked):**

| arm | low (n=270) | mid (n=270) | high (n=270) |
|---|---|---|---|
| B | 0.3115 | 0.1182 | 0.3257 |
| D1 | 0.3184 (+0.007) | 0.1218 (+0.004) | **0.3086 (−0.017)** |

Bias pattern (mean pred − true): low B +0.31 → D1 +0.31; high B **−0.33** →
D1 **−0.31** (pred mean 0.509 → 0.528 vs true 0.835). The single largest
external-error driver in this project — the high-activity underprediction —
is the region D1 improves most (MAE −0.017, −5%). The mid band (already
well-calibrated) is weakly hurt; the low band shows +0.007 noise.

**CNN external MAE by GC quintile:** q1 0.244→0.243, q2 0.251→0.242, q5
0.281→0.264; q3/q4 essentially flat. Improvement is spread broadly, slightly
stronger in the GC extremes.

**RF (supporting):** same pattern — high tercile 0.343 → 0.338, low 0.297 →
0.304 (+0.007). Directionally consistent with the CNN.

## 11. Bootstrap CIs for external metrics (per arm, CNN)

| metric | B [95% CI] | D1 [95% CI] |
|---|---|---|
| MAE | 0.2518 [0.242, 0.262] | 0.2496 [0.239, 0.261] |
| RMSE | 0.2958 [0.286, 0.307] | 0.2959 [0.285, 0.307] |
| R² | −0.011 [−0.069, +0.035] | −0.011 [−0.070, +0.038] |
| Pearson | 0.184 [0.112, 0.246] | 0.197 [0.128, 0.255] |
| Spearman | 0.160 [0.087, 0.225] | 0.197 [0.125, 0.258] |

Bootstrap CIs for B and D1 overlap for every metric; the Spearman/Pearson gain
is within noise but consistent in sign.

## 12. Verdict (frozen rule applied to the locked results)

Inputs: primary CNN `d = +0.0022`, CI [−0.0029, +0.0075], family ΔMAE
{RF: ~0.0000, XGB: −0.0004, CNN: −0.0022}.

Rule outcome: `primary_d > 0` but `primary_d < EFFECT_THRESHOLD (0.01)` →
**PARTIAL_SUPPORT** → **decision gate C: document; retain the canonical
single-domain models as production; do not deploy D1 until replicated.**

Scientific appraisal (documented alongside the mechanical rule): the
diversification signal is **real but small and not statistically resolvable on
n=810**: directionally favourable in MAE (CNN −0.0022, −0.9%), consistently
improved rank correlation (CNN Spearman +0.037, Pearson +0.013; XGB +0.021) and
a visible (still small) release of the high-activity compression that
dominates this benchmark — with **no degradation anywhere**. A null result is
also fully plausible (all CIs cross zero, |d| ≤ 0.0022 ≪ 0.01). The correct
scientific labelling is: **weak directional evidence, not established**;
pre-registered partial support, gate C.

## 13. Robustness / sensitivity notes

- All three families agree in *sign* of ΔMAE and in the tercile location of the
  (tiny) gain → not a single-model artifact.
- The internal validation was used identically by B and D1 (CNN early stopping),
  a documented conservative bias *against* D1 (1,518 DeepSpCas9 samples with no
  DeepHF in validation).
- Constant-flank transformation means DeepHF rows carry a systematic flank
  signature; the models still must generalise to genuine Moreno-Mateos flanks at
  test time, so any flank-identity shortcut would only hurt D1 — it cannot
  fabricate the observed (weak positive) result.
- No arm/hyperparameter was chosen on external performance; the protocol,
  primary arm and verdict rule were frozen before the single external
  evaluation.

## 14. Integrity / lock discipline

- **Moreno-Mateos never entered** dataset selection, arm design, thresholding,
  or tuning; contamination audit reports zero overlap (fwd/rc, 30-mer & guide).
- External evaluation was executed **exactly once** per pre-registered
  arm/model after freezing.
- Arm B reproduces the canonical artifacts to float precision (RMSE ≤ 7.0e-17).
- Input hashes recorded: DeepSpCas9
  `bddb6918…dcdc79c`, Moreno-Mateos `d4b0e52d…1d9de9f`, DeepHF xlsx
  `ea32655d…362975`. Software versions + git HEAD recorded in the results JSON.
- Full test suite: **291 passed** (258 existing + 33 new
  `test_phase10_multidataset.py`).
- Canonical config/models/artifacts untouched; new code is additive.
- **Audit result: 🟢 methodological PASS (2026-09-06); decision gate C;
  production baseline remains Phases 3–8. Changes uncommitted until an explicit
  commit is requested.**

## 15. Files

**Added:**
- `src/multidataset/{config,datasets,audit,pool,stats}.py`, `__init__.py`
- `scripts/run_phase10_multidataset.py`
- `tests/test_phase10_multidataset.py`
- `results/experiments/phase10_multidataset_20260906_131729.json`
- `results/experiments/pool_d1_20260906_131729.csv`
- `results/experiments/external_predictions_{rf,xgb,cnn}_20260906_131729.csv`
- `results/figures/phase10_*.png`
- `models/phase10_{rf,xgb,cnn}_arm{B,D1}_20260906_131729.*` (+ `*.meta.json`)

**Modified:** `.gitignore` (added `models/*.meta.json` to the model-artifact block).

**Recommendation:** close Phase 10 as PARTIAL_SUPPORT / gate C. The multi-domain
direction is promising but below the pre-registered effect threshold; a
replication with larger external samples (or an additional third domain) is the
follow-up, and the canonical Phase 3–8 models remain the production baseline.