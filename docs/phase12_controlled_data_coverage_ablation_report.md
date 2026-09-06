# Phase 12 — Controlled Data-Coverage Ablation (Feasibility-Gated)

**Status:** completed · **Outcome:** INFEASIBLE → **STOP** (gate **F**)
**Type:** analysis-only (no arm construction, no model training, no external evaluation)
**Result JSON:** `results/experiments/phase12_controlled_data_coverage_ablation_20260906_145652.json`
**Provenance CSVs:** `results/pools/phase12_source_provenance_{deepspcas9,deephf}_20260906_145652.csv`
**Figures:** `results/figures/phase12_*.png` (3)
**Environment:** system `python3 3.14.4` · `git HEAD d5cf9fd67b3002ce905b0e357beff30c4cd78f50`
**Tests:** `tests/test_phase12_coverage_ablation.py` (22) — full suite **345 passed**

---

## 1. Executive Summary

Phase 12 was pre-registered as a **controlled data-coverage ablation**: a
five-arm experiment (A canonical baseline, B sample-size control, C
activity-coverage control, D diversity control, E frozen Phase 10 reference)
whose primary contrasts (B/A, C/A, D/A, E/A) were meant to isolate *which*
data-coverage lever produced the partial Phase 10 gain.

The first and mandatory step of the protocol is the **Section-4 feasibility
audit**. That audit **failed deterministically**. With the available data no
subset construction can vary sample size, activity coverage, or sequence
diversity *independently* of one another, so the 5-arm design is **not
identifiable**. Per the protocol this is a valid scientific outcome, not a
defect:

> **"Phase 12 controlled isolation is not identifiable under the available
> data."**

**Decision gate: F** (CONTROLLED ABLATION INFEASIBLE WITH AVAILABLE DATA).
Triggered stop conditions: **1, 7, 8, 9**. No arm was constructed, no model was
trained, no external (Moreno-Mateos) evaluation was performed, and no Moreno-
Mateos data was read.

## 2. Objective & Frozen Scope

- **Objective:** determine whether the Phase 10 benefit (CNN `d = +0.0022`,
  CI `[−0.0029, +0.0075]`, above-combined-arms) is attributable to (H1) sample
  size, (H2) activity coverage, (H3) sequence/domain diversity, (H4) a
  combination, or (H5) is not identifiable to a single lever.
- **Frozen scope:** Moreno-Mateos is COMPLETELY LOCKED — never read, matched,
  inspected, or used for any selection. Phase 10/11 result JSONs are quoted
  only as historical records. No new external evaluation is performed
  (pre-registration of any confirmatory evaluation would be required first).
- **No new architecture** is introduced (Section-28 exception is not triggered
  because gate F already establishes non-identifiability).

## 3. Methodology (pre-registered, executed as written)

1. **Reproduce canonical domain:** DeepSpCas9, seed-42 85/15 split → train
   8,599 / validation 1,518.
2. **Load the only permitted additional source** DeepHF (48,295 labelled rows,
   3,670 NaN-label rows dropped before any statistics).
3. **Run the Section-4 feasibility audit** — quantitative, per-dimension,
   deterministic:
   - same-domain extra data check (ARM B structural requirement),
   - sequence/domain gap probes (GC, nucleotides, 2-mer/3-mer JSD,
     duplicates, length, overlap) between canonical and each extra source,
   - activity gap probes (P5–P95 quantiles, canonical bins, Wasserstein,
     Cohen's d),
   - within-extra activity ↔ composition entanglement probe,
4. **Apply stop conditions 1–11;** if triggered → STOP, assemble the result
   artifact set, do not build arms.
5. Arm construction (`construct_arm_pools`, `stratified_subsample`) exists as a
   tested code path but is **unreachable** with this inventory — the audit
   gates it, and gates it closed.

All random selection is seed-fixed (`numpy.random.default_rng(seed)`); no
subset search, no p-hacking, no model, no external data in any selection.

## 4. Feasibility Audit — Quantitative Results

### 4.1 Same-domain extra data (ARM B)

| Check | Result |
|---|---|
| Sources allowed as additional training | `['DeepHF']` (only) |
| Same-domain (= DeepSpCas9-freshscreen) extra rows available | **none** |
| Consequence | ARM B (sample size at constant activity AND constant
  canonical sequence/domain) is **structurally impossible** → STOP-1, STOP-9 |

### 4.2 Sequence / domain gap (canonical train vs DeepHF)

| Dimension | Canonical train | DeepHF | Gap |
|---|---|---|---|
| GC mean (sd) | 0.5512 (0.1151) | 0.4274 (0.0729) | **d = +1.533**, W1 = 0.124 |
| k-mer 3 JSD | — | — | **0.0796** |
| k-mer 2 JSD | — | — | 0.0632 |
| Nucleotide mean-abs-delta (freqs) | — | — | 0.1003 |
| Duplicate/unique structure | 8,599 unique / 0 dup | 48,295 unique / 0 dup | separate families |
| 20-mer guide overlap | — | — | 330 guides (Jaccard 0.0058) |
| Exact 30-mer overlap | 8,599 | 48,295 | shared 0, Jaccard 0.0 |

Any pool containing DeepHF changes BOTH sequence diversity and (due to the
layout below) activity coverage simultaneously → **STOP-8**.

### 4.3 Activity gap

| Quantile | Canonical train | DeepHF |
|---|---|---|
| P5 | 0.062 | 0.279 |
| P10 | 0.116 | 0.395 |
| P25 | 0.251 | 0.618 |
| **P50** | **0.437** | **0.806** |
| P75 | 0.601 | 0.898 |
| P90 | 0.714 | 0.937 |
| P95 | 0.772 | 0.953 |

Activity Wasserstein = **0.306**; Cohen's `d = −1.432`; high-activity
(> 0.8) coverage 0.0327 → 0.5116 (from locked Phase 11 artifact).

### 4.4 Activity ↔ composition entanglement *within* DeepHF

| Canonical bin | n | Mean GC (sd) |
|---|---|---|
| [0.0, 0.2] | 1,254 | 0.3740 (0.0849) |
| [0.2, 0.4] | 3,682 | 0.3917 (0.0763) |
| [0.4, 0.6] | 6,343 | 0.4035 (0.0718) |
| [0.6, 0.8] | 12,306 | 0.4193 (0.0700) |
| [0.8, 1.0] | 24,710 | 0.4456 (0.0674) |

GC rises **monotonically** with activity (range 0.072 ≥ 0.04 detection
threshold). Selecting DeepHF rows by activity therefore co-selects sequence
composition, and selecting by composition co-selects activity → **STOP-7,
STOP-8**. Activity coverage and sequence diversity cannot be separated even
*inside* a single source.

### 4.5 Decision

```
triggered_stop_conditions: [1, 7, 8, 9]
decision: INFEASIBLE
conclusion: Phase 12 controlled isolation is not identifiable under the
            available data.
```

## 5. Why the Design Was Not Forced

The Section-4 result is the intended, pre-registered exit. The protocol is
explicit that **non-identifiability is an acceptable scientific result** and
that the experiment must **not** be forced by relaxing tolerances, selecting
subsets by inspection, or finding "any" plausible control. Constructing arms
after this audit would yield comparisons that confound all three levers
simultaneously (C-vs-D is entangled within DeepHF; B requires data that does
not exist). Doing so would manufacture an answer instead of measuring one.

## 6. Validity & Guarantees

| Guarantee | Status |
|---|---|
| Moreno-Mateos raw data never read in Phase 12 | ✅ (asserted in runner; loader not imported) |
| No new external evaluation | ✅ |
| No model trained, no hyperparameter changed | ✅ |
| Canonical artifacts untouched (checked hashes/status) | ✅ |
| Arm construction not executed (audit gate closed) | ✅ |
| Deterministic reproduction (fixed seeds, no search) | ✅ |
| Full test suite (345) passes | ✅ |
| No commit, no push | ✅ |

## 7. Evidence Base (locked records, quoted — never recomputed)

| Artifact | Role |
|---|---|
| `results/experiments/phase10_multidataset_20260906_131729.json` | Phase 10 verdict PARTIAL_SUPPORT, CNN `d = +0.0022` CI `[−0.0029, +0.0075]` |
| `results/experiments/phase11_targeted_dataset_analysis_20260906_141922.json` | Phase 11 dataset analysis, gate **D** (evidence too weak to attribute the gain) |
| `data/raw/DeepSpCas9.csv` | sha256 `` ... (verified vs Phase 10 JSON) |
| `data/external/.../DeepHF_training.xlsx` | sha256 recorded in result JSON |

## 8. What the Data Does Support

- The additional source genuinely adds **new sequence diversity** (GC d =
  +1.533 vs the canonical domain; nucleotides/k-mers shifted; distinct screen
  context).
- The additional source genuinely adds **high-activity coverage** (P50 0.437 →
  0.806; >0.8 coverage 0.033 → 0.512).
- These two facts are **jointly true** — which is precisely the problem: they
  cannot be varied separately. "More data" in this repository means "data of a
  different distribution", and the two levers move together both across and
  within sources.

## 9. Causal Language

Consistent with the interpretation rule recorded in the result JSON, no causal
claim is made. The conclusion is a statement about **constructibility /
identifiability**, not effect attribution.

## 10. Decision Gate (Section 27)

```
gate:  F — CONTROLLED ABLATION INFEASIBLE WITH AVAILABLE DATA
consequence for mechanism identifiability: E — factors non-identifiable
```

## 11. Consequences & Recommendations

1. The partial Phase 10 gain **cannot** be attributed to any single data
   lever with the available inventory. H1–H5 of the Phase 12 experiment remain
   unresolved **by design**, not by lack of effort.
2. **Do not force the experiment.** Any future attempt requires either
   (a) a genuinely same-domain, larger-N secondary pool (to construct ARM B),
   or (b) data acquisition engineered to decouple activity coverage from
   composition — neither exists in the current repository.
3. A third-party external dataset (e.g., an independent screen in the *canonical*
   domain) would be needed to re-open the design; that is a data-acquisition
   decision, not a computational one.
4. Per section 28, gate E/F may make *architecture/representation* the next
   hypothesis — that direction is noted for review only and was NOT started.

## 12. Files Produced

| File | Purpose |
|---|---|
| `src/coverage_ablation/__init__.py` | Phase 12 package |
| `src/coverage_ablation/config.py` | protocol constants, stop conditions, gates, tolerances |
| `src/coverage_ablation/matching.py` | feasibility audit, matching reports, gated arm construction |
| `src/coverage_ablation/analysis.py` | arm-pool metadata, high-activity summary, feasibility output assembly |
| `src/coverage_ablation/stats.py` | paired contrasts (bootstrap CI, dz, t, Wilcoxon) |
| `scripts/run_phase12_coverage_ablation.py` | orchestration (audit → STOP) |
| `tests/test_phase12_coverage_ablation.py` | 22 unit tests |
| `results/experiments/phase12_controlled_data_coverage_ablation_20260906_145652.json` | full machine-readable result |
| `results/pools/phase12_source_provenance_*.csv` | audit-input provenance snapshots (NOT arms) |
| `results/figures/phase12_*.png` | activity / GC / entanglement feasibility figures |
| `docs/phase12_controlled_data_coverage_ablation_report.md` | this report |

## 13. Reproducibility

- Single entry point: `python3 scripts/run_phase12_coverage_ablation.py`.
- Inputs locked by sha256 (recorded in the result JSON).
- Seeded only; `numpy.random.default_rng`; no network; no cache.
- Re-running reproduces the same decision (audit is deterministic).

## 14. Git Policy

Per project policy nothing is committed or pushed. `git HEAD` and
`git status --porcelain` are recorded in the result JSON for audit.

## 15. Status

**COMPLETED — STOP (INFEASIBLE, gate F).** Next phase not started.