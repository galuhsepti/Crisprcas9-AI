# Phase 15 Data Feasibility Report

**Status:** DATA AUDIT ONLY
**Audit timestamp:** 2026-09-07T23:09:35+07:00
**Git HEAD:** `d54d2e228704d8f7b77737b0d22205115f7c54f7`
**Scope:** Gate 0 and Gate 1 feasibility only

No model was trained or retrained. No hyperparameter was tuned. No prediction file was created. Moreno-Mateos raw content was not read, screened, or used for a decision. No commit or push was performed.

## 1. Gate 0 Environment

- `HEAD`: `d54d2e2`
- `origin/main`: `d54d2e2`
- `HEAD == origin/main`: yes
- Worktree: dirty because of 21 pre-existing local modifications; no Phase 15 files were modified before this audit.
- Python: 3.11.9 from `.venv`
- Packages: NumPy 2.4.6; pandas 3.0.5; SciPy 1.17.1; scikit-learn 1.9.0; PyTorch 2.14.0+cpu; XGBoost 3.2.0; matplotlib 3.11.1; PyYAML 6.0.3.
- Phase 14 `project_root`: `/home/konta/crispr-prediction`, incompatible with this Windows workspace. It is hard-coded in `src/domain_shift/config.py`; it was not changed during this audit.
- Canonical model binaries: local `models/` directory absent.
- Canonical artifact status: `RECORDED_BUT_NOT_LOCALLY_VERIFIED`.
- Recorded canonical hashes are preserved in Phase 14 final metadata and were not recomputed because binaries are absent.

Gate 0 is conditionally reproducible for data audit purposes, but not fully reproducible for model artifact verification.

## 2. Dataset Inventory

Only these data files are physically available:

| Dataset | Local status | Scientific role |
|---|---|---|
| DeepSpCas9 | available at `data/raw/DeepSpCas9.csv` | canonical reference/training domain |
| Moreno-Mateos | present but locked | external test; no new access permitted |
| `sample_dataset.csv` | available | test fixture, not scientific data |
| DeepHF and other benchmark sources | absent | documented historical candidates only |

### DeepSpCas9 reference audit

The canonical validator and seed-42 split were used only to establish the reference distribution. This is not a new model experiment.

- Raw rows: 12,832
- Canonical-valid rows: 10,117
- Training reference: 8,599
- Validation reference: 1,518
- Sequence: 30 bp; guide `[4:24]`; PAM `[24:27]`
- Organism: human
- Assay: pooled SpCas9 lentiviral screen with NGS modification frequency
- Label: `activity` / `modFreq`; continuous modification-frequency fraction
- Missing labels after validation: 0
- Training exact 30-mer duplicates: 0
- Training GC mean/SD: 0.5512 / 0.1153
- Training activity mean/SD: 0.4260 / 0.2202
- Training activity median: 0.4363
- Training activity quantiles q05/q25/q50/q75/q95: 0.0609 / 0.2498 / 0.4363 / 0.6027 / 0.7699
- Fixed activity-bin counts `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1.0]`: 1,654 / 2,218 / 2,546 / 1,910 / 271
- DeepSpCas9 SHA-256: `6aeab30f55155ca9f1b80aae3159231ce113815354597ea4ca1d6c84ba0296df`

Full positional and k-mer vectors were not used to make a candidate-selection decision. They remain specified for a future accepted-dataset audit.

## 3. Candidate Provenance

The following candidates are documented in repository Phase 10 inventory, but their primary files are absent from this workspace. Therefore their records are not acceptance decisions.

| Candidate | Primary evidence locally available | Label/geometry summary | Status |
|---|---|---|---|
| DeepHF | No; only Phase 10 records | 21-mer guide+PAM, no genomic flanks; `Wt_Efficiency` 0–1; human pooled screens | `PREVIOUSLY_ANALYZED_CANDIDATE` |
| Doench A375 | No | A375 pooled screen; normalized activity documented as 0–1; overlap risk | `PENDING_PRIMARY_VERIFICATION` |
| Chen HEK293T | No | edited-read fraction; documented repeated/synthetic flanks | `PENDING_PRIMARY_VERIFICATION` |
| Hart HCT116 | No | log2 fitness score, not editing-efficiency fraction | `PENDING_PRIMARY_VERIFICATION` |
| Labuhn HEL | No | fluorescent-reporter fraction; small and assay-specific | `PENDING_PRIMARY_VERIFICATION` |
| Koike-Yusa mESC | No | log2 fitness score; mouse context | `PENDING_PRIMARY_VERIFICATION` |
| Gagnon zebrafish | No | percent germline mutation rate; very small | `PENDING_PRIMARY_VERIFICATION` |
| Varshney zebrafish | No | percent indel rate; very small | `PENDING_PRIMARY_VERIFICATION` |
| Teboul mouse | No | in-vivo embryo/oocyte indel outcome; extremely small | `PENDING_PRIMARY_VERIFICATION` |
| Xiang/DeepHF benchmark test set | No | benchmark/test derivative risk; flank provenance uncertain | `PENDING_PRIMARY_VERIFICATION` |

A candidate is not accepted from this inventory alone because primary sequence, label, filtering, and provenance files were not available for inspection.

## 4. DeepHF Special Assessment

DeepHF was already analyzed in Phase 10 and is not a newly discovered independent candidate.

Documented Phase 10 facts:

- 59,852 raw rows; 48,295 retained labelled rows.
- Official source contains guide+PAM without genomic flanks.
- Canonical 30-mer was constructed using constant `AAAA` and `AAA` flanks.
- 3,670 rows with missing `Wt_Efficiency` were excluded.
- Phase 10 found a small, non-significant directional CNN improvement: MAE change approximately -0.0022, with CI crossing zero.
- Activity coverage and sequence composition were confounded.

DeepHF could serve as an independent held-out domain in a future protocol only if its primary file and provenance are restored and its role is explicitly separated from new dataset discovery. Its constant-flank construction makes it unsuitable as an uncontested canonical 30-mer primary training source. It must not be rerun in Phase 15 feasibility.

## 5. Label Compatibility

Categories:

- **A — DIRECTLY COMPATIBLE:** experimentally measured SpCas9 editing-efficiency fraction with documented semantics and compatible geometry.
- **B — POTENTIALLY COMPATIBLE — SUPERVISOR REVIEW:** bounded score or related activity measurement whose biological meaning, assay comparability, or geometry requires review.
- **C — BIOLOGICALLY DIFFERENT — REJECT:** fitness/log2, reporter, percent, or in-vivo outcome not demonstrably equivalent to DeepSpCas9 modification frequency.

Current classifications:

| Candidate | Label category | Reason |
|---|---|---|
| DeepHF | B | comparable-looking fraction but different screens and missing genomic flanks; already analyzed |
| Doench A375 | B | quantitative pooled-screen activity, but semantics and overlap require primary verification |
| Chen HEK293T | B/C boundary; not accepted | edited-read fraction is plausible, but synthetic flanks prevent canonical 30-mer use |
| Hart HCT116 | C | fitness/log2 quantity |
| Labuhn HEL | B | reporter fraction, but modality and size require review |
| Koike-Yusa mESC | C | fitness/log2 quantity and mouse context |
| Gagnon/Varshney/Teboul | C or B only for secondary analysis | different organism/context, scale, or extremely small sample |
| Xiang/DeepHF benchmark test | B/C boundary; not accepted | derivative/test-set and flank provenance risk |

No label transformation or calibration was performed.

## 6. Exact Overlap Results

No new candidate primary sequences were available, so no new candidate-versus-candidate or candidate-versus-Moreno overlap audit was performed.

Documented historical results may be cited only as prior records:

- DeepHF vs Moreno: zero forward/reverse-complement overlap was reported in Phase 10.
- DeepHF vs DeepSpCas9: zero exact 30-mer overlap was reported; shared 20-mer guides were reported.
- Doench A375: 1,568/1,934 valid 30-mers were reported as exact duplicates of canonical DeepSpCas9.
- The locked Moreno raw CSV was not read in this audit.

The available DeepSpCas9 training reference itself has zero within-training exact duplicates after canonical filtering.

Required future overlap checks for any accepted candidate:

1. exact 30-mer forward;
2. exact 20-mer guide;
3. reverse-complement 30-mer where strand conventions require it;
4. reverse-complement guide where relevant;
5. within-dataset duplicate rate and label conflicts.

## 7. Near-Duplicate Methodology Review

The repository supports exact and reverse-complement overlap. It does not currently provide a validated near-duplicate audit.

Possible descriptive metrics:

- Hamming distance for equal-length aligned 20-mer or 30-mer sequences;
- edit distance for variable-length records;
- k-mer Jaccard or MinHash for larger datasets;
- source/locus grouping where target identity is available.

No near-duplicate threshold was selected, and no sequence was removed using an unapproved threshold.

**SUPERVISOR REVIEW REQUIRED:** approve the metric, threshold, grouping policy, and whether near-duplicates are excluded or retained for sensitivity analysis.

## 8. Activity Coverage

Only the DeepSpCas9 reference distribution was computed from primary data in this audit. Its fixed-bin counts are:

| Activity bin | DeepSpCas9 training count |
|---|---:|
| `[0,.2)` | 1,654 |
| `[.2,.4)` | 2,218 |
| `[.4,.6)` | 2,546 |
| `[.6,.8)` | 1,910 |
| `[.8,1.0]` | 271 |

Candidate activity coverage cannot be calculated without primary candidate files. Phase 10/11 historical descriptions are not recomputed here.

## 9. Sequence Coverage and Diversity

For future candidates with primary sequence files, the audit will calculate:

- full 30-mer and guide GC;
- A/C/G/T composition;
- 2-mer and 3-mer frequency vectors;
- positional nucleotide frequencies;
- unique sequence and unique-guide fractions;
- sequence entropy where defined;
- Jensen-Shannon divergence from the DeepSpCas9 training reference;
- exact, reverse-complement, and approved near-duplicate overlap.

For DeepSpCas9, the training reference has 8,599 unique 30-mers and 0 duplicate rows. Its GC mean is 0.5512 and nucleotide frequencies are approximately A 0.2207, C 0.2552, G 0.2997, T 0.2244.

No candidate is accepted merely because it would increase sample count. Size, diversity, and domain coverage must be reported separately.

## 10. Joint Support

For every candidate with primary data, calculate:

- GC × fixed activity-bin counts;
- common-support cell counts against DeepSpCas9;
- activity-tail counts within GC strata;
- sequence-composition support by activity bin;
- domain-specific empty cells.

A candidate that increases `N` but does not add support in underrepresented activity or sequence regions will not qualify as meaningful diversification.

No model predictions or prediction errors will be used in this feasibility audit.

## 11. Candidate Decision Table

| Candidate | Status | Gate rationale |
|---|---|---|
| DeepHF | `PREVIOUSLY_ANALYZED_CANDIDATE` | Not new; primary file absent; constant-flank limitation; Phase 10 already confounded coverage and composition |
| Doench A375 | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; documented overlap and label semantics require verification |
| Chen HEK293T | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; historical synthetic-flank concern must be confirmed from source |
| Hart HCT116 | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; documented label appears biologically different |
| Labuhn HEL | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; small reporter-specific dataset |
| Koike-Yusa mESC | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; documented fitness/log2 label |
| Gagnon/Varshney/Teboul | `PENDING_PRIMARY_VERIFICATION` | Primary files absent; small or biologically different contexts |
| Xiang/DeepHF benchmark test | `PENDING_PRIMARY_VERIFICATION` | Primary file absent; derivative/test-set and flank risks |
| `sample_dataset.csv` | `REJECT` | Test fixture, not a documented scientific dataset |

No candidate receives `ACCEPT_FOR_PHASE15`.

## 12. Limitations

- No new candidate primary data were available locally.
- Public repository documentation is insufficient for final acceptance.
- Canonical model binaries are absent and hashes cannot be locally re-verified.
- Phase 14 has a hard-coded Linux root incompatible with this Windows workspace.
- The current repository lacks near-duplicate implementation and an approved threshold.
- Moreno was intentionally not read, so new candidate-versus-Moreno contamination checks were not performed.
- No causal claim can be made about activity coverage, diversity, or domain compatibility from the current feasibility inventory.

## 13. Gate 1 Decision

**Gate 1: FAIL for new accepted training data.**

No candidate currently satisfies the acceptance criteria because no newly inspected independent candidate has accessible primary data and complete documentation in this workspace. DeepHF cannot satisfy the “new candidate” requirement because it was previously analyzed and has a documented constant-flank limitation.

**Feasibility outcome: INFEASIBLE under the currently available workspace evidence.**

Per protocol, stop new model development. This is a data-readiness stop, not a claim that no suitable dataset exists anywhere.

## 14. Supervisor Decisions Required

1. Approve whether DeepHF may be used again only as a previously analyzed held-out domain.
2. Approve acceptable label semantics and assay differences.
3. Approve primary-data sources and archival requirements.
4. Approve near-duplicate metric and threshold.
5. Approve treatment of shared guides across independent assays.
6. Approve whether reporter, organism-different, or fitness datasets may be secondary-only.
7. Approve restoration and archival of canonical model binaries.
8. Approve a platform-independent Phase 14 path configuration before reproducibility work.

## 15. Recommended Next Step

STOP new model development and obtain primary files plus documentation for one or more candidate datasets before reopening Phase 15 Gate 1.

## Final Status

PHASE 15 FEASIBILITY STATUS:
INFEASIBLE

GATE 0:
CONDITIONAL

GATE 1:
FAIL

MORENO:
LOCKED — NO NEW ACCESS

MODEL TRAINING:
NOT PERFORMED

RECOMMENDED NEXT ACTION:
Obtain and archive primary candidate data and documentation, then repeat Gate 1 without accessing Moreno.
