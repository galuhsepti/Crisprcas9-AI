# Phase 16F Final Dataset Landscape Gate

**Protocol:** `phase16f-dataset-gate-v1`  
**Scope:** Final data-readiness and dataset-acceptance decision only.

This gate consumes the frozen Phase 16B-16E JSON artifacts. No modeling,
evaluation, hyperparameter tuning, pooling, label transformation, sequence
transformation, calibration, adaptation, or automatic dataset promotion was
performed. Moreno raw data remained locked and was not accessed. No
generalization claim was made.

## Decision Table

| candidate | provenance | sequence | label | overlap / independence | final status |
|---|---|---|---|---|---|
| Doench 2016 ORCS screens | primary files verified | 20-nt guide-only; PAM/orientation unresolved | rank, enrichment, and log-fold-change semantics; not the target label | 2,333 exact guide overlaps; duplicate/conflict evidence; independence not established; RC not assessed | `REJECTED_FOR_CURRENT_TARGET` |
| CRISPRpred(SEQ) | supplementary files verified | 23-mer guide+PAM; noncanonical geometry | binary 0/1; no rescue | 1,966 exact guide overlaps; 4,284 duplicate excess; 402 conflicts; derived/processed independence risk; RC not assessed | `REJECTED_FOR_CURRENT_TARGET` |
| CRISPRon 2021 | primary data not verified | unavailable | unavailable | not assessable | `INSUFFICIENT_EVIDENCE` |
| DeepHF | previously analyzed | prior Phase 10 evidence only | prior Phase 10 evidence only | not a new Phase 16 candidate | `ALREADY_ANALYZED_NOT_NEW` |
| SgDesigner | primary data not verified | unavailable | unavailable | not assessable | `INSUFFICIENT_EVIDENCE` |

No candidate was accepted for the future pipeline. The rejected candidates are
incompatible with the current thesis target, not necessarily scientifically
useless for other questions.

## Evidence And Limitations

Doench cannot be accepted because its observed screen outputs are rank,
enrichment, and log-fold-change-style measures, while its sequence schema is
guide-only. Converting these labels or creating canonical 30-mers would be
prohibited harmonization/transformation.

CRISPRpred(SEQ) cannot be accepted because its labels are binary and its
observed sequence is a 23-mer guide+PAM. The exact overlap audit found 1,966
shared guides with DeepSpCas9, 4,284 duplicate-record excess, and 402
identical-sequence label conflicts. No binary-label rescue was performed.

CRISPRon and SgDesigner remain unresolved because Phase 16B-16E did not verify
their primary experimental files. Absence of assessable overlap is not evidence
of independence.

DeepHF is explicitly preserved as prior Phase 10-12 evidence. It is not counted
as an independent Phase 16 discovery or a rediscovered candidate.

## Phase 16E Clarification Checks

- DeepSpCas9 distribution-audit population: **12,832 raw rows**.
- DeepSpCas9 canonical-valid modeling population: **10,117 rows**.
- These populations are distinct. The checked-in Phase 16E report does not yet
  state that distinction explicitly; this remains an unresolved documentation
  limitation and no prior-phase file was modified.
- The overlap parser only emits sequence values from selected workbook fields
  when they contain A/C/G/T characters. Ordinary non-DNA workbook metadata,
  rank values, and STARS values are therefore excluded from sequence parsing.
  A DNA-looking string in a selected field could still pass this character
  filter.
- Phase 16D reports 214,820 Doench sequence records while Phase 16E reports
  194,653 usable sequence records. This discrepancy remains unresolved.

## Locks

- No modeling or model evaluation was performed.
- Moreno remained locked; no Moreno raw data was accessed.
- No pooling, label transformation, or sequence transformation occurred.
- No generalization claim was made.
- No dataset was automatically promoted.

## Unresolved Questions

See the timestamped Phase 16F JSON for the complete machine-readable list.
The principal limitations are unresolved Doench count reconciliation, unverified
orientation/PAM information, incomplete provenance for CRISPRon and SgDesigner,
and unresolved processed-parent relationships for CRISPRpred(SEQ).