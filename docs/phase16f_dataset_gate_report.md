# Phase 16F - Final Dataset Landscape Gate

## 1. Objective

Phase 16F performs the final evidence-based dataset-readiness gate for the Phase 16 dataset landscape. This is a data readiness and dataset acceptance decision only.

No candidate is selected because it is large, public, has a primary file, or has sequence records. A candidate can be accepted only when provenance, primary data, sequence definition, label semantics, compatibility, and independence are all sufficiently established without prohibited transformation.

## 2. Locked Constraints

Moreno remains locked. Phase 16F did not open Moreno raw data, read Moreno raw files, recalculate Moreno statistics, use Moreno as an acceptance criterion, or modify any Moreno artifact.

Phase 16F did not train models, evaluate models, tune hyperparameters, pool datasets, transform labels, reconstruct sequences, adapt or calibrate models, create a new training dataset, claim improved generalization, implement continual learning, or promote any dataset into model training.

Phase 16F does not establish model improvement or generalization.

## 3. Evidence Sources

This gate uses only Phase 16A-16E evidence and existing project documentation. The machine-readable gate consumes these frozen artifacts:

| Phase | Artifact |
|---|---|
| 16B | `results/phase16b_primary_file_verification_20260908_151501.json` |
| 16C | `results/phase16c_compatibility_audit_20260908_152340.json` |
| 16D | `results/phase16d_overlap_independence_audit_20260908_153958.json` |
| 16E | `results/phase16e_distribution_audit_20260908_170415.json` |

Phase 16C compatibility decisions are carried forward and not overridden. Phase 16D exact overlap and independence evidence is carried forward without near-duplicate thresholds. Phase 16E distribution findings are descriptive only.

Important Phase 16E clarification: DeepSpCas9 n=12,832 is the Phase 16E distribution-audit population, not the canonical modeling population. The canonical modeling population remains valid n=10,117, train n=8,599, and validation n=1,518.

Important Phase 16D clarification: overlap parsing excludes ordinary non-DNA workbook metadata, rank values, and STARS values from sequence parsing by restricting emitted sequence values to selected fields containing A/C/G/T characters. DNA-looking strings in selected fields remain a parser limitation.

## 4. Candidate-by-Candidate Assessment

### crispron_2021_rth_tools

CRISPRon remains insufficiently evidenced for current acceptance. Phase 16B obtained the official repository source archive, but the primary experimental sgRNA training table was not identified. Phase 16C therefore retained insufficient provenance. Phase 16D had no verified sequence file for exact overlap assessment. Phase 16E had no records for distribution analysis.

Final status: `INSUFFICIENT_EVIDENCE`.

### crisprpredseq_2020_bmc_additional_files

CRISPRpred(SEQ) supplementary CSV files were obtained and hashed, but Phase 16B/16C left the parent or processed upstream relationship unresolved. Phase 16C found binary 0/1 labels and 23-mer guide+PAM geometry, which are incompatible with the current continuous 30-mer DeepSpCas9 activity target. No binary-label rescue, label harmonization, or sequence transformation was performed.

Phase 16D found no assessable 30-mer overlap, 1,966 shared unique guides with DeepSpCas9, 4,284 duplicate-record excess, 402 identical-sequence label conflicts, and `RC_OVERLAP_NOT_ASSESSED`. Phase 16E recorded GC n=16,749, mean 0.5801, median 0.5652, and sd 0.0840. Labels were not analyzed as activity.

Final status: `REJECTED_FOR_CURRENT_TARGET`.

### deep_hf_2019_public_data_listing

DeepHF is preserved as prior Phase 10-12 project evidence and is not counted as a new independent Phase 16 dataset. Phase 16B did not restore the primary file in the Phase 16 raw acquisition. Phase 16C therefore retained insufficient provenance in the Phase 16 context, and Phase 16D/16E did not independently re-establish sequence, overlap, or distribution evidence.

Final status: `ALREADY_ANALYZED_NOT_NEW`.

### doench_2016_orcs_publication_screens

Doench ORCS supplementary screen workbooks were obtained and hashed. Phase 16C found guide-only sequence geometry and screen rank, enrichment, STARS, or log-fold-change style outputs, not continuous SpCas9 editing activity compatible with the thesis target. Accepting the dataset would require prohibited sequence transformation and label harmonization.

Phase 16D found no assessable 30-mer overlap, 2,333 shared unique guides with DeepSpCas9, 57,240 duplicate-record excess, 296 identical-sequence label conflicts, and `RC_OVERLAP_NOT_ASSESSED`. Independence was not established. Phase 16E recorded GC n=194,653, mean 0.5218, median 0.5000, and sd 0.1163. Labels were not analyzed as activity. The Phase 16D record count of 214,820 and Phase 16E usable sequence count of 194,653 remain an unresolved documentation/data-processing limitation, not a basis for acceptance.

Final status: `REJECTED_FOR_CURRENT_TARGET`.

### sgdesigner_2020_public_data_listing

SgDesigner remains a discovery lead only. No official primary file or confirmed repository data file was verified in Phase 16B. Phase 16C retained insufficient provenance. Phase 16D had no verified sequence file for exact overlap assessment, and Phase 16E had no records for distribution analysis.

Final status: `INSUFFICIENT_EVIDENCE`.

## 5. Final Evidence Matrix

| Candidate | Provenance | Primary File | Sequence | Compatibility | Label | Overlap | Independence | Distribution/Info Value | Final Status | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| `crispron_2021_rth_tools` | Insufficient; primary experimental table unresolved | Source archive obtained, primary table unresolved | No verified Phase 16 sequence file | `INSUFFICIENT_PROVENANCE` carried forward | Unknown/unverified | No 30-mer or guide overlap assessable; RC not assessed | Insufficient sequence/provenance information | No records analyzed | `INSUFFICIENT_EVIDENCE` | Critical provenance, sequence, label, overlap, and distribution evidence unavailable |
| `crisprpredseq_2020_bmc_additional_files` | Supplementary files verified; upstream processed relationship unresolved | Four official BMC supplementary CSV files obtained and hashed | 23-mer guide+PAM, not canonical 30-mer; orientation unknown | `INCOMPATIBLE` carried forward | Binary 0/1, not continuous activity | No 30-mer assessable; 1,966 shared guides; 4,284 duplicate excess; 402 conflicts; RC not assessed | Not established; derived/processed risk | GC differs descriptively; labels not analyzed as activity | `REJECTED_FOR_CURRENT_TARGET` | Binary labels and noncanonical sequence geometry require prohibited rescue for current thesis target |
| `deep_hf_2019_public_data_listing` | Prior Phase 10-12 relationship preserved; Phase 16 primary file not restored | Primary file not obtained in 16B | No verified Phase 16 sequence file | `INSUFFICIENT_PROVENANCE` in Phase 16C context | Not re-established in Phase 16 | No Phase 16 overlap assessable; RC not assessed | Not a new Phase 16 independent candidate | No Phase 16E records analyzed | `ALREADY_ANALYZED_NOT_NEW` | Rediscovery does not create a new independent accepted dataset |
| `doench_2016_orcs_publication_screens` | Publication/ORCS supplementary files verified; same-study relationships unresolved at acceptance level | Four supplementary XLSX screen files obtained and hashed | Guide-only; PAM/orientation unknown; canonical 30-mer unavailable without transformation | `INCOMPATIBLE` carried forward | Rank/enrichment/STARS/log-fold-change screen outputs, not continuous editing activity | No 30-mer assessable; 2,333 shared guides; 57,240 duplicate excess; 296 conflicts; RC not assessed | Not established | GC differs descriptively; labels not analyzed as activity | `REJECTED_FOR_CURRENT_TARGET` | Screen labels and guide-only geometry require prohibited harmonization/transformation for current thesis target |
| `sgdesigner_2020_public_data_listing` | Insufficient primary-source verification | No official primary file obtained | No verified sequence file | `INSUFFICIENT_PROVENANCE` carried forward | Unknown/unverified | No 30-mer or guide overlap assessable; RC not assessed | Insufficient sequence/provenance information | No records analyzed | `INSUFFICIENT_EVIDENCE` | Primary file, label semantics, sequence geometry, and independence are not established |

## 6. Dataset Acceptance Decisions

No Phase 16 candidate is accepted for the future pipeline.

Accepted for future pipeline: none.

Rejected for current target: `crisprpredseq_2020_bmc_additional_files`, `doench_2016_orcs_publication_screens`.

Insufficient evidence: `crispron_2021_rth_tools`, `sgdesigner_2020_public_data_listing`.

Already analyzed, not new: `deep_hf_2019_public_data_listing`.

`INCOMPATIBLE` means incompatible with the current thesis target. It does not mean a dataset is scientifically useless for other questions.

## 7. Unresolved Scientific Questions

- CRISPRon primary experimental table identity, row schema, sequence geometry, label units, SHA-256, and parent relationships remain unresolved.
- CRISPRpred(SEQ) primary parent or processed-source relationship remains incompletely established.
- DeepHF Phase 16 primary file restoration remains unresolved; future reuse requires explicit protocol handling of Phase 10-12 evidence.
- Doench PAM, orientation, guide-to-target context, and Phase 16D versus 16E record-count reconciliation remain unresolved.
- SgDesigner primary repository/file identity, publication linkage, label definition, sequence geometry, and independence remain unresolved.
- Reverse-complement overlap remains `RC_OVERLAP_NOT_ASSESSED` for all candidates because orientation/strand was not verified.
- Near-duplicate status remains not established because no Hamming, edit-distance, or near-duplicate threshold was approved.

## 8. What This Phase Does NOT Establish

Phase 16F does not establish model improvement, improved generalization, external performance, biological causality, dataset superiority, or readiness for retraining.

Phase 16F does not establish that any candidate is independent merely because 30-mer overlap was not assessable. Phase 16F does not call overlap data leakage because leakage was not directly established.

Phase 16F does not convert fractions to percentages, convert ranks or enrichment scores to activity, convert binary labels to continuous activity, reconstruct 30-mers, or infer reverse-complement overlap.

## 9. Reproducibility

The Phase 16F implementation is in `src/dataset_landscape/gates.py` and `scripts/run_phase16f_dataset_gate.py`. The timestamped JSON artifact is written to `results/phase16f_dataset_gate_<timestamp>.json`.

Required guards are encoded in the JSON: no modeling, no Moreno raw-data access, no pooling, no label transformation, no sequence transformation, no generalization claim, and no dataset promotion into training.

## 10. Final Phase 16 Status

Phase 16F stops after the final dataset-readiness decision.

Final Phase 16 dataset landscape decision: no candidate is accepted for future pipeline inclusion under the current evidence and current thesis target.
