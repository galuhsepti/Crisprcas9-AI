# Phase 17F - Final Dataset Acceptance Gate

## 1. Final candidate decision table

| candidate_id | provenance status | experimental status | sequence status | label status | independence status | information-value status | final acceptance status | exact reason |
|---|---|---|---|---|---|---|---|---|
| `crispron_2021_rth_tools` | `PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED` | `NOT_ESTABLISHED` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | `LABEL_INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_PROVENANCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | Primary experimental training table was not identified; sequence and label evidence remain insufficient. |
| `crisprpredseq_2020_bmc_additional_files` | `DERIVED_OR_PROCESSED` | `ESTABLISHED_SCREEN_OR_OBSERVATION_NOT_NECESSARILY_COMPATIBLE` | `SEQUENCE_INCOMPATIBLE` | `LABEL_INCOMPATIBLE` | `DERIVED_OR_PROCESSED_RELATIONSHIP` | `LIMITED_INFORMATION_VALUE` | `REJECTED_FOR_CURRENT_TARGET` | Observed 23-mer guide+PAM records have binary labels and processed/derived relationship; accepting would require prohibited label and sequence rescue. |
| `deep_hf_2019_public_data_listing` | `PRIMARY_FILE_NOT_RECOVERED` | `NOT_ESTABLISHED` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | `LABEL_INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_PROVENANCE` | `INSUFFICIENT_EVIDENCE` | `ALREADY_ANALYZED_NOT_NEW` | DeepHF was already analyzed in Phases 10-12 and no new Phase 17 primary local file established it as a new independent dataset. |
| `doench_2016_orcs_publication_screens` | `PRIMARY_VERIFIED_WITH_METADATA_GAPS` | `ESTABLISHED_SCREEN_OR_OBSERVATION_NOT_NECESSARILY_COMPATIBLE` | `SEQUENCE_INCOMPATIBLE` | `LABEL_INCOMPATIBLE` | `SHARED_SOURCE_STUDY` | `LIMITED_INFORMATION_VALUE` | `REJECTED_FOR_CURRENT_TARGET` | Guide-only screen files expose STARS/rank/enrichment/log-fold-change style values, not compatible continuous activity labels; same-study relationships remain unresolved. |
| `sgdesigner_2020_public_data_listing` | `PRIMARY_FILE_NOT_RECOVERED` | `NOT_ESTABLISHED` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | `LABEL_INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_PROVENANCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | No verified primary file, sequence evidence, label evidence, or independence evidence was recovered. |
| `corsi_2022_free_energy_pam_context` | `DISCOVERY_ONLY` | `NOT_ESTABLISHED` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | `LABEL_INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_PROVENANCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | Discovery-only candidate with no recovered primary file and no verified sequence or label evidence. |

## 2. Accepted candidates

accepted_for_future_pipeline = []

## 3. Rejected candidates

- `crisprpredseq_2020_bmc_additional_files`: Observed 23-mer guide+PAM records have binary labels and processed/derived relationship; accepting would require prohibited label and sequence rescue.
- `doench_2016_orcs_publication_screens`: Guide-only screen files expose STARS/rank/enrichment/log-fold-change style values, not compatible continuous activity labels; same-study relationships remain unresolved.

## 4. Conditional candidates

- None.

## 5. Already-analyzed candidates

- `deep_hf_2019_public_data_listing`: DeepHF was already analyzed in Phases 10-12 and no new Phase 17 primary local file established it as a new independent dataset.

## 6. Tests

- Focused Phase 17F tests: 14 passed.
- Phase 17A-17E deterministic tests: 45 passed across separate deterministic runs.
- Existing project regression tests excluding Phase 17A-17F: 457 passed.
- Warnings: openpyxl unsupported-extension warnings while reading Doench XLSX files in Phase 17B/17D/17E and Phase 16 tests; one XGBoost pickle-version warning; constant-input correlation warnings in a Phase 9D edge-case test.
- Failures: none in completed runs.

## 7. Canonical integrity

- DeepSp unchanged: True
- Moreno untouched: True; raw Moreno hash was not computed because Moreno remains locked.
- Models unchanged: True according to canonical integrity guard.
- Canonical predictions unchanged: True; `results/predictions/*` was not modified by this phase.

## 8. Scope confirmation

- `no_model_training`: True
- `no_model_evaluation`: True
- `no_hyperparameter_tuning`: True
- `no_moreno_raw_data_accessed`: True
- `no_dataset_pooling`: True
- `no_combined_training_dataset`: True
- `no_new_training_split`: True
- `no_label_transformation`: True
- `no_label_harmonization`: True
- `no_sequence_transformation`: True
- `no_30mer_construction`: True
- `no_reverse_complement_assumption`: True
- `no_near_duplicate_or_similarity_threshold_analysis`: True
- `no_dataset_construction`: True
- `no_accepted_dataset_integrated`: True
- `no_continual_learning`: True
- `no_phase18`: True
- `no_canonical_artifact_modifications`: True
- `no_model_improvement_claim`: True
- `no_generalization_claim`: True

Population distinction preserved:

- DeepSp Phase 16E distribution n=12832
- DeepSp canonical modeling n=10117
- train=8599
- validation=1518

## 9. Git state

```bash
git status
?? data/phase17/
?? docs/phase17a_dataset_recovery_report.md
?? docs/phase17b_primary_file_verification_report.md
?? docs/phase17c_sequence_label_compatibility_report.md
?? docs/phase17d_independence_overlap_report.md
?? docs/phase17e_distribution_information_value_report.md
?? results/phase17a_dataset_recovery_20260911_140154.json
?? results/phase17a_provenance_20260911_140154.json
?? results/phase17b_primary_file_verification_20260911_143300.json
?? results/phase17c_sequence_label_compatibility_20260911_144924.json
?? results/phase17d_independence_overlap_20260911_145754.json
?? results/phase17e_distribution_information_value_20260911_190523.json
?? scripts/run_phase17a_dataset_recovery.py
?? scripts/run_phase17b_primary_file_verification.py
?? scripts/run_phase17c_compatibility_audit.py
?? scripts/run_phase17d_independence_overlap_audit.py
?? scripts/run_phase17e_distribution_audit.py
?? scripts/run_phase17f_dataset_acceptance_gate.py
?? src/dataset_recovery/
?? tests/test_phase17a_dataset_recovery.py
?? tests/test_phase17b_primary_file_verification.py
?? tests/test_phase17c_compatibility.py
?? tests/test_phase17d_independence_overlap.py
?? tests/test_phase17e_distribution.py
?? tests/test_phase17f_dataset_acceptance_gate.py

git diff --stat


git diff --name-only


git log -2 --oneline
e49cdfe audit Phase 16 dataset landscape and reproducibility closure
0853b66 docs: add project master handover
```

## 10. Final status

PHASE 17F PASS — FINAL DATASET ACCEPTANCE GATE COMPLETE
