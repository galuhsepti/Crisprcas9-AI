# Phase 17E Distribution and Information-Value Audit

## 1. Dataset classification

| candidate_id | dataset type | verified sequence? | verified continuous activity? | analysis performed | information-value status |
|---|---|---|---|---|---|
| `corsi_2022_free_energy_pam_context` | `INSUFFICIENT_DATA` | False | False | sequence=False; activity=False | `INSUFFICIENT_EVIDENCE` |
| `crispron_2021_rth_tools` | `INSUFFICIENT_DATA` | False | False | sequence=False; activity=False | `INSUFFICIENT_EVIDENCE` |
| `crisprpredseq_2020_bmc_additional_files` | `INCOMPATIBLE_LABEL_DATA` | True | False | sequence=True; activity=False | `LIMITED_INFORMATION_VALUE` |
| `deep_hf_2019_public_data_listing` | `INSUFFICIENT_DATA` | False | False | sequence=False; activity=False | `INSUFFICIENT_EVIDENCE` |
| `doench_2016_orcs_publication_screens` | `INCOMPATIBLE_LABEL_DATA` | True | False | sequence=True; activity=False | `LIMITED_INFORMATION_VALUE` |
| `sgdesigner_2020_public_data_listing` | `INSUFFICIENT_DATA` | False | False | sequence=False; activity=False | `INSUFFICIENT_EVIDENCE` |

## 2. Sequence distribution

Reference for sequence composition comparison = DeepSpCas9 Phase 16E distribution population (n=12,832).

| candidate_id | n valid A/C/G/T | length distribution | mean GC | median GC | sd GC | A | C | G | T | comparison |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `corsi_2022_free_energy_pam_context` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | insufficient verified sequence evidence |
| `crispron_2021_rth_tools` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | insufficient verified sequence evidence |
| `crisprpredseq_2020_bmc_additional_files` | 16749 | {23: 16749} | 0.5801 | 0.5652 | 0.0840 | 0.2341 | 0.2475 | 0.3326 | 0.1859 | candidate exhibits a different sequence-composition regime; mean GC delta 0.0259 |
| `deep_hf_2019_public_data_listing` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | insufficient verified sequence evidence |
| `doench_2016_orcs_publication_screens` | 194653 | {20: 194653} | 0.5218 | 0.5000 | 0.1163 | 0.2633 | 0.2713 | 0.2505 | 0.2149 | candidate exhibits a different sequence-composition regime; mean GC delta -0.0324 |
| `sgdesigner_2020_public_data_listing` | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | insufficient verified sequence evidence |

## 3. Activity distribution

NO_CANDIDATE_AVAILABLE_FOR_ACTIVITY_DISTRIBUTION_ANALYSIS

No candidate label was analyzed as thesis activity because Phase 17C did not establish `LABEL_COMPATIBLE` for any recovered candidate. CRISPRpred binary labels and Doench screen/rank/enrichment/log-fold-change values were not transformed or treated as activity.

## 4. Information-value assessment

- `corsi_2022_free_energy_pam_context`: `INSUFFICIENT_EVIDENCE`. Verified local sequence/activity evidence is insufficient.
- `crispron_2021_rth_tools`: `INSUFFICIENT_EVIDENCE`. Verified local sequence/activity evidence is insufficient.
- `crisprpredseq_2020_bmc_additional_files`: `LIMITED_INFORMATION_VALUE`. Sequence composition can be described, but labels are incompatible or not usable as thesis activity.
- `deep_hf_2019_public_data_listing`: `INSUFFICIENT_EVIDENCE`. Verified local sequence/activity evidence is insufficient.
- `doench_2016_orcs_publication_screens`: `LIMITED_INFORMATION_VALUE`. Sequence composition can be described, but labels are incompatible or not usable as thesis activity.
- `sgdesigner_2020_public_data_listing`: `INSUFFICIENT_EVIDENCE`. Verified local sequence/activity evidence is insufficient.

## 5. Canonical population distinction

DeepSp Phase 16E distribution n=12,832

DeepSp canonical modeling n=10,117

train=8,599

validation=1,518

## 6. Tests

- Focused Phase 17E tests: 10 passed.
- Phase 17A-17D regression tests: 35 passed across separate deterministic runs.
- Existing project regression tests excluding Phase 17A-17E: 457 passed.
- Full all-in-one `pytest -q`: timed out after 15 minutes before returning a complete summary; results above were obtained through scoped deterministic runs.
- Warnings: openpyxl unsupported-extension warnings while reading Doench XLSX files; one XGBoost pickle-version warning; constant-input correlation warnings in a Phase 9D edge-case test; one joblib worker warning during XGBoost cross-validation.
- Failures: none in completed runs.

## 7. Scope confirmation

- `no_model_training`: True
- `no_model_evaluation`: True
- `no_hyperparameter_tuning`: True
- `no_moreno_raw_data_accessed`: True
- `no_dataset_pooling`: True
- `no_combined_training_dataset`: True
- `no_retraining`: True
- `no_calibration`: True
- `no_adaptation_or_domain_weighting`: True
- `no_label_transformation`: True
- `no_label_harmonization`: True
- `no_sequence_transformation`: True
- `no_30mer_construction`: True
- `no_reverse_complement`: True
- `no_near_duplicate_analysis`: True
- `no_continual_learning`: True
- `no_dataset_acceptance`: True

## 8. Git state

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
?? src/dataset_recovery/
?? tests/test_phase17a_dataset_recovery.py
?? tests/test_phase17b_primary_file_verification.py
?? tests/test_phase17c_compatibility.py
?? tests/test_phase17d_independence_overlap.py
?? tests/test_phase17e_distribution.py

git diff --stat


git diff --name-only


git log -2 --oneline
e49cdfe audit Phase 16 dataset landscape and reproducibility closure
0853b66 docs: add project master handover
```

## 9. Final status

PHASE 17E PASS — DISTRIBUTION/INFORMATION-VALUE AUDIT COMPLETE
