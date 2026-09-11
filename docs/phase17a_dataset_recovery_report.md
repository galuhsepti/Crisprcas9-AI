# Phase 17A - Public Dataset Recovery and Acquisition Audit

## 1. Objective

Phase 17A is an acquisition/provenance audit only. It recovers public candidate primary files where possible, preserves original files under `data/phase17/raw/`, records SHA-256 identities, and inspects file structure only.

No model training, model evaluation, tuning, pooling, label transformation, label harmonization, Moreno access, reverse-complement matching, near-duplicate analysis, continual learning, or dataset acceptance was performed.

## 2. Candidate Summary

| Candidate | Carry-over/New | Primary file located? | Status | Local files / SHA-256 | Key unresolved items |
|---|---|---:|---|---|---|
| `crispron_2021_rth_tools` | Phase 16 carry-over | True | `RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA` | `data/phase17/raw/crispron_2021_rth_tools_main.zip`<br>`3fa17a7cb3de043703ec3090505c96340e6da13f23acda8fd7f7268a3f572a9b` | sequence_orientation, sequence_geometry, pam_definition, label_semantics |
| `crisprpredseq_2020_bmc_additional_files` | Phase 16 carry-over | True | `RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA` | `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`<br>`f6da050009616e5bb50a8e66e9429830e89b59e607d0c740cebc364066f34a93`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`<br>`6dbd8c62a9e1e93d2a6b265de13e5ea6b287e9c605186223611a2be8c7968357`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`<br>`466813efdaf0dcca7cbd6c4690a99be9ca47593671174f3d72d87fef669e204d`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`<br>`9b19ca010304b15aab9f2e58962d00b811b0c6c368b2e6b2847da8dc65a00a47` | sequence_orientation, sequence_geometry, pam_definition, label_semantics |
| `doench_2016_orcs_publication_screens` | Phase 16 carry-over | True | `RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA` | `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`<br>`3fa01c4adad9dd624b67d5eacb982fc8f7448461e6a2febe8faefe5e7eefe7b8`<br>`data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`<br>`88ff9066c2f6f93cf2917a83c8bfd10a26378af60853358dda3442ade24c9030`<br>`data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`<br>`2a7018e31f9e6abd2ac5355b7d8c900c9f979e133db22f638197ee908fd7f399`<br>`data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`<br>`9d00508ea5f8af2ac809b0646497ef7a56e484aa4e3b99d779e89cdea21eea58` | sequence_orientation, sequence_geometry, pam_definition, label_semantics |
| `deep_hf_2019_public_data_listing` | Phase 16 carry-over | False | `PRIMARY_FILE_NOT_FOUND` | none | sequence_orientation, sequence_geometry, pam_definition, label_semantics |
| `sgdesigner_2020_public_data_listing` | Phase 16 carry-over | False | `INSUFFICIENT_PROVENANCE` | none | sequence_orientation, sequence_geometry, pam_definition, label_semantics |
| `corsi_2022_free_energy_pam_context` | New discovery | False | `DISCOVERY_ONLY` | none | sequence_orientation, sequence_geometry, pam_definition, label_semantics |

## 3. Recovered Primary Files

- `data/phase17/raw/crispron_2021_rth_tools_main.zip`; SHA-256 `3fa17a7cb3de043703ec3090505c96340e6da13f23acda8fd7f7268a3f572a9b`; candidate `crispron_2021_rth_tools`
- `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`; SHA-256 `f6da050009616e5bb50a8e66e9429830e89b59e607d0c740cebc364066f34a93`; candidate `crisprpredseq_2020_bmc_additional_files`
- `data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`; SHA-256 `6dbd8c62a9e1e93d2a6b265de13e5ea6b287e9c605186223611a2be8c7968357`; candidate `crisprpredseq_2020_bmc_additional_files`
- `data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`; SHA-256 `466813efdaf0dcca7cbd6c4690a99be9ca47593671174f3d72d87fef669e204d`; candidate `crisprpredseq_2020_bmc_additional_files`
- `data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`; SHA-256 `9b19ca010304b15aab9f2e58962d00b811b0c6c368b2e6b2847da8dc65a00a47`; candidate `crisprpredseq_2020_bmc_additional_files`
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`; SHA-256 `3fa01c4adad9dd624b67d5eacb982fc8f7448461e6a2febe8faefe5e7eefe7b8`; candidate `doench_2016_orcs_publication_screens`
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`; SHA-256 `88ff9066c2f6f93cf2917a83c8bfd10a26378af60853358dda3442ade24c9030`; candidate `doench_2016_orcs_publication_screens`
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`; SHA-256 `2a7018e31f9e6abd2ac5355b7d8c900c9f979e133db22f638197ee908fd7f399`; candidate `doench_2016_orcs_publication_screens`
- `data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`; SHA-256 `9d00508ea5f8af2ac809b0646497ef7a56e484aa4e3b99d779e89cdea21eea58`; candidate `doench_2016_orcs_publication_screens`

## 4. Candidates Not Recovered

- `deep_hf_2019_public_data_listing`: PRIMARY_FILE_NOT_FOUND - Official supplementary XLSX URL located but not recoverable locally due media download/browser-check response.
- `sgdesigner_2020_public_data_listing`: INSUFFICIENT_PROVENANCE - Only discovery/catalogue-level evidence available in current project records.
- `corsi_2022_free_energy_pam_context`: DISCOVERY_ONLY - Discovery-only candidate; primary file acquisition deferred because direct Springer media download was not verified locally.

## 5. Structure Inspection Only

The audit records row counts, column names, obvious sequence columns, obvious label columns, sheet names, missingness, and duplicate rows where technically available. It does not decide compatibility with the canonical target.

## 6. Canonical Integrity

Canonical protected-file hashes unchanged: `True`.

## 7. Final Phase 17A Status

Phase 17A stops after public-file recovery and provenance/structure recording. Dataset acceptance is deferred to later gates.
