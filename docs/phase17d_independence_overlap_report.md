# Phase 17D - Independence and Exact Overlap Audit

## 1. Objective

Phase 17D audits exact sequence/guide overlap, internal duplication, label conflicts, and provenance relationships. It does not decide acceptance.

No model training, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, near-duplicate analysis, reverse-complement matching, continual learning, or dataset acceptance was performed.

## 2. Candidate Overlap Table

| Candidate | Sequence availability | Geometry | Exact 30-mer overlap | Exact guide overlap | Duplicate sequence excess | Label-conflict sequences | Provenance status |
|---|---|---|---|---|---:|---:|---|
| `corsi_2022_free_energy_pam_context` | INSUFFICIENT_SEQUENCE_INFORMATION | NO_VERIFIED_SEQUENCE_TABLE | INSUFFICIENT_SEQUENCE_INFORMATION | INSUFFICIENT_SEQUENCE_INFORMATION | 0 | 0 | `INSUFFICIENT_PROVENANCE` |
| `crispron_2021_rth_tools` | INSUFFICIENT_SEQUENCE_INFORMATION | NO_VERIFIED_SEQUENCE_TABLE | INSUFFICIENT_SEQUENCE_INFORMATION | INSUFFICIENT_SEQUENCE_INFORMATION | 0 | 0 | `INSUFFICIENT_PROVENANCE` |
| `crisprpredseq_2020_bmc_additional_files` | VERIFIED_OBSERVED_SEQUENCE_OR_GUIDE_FIELD | ['GUIDE_PLUS_PAM_23MER'] | NO_CANONICAL_30MER_GEOMETRY | 1966 | 4284 | 402 | `DERIVED_OR_PROCESSED_RELATIONSHIP` |
| `deep_hf_2019_public_data_listing` | INSUFFICIENT_SEQUENCE_INFORMATION | NO_VERIFIED_SEQUENCE_TABLE | INSUFFICIENT_SEQUENCE_INFORMATION | INSUFFICIENT_SEQUENCE_INFORMATION | 0 | 0 | `INSUFFICIENT_PROVENANCE` |
| `doench_2016_orcs_publication_screens` | VERIFIED_OBSERVED_SEQUENCE_OR_GUIDE_FIELD | ['GUIDE_ONLY'] | NO_CANONICAL_30MER_GEOMETRY | 2333 | 37080 | 1516 | `SHARED_SOURCE_STUDY` |
| `sgdesigner_2020_public_data_listing` | INSUFFICIENT_SEQUENCE_INFORMATION | NO_VERIFIED_SEQUENCE_TABLE | INSUFFICIENT_SEQUENCE_INFORMATION | INSUFFICIENT_SEQUENCE_INFORMATION | 0 | 0 | `INSUFFICIENT_PROVENANCE` |

## 3. CRISPRpred File-by-File Results

- `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`: records 4239; guide overlap 32; duplicate sequence excess 0; label-conflict sequences 0.
- `data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`: records 2333; guide overlap 1899; duplicate sequence excess 0; label-conflict sequences 0.
- `data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`: records 8101; guide overlap 53; duplicate sequence excess 0; label-conflict sequences 0.
- `data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`: records 2076; guide overlap 16; duplicate sequence excess 0; label-conflict sequences 0.

## 4. Doench Workbook-by-Workbook Results

- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`: records 17586; guide overlap 1115; duplicate sequence excess 703; label-conflict sequences 0.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`: records 27830; guide overlap 177; duplicate sequence excess 2833; label-conflict sequences 0.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`: records 69596; guide overlap 1155; duplicate sequence excess 28131; label-conflict sequences 0.
- `data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`: records 79641; guide overlap 45; duplicate sequence excess 331; label-conflict sequences 296.

## 5. Candidate-to-Candidate Overlap

- `corsi_2022_free_energy_pam_context_vs_crispron_2021_rth_tools`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `corsi_2022_free_energy_pam_context_vs_crisprpredseq_2020_bmc_additional_files`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `corsi_2022_free_energy_pam_context_vs_deep_hf_2019_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `corsi_2022_free_energy_pam_context_vs_doench_2016_orcs_publication_screens`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `corsi_2022_free_energy_pam_context_vs_sgdesigner_2020_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crispron_2021_rth_tools_vs_crisprpredseq_2020_bmc_additional_files`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crispron_2021_rth_tools_vs_deep_hf_2019_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crispron_2021_rth_tools_vs_doench_2016_orcs_publication_screens`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crispron_2021_rth_tools_vs_sgdesigner_2020_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crisprpredseq_2020_bmc_additional_files_vs_deep_hf_2019_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `crisprpredseq_2020_bmc_additional_files_vs_doench_2016_orcs_publication_screens`: guide overlap 1543; provenance `INDEPENDENCE_UNRESOLVED`.
- `crisprpredseq_2020_bmc_additional_files_vs_sgdesigner_2020_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `deep_hf_2019_public_data_listing_vs_doench_2016_orcs_publication_screens`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `deep_hf_2019_public_data_listing_vs_sgdesigner_2020_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.
- `doench_2016_orcs_publication_screens_vs_sgdesigner_2020_public_data_listing`: guide overlap INSUFFICIENT_SEQUENCE_INFORMATION; provenance `INSUFFICIENT_PROVENANCE`.

## 6. Interpretation Safeguards

- Exact overlap was not called leakage automatically.
- Zero overlap was not treated as proof of independence.
- No near-duplicate analysis was performed.
- Reverse-complement matching was not performed without verified orientation.

## 7. Final Phase 17D Status

Independence/overlap audit complete; acceptance remains deferred.
