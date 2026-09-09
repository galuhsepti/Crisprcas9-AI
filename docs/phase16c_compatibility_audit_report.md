# Phase 16C Compatibility Audit

**Status:** COMPATIBILITY_AUDIT_RECORDED
**Scope:** Biological/measurement compatibility only.

No label harmonization, sequence transformation, overlap analysis, distribution analysis, pooling, model training, or model evaluation was performed. Moreno raw data was not accessed.

## Candidate Decisions

| candidate_id | compatibility_status | reason | transformations |
|---|---|---|---|
| doench_2016_orcs_publication_screens | INCOMPATIBLE | Primary files are screen-enrichment/rank/log-fold-change outputs, not substantively compatible continuous SpCas9 editing activity. | REQUIRED_NOT_PERFORMED |
| crisprpredseq_2020_bmc_additional_files | INCOMPATIBLE | Verified files contain binary labels and 23-mer guide+PAM schema; no label harmonization or sequence transformation performed. | REQUIRED_NOT_PERFORMED |
| crispron_2021_rth_tools | INSUFFICIENT_PROVENANCE | Phase 16C evidence does not resolve this candidate beyond Phase 16B. | NONE |
| deep_hf_2019_public_data_listing | INSUFFICIENT_PROVENANCE | Phase 16C evidence does not resolve this candidate beyond Phase 16B. | NONE |
| sgdesigner_2020_public_data_listing | INSUFFICIENT_PROVENANCE | Phase 16C evidence does not resolve this candidate beyond Phase 16B. | NONE |

## File-Level Evidence

### doench_2016_orcs_publication_screens

- File: `data/phase16/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_only` length `20`
  Activity: Workbook contains STARS/rank/enrichment-style screen outputs
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_only` length `20`
  Activity: Workbook contains STARS/rank/enrichment-style screen outputs
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_only` length `20`
  Activity: Workbook contains STARS/rank/enrichment-style screen outputs
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_only` length `20`
  Activity: Workbook contains sgRNA and average log fold change IFNgamma - mock
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`

### crisprpredseq_2020_bmc_additional_files

- File: `data/phase16/raw/crisprpredseq_additional_file_1.csv`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_plus_pam` length `23`
  Activity: CSV columns sgRNA,label; observed label values are 0/1.
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/crisprpredseq_additional_file_2.csv`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_plus_pam` length `23`
  Activity: CSV columns sgRNA,label; observed label values are 0/1.
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/crisprpredseq_additional_file_3.csv`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_plus_pam` length `23`
  Activity: CSV columns sgRNA,label; observed label values are 0/1.
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`
- File: `data/phase16/raw/crisprpredseq_additional_file_4.csv`
  Status: `INCOMPATIBLE`
  Sequence schema: `guide_plus_pam` length `23`
  Activity: CSV columns sgRNA,label; observed label values are 0/1.
  Required transformation: `REQUIRED_NOT_PERFORMED`; performed: `NOT_PERFORMED`

### crispron_2021_rth_tools

- No new Phase 16C file-level evidence beyond unresolved Phase 16B status.

### deep_hf_2019_public_data_listing

- No new Phase 16C file-level evidence beyond unresolved Phase 16B status.

### sgdesigner_2020_public_data_listing

- No new Phase 16C file-level evidence beyond unresolved Phase 16B status.

## Locks

- No Moreno raw data accessed.
- No model training/evaluation performed.
- No dataset pooling performed.
- No canonical Phase 3-15 artifacts modified.
