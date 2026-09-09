# Phase 16D Exact Overlap / Independence Audit

**Status:** OVERLAP_AUDIT_RECORDED
**Scope:** Exact overlap and provenance independence only.

No label rescue, sequence transformation, dataset pooling, distribution analysis, model training, or model evaluation was performed. Moreno raw data was not accessed.

## Canonical Reference

- DeepSpCas9 rows: 12832
- Unique 30-mers: 12832
- Unique guides: 12825

## Candidate Results

| candidate_id | sequence_available | records | 30mer overlap | guide overlap | duplicate excess | label conflicts | RC status | independence |
|---|---:|---:|---|---|---:|---:|---|---|
| crispron_2021_rth_tools | False | 0 | NO_30MER_SEQUENCE_AVAILABLE | NO_GUIDE_SEQUENCE_AVAILABLE | 0 | 0 | RC_OVERLAP_NOT_ASSESSED | INSUFFICIENT_SEQUENCE_INFORMATION |
| crisprpredseq_2020_bmc_additional_files | True | 16749 | NO_30MER_SEQUENCE_AVAILABLE | 1966 | 4284 | 402 | RC_OVERLAP_NOT_ASSESSED | NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK |
| deep_hf_2019_public_data_listing | False | 0 | NO_30MER_SEQUENCE_AVAILABLE | NO_GUIDE_SEQUENCE_AVAILABLE | 0 | 0 | RC_OVERLAP_NOT_ASSESSED | INSUFFICIENT_SEQUENCE_INFORMATION |
| doench_2016_orcs_publication_screens | True | 214820 | NO_30MER_SEQUENCE_AVAILABLE | 2333 | 57240 | 296 | RC_OVERLAP_NOT_ASSESSED | NOT_ESTABLISHED |
| sgdesigner_2020_public_data_listing | False | 0 | NO_30MER_SEQUENCE_AVAILABLE | NO_GUIDE_SEQUENCE_AVAILABLE | 0 | 0 | RC_OVERLAP_NOT_ASSESSED | INSUFFICIENT_SEQUENCE_INFORMATION |

## Pairwise Candidate Overlap

- `crispron_2021_rth_tools_vs_crisprpredseq_2020_bmc_additional_files`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crispron_2021_rth_tools_vs_deep_hf_2019_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crispron_2021_rth_tools_vs_doench_2016_orcs_publication_screens`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crispron_2021_rth_tools_vs_sgdesigner_2020_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crisprpredseq_2020_bmc_additional_files_vs_deep_hf_2019_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crisprpredseq_2020_bmc_additional_files_vs_doench_2016_orcs_publication_screens`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `1543`, RC `RC_OVERLAP_NOT_ASSESSED`
- `crisprpredseq_2020_bmc_additional_files_vs_sgdesigner_2020_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `deep_hf_2019_public_data_listing_vs_doench_2016_orcs_publication_screens`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `deep_hf_2019_public_data_listing_vs_sgdesigner_2020_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`
- `doench_2016_orcs_publication_screens_vs_sgdesigner_2020_public_data_listing`: exact 30-mer `NO_30MER_SEQUENCE_AVAILABLE`, exact guide `NO_GUIDE_SEQUENCE_AVAILABLE`, RC `RC_OVERLAP_NOT_ASSESSED`

## Locks

- No Moreno raw data accessed.
- No model training/evaluation performed.
- No dataset pooling performed.
- No canonical Phase 3-15 artifacts modified.
