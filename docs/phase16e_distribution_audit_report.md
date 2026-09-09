# Phase 16E Distribution / Composition Audit

**Status:** DISTRIBUTION_AUDIT_RECORDED
**Scope:** Descriptive distribution and composition audit only.

No modeling, label transformation, pooling, overlap reinterpretation, gate decision, or Moreno access was performed.

## Dataset Counts

| dataset | sequence records | compatibility | activity analyzed |
|---|---:|---|---|
| DeepSpCas9 | 12832 | canonical reference | yes |
| crispron_2021_rth_tools | 0 | INSUFFICIENT_PROVENANCE | no |
| crisprpredseq_2020_bmc_additional_files | 16749 | INCOMPATIBLE | no |
| deep_hf_2019_public_data_listing | 0 | INSUFFICIENT_PROVENANCE | no |
| doench_2016_orcs_publication_screens | 194653 | INCOMPATIBLE | no |
| sgdesigner_2020_public_data_listing | 0 | INSUFFICIENT_PROVENANCE | no |

## Sequence GC Summary

| dataset | n | mean | median | sd |
|---|---:|---:|---:|---:|
| DeepSpCas9 | 12832 | 0.5542 | 0.5667 | 0.1319 |
| crispron_2021_rth_tools | 0 | n/a | n/a | n/a |
| crisprpredseq_2020_bmc_additional_files | 16749 | 0.5801 | 0.5652 | 0.0840 |
| deep_hf_2019_public_data_listing | 0 | n/a | n/a | n/a |
| doench_2016_orcs_publication_screens | 194653 | 0.5218 | 0.5000 | 0.1163 |
| sgdesigner_2020_public_data_listing | 0 | n/a | n/a | n/a |

## Activity / Label Distribution

- DeepSpCas9 activity was analyzed as the canonical established continuous label.
- Candidate labels were not analyzed as activity because Phase 16C did not establish target-compatible continuous activity.

## Phase 16D Context Carried Forward

- `crispron_2021_rth_tools` RC: `RC_OVERLAP_NOT_ASSESSED`; independence: `INSUFFICIENT_SEQUENCE_INFORMATION`
- `crisprpredseq_2020_bmc_additional_files` RC: `RC_OVERLAP_NOT_ASSESSED`; independence: `NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK`
- `deep_hf_2019_public_data_listing` RC: `RC_OVERLAP_NOT_ASSESSED`; independence: `INSUFFICIENT_SEQUENCE_INFORMATION`
- `doench_2016_orcs_publication_screens` RC: `RC_OVERLAP_NOT_ASSESSED`; independence: `NOT_ESTABLISHED`
- `sgdesigner_2020_public_data_listing` RC: `RC_OVERLAP_NOT_ASSESSED`; independence: `INSUFFICIENT_SEQUENCE_INFORMATION`

## Figures

- gc_distribution: `results\figures\phase16e_gc_distribution_20260908_170415.png`

## Locks

- No Moreno raw data accessed.
- No model training/evaluation performed.
- No dataset pooling performed.
- No Phase 16F gate decision performed.
- No canonical Phase 3-15 artifacts modified.
