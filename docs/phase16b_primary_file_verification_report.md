# Phase 16B Primary File Verification

**Status:** PRIMARY_FILE_VERIFICATION_RECORDED
**Scope:** Primary-file verification only.

No label compatibility decision, sequence transformation, overlap analysis, distribution analysis, model training, or model evaluation was performed. Moreno raw data was not accessed.

## Candidate Status

| candidate_id | primary_file_status | status_after_16b | files verified |
|---|---|---|---:|
| doench_2016_orcs_publication_screens | PRIMARY_SUPPLEMENTARY_FILES_OBTAINED | PENDING_PRIMARY_VERIFICATION | 4 |
| crisprpredseq_2020_bmc_additional_files | OFFICIAL_SUPPLEMENTARY_FILES_OBTAINED | PENDING_PRIMARY_VERIFICATION | 4 |
| crispron_2021_rth_tools | SOURCE_ARCHIVE_OBTAINED_PRIMARY_EXPERIMENTAL_FILE_UNRESOLVED | INSUFFICIENT_PROVENANCE | 1 |
| deep_hf_2019_public_data_listing | PRIMARY_FILE_NOT_OBTAINED_IN_16B_PRIOR_EVIDENCE_EXISTS | PENDING_PRIMARY_VERIFICATION | 0 |
| sgdesigner_2020_public_data_listing | INSUFFICIENT_PRIMARY_SOURCE_VERIFICATION | INSUFFICIENT_PROVENANCE | 0 |

## Verified Files

### doench_2016_orcs_publication_screens

Multiple supplementary screen tables from the same original Doench 2016 study; independence from canonical data is not assessed in Phase 16B.

- `data/phase16/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`; sha256 `3FA01C4ADAD9DD624B67D5EACB982FC8F7448461E6A2FEBE8FAEFE5E7EEFE7B8`; source: https://orcs.thebiogrid.org/uploads/processed/5aeb49259de3c/STable_06_Vem_STARSOutputs.xlsx
- `data/phase16/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`; sha256 `88FF9066C2F6F93CF2917A83C8BFD10A26378AF60853358DDA3442ADE24C9030`; source: https://orcs.thebiogrid.org/uploads/processed/5e4d50e13c7d6/STable%2009%20Sel_STARSOutput.xlsx
- `data/phase16/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`; sha256 `2A7018E31F9E6ABD2AC5355B7D8C900C9F979E133DB22F638197EE908FD7F399`; source: https://orcs.thebiogrid.org/uploads/processed/5af0c639e9d96/STable%2012%20NegativeSelection_individual_STARS.xlsx
- `data/phase16/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`; sha256 `9D00508EA5F8AF2AC809B0646497EF7A56E484AA4E3B99D779E89CDEA21EEA58`; source: https://orcs.thebiogrid.org/uploads/processed/5afb153562639/STable_15_IFNg_data_STARS.xlsx

Unresolved:
- Row-level label semantics not assessed.
- Sequence geometry not assessed.
- Subset/overlap relationship not assessed.

### crisprpredseq_2020_bmc_additional_files

Article reports HCT116, HEK293, HeLa, and HL60 files as data used by previous DeepCRISPR/Haeussler work; independent identity cannot be assumed.

- `data/phase16/raw/crisprpredseq_additional_file_1.csv`; sha256 `F6DA050009616E5BB50A8E66E9429830E89B59E607D0C740CEBC364066F34A93`; source: https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM1_ESM.csv
- `data/phase16/raw/crisprpredseq_additional_file_2.csv`; sha256 `6DBD8C62A9E1E93D2A6B265DE13E5EA6B287E9C605186223611A2BE8C7968357`; source: https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM2_ESM.csv
- `data/phase16/raw/crisprpredseq_additional_file_3.csv`; sha256 `466813EFDAF0DCCA7CBD6C4690A99BE9CA47593671174F3D72D87FEF669E204D`; source: https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM3_ESM.csv
- `data/phase16/raw/crisprpredseq_additional_file_4.csv`; sha256 `9B19CA010304B15AAB9F2E58962D00B811B0C6C368B2E6B2847DA8DC65A00A47`; source: https://static-content.springer.com/esm/art%3A10.1186%2Fs12859-020-3531-9/MediaObjects/12859_2020_3531_MOESM4_ESM.csv

Unresolved:
- Whether files are primary experimental data or processed reuse remains unresolved.
- Original upstream source relationship requires verification.
- Label and sequence definitions deferred to Phase 16C.

### crispron_2021_rth_tools

Official repository archive was obtained, but the primary experimental sgRNA training table was not located from this archive during Phase 16B.

- `data/phase16/raw/crispron_main.zip`; sha256 `3FA17A7CB3DE043703EC3090505C96340E6DA13F23ACDA8FD7F7268A3F572A9B`; source: https://github.com/RTH-tools/crispron/archive/refs/heads/main.zip

Unresolved:
- Actual primary experimental data filename unresolved.
- Repository archive appears to contain model artifacts/test output rather than raw study table.

### deep_hf_2019_public_data_listing

DeepHF was already analyzed in Phases 10-12 and must not be counted as a new independent dataset merely because it was rediscovered. Phase 16B did not restore the primary DeepHF file in data/phase16/raw.

- No local primary file verified in Phase 16B.

Unresolved:
- Primary file unavailable in current Phase 16B acquisition.
- Role requires supervisor decision because DeepHF is prior evidence, not new discovery.

### sgdesigner_2020_public_data_listing

Only catalog-level source-link evidence was retained in Phase 16B; no official primary file was obtained.

- No local primary file verified in Phase 16B.

Unresolved:
- Primary repository URL and original data file unresolved.
- Publication identity requires primary-source confirmation.

## Locks

- No Moreno raw data accessed.
- No model training/evaluation performed.
- No canonical Phase 3-15 artifacts modified.
