# Phase 16A Dataset Discovery / Provenance Inventory

**Status:** DISCOVERY_INVENTORY_CREATED
**Scope:** Discovery and provenance inventory only.

No model was trained, tuned, calibrated, adapted, or evaluated. Moreno raw data was not accessed. No reverse-complement overlap or near-duplicate threshold analysis was performed.

## Discovery Sources Consulted

- BioGRID ORCS Dataset 19: https://orcs.thebiogrid.org/Dataset/19 (primary repository / publication supplementary file index)
- RTH-tools CRISPRon GitHub: https://github.com/RTH-tools/crispron (official project repository)
- CRISPRon webserver: https://rth.dk/resources/crispr/crispron/ (official project page)
- CRISPRpred(SEQ) BMC Bioinformatics article: https://link.springer.com/article/10.1186/s12859-020-3531-9 (publisher-hosted supplementary-file index)
- dagrate/public_data_crisprCas9: https://github.com/dagrate/public_data_crisprCas9 (public catalog used only to discover primary source links)

## Candidate Inventory

| candidate_id | candidate_name | source_repository | status | primary evidence |
|---|---|---|---|---|
| crispron_2021_rth_tools | CRISPRon 2021 sgRNA efficiency data | RTH-tools/crispron | PENDING_PRIMARY_VERIFICATION | UNKNOWN; repository files require 16B enumeration |
| crisprpredseq_2020_bmc_additional_files | CRISPRpred(SEQ) additional files HCT116/HEK293/HeLa/HL60 | BMC Bioinformatics supplementary additional files | PENDING_PRIMARY_VERIFICATION | Additional file 1; Additional file 2; Additional file 3; Additional file 4 |
| deep_hf_2019_public_data_listing | DeepHF on-target activity data | DeepHF website/GitHub link reported by public_data_crisprCas9 | PENDING_PRIMARY_VERIFICATION | UNKNOWN; public_data_crisprCas9 points to DeepHF website/GitHub |
| doench_2016_orcs_publication_screens | Doench 2016 published CRISPR screens | BioGRID ORCS Dataset 19 | PENDING_PRIMARY_VERIFICATION | STable_06_Vem_STARSOutputs.xlsx; STable 12 NegativeSelection_individual_STARS.xlsx; STable_15_IFNg_data_STARS.xlsx; STable 09 Sel_STARSOutput.xlsx |
| sgdesigner_2020_public_data_listing | SgDesigner / unique plasmid library sgRNA potency data | sgDesigner GitHub link reported by public_data_crisprCas9 | PENDING_PRIMARY_VERIFICATION | UNKNOWN; source GitHub data file pending |

## Provenance Questions Remaining

- `crispron_2021_rth_tools` unknown identity fields: parent_dataset
- `crisprpredseq_2020_bmc_additional_files` identity fields populated for Phase 16A discovery.
- `deep_hf_2019_public_data_listing` unknown identity fields: parent_dataset, library_design
- `doench_2016_orcs_publication_screens` unknown identity fields: parent_dataset
- `sgdesigner_2020_public_data_listing` unknown identity fields: parent_dataset, subset_id

## Explicit Limits

- Every candidate remains `PENDING_PRIMARY_VERIFICATION`.
- Dataset identity is not determined from candidate name alone.
- Literature mention alone is not treated as primary-data verification.
- Compatibility, overlap, distribution analysis, and gate decisions are deferred.
