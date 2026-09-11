# Phase 17B - Primary-File Verification Audit

## 1. Objective

Phase 17B verifies primary-file identity, provenance, raw sequence structure, and raw label structure for Phase 17A candidates. It does not make final dataset acceptance decisions.

No modeling, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, overlap analysis, reverse-complement matching, continual learning, or dataset acceptance was performed.

## 2. Candidate Verification Table

| Candidate | Publication | Repository/Accession | Local file(s) | Primary/derived/processed status | Verification status | Unresolved issues |
|---|---|---|---|---|---|---|
| `crispron_2021_rth_tools` | Xu F et al. 2021. Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning. | RTH-tools/crispron GitHub repository; NOT_APPLICABLE | `data/phase17/raw/crispron_2021_rth_tools_main.zip`<br>`3fa17a7cb3de043703ec3090505c96340e6da13f23acda8fd7f7268a3f572a9b` | PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED | `PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED` | PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED, label_semantics, pam_definition, processed_or_derived_status, sequence_geometry, sequence_orientation |
| `crisprpredseq_2020_bmc_additional_files` | Rafid AHM et al. 2020. CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction using traditional machine learning. | BMC Bioinformatics supplementary files; NOT_APPLICABLE | `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`<br>`f6da050009616e5bb50a8e66e9429830e89b59e607d0c740cebc364066f34a93`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`<br>`6dbd8c62a9e1e93d2a6b265de13e5ea6b287e9c605186223611a2be8c7968357`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`<br>`466813efdaf0dcca7cbd6c4690a99be9ca47593671174f3d72d87fef669e204d`<br>`data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`<br>`9b19ca010304b15aab9f2e58962d00b811b0c6c368b2e6b2847da8dc65a00a47` | author-provided processed / binary classification supplementary dataset | `DERIVED_OR_PROCESSED` | label_semantics, pam_definition, processed_or_derived_status, processed_or_partitioned_source_relationship_requires_later_gate, sequence_geometry, sequence_orientation |
| `doench_2016_orcs_publication_screens` | Doench JG et al. 2016. Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. | BioGRID ORCS Dataset 19 / publication supplementary files; BioGRID ORCS Dataset 19 | `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`<br>`3fa01c4adad9dd624b67d5eacb982fc8f7448461e6a2febe8faefe5e7eefe7b8`<br>`data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`<br>`88ff9066c2f6f93cf2917a83c8bfd10a26378af60853358dda3442ade24c9030`<br>`data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`<br>`2a7018e31f9e6abd2ac5355b7d8c900c9f979e133db22f638197ee908fd7f399`<br>`data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`<br>`9d00508ea5f8af2ac809b0646497ef7a56e484aa4e3b99d779e89cdea21eea58` | publication supplementary screen/workbook outputs with metadata gaps | `PRIMARY_VERIFIED_WITH_METADATA_GAPS` | label_semantics, pam_definition, processed_or_derived_status, sequence_geometry, sequence_orientation, study_independence |
| `deep_hf_2019_public_data_listing` | Wang D et al. 2019. Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning. | Nature Communications supplementary data / DeepHF web server; NOT_APPLICABLE | not recovered | PRIMARY_STATUS_UNRESOLVED | `PRIMARY_FILE_NOT_RECOVERED` | label_semantics, pam_definition, primary_file_not_recovered_locally, processed_or_derived_status, sequence_geometry, sequence_orientation |
| `sgdesigner_2020_public_data_listing` | SgDesigner / unique plasmid library sgRNA potency data; primary publication identity unresolved in project records | public_data_crisprCas9 catalogue only; UNKNOWN | not recovered | PRIMARY_STATUS_UNRESOLVED | `PRIMARY_FILE_NOT_RECOVERED` | label_semantics, pam_definition, primary_file_not_recovered_locally, processed_or_derived_status, sequence_geometry, sequence_orientation |
| `corsi_2022_free_energy_pam_context` | Corsi GI et al. 2022. CRISPR/Cas9 gRNA activity depends on free energy changes and on the target PAM context. | Nature Communications supplementary data; NOT_APPLICABLE | not recovered | DISCOVERY_ONLY | `DISCOVERY_ONLY` | label_semantics, pam_definition, processed_or_derived_status, sequence_geometry, sequence_orientation, study_independence |

## 3. CRISPRon Archive Analysis

- Archive members: 53
- Documentation/code files observed: 4 documentation/license, 5 code files.
- Data-like files observed: ['crispron-main/test/outdir.original/CRISPRparams.tsv', 'crispron-main/test/outdir.original/crispron.csv']
- Model-artifact-like files observed: 30
- Experimental training table found: `False`
- Preserved status: `PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED`

## 4. CRISPRpred File-by-File Analysis

### `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`
- Rows: 4239
- Columns: `['sgRNA', 'label']`
- Sequence columns: `['sgRNA']`
- Label columns: `['label']`
- File interpretation: processed/partitioned binary-label supplementary CSV; no merge or label conversion performed.

### `data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`
- Rows: 2333
- Columns: `['sgRNA', 'label']`
- Sequence columns: `['sgRNA']`
- Label columns: `['label']`
- File interpretation: processed/partitioned binary-label supplementary CSV; no merge or label conversion performed.

### `data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`
- Rows: 8101
- Columns: `['sgRNA', 'label']`
- Sequence columns: `['sgRNA']`
- Label columns: `['label']`
- File interpretation: processed/partitioned binary-label supplementary CSV; no merge or label conversion performed.

### `data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`
- Rows: 2076
- Columns: `['sgRNA', 'label']`
- Sequence columns: `['sgRNA']`
- Label columns: `['label']`
- File interpretation: processed/partitioned binary-label supplementary CSV; no merge or label conversion performed.

## 5. Doench Workbook-by-Workbook Analysis

### `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx`
- Sheet count: 5
- Sheet names: `['STARS Vem Avana lentiCRISPRv2', 'STARS Vem Avana lentiGuide', 'STARS Vem GeCKOv1 lentiCRISPRv1', 'STARS Vem GeCKOv1 lentiGuide', 'STARS Vem GeCKOv2 lentiGuide']`
- File interpretation: publication/ORCS screen workbook; screen-derived statistics are not transformed into activity.

### `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx`
- Sheet count: 2
- Sheet names: `['Avana Sel lentiGuide', 'Avana Sel lentiCRISPRv2']`
- File interpretation: publication/ORCS screen workbook; screen-derived statistics are not transformed into activity.

### `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx`
- Sheet count: 5
- Sheet names: `['Avana6_lentiCRISPRv2_A375', 'Avana6_lentiGuide_A375', 'Avana4_lentiGuide_HT29', 'GeCKOv2_lentiGuide_A375', 'GeCKOv2_lentiGuide_HT29']`
- File interpretation: publication/ORCS screen workbook; screen-derived statistics are not transformed into activity.

### `data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx`
- Sheet count: 2
- Sheet names: `['Data', 'STARS analysis']`
- File interpretation: publication/ORCS screen workbook; screen-derived statistics are not transformed into activity.

## 6. DeepHF / SgDesigner / Corsi Recovery Status

- `deep_hf_2019_public_data_listing`: `PRIMARY_FILE_NOT_RECOVERED`; PRIMARY_STATUS_UNRESOLVED.
- `sgdesigner_2020_public_data_listing`: `PRIMARY_FILE_NOT_RECOVERED`; PRIMARY_STATUS_UNRESOLVED.
- `corsi_2022_free_energy_pam_context`: `DISCOVERY_ONLY`; DISCOVERY_ONLY.

## 7. Canonical Integrity

Canonical protected state unchanged: `True`.

## 8. Final Phase 17B Status

Primary-file verification is complete for recovered files; acceptance remains deferred.
