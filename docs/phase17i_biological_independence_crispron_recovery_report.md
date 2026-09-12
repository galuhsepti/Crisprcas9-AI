# Phase 17I - Biological Independence and CRISPRon Recovery

## Scope

- Phase 18 not started.
- No model training.
- Moreno not accessed.
- Phase 17G not rerun.
- No label harmonization or condition pooling.

## Corsi Biological-Independence Audit

- Raw observations: 2010
- Unique 30-mers: 1022
- Unique 23-mer target+PAM sequences: 256
- Unique 20-mer spacers: 4
- Unique spacer-condition pairs: 8
- Effective guide diversity ratio: 0.0039
- Shared 20-mer spacers between Dox- and Dox+: 4
- Identical 30-mers shared between Dox- and Dox+: 966
- Explicit gene/locus count from gRNA_ID: 4

The approximately 1022 unique 30-mers are primarily PAM/context variants derived from a much smaller set of spacer sequences. They should not be treated as 1022 biologically independent sgRNA observations merely because the full 30-mer differs.

## Corsi Condition Conflict Audit

- Shared sequence count: 966
- Shared sequences with label differences: 939
- Median absolute label difference: 1.029335
- Mean absolute label difference: 8.967720
- Pooling ambiguity for sequence-only model: True
- CORSI_MAIN_TRAINING_SUITABILITY: AUXILIARY_ONLY

Sequence-only pooling would assign different labels to identical 30-mers across Dox conditions, and the 30-mer count substantially overstates independent spacer diversity.

## CRISPRon Recovery

- Retrieval status: RECOVERED
- Source URL: https://rth.dk/resources/crispr/crispron/downloads//Luo2020_Kim2019.xlsx
- File size: 1544158
- SHA-256: 146f211575179fc31669dbd5eaef46434b0866013a2d03767354375a1cc5d61f
- HTTP/content type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
- Publication relationship: Official CRISPRon/RTH training-data download associated with Xiang et al. 2021 Nature Communications 12, 3238.

## CRISPRon Workbook Inventory

- `README_first`: 0 rows; columns: 
- `Luo2020_Kim2019`: 23902 rows; columns: gRNA+PAM, Dataset, Gene, Transcript, HEK293T_indel_freq_avg_d8_d10, HEK293T_indel_freq_d8, HEK293T_indel_freq_d10, 30mer_gRNA, Indel_freq_HEK293T

## CRISPRon Source Separation

- Source identity recoverable: True
- Xiang/Luo raw rows: 10592
- Xiang/Luo unique 30-mers: 10592
- Xiang/Luo unique 20-mers: 10592
- Kim raw rows: 13359
- Kim unique 30-mers: 13359

## DeepSpCas9 Overlap

- All recoverable raw rows: 23902
- Unique 30-mers: 23902
- Exact DeepSpCas9 overlap: 12832
- Exact overlap fraction: 0.536858840264413
- New unique 30-mers: 11070
- Unique 20-mer spacers: 23895
- Spacer overlap with DeepSpCas9: 12825
- New unique 20-mer spacers: 11070

## CRISPRon Label Audit

- label_name: HEK293T_indel_freq_avg_d8_d10
- label_definition: Workbook-native SpCas9 on-target indel-frequency/efficiency label; inspected without rescaling.
- experimental_measurement: Experimentally measured HEK293T indel frequency for the Luo/Xiang component; Kim rows are integrated published data and remain source-separated.
- scale: percent-like numeric scale in the workbook
- minimum: 0.0380228136882129
- maximum: 95.8512582144575
- mean: 37.16010516315937
- median: 35.13975115371885
- SD: 22.993814153002077
- IQR: 38.11274991760083
- number_of_unique_values: 10592
- missing_values: 13310
- classification: PRIMARY_CONTINUOUS
- derivation_assessment: Quant_norm_efficiency is not present in the recovered workbook. The available average day-8/day-10 indel-frequency field is treated as a primary continuous measurement for audit purposes only.

## Preliminary CRISPRon Suitability

CRISPRON_PRELIMINARY_STATUS: PROMISING_FOR_17G

This is not acceptance. Acceptance can only occur in a later Phase 17G re-evaluation.

## Canonical Integrity

- Canonical integrity unchanged: True
- Canonical data not modified.

## Git

- HEAD: a7a7e8298d601d61c58ecab4c2a824a56f1f5dd5
```
config.yaml | 8 ++++++++
 1 file changed, 8 insertions(+)
```
```
M config.yaml
?? data/phase17h/
?? data/phase17i/
?? docs/phase17g_data_sufficiency_report.md
?? docs/phase17h_primary_dataset_acquisition_report.md
?? docs/phase17i_biological_independence_crispron_recovery_report.md
?? results/phase17g_data_sufficiency_20260911_222112.json
?? results/phase17h_primary_dataset_acquisition_20260911_223854.json
?? results/phase17i_biological_independence_crispron_recovery_20260911_232332.json
?? results/phase17i_biological_independence_crispron_recovery_20260911_232536.json
?? scripts/run_phase17g_data_sufficiency_audit.py
?? scripts/run_phase17h_primary_dataset_acquisition.py
?? scripts/run_phase17i_biological_independence_crispron_recovery.py
?? src/dataset_recovery/biological_independence_crispron.py
?? src/dataset_recovery/data_sufficiency.py
?? src/dataset_recovery/primary_acquisition.py
?? src/dataset_recovery/provenance_chain.py
?? tests/test_phase17g_data_sufficiency.py
?? tests/test_phase17h_primary_dataset_acquisition.py
?? tests/test_phase17i_biological_independence_crispron.py
```
