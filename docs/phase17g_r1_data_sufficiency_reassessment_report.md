# Phase 17G-R1 - Data Sufficiency and Readiness Re-Evaluation

## Scope

- Original Phase 17G artifacts were not overwritten.
- Phase 18 not started.
- No model training, evaluation, calibration, or tuning.
- Moreno remains locked and was not accessed.

## Basis

- Original Phase 17G decision: NO_GO
- New evidence phases: 17H, 17I
- CRISPRon SHA-256: 146f211575179fc31669dbd5eaef46434b0866013a2d03767354375a1cc5d61f
- SHA-256 matches expected: True

## CRISPRon Xiang/Luo Gates

- Provenance: PASS
- Task compatibility: PASS
- Sequence usability: PASS
- Label classification: PRIMARY_CONTINUOUS
- Label status: CONTINUOUS_USABLE
- Independence: PASS
- Quantity: RECOMMENDED_QUANTITY_MET
- Guide diversity: HIGH_GUIDE_DIVERSITY (1.0000)
- Information value: HIGH
- Final candidate status: ACCEPTED_READY_CANDIDATE

Evidence chain: primary publication -> official CRISPRon/RTH resource -> recovered workbook -> explicit Dataset component -> 30mer_gRNA -> HEK293T_indel_freq_avg_d8_d10.

## Xiang/Luo Sequence And Label Audit

- raw_n: 10592
- valid_sequence_n: 10592
- unique_30mer_n: 10592
- unique_23mer_n: 10592
- unique_spacer20_n: 10592
- malformed_sequence_n: 0
- ambiguous_base_n: 0
- DeepSpCas9 exact 30-mer overlap: 48
- New unique 30-mers: 10544
- DeepSpCas9 spacer overlap: 48
- New unique spacers: 10544
- label n / finite / missing: 10592 / 10592 / 0
- label min / median / max: 0.0380228136882129 / 35.13975115371885 / 95.8512582144575
- label mean / SD / IQR: 37.16010516315937 / 22.993814153002077 / 38.11274991760083
- label unique values: 10592

## Corsi Handling

- Unique 30-mers: 1022
- Unique 20-mer spacers: 4
- Guide diversity: LOW_GUIDE_DIVERSITY (0.0039)
- Final status: AUXILIARY_ONLY
- Exclusion reason: AUXILIARY_ONLY due low spacer diversity and Dox condition label conflicts.

Corsi is not counted toward the accepted training-data total because its 30-mer count is dominated by PAM/context permutations over only 4 spacers and Dox-/Dox+ labels conflict for identical sequences.

## CRISPRon Kim Handling

- Unique 30-mers: 13359
- DeepSpCas9 exact overlap: 12832
- Overlap fraction: 0.960550939441575
- Final status: ALREADY_REPRESENTED
- Exclusion reason: Kim component overlaps the canonical DeepSpCas9 domain and is not counted as new data.

## Accepted-Data Accounting

| candidate | raw_n | unique_30mer_n | unique_spacer20_n | DeepSpCas9_overlap_n | new_unique_n | accepted_for_future_pipeline | accepted_new_unique_n | exclusion_reason |
|---|---:|---:|---:|---:|---:|---|---:|---|
| crispron_xiang_luo | 10592 | 10592 | 10592 | 48 | 10544 | True | 10544 |  |
| crispron_kim | 13359 | 13359 | 13352 | 12832 | 527 | False | 0 | Kim component overlaps the canonical DeepSpCas9 domain and is not counted as new data. |
| corsi_2022 | 2010 | 1022 | 4 | 0 | 0 | False | 0 | AUXILIARY_ONLY due low spacer diversity and Dox condition label conflicts. |

## Final Decision

- accepted_candidate_count: 1
- total_new_unique_compatible_observations: 10544
- minimum_required: 2000
- surplus_or_deficit: 8544
- final_decision: GO

Decision reasons:
- CRISPRon Xiang/Luo passes provenance, task, sequence, label, independence, diversity, quantity, and information-value gates under existing Phase 17G thresholds.

## Canonical Integrity

- unchanged: True

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
?? docs/phase17g_r1_data_sufficiency_reassessment_report.md
?? docs/phase17h_primary_dataset_acquisition_report.md
?? docs/phase17i_biological_independence_crispron_recovery_report.md
?? results/phase17g_data_sufficiency_20260911_222112.json
?? results/phase17g_r1_data_sufficiency_reassessment_20260911_233322.json
?? results/phase17h_primary_dataset_acquisition_20260911_223854.json
?? results/phase17i_biological_independence_crispron_recovery_20260911_232332.json
?? results/phase17i_biological_independence_crispron_recovery_20260911_232536.json
?? results/phase17i_biological_independence_crispron_recovery_20260911_232634.json
?? scripts/run_phase17g_data_sufficiency_audit.py
?? scripts/run_phase17g_r1_data_sufficiency_reassessment.py
?? scripts/run_phase17h_primary_dataset_acquisition.py
?? scripts/run_phase17i_biological_independence_crispron_recovery.py
?? src/dataset_recovery/biological_independence_crispron.py
?? src/dataset_recovery/data_sufficiency.py
?? src/dataset_recovery/data_sufficiency_reassessment.py
?? src/dataset_recovery/primary_acquisition.py
?? src/dataset_recovery/provenance_chain.py
?? tests/test_phase17g_data_sufficiency.py
?? tests/test_phase17g_r1_data_sufficiency_reassessment.py
?? tests/test_phase17h_primary_dataset_acquisition.py
?? tests/test_phase17i_biological_independence_crispron.py
```
