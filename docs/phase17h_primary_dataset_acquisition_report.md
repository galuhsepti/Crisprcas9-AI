# Phase 17H - Targeted Primary Dataset Acquisition and Provenance Recovery

## 1. What Was Searched

- Phase 17 carryover + GitHub/Nature/publication search: CRISPRon, 10,592 SpCas9 gRNAs, s41467-021-23576-0
- Nature article supplementary files: Corsi 2022, Supplementary Data 1, Indel frequency, 30mer_gRNA
- Phase 17 carryover / prior reports: sgDesigner, sgRNA Designer, on-target activity
- Phase 17 carryover: CRISPRpred, binary label, BMC supplementary
- Phase 17 carryover: Doench ORCS, STARS, rank, screen

## 2. Primary Publications Investigated

- Xiang/Xu et al. 2021 CRISPRon (10.1038/s41467-021-23576-0)
- Corsi et al. 2022 PAM-context SpCas9 cleavage efficiency (10.1038/s41467-022-30515-0)
- sgDesigner / Broad sgRNA Designer lineage (NOT_ESTABLISHED)
- CRISPRpred(SEQ) 2020 (10.1186/s12859-020-3531-9)
- Doench et al. 2016 ORCS screens (PMID:26780180)

## 3. Supplementary Files Located

- `corsi_2022_pam_context_supplementary_data_1`: Supplementary Data 1 -> `data/phase17h/raw/corsi_2022_supplementary_data_1.xlsx`

## 4. Files Actually Recovered

- `data/phase17h/raw/corsi_2022_supplementary_data_1.xlsx` SHA-256 `87540fc4e22bd95dce95f39efb1b54cc3a761026ed3e2faeafbc34e3ba8549fd`

## 5. Row-Level sgRNA Sequence Evidence

- `corsi_2022_pam_context_supplementary_data_1`: sequence column `30mer_gRNA`, unique sequences 1022.

## 6. Continuous Experimental On-Target Activity

- `corsi_2022_pam_context_supplementary_data_1`: label `Indel frequency (% avg D6-D10)`; definition: continuous indel frequency percentage averaged across day 6 and day 10; no rescaling performed.

## 7. Incompatible Candidates

- Xiang/Xu et al. 2021 CRISPRon
- sgDesigner / Broad sgRNA Designer lineage
- CRISPRpred(SEQ) 2020
- Doench et al. 2016 ORCS screens

## 8. Promising Candidates for Phase 17G Re-Evaluation

- `corsi_2022_pam_context_supplementary_data_1`

## 9. Potential Compatible Unique Observations

potential_total_unique_observations = 1022

phase17g_minimum_total_new_unique_n = 2000

remaining_deficit_relative_to_2000 = 978

## 10. Realistic Path Toward 2000

Corsi Supplementary Data 1 may contribute about 1,022 unique 30 nt sequence-linked indel-frequency observations if it passes full Phase 17G re-evaluation. The project still needs at least 978 additional compatible unique observations from another verified primary source, or a larger independently compatible dataset.

## 11. Missing Evidence

- CRISPRon: official row-level primary experimental table still not recovered.
- sgDesigner: primary experimental dataset identity remains unresolved.
- Corsi: requires full Phase 17G re-evaluation, independence audit, and metadata review before any acceptance.

## 12. Canonical Protection

- Moreno untouched: True.
- Canonical models and predictions untouched: True.
- Canonical integrity unchanged: True.

## 13. Scope Confirmation

- No model training.
- No model evaluation.
- No label harmonization.
- No sequence reconstruction.
- No Phase 17G rerun.
- No Phase 18.
