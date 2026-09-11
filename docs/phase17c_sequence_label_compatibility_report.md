# Phase 17C - Sequence and Label Compatibility Audit

## 1. Objective

Phase 17C audits whether recovered Phase 17 candidates have observed sequence structure and documented label semantics compatible with the thesis target. It does not accept datasets or create training data.

No model training, model evaluation, tuning, Moreno access, pooling, label transformation, sequence transformation, 30-mer construction, reverse-complement matching, near-duplicate analysis, continual learning, or dataset acceptance was performed.

## 2. Candidate-Level Compatibility Table

| Candidate | Sequence status | Observed sequence geometry | Label status | Label semantics | Overall status |
|---|---|---|---|---|---|
| `crispron_2021_rth_tools` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | ARCHIVE_NO_VERIFIED_EXPERIMENTAL_SEQUENCE_TABLE | `LABEL_INSUFFICIENT_EVIDENCE` | CRISPRon archive does not identify a primary experimental activity label table. | `INSUFFICIENT_EVIDENCE` |
| `crisprpredseq_2020_bmc_additional_files` | `SEQUENCE_INCOMPATIBLE` | GUIDE_PLUS_PAM_23MER_OR_UNVERIFIED_23MER | `LABEL_INCOMPATIBLE` | Binary classification label is not a continuous experimentally measured activity label. | `INCOMPATIBLE_FOR_CURRENT_TARGET` |
| `doench_2016_orcs_publication_screens` | `SEQUENCE_INCOMPATIBLE` | GUIDE_ONLY_OR_SHORT_GUIDE; NONCANONICAL_OR_MIXED_LENGTH | `LABEL_INCOMPATIBLE` | Rank/enrichment/log-fold-change/screen statistics are not canonical continuous activity labels. | `INCOMPATIBLE_FOR_CURRENT_TARGET` |
| `deep_hf_2019_public_data_listing` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_SEQUENCE_EVIDENCE | `LABEL_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_LABEL_EVIDENCE | `INSUFFICIENT_EVIDENCE` |
| `sgdesigner_2020_public_data_listing` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_SEQUENCE_EVIDENCE | `LABEL_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_LABEL_EVIDENCE | `INSUFFICIENT_EVIDENCE` |
| `corsi_2022_free_energy_pam_context` | `SEQUENCE_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_SEQUENCE_EVIDENCE | `LABEL_INSUFFICIENT_EVIDENCE` | NO_RECOVERED_LABEL_EVIDENCE | `INSUFFICIENT_EVIDENCE` |

## 3. CRISPRpred File-by-File Results

- `data/phase17/raw/crisprpredseq_2020_additional_file_1.csv`: sequence `SEQUENCE_INCOMPATIBLE`; lengths `{23: 4239}`; labels `LABEL_INCOMPATIBLE`.
- `data/phase17/raw/crisprpredseq_2020_additional_file_2.csv`: sequence `SEQUENCE_INCOMPATIBLE`; lengths `{23: 2333}`; labels `LABEL_INCOMPATIBLE`.
- `data/phase17/raw/crisprpredseq_2020_additional_file_3.csv`: sequence `SEQUENCE_INCOMPATIBLE`; lengths `{23: 8101}`; labels `LABEL_INCOMPATIBLE`.
- `data/phase17/raw/crisprpredseq_2020_additional_file_4.csv`: sequence `SEQUENCE_INCOMPATIBLE`; lengths `{23: 2076}`; labels `LABEL_INCOMPATIBLE`.

## 4. Doench Workbook-by-Workbook Results

- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv1 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv1 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv1 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv1 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv2 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv2 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv2 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_06_Vem_STARSOutputs.xlsx` / sheet `STARS Vem GeCKOv2 lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiGuide`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiCRISPRv2`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiCRISPRv2`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiCRISPRv2`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_09_Sel_STARSOutput.xlsx` / sheet `Avana Sel lentiCRISPRv2`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiCRISPRv2_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiCRISPRv2_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiCRISPRv2_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiCRISPRv2_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana6_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana4_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana4_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana4_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `Avana4_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_A375`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_12_NegativeSelection_individual_STARS.xlsx` / sheet `GeCKOv2_lentiGuide_HT29`: sequence `SEQUENCE_INCOMPATIBLE`; geometry NONCANONICAL_OR_MIXED_LENGTH; no transformation performed.
- `data/phase17/raw/doench2016_orcs_STable_15_IFNg_data_STARS.xlsx` / sheet `Data`: sequence `SEQUENCE_INCOMPATIBLE`; geometry GUIDE_ONLY_OR_SHORT_GUIDE; no transformation performed.

## 5. CRISPRon Results

Verified experimental activity table exists: `False`.
Sequence status: `SEQUENCE_INSUFFICIENT_EVIDENCE`.
Label status: `LABEL_INSUFFICIENT_EVIDENCE`.

## 6. Unrecovered Candidates

- `deep_hf_2019_public_data_listing`: sequence `SEQUENCE_INSUFFICIENT_EVIDENCE`, label `LABEL_INSUFFICIENT_EVIDENCE`, overall `INSUFFICIENT_EVIDENCE`.
- `sgdesigner_2020_public_data_listing`: sequence `SEQUENCE_INSUFFICIENT_EVIDENCE`, label `LABEL_INSUFFICIENT_EVIDENCE`, overall `INSUFFICIENT_EVIDENCE`.
- `corsi_2022_free_energy_pam_context`: sequence `SEQUENCE_INSUFFICIENT_EVIDENCE`, label `LABEL_INSUFFICIENT_EVIDENCE`, overall `INSUFFICIENT_EVIDENCE`.

## 7. Safety Checks

- Numeric ranges were not treated as activity without source semantics.
- Binary labels were not converted into continuous labels.
- Guide-only or 23-mer sequences were not converted into canonical 30-mers.
- PAM/flanking sequence was not invented.
- Rank/enrichment/log-fold-change fields were not treated as activity.
- Processed benchmark files were not treated as independent primary experiments.

## 8. Final Phase 17C Status

Sequence/label compatibility audit complete; acceptance remains deferred.
