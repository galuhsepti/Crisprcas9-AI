# CRISPR-Cas9 AI Thesis — Project Master Handover

## 1. Project Identity

- **Project objective:** build and audit an in-silico CRISPR-Cas9 sgRNA activity prediction pipeline from DNA sequence, with reproducible thesis-level records for data processing, model training, evaluation, interpretation, and dataset-readiness decisions.
- **Thesis-level research question:** whether sequence-based machine learning models can predict and rank SpCas9 sgRNA activity with reproducible internal performance and scientifically defensible external generalization under a locked evaluation policy.
- **Current modeling target:** continuous on-target SpCas9 activity / modification frequency on a 0-1 scale, using DeepSpCas9 as the canonical training/validation domain.
- **Intended future application:** computational screening, prioritization, or ranking of candidate sgRNAs before experimental validation.
- **Explicit scope limitations:** the project is limited to in-silico prediction/ranking from sequence representations and recorded experimental activity labels. It does not perform wet-lab validation, clinical validation, off-target safety assessment, therapeutic design approval, or causal biological inference from model outputs or attribution.
- **Experimental-success limitation:** this is an in-silico prediction/ranking system. It does not guarantee experimental editing success, biological efficacy, safety, or transferability to untested organisms, assays, cell systems, or Cas9 variants.

## 2. System Architecture

- **Canonical input representation:** validated 30-mer DNA target sequences over A/C/G/T. The CNN uses one-hot encoding with shape `(n, 30, 4)`. The tabular baselines use 197 engineered features.
- **30-mer geometry:** positions `[0:4]` are 5' context, `[4:24]` are the 20 bp guide, `[24:27]` are the PAM, and `[27:30]` are 3' context.
- **Guide/PAM definition:** guide is the 20 bp target sequence at `[4:24]`; PAM is `[24:27]` and is treated as NGG for canonical SpCas9 data.
- **Continuous activity target:** `modFreq` / activity fraction on a 0-1 scale, retained as a continuous regression target.
- **CNN primary model:** CRISPRpred-style parallel 1D convolution over the one-hot 30-mer, with kernel sizes `[5, 7, 9]`, 64 filters per branch, dense layer of 64 units, dropout 0.3, Adam optimizer, MSE loss, batch size 32, max 100 epochs, and patience 10 early stopping.
- **Random Forest baseline:** fixed Random Forest regressor using the 197 engineered feature set; canonical hyperparameters include 200 trees, max depth 20, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, seed 42.
- **XGBoost baseline:** fixed XGBoost regressor using the same 197 engineered feature set; canonical hyperparameters include 100 estimators, max depth 6, learning rate 0.1, `reg:squarederror`, seed 42, and no early stopping.
- **Evaluation metrics:** primary metrics are MAE, RMSE, R2, Pearson correlation, and Spearman correlation. Kendall tau, ranking metrics, paired tests, and bootstrap confidence intervals are used in specific audit phases. MAPE is not a primary metric because activity values near zero make it unstable.
- **Interpretability methods:** CNN gradient saliency and Integrated Gradients; Random Forest impurity-decrease feature importance; XGBoost split-gain feature importance. Attribution findings are model-behavior evidence only, not causal biology.

## 3. Canonical Dataset and Evaluation Policy

- **DeepSpCas9:** canonical training/validation dataset. Repository records describe 12,832 raw rows and 10,117 canonical-valid rows after the homopolymer/sequence validation filter.
- **Canonical valid n:** 10,117.
- **Train n:** 8,599, using an 85/15 split with fixed seed 42.
- **Validation n:** 1,518, using the same fixed split.
- **Moreno external evaluation dataset:** Moreno-Mateos external test set, 810 valid rows after canonical validation.
- **Moreno LOCKED policy:** Moreno-Mateos is held out for final external evaluation and must not be used for training, tuning, early stopping, calibration, adaptation, arm selection, dataset selection, threshold setting, or iterative model development.
- **No external fitting/calibration/adaptation:** external data must not influence fitted parameters, calibrators, weights, transformations, gates, or model-selection decisions.
- **CNN validation early stopping policy:** CNN may use the DeepSpCas9 validation split for early stopping / best-epoch model selection as a documented exception (D-006). This does not permit Moreno use.
- **Strict RF/XGB baseline policy:** Random Forest and XGBoost baselines use fixed hyperparameters and must not use validation for early stopping or model selection; validation is evaluation-only for these baselines.

## 4. Canonical Model Artifacts

| Artifact | Role | SHA-256 |
|---|---|---|
| `models/rf_baseline_fixed_20260905_001107.pkl` | Canonical Random Forest baseline artifact used for Phase 6 and later audits. | `1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285740280b6` |
| `models/xgboost_baseline_20260905_002839.pkl` | Canonical XGBoost baseline artifact used for Phase 6 and later audits. | `129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd` |
| `models/cnn_baseline_20260905_011720.pt` | Canonical CNN primary-model artifact used for Phase 6 and later audits. | `76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616` |

The hashes above are established in `docs/phase15_reproducibility_recovery_addendum.md`. Canonical artifact files must be protected from silent regeneration, overwrite, conversion, or promotion.

## 5. Phase Decision Log

| Phase | Purpose | Main Result | Decision/Gate | Status |
|---|---|---|---|---|
| Phase 1 | Dataset selection, validation, and reporting. | DeepSpCas9 selected as canonical dataset; Moreno-Mateos identified as independent external test candidate; 30-mer geometry documented. | Establish DeepSpCas9 + Moreno evaluation direction. | Completed in project records. |
| Phase 2 | Feature extraction and canonical representations. | 197 tabular features and CNN one-hot representation established; guide/PAM slicing fixed to canonical geometry. | Use engineered features for RF/XGB and one-hot 30-mer for CNN. | Completed in project records. |
| Phase 3 | Random Forest baseline correction. | Earlier invalid leakage/geometry results superseded; corrected RF external Moreno R2 0.0456, Pearson 0.2352. | Use corrected RF only; discard invalid earlier results. | Completed. |
| Phase 4 | XGBoost baseline under fair fixed protocol. | Stronger validation performance than RF, weaker external performance than RF; no early stopping. | Strict fixed-baseline policy retained. | Completed. |
| Phase 5 | CNN primary model. | CNN learned in-domain signal from one-hot 30-mer; external Moreno performance remained weak. | CNN is primary model; validation early stopping documented as D-006. | Completed. |
| Phase 6 | Cross-model evaluation and statistical comparison. | RF strongest on locked external Moreno; XGB strongest in-domain; all models show limited cross-dataset generalization. | Canonical evaluation package established; MAPE not primary. | Completed. |
| Phase 7 | Sequence-region ablation. | Guide `[4:24]` carried most predictive signal; context/PAM alone collapsed in-domain. | Guide-dominant interpretation, descriptive only. | Completed. |
| Phase 8 | Interpretability. | CNN attribution assigned about 68-70% mass to guide; tabular models highlighted PAM-proximal guide features. | Attribution supports model-behavior interpretation only; no causal biology. | Completed. |
| Phase 9A | Domain diagnostics. | External Moreno differs in activity, GC, k-mers, and has zero exact sequence overlap with DeepSpCas9; predictions are under-dispersed. | Domain-shift diagnostic; no model change. | Completed. |
| Phase 9B | Prediction-scale / calibration hypothesis test. | Internal calibration did not rescue external performance; ranking unchanged by monotonic transforms. | Calibration hypothesis refuted as solution. | Completed. |
| Phase 9C | Domain-weighted retraining audit. | Pre-registered LG weighting did not improve locked Moreno performance; sensitivity gains were tiny/non-significant. | Domain-weighting hypothesis refuted. | Completed in project records. |
| Phase 9D | Final generalization audit. | Canonical models reproducible; external degradation associated with domain shift and under-dispersion; weak ranking signal remains. | Decision B: need more data / dataset diversification first. | Completed. |
| Phase 10 | Multi-dataset diversification with DeepHF. | DeepHF D1 arm gave small, non-significant directional CNN improvement; CIs crossed zero. | PARTIAL_SUPPORT, gate C; retain canonical Phase 3-8 baseline. | Completed/audited. |
| Phase 11 | Targeted dataset analysis after Phase 10. | Phase 10 effect could not be attributed to a single mechanism because size, coverage, and composition were confounded. | Gate D: evidence too weak to attribute; freeze dataset experiments. | Completed. |
| Phase 12 | Controlled data-coverage ablation feasibility. | Controlled isolation was not identifiable with available data; no arms, training, or external evaluation. | Gate F: controlled ablation infeasible. | Completed/STOP. |
| Phase 13 | Controlled representation audit. | Learned nucleotide embedding changed dispersion slightly but did not materially improve external generalization. | Gate B/E; data/domain shift remains dominant limitation. | PASS. |
| Phase 14 | Domain-shift attribution / robustness audit. | Activity extremes and common-support cells associated with external error increase; GC/error rank association not supported at threshold. | Gate B/E; read-only associational audit. | PASS. |
| Phase 15 | Data feasibility and reproducibility recovery. | New accepted training data not established; canonical model binaries later recovered and SHA-256 verified; path portability repaired. | Gate 1 FAIL for new accepted data; technical reproducibility recovered. | Completed. |
| Phase 16A | Dataset discovery / provenance inventory. | Candidate inventory created for CRISPRon, CRISPRpred(SEQ), DeepHF, Doench ORCS, and sgDesigner. | Discovery only; all candidates pending or unresolved. | PASS. |
| Phase 16B | Primary file verification. | Doench and CRISPRpred(SEQ) supplementary files obtained; CRISPRon archive obtained but primary table unresolved; DeepHF and sgDesigner primary files not obtained. | Primary-file verification recorded. | PASS. |
| Phase 16C | Compatibility audit. | Doench and CRISPRpred(SEQ) found incompatible with continuous canonical activity target; other candidates insufficient provenance. | No label harmonization or sequence transformation. | PASS. |
| Phase 16D | Exact overlap / independence audit. | 30-mer overlap not assessable for candidates lacking 30-mers; guide overlaps, duplicates, and conflicts documented for CRISPRpred(SEQ) and Doench; RC not assessed. | PASS WITH MINOR CLARIFICATION. | PASS WITH MINOR CLARIFICATION. |
| Phase 16E | Distribution / composition audit. | GC summaries recorded for DeepSpCas9, CRISPRpred(SEQ), and Doench; candidate labels not treated as activity. | Descriptive distribution audit only; no Phase 16F gate. | PASS. |

## 6. Locked Scientific Decisions

- Continuous activity target must not be silently changed.
- Canonical 30-mer representation and geometry must not be silently changed.
- Moreno-Mateos remains locked.
- No arbitrary label harmonization.
- No near-duplicate thresholds in Phase 16.
- Exact overlap only unless a future protocol explicitly approves otherwise.
- Reverse-complement overlap requires verified strand/orientation.
- Public dataset does not automatically mean independent dataset.
- Dataset size alone is not an acceptance criterion.
- No generalization claims without evidence.
- No causal biological interpretation from model attribution.
- Canonical Phase 3-15 results must remain reproducible.

## 7. Phase 16 Dataset Landscape

| Candidate ID | Provenance status | Primary file status | Compatibility status | Sequence status | Overlap status | Independence status | Distribution status | Current interpretation |
|---|---|---|---|---|---|---|---|---|
| `crispron_2021_rth_tools` | Insufficient provenance after 16B/16C. | Source archive obtained; primary experimental file unresolved. | INSUFFICIENT_PROVENANCE. | No verified sequence file available in 16D/16E. | No 30-mer or guide overlap assessable; RC_OVERLAP_NOT_ASSESSED. | INSUFFICIENT_SEQUENCE_INFORMATION. | No records analyzed; GC n/a. | Cannot be accepted or rejected scientifically beyond insufficient provenance in current records. |
| `crisprpredseq_2020_bmc_additional_files` | Official supplementary files obtained; primary-vs-processed upstream relationship unresolved. | Four BMC supplementary CSV files obtained and hashed. | INCOMPATIBLE: binary 0/1 labels and 23-mer guide+PAM schema; no transformation performed. | 16,749 guide+PAM records observed. | No 30-mer sequence available; 1,966 shared unique guides with DeepSpCas9; 4,284 duplicate-record excess; 402 identical-sequence label conflicts; RC_OVERLAP_NOT_ASSESSED. | NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK. | GC n=16,749, mean 0.5801, median 0.5652, sd 0.0840; labels not analyzed as activity. | Not compatible with current continuous 30-mer activity target; independence not established. |
| `deep_hf_2019_public_data_listing` | Prior Phase 10-12 evidence exists, but Phase 16B did not obtain primary file; role requires supervisor decision. | Primary file not obtained in Phase 16B. | INSUFFICIENT_PROVENANCE in Phase 16C. | No verified Phase 16 sequence file available. | No 30-mer or guide overlap assessable in Phase 16; RC_OVERLAP_NOT_ASSESSED. | INSUFFICIENT_SEQUENCE_INFORMATION in Phase 16. | No Phase 16E records analyzed; GC n/a. | Previously analyzed candidate, not a new accepted dataset from Phase 16 evidence. |
| `doench_2016_orcs_publication_screens` | BioGRID ORCS / publication supplementary files verified; same-study relationships unresolved at acceptance level. | Four supplementary XLSX screen files obtained and hashed. | INCOMPATIBLE: screen-enrichment/rank/log-fold-change outputs, not continuous SpCas9 editing activity; no transformation performed. | 214,820 guide-only sequence records in 16D; 194,653 sequence records in 16E distribution audit. | No 30-mer sequence available; 2,333 shared unique guides with DeepSpCas9; 57,240 duplicate-record excess; 296 identical-sequence label conflicts; RC_OVERLAP_NOT_ASSESSED. | NOT_ESTABLISHED. | GC n=194,653, mean 0.5218, median 0.5000, sd 0.1163; labels not analyzed as activity. | Not compatible with current continuous target; guide overlap and duplicate/conflict structure require caution; independence not established. |
| `sgdesigner_2020_public_data_listing` | Insufficient primary-source verification. | No official primary file obtained. | INSUFFICIENT_PROVENANCE. | No verified sequence file available. | No 30-mer or guide overlap assessable; RC_OVERLAP_NOT_ASSESSED. | INSUFFICIENT_SEQUENCE_INFORMATION. | No records analyzed; GC n/a. | Discovery lead only; not accepted under current evidence. |

## 8. Phase 16 Overlap Findings

- **Exact guide overlap findings:** CRISPRpred(SEQ) additional files share 1,966 unique guides with DeepSpCas9. Doench ORCS publication screens share 2,333 unique guides with DeepSpCas9. CRISPRpred(SEQ) and Doench share 1,543 unique guides with each other.
- **Exact 30-mer limitations:** candidate exact 30-mer overlap could not be assessed for the Phase 16 candidates because the verified CRISPRpred(SEQ) records are 23-mer guide+PAM, Doench records are guide-only, and CRISPRon/DeepHF/sgDesigner lacked verified Phase 16 sequence files.
- **Duplicate findings:** CRISPRpred(SEQ) had 16,749 sequence records, 12,465 unique sequences, 4,217 duplicate sequences, and 4,284 duplicate-record excess. Doench had 214,820 sequence records, 157,580 unique sequences, 26,370 duplicate sequences, and 57,240 duplicate-record excess.
- **Identical-sequence label conflicts:** CRISPRpred(SEQ) had 402 conflicting sequences with observed 0/1 label conflicts. Doench had 296 conflicting guide sequences with differing screen/log-fold-change values.
- **RC_OVERLAP_NOT_ASSESSED:** reverse-complement overlap was not assessed in Phase 16D because strand/orientation was not verified.
- **DeepHF prior relationship:** DeepHF was previously analyzed in Phases 10-12. Phase 10 reported zero DeepHF-vs-Moreno forward/reverse-complement overlap and zero exact 30-mer overlap with DeepSpCas9, plus 330 shared 20 bp guides with the DeepSpCas9 training split. In Phase 16, the DeepHF primary file was not restored, so Phase 16 does not independently re-establish those facts.

These findings must not be called data leakage unless leakage is directly established. Candidates must not be called independent merely because overlap was not assessable.

## 9. Phase 16 Distribution Findings

Phase 16E verified the following GC summaries:

| Dataset | n | Mean GC | Median GC | SD |
|---|---:|---:|---:|---:|
| DeepSpCas9 | 12,832 | 0.5542 | 0.5667 | 0.1319 |
| CRISPRpred(SEQ) additional files | 16,749 | 0.5801 | 0.5652 | 0.0840 |
| Doench ORCS publication screens | 194,653 | 0.5218 | 0.5000 | 0.1163 |
| CRISPRon | 0 | n/a | n/a | n/a |
| DeepHF Phase 16 listing | 0 | n/a | n/a | n/a |
| sgDesigner | 0 | n/a | n/a | n/a |

DeepSpCas9 n=12,832 in Phase 16E is the distribution-audit population, not the canonical modeling population of n=10,117. The canonical modeling population remains 10,117 valid rows after canonical filtering.

CRISPRpred(SEQ) labels were not treated as continuous activity. Doench screen outputs were not treated as continuous activity. No prohibited label transformations were performed.

## 10. Reproducibility and Git Workflow

- **VS Code:** local development and file-editing environment.
- **Codex implementation role:** implement approved technical phases, produce reproducible artifacts, run permitted tests/checks, and report changes. Codex must not independently make scientific design decisions.
- **GitHub repository role:** authoritative shared version-control record for code, documentation, reports, results, and approved handover material. Canonical model artifacts remain protected and are not silently regenerated.
- **ChatGPT supervisor/auditor role:** review scientific decisions, approve phase scope, audit results, decide PASS/REVISE gates, and authorize commits/pushes where applicable.
- **Scientific decision authority:** supervisor/auditor, not Codex acting alone.
- **Workflow:** implementation -> audit -> PASS/REVISE -> commit/push only after PASS and explicit authorization.
- **Canonical artifact protection:** canonical model binaries, canonical metrics, canonical splits, locked external policy, and Phase 3-15 result records must not be overwritten, regenerated, or silently reinterpreted.
- **Working-tree protection:** do not revert or overwrite unreviewed local work; inspect status before changes; keep documentation-only tasks limited to documentation-only outputs.

## 11. Prohibited Actions

The following require explicit supervisor approval or are currently prohibited:

- Retraining canonical models.
- Regenerating, overwriting, converting, or promoting canonical model artifacts.
- Changing labels or redefining the continuous activity target.
- Merging datasets into training without an approved protocol.
- Accessing Moreno raw data outside an approved locked-evaluation or audit protocol.
- Changing canonical 30-mer geometry.
- Changing guide/PAM slicing.
- Changing evaluation protocol or primary metrics.
- Using near-duplicate thresholds without an approved metric/threshold/grouping policy.
- Claiming independent provenance without primary evidence.
- Treating public discovery catalogs as acceptance evidence.
- Automatic model promotion.
- Interpreting attribution as biological causality.
- Label harmonization, rescaling, binarization, or rank conversion without approval.
- Sequence transformation from guide-only or guide+PAM into canonical 30-mer without approval.
- Calling absence of assessable overlap "independence."
- Calling guide overlap "leakage" unless leakage is directly established.

## 12. Current Project Status

- Phase 16A = PASS
- Phase 16B = PASS
- Phase 16C = PASS
- Phase 16D = PASS WITH MINOR CLARIFICATION
- Phase 16E = PASS
- Phase 16F = UNLOCKED

Current immediate next step:

**PHASE 16F - FINAL DATASET LANDSCAPE GATE**

Phase 16F has NOT yet been implemented.

## 13. Open Scientific Questions

- Whether any candidate has sufficient provenance, compatibility, overlap/independence evidence, and distributional support for future pipeline inclusion.
- Which unresolved primary experimental files, if any, can be obtained for CRISPRon, DeepHF in the Phase 16 context, and sgDesigner.
- Whether CRISPRpred(SEQ)'s supplementary files can be traced to primary experimental sources or remain processed/derived records unsuitable for acceptance.
- Whether Doench screen outputs can ever be used for a secondary analysis without changing the canonical continuous activity target.
- Whether strand/orientation can be verified well enough to permit reverse-complement overlap assessment.
- Whether additional public datasets can independently address the known activity-coverage/domain-shift bottleneck without confounding sample size, activity coverage, sequence composition, and provenance.
- Whether future data acquisition can provide canonical 30-mer sequences with compatible continuous SpCas9 editing-efficiency labels and independent provenance.

## 14. Future Continual-Update Architecture

This section describes a future design, not an implemented feature.

Intended future architecture:

```text
public discovery
-> provenance
-> quarantine
-> validation
-> deduplication
-> versioning
-> compatibility
-> candidate training
-> frozen evaluation
-> promotion only if predefined criteria are met
```

Daily discovery does not mean daily automatic model modification. Discovery may add candidates to a quarantined registry, but training, evaluation, promotion, or canonical-model replacement must occur only under a predefined approved protocol with locked criteria.

Suggested registry fields:

- `dataset_id`
- `source_study_id`
- `accession`
- `source_url`
- `first_seen`
- `last_seen`
- `file_sha256`
- `version`
- `n_records`
- `label_definition`
- `sequence_definition`
- `status`

## 15. Handover Instructions for Collaborators

1. Read this file.
2. Read the relevant phase report.
3. Do not assume undocumented decisions.
4. Implement only the approved phase.
5. Return an implementation report.
6. Wait for scientific audit.
7. Commit/push only after PASS.

When information is missing, write `NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.` rather than inventing a value or importing assumptions from general literature.

## 16. Evidence Hierarchy

1. Primary experimental source/provenance.
2. Repository raw/primary files.
3. Project audit results.
4. Project documentation.
5. Discovery catalogs/reviews.
6. Informal/search results.

Discovery sources must not be treated as primary acceptance evidence.

## 17. Final Disclaimer

This document is a project handover and decision record, not a replacement for the underlying experimental data, provenance records, or phase-specific audit reports. Any future scientific decision must return to the relevant primary files, audit JSONs, phase reports, and supervisor-approved protocol.

## Not Established / Not Available In Current Project Record

- A single formal thesis title beyond the repository/project title is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Wet-lab validation results for predicted guides are NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Clinical, therapeutic, safety, or off-target validation is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Primary experimental CRISPRon training table identity is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Phase 16 primary DeepHF file restoration is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Phase 16 sgDesigner primary file identity and publication linkage are NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Reverse-complement overlap for Phase 16 candidates is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Near-duplicate thresholds for Phase 16 are NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
- Accepted Phase 16 candidate dataset for pipeline inclusion is NOT ESTABLISHED / NOT AVAILABLE IN CURRENT PROJECT RECORD.
