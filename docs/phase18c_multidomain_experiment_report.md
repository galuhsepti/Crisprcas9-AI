# Phase 18C - Multi-domain Experiment Final Report

## Final Status

- Official classification: **NO_CLEAR_BENEFIT**
- Final campaign: `campaign_20260914_141742`
- Runs completed: **35 / 35**
- Failed: **0**
- Skipped: **0**
- Locked code commit: `b8323b671050256cf1a85c3b1b9eec27ae1ed833`

This report documents the completed, preregistered Phase 18C campaign. It does
not change the scientific classification, introduce a rescue analysis, or
authorize Phase 18D. No model was retrained during finalization.

## 1. Scientific Question

Phase 18C asked whether a shared sequence encoder with separate domain-specific
heads could improve within-domain prediction for at least one SpCas9 activity
domain while preserving performance in both domains. The confirmatory
comparison was the multi-domain CNN arm D against the corresponding
single-domain CNN arm A or B, paired by seed and evaluated on locked internal
test partitions.

The two labels were not assumed to be numerically exchangeable. The design
tested shared sequence representation while retaining separate response
mappings for the two assays.

## 2. Phase 18B Preregistered Design

The Phase 18B protocol fixed the datasets, eligibility rules, split membership,
model matrix, architecture, optimization seeds, stopping rules, metrics,
bootstrap procedure, and scientific decision rules before Phase 18C execution.
The primary metric was Spearman correlation. Pearson correlation, R-squared,
MAE, and RMSE were secondary metrics.

The primary inference used paired D-minus-baseline effects for seeds 42 through
46. It used 10,000 hierarchical paired bootstrap replicates with bootstrap seed
42. The bridge was diagnostic and was evaluated only after primary inference.

No architecture search, hyperparameter search, adaptive loss weighting,
best-seed selection, post-result calibration, or rescue training was allowed.

## 3. Datasets and Domains

| Domain | Source label | Unit and timing | Eligible n |
|---|---|---|---:|
| A: DeepSpCas9 | `activity` | Day-2.9 indel fraction | 10,117 |
| B: CRISPRon Xiang/Luo | `HEK293T_indel_freq_avg_d8_d10` | Dox-free day-8/day-10 mean indel percentage | 10,592 |

Both domains measure sequence-linked SpCas9 on-target activity in HEK293T
surrogate-target systems, but their units, timing, selection conditions,
background handling, and response mappings differ. Raw targets therefore
remained domain-specific.

## 4. Split and Bridge Policy

The grouping unit was the exact forward 20-nt spacer at positions `[4:24]`,
globally across both domains. The locked deterministic split used a 70/15/15
hash allocation for train, validation, and internal test.

| Domain | Train | Validation | Internal test | Bridge | Quarantine |
|---|---:|---:|---:|---:|---:|
| DeepSpCas9 | 7,040 | 1,510 | 1,526 | 41 | 0 |
| Xiang/Luo | 7,465 | 1,520 | 1,559 | 41 | 7 |

The 41 canonical exact pairs were held out as the diagnostic bridge. Seven
additional Xiang/Luo rows overlapping canonical-ineligible DeepSpCas9 rows were
quarantined. Bridge and quarantine rows were excluded from training,
validation, internal-test inference, and model selection.

## 5. Experimental Arms

| Arm | Model | Training domains | Confirmatory role |
|---|---|---|---|
| A | Single-domain CNN encoder and DeepSpCas9 head | DeepSpCas9 | Baseline for D on DeepSpCas9 |
| B | Single-domain CNN encoder and Xiang/Luo head | Xiang/Luo | Baseline for D on Xiang/Luo |
| D | Shared CNN encoder and separate domain heads | Both | Primary multi-domain arm |
| RF_A / RF_B | Fixed Random Forest, 197 sequence features | One domain each | Descriptive tree baseline |
| XGB_A / XGB_B | Fixed XGBoost, 197 sequence features | One domain each | Descriptive tree baseline |

Each arm used seeds 42, 43, 44, 45, and 46, producing 35 registered runs.
The CNN encoder contained 17,920 trainable parameters; A and B each contained
17,985 parameters, and D contained 18,050 parameters.

CNN training used CPU float32 deterministic execution, Adam with learning rate
0.001, batch size 32, 467 optimizer steps per epoch, a maximum of 100 epochs,
and patience 10. Arm D used equal domain weighting and separate raw-scale
output heads with train-only numerical preconditioning.

## 6. Execution Environment and Runtime

| Component | Recorded value |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.1.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| PyTorch | 2.14.0+cpu |
| XGBoost | 3.2.0 |
| Platform | Windows 10 |
| CPU | 12th Gen Intel Core i3-1215U, 6 cores / 8 logical processors |
| GPU use | None; CPU-only campaign |
| Physical memory | 8,301,043,712 bytes |

Approximate campaign wall time was 6,579.96 seconds (1 hour, 49 minutes,
40 seconds). Recorded aggregate fit time was 4,322.64 seconds for CNNs, 38.51
seconds for Random Forests, and 6.75 seconds for XGBoost models. Python
tracemalloc peaks were approximately 2.10 MiB for CNN, 7.10 MiB for Random
Forest, and 0.04 MiB for XGBoost. Native process peak working-set memory was not
recorded, so the tracemalloc values must not be interpreted as total process
memory.

## 7. CNN Results: DeepSpCas9

| Seed | A Spearman | D Spearman | Delta Spearman | A RMSE | D RMSE | Relative RMSE gain |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.63107 | 0.63760 | 0.00652 | 0.16908 | 0.16755 | 0.00904 |
| 43 | 0.63935 | 0.64547 | 0.00613 | 0.16770 | 0.16706 | 0.00382 |
| 44 | 0.62457 | 0.65619 | 0.03162 | 0.17051 | 0.16500 | 0.03230 |
| 45 | 0.62420 | 0.65794 | 0.03373 | 0.16994 | 0.16504 | 0.02884 |
| 46 | 0.63641 | 0.65001 | 0.01359 | 0.16795 | 0.16542 | 0.01508 |
| Mean | 0.63112 | 0.64944 | 0.01832 | 0.16903 | 0.16601 | 0.01782 |

Across seeds, A had mean R-squared 0.41429 and mean MAE 0.13412. Arm D
had mean R-squared 0.43505 and mean MAE 0.13231 on the same domain.

The registered decision confidence interval for DeepSpCas9 delta Spearman was
`[-0.00372, 0.04096]`. The interval for relative RMSE gain was
`[-0.00361, 0.03885]`.

## 8. CNN Results: Xiang/Luo

| Seed | B Spearman | D Spearman | Delta Spearman | B RMSE | D RMSE | Relative RMSE gain |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.67718 | 0.68763 | 0.01045 | 17.31978 | 17.00918 | 0.01793 |
| 43 | 0.67582 | 0.68387 | 0.00806 | 17.26724 | 17.14293 | 0.00720 |
| 44 | 0.67379 | 0.68217 | 0.00838 | 17.30253 | 17.16612 | 0.00788 |
| 45 | 0.67528 | 0.68315 | 0.00787 | 17.23660 | 17.26619 | -0.00172 |
| 46 | 0.68703 | 0.68799 | 0.00096 | 17.13222 | 17.13432 | -0.00012 |
| Mean | 0.67782 | 0.68496 | 0.00714 | 17.25168 | 17.14375 | 0.00624 |

Across seeds, B had mean R-squared 0.45216 and mean MAE 13.79698. Arm D
had mean R-squared 0.45899 and mean MAE 13.73650 on the same domain.

The registered decision confidence interval for Xiang/Luo delta Spearman was
`[-0.00680, 0.02157]`. The interval for relative RMSE gain was
`[-0.01234, 0.02455]`.

## 9. Confirmatory Decision

The multi-domain arm showed modest positive mean effects:

- DeepSpCas9 mean delta Spearman: approximately **+0.01832**.
- Xiang/Luo mean delta Spearman: approximately **+0.00714**.
- DeepSpCas9 mean relative RMSE gain: approximately **+0.01782**.
- Xiang/Luo mean relative RMSE gain: approximately **+0.00624**.

No invalid bootstrap replicate was recorded. Seed ranges were within the locked
stability limits. Both domains passed the registered preservation conditions,
and negative transfer was not detected.

The benefit condition nevertheless did not pass. DeepSpCas9 exceeded the
registered +0.01 mean delta-Spearman threshold, but its decision confidence
interval crossed zero. Xiang/Luo did not exceed the +0.01 mean threshold, and
its decision confidence interval also crossed zero. Under the locked decision
rules, the official classification is therefore:

**NO_CLEAR_BENEFIT**

This classification is not changed by descriptive tree performance or bridge
diagnostics.

## 10. Tree Baselines

| Domain and model | Spearman mean (SD) | Pearson mean (SD) | R-squared mean (SD) | MAE mean (SD) | RMSE mean (SD) |
|---|---:|---:|---:|---:|---:|
| DeepSpCas9 RF | 0.64177 (0.00204) | 0.64452 (0.00195) | 0.36031 (0.00262) | 0.14637 (0.00044) | 0.17666 (0.00036) |
| DeepSpCas9 XGBoost | 0.70805 (0.00000) | 0.71458 (0.00000) | 0.50746 (0.00000) | 0.12277 (0.00000) | 0.15501 (0.00000) |
| Xiang/Luo RF | 0.68188 (0.00189) | 0.67694 (0.00154) | 0.41152 (0.00126) | 14.80997 (0.01108) | 17.88022 (0.01907) |
| Xiang/Luo XGBoost | 0.75578 (0.00000) | 0.75078 (0.00000) | 0.55667 (0.00000) | 12.15415 (0.00000) | 15.51928 (0.00000) |

XGBoost was the strongest internal predictive baseline in both domains. Its
locked configuration produced identical results across the five seeds because
the selected tree setup did not use stochastic row or feature subsampling.
These tree results are descriptive and do not replace the paired CNN
confirmatory comparison.

## 11. Bridge Diagnostic

The bridge contained 41 paired canonical sequences and was evaluated after the
primary classifier. Across seeds, moving from separate single-domain CNNs to D
changed the diagnostic ranges as follows:

| Diagnostic | Single-domain range | Multi-domain D range |
|---|---:|---:|
| Head-to-head Spearman | 0.68171 to 0.75557 | 0.89286 to 0.94861 |
| Head-to-head Kendall | 0.51463 to 0.58293 | 0.77317 to 0.83902 |
| Mean absolute percentile-rank disagreement | 0.15610 to 0.17805 | 0.06341 to 0.08659 |
| DeepSpCas9 head-to-own-label Spearman | 0.57857 to 0.61829 | 0.61359 to 0.69233 |
| Xiang/Luo head-to-own-label Spearman | 0.71132 to 0.77003 | 0.71446 to 0.80958 |

The bridge diagnostics improved strongly, but they are diagnostic only. They do
not demonstrate endpoint equivalence, do not alter the internal-test decision,
and do not change the official classification.

## 12. Limitations

- Evidence is limited to two internal domains and five optimization seeds.
- The assays have different units, timing, selection, and response mappings.
- The bridge contains only 41 canonical paired sequences.
- Spacer grouping protects exact reuse but not gene, locus, or near-homology
  dependence.
- Internal-test performance does not establish external generalization.
- XGBoost seed invariance reflects its deterministic locked configuration and
  does not provide five independent stochastic fits.
- Native process peak memory was unavailable.
- No conclusion is drawn from the inaccessible locked external dataset.

## 13. Reproducibility and Artifact Integrity

| Item | SHA-256 or identifier |
|---|---|
| Locked code commit | `b8323b671050256cf1a85c3b1b9eec27ae1ed833` |
| Protocol | `79ec4f03fe7314bc54a337c0d235c4f40e93b441aa40633828e1a50a3ee4d49b` |
| Implementation | `cfacd0cefae25e897b08d363bbdb68f1ff072977317a2e7230ef39ad04a5a093` |
| Split | `b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8` |
| Artifact manifest | `a17be67fe2ebbc0ae92c3b07131f8402ffc1995e8ce783dfcfdbbf0d9cf81d06` |
| Campaign results | `5a2ec302fc73acbd2f121dfa34be7641dc9fbe27d6a8b283d3f32426141b2d78` |
| Execution lock | `3ff5465c030a710bcff9b1e97755c8f832d6cf83a366b23138696b4b520bda89` |

The final artifact manifest records 35 model artifacts and 35 prediction
artifacts. Every record includes experiment ID, seed, protocol hash, split hash,
implementation hash, code commit, artifact path, byte size, and artifact hash.
All 35 experiment IDs are represented exactly once per artifact type.

The final campaign reports `canonical_integrity_after=true`,
`checkpoint_freeze_complete=true`, and `implementation_unchanged=true`.

## 14. Finalization Boundaries

The Moreno-Mateos dataset remained untouched: it was not opened, read, hashed,
searched, trained on, or evaluated during Phase 18C finalization.

The interrupted campaign preserved in `stash@{0}` was excluded from final
inference. Only `campaign_20260914_141742` is the official final campaign.

No post-result rescue training, tuning, calibration, refit, replacement seed,
new experiment, or Phase 18D work occurred. The Phase 18C result remains
**NO_CLEAR_BENEFIT**.

## 15. Archival Boundary

Recommended for version control:

- this final report;
- `campaign_results.json`;
- `execution_lock.json`;
- `artifact_manifest.json`;
- the compact training logs.

The `models/` and `predictions/` directories should remain local because they
contain approximately 461.3 MB and 32.2 MB respectively. Their hashes and
metadata remain represented by `artifact_manifest.json`. If remote binary
retention is later required, archive these two directories through Git LFS or a
versioned release asset rather than ordinary Git history.
