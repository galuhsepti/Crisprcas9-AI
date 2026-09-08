# Phase 15 Reproducibility Recovery Addendum

**Addendum timestamp:** 2026-09-08T01:10:50+07:00
**Purpose:** append-only technical reproducibility recovery record
**Original feasibility snapshot:** `2026-09-07T23:09:35+07:00`
**Original feasibility result:** Gate 1 `FAIL` for new accepted training data

This document is an addendum. It does not replace or rewrite the original Phase 15 feasibility snapshot or its historical state.

## 1. Original Phase 15 State

The original feasibility audit recorded:

- Gate 1: `FAIL` for new accepted training data.
- Feasibility outcome: `INFEASIBLE` under the available workspace evidence.
- Canonical model state: `RECORDED_BUT_NOT_LOCALLY_VERIFIED`.
- No model training, prediction generation, Moreno access, or new dataset analysis.

The original files remain unchanged:

- `docs/phase15_data_feasibility_report.md`
- `results/phase15_data_feasibility_20260907_230935.json`
- `results/phase15_data_provenance_20260907_230935.json`

## 2. Canonical Artifact Recovery

The canonical binaries were recovered from the original Linux environment/archive:

- `/home/konta/crispr-prediction/models/rf_baseline_fixed_20260905_001107.pkl`
- `/home/konta/crispr-prediction/models/xgboost_baseline_20260905_002839.pkl`
- `/home/konta/crispr-prediction/models/cnn_baseline_20260905_011720.pt`

The local runtime copies were verified against the recovered source files by raw-file SHA-256. All corresponding sizes and hashes matched. No model was regenerated, retrained, loaded and resaved, converted, or scientifically altered.

| Model | Local size | Authoritative SHA-256 | Actual local SHA-256 | Source identical |
|---|---:|---|---|---|
| RF | 53,056,052 bytes | `1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285740280b6` | `1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285740280b6` | yes |
| XGBoost | 477,707 bytes | `129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd` | `129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd` | yes |
| CNN | 78,101 bytes | `76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616` | `76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616` | yes |

The authoritative values are read from the existing Phase 14 configuration/provenance lineage, not manually substituted.

## 3. Portability Recovery

Only project-root resolution was repaired:

- `src/domain_shift/config.py`
- `src/representation_audit/config.py`

Both now derive the project root from the configuration module location using `Path(__file__).resolve().parents[2]`.

No model filename, hash, architecture, preprocessing, split, dataset, metric, threshold, or scientific protocol changed.

## 4. Validation

Targeted Phase 14 reproducibility tests:

```text
3 passed, 41 deselected
```

Full test suite:

```text
410 passed, 3 warnings
```

The warnings were existing XGBoost serialized-model compatibility and constant-input correlation warnings. They did not produce test failures.

## 5. Scientific Immutability

Confirmed for this recovery:

- No model retraining or regeneration.
- No canonical model binary modification.
- No scientific parameter change.
- No canonical metric/result change.
- No Moreno raw-data access.
- No new dataset analysis.
- No prediction artifact generated.
- No commit or push.

## 6. Technical Gate

The original scientific feasibility decision remains historical and unchanged: no new accepted training dataset was established.

The separate technical reproducibility gate now passes:

`REPRODUCIBILITY_RECOVERED`

This addendum records artifact recovery and path portability only. It does not authorize Phase 16 or any new modeling experiment.
