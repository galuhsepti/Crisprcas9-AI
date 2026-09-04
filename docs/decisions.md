# Decisions Log

This file records important methodological decisions and corrections made
during the project. Each entry explains what changed, why, and its impact.

---

## D-001: Correct 30-mer sequence geometry (guide_start = 4)

**Date:** 2026-09-05
**Status:** Applied

### Issue
In the initial Phase 2/3 implementation, the guide start position was derived
as `guide_start = (context_length - guide_length) // 2`, which evaluates to
`(30 - 20) // 2 = 5`. This is **incorrect** for the standard DeepSpCas9 30-mer
format, where the guide spans positions **4-23**, not 5-24.

### What was wrong
With the derived `guide_start = 5`:
- guide slice was `[5:25]` instead of `[4:24]` (last guide base was actually
  the first PAM base at position 24)
- `gc_pam` used `[25:28]` instead of `[24:27]`
- `gc_3prime_context` used `[28:]` instead of `[27:]`
- positional one-hot features were extracted from the wrong guide window
- PAM validation in `validate_sequence(check_pam=True)` checked position 25
  instead of 24

### Fix
- `guide_start` is now a fixed constant (`4`) in:
  - `src/bioinformatics/sequence_features.py`
  - `src/bioinformatics/gc_content.py`
  - `src/bioinformatics/kmer.py`
  - `src/data/preprocessing.py`
  - `src/data/validation.py`
- Explicit from/to bounds added to `SequenceFeatureExtractor`
  (`guide_end=24`, `pam_start=24`, `pam_end=27`)
- Documented in `config.yaml` under `data`
- Regression tests added in `tests/test_bioinformatics.py`
  (`TestGuideGeometry`)

### Impact
- Guide-based features now describe the correct biologically meaningful window
- Slightly different valid-sequence set after filtering (guide homopolymer
  check now uses correct guide sequence)

---

## D-002: Fix train/validation leakage in RF baseline

**Date:** 2026-09-05
**Status:** Applied

### Issue
In the initial Phase 3 training script, the Random Forest was fitted with
`model.fit(X_train_full, y_train_full, ...)`, i.e. on the **entire**
DeepSpCas9 set. The 15% held-out "validation" split was therefore a subset of
the training data, making validation metrics unreliable (they were not
measuring generalization to unseen data).

### Fix
- Model is now fitted **only** on `X_train` / `y_train` (85% split)
- Validation (15%) is a held-out set not used during model fitting; used only
  for evaluation / model selection
- Moreno-Mateos held-out set is used **only** for final evaluation — never for
  tuning or feature engineering
- Split seed fixed (`random_seed = 42`)

### Impact
- Validation metrics dropped from the inflated
  MAE 0.0745 / RMSE 0.0905 / R² 0.8242 (leaked) to the honest
  MAE 0.1499 / RMSE 0.1787 / R² 0.3619.
- The leaked numbers were discarded and must not appear in the thesis.
  See D-003.

---

## D-003: Discard invalid baseline numbers

**Date:** 2026-09-05
**Status:** Applied

### Decision
The following metrics from the first Phase 3 run are **invalid** and are
excluded from all thesis reporting:

| Metric | Invalid value | Reason |
|--------|---------------|--------|
| Validation MAE | 0.0745 | Data leakage (D-002) + wrong geometry (D-001) |
| Validation RMSE | 0.0905 | same |
| Validation R² | 0.8242 | same |

### Rationale
These numbers came from training on data that included the validation samples
and from features computed with incorrect sequence slicing. They do not
represent generalization performance and would misrepresent the model's
capability.

### Replacement
The corrected Phase 3 results (see `docs/phase3_rf_report.md`) are the
authoritative baseline:
- Validation: MAE 0.1499, RMSE 0.1787, R² 0.3619, Pearson r 0.6375
- Independent test: MAE 0.2485, RMSE 0.2874, R² 0.0456, Pearson r 0.2352

---

## D-004: Homopolymer filtering

**Status:** Kept

### Decision
Sequences whose guide contains a homopolymer run of ≥4 identical nucleotides
are filtered out before training (2715 from DeepSpCas9, 210 from
Moreno-Mateos).

### Rationale
- Consistent with common practice in CRISPR screening studies
- Reflects that such guides are avoided experimentally (poor synthesis/activity)
- Prevents trivial sequence-level artifacts from dominating features

### Re-evaluation trigger
If filtered-out sequences materially change sample size or skew the
distribution, revisit whether threshold should be relaxed.

---

## D-005: Remove early stopping / validation-based model selection from XGBoost baseline

**Date:** 2026-09-05
**Status:** Applied

### Issue
The first Phase 4 XGBoost run used `early_stopping_rounds=20` with the
validation set as the `eval_set`. This made validation influence the number of
boosting rounds (a form of model selection on the validation set), which is
**not fair** relative to the Random Forest baseline. Random Forest used a fixed
`n_estimators=200` with no validation feedback.

### Fix
- The XGBoost baseline is now trained with a **fixed** `n_estimators=100`
  (from `config.yaml`) and **no early stopping**.
- Validation is used strictly for evaluation, never for training/model
  selection — exactly matching the Random Forest protocol.
- Moreno-Mateos remains held out and is used only for final evaluation.
- Early-stopping remains available in `XGBoostModel.fit` as an option but is
  **not** used for baselines.
- The old early-stopping experiment JSON/model was removed.

### Impact
- The methodology is now symmetric and fair across RF and XGBoost baselines.
- Reported numbers are unchanged (early stopping had stopped at round 99, so
  fixed 100 rounds produces identical predictions), but the protocol is now
  methodologically correct and defensible for the thesis.

### Files updated
- `scripts/train_xgboost.py` — remove `eval_set`/`early_stopping_rounds` from
  baseline fit; record `early_stopping: false` in results metadata.
- `docs/phase4_xgboost_report.md` — update methodology section.
- Removed `xgboost_baseline_20260905_002128.*` (early-stopping run).
- Regenerated `xgboost_baseline_20260905_002839.json` with corrected protocol.

---

## D-006: CNN is the primary model — early stopping on validation allowed

**Date:** 2026-09-05
**Status:** Applied

### Decision
The CNN (Phase 5) is the **primary model** of the thesis. It **may** use the
DeepSpCas9 validation split (15%) for early stopping (best model by validation
loss retained). RF and XGBoost remain strict evaluation baselines with fixed
hyperparameters and **no** validation influence (D-005).

### Rationale
- D-005's ban on validation-based model selection exists to keep the baseline
  comparison fair. It does not apply to the primary model.
- Early stopping on a train/validation split is standard deep-learning practice
  and does not consume the external Moreno-Mateos test set, which remains fully
  independent and is used only for final evaluation.
- Framework: PyTorch (CPU; no CUDA in this environment). TensorFlow was
  considered but not installed.

### Impact
- The validation set is a **held-out validation set**: it is not used for
  gradient updates, but it is used for early stopping / model selection. The
  CNN is fitted on the 85% split only; validation decides when to stop and
  which checkpoint is kept. Terminology used in reports reflects this
  precisely (never "truly unseen validation").
- This is exactly why RF/XGBoost validation results must not be compared with
  the CNN on a head-to-head "no validation contact" basis — the thesis text
  states the asymmetry explicitly.
- Moreno-Mateos results remain a clean, equal-footing comparison across all
  three models.

### Files updated
- `src/models/cnn.py`, `scripts/train_cnn.py`, `config.yaml` (CNN section),
  `tests/test_cnn.py`, `docs/phase5_cnn_report.md`.

---

## D-007: Audit fixes — best_epoch metadata, terminology, MAPE, interpretation

**Date:** 2026-09-05
**Status:** Applied

### Decision
A methodology audit of Phase 5 (CNN) was performed. The following corrections
were applied without changing the research goal, datasets, labels, sequence
geometry, CNN architecture, hyperparameters, split ratio, or random seed:

1. **`best_epoch` metadata fixed.** Previously `best_epoch` was set to
   `len(val_loss)` (the number of epochs run). It now correctly records the
   **1-indexed epoch with the minimum validation loss**. `best_epoch` is
   guaranteed `<= total_epochs_run`, `best_val_loss ==
   val_loss[best_epoch - 1]`, and checkpoint selection logic is unchanged.
   Without a validation set, `best_epoch = total_epochs_run` and
   `best_val_loss = None` (no validation contact). The epoch index convention
   (1-indexed) is documented in code.
2. **Validation terminology.** The CNN validation set is described as a
   "held-out validation set, not used for gradient updates but used for early
   stopping / model selection" — not "truly unseen"/"completely unseen".
   Terminology applied consistently in `src/models/cnn.py`,
   `scripts/train_cnn.py`, the experiment JSON, and Phase 3/4/5 reports.
3. **Cross-dataset interpretation made conservative.** Reported as: all
   three models showed substantially reduced performance on Moreno-Mateos,
   indicating limited cross-dataset generalization under the present feature
   and model configurations; Random Forest had the strongest measured
   performance there. Causal claims such as "XGBoost overfits" and "all three
   models generalize comparably" were removed. In-domain validation performance
   and external cross-dataset performance are reported separately.
   Result numbers were not changed to fit a narrative.
4. **MAPE.** Mean Absolute Percentage Error is **not** a primary metric (it is
   unstable when targets approach zero). Primary metrics are MAE, RMSE, R²,
   Pearson, Spearman. `calculate_mape` remains for compatibility, is not shown
   by `format_metrics_report`, and a test documents its instability.

### Rationale
- Scientific correctness and reproducibility take priority over performance or
  convenience.
- Moreno-Mateos remains completely untouched until final evaluation.

### Files updated
- `src/models/cnn.py` (best_epoch tracking, docstrings)
- `scripts/train_cnn.py` (terminology; JSON now stores the full validation-loss
  curve and corrected validation_split description)
- `src/evaluation/metrics.py` (MAPE warning docstring)
- `docs/phase5_cnn_report.md`, `docs/phase4_xgboost_report.md`,
  `docs/phase3_rf_report.md` (terminology + conservative interpretation)
- `tests/test_cnn.py`, `tests/test_evaluation.py` (best_epoch + MAPE tests)
- Regenerated experiment `results/experiments/cnn_baseline_*.json` with the
  same data/split/seed/architecture/hyperparameters (numbers unchanged;
  `best_epoch`, `total_epochs_run`, `best_val_loss` now consistent).