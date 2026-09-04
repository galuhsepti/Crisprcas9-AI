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
- Validation (15%) is truly unseen and used only for model selection
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