# Phase 5 Report - CNN (Primary Model)

## Overview

Implementation and evaluation of a Convolutional Neural Network (CNN) for
sgRNA activity prediction using **PyTorch**. This is the **primary model** of
the thesis. Unlike Random Forest and XGBoost (baselines), the CNN consumes the
raw one-hot encoded 30-mer directly (`n, 30, 4`) and learns the relevant
sequence features from data, rather than from engineered features.

## Methodology

The overall protocol follows the same rigorous safeguards as Phases 3/4
(`docs/decisions.md` D-001, D-002, D-003, D-004):

- **Training set:** DeepSpCas9 85% split (n = 8,599)
- **Validation set:** DeepSpCas9 15% split (n = 1,518), a held-out validation
  set. It is **not** used for gradient updates, but it is used for early
  stopping / model selection (D-006).
- **Independent test set:** Moreno-Mateos (n = 810), held out; used only for
  final evaluation, never for training, tuning, early stopping, or model
  selection
- Fixed seed split (`random_seed = 42`)
- Consistent sequence geometry: guide `[4:24]`, PAM `[24:27]`
- One-hot encoding produced by the same extractor used everywhere else
  (`extract_one_hot_for_cnn`), from validated sequences only

### Early stopping — deliberate exception (D-006, documented)

The CNN **is allowed to use the validation set for early stopping** (best model
by validation loss is retained). This is a documented asymmetry with the
baselines:

- RF and XGBoost are **evaluation baselines** — to compare fairly they must use
  fixed hyperparameters and never touch validation during training, otherwise
  the comparison between models is unfair (D-005).
- The CNN is the **primary model** of the thesis. Early stopping on a
  validation split is standard practice for deep networks and does not use the
  external test set. The test set still remains fully independent of all
  training and model-selection decisions.

The selected checkpoint corresponds to the epoch with the **minimum
validation loss** (`best_epoch`), and `total_epochs_run` records the actual
number of epochs trained. Both are 1-indexed and stored consistently in the
experiment JSON.

This decision is recorded in `docs/decisions.md` (D-006) so the thesis text can
state it explicitly and defensibly.

### CNN Architecture (`config.yaml`)

CRISPRpred-style parallel multi-kernel convolutions over the one-hot encoding:

```yaml
cnn:
  input_length: 30
  n_nucleotides: 4
  conv_n_filters: 64            # per branch
  conv_kernel_sizes: [5, 7, 9]  # 3 parallel receptive fields
  dense_units: 64
  dropout_rate: 0.3
  learning_rate: 0.001
  batch_size: 32
  epochs: 100    # max
  patience: 10   # early stopping on validation
  optimizer: adam
  loss: mse
```

Each branch: `Conv1d(4, 64, k)` -> ReLU -> MaxPool1d(2) -> GlobalAvgPool.
Branch outputs are concatenated (192-dim) -> Dense(64) -> ReLU -> Dropout(0.3)
-> Linear(1). Ran on CPU (no CUDA in this environment), 4 threads.

## Results

### Training run (reproducible)

Same dataset, split (85/15, seed 42), architecture and hyperparameters as the
initial run. Results are numerically identical, confirming reproducibility.

| Quantity | Value |
|----------|-------|
| Best epoch (1-indexed, min validation loss) | 24 |
| Total epochs run | 34 |
| Best validation loss | 0.029134 |
| Early stopping | Triggered at epoch 34 (patience 10) |

### Validation set (held-out validation, not used for gradient updates, n = 1,518)

| Metric | Value |
|--------|-------|
| MAE | 0.1385 |
| RMSE | 0.1709 |
| R² | 0.4164 |
| Pearson r | 0.6468 (p = 1.2e-180) |
| Spearman ρ | 0.6313 (p = 1.5e-169) |
| Kendall τ | 0.4519 (p = 2.4e-153) |

### Independent test set (Moreno-Mateos, n = 810)

| Metric | Value |
|--------|-------|
| MAE | 0.2518 |
| RMSE | 0.2958 |
| R² | -0.0111 |
| Pearson r | 0.1838 (p = 1.4e-07) |
| Spearman ρ | 0.1602 (p = 4.6e-06) |
| Kendall τ | 0.1092 (p = 3.3e-06) |

### MAPE is not a primary metric

The primary metrics of this research are MAE, RMSE, R², Pearson correlation,
and Spearman correlation. Mean Absolute Percentage Error (MAPE) is **not** used
as a primary metric. Many sgRNA activity targets are close to zero, making the
per-sample percentage error explode and yielding unstable, near-arbitrarily
large MAPE values. MAPE therefore cannot be used for model selection or for
concluding that one model is better. It is still computed by the generic
evaluation utility (`calculate_all_metrics`) for compatibility, but it is not
shown in the primary results and must not be used for comparison.

## Three-Model Comparison

### Validation (in-domain, DeepSpCas9 15%)

| Metric | CNN | XGBoost | RandomForest |
|--------|---------|---------|--------------|
| MAE | 0.1385 | **0.1263** | 0.1499 |
| RMSE | 0.1709 | **0.1564** | 0.1787 |
| R² | 0.4164 | **0.5112** | 0.3619 |
| Pearson r | 0.6468 | **0.7173** | 0.6375 |
| Spearman ρ | 0.6313 | **0.7051** | 0.6262 |

### Test (cross-domain, Moreno-Mateos)

| Metric | CNN | XGBoost | RandomForest |
|--------|---------|---------|--------------|
| MAE | 0.2518 | 0.2540 | **0.2485** |
| RMSE | 0.2958 | 0.2977 | **0.2874** |
| R² | -0.0111 | -0.0240 | **0.0456** |
| Pearson r | 0.1838 | 0.1833 | **0.2352** |
| Spearman ρ | 0.1602 | 0.1772 | **0.2263** |

## Interpretation

**In-domain (validation, DeepSpCas9 15%):** the CNN (R² 0.4164, Pearson 0.6468)
learns a meaningful sequence-activity relationship from raw sequence alone. It
outperforms the Random Forest baseline on this split but does not match XGBoost
(R² 0.5112), which is trained on 197 engineered features (GC content,
composition, k-mers, positional features) rather than the raw one-hot encoding.
No claim is made here about the cause of this difference beyond the difference
in input representation and model family.

**Cross-dataset (Moreno-Mateos):** all three models showed substantially
reduced performance on the independent Moreno-Mateos dataset (negative R²),
indicating **limited cross-dataset generalization under the present feature and
model configurations**. Random Forest showed the strongest performance among
the three models on this external test set. This comparison is observational:
it reflects the measured performance on this dataset under the fixed
configurations used. It should not be interpreted as causal evidence about
overfitting or about the general transferability of any individual model
without further dedicated analysis.

**Statistical significance:** all three models retain statistically significant
correlations on the external test set (Pearson p ≈ 1e-7 for the CNN), i.e. the
predictions preserve a weak but detectable monotonic relationship with measured
activity, while absolute accuracy (R²) is poor out-of-domain.

**Role of engineered features:** the close CNN vs XGBoost split reflects the
different input representations (raw one-hot vs 197 engineered features). It
shows the Phase 2 engineered features capture a large share of the learnable
in-domain signal; the CNN demonstrates end-to-end learning is viable on this
data at modest size.

## Conclusion for Thesis

- The CNN (primary model) achieves moderate in-domain performance (R² 0.42,
  Pearson 0.65) purely from raw sequence, confirming end-to-end deep learning
  for sgRNA activity prediction is workable in this pipeline.
- On the validation split, XGBoost had the highest in-domain performance among
  the three models; on the external Moreno-Mateos test set, Random Forest had
  the highest measured performance. All three models show limited
  cross-dataset performance (see Interpretation).
- The early-stopping asymmetry vs baselines is deliberately recorded (D-006) so
  the method section is precise about how each model's validation split was used.

## Files

- Model: `models/cnn_baseline_20260905_011720.pt` (regenerated artifacts)
- Results: `results/experiments/cnn_baseline_20260905_011720.json` (includes
  the full validation-loss curve, best_epoch, total_epochs_run, and the
  XGBoost + RF comparison). The earlier run
  `results/experiments/cnn_baseline_20260905_004728.json` is preserved for
  traceability.
- Modules: `src/models/cnn.py` (`CNNModel`, `CRISPRsvGN`);
  exported from `src/models/__init__.py`
- Training script: `scripts/train_cnn.py`
- Tests: `tests/test_cnn.py` (CNN, incl. best_epoch consistency),
  `tests/test_evaluation.py` (MAPE-not-primary documentation); 11 + 3
  respective tests, passing