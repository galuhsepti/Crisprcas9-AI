# Phase 13 Report: Controlled Representation Audit

**Testing Whether Sequence Representation Limits External Generalization**

- **Date:** 2026-09-06
- **Status:** PASS
- **Decision gate:** **B/E**
- **Git HEAD (before):** `7cde40039f32013b7cbdc09215dea965a553abe9`
- **Primary representation:** Learned nucleotide (character) embedding of the 30-mer

---

## 1. Title

Controlled Representation Audit: Testing Whether Sequence Representation Limits External Generalization.

## 2. Scientific motivation

Phases 9A–12 established that the canonical CNN (Phase 5) transfers poorly to the locked Moreno-Mateos external test set. Phase 9A demonstrated strong external *under-dispersion* (prediction SD far below target SD). Phase 9B showed calibration did not resolve external generalization; Phase 9C showed domain-weighting did not; Phase 9D concluded that the immediate limitation is dataset/domain shift and recommended data diversification. Phase 10 (multidataset training) produced only partial support; Phase 11 could not isolate the mechanism; Phase 12 concluded that a controlled data-coverage ablation is infeasible under the available data.

Phase 13 tests a *narrower* hypothesis: whether the **sequence representation** itself limits predictive resolution and external generalization, holding the dataset, splits, external test, and model-development procedure fixed. This is a controlled model-capacity/representation experiment, not a data experiment and not a performance-chasing exercise.

## 3. Research question

Does replacing the canonical one-hot sequence representation with a richer (learned nucleotide-embedding) sequence representation materially improve predictive resolution/generalization, under the same training data, same split, same external test, and controlled model-development procedure?

This is **not** about beating a benchmark, maximizing Moreno-Mateos R², or tuning until the external metric improves. The canonical CNN remains the production/reference model.

## 4. Pre-registered hypotheses

These were defined BEFORE training:

- **H1 — REPRESENTATION:** A richer sequence representation can extract predictive information not fully captured by 30-mer one-hot input.
- **H2 — RESOLUTION:** A richer representation may reduce prediction compression / under-dispersion relative to the canonical CNN.
- **H3 — GENERALIZATION:** If the representation captures more transferable sequence signal, external performance may improve.
- **H4 — NULL:** A richer representation may not materially improve external generalization because the dominant limitation may be dataset/domain shift rather than representational capacity.
- **H5 — NON-IDENTIFIABILITY:** Even if a richer model improves performance, the experiment cannot prove representation is the sole cause unless all other experimental factors are controlled.

## 5. Relationship to Phase 9A–12

- **Phase 9A:** external under-dispersion demonstrated → Phase 13 explicitly tests whether a richer representation changes dispersion (Hypothesis H2).
- **Phase 9B (calibration), 9C (weighting):** failed to resolve external generalization → Phase 13 changes the *input representation* rather than the prediction scale or sample weights.
- **Phase 9D:** concluded "more data / diversification first" but noted architecture/representation remained plausible but untested → Phase 13 tests representation specifically.
- **Phase 10–12:** diversified training (10), dataset-contribution analysis (11), and the controlled data-ablation gate (12, INFEASIBLE) → Phase 13 must NOT reopen Phase 12 and must NOT change the dataset. Phase 13 intentionally keeps the canonical DeepSpCas9 data.

## 6. Experimental design

Phase 13 is a **paired, controlled representation comparison**. The ONLY intended change between the canonical CNN (Phase 5) and the Phase 13 model is the input sequence representation:

- Canonical CNN: 30-mer one-hot encoding `(n, 30, 4)` → Conv1d with 4 input channels.
- Phase 13 model: 30-mer learned nucleotide-embedding `(n, 30) → nn.Embedding(4, 8) → (n, 8, 30)` → Conv1d with 8 input channels.

Everything else is held as identical as practical to the canonical Phase 5 CNN: three parallel Conv1d branches (kernels [5,7,9], 64 filters each), ReLU, MaxPool, AdaptiveAvgPool, concatenation, Dense(64)+ReLU+Dropout(0.3), linear output(1), Adam, lr 0.001, batch 32, MSE loss, max 100 epochs, patience 10, early stopping on validation loss, seed 42.

## 7. Dataset and split policy

- **Primary data:** canonical DeepSpCas9 (`data/raw/DeepSpCas9.csv`), exactly as used in Phase 5.
- **Canonical split:** `train_test_split(test_size=0.15, random_state=42)` → 8599 train / 1518 validation (10117 valid 30-mers after validation, matching Phase 5's figures exactly: n_train=8599, n_val=1518).
- **External:** Moreno-Mateos locked, used ONLY for the single final evaluation after freeze.
- No Phase 10 pool, no DeepHF, no new external datasets, no synthetic samples, no manual removal/rebalancing/oversampling/undersampling, no target-distribution alteration.
- Sequence geometry remains the 30-mer with guide at [4:24] and PAM at [24:27].

## 8. Representation definition

**Primary (pre-registered): nucleotide (character) embedding.**

- Tokenization is deterministic: each of the 30 positions maps to its nucleotide index over the fixed vocabulary `{A,C,G,T}` → indices `{0,1,2,3}`.
- The vocabulary/index mapping is a fixed deterministic transformation shared by train/val/external; it does NOT depend on the training distribution and does NOT depend on Moreno-Mateos.
- The ONLY learned parameters are the `nn.Embedding(4, 8)` vectors (an 8-dim vector per nucleotide), fitted on TRAIN ONLY by the model.
- input dimension per position: embedding_dim = 8 (vs canonical one-hot dim 4).
- Sequence length after tokenization: 30 (unchanged).

**Secondary (pre-registered, not run): overlapping 3-mer token embedding** (`KMerEmbeddingRepresentation`). Documented for completeness; not trained because a single primary representation is sufficient for the controlled comparison and the protocol bars an open-ended sweep.

## 9. Model architecture

Phase 13 `EmbeddingCNN`:

```
30-mer -> integer tokens (n,30)
  -> nn.Embedding(4,8)                     # learned, train-only
  -> (n,8,30)
  -> Conv1d(8,64,k=5) + ReLU + MaxPool(2) + GlobalAvgPool
  -> Conv1d(8,64,k=7) + ReLU + MaxPool(2) + GlobalAvgPool   (parallel branch)
  -> Conv1d(8,64,k=9) + ReLU + MaxPool(2) + GlobalAvgPool   (parallel branch)
  -> concat -> Dense(64) + ReLU + Dropout(0.3)
  -> Linear(64,1)
```

This mirrors the canonical `CRISPRsvGN` architecture exactly in the downstream, with 8 embedding channels feeding the (otherwise identical) Conv1d branches instead of the 4 one-hot channels.

## 10. Training protocol

- Optimizer: Adam (lr 0.001) — same as canonical.
- Loss: MSE — same as canonical.
- Batch size 32, max epochs 100, patience 10 — same as canonical.
- Validation used for early stopping / model selection (best validation-loss checkpoint) — same as canonical.
- Random seed 42, CPU threads 4.
- Moreno-Mateos never used for training, tuning, early stopping, or selection.

**No hyperparameter search, no architecture search, no seed fishing, no multiple hidden experiments.** One pre-registered primary representation, trained once under the fixed protocol.

## 11. Leakage controls

- Tokenization/vocabulary is a fixed deterministic function (no data fitting, no external info).
- The only learned preprocessing parameter, the embedding, is fitted on TRAIN ONLY.
- The runner uses an explicit state machine `DESIGN → FEASIBILITY → TRAINING → INTERNAL_EVALUATION → FREEZE → FINAL_EXTERNAL_EVALUATION → COMPLETE`; external evaluation is forbidden before FREEZE and fails programmatically otherwise.
- Moreno-Mateos is loaded only AFTER the experiment is frozen, and only once.
- No module in `src/representation_audit/representation.py` or `model.py` imports or reads Moreno-Mateos data (verified by test).

## 12. Internal results (canonical validation, descriptive only)

These are model-selection context and do NOT prove external generalization.

| Metric | Phase 13 | Canonical CNN |
|---|---|---|
| MAE | 0.1378 | 0.1385 |
| RMSE | 0.1702 | 0.1709 |
| R² | 0.4209 | 0.4164 |
| Pearson r | 0.6493 | 0.6468 |
| Spearman ρ | 0.6323 | 0.6313 |

Prediction mean / SD: Phase 13 = 0.4273 / 0.1463; Canonical = 0.4192 / 0.1351.
Best validation MSE: 0.0289 (epoch 25 of 35 run).

The Phase 13 model performs essentially identically to the canonical CNN on the validation set (differences < 0.001 in MAE/RMSE), within a modest dispersion increase.

## 13. External evaluation policy

- Primary external metric: **MAE** (locked).
- Secondary: RMSE, R², Pearson, Spearman (Kendall also reported as in canonical reporting).
- ONE final evaluation after all decisions frozen, on the canonical Phase 6 external protocol.
- No re-running, no external-test-driven selection.

## 14. External results (locked Moreno-Mateos, n=810)

| Metric | Phase 13 | Canonical |
|---|---|---|
| MAE | **0.2508** | 0.2518 |
| RMSE | 0.2960 | 0.2958 |
| R² | -0.0123 | -0.0111 |
| Pearson r | 0.1959 | 0.1838 |
| Spearman ρ | 0.1915 | 0.1602 |

## 15. Paired statistical comparison

Per-example absolute-error difference, new − canonical across the same 810 external sequences:

- mean ΔMAE (new − canon): **−0.0010** (negative → new marginally better)
- median ΔAE: −0.0026
- bootstrap 95% CI: **[−0.0045, 0.0026]** (crosses zero)
- paired t: t = −0.53, p = 0.599
- Wilcoxon signed-rank: p = 0.307

The CI crosses zero and the point effect is ~0.001 MAE units (~0.4% of the canonical MAE). This is small and statistically non-significant. No strong evidence that the alternative representation materially improves external generalization.

## 16. Prediction compression analysis

External (target SD = 0.294):

- Phase 13 prediction SD = 0.1239 → SD ratio (pred/target) = **0.421**
- Canonical prediction SD = 0.1149 → SD ratio = **0.390**

The Phase 13 representation slightly increases external prediction dispersion (0.421 vs 0.390) toward the target dispersion, consistent with Hypothesis H2 (slightly reduced under-dispression). **However**, better dispersion did NOT translate into improved external accuracy — MAE/R² essentially unchanged. This matches the protocol warning: variance matching ≠ calibration success.

Per-bin MAE (same predefined bins, **not** created after inspecting results):

| Bin | n | Phase 13 MAE | Canonical MAE |
|---|---|---|---|
| [0.0, 0.2) | 177 | 0.377 | 0.370 |
| [0.2, 0.4) | 148 | 0.182 | 0.170 |
| [0.4, 0.6) | 156 | 0.113 | 0.104 |
| [0.6, 0.8) | 165 | 0.198 | 0.211 |
| [0.8, 1.0] | 164 | 0.361 | 0.379 |

Phase 13 is marginally better in the two highest-activity bins and marginally worse in the three lower bins; overall effect is small and within noise.

## 17. Interpretability

Performed as documented, not extensively. No attribution analysis is reported for the Phase 13 model. Integrated Gradients / saliency on a learned embedding is technically noisy and the protocol requires a clearly defined baseline; given the negligible empirical difference from the canonical model and the explicit instruction not to over-claim biological mechanism, attribution was **omitted** and no biological claims are made. (No prohibition on this omission: the protocol states attribution is secondary and may be omitted if technically unreliable.)

## 18. Limitations

- The embedding dimension (8) increases downstream input channels (8 vs 4) and thus capacity slightly; this is a representation change, not a controlled-capacity-neutral swap, though the increase is modest (32 embedding params replacing; downstream stays the same).
- The effect is small and CIs cross zero; the experiment cannot rule out a modest real effect too small to detect at n=810.
- Non-identifiability caveat (H5): even the small observed differences cannot be attributed solely to representation, because embedding introduces new capacity/learning dynamics.
- External under-dispersion persists (both SD ratios ~0.4), confirming domain/coverage shift remains prominent.

## 19. Decision gate

**Gate: B/E**

- **B (Partial support):** an improvement exists but is small/uncertain and only some aspects improve. Externally, MAE improved only marginally (mean ΔMAE −0.0010, 95% CI [−0.0045, 0.0026] crossing zero, paired tests non-significant) and prediction dispersion increased slightly (0.421 vs 0.390) without an accuracy gain.
- **E (Data/domain shift remains dominant limitation):** the representation change did not resolve external under-dispersion (both models remain ~0.4 of target SD) and did not materially change external R² (~0). The dominant limitation remains dataset/domain shift.

Both B and E are scientifically valid and reported together, as the protocol permits.

Conclusion as per H4: *A richer representation did not materially improve external generalization under the tested canonical-data condition; the dominant limitation remains dataset/domain shift rather than representational capacity.*

## 20. Reproducibility / provenance

- Git HEAD before: `7cde40039f32013b7cbdc09215dea965a553abe9` (Phase 12 commit exists).
- Python 3.14.4; PyTorch 2.14.0+cpu; NumPy 2.5.2; scikit-learn 1.9.0; pandas 3.0.5; scipy 1.18.1.
- Dataset: `data/raw/DeepSpCas9.csv` (canonical), `data/raw/Moreno-Mateos.csv` (locked).
- Canonical checkpoint: `models/cnn_baseline_20260905_011720.pt` (loaded, not modified).
- Canonical artifact hashes recorded before and verified unchanged after the experiment.
- Phase 13 checkpoint: `models/phase13_nucleotide_embedding_20260906_203923.pt`.
- Random seed 42; bootstrap seed 20260906; n_boot 1000; percentile (not BCa) CI.
- Split: val_ratio 0.15, seed 42 (identical to Phase 5).
- Parameter counts: total 23,393; embedding 32; downstream 23,361 (included in results JSON).
- Minimal configuration in `src/representation_audit/config.py` (frozen).
- No undocumented manual preprocessing; no canonical artifact modified.

## 21. Final conclusion

Phase 13 provides **no strong evidence** that replacing the canonical one-hot representation with a learned nucleotide embedding materially improves external generalization under the tested canonical-data condition. The small external MAE improvement is within the bootstrap CI crossing zero and is not statistically significant. The representation did modestly increase prediction dispersion toward the target (H2 direction), but this did not translate into external accuracy gains.

Therefore: **representation remains plausible but is not sufficient to explain external generalization.** The dominant limitation continues to be dataset/domain shift, consistent with Phases 9A–12. This is a scientifically valid null-to-partial result; it does not reopen Phase 12 or justify dataset expansion.

---

*Generated by the Phase 13 runner. This is the frozen report; no further changes were made after external evaluation.*
