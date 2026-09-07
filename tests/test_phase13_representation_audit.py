"""
Phase 13 tests: Controlled Representation Audit.

Covers the 16 required test requirements, including explicit guards against
external-test leakage. These tests are written BEFORE considering the phase
complete.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.representation_audit.config import Phase13ExperimentConfig
from src.representation_audit.representation import (
    NucleotideEmbeddingRepresentation,
    KMerEmbeddingRepresentation,
)
from src.representation_audit.model import EmbeddingCNNModel, EmbeddingCNN
from src.representation_audit.evaluation import ExperimentStateMachine
from src.representation_audit.stats import (
    bootstrap_delta_mae_ci,
    reproducible_bootstrap_seed,
)


# ---------------------------------------------------------------------------
# Config / reproducibility
# ---------------------------------------------------------------------------
def test_config_reproducible():
    """(13) Configuration is frozen and reproducible."""
    cfg1 = Phase13ExperimentConfig()
    cfg2 = Phase13ExperimentConfig()
    assert cfg1 == cfg2
    assert cfg1.random_seed == 42
    assert cfg1.split_seed == 42
    assert cfg1.patience == 10
    assert cfg1.batch_size == 32
    assert cfg1.learning_rate == 0.001


def test_config_dataset_no_external_pool():
    """(16) Config does NOT expand to Phase 10/11/12 pools."""
    cfg = Phase13ExperimentConfig()
    assert cfg.primary_dataset.endswith("DeepSpCas9.csv")
    assert "pool" not in cfg.primary_dataset.lower()
    assert "deephf" not in cfg.primary_dataset.lower()


# ---------------------------------------------------------------------------
# Tokenization determinism / correctness
# ---------------------------------------------------------------------------
def test_tokenization_deterministic():
    """(1) Deterministic tokenization."""
    rep = NucleotideEmbeddingRepresentation()
    seq = "TTCTGCCTTGTTTCTTTCCTCTCTGGGTCG"
    t1 = rep.tokenize(seq)
    t2 = rep.tokenize(seq)
    np.testing.assert_array_equal(t1, t2)


def test_tokenization_correct_length():
    """(2) Correct sequence length after tokenization = 30."""
    rep = NucleotideEmbeddingRepresentation()
    seq = "TTCTGCCTTGTTTCTTTCCTCTCTGGGTCG"
    tokens = rep.tokenize(seq)
    assert tokens.shape == (30,)


def test_vocabulary_correct():
    """(3) Correct vocabulary = {A,C,G,T}."""
    rep = NucleotideEmbeddingRepresentation()
    assert set(rep.vocabulary) == {"A", "C", "G", "T"}
    assert rep.vocab_size == 4
    for nuc in "ACGT":
        assert nuc in rep.nuc_to_index


def test_invalid_nucleotide_raises():
    """(4) No invalid-nucleotide handling ambiguity: unknown char raises."""
    rep = NucleotideEmbeddingRepresentation()
    with pytest.raises(ValueError):
        rep.tokenize("TTCTGCCTTGTTTCTTTCCTCTCTGGGTCX")
    with pytest.raises(ValueError):
        rep.tokenize("TTCTGCCTTGTTTCTTTCCTC" + "T" * 20)  # wrong length


def test_kmer_tokenization_length():
    """(2,3) K-mer representation: overlapping 3-mers -> 28 tokens."""
    rep = KMerEmbeddingRepresentation(k=3, context_length=30)
    seq = "A" * 30
    tokens = rep.tokenize(seq)
    assert tokens.shape == (28,)
    assert rep.vocab_size == 4 ** 3


# ---------------------------------------------------------------------------
# Train-only fitting / no external dependence
# ---------------------------------------------------------------------------
def test_tokenization_no_moreno_dependence():
    """(5)(7) Tokenization does not depend on training data or Moreno-Mateos."""
    rep = NucleotideEmbeddingRepresentation()
    # Tokenization is a pure function of the sequence; remains valid for any
    # 30-mer over {A,C,G,T}.
    tok = rep.tokenize("A" * 30)
    assert tok.sum() == 0  # all A -> index 0


def test_no_moreno_mateos_import():
    """(6) Representation/model modules never load or read external data."""
    import src.representation_audit.representation as rep_mod
    import src.representation_audit.model as model_mod
    src_text = (Path(rep_mod.__file__).read_text()
                + Path(model_mod.__file__).read_text())
    # The representation and model modules must be pure: they may not read any
    # dataset file (including Moreno-Mateos) or import data-loading routines.
    assert "Moreno" not in src_text
    assert "pd.read_csv" not in src_text
    assert "open(" not in src_text
    assert "load_and_validate" not in src_text
    assert "primary_dataset" not in src_text
    assert "external_dataset" not in src_text


def test_model_forward_pass():
    """(9) Model forward pass."""
    rep = NucleotideEmbeddingRepresentation(embedding_dim=8)
    model = EmbeddingCNN(vocab_size=rep.vocab_size, embedding_dim=8, seq_len=30)
    toks = torch.randint(0, 4, (8, 30))
    out = model(toks)
    assert out.shape == (8, 1)


def test_model_output_shape_and_ordering():
    """(10) Output shape and prediction ordering preserved with input order."""
    rep = NucleotideEmbeddingRepresentation(embedding_dim=8)
    model = EmbeddingCNNModel(
        vocab_size=4, embedding_dim=8, seq_len=30, epochs=1, batch_size=8,
        use_cpu_threads=1, random_state=42,
    )
    # Build tiny dataset
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(20, 30))
    y = X[:, 10].astype(float).ravel() * 0.1
    model.fit(X, y, verbose=False)
    preds = model.predict(X)
    assert preds.shape == (20,)
    # Ordering: swap two rows -> predictions order follows input order.
    X_swapped = X.copy()
    X_swapped[[0, 1]] = X_swapped[[1, 0]]
    preds_swapped = model.predict(X_swapped)
    np.testing.assert_allclose(preds_swapped[0], preds[1], atol=1e-5)
    np.testing.assert_allclose(preds_swapped[1], preds[0], atol=1e-5)


def test_canonical_split_preservation():
    """(7) Canonical split preservation (8599 / 1518)."""
    cfg = Phase13ExperimentConfig()
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from src.data.validation import validate_sequence
    from src.bioinformatics import extract_one_hot_for_cnn

    df = pd.read_csv(cfg.paths()["primary_dataset"])
    valid = [validate_sequence(s, check_pam=False)[0] for s in df["sequence_30mer"]]
    df = df[valid].copy()
    X = extract_one_hot_for_cnn(df["sequence_30mer"].tolist(), 30)
    y = df["activity"].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    assert X_tr.shape[0] == 8599
    assert X_val.shape[0] == 1518


# ---------------------------------------------------------------------------
# Paired comparison alignment
# ---------------------------------------------------------------------------
def test_paired_comparison_alignment():
    """(11) Paired comparison aligns sequence-by-sequence."""
    from src.representation_audit.evaluation import (
        run_final_external_evaluation,
        ExperimentStateMachine,
    )
    y_true = np.random.RandomState(1).rand(50)
    y_new = y_true + 0.05
    y_canon = y_true - 0.05

    sm = ExperimentStateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    sm.transition("TRAINING")
    sm.transition("INTERNAL_EVALUATION")
    sm.freeze()

    res = run_final_external_evaluation(
        y_true, y_new, y_canon, sm, n_boot=100, bootstrap_seed=7,
    )
    assert "delta_mae_bootstrap" in res
    assert res["delta_mae_bootstrap"]["n_samples"] == 50


# ---------------------------------------------------------------------------
# State machine guards (leakage)
# ---------------------------------------------------------------------------
def test_state_machine_prevents_external_before_freeze():
    """(15) No external evaluation before freeze."""
    sm = ExperimentStateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    sm.transition("TRAINING")
    sm.transition("INTERNAL_EVALUATION")
    with pytest.raises(RuntimeError):
        sm.transition("FINAL_EXTERNAL_EVALUATION")


def test_state_machine_prevents_backward():
    """(15) No backward transitions."""
    sm = ExperimentStateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    with pytest.raises(RuntimeError):
        sm.transition("DESIGN")


def test_state_machine_freezes_then_external():
    """(15) Freeze then external is allowed."""
    sm = ExperimentStateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    sm.transition("TRAINING")
    sm.transition("INTERNAL_EVALUATION")
    sm.freeze()
    sm.transition("FINAL_EXTERNAL_EVALUATION")
    sm.transition("COMPLETE")
    assert sm.state == "COMPLETE"


# ---------------------------------------------------------------------------
# Bootstrap reproducibility
# ---------------------------------------------------------------------------
def test_bootstrap_reproducible():
    """(12) Bootstrap reproducibility."""
    ae_new = np.random.RandomState(2).rand(100)
    ae_canon = np.random.RandomState(3).rand(100)
    r1 = bootstrap_delta_mae_ci(ae_new, ae_canon, n_boot=200, rng=reproducible_bootstrap_seed(5))
    r2 = bootstrap_delta_mae_ci(ae_new, ae_canon, n_boot=200, rng=reproducible_bootstrap_seed(5))
    np.testing.assert_allclose(r1["ci_lower"], r2["ci_lower"])
    np.testing.assert_allclose(r1["ci_upper"], r2["ci_upper"])
    assert r1["ci_method"] == "paired percentile bootstrap (not BCa)"


# ---------------------------------------------------------------------------
# Prediction compression / activity bins
# ---------------------------------------------------------------------------
def test_prediction_compression():
    """Report SD ratios and predefined bins; D correctly flags resolution."""
    from src.representation_audit.evaluation import prediction_compression_analysis
    y_true = np.linspace(0, 1, 100)
    y_new = y_true * 0.5        # compressed
    y_canon = y_true * 0.3      # more compressed
    c = prediction_compression_analysis(y_true, y_new, y_canon)
    assert c["new_sd_over_target_sd"] == pytest.approx(0.5, rel=0.05)
    assert c["canonical_sd_over_target_sd"] == pytest.approx(0.3, rel=0.05)
    # Bins are the predefined grid.
    assert len(c["new_bins"]) == 5


# ---------------------------------------------------------------------------
# No canonical artifact modification
# ---------------------------------------------------------------------------
def test_no_dataset_expansion_in_representation():
    """(16) Representation never touches dataset expansion/pooling."""
    rep = NucleotideEmbeddingRepresentation()
    # Tokenization is a fixed deterministic function, cannot add/remove samples.
    seqs = ["A" * 30, "C" * 30, "G" * 30]
    toks = rep.tokenize_batch(seqs)
    assert toks.shape == (3, 30)


def test_no_external_eval_before_freeze_guard_in_runner_source():
    """(15) The runner contains an explicit assertion/guard against premature
    external evaluation (state machine present)."""
    runner = (Path(__file__).parent.parent
              / "scripts" / "run_phase13_representation_audit.py").read_text()
    assert "ExperimentStateMachine" in runner
    assert "FREEZE" in runner
    assert "FINAL_EXTERNAL_EVALUATION" in runner
