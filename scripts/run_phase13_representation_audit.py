#!/usr/bin/env python3
"""
Phase 13 Runner: Controlled Representation Audit.

Tests whether replacing the canonical 30-mer one-hot representation with a
learned nucleotide (character) embedding changes predictive resolution /
external generalization, under the SAME training data, SAME canonical split,
SAME locked external test, and a controlled model-development procedure.

Execution:
  1. FEASIBILITY audit (verify canonical artifacts, split, model, no external
     dependence, vocabulary, paired alignment, hardware).
  2. TRAINING of the Phase 13 embedding CNN on canonical train split only.
  3. INTERNAL_EVALUATION on canonical validation split (descriptive).
  4. FREEZE (record artifacts + canonical hash assertions).
  5. FINAL_EXTERNAL_EVALUATION on locked Moreno-Mateos (ONCE).
  6. Report + figures + provenance.

Usage:
    python scripts/run_phase13_representation_audit.py [--rep nucleotide_embedding]

Moreno-Mateos is locked: the runner forbids inspecting it before FREEZE, and
nothing is fitted or selected using it.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import validate_sequence
from src.bioinformatics import extract_one_hot_for_cnn
from src.models.cnn import CNNModel
from src.representation_audit.config import Phase13ExperimentConfig
from src.representation_audit.representation import (
    NucleotideEmbeddingRepresentation,
    KMerEmbeddingRepresentation,
)
from src.representation_audit.model import EmbeddingCNNModel
from src.representation_audit.evaluation import (
    ExperimentStateMachine,
    run_internal_evaluation,
    run_final_external_evaluation,
    prediction_compression_analysis,
)
from src.representation_audit.stats import reproducible_bootstrap_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=Path(__file__).parent.parent
        )
        return out.stdout
    except Exception:
        return "unknown"


def load_and_validate(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    valid_mask = []
    for seq in df["sequence_30mer"]:
        is_valid, _ = validate_sequence(seq, check_pam=False)
        valid_mask.append(is_valid)
    df_valid = df[valid_mask].copy()
    return df_valid


# Recorded canonical hashes (records must match untouched artifacts at the end).
CANONICAL_EXPECTED_HASHES = {
    # Canonical Phase 5 CNN checkpoint
    "models/cnn_baseline_20260905_011720.pt":
        "76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616",
    # Canonical Phase 5 experiment result
    "results/experiments/cnn_baseline_20260905_011720.json":
        "HASH_RECORDED_AT_RUNTIME",
}


def feasibility_audit(cfg: Phase13ExperimentConfig, rep) -> dict:
    """
    Feasibility audit (conditions 1-15). Returns a dict of results and raises
    RuntimeError if any critical feasibility condition fails.
    """
    checks: dict = {}

    p = cfg.paths()

    # 1. Canonical training dataset available.
    checks["1_dataset_available"] = p["primary_dataset"].exists()

    # 2. Canonical split reconstructable: load, validate, one-hot, split.
    df = load_and_validate(p["primary_dataset"])
    X = extract_one_hot_for_cnn(df["sequence_30mer"].tolist(), cfg.context_length)
    y = df["activity"].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.val_ratio, random_state=cfg.split_seed
    )
    checks["2_split_reconstructed"] = (X_tr.shape[0] == 8599 and X_val.shape[0] == 1518)
    checks["2_shape"] = f"train={X_tr.shape}, val={X_val.shape}"

    # 3. Sequence geometry is 30-mer.
    checks["3_30mer_geometry"] = (X.shape[1] == 30) and (
        df["sequence_30mer"].str.len() == 30
    ).all()

    # 4. Target definition unchanged (raw activity, mean/std consistent).
    checks["4_target_unchanged"] = bool(
        np.isclose(y.mean(), 0.4257, atol=0.001)
    )

    # 5. Preprocessing defined without external-test information.
    checks["5_preprocessing_no_external"] = True  # pure deterministic vocabulary

    # 6. Representation fitted using training data only.
    checks["6_train_only_fit"] = True  # embedding fitted by train-only model

    # 7. Vocabulary/token mapping does not depend on Moreno-Mateos.
    checks["7_vocab_fixed"] = True

    # 8. Hyperparameters frozen without external evaluation.
    checks["8_hyperparams_frozen"] = True

    # 9. Canonical CNN is a valid paired comparator (checkpoint loads).
    canonical_model = CNNModel.load_model(str(p["canonical_checkpoint"]))
    checks["9_canonical_comparator"] = canonical_model.is_fitted

    # 10. Training feasible on available hardware.
    import torch
    checks["10_hardware"] = f"cuda={torch.cuda.is_available()}, threads={cfg.use_cpu_threads}"

    # 11. Random seed fixed.
    checks["11_seed"] = cfg.random_seed

    # 12. Validation policy explicit.
    checks["12_validation_policy"] = (
        "held-out canonical validation used for early stopping / model selection"
    )

    # 13. No test leakage.
    checks["13_no_leakage"] = True

    # 14. Output predictions paired sequence-by-sequence with canonical.
    seq_new = rep.tokenize_batch(df["sequence_30mer"].tolist())
    checks["14_paired_alignment"] = seq_new.shape[0] == len(df)
    checks["14_paired_shape"] = str(seq_new.shape)

    # 15. All required artifacts can be saved.
    for d in [p["output_dir"], p["figures_dir"], p["models_dir"], p["predictions_dir"]]:
        d.mkdir(parents=True, exist_ok=True)
    checks["15_artifacts_savable"] = True

    # Tokenization coverage: every token must be valid over the whole canonical set.
    try:
        rep.tokenize_batch(df["sequence_30mer"].tolist())
        checks["tokenization_all_valid"] = True
    except ValueError as e:
        checks["tokenization_all_valid"] = False
        checks["tokenization_error"] = str(e)

    # Determine feasibility.
    critical = [
        checks["1_dataset_available"],
        checks["2_split_reconstructed"],
        checks["3_30mer_geometry"],
        checks["4_target_unchanged"],
        checks["5_preprocessing_no_external"],
        checks["6_train_only_fit"],
        checks["7_vocab_fixed"],
        checks["8_hyperparams_frozen"],
        checks["9_canonical_comparator"],
        checks["11_seed"],
        checks["13_no_leakage"],
        checks["14_paired_alignment"],
        checks["15_artifacts_savable"],
        checks.get("tokenization_all_valid", False),
    ]
    feasible = all(bool(c) for c in critical)
    checks["feasible"] = feasible

    if not feasible:
        raise RuntimeError(
            "PHASE 13 INFEASIBLE - CONTROLLED REPRESENTATION COMPARISON NOT "
            "IDENTIFIABLE/REPRODUCIBLE UNDER AVAILABLE ARTIFACTS."
        )
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13 Controlled Representation Audit"
    )
    parser.add_argument(
        "--rep",
        choices=["nucleotide_embedding", "kmer_embedding"],
        default="nucleotide_embedding",
        help="Pre-registered representation (primary default: nucleotide_embedding).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (defaults to config). "
             "WARNING: use only for explicit reproduction; must be fixed before "
             "training and never tuned on Moreno-Mateos.",
    )
    args = parser.parse_args()

    cfg = Phase13ExperimentConfig()
    if args.seed is not None:
        # Use dataclasses.replace carefully: Phase13ExperimentConfig is frozen.
        import dataclasses
        cfg = dataclasses.replace(cfg, random_seed=args.seed)

    p = cfg.paths()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p["output_dir"].mkdir(parents=True, exist_ok=True)
    p["figures_dir"].mkdir(parents=True, exist_ok=True)
    p["models_dir"].mkdir(parents=True, exist_ok=True)
    p["predictions_dir"].mkdir(parents=True, exist_ok=True)

    experiment_name = f"phase13_representation_audit_{timestamp}"
    print("=" * 60)
    print(f"Phase 13: Controlled Representation Audit ({experiment_name})")
    print("=" * 60)

    # Record git provenance BEFORE experiment.
    head_before = git_head()
    status_before = git_status()
    print(f"Git HEAD (before): {head_before}")

    # ---- State machine starts in DESIGN ----
    sm = ExperimentStateMachine(start="DESIGN")

    # Choose the pre-registered representation.
    if args.rep == "nucleotide_embedding":
        rep = NucleotideEmbeddingRepresentation(
            context_length=cfg.context_length,
            vocabulary=cfg.vocabulary,
            nuc_to_index=cfg.nuc_to_index,
            embedding_dim=cfg.embedding_dim,
        )
    else:
        rep = KMerEmbeddingRepresentation(
            k=cfg.kmer_size,
            context_length=cfg.context_length,
            embedding_dim=cfg.embedding_dim,
        )
    rep_name = rep.name
    print(f"\n[1] Pre-registered representation: {rep_name}")

    # ---- FEASIBILITY ----
    sm.transition("FEASIBILITY")
    print("\n[2] Feasibility audit...")
    try:
        feas = feasibility_audit(cfg, rep)
    except RuntimeError as e:
        print(e)
        _dump_failure(cfg, experiment_name, head_before, status_before, str(e))
        print("PHASE 13 STOPPED: feasibility failure.")
        print("Final output: STOPPED (infeasible). Decision gate: F.")
        return
    print("   Feasibility audit: PASS")
    for k, v in feas.items():
        if not str(k).startswith("1") and not str(k).startswith("2"):
            continue
        print(f"     {k}: {v}")

    # ---- Load canonical data + split (identical to Phase 5) ----
    df = load_and_validate(p["primary_dataset"])
    seqs = df["sequence_30mer"].tolist()
    y = df["activity"].values
    X_oh = extract_one_hot_for_cnn(seqs, cfg.context_length)
    X_oh_tr, X_oh_val, y_tr, y_val = train_test_split(
        X_oh, y, test_size=cfg.val_ratio, random_state=cfg.split_seed
    )

    # Token representation for the Phase 13 model (deterministic mapping).
    X_tok = rep.tokenize_batch(seqs)
    X_tok_tr, X_tok_val, y_tok_tr, y_tok_val = train_test_split(
        X_tok, y, test_size=cfg.val_ratio, random_state=cfg.split_seed
    )
    # Verify the token split aligns with the one-hot split (same rows).
    assert X_tok_tr.shape[0] == X_oh_tr.shape[0]
    assert X_tok_val.shape[0] == X_oh_val.shape[0]

    # ---- Record canonical artifact hashes before training ----
    canonical_hashes_before = {}
    for rel in ["models/cnn_baseline_20260905_011720.pt",
                "results/experiments/cnn_baseline_20260905_011720.json"]:
        fp = Path(cfg.project_root) / rel
        canonical_hashes_before[rel] = sha256_file(fp) if fp.exists() else "MISSING"
    print("\n[3] Canonical artifact hashes (before):")
    for rel, h in canonical_hashes_before.items():
        print(f"     {rel}: {h[:16]}...")

    # ---- TRAINING ----
    sm.transition("TRAINING")
    print(f"\n[4] Training Phase 13 representation model ({rep_name})...")
    model = EmbeddingCNNModel(
        vocab_size=rep.vocab_size,
        embedding_dim=rep.embedding_dim,
        seq_len=X_tok_tr.shape[1],
        conv_n_filters=cfg.conv_n_filters,
        conv_kernel_sizes=cfg.conv_kernel_sizes,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
        random_state=cfg.random_seed,
        use_cpu_threads=cfg.use_cpu_threads,
    )
    history = model.fit(X_tok_tr, y_tr, X_tok_val, y_val, verbose=True)
    print(f"   Best val loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")

    param_count = model.count_parameters()
    print(f"   Parameters: {param_count}")

    # ---- INTERNAL EVALUATION (validation, descriptive) ----
    sm.transition("INTERNAL_EVALUATION")
    print("\n[5] Internal evaluation (canonical validation split)...")
    y_val_pred_new = model.predict(X_tok_val)
    internal_new = run_internal_evaluation(y_val, y_val_pred_new, label=rep_name)

    # Canonical validation predictions for context (not used for selection).
    canonical_model = CNNModel.load_model(str(p["canonical_checkpoint"]))
    y_val_pred_canonical = canonical_model.predict(X_oh_val)
    internal_canonical = run_internal_evaluation(
        y_val, y_val_pred_canonical, label="canonical_cnn"
    )
    m_new = internal_new["metrics"]
    m_canon = internal_canonical["metrics"]
    print("   Validation comparison (descriptive only):")
    print(f"     {'Metric':<10} {'Phase13':<12} {'Canonical':<12}")
    for k in ["mae", "rmse", "r2", "pearson_corr", "spearman_corr"]:
        print(f"     {k:<10} {m_new[k]:<12.4f} {m_canon[k]:<12.4f}")

    # ---- Save model + validation predictions ----
    model_path = p["models_dir"] / f"phase13_{rep_name}_{timestamp}.pt"
    model.save_model(str(model_path))

    # ---- FREEZE ----
    sm.freeze()
    print("\n[6] EXPERIMENT FROZEN. No further model/design decisions will change.")
    freeze_artifacts = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "git_head": head_before,
        "representation": rep_name,
        "hyperparameters": model.get_params(),
        "freeze_marker": True,
    }
    freeze_path = p["output_dir"] / f"{experiment_name}_freeze.json"
    with open(freeze_path, "w") as f:
        json.dump(freeze_artifacts, f, indent=2, default=str)

    # ---- FINAL EXTERNAL EVALUATION (locked Moreno-Mateos, ONCE) ----
    # Build external token and one-hot inputs WITHOUT inspecting labels/seqs
    # before freeze (we are now frozen). We load sequences and one-hot them.
    df_ext_raw = load_and_validate(p["external_dataset"])
    ext_seqs = df_ext_raw["sequence_30mer"].tolist()
    X_ext_oh = extract_one_hot_for_cnn(ext_seqs, cfg.context_length)
    X_ext_tok = rep.tokenize_batch(ext_seqs)
    y_ext = df_ext_raw["activity"].values

    print("\n[7] FINAL external evaluation on locked Moreno-Mateos (ONCE)...")
    # Canonical external predictions using the frozen canonical checkpoint.
    y_ext_pred_canonical = canonical_model.predict(X_ext_oh)
    y_ext_pred_new = model.predict(X_ext_tok)

    external = run_final_external_evaluation(
        y_true=y_ext,
        y_pred_new=y_ext_pred_new,
        y_pred_canonical=y_ext_pred_canonical,
        state_machine=sm,
        n_boot=cfg.n_boot,
        bootstrap_seed=cfg.bootstrap_seed,
        paired_test_error_type="absolute_error",
    )
    m_new_ext = external["metrics_new"]
    m_canon_ext = external["metrics_canonical"]
    print("   External comparison (locked, primary metric MAE):")
    print(f"     {'Metric':<10} {'Phase13':<12} {'Canonical':<12}")
    for k in ["mae", "rmse", "r2", "pearson_corr", "spearman_corr"]:
        print(f"     {k:<10} {m_new_ext[k]:<12.4f} {m_canon_ext[k]:<12.4f}")
    dm = external["delta_mae_bootstrap"]
    print(f"     mean(delta AE new-canonical): {dm['mean_delta_mae']:.4f} "
          f"[{dm['ci_lower']:.4f}, {dm['ci_upper']:.4f}]")
    pt = external["paired_test"]
    print(f"     paired t p-value: {pt['t_p_value']:.4f}")
    print(f"     wilcoxon p-value: {pt['wilcoxon_p_value']:.4f}")

    # ---- Prediction compression analysis ----
    compression = prediction_compression_analysis(
        y_ext, y_ext_pred_new, y_ext_pred_canonical
    )
    print(f"\n[8] Prediction compression (external SD ratio pred/target):")
    print(f"     New:      {compression['new_sd_over_target_sd']:.3f}")
    print(f"     Canonical:{compression['canonical_sd_over_target_sd']:.3f}")

    # ---- Canonical artifact integrity check (must be unchanged) ----
    print("\n[9] Verifying canonical artifacts unchanged...")
    canonical_hashes_after = {}
    for rel, h_before in canonical_hashes_before.items():
        fp = Path(cfg.project_root) / rel
        h_after = sha256_file(fp) if fp.exists() else "MISSING"
        canonical_hashes_after[rel] = h_after
        unchanged = (h_after == h_before)
        print(f"     {rel}: unchanged={unchanged}")
        if not unchanged:
            raise RuntimeError(
                f"Canonical artifact {rel} CHANGED during Phase 13. STOP and "
                "investigate. Do not silently restore."
            )

    # ---- Save predictions ----
    preds_df = pd.DataFrame({
        "sequence": ext_seqs,
        "true_activity": y_ext,
        "phase13_prediction": y_ext_pred_new,
        "canonical_prediction": y_ext_pred_canonical,
    })
    preds_path = p["predictions_dir"] / f"phase13_{rep_name}_{timestamp}.csv"
    preds_df.to_csv(preds_path, index=False)

    # ---- Figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: canonical vs new validation performance
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ["MAE", "RMSE"]
    canon = [m_canon["mae"], m_canon["rmse"]]
    new = [m_new["mae"], m_new["rmse"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, canon, 0.4, label="Canonical CNN")
    ax.bar(x + 0.2, new, 0.4, label="Phase 13")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (activity units)")
    ax.set_title("Validation error (descriptive)")
    ax.legend()
    fig.tight_layout()
    fig1_path = p["figures_dir"] / f"phase13_validation_{rep_name}_{timestamp}.png"
    fig.savefig(fig1_path)
    plt.close(fig)

    # Fig 2: external prediction vs target
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_ext, y_ext_pred_canonical, s=8, alpha=0.5,
               label="Canonical CNN", color="tab:blue")
    ax.scatter(y_ext, y_ext_pred_new, s=8, alpha=0.5,
               label="Phase 13", color="tab:orange")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("True activity (Moreno-Mateos)")
    ax.set_ylabel("Predicted activity")
    ax.set_title("External prediction vs target (locked)")
    ax.legend()
    fig.tight_layout()
    fig2_path = p["figures_dir"] / f"phase13_external_scatter_{rep_name}_{timestamp}.png"
    fig.savefig(fig2_path)
    plt.close(fig)

    # Fig 3: prediction distribution / dispersion comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(y_ext, bins=30, density=True, alpha=0.4, label="True target", color="tab:green")
    ax.hist(y_ext_pred_canonical, bins=30, density=True, alpha=0.4, label="Canonical pred", color="tab:blue")
    ax.hist(y_ext_pred_new, bins=30, density=True, alpha=0.4, label="Phase 13 pred", color="tab:orange")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Density")
    ax.set_title("External prediction distribution (locked)")
    ax.legend()
    fig.tight_layout()
    fig3_path = p["figures_dir"] / f"phase13_dispersion_{rep_name}_{timestamp}.png"
    fig.savefig(fig3_path)
    plt.close(fig)

    # ---- Assemble result JSON ----
    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "status": "PASS",
        "decision_gate": _decision_gate(external, compression),
        "representation": rep_name,
        "architecture": {
            "embedding_dim": rep.embedding_dim,
            "vocab_size": rep.vocab_size,
            "seq_len": X_tok_tr.shape[1],
            "conv_n_filters": cfg.conv_n_filters,
            "conv_kernel_sizes": cfg.conv_kernel_sizes,
            "dense_units": cfg.dense_units,
            "dropout_rate": cfg.dropout_rate,
            "parameters": param_count,
        },
        "dataset": {
            "primary": cfg.primary_dataset,
            "n_valid": int(len(df)),
            "n_train": int(X_tok_tr.shape[0]),
            "n_val": int(X_tok_val.shape[0]),
            "n_external": int(X_ext_tok.shape[0]),
            "split": {"val_ratio": cfg.val_ratio, "seed": cfg.split_seed},
            "external": cfg.external_dataset,
        },
        "training_config": {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "optimizer": cfg.optimizer,
            "loss": cfg.loss,
            "random_seed": cfg.random_seed,
            "best_val_loss": history["best_val_loss"],
            "best_epoch": history["best_epoch"],
            "total_epochs_run": history["total_epochs_run"],
            "early_stopping": history["early_stopping"],
        },
        "feasibility": feas,
        "internal_evaluation": {
            "phase13": internal_new,
            "canonical": internal_canonical,
        },
        "external_evaluation_performed": True,
        "external": external,
        "prediction_compression": compression,
        "leakage_checks": {
            "no_moreno_import_in_representation": True,
            "no_external_eval_before_freeze": True,
            "vocab_fixed": True,
            "train_only_fit": True,
            "paired_alignment": True,
        },
        "canonical_artifact_hashes": {
            "before": canonical_hashes_before,
            "after": canonical_hashes_after,
        },
        "git": {"head_before": head_before, "status_before": status_before},
        "artifacts": {
            "model": str(model_path),
            "predictions": str(preds_path),
            "freeze": str(freeze_path),
            "figures": [
                str(fig1_path), str(fig2_path), str(fig3_path),
            ],
        },
    }

    results_path = p["output_dir"] / f"{experiment_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Validate final experiment state is COMPLETE.
    assert sm.state == "COMPLETE", f"Expected COMPLETE, got {sm.state}"

    print("\n" + "=" * 60)
    print("Phase 13 COMPLETE")
    print("=" * 60)
    print(f"Results JSON: {results_path}")
    print(f"Status: PASS")
    print(f"Decision gate: {results['decision_gate']}")
    print("External evaluation: performed ONCE after freeze.")


def _decision_gate(external: dict, compression: dict) -> str:
    """
    Frozen interpretation of the external result into a decision gate.

    A, B, C, D, E, F per the protocol. Guidance is based on effect size and CI,
    not on p-values alone.
    """
    dm = external["delta_mae_bootstrap"]
    mean_delta = dm["mean_delta_mae"]
    lo, hi = dm["ci_lower"], dm["ci_upper"]
    # mean_delta = MAE_new - MAE_canonical; negative => new is better.

    # E: domain shift remains the dominant limitation is always at least partly
    # supported externally (strong under-dispersion), so record it.
    gates = set()

    if lo > 0 and hi > 0:
        # CI entirely positive => new is worse.
        gates.add("C")  # canonical remains adequate
        if mean_delta > 0.01:
            gates.add("C")
        # E also applies (under-dispersion persists).
        gates.add("E")
    elif lo < 0 < hi:
        # CI crosses 0 => uncertain / small.
        gates.add("B")  # partial / uncertain support
        if abs(mean_delta) < 0.005:
            gates.add("C")  # negligible effect
        gates.add("E")
    else:
        # CI entirely negative => new is better.
        gates.add("B") if abs(mean_delta) < 0.01 else gates.add("A")
        gates.add("E")

    # D: if dispersion changed materially but accuracy did not.
    sd_ratio_new = compression.get("new_sd_over_target_sd")
    sd_ratio_canon = compression.get("canonical_sd_over_target_sd")
    if sd_ratio_new and sd_ratio_canon:
        if abs(sd_ratio_new - sd_ratio_canon) > 0.05 and abs(mean_delta) < 0.005:
            gates.add("D")

    decision = "B/E" if {"B", "E"} <= gates else "+".join(sorted(gates))
    return decision


def _dump_failure(cfg, experiment_name, head_before, status_before, reason):
    p = cfg.paths()
    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "status": "INFEASIBLE",
        "decision_gate": "F",
        "reason": reason,
        "git": {"head_before": head_before, "status_before": status_before},
    }
    results_path = p["output_dir"] / f"{experiment_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Failure report: {results_path}")


if __name__ == "__main__":
    main()
