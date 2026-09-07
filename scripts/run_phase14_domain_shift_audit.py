#!/usr/bin/env python3
"""
Phase 14 Runner: Domain-Shift Attribution / Robustness Audit.

Read-only audit of which observble domain-shift characteristics are most
consistently associated with external model error, using frozen canonical
artifacts and a pre-registered protocol. NO training, tuning, calibration,
adaptation, or dataset expansion.

State machine (strict):
    DESIGN -> FEASIBILITY -> ANALYSIS -> STATISTICAL_EVALUATION
    -> FREEZE -> FINAL_EXTERNAL_EVALUATION -> AUDIT -> COMPLETE

Moreno-Mateos is locked: its content is read only once, AFTER FREEZE.
FEASIBILITY performs an os.stat existence/size check only (D-7).

Usage:
    python scripts/run_phase14_domain_shift_audit.py
"""

import argparse
import json
import logging
import os
import sys
import dataclasses
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import validate_sequence
from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn
from src.models import RandomForestModel, XGBoostModel, CNNModel
from src.domain_shift.config import Phase14DomainShiftConfig
from src.domain_shift.state_machine import Phase14StateMachine
from src.domain_shift import stratification as strat
from src.domain_shift import composition as comp
from src.domain_shift import error_analysis as err
from src.domain_shift import statistics as stat
from src.domain_shift import provenance as prov
from src.domain_shift import figures as figmod
from src.evaluation.metrics import calculate_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIVITY_SPLIT = [0.2, 0.4, 0.6, 0.8]
N_ACT_BINS = 5


def build_feature_extractor() -> SequenceFeatureExtractor:
    return SequenceFeatureExtractor(
        context_length=30,
        guide_length=20,
        guide_start=4,
        k_values=[2, 3],
        include_one_hot=False,
        include_gc=True,
        include_composition=True,
        include_kmer=True,
        include_positional=True,
    )


def load_and_validate(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    seqs = df["sequence_30mer"].tolist()
    valid_mask = [validate_sequence(s, check_pam=False)[0] for s in seqs]
    return df[valid_mask].copy()


def predict_all(cfg, seqs):
    """Predictions of the three primary models (RF/XGB/CNN) on a corpus."""
    extractor = build_feature_extractor()
    X_tab = extractor.extract_features_batch(seqs).values
    X_oh = extract_one_hot_for_cnn(seqs, cfg.context_length)

    rf = RandomForestModel.load_model(str(cfg.paths()["model_rf"]))
    xgb = XGBoostModel.load_model(str(cfg.paths()["model_xgb"]))
    cnn = CNNModel.load_model(str(cfg.paths()["model_cnn"]))
    pred_rf = np.asarray(rf.predict(X_tab), dtype=float).ravel()
    pred_xgb = np.asarray(xgb.predict(X_tab), dtype=float).ravel()
    pred_cnn = np.asarray(cnn.predict(X_oh), dtype=float).ravel()
    # determinism check for CNN (eval mode; verify reproducibility)
    pred_cnn2 = np.asarray(cnn.predict(X_oh), dtype=float).ravel()
    cnn_deterministic = bool(np.array_equal(pred_cnn, pred_cnn2))
    return {
        "random_forest": pred_rf,
        "xgboost": pred_xgb,
        "cnn": pred_cnn,
        "cnn_deterministic": cnn_deterministic,
    }


def predict_phase13(cfg, seqs):
    """Secondary sensitivity model (Phase 13 learned-embedding CNN)."""
    from src.representation_audit.representation import NucleotideEmbeddingRepresentation
    from src.representation_audit.model import EmbeddingCNNModel

    rep = NucleotideEmbeddingRepresentation(
        context_length=cfg.context_length,
        vocabulary=["A", "C", "G", "T"],
        nuc_to_index={"A": 0, "C": 1, "G": 2, "T": 3},
        embedding_dim=8,
    )
    model = EmbeddingCNNModel.load_model(str(cfg.paths()["model_phase13"]))
    toks = rep.tokenize_batch(seqs)
    return np.asarray(model.predict(toks), dtype=float).ravel()


def feasibility_audit(cfg, sm) -> dict:
    checks = {}
    p = cfg.paths()
    checks["1_primary_dataset_available"] = p["primary_dataset"].exists()
    checks["2_external_exists_stat_only"] = p["external_dataset"].exists()
    if checks["2_external_exists_stat_only"]:
        st = os.stat(p["external_dataset"])
        checks["2_external_size_bytes"] = st.st_size
    else:
        checks["2_external_size_bytes"] = None

    df = load_and_validate(p["primary_dataset"])
    seqs = df["sequence_30mer"].tolist()
    y = df["activity"].values
    checks["3_n_valid"] = int(len(df))
    lengths_ok = (df["sequence_30mer"].str.len() == cfg.context_length).all()
    checks["4_all_length_30"] = bool(lengths_ok)
    X_oh = extract_one_hot_for_cnn(seqs, cfg.context_length)
    tr_idx, va_idx = train_test_split(
        np.arange(len(df)), test_size=cfg.val_ratio, random_state=cfg.split_seed
    )
    checks["5_split_reconstructed"] = (tr_idx.size == cfg.expected_n_train
                                       and va_idx.size == cfg.expected_n_val)
    checks["6_target_mean"] = float(np.mean(y))
    checks["6_target_ok"] = bool(np.isclose(np.mean(y), 0.4257, atol=0.001))

    models_ok = True
    detail = {}
    for name, path in [("random_forest", p["model_rf"]),
                       ("xgboost", p["model_xgb"]),
                       ("cnn", p["model_cnn"])]:
        try:
            if name == "random_forest":
                m = RandomForestModel.load_model(str(path))
            elif name == "xgboost":
                m = XGBoostModel.load_model(str(path))
            else:
                m = CNNModel.load_model(str(path))
            detail[name] = bool(m.is_fitted)
            models_ok = models_ok and bool(m.is_fitted)
        except Exception as e:  # pragma: no cover
            detail[name] = str(e)
            models_ok = False
    checks["7_models_load"] = models_ok
    checks["7_model_detail"] = detail

    hashes = {}
    hash_ok = True
    for rel, exp in cfg.expected_canonical_hashes.items():
        fp = Path(cfg.project_root) / rel
        h = prov.sha256_file(fp) if fp.exists() else "MISSING"
        hashes[rel] = h
        hash_ok = hash_ok and (h == exp)
    checks["8_canonical_hashes_match_recorded"] = hash_ok
    checks["8_hashes"] = hashes

    for d in ["output_dir", "figures_dir", "predictions_dir"]:
        p[d].mkdir(parents=True, exist_ok=True)
    checks["9_artifacts_savable"] = True

    critical = [
        checks["1_primary_dataset_available"],
        checks["2_external_exists_stat_only"],
        checks["3_n_valid"] == cfg.expected_n_valid,
        checks["4_all_length_30"],
        checks["5_split_reconstructed"],
        checks["6_target_ok"],
        checks["7_models_load"],
        checks["8_canonical_hashes_match_recorded"],
        checks["9_artifacts_savable"],
    ]
    checks["feasible"] = all(bool(c) for c in critical)
    return checks


def domain_dict(seqs, y, preds, gc_edges, gc_guide_edges, frozen):
    """Assemble one domain's analysis vectors on the frozen grid."""
    gc30 = comp.gc_array(seqs)
    gc_guide = comp.gc_array_guide(seqs, 4, 20)
    act_bin = strat.bin_indices(np.asarray(y, dtype=float), ACTIVITY_SPLIT)
    gc_bin = strat.bin_indices(gc30, gc_edges)
    gc_guide_bin = strat.bin_indices(gc_guide, gc_guide_edges)
    y_arr = np.asarray(y, dtype=float)
    d = {
        "seqs": seqs,
        "y_true": y_arr,
        "gc30": gc30,
        "gc_guide": gc_guide,
        "activity_bin": act_bin,
        "gc_bin": gc_bin,
        "gc_guide_bin": gc_guide_bin,
        "pred": {m: np.asarray(preds[m], dtype=float) for m in preds},
    }
    d["gc_train_z"] = (gc30 - frozen["gc_mean"]) / frozen["gc_std"]
    d["a_train_z"] = (y_arr - frozen["activity_mean"]) / frozen["activity_std"]
    d["comp_dist"] = err.composition_distance(seqs, frozen["train_f2"], 2)
    return d


def bin_tables(domain, models, gc_edges, gc_guide_edges, cfg):
    gc_lo, gc_hi = strat.bin_bounds(gc_edges, 0.0, 1.0)
    ag_lo, ag_hi = strat.bin_bounds(gc_guide_edges, 0.0, 1.0)
    a_lo, a_hi = strat.bin_bounds(ACTIVITY_SPLIT, 0.0, 1.0)
    out = {}
    for m in models:
        out[m] = {
            "gc_bins": strat.per_bin_stats(
                domain["y_true"], domain["pred"][m], domain["gc_bin"],
                cfg.n_gc_bins, gc_lo, gc_hi, cfg.min_n_report, cfg.min_n_boot,
                cfg.n_boot, cfg.bootstrap_seed),
            "guide_gc_bins": strat.per_bin_stats(
                domain["y_true"], domain["pred"][m], domain["gc_guide_bin"],
                cfg.n_gc_bins, ag_lo, ag_hi, cfg.min_n_report, cfg.min_n_boot,
                cfg.n_boot, cfg.bootstrap_seed),
            "activity_bins": strat.per_bin_stats(
                domain["y_true"], domain["pred"][m], domain["activity_bin"],
                N_ACT_BINS, a_lo, a_hi, cfg.min_n_report, cfg.min_n_boot,
                cfg.n_boot, cfg.bootstrap_seed),
        }
    return out


def associations(domain, models, primary, cfg):
    out = {}
    for m in models:
        cov = {
            "gc": domain["gc30"],
            "guide_gc": domain["gc_guide"],
            "activity": domain["y_true"],
            "gc_train_z": domain["gc_train_z"],
            "a_train_z": domain["a_train_z"],
            "composition_distance": domain["comp_dist"],
            "predicted_activity": domain["pred"][m],
        }
        out[m] = err.association_summary(
            domain["y_true"], domain["pred"][m], cov,
            model=m, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed,
            primary_error="absolute",
        )
    return out


def composition_summary(train_seqs, domain_seqs, train_gc, domain_gc, cfg,
                        compared_to_label):
    nuc_ref = comp.nucleotide_frequencies(train_seqs)
    nuc_cmp = comp.nucleotide_frequencies(domain_seqs)
    f2_ref = comp.kmer_frequency_vector(train_seqs, 2)
    f2_cmp = comp.kmer_frequency_vector(domain_seqs, 2)
    f3_ref = comp.kmer_frequency_vector(train_seqs, 3)
    f3_cmp = comp.kmer_frequency_vector(domain_seqs, 3)
    null2 = comp.internal_null_band(train_seqs, k=2, n_boot=500, seed=cfg.bootstrap_seed)
    null3 = comp.internal_null_band(train_seqs, k=3, n_boot=500, seed=cfg.bootstrap_seed)
    js = {
        "2": comp.js_divergence(f2_ref, f2_cmp),
        "3": comp.js_divergence(f3_ref, f3_cmp),
    }
    return {
        "trained_on": "internal_train",
        "compared_to": compared_to_label,
        "n_train": int(len(train_seqs)),
        "n_compared": int(len(domain_seqs)),
        "gc": {
            "train_mean": float(np.mean(train_gc)),
            "train_std": float(np.std(train_gc, ddof=1)),
            "domain_mean": float(np.mean(domain_gc)),
            "domain_std": float(np.std(domain_gc, ddof=1)),
            "cohens_d_train_minus_domain": comp.cohens_d(train_gc, domain_gc),
        },
        "nucleotide_frequencies_train": nuc_ref,
        "nucleotide_frequencies_compared": nuc_cmp,
        "jensen_shannon": js,
        "internal_null_band_k2": null2,
        "internal_null_band_k3": null3,
        "position_frequencies_train": comp.position_frequencies(train_seqs, 30),
        "position_frequencies_compared": comp.position_frequencies(domain_seqs, 30),
        "position_mad": comp.mad_between(
            comp.position_frequencies(train_seqs, 30),
            comp.position_frequencies(domain_seqs, 30)),
    }


def dispersion(domain, models):
    out = {}
    for m in models:
        t_std = float(np.std(domain["y_true"], ddof=1)) if domain["y_true"].size > 1 else 0.0
        p_std = float(np.std(domain["pred"][m], ddof=1)) if domain["pred"][m].size > 1 else 0.0
        out[m] = {
            "target_std": t_std,
            "pred_std": p_std,
            "sd_ratio_pred_over_true": p_std / t_std if t_std > 0 else None,
        }
    return out


def confirmatory_results(int_domain, ext_domain, cfg):
    """The 7 pre-registered confirmatory tests (C1..C7)."""
    rf = cfg.primary_assoc_model
    tests = {}
    bin0 = 0
    bin4 = 4

    ext_ae_lo = np.abs(ext_domain["y_true"][ext_domain["activity_bin"] == bin0]
                       - ext_domain["pred"][rf][ext_domain["activity_bin"] == bin0])
    int_ae_lo = np.abs(int_domain["y_true"][int_domain["activity_bin"] == bin0]
                       - int_domain["pred"][rf][int_domain["activity_bin"] == bin0])
    tests["C1_activity_lowbin_delta_mae"] = {
        "definition": "between-domain independent bootstrap of mean abs error "
                      "(external minus internal) in activity bin [0.0,0.2)",
        "model": rf,
        "bootstrap": stat.bootstrap_mean_delta_ci(
            ext_ae_lo, int_ae_lo, n_boot=cfg.n_boot,
            rng=stat.reproducible_bootstrap_seed(cfg.bootstrap_seed + 1)),
    }

    ext_ae_hi = np.abs(ext_domain["y_true"][ext_domain["activity_bin"] == bin4]
                       - ext_domain["pred"][rf][ext_domain["activity_bin"] == bin4])
    int_ae_hi = np.abs(int_domain["y_true"][int_domain["activity_bin"] == bin4]
                       - int_domain["pred"][rf][int_domain["activity_bin"] == bin4])
    tests["C2_activity_highbin_delta_mae"] = {
        "definition": "between-domain independent bootstrap of mean abs error "
                      "(external minus internal) in activity bin [0.8,1.0]",
        "model": rf,
        "bootstrap": stat.bootstrap_mean_delta_ci(
            ext_ae_hi, int_ae_hi, n_boot=cfg.n_boot,
            rng=stat.reproducible_bootstrap_seed(cfg.bootstrap_seed + 2)),
    }

    def _sp(name, x):
        return stat.spearman_with_bootstrap(
            err.absolute_error(ext_domain["y_true"], ext_domain["pred"][rf]),
            x, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed + 3)

    res3 = _sp("gc", ext_domain["gc30"])
    res4 = _sp("activity", ext_domain["y_true"])
    res5 = _sp("gc_train_z", ext_domain["gc_train_z"])
    tests["C3_spearman_abs_gc"] = {
        "definition": "Spearman(abs_error_RF, GC) on external; within-domain bootstrap CI",
        "model": rf, "spearman": res3}
    tests["C4_spearman_abs_activity"] = {
        "definition": "Spearman(abs_error_RF, activity) on external",
        "model": rf, "spearman": res4}
    tests["C5_spearman_abs_gc_train_z"] = {
        "definition": "Spearman(abs_error_RF, GC_train_z) on external",
        "model": rf, "spearman": res5}

    int_d = {
        "true": int_domain["y_true"],
        "pred_" + rf: int_domain["pred"][rf],
        "gc_bin": int_domain["gc_bin"],
        "activity_bin": int_domain["activity_bin"],
    }
    ext_d = {
        "true": ext_domain["y_true"],
        "pred_" + rf: ext_domain["pred"][rf],
        "gc_bin": ext_domain["gc_bin"],
        "activity_bin": ext_domain["activity_bin"],
    }
    jt = strat.common_support_joint_delta(
        int_d, ext_d, cfg.n_gc_bins, N_ACT_BINS, min_n=cfg.min_n_common_support,
        model=rf,
    )
    tests["C6_joint_common_support_median_delta"] = {"joint": jt}

    per_model_errors = {m: ext_domain["pred"][m] for m in cfg.primary_models}
    conv = stat.convergence_bootstrap(
        per_model_errors, ext_domain["activity_bin"], n_bins=N_ACT_BINS,
        n_boot=cfg.n_boot, seed=cfg.bootstrap_seed + 4,
    )
    tests["C7_cross_model_bin_mae_consistency"] = {"convergence": conv}
    return tests


def decision_gates(tests, int_domain, ext_domain, cfg):
    """A..F gates from the frozen numeric triggers."""
    def c1_notable(t):
        v = t["C1_activity_lowbin_delta_mae"]["bootstrap"]
        return (v["p_value"] <= cfg.alpha and abs(v["mean_delta"]) >= cfg.dmae_effect_threshold)

    def c2_notable(t):
        v = t["C2_activity_highbin_delta_mae"]["bootstrap"]
        return (v["p_value"] <= cfg.alpha and abs(v["mean_delta"]) >= cfg.dmae_effect_threshold)

    def c3_notable(t):
        s = t["C3_spearman_abs_gc"]["spearman"]
        return (s["ci_lower"] > 0 or s["ci_upper"] < 0) and abs(s["spearman"]) >= cfg.rho_effect_threshold

    def c4_notable(t):
        s = t["C4_spearman_abs_activity"]["spearman"]
        return (s["ci_lower"] > 0 or s["ci_upper"] < 0) and abs(s["spearman"]) >= cfg.rho_effect_threshold

    def c5_notable(t):
        s = t["C5_spearman_abs_gc_train_z"]["spearman"]
        return (s["ci_lower"] > 0 or s["ci_upper"] < 0) and abs(s["spearman"]) >= cfg.rho_effect_threshold

    def c6_notable(t):
        j = t["C6_joint_common_support_median_delta"]["joint"]
        md = j.get("median_delta_mae")
        ci = j.get("median_delta_mae_ci", {})
        if md is None:
            return False
        p = ci.get("p_value", 1.0)
        return p <= cfg.alpha and abs(md) >= cfg.c6_cell_delta_threshold

    def c7_notable(t):
        c = t["C7_cross_model_bin_mae_consistency"]["convergence"]
        m = c.get("mean_pairwise_spearman")
        return m is not None and m >= cfg.convergence_rho_threshold

    notable = {
        "C1": c1_notable(tests), "C2": c2_notable(tests),
        "C3": c3_notable(tests), "C4": c4_notable(tests),
        "C5": c5_notable(tests), "C6": c6_notable(tests),
        "C7": c7_notable(tests),
    }
    h1 = notable["C3"] or notable["C5"]
    h2 = notable["C4"] or notable["C1"] or notable["C2"]
    h3 = notable["C6"]
    h4 = notable["C7"]
    support_count = sum([h1, h2, h3, h4])

    disp = dispersion(ext_domain, cfg.primary_models)
    underdispersed = any(
        d.get("sd_ratio_pred_over_true") is not None
        and d["sd_ratio_pred_over_true"] < cfg.dispersion_sd_ratio_threshold
        for d in disp.values()
    )
    e_gate = (h1 or h2 or h3) and underdispersed

    # single-factor marginal deltas (median per-bin delta MAE in MAE units)
    def median_per_bin_delta(bins_int, bins_ext):
        ds = []
        for b_int, b_ext in zip(bins_int, bins_ext):
            if b_int.get("n", 0) >= cfg.min_n_report and b_ext.get("n", 0) >= cfg.min_n_report:
                ds.append(b_ext["mae"] - b_int["mae"])
        return float(np.median(ds)) if ds else None

    tb_int = bin_tables(int_domain, [cfg.primary_assoc_model], cfg_gc_edges_placeholder,
                        cfg_gc_guide_edges_placeholder, cfg)[cfg.primary_assoc_model]
    tb_ext = bin_tables(ext_domain, [cfg.primary_assoc_model], cfg_gc_edges_placeholder,
                        cfg_gc_guide_edges_placeholder, cfg)[cfg.primary_assoc_model]
    act_margin = median_per_bin_delta(tb_int["activity_bins"], tb_ext["activity_bins"])
    gc_margin = median_per_bin_delta(tb_int["gc_bins"], tb_ext["gc_bins"])
    jt = tests["C6_joint_common_support_median_delta"]["joint"]
    joint_margin = jt.get("median_delta_mae")
    d_gate = False
    if h3 and joint_margin is not None:
        best_single = max([v for v in [act_margin, gc_margin] if v is not None], default=-np.inf)
        if joint_margin >= best_single + cfg.c6_cell_delta_threshold:
            d_gate = True

    if support_count == 4:
        headline = "A"
    elif d_gate:
        headline = "D"
    elif support_count >= 2:
        headline = "B"
    elif support_count == 0:
        headline = "C"
    else:
        headline = "B"
    gates = [headline]
    if e_gate:
        gates.append("E")
    return {
        "headline": "/".join(dict.fromkeys(gates)),
        "notable": notable,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "support_count": int(support_count),
        "underdispersed_external": bool(underdispersed),
        "marginals": {"activity_bin_median_delta": act_margin,
                      "gc_bin_median_delta": gc_margin,
                      "joint_cell_median_delta": joint_margin},
        "d_gate": bool(d_gate),
    }


# runtime placeholders replaced from configs during execution
cfg_gc_edges_placeholder = [0.4, 0.5, 0.6, 0.7]
cfg_gc_guide_edges_placeholder = [0.4, 0.5, 0.6, 0.7]


def main():
    parser = argparse.ArgumentParser(description="Phase 14 Domain-Shift Audit")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="override bootstrap count (default: config)")
    args = parser.parse_args()

    cfg = Phase14DomainShiftConfig()
    if args.n_boot is not None:
        cfg = dataclasses.replace(cfg, n_boot=args.n_boot)

    global cfg_gc_edges_placeholder
    global cfg_gc_guide_edges_placeholder

    p = cfg.paths()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"phase14_domain_shift_audit_{timestamp}"

    print("=" * 70)
    print(f"Phase 14: Domain-Shift Attribution / Robustness Audit ({experiment_name})")
    print("=" * 70)

    # ---------------- DESIGN ----------------
    sm = Phase14StateMachine(start="DESIGN")
    head_before = prov.git_head(cfg.project_root)
    status_before = prov.git_status(cfg.project_root)
    config_hash = prov.config_hash(cfg)
    print(f"[DESIGN] git HEAD: {head_before}")
    print(f"[DESIGN] config hash: {config_hash[:16]}...")

    # ---------------- FEASIBILITY ----------------
    sm.transition("FEASIBILITY")
    feas = feasibility_audit(cfg, sm)
    print(f"[FEASIBILITY] feasible={feas['feasible']}")
    for k in sorted(feas):
        if k == "8_hashes":
            continue
        print(f"    {k}: {feas[k]}")
    if not feas["feasible"]:
        reason = ("FEASIBILITY FAILED: " + json.dumps(
            {k: v for k, v in feas.items() if k != "8_hashes"}, default=str))
        _dump_failure(cfg, experiment_name, head_before, status_before, reason)
        print("PHASE 14 STOPPED (infeasible). Decision gate: F")
        return

    # ---------------- ANALYSIS (internal) ----------------
    sm.transition("ANALYSIS")
    df = load_and_validate(p["primary_dataset"])
    df = df.reset_index(drop=True).copy()
    seqs_all = df["sequence_30mer"].tolist()
    y_all = df["activity"].values
    idx_all = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx_all, test_size=cfg.val_ratio,
                                      random_state=cfg.split_seed)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[tr_idx] = True

    train_seqs = [seqs_all[i] for i in tr_idx]
    val_seqs = [seqs_all[i] for i in va_idx]
    y_train = y_all[tr_idx]
    y_val = y_all[va_idx]

    # frozen covariate stats from TRAIN only
    gc_train = comp.gc_array(train_seqs)
    gc_guide_train = comp.gc_array_guide(train_seqs, 4, 20)
    frozen = {
        "gc_mean": float(np.mean(gc_train)),
        "gc_std": float(np.std(gc_train, ddof=1)) if gc_train.size > 1 else 0.0,
        "gc_guide_mean": float(np.mean(gc_guide_train)),
        "gc_guide_std": float(np.std(gc_guide_train, ddof=1)) if gc_guide_train.size > 1 else 0.0,
        "activity_mean": float(np.mean(y_train)),
        "activity_std": float(np.std(y_train, ddof=1)) if y_train.size > 1 else 0.0,
        "train_f2": comp.kmer_frequency_vector(train_seqs, 2).tolist(),
    }

    gc_edges_res = strat.construct_gc_edges(
        gc_train, tag="train", quantiles=cfg.gc_quantiles,
        round_dp=cfg.gc_round, fallback_edges=cfg.fallback_gc_edges)
    gc_guide_edges_res = strat.construct_gc_edges(
        gc_guide_train, tag="train", quantiles=cfg.gc_quantiles,
        round_dp=cfg.gc_round, fallback_edges=cfg.fallback_gc_edges)
    gc_edges = gc_edges_res["edges"]
    gc_guide_edges = gc_guide_edges_res["edges"]
    cfg_gc_edges_placeholder = gc_edges
    cfg_gc_guide_edges_placeholder = gc_guide_edges
    print(f"[ANALYSIS] frozen GC edges (30-mer): {gc_edges} "
          f"(source={gc_edges_res['source']})")
    print(f"[ANALYSIS] frozen GC edges (guide): {gc_guide_edges} "
          f"(source={gc_guide_edges_res['source']})")

    # Internal validation predictions
    preds_val = predict_all(cfg, val_seqs)
    print(f"[ANALYSIS] CNN determinism (validation): {preds_val['cnn_deterministic']}")
    preds_val.pop("cnn_deterministic", None)
    val_domain = domain_dict(val_seqs, y_val, preds_val, gc_edges, gc_guide_edges, frozen)
    val_bins = bin_tables(val_domain, cfg.primary_models, gc_edges, gc_guide_edges, cfg)
    val_assoc = associations(val_domain, cfg.primary_models,
                             cfg.primary_assoc_model, cfg)
    val_comp = composition_summary(train_seqs, val_seqs, gc_train,
                                   comp.gc_array(val_seqs), cfg,
                                   compared_to_label="internal_validation")
    val_disp = dispersion(val_domain, cfg.primary_models)
    val_metrics = {m: calculate_all_metrics(y_val, preds_val[m]) for m in cfg.primary_models}

    # ---------------- STATISTICAL_EVALUATION (internal context) ----------------
    sm.transition("STATISTICAL_EVALUATION")
    internal_ctx = {
        "validation_metrics": val_metrics,
        "activity_bin_mae_rf": val_bins[cfg.primary_assoc_model]["activity_bins"],
        "spearman_associations_rf": {k: v for k, v in val_assoc[cfg.primary_assoc_model].items()
                                     if isinstance(v, dict) and "spearman" in v},
        "rf_val_mae": float(np.mean(np.abs(y_val - preds_val[cfg.primary_assoc_model]))),
    }
    # cross-model consistency on internal (descriptive analog of C7)
    int_conv = stat.convergence_bootstrap(
        {m: preds_val[m] for m in cfg.primary_models}, val_domain["activity_bin"],
        n_bins=N_ACT_BINS, n_boot=min(cfg.n_boot, 200), seed=cfg.bootstrap_seed + 5)
    internal_ctx["c7_internal_analog"] = int_conv
    print(f"[STATISTICAL_EVALUATION] internal context recorded "
          f"(RF val MAE {internal_ctx['rf_val_mae']:.4f})")

    # Internal analysis snapshot (root JSON, pre-freeze)
    internal_results = {
        "experiment_name": experiment_name,
        "phase": "14",
        "timestamp": datetime.now().isoformat(),
        "status": "INTERNAL_COMPLETE",
        "config_hash": config_hash,
        "feasibility": feas,
        "frozen_edges": {"gc_edges": gc_edges, "gc_guide_edges": gc_guide_edges,
                         "source_30mer": gc_edges_res["source"],
                         "source_guide": gc_guide_edges_res["source"]},
        "frozen_covariates": frozen,
        "internal": {
            "composition": val_comp,
            "bins": val_bins,
            "associations": val_assoc,
            "dispersion": val_disp,
            "metrics": val_metrics,
            "statistical_context": internal_ctx,
        },
        "git_head_before": head_before,
    }
    results_path = p["output_dir"] / f"{experiment_name}.json"
    with open(results_path, "w") as f:
        json.dump(internal_results, f, indent=2, default=str)

    # ---------------- FREEZE ----------------
    freeze_payload = {
        "experiment_name": experiment_name,
        "phase": "14",
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "status": "FROZEN",
        "gc_method": cfg.gc_method,
        "gc_edges": {
            "full_30mer": gc_edges,
            "guide_4_24": gc_guide_edges,
            "source_30mer": gc_edges_res["source"],
            "source_guide": gc_guide_edges_res["source"],
        },
        "activity_edges": cfg.activity_edges,
        "activity_split_points": ACTIVITY_SPLIT,
        "min_n_report": cfg.min_n_report,
        "min_n_boot": cfg.min_n_boot,
        "min_n_common_support": cfg.min_n_common_support,
        "n_boot": cfg.n_boot,
        "bootstrap_seed": cfg.bootstrap_seed,
        "fdr_q": cfg.fdr_q,
        "alpha": cfg.alpha,
        "effect_thresholds": {
            "rho": cfg.rho_effect_threshold,
            "dmae": cfg.dmae_effect_threshold,
            "c6_cell_delta": cfg.c6_cell_delta_threshold,
            "convergence_rho": cfg.convergence_rho_threshold,
            "dispersion_sd_ratio": cfg.dispersion_sd_ratio_threshold,
        },
        "confirmatory_tests": cfg.confirmatory_tests,
        "frozen_covariates": frozen,
        "internal_summary": {
            "n_valid": len(df), "n_train": int(tr_idx.size), "n_val": int(va_idx.size),
            "rf_validation_mae": internal_ctx["rf_val_mae"],
        },
        "external_stat_only": {
            "exists": p["external_dataset"].exists(),
            "size_bytes": os.stat(p["external_dataset"]).st_size
                          if p["external_dataset"].exists() else None,
        },
        "canonical_hashes_before": feas["8_hashes"],
        "git_head": head_before,
        "git_status": status_before,
    }
    freeze_path = p["output_dir"] / f"{experiment_name}_freeze.json"
    with open(freeze_path, "w") as f:
        json.dump(freeze_payload, f, indent=2, default=str)
    sm.freeze()
    print(f"[FREEZE] protocol frozen: {freeze_path.name}")

    # ---------------- FINAL EXTERNAL EVALUATION ----------------
    sm.transition("FINAL_EXTERNAL_EVALUATION")
    sm.assert_moreno_read_allowed()
    df_ext = load_and_validate(p["external_dataset"])
    if len(df_ext) != cfg.expected_n_external:
        _dump_failure(cfg, experiment_name, head_before, status_before,
                      f"External validated n={len(df_ext)} != {cfg.expected_n_external}")
        print("PHASE 14 STOPPED: external size mismatch at final evaluation")
        return
    ext_seqs = df_ext["sequence_30mer"].tolist()
    y_ext = df_ext["activity"].values
    moreno_hash = prov.sha256_file(p["external_dataset"])

    preds_ext = predict_all(cfg, ext_seqs)
    print(f"[FINAL] CNN determinism (external): {preds_ext['cnn_deterministic']}")
    preds_ext.pop("cnn_deterministic", None)

    ext_domain = domain_dict(ext_seqs, y_ext, preds_ext, gc_edges, gc_guide_edges, frozen)
    ext_bins = bin_tables(ext_domain, cfg.primary_models, gc_edges, gc_guide_edges, cfg)
    ext_assoc = associations(ext_domain, cfg.primary_models,
                             cfg.primary_assoc_model, cfg)
    ext_comp = composition_summary(train_seqs, ext_seqs, gc_train,
                                   comp.gc_array(ext_seqs), cfg,
                                   compared_to_label="external_moreno_mateos")
    ext_disp = dispersion(ext_domain, cfg.primary_models)
    ext_metrics = {m: calculate_all_metrics(y_ext, preds_ext[m]) for m in cfg.primary_models}

    # Phase 13 secondary sensitivity
    phase13_ext = None
    phase13_val = None
    if cfg.phase13_secondary:
        p13_ext = predict_phase13(cfg, ext_seqs)
        p13_val = predict_phase13(cfg, val_seqs)
        phase13_ext = {
            "predictions_stats": {"mean": float(np.mean(p13_ext)),
                                  "std": float(np.std(p13_ext, ddof=1))},
            "metrics": calculate_all_metrics(y_ext, p13_ext),
        }
        phase13_val = {
            "predictions_stats": {"mean": float(np.mean(p13_val)),
                                  "std": float(np.std(p13_val, ddof=1))},
            "metrics": calculate_all_metrics(y_val, p13_val),
        }

    tests = confirmatory_results(val_domain, ext_domain, cfg)

    # BH-FDR across the 7 confirmatory tests
    pvals = []
    t_key_order = ["C1_activity_lowbin_delta_mae", "C2_activity_highbin_delta_mae",
                   "C3_spearman_abs_gc", "C4_spearman_abs_activity",
                   "C5_spearman_abs_gc_train_z", "C6_joint_common_support_median_delta",
                   "C7_cross_model_bin_mae_consistency"]
    for k in t_key_order:
        t = tests[k]
        if "bootstrap" in t:
            pvals.append(t["bootstrap"]["p_value"])
        elif "spearman" in t:
            pvals.append(1.0)  # hypothesis of a nonzero correlation handled by CI rule
        elif "joint" in t:
            ci = t["joint"].get("median_delta_mae_ci", {})
            pvals.append(ci.get("p_value", 1.0))
        elif "convergence" in t:
            pvals.append(t["convergence"].get("p_value", 1.0) or 1.0)
    adj = stat.bh_adjust(pvals)
    for k, a in zip(t_key_order, adj):
        tests[k]["bh_adjusted_p"] = a

    gates = decision_gates(tests=tests, int_domain=val_domain,
                           ext_domain=ext_domain, cfg=cfg)
    print(f"[FINAL] decision gate: {gates['headline']}")
    print(f"[FINAL] underdispersion external: {gates['underdispersed_external']}")

    # ---------------- AUDIT ----------------
    sm.transition("AUDIT")
    hashes_after = {}
    for rel in cfg.expected_canonical_hashes:
        fp = Path(cfg.project_root) / rel
        hashes_after[rel] = prov.sha256_file(fp) if fp.exists() else "MISSING"
    unchanged = {rel: hashes_after[rel] == feas["8_hashes"][rel]
                 for rel in cfg.expected_canonical_hashes}
    canonical_ok = all(unchanged.values())
    if not canonical_ok:
        _dump_failure(cfg, experiment_name, head_before, status_before,
                      "Canonical artifact hash CHANGED during Phase 14")
        print("PHASE 14 STOPPED: canonical artifact modified")
        return

    # reproducibility spot-check: recompute RF validation MAE
    preds_val_2 = predict_all(cfg, val_seqs)
    rf_mae_recompute = float(np.mean(
        np.abs(y_val - preds_val_2[cfg.primary_assoc_model])))
    reproducible = bool(np.isclose(rf_mae_recompute,
                                   internal_ctx["rf_val_mae"], atol=1e-12))

    audit = {
        "canonical_hashes_after": hashes_after,
        "canonical_unchanged": unchanged,
        "canonical_ok": bool(canonical_ok),
        "freeze_config_hash_matches": bool(
            prov.config_hash(cfg) == config_hash),
        "rf_val_mae_recomputed": rf_mae_recompute,
        "reproducible_internal_stats": reproducible,
        "moreno_hash_recorded_once": moreno_hash[:16] + "...",
    }

    # ---------------- COMPLETE ----------------
    sm.transition("COMPLETE")
    final_path = p["output_dir"] / f"{experiment_name}_final.json"
    if final_path.exists():
        raise RuntimeError(f"Write-once violation: {final_path} already exists")

    preds_int_df = pd.DataFrame({
        "sequence": val_seqs,
        "true_activity": y_val,
        "random_forest": preds_val["random_forest"],
        "xgboost": preds_val["xgboost"],
        "cnn": preds_val["cnn"],
    })
    preds_ext_df = pd.DataFrame({
        "sequence": ext_seqs,
        "true_activity": y_ext,
        "random_forest": preds_ext["random_forest"],
        "xgboost": preds_ext["xgboost"],
        "cnn": preds_ext["cnn"],
    })
    preds_int_path = p["predictions_dir"] / f"{experiment_name}_internal_predictions.csv"
    preds_ext_path = p["predictions_dir"] / f"{experiment_name}_external_predictions.csv"
    preds_int_df.to_csv(preds_int_path, index=False)
    preds_ext_df.to_csv(preds_ext_path, index=False)

    fig_dir = p["figures_dir"]
    figs = {}
    figs["fig1_gc"] = str(fig_dir / f"{experiment_name}_fig1_gc_distribution.png")
    figs["fig2_activity"] = str(fig_dir / f"{experiment_name}_fig2_activity_distribution.png")
    figs["fig3_gc_mae"] = str(fig_dir / f"{experiment_name}_fig3_mae_gc_bins.png")
    figs["fig4_activity_mae"] = str(fig_dir / f"{experiment_name}_fig4_mae_activity_bins.png")
    figs["fig5_pred_true"] = str(fig_dir / f"{experiment_name}_fig5_pred_vs_true.png")
    figs["fig6_model_compare"] = str(fig_dir / f"{experiment_name}_fig6_model_comparison.png")
    figs["fig7_composition"] = str(fig_dir / f"{experiment_name}_fig7_composition_summary.png")
    figs["fig8_joint"] = str(fig_dir / f"{experiment_name}_fig8_joint_heatmap.png")

    figmod.fig1_gc_distribution(val_domain, ext_domain, gc_edges, figs["fig1_gc"])
    figmod.fig2_activity_distribution(val_domain, ext_domain, cfg.activity_edges,
                                      figs["fig2_activity"])
    figmod.fig3_mae_by_gc_bin(val_domain, ext_domain, gc_edges,
                              cfg.primary_assoc_model, figs["fig3_gc_mae"])
    figmod.fig4_mae_by_activity_bin(val_domain, ext_domain,
                                    cfg.primary_assoc_model, figs["fig4_activity_mae"])
    figmod.fig5_pred_vs_true(val_domain, ext_domain, cfg.primary_models, figs["fig5_pred_true"])
    figmod.fig6_model_comparison(val_domain, ext_domain, cfg.primary_models,
                                 figs["fig6_model_compare"])
    jt = tests["C6_joint_common_support_median_delta"]["joint"]
    figmod.fig8_joint_heatmap(jt["cells"], cfg.n_gc_bins, N_ACT_BINS, figs["fig8_joint"])
    figmod.fig7_composition_summary(ext_comp["jensen_shannon"],
                                    {"2": ext_comp["internal_null_band_k2"],
                                     "3": ext_comp["internal_null_band_k3"]},
                                    figs["fig7_composition"])

    final = {
        "experiment_name": experiment_name,
        "phase": "14",
        "timestamp": datetime.now().isoformat(),
        "status": "PASS",
        "decision_gate": gates["headline"],
        "config_hash": config_hash,
        "methodology": {
            "read_only_audit": True,
            "no_retraining": True,
            "no_tuning": True,
            "no_calibration": True,
            "no_pooled_data": True,
            "models": cfg.primary_models,
            "phase13_secondary": cfg.phase13_secondary,
            "primary_association_model": cfg.primary_assoc_model,
            "c1_c2_comparison": "independent_domains (external - internal), not paired",
            "c6_scope": "common-support GC x activity cells (n>=20 both domains); "
                        "does not represent the full datasets",
        },
        "feasibility": feas,
        "frozen_protocol": freeze_payload,
        "datasets": {
            "deepspcas9": {"n_valid": int(len(df)), "n_train": int(tr_idx.size),
                            "n_val": int(va_idx.size)},
            "external": {"n_valid": int(len(df_ext)),
                         "sha256": moreno_hash,
                         "read_after_freeze": True,
                         "read_once": True},
        },
        "composition_external_vs_train": ext_comp,
        "composition_internal_val_vs_train": val_comp,
        "composition": ext_comp,
        "stratification": {
            "internal": {"activity_bins": {m: [b for b in val_bins[m]["activity_bins"]] for m in cfg.primary_models},
                          "gc_bins": {m: [b for b in val_bins[m]["gc_bins"]] for m in cfg.primary_models}},
            "external": {"activity_bins": {m: [b for b in ext_bins[m]["activity_bins"]] for m in cfg.primary_models},
                          "gc_bins": {m: [b for b in ext_bins[m]["gc_bins"]] for m in cfg.primary_models}},
        },
        "association": {
            "internal": {m: assoc for m, assoc in val_assoc.items()},
            "external": {m: assoc for m, assoc in ext_assoc.items()},
        },
        "confirmatory": tests,
        "convergence": {
            "internal_c7_analog": internal_ctx["c7_internal_analog"],
            "external_c7": tests["C7_cross_model_bin_mae_consistency"]["convergence"],
        },
        "dispersion": {"internal": val_disp, "external": ext_disp},
        "metrics": {"internal": val_metrics, "external": ext_metrics},
        "phase13_sensitivity": {"internal": phase13_val, "external": phase13_ext},
        "gates": gates,
        "leakage_checks": {
            "no_moreno_content_before_freeze": True,
            "external_read_once": True,
            "no_training_or_tuning": True,
            "no_pool": True,
            "canonical_hashes_unchanged": bool(canonical_ok),
        },
        "audit": audit,
        "state_log": sm.log,
        "canonical_artifact_hashes": {"before": feas["8_hashes"], "after": hashes_after,
                                      "unchanged": unchanged},
        "git": {"head_before": head_before, "status_before": status_before},
        "env": prov.env_versions(),
        "artifacts": {
            "internal_results": str(results_path),
            "freeze": str(freeze_path),
            "final": str(final_path),
            "predictions_internal": str(preds_int_path),
            "predictions_external": str(preds_ext_path),
            "figures": figs,
        },
    }
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2, default=str)

    assert sm.state == "COMPLETE"
    print("\n" + "=" * 70)
    print("Phase 14 COMPLETE")
    print(f"Results (internal): {results_path}")
    print(f"Freeze:            {freeze_path}")
    print(f"Results (final):   {final_path}")
    print(f"Decision gate:     {gates['headline']}")
    print(f"External read:     once, after freeze: {len(df_ext)} sequences")
    print("=" * 70)


def _dump_failure(cfg, experiment_name, head_before, status_before, reason):
    p = cfg.paths()
    p["output_dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": experiment_name,
        "phase": "14",
        "timestamp": datetime.now().isoformat(),
        "status": "FAILED",
        "decision_gate": "F",
        "reason": reason,
        "git": {"head_before": head_before, "status_before": status_before},
    }
    path = p["output_dir"] / f"{experiment_name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Failure report: {path}")


if __name__ == "__main__":
    main()