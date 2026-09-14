"""Phase 18B deterministic preregistration and leakage audit infrastructure.

No model is imported, constructed, fitted, or evaluated here. Labels are read
only by the existing allowlisted loader and fingerprinted without transformation.
Training statistics, batches, bootstrap samples and predictions are NOT made.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from src.dataset_integration.label_compatibility import (
    EXPECTED_HASHES,
    integrity_snapshot,
    load_phase18a_inputs,
    prohibit_locked_path,
    require_expected_integrity,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("deepspcas9", "crispron_xiang_luo")
SEEDS = (42, 43, 44, 45, 46)
SPLIT_SEED = 42
VERSION = "phase18b-multidomain-v1"
PHASE17 = (
    "results/phase17g_r1_data_sufficiency_reassessment_20260911_233749.json"
)
PHASE18A = "results/phase18a_cross_dataset_compatibility_20260912_160313.json"
REPORT = "docs/phase18b_multidomain_experimental_protocol.md"
PARTITIONS = ("train", "validation", "internal_test", "bridge", "quarantine")
EXPECTED_COUNTS = {
    DOMAINS[0]: dict(zip(PARTITIONS, (7040, 1510, 1526, 41, 0))),
    DOMAINS[1]: dict(zip(PARTITIONS, (7465, 1520, 1559, 41, 7))),
}


def serialize(payload: object) -> str:
    """Canonical UTF-8 JSON representation, independent of mapping order."""
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def fingerprint(payload: object) -> str:
    return hashlib.sha256(serialize(payload).encode("utf-8")).hexdigest()


def safe_path(path: str | Path) -> Path:
    """Reject locked names before resolution, and reject resolved aliases."""
    path = Path(path)
    prohibit_locked_path(path)
    resolved = path.resolve()
    prohibit_locked_path(resolved)
    return resolved


@contextmanager
def audit_only_guard():
    """Fail before Python opens locked data or writes protected artifacts.

    Audit hooks also intercept ordinary builtins/io opens inside dependencies.
    The profile guard rejects fitting/gradient entry points before their body.
    This is an execution guard for this Python process, not an OS sandbox.
    """
    state = {"active": True}
    approved = {safe_path(ROOT / p) for p in EXPECTED_HASHES}
    data_root = ROOT / "data"
    model_root = ROOT / "models"

    def on_audit(event, args):
        if not state["active"] or event != "open":
            return
        name, mode, flags = args
        if not isinstance(name, (str, bytes, Path)):
            return
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        path = safe_path(name)
        write_flags = (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        writing = bool(flags & write_flags)
        writing = writing or bool(mode and any(c in mode for c in "wax+"))
        if path.is_relative_to(data_root) or path.is_relative_to(model_root):
            if writing or path not in approved:
                raise PermissionError("Phase 18B data/model access prohibited")
        if writing and (
            path == ROOT / "config.yaml"
            or path.is_relative_to(ROOT / "results" / "predictions")
            or path.is_relative_to(ROOT / "results" / "experiments")
        ):
            raise PermissionError(
                "Phase 18B canonical output overwrite prohibited"
            )

    previous_profile = sys.getprofile()

    def no_training(frame, event, arg):
        if event == "call" and frame.f_code.co_name in {
            "fit",
            "partial_fit",
            "train",
            "backward",
            "step",
        }:
            module = frame.f_globals.get("__name__", "")
            if module.startswith(
                (
                    "torch",
                    "sklearn",
                    "xgboost",
                    "src.models",
                    "src.calibration",
                )
            ):
                raise RuntimeError(
                    "Phase 18B model training/calibration prohibited"
                )
        if previous_profile:
            previous_profile(frame, event, arg)

    sys.addaudithook(on_audit)
    sys.setprofile(no_training)
    try:
        yield
    finally:
        state["active"] = False
        sys.setprofile(previous_profile)


def verify_integrity(expected=None) -> dict:
    """Reuse Phase 18A's corrected hashes, with resolved-path protection."""
    expected = EXPECTED_HASHES if expected is None else expected
    if not expected:
        raise ValueError("Empty integrity manifest")
    paths = {}
    for p, value in expected.items():
        resolved = safe_path(ROOT / p)
        if resolved != (ROOT / p).absolute():
            raise PermissionError("Integrity input alias prohibited")
        paths[str(resolved)] = value
    snapshot = integrity_snapshot(paths)
    require_expected_integrity(snapshot)
    return {p: snapshot[str(safe_path(ROOT / p))] for p in expected}


def group_key(sequence: str, unit: str = "spacer20") -> str:
    """Exact forward grouping; never reconstruct or reverse complement DNA."""
    if not isinstance(sequence, str) or len(sequence) != 30:
        raise ValueError("Expected canonical 30-mer geometry")
    if not set(sequence) <= set("ACGT"):
        raise ValueError("Expected uppercase unambiguous DNA")
    if unit == "30mer":
        return sequence
    if unit == "spacer20":
        return sequence[4:24]
    raise ValueError("Unsupported grouping unit")


def split_for_group(group: str, seed: int = SPLIT_SEED) -> str:
    if type(seed) is not int or seed != SPLIT_SEED:
        raise ValueError("Split seed is locked")
    digest = hashlib.sha256(f"{VERSION}|{seed}|{group}".encode("ascii"))
    value = int(digest.hexdigest(), 16)
    # Exact integer comparison avoids platform-specific float boundaries.
    if value * 100 < 70 * 2**256:
        return "train"
    if value * 100 < 85 * 2**256:
        return "validation"
    return "internal_test"


def make_split_manifest(domain_sequences, raw_bridge, canonical_bridge):
    """Write only hashed sequence identifiers and membership, never labels."""
    if set(domain_sequences) != set(DOMAINS):
        raise ValueError("Exactly two development domains required")
    raw_bridge = set(raw_bridge)
    canonical_bridge = set(canonical_bridge)
    if not canonical_bridge <= raw_bridge:
        raise ValueError("Canonical bridge must be a subset of raw bridge")
    held_groups = {group_key(s) for s in raw_bridge}
    rows = []
    for domain in DOMAINS:
        sequences = list(domain_sequences[domain])
        if len(sequences) != len(set(sequences)):
            raise ValueError("Unexpected within-domain 30-mer duplicate")
        for sequence in sorted(sequences):
            group = group_key(sequence)
            split = (
                ("bridge" if sequence in canonical_bridge else "quarantine")
                if group in held_groups
                else split_for_group(group)
            )
            rows.append(
                {
                    "domain": domain,
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode()
                    ).hexdigest(),
                    "group_sha256": hashlib.sha256(group.encode()).hexdigest(),
                    "split": split,
                }
            )
    validate_leakage(
        rows,
        held_groups={
            hashlib.sha256(g.encode()).hexdigest() for g in held_groups
        },
    )
    return rows


def validate_leakage(rows, held_groups):
    """Detect leakage within and across domains, including bridge relatives."""
    seen = set()
    groups = {}
    exact = {}
    for row in rows:
        domain, split = row["domain"], row["split"]
        if domain not in DOMAINS or split not in PARTITIONS:
            raise ValueError("Unknown domain or split")
        identity = (domain, row["sequence_sha256"])
        if identity in seen:
            raise ValueError("Duplicate manifest identity")
        seen.add(identity)
        group = row["group_sha256"]
        holding = split in {"bridge", "quarantine"}
        if holding != (group in held_groups):
            raise ValueError("Bridge exclusion/holding policy violated")
        partition = "held_out" if holding else split
        for key, mapping in ((group, groups), (row["sequence_sha256"], exact)):
            if key in mapping and mapping[key] != partition:
                raise ValueError("Cross-domain or within-domain split leakage")
            mapping[key] = partition


def policies() -> dict:
    """All numerical choices are fixed before Phase 18C results exist."""
    return {
        "hypothesis": (
            "A shared sequence encoder can improve sequence-linked SpCas9 "
            "indel-activity prediction while separate heads retain each assay's "
            "measurement mapping; benefit must not meaningfully harm either domain."
        ),
        "architecture": {
            "primary": "shared_encoder_domain_specific_heads",
            "shared": "Conv1d(4,64,k=5/7/9,padding=k//2), ReLU, MaxPool1d(2), "
            "AdaptiveAvgPool1d(1), concat(192), Linear(192,64), "
            "ReLU, Dropout(0.3)",
            "heads": "one Linear(64,1) per domain; explicit required domain ID",
            "raw_output": "mu_train_d + sd_train_d * head_d(encoder(X))",
            "routing": "Route only to the observed domain head; no inferred domain, "
            "no cross-head loss, unknown domain fails",
            "shared_parameter_n": 17920,
            "single_head_parameter_n": 65,
            "single_domain_total_n": 17985,
            "multidomain_total_n": 18050,
            "output_buffers": "per-domain train-only mean and population SD; "
            "not learned, not calibration, no Phase 18B fit",
        },
        "label": {
            "primary": "domain_specific_raw_regression",
            "targets": {
                DOMAINS[0]: "activity",
                DOMAINS[1]: "HEK293T_indel_freq_avg_d8_d10",
            },
            "mutate_labels": False,
            "clipping": False,
            "output_preconditioning": "train_only_affine_output",
            "interpretation": "Raw fraction A and raw percent B remain separate "
            "endpoints; affine output is algebraically equivalent "
            "to domain-local target standardization, not harmonization",
        },
        "loss": {
            "name": "train_variance_normalized_raw_mse",
            "formula": "L_d = mean((yhat_d-y_d)^2) / var_train_d; "
            "L_D = 0.5*L_A + 0.5*L_B",
            "variance_source": "training_only_ddof_0_float64",
            "variance_floor": None,
            "degenerate_policy": "stop if SD is nonfinite or <= 1e-12 native units",
            "adaptive_weighting": False,
            "domain_weights": {DOMAINS[0]: 0.5, DOMAINS[1]: 0.5},
            "single_domain_weight": 1.0,
        },
        "split": {
            "unit": "spacer20",
            "seed": SPLIT_SEED,
            "algorithm": "SHA256(version|42|forward_spacer20), integer thresholds",
            "ratios": {
                "train": 0.70,
                "validation": 0.15,
                "internal_test": 0.15,
            },
            "label_stratification": False,
            "reseed_for_counts": False,
            "bridge": "hold all raw48 spacer groups; canonical41 paired diagnostics; "
            "remaining7 B-only quarantine; all group relatives held out",
            "old_canonical_split": "historical only; all Phase 18C baselines refitted "
            "from scratch on new experimental train partitions",
            "B_eligibility": "accepted10592 geometry-valid rows; no additional "
            "homopolymer filter; 10544 nonoverlapping development rows",
        },
        "batching": {
            "batch_size": 32,
            "multidomain_composition": {DOMAINS[0]: 16, DOMAINS[1]: 16},
            "single_domain_composition": "32 from that domain",
            "steps_per_epoch": "ceil(max(n_train_A,n_train_B)/16) for every CNN arm",
            "sampling": "domain-local sorted IDs, NumPy PCG64 SeedSequence "
            "[run_seed,domain_index,epoch,cycle]; permute without "
            "replacement, cycle/re-permute on exhaustion, full batches",
            "epoch": "fixed optimizer-step budget, not one pass for single-domain arms",
            "optimizer_steps": "one update per batch, both D losses accumulated first",
            "fairness": "same steps and total examples per CNN; D has half as many "
            "examples per individual domain; log steps/exposures/time",
        },
        "training": {
            "seeds": list(SEEDS),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_epochs": 100,
            "patience": 10,
            "min_improvement": 0.0001,
            "validation_metric": "same domain-normalized MSE objective as training; "
            "D equal-domain mean; whole validation partitions",
            "checkpoint": "lowest unrounded validation objective, earliest tie; "
            "patience reset only when loss < significant_best-0.0001; "
            "significant_best updated only on reset; evaluate each epoch",
            "initialization": "fresh PyTorch default Conv/Linear initialization; "
            "encoder RNG=run_seed; independent head RNG="
            "run_seed+1000+domain_index; paired across arms",
            "determinism": "CPU float32 model, four threads, deterministic algorithms, "
            "Python/NumPy/Torch seeded, num_workers=0; log versions",
            "hyperparameter_search": [],
            "refit_on_train_plus_validation": False,
        },
        "tabular": {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "criterion": "squared_error",
                "bootstrap": True,
                "n_jobs": 4,
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "n_jobs": 4,
                "tree_method": "hist",
                "device": "cpu",
            },
            "random_state": "run_seed",
            "early_stopping": False,
            "tuning": False,
            "feature_n": 197,
        },
        "metrics": {
            "primary": ["spearman"],
            "guardrail": ["rmse"],
            "secondary": ["pearson", "r2", "mae", "rmse"],
            "domain_aggregation": "none; conjunctive domain-level decision",
            "seed_aggregation": "arithmetic mean of per-seed metrics, not ensemble predictions",
            "undefined": "UNSTABLE_RESULT; no silent zero correlation or dropped seed",
        },
        "statistics": {
            "primary_comparisons": ["D_vs_A_on_A", "D_vs_B_on_B"],
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 42,
            "rng": "numpy.random.Generator(PCG64(42))",
            "nominal_confidence": 0.95,
            "decision_interval_confidence": 0.9875,
            "decision_quantiles": [0.00625, 0.99375],
            "multiplicity": "Bonferroni four endpoints: rho and relative RMSE gain "
            "in each domain; familywise nominal 95%",
            "resampling": "each replicate: draw 5 paired seed indices, then sorted "
            "test spacer groups with replacement separately A then B; "
            "retain all rows of each drawn group and use same draw "
            "for both arms and all seeds; recompute metrics per seed, "
            "then mean the paired deltas; percentile CI, linear quantiles",
            "invalid_replicate": "no replacement draws; any undefined primary "
            "replicate -> UNSTABLE_RESULT and report count",
            "secondary": "95% descriptive paired CIs; no additional confirmatory claims",
            "limitations": "single fixed split, five seeds, approximate hierarchical "
            "bootstrap, not a prospective power guarantee or gene holdout",
        },
        "outcomes": {
            "rho_preservation_margin": 0.01,
            "rho_minimum_benefit": 0.01,
            "relative_rmse_harm_margin": 0.02,
            "rho_seed_range_limit": 0.10,
            "rmse_gain_seed_range_limit": 0.20,
            "rmse_gain_definition": "(RMSE_baseline-RMSE_D)/RMSE_baseline per seed",
            "precedence": [
                "PROTOCOL_VIOLATION",
                "UNSTABLE_RESULT",
                "NEGATIVE_TRANSFER",
                "MULTI_DOMAIN_BENEFIT",
                "NO_CLEAR_BENEFIT",
            ],
            "benefit": "both domains rho CI lower >= -0.01 and RMSE gain CI lower "
            ">= -0.02; at least one domain mean delta rho >= 0.01 with "
            "CI lower > 0; in both domains >=4/5 seeds preserve both margins",
            "negative_transfer": "any domain rho CI upper < -0.01 OR relative "
            "RMSE gain CI upper < -0.02; directional harm "
            "flags also reported even when CI is inconclusive",
            "unstable": "missing/nonfinite seed or metric or invalid CI/bootstrap; "
            "any domain delta-rho range >=0.10 or RMSE-gain range >=0.20",
            "no_clear_benefit": (
                "all remaining valid, stable outcomes, including "
                "preservation alone or inconclusive harm"
            ),
            "threshold_basis": "prospective project practical tolerances, not "
            "assay-derived biological cutoffs; no external-data basis",
        },
        "bridge_analysis": {
            "role": "secondary diagnostic only; previously inspected in Phase 18A, "
            "not a pristine confirmatory test; no selection or calibration",
            "population": "41 exact canonical pairs, evaluated after all checkpoints frozen",
            "metrics": [
                "head_to_head_spearman",
                "head_to_head_kendall",
                "head_to_own_label_spearman",
                "head_to_own_label_pearson",
                "mean_absolute_percentile_rank_disagreement",
            ],
            "comparison": "D head pair versus paired A/B single-domain predictions; "
            "report delta agreement and own-label fidelity; consensus "
            "alone cannot establish correctness",
            "ranks": "average ties, (rank-1)/(41-1), within bridge only",
            "inference": "descriptive per-seed values and ranges; no success gate, "
            "no cross-domain raw-error comparison, no additional fitting",
        },
        "stops": [
            "canonical integrity mismatch",
            "Moreno access attempt",
            "split leakage",
            "bridge or quarantine leakage",
            "nondeterministic preprocessing or split fingerprint mismatch",
            "label fingerprint mutation",
            "30-mer geometry change",
            "canonical artifact/prediction overwrite attempt",
            "unexpected dataset/component/partition counts",
            "unregistered arm/seed/hyperparameter",
            "nonfinite training objective",
        ],
    }


def experiment_matrix(policy=None):
    policy = policies() if policy is None else policy
    arms = [
        (
            "A",
            "cnn",
            [DOMAINS[0]],
            "single_domain_encoder_head",
            "Canonical-domain reference and no-sharing ablation",
        ),
        (
            "B",
            "cnn",
            [DOMAINS[1]],
            "single_domain_encoder_head",
            "Independent new-domain reference and no-sharing ablation",
        ),
        (
            "D",
            "cnn",
            list(DOMAINS),
            "shared_encoder_domain_specific_heads",
            policy["hypothesis"],
        ),
    ]
    for family in ("random_forest", "xgboost"):
        for suffix, domain in zip(("A", "B"), DOMAINS):
            arms.append(
                (
                    f"{family}_{suffix}",
                    family,
                    [domain],
                    "single_domain_197_features",
                    "Fixed tabular domain reference",
                )
            )
    rows = []
    for arm, family, domains, architecture, hypothesis in arms:
        for seed in SEEDS:
            rows.append(
                {
                    "experiment_id": f"18C_{arm}_s{seed}",
                    "arm": arm,
                    "model_family": family,
                    "architecture": architecture,
                    "training_domains": domains,
                    "prediction_heads": domains,
                    "input": (
                        "one_hot_30x4"
                        if family == "cnn"
                        else "197_sequence_features"
                    ),
                    "target": {
                        d: policy["label"]["targets"][d] for d in domains
                    },
                    "domain_handling": (
                        "explicit domain routing"
                        if arm == "D"
                        else "independent single-domain fit"
                    ),
                    "label_policy": policy["label"]["primary"],
                    "loss_policy": (
                        policy["loss"]["name"]
                        if family == "cnn"
                        else "native_squared_error_raw_target"
                    ),
                    "domain_weighting": {d: 1 / len(domains) for d in domains},
                    "split_policy": "locked_spacer20_70_15_15_manifest",
                    "bridge_policy": "raw48_groups_held_canonical41_diagnostic",
                    "seed": seed,
                    "hypothesis": hypothesis,
                    "primary_metrics": policy["metrics"]["primary"],
                    "secondary_metrics": policy["metrics"]["secondary"],
                    "expected_benefit": (
                        "complementary sequence information"
                        if arm == "D"
                        else "domain-specific reference performance"
                    ),
                    "major_risk": (
                        "negative transfer"
                        if arm == "D"
                        else "limited representation or domain-specific overfitting"
                    ),
                    "interpretation": (
                        "confirmatory D vs respective CNN baseline"
                        if family == "cnn"
                        else "descriptive fixed baseline; no neural routing"
                    ),
                }
            )
    return rows


def validate_policy(policy, matrix):
    """Reject drift from the preregistration, including numerical policy changes."""
    reference = policies()
    for section in reference:
        if policy.get(section) != reference[section]:
            raise ValueError(f"Locked {section} policy mismatch")
    if set(policy) != set(reference):
        raise ValueError("Unexpected policy section")
    if matrix != experiment_matrix(reference):
        raise ValueError("Locked experiment matrix/seed mismatch")


def classify_outcome(evidence, violations=()):
    """Apply future decision rules to synthetic/externally supplied summaries.

    No metrics or predictions are computed. Each domain needs five paired rho
    deltas, five paired relative RMSE gains and their decision CIs.
    """
    if violations:
        return "PROTOCOL_VIOLATION"
    p = policies()["outcomes"]
    if set(evidence) != set(DOMAINS):
        return "UNSTABLE_RESULT"
    for record in evidence.values():
        if record.get("seeds") != list(SEEDS) or record.get(
            "invalid_bootstrap_n", 0
        ):
            return "UNSTABLE_RESULT"
        for metric, limit in (
            ("rho", p["rho_seed_range_limit"]),
            ("rmse_gain", p["rmse_gain_seed_range_limit"]),
        ):
            values = record.get(metric, [])
            ci = record.get(metric + "_ci", [])
            if len(values) != 5 or len(ci) != 2:
                return "UNSTABLE_RESULT"
            if not all(
                isinstance(v, (float, int)) and math.isfinite(v)
                for v in values + ci
            ):
                return "UNSTABLE_RESULT"
            if ci[0] > ci[1] or max(values) - min(values) >= limit:
                return "UNSTABLE_RESULT"
    if any(
        r["rho_ci"][1] < -p["rho_preservation_margin"]
        or r["rmse_gain_ci"][1] < -p["relative_rmse_harm_margin"]
        for r in evidence.values()
    ):
        return "NEGATIVE_TRANSFER"
    preserved = all(
        r["rho_ci"][0] >= -p["rho_preservation_margin"]
        and r["rmse_gain_ci"][0] >= -p["relative_rmse_harm_margin"]
        and sum(
            a >= -p["rho_preservation_margin"]
            and b >= -p["relative_rmse_harm_margin"]
            for a, b in zip(r["rho"], r["rmse_gain"])
        )
        >= 4
        for r in evidence.values()
    )
    benefit = any(
        sum(r["rho"]) / 5 >= p["rho_minimum_benefit"] and r["rho_ci"][0] > 0
        for r in evidence.values()
    )
    return (
        "MULTI_DOMAIN_BENEFIT" if preserved and benefit else "NO_CLEAR_BENEFIT"
    )


def check_authoritative_artifacts():
    a = json.loads(safe_path(ROOT / PHASE18A).read_text(encoding="utf-8"))
    g = json.loads(safe_path(ROOT / PHASE17).read_text(encoding="utf-8"))
    sensitivity = a["shared_sequence_bridge"][
        "canonical_modeling_population_sensitivity"
    ]
    checks = [
        g["final_decision"] == "GO",
        g["aggregate"]["total_new_unique_compatible_observations"] == 10544,
        g["accepted_candidates"] == [DOMAINS[1]],
        a["recommended_strategy"] == "MULTI_DOMAIN_EXPERIMENT_RECOMMENDED",
        a["immediate_pooling_justified"] is False,
        a["controlled_integration_experiments_justified"] is True,
        a["datasets"][DOMAINS[0]]["canonical_modeling_n"] == 10117,
        a["datasets"][DOMAINS[1]]["new_nonoverlapping_n"] == 10544,
        a["shared_sequence_bridge"]["n"] == 48,
        sensitivity["shared_30mer_n"] == 41,
        a["canonical_integrity"]["all_expected_hashes_match"] is True,
        {
            p: r["expected_sha256"]
            for p, r in a["canonical_integrity"]["after"].items()
        }
        == EXPECTED_HASHES,
    ]
    if not all(checks):
        raise RuntimeError(
            "STOP: authoritative Phase 17/18A artifact discrepancy"
        )


def build_protocol():
    """Verify data structure, then lock a label-free split and future run matrix."""
    with audit_only_guard():
        check_authoritative_artifacts()
        before = verify_integrity()
        deep_raw, deep, xiang = load_phase18a_inputs()
        sources = set(xiang["Dataset"])
        if sources != {"LuoSpCas92020_min200", "Overlap_Luo_Kim2019"}:
            raise ValueError("Unexpected Xiang/Luo source components")
        sequences = {
            DOMAINS[0]: deep["sequence_30mer"].tolist(),
            DOMAINS[1]: xiang["30mer_gRNA"].tolist(),
        }
        raw_overlap = set(deep_raw["sequence_30mer"]) & set(
            sequences[DOMAINS[1]]
        )
        bridge = set(sequences[DOMAINS[0]]) & set(sequences[DOMAINS[1]])
        raw_spacers = {group_key(s) for s in deep_raw["sequence_30mer"]}
        xiang_spacers = {group_key(s) for s in sequences[DOMAINS[1]]}
        if (
            len(raw_overlap),
            len(bridge),
            len(raw_spacers & xiang_spacers),
            len(xiang_spacers),
        ) != (48, 41, 48, 10592):
            raise ValueError("Unexpected overlap/diversity counts")
        label_hashes = {}
        for domain, frame, label in (
            (DOMAINS[0], deep, "activity"),
            (DOMAINS[1], xiang, "HEK293T_indel_freq_avg_d8_d10"),
        ):
            values = frame[label].tolist()
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite label")
            pairs = sorted(
                zip(sequences[domain], (float(v).hex() for v in values))
            )
            label_hashes[domain] = fingerprint(pairs)
        manifest = make_split_manifest(sequences, raw_overlap, bridge)
        counts = {
            d: {
                p: sum(r["domain"] == d and r["split"] == p for r in manifest)
                for p in PARTITIONS
            }
            for d in DOMAINS
        }
        if (
            counts[DOMAINS[0]]["bridge"] != 41
            or counts[DOMAINS[1]]["bridge"] != 41
        ):
            raise ValueError("Unexpected bridge counts")
        if (
            counts[DOMAINS[0]]["quarantine"] != 0
            or counts[DOMAINS[1]]["quarantine"] != 7
        ):
            raise ValueError("Unexpected bridge-relative/quarantine counts")
        if any(counts[d][p] == 0 for d in DOMAINS for p in PARTITIONS[:3]):
            raise ValueError("Empty development partition")
        if counts != EXPECTED_COUNTS:
            raise ValueError("Locked partition counts changed")
        policy = policies()
        matrix = experiment_matrix(policy)
        validate_policy(policy, matrix)
        after = verify_integrity()
        if before != after:
            raise RuntimeError("Protected inputs changed during Phase 18B")
        protected_sources = [
            "config.yaml",
            "src/models/cnn.py",
            "src/models/random_forest.py",
            "src/models/xgboost_model.py",
            "src/data/validation.py",
            "src/data/preprocessing.py",
            "src/bioinformatics/sequence_features.py",
            "src/dataset_integration/label_compatibility.py",
            PHASE17,
            PHASE18A,
            "docs/phase17g_r1_data_sufficiency_reassessment_report.md",
            "docs/phase18a_cross_dataset_compatibility_report.md",
            REPORT,
            "src/experiment_protocols/multidomain_protocol.py",
            "scripts/run_phase18b_multidomain_protocol.py",
        ]
        source_hashes = {
            p: sha256_file(safe_path(ROOT / p)) for p in protected_sources
        }
        steps = math.ceil(max(counts[d]["train"] for d in DOMAINS) / 16)
        return {
            "phase": "18B",
            "protocol_version": VERSION,
            "final_decision": "READY_FOR_PHASE18C",
            "scope": {
                "design_only": True,
                "model_training_occurred": False,
                "moreno_accessed": False,
                "phase18c_started": False,
            },
            "policy": policy,
            "experiment_matrix": matrix,
            "split_manifest": manifest,
            "split_counts": counts,
            "split_sha256": fingerprint(manifest),
            "label_fingerprints": label_hashes,
            "label_fingerprint_format": "sorted [sequence, float64.hex(label)] pairs",
            "overlap": {
                "raw_30mer": 48,
                "raw_spacer20": 48,
                "canonical_30mer": 41,
                "B_new_nonoverlapping": 10544,
            },
            "group_counts": {
                d: len(
                    {r["group_sha256"] for r in manifest if r["domain"] == d}
                )
                for d in DOMAINS
            },
            "canonical_integrity": {
                "before": before,
                "after": after,
                "unchanged": True,
                "all_expected_hashes_match": True,
            },
            "protected_source_hashes": source_hashes,
            "budget": {
                "total_runs": len(matrix),
                "cnn_runs": 15,
                "rf_runs": 10,
                "xgboost_runs": 10,
                "seeds": 5,
                "cnn_steps_per_epoch": steps,
                "max_steps_per_cnn_run": steps * 100,
                "max_total_cnn_steps": steps * 100 * 15,
                "relative_cost": "15 capacity/step-matched CNN runs plus "
                "20 fixed tabular fits; no tuning grid",
                "wall_time": "not estimated empirically; no Phase 18B training",
            },
            "excluded_arms": {
                "C_naive_pooling": "unit-confounded straw control; cannot identify "
                "sharing benefit and violates exchangeability evidence",
                "E_domain_conditioning": "additional interaction/capacity confound; "
                "not needed for first shared-encoder question",
                "F_transfer": "direction, freezing and sequential budget add a "
                "different question; defer to separately approved protocol",
                "separate_encoders": "already supplied by paired A and B baselines",
                "weighting_grid": "no evidence for adaptive/sample weighting search",
            },
        }


def validate_locked_artifact(payload, approved_digest=None):
    """Fail if the protocol lock, matrix, counts, or split assignments drift."""
    body = {k: v for k, v in payload.items() if k != "protocol_sha256"}
    if payload.get("protocol_sha256") != fingerprint(body):
        raise ValueError("Protocol serialization hash mismatch")
    if (
        approved_digest is not None
        and payload["protocol_sha256"] != approved_digest
    ):
        raise ValueError("Approved protocol hash mismatch")
    validate_policy(payload["policy"], payload["experiment_matrix"])
    rows = payload["split_manifest"]
    if fingerprint(rows) != payload["split_sha256"]:
        raise ValueError("Split fingerprint mismatch")
    held = {r["group_sha256"] for r in rows if r["split"] in PARTITIONS[3:]}
    validate_leakage(rows, held)
    for domain in DOMAINS:
        counts = Counter(r["split"] for r in rows if r["domain"] == domain)
        if any(
            counts[p] != payload["split_counts"][domain][p] for p in PARTITIONS
        ):
            raise ValueError("Partition count mismatch")
    if payload["split_counts"] != EXPECTED_COUNTS:
        raise ValueError("Locked partition counts changed")


def verify_locked_inputs(payload, approved_digest):
    """Phase 18C preflight: no fitting, automatic repair, or new split policy."""
    with audit_only_guard():
        validate_locked_artifact(payload, approved_digest)
        verify_integrity()
        for path, expected in payload["protected_source_hashes"].items():
            if sha256_file(safe_path(ROOT / path)) != expected:
                raise RuntimeError(f"Locked source hash mismatch: {path}")
        rebuilt = build_protocol()
        for key in (
            "split_manifest",
            "split_sha256",
            "split_counts",
            "label_fingerprints",
            "group_counts",
            "overlap",
        ):
            if rebuilt[key] != payload[key]:
                raise RuntimeError(
                    f"Locked input/preprocessing mismatch: {key}"
                )


def lock_protocol(payload):
    if "protocol_sha256" in payload:
        raise ValueError("Already locked")
    result = dict(payload, protocol_sha256=fingerprint(payload))
    validate_locked_artifact(result)
    return result
