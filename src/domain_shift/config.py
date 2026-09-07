"""
Phase 14 configuration.

PRE-REGISTERED and FROZEN before execution. None of these values depend on
Moreno-Mateos data. The GC-bin edges are constructed from the INTERNAL TRAIN
distribution only (quantile method, with a fixed fallback grid), and the
realized numeric edges are written to freeze.json before any external access.

C1/C2 are BETWEEN-DOMAIN comparisons of subgroup MAE (external minus internal)
using an independent-groups bootstrap: observations are never paired across
datasets. C6 is restricted to common-support GC x activity cells with n >= 20
in BOTH domains and is labelled as such (it does not represent the full
datasets).
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass(frozen=True)
class Phase14DomainShiftConfig:
    # ---- Paths / datasets ----
    project_root: str = "/home/konta/crispr-prediction"
    primary_dataset: str = "data/raw/DeepSpCas9.csv"
    external_dataset: str = "data/raw/Moreno-Mateos.csv"
    context_length: int = 30
    guide_start: int = 4
    guide_length: int = 20

    # ---- Canonical split (identical to Phase 5/6/13) ----
    val_ratio: float = 0.15
    split_seed: int = 42
    expected_n_valid: int = 10117
    expected_n_train: int = 8599
    expected_n_val: int = 1518
    expected_n_external: int = 810

    # ---- Canonical model artifacts ----
    model_rf: str = "models/rf_baseline_fixed_20260905_001107.pkl"
    model_xgb: str = "models/xgboost_baseline_20260905_002839.pkl"
    model_cnn: str = "models/cnn_baseline_20260905_011720.pt"
    model_phase13: str = "models/phase13_nucleotide_embedding_20260906_203923.pt"
    primary_models: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "cnn"]
    )
    phase13_secondary: bool = True
    primary_assoc_model: str = "random_forest"

    # Recorded canonical artifact hashes (Phase 9D integrity table). Asserted at
    # FEASIBILITY and re-asserted at AUDIT (must remain unchanged).
    expected_canonical_hashes: Dict[str, str] = field(
        default_factory=lambda: {
            "models/rf_baseline_fixed_20260905_001107.pkl":
                "1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285750740280b6",
            "models/xgboost_baseline_20260905_002839.pkl":
                "129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd",
            "models/cnn_baseline_20260905_011720.pt":
                "76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616",
        }
    )

    # ---- Frozen bin construction ----
    # GC bins: quintile edges of the INTERNAL TRAIN GC distribution, rounded to
    # 2 dp, deduplicated. If fewer than 5 distinct bins result, use the fixed
    # absolute fallback grid below.
    gc_method: str = "internal_train_quantile"
    gc_quantiles: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    gc_round: int = 2
    n_gc_bins: int = 5
    fallback_gc_edges: List[float] = field(
        default_factory=lambda: [0.40, 0.50, 0.60, 0.70]
    )

    # Activity bins: FIXED equal-width grid on true activity (last bin closed).
    activity_edges: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # ---- Stratification parameters ----
    gc_definition: str = "full_30mer"  # primary; guide GC [4:24] is sensitivity
    min_n_report: int = 20
    min_n_boot: int = 30
    min_n_common_support: int = 20
    n_boot: int = 1000
    bootstrap_seed: int = 20260907

    # ---- Confirmatory thresholds ----
    fdr_q: float = 0.05
    alpha: float = 0.05
    rho_effect_threshold: float = 0.15
    dmae_effect_threshold: float = 0.02
    convergence_rho_threshold: float = 0.6
    dispersion_sd_ratio_threshold: float = 0.6
    c6_cell_delta_threshold: float = 0.02

    # ---- Confirmatory tests (frozen) ----
    confirmatory_tests: List[str] = field(
        default_factory=lambda: [
            "C1_activity_lowbin_delta_mae",
            "C2_activity_highbin_delta_mae",
            "C3_spearman_abs_gc",
            "C4_spearman_abs_activity",
            "C5_spearman_abs_gc_train_z",
            "C6_joint_common_support_median_delta",
            "C7_cross_model_bin_mae_consistency",
        ]
    )

    # ---- Output dirs ----
    output_dir: str = "results/experiments"
    figures_dir: str = "results/figures"
    predictions_dir: str = "results/predictions"

    def paths(self) -> Dict[str, Path]:
        root = Path(self.project_root)
        return {
            "primary_dataset": root / self.primary_dataset,
            "external_dataset": root / self.external_dataset,
            "model_rf": root / self.model_rf,
            "model_xgb": root / self.model_xgb,
            "model_cnn": root / self.model_cnn,
            "model_phase13": root / self.model_phase13,
            "output_dir": root / self.output_dir,
            "figures_dir": root / self.figures_dir,
            "predictions_dir": root / self.predictions_dir,
        }


PHASE14_CONFIG = Phase14DomainShiftConfig()