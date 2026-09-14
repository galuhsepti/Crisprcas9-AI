"""Phase 18A cross-dataset label compatibility audit.

This module is deliberately audit-only. It reads two allowlisted inputs,
computes descriptive diagnostics, and never creates training data or fits a
predictive model.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.bioinformatics.kmer import (
    calculate_kmer_complexity,
    calculate_kmer_entropy,
)
from src.bioinformatics.nucleotide_composition import calculate_heterogeneity
from src.data.validation import validate_sequence
from src.dataset_recovery.biological_independence_crispron import clean_dna
from src.dataset_recovery.distribution import (
    jensen_shannon_divergence,
    kmer_summary,
)

DEEPSPCAS9_PATH = Path("data/raw/DeepSpCas9.csv")
CRISPRON_PATH = Path("data/phase17i/raw/crispron/Luo2020_Kim2019.xlsx")
CRISPRON_SHEET = "Luo2020_Kim2019"

EXPECTED_HASHES = {
    "data/raw/DeepSpCas9.csv": "6aeab30f55155ca9f1b80aae3159231ce113815354597ea4ca1d6c84ba0296df",
    "data/phase17i/raw/crispron/Luo2020_Kim2019.xlsx": "146f211575179fc31669dbd5eaef46434b0866013a2d03767354375a1cc5d61f",
    "models/rf_baseline_fixed_20260905_001107.pkl": "1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285750740280b6",
    "models/xgboost_baseline_20260905_002839.pkl": "129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd",
    "models/cnn_baseline_20260905_011720.pt": "76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616",
}

QUANTILES = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]


def _portable(path: Path) -> str:
    return str(path).replace("\\", "/")


def prohibit_locked_path(path: str | Path) -> None:
    """Reject any attempt to route this audit through the locked dataset."""
    text = _portable(Path(path)).casefold()
    if "moreno" in text or "mateos" in text:
        raise PermissionError(
            "Phase 18A may only read the two allowlisted development datasets"
        )


def require_canonical_path(path: str | Path, expected: str | Path) -> Path:
    """Require an exact approved input path and reject locked-path aliases."""
    candidate = Path(path)
    approved = Path(expected)
    prohibit_locked_path(candidate)
    if candidate.resolve() != approved.resolve():
        raise ValueError(f"Unapproved Phase 18A input path: {_portable(candidate)}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrity_snapshot(paths: Mapping[str, str]) -> dict:
    """Hash only explicit canonical/approved paths; no directory traversal."""
    records = {}
    for path_text, expected in paths.items():
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"Invalid expected SHA-256 for {path_text}")
        path = Path(path_text)
        prohibit_locked_path(path)
        exists = path.is_file()
        observed = sha256_file(path) if exists else None
        records[path_text] = {
            "exists": exists,
            "sha256": observed,
            "expected_sha256": expected,
            "matches_expected": observed == expected if exists else False,
        }
    return records


def require_expected_integrity(snapshot: Mapping[str, Mapping[str, object]]) -> None:
    """Stop the audit if an approved input or protected artifact has drifted."""
    mismatches = [
        path for path, record in snapshot.items() if not record["matches_expected"]
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(f"Phase 18A integrity check failed: {joined}")


def canonical_modeling_mask(sequences: Sequence[object]) -> np.ndarray:
    """Reproduce the canonical guide homopolymer eligibility rule."""
    mask = []
    for value in sequences:
        sequence = clean_dna(value)
        valid = (
            False
            if sequence is None
            else validate_sequence(sequence, check_pam=False)[0]
        )
        mask.append(valid)
    return np.asarray(mask, dtype=bool)


def load_phase18a_inputs(
    deepspcas9_path: Path = DEEPSPCAS9_PATH,
    crispron_path: Path = CRISPRON_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw DeepSpCas9, its canonical subset, and Xiang/Luo only."""
    deep_path = require_canonical_path(deepspcas9_path, DEEPSPCAS9_PATH)
    crispron_input = require_canonical_path(crispron_path, CRISPRON_PATH)
    deep_raw = pd.read_csv(deep_path, usecols=["sequence_30mer", "activity"])
    deep_raw["sequence_30mer"] = deep_raw["sequence_30mer"].map(clean_dna)
    deep_canonical = deep_raw.loc[
        canonical_modeling_mask(deep_raw["sequence_30mer"])
    ].copy()

    workbook = pd.read_excel(
        crispron_input,
        sheet_name=CRISPRON_SHEET,
        usecols=[
            "Dataset",
            "Gene",
            "Transcript",
            "30mer_gRNA",
            "HEK293T_indel_freq_avg_d8_d10",
        ],
    )
    source = workbook["Dataset"].astype(str).str.lower()
    mask = source.str.contains("xiang|luo|xu", regex=True, na=False)
    xiang_luo = workbook.loc[mask].copy()
    xiang_luo["30mer_gRNA"] = xiang_luo["30mer_gRNA"].map(clean_dna)

    if len(deep_raw) != 12832 or len(deep_canonical) != 10117:
        raise ValueError(
            "DeepSpCas9 population does not match canonical project records"
        )
    if len(xiang_luo) != 10592:
        raise ValueError(
            "Xiang/Luo population does not match accepted Phase 17G-R1 records"
        )
    return deep_raw, deep_canonical, xiang_luo


def label_statistics(values: Sequence[object]) -> dict:
    """Return complete finite-label descriptives without modifying values."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))].astype(
        float
    )
    n = int(len(series))
    finite_n = int(len(finite))
    if not finite_n:
        return {
            "n": n,
            "finite_n": 0,
            "missing_n": n,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "SD": None,
            "IQR": None,
            "unique_value_n": 0,
            "quantiles_percent": {},
            "skewness": None,
            "fraction_at_min": None,
            "fraction_at_max": None,
            "empirical_cdf": {"n_points": 0, "x": [], "probability": []},
        }

    array = finite.to_numpy(dtype=float)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    quantiles = np.quantile(array, QUANTILES)
    if finite_n < 3 or maximum == minimum:
        skewness = 0.0
    else:
        skewness = float(stats.skew(array, bias=False))
    sorted_values = np.sort(array)
    indices = np.linspace(0, finite_n - 1, min(21, finite_n)).astype(int)
    return {
        "n": n,
        "finite_n": finite_n,
        "missing_n": n - finite_n,
        "min": minimum,
        "max": maximum,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "SD": float(np.std(array, ddof=1)) if finite_n > 1 else 0.0,
        "IQR": float(quantiles[6] - quantiles[4]),
        "unique_value_n": int(np.unique(array).size),
        "quantiles_percent": {
            str(int(q * 100)): float(value) for q, value in zip(QUANTILES, quantiles)
        },
        "skewness": skewness,
        "fraction_at_min": float(np.mean(array == minimum)),
        "fraction_at_max": float(np.mean(array == maximum)),
        "empirical_cdf": {
            "n_points": int(len(indices)),
            "x": [float(sorted_values[index]) for index in indices],
            "probability": [float((index + 1) / finite_n) for index in indices],
        },
    }


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> dict:
    if x.size < 2:
        return {"coefficient": None, "p_value": None, "status": "INSUFFICIENT_N"}
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "coefficient": None,
            "p_value": None,
            "status": "UNDEFINED_CONSTANT_LABEL",
        }
    function = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[method]
    coefficient, p_value = function(x, y)
    return {
        "coefficient": float(coefficient),
        "p_value": float(p_value),
        "status": "COMPUTED",
    }


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros(values.size, dtype=float)
    return (stats.rankdata(values, method="average") - 1) / (values.size - 1)


def _relationship_label(spearman: dict) -> str:
    value = spearman.get("coefficient")
    if value is None:
        return "NOT_ESTABLISHED"
    strength = (
        "STRONG"
        if abs(value) >= 0.70
        else (
            "MODERATE"
            if abs(value) >= 0.40
            else "WEAK" if abs(value) >= 0.20 else "LITTLE_OR_NONE"
        )
    )
    direction = "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "NONE"
    return f"{strength}_{direction}_MONOTONIC_ASSOCIATION"


def extract_shared_sequence_bridge(
    deepspcas9: pd.DataFrame,
    xiang_luo: pd.DataFrame,
    deep_sequence_col: str = "sequence_30mer",
    deep_label_col: str = "activity",
    xiang_sequence_col: str = "30mer_gRNA",
    xiang_label_col: str = "HEK293T_indel_freq_avg_d8_d10",
) -> dict:
    """Match exact 30-mers and compute diagnostic-only label agreement."""
    left = deepspcas9[[deep_sequence_col, deep_label_col]].copy()
    right = xiang_luo[[xiang_sequence_col, xiang_label_col]].copy()
    left[deep_sequence_col] = left[deep_sequence_col].map(clean_dna)
    right[xiang_sequence_col] = right[xiang_sequence_col].map(clean_dna)
    left = left.dropna(subset=[deep_sequence_col])
    right = right.dropna(subset=[xiang_sequence_col])
    if (
        left[deep_sequence_col].duplicated().any()
        or right[xiang_sequence_col].duplicated().any()
    ):
        raise ValueError("Exact bridge requires one label per 30-mer in each dataset")
    bridge = left.merge(
        right, left_on=deep_sequence_col, right_on=xiang_sequence_col, how="inner"
    )
    bridge = bridge.rename(
        columns={
            deep_sequence_col: "sequence_30mer",
            deep_label_col: "deepspcas9_label",
            xiang_label_col: "crispron_xiang_luo_label",
        }
    )
    bridge = bridge.drop(columns=[xiang_sequence_col], errors="ignore")
    matched_n = int(len(bridge))
    bridge["deepspcas9_label"] = pd.to_numeric(
        bridge["deepspcas9_label"], errors="coerce"
    )
    bridge["crispron_xiang_luo_label"] = pd.to_numeric(
        bridge["crispron_xiang_luo_label"], errors="coerce"
    )
    finite = bridge[
        np.isfinite(bridge["deepspcas9_label"])
        & np.isfinite(bridge["crispron_xiang_luo_label"])
    ].copy()
    finite = finite.sort_values("sequence_30mer").reset_index(drop=True)
    x = finite["deepspcas9_label"].to_numpy(dtype=float)
    y = finite["crispron_xiang_luo_label"].to_numpy(dtype=float)
    pearson = _correlation(x, y, "pearson")
    spearman = _correlation(x, y, "spearman")
    kendall = _correlation(x, y, "kendall")

    deep_rank = _rank_percentiles(x)
    xiang_rank = _rank_percentiles(y)
    rank_difference = np.abs(deep_rank - xiang_rank)
    if len(finite):
        finite["deepspcas9_percentile_rank"] = deep_rank
        finite["crispron_percentile_rank"] = xiang_rank
        finite["absolute_percentile_rank_disagreement"] = rank_difference
        finite["raw_numeric_absolute_disagreement"] = np.abs(x - y)
    quartile_deep = set(np.flatnonzero(deep_rank >= 0.75))
    quartile_xiang = set(np.flatnonzero(xiang_rank >= 0.75))
    quartile_union = quartile_deep | quartile_xiang

    linear = {"status": "UNDEFINED_CONSTANT_LABEL"}
    robust = {"status": "UNDEFINED_CONSTANT_LABEL"}
    outlier_indices: list[int] = []
    if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
        fit = stats.linregress(x, y)
        residuals = y - (fit.intercept + fit.slope * x)
        linear = {
            "status": "COMPUTED_DIAGNOSTIC_ONLY",
            "direction": "crispron_xiang_luo_label ~ deepspcas9_label",
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r_value": float(fit.rvalue),
            "r_squared": float(fit.rvalue**2),
            "p_value": float(fit.pvalue),
            "slope_standard_error": float(fit.stderr),
            "intercept_standard_error": float(fit.intercept_stderr),
            "residual_SD": (
                float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
            ),
            "residual_MAE": float(np.mean(np.abs(residuals))),
        }
        slope, intercept, low_slope, high_slope = stats.theilslopes(y, x, alpha=0.95)
        robust = {
            "status": "COMPUTED_DIAGNOSTIC_ONLY",
            "method": "Theil-Sen",
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_95pct_CI": [float(low_slope), float(high_slope)],
        }
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        if mad > 0:
            robust_z = 0.6745 * (residuals - median_residual) / mad
            outlier_indices = np.flatnonzero(np.abs(robust_z) > 3.5).tolist()
            finite["linear_residual"] = residuals
            finite["robust_residual_z"] = robust_z

    rows = []
    for index, row in finite.iterrows():
        record = {
            "sequence_30mer": row["sequence_30mer"],
            "deepspcas9_label": float(row["deepspcas9_label"]),
            "crispron_xiang_luo_label": float(row["crispron_xiang_luo_label"]),
            "raw_numeric_absolute_disagreement": float(
                row["raw_numeric_absolute_disagreement"]
            ),
            "deepspcas9_percentile_rank": float(row["deepspcas9_percentile_rank"]),
            "crispron_percentile_rank": float(row["crispron_percentile_rank"]),
            "absolute_percentile_rank_disagreement": float(
                row["absolute_percentile_rank_disagreement"]
            ),
            "linear_outlier": index in outlier_indices,
        }
        if "linear_residual" in finite:
            record["linear_residual"] = float(row["linear_residual"])
            record["robust_residual_z"] = float(row["robust_residual_z"])
        rows.append(record)

    diagnostics = [
        "Bridge observations are diagnostic only and were not used to derive a harmonization transform.",
        "Raw absolute differences mix fraction and percentage units and are not biological effect sizes.",
        "Correlation and regression cannot remove timing, selection, expression, or assay-context differences.",
    ]
    if outlier_indices:
        diagnostics.append(
            f"{len(outlier_indices)} bridge observations exceed the pre-specified absolute robust residual z threshold of 3.5."
        )
    return {
        "n": matched_n,
        "finite_pair_n": int(len(finite)),
        "missing_pair_n": matched_n - int(len(finite)),
        "matching": "exact forward 30-mer only",
        "diagnostic_only": True,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "monotonic_relationship": _relationship_label(spearman),
        "rank_agreement": {
            "mean_absolute_percentile_rank_disagreement": (
                float(np.mean(rank_difference)) if rank_difference.size else None
            ),
            "median_absolute_percentile_rank_disagreement": (
                float(np.median(rank_difference)) if rank_difference.size else None
            ),
            "maximum_absolute_percentile_rank_disagreement": (
                float(np.max(rank_difference)) if rank_difference.size else None
            ),
            "same_quartile_fraction": (
                float(
                    np.mean(
                        np.floor(deep_rank * 4 - 1e-12)
                        == np.floor(xiang_rank * 4 - 1e-12)
                    )
                )
                if rank_difference.size
                else None
            ),
            "top_quartile_overlap_n": len(quartile_deep & quartile_xiang),
            "top_quartile_jaccard": (
                float(len(quartile_deep & quartile_xiang) / len(quartile_union))
                if quartile_union
                else None
            ),
        },
        "absolute_disagreement": {
            "raw_numeric_MAE": float(np.mean(np.abs(x - y))) if x.size else None,
            "raw_numeric_median": float(np.median(np.abs(x - y))) if x.size else None,
            "interpretation": "Not directly interpretable because the original labels use different units and endpoints; no conversion was applied.",
        },
        "linear_regression": linear,
        "robust_regression": robust,
        "outlier_n": len(outlier_indices),
        "rows": rows,
        "diagnostics": diagnostics,
    }


def _numeric_summary(values: Sequence[float]) -> dict:
    return label_statistics(values)


def _position_frequencies(sequences: Sequence[str]) -> list[dict]:
    records = []
    for position in range(30):
        counts = Counter(sequence[position] for sequence in sequences)
        total = len(sequences)
        region = (
            "upstream_context"
            if position < 4
            else (
                "spacer"
                if position < 24
                else "PAM" if position < 27 else "downstream_context"
            )
        )
        records.append(
            {
                "position_0_based": position,
                "region": region,
                "frequencies": {
                    base: float(counts.get(base, 0) / total) for base in "ACGT"
                },
            }
        )
    return records


def _frequency(counter: Counter, total: int) -> dict[str, float]:
    return {key: float(value / total) for key, value in sorted(counter.items())}


def _sequence_profile(sequences: Sequence[object]) -> dict:
    valid = [clean_dna(value) for value in sequences]
    valid30 = [
        sequence for sequence in valid if sequence is not None and len(sequence) == 30
    ]
    spacers = [sequence[4:24] for sequence in valid30]
    pam_counts = Counter(sequence[24:27] for sequence in valid30)
    full_kmers = {str(k): kmer_summary(valid30, k) for k in (2, 3)}
    spacer_kmers = {str(k): kmer_summary(spacers, k) for k in (2, 3)}
    return {
        "n": len(sequences),
        "valid_30mer_n": len(valid30),
        "unique_30mer_n": len(set(valid30)),
        "full_30mer_gc": _numeric_summary(
            [(s.count("G") + s.count("C")) / 30 for s in valid30]
        ),
        "spacer_gc": _numeric_summary(
            [(s.count("G") + s.count("C")) / 20 for s in spacers]
        ),
        "positional_nucleotide_frequencies": _position_frequencies(valid30),
        "PAM_distribution": {
            "counts": dict(sorted(pam_counts.items())),
            "frequencies": _frequency(pam_counts, len(valid30)),
        },
        "spacer_sequence_complexity": {
            "nucleotide_entropy": _numeric_summary(
                [calculate_heterogeneity(s) for s in spacers]
            ),
            "kmer_2_entropy": _numeric_summary(
                [calculate_kmer_entropy(s, 2) for s in spacers]
            ),
            "kmer_3_entropy": _numeric_summary(
                [calculate_kmer_entropy(s, 3) for s in spacers]
            ),
            "kmer_2_complexity": _numeric_summary(
                [calculate_kmer_complexity(s, 2) for s in spacers]
            ),
            "kmer_3_complexity": _numeric_summary(
                [calculate_kmer_complexity(s, 3) for s in spacers]
            ),
        },
        "kmer_distributions": {"full_30mer": full_kmers, "spacer20": spacer_kmers},
    }


def _top_frequency_deltas(
    left: Mapping[str, float], right: Mapping[str, float], n: int = 10
) -> list[dict]:
    records = [
        {
            "kmer": key,
            "deepspcas9": float(left.get(key, 0.0)),
            "crispron_xiang_luo": float(right.get(key, 0.0)),
            "delta": float(right.get(key, 0.0) - left.get(key, 0.0)),
        }
        for key in sorted(set(left) | set(right))
    ]
    return sorted(records, key=lambda item: (-abs(item["delta"]), item["kmer"]))[:n]


def sequence_domain_comparison(
    deep_sequences: Sequence[object], xiang_sequences: Sequence[object]
) -> dict:
    deep = _sequence_profile(deep_sequences)
    xiang = _sequence_profile(xiang_sequences)
    position_jsd = []
    for left, right in zip(
        deep["positional_nucleotide_frequencies"],
        xiang["positional_nucleotide_frequencies"],
    ):
        position_jsd.append(
            {
                "position_0_based": left["position_0_based"],
                "region": left["region"],
                "JSD_bits": jensen_shannon_divergence(
                    left["frequencies"], right["frequencies"]
                ),
            }
        )
    kmer_comparison = {}
    for region in ("full_30mer", "spacer20"):
        kmer_comparison[region] = {}
        for k in ("2", "3"):
            left = deep["kmer_distributions"][region][k]["mean_frequencies"]
            right = xiang["kmer_distributions"][region][k]["mean_frequencies"]
            kmer_comparison[region][f"k{k}"] = {
                "JSD_bits": jensen_shannon_divergence(left, right),
                "largest_absolute_frequency_deltas": _top_frequency_deltas(left, right),
            }
    jsd_values = [record["JSD_bits"] for record in position_jsd]
    return {
        "status": "DESCRIPTIVE_DIFFERENCE_NOT_A_GATE",
        "deepspcas9": deep,
        "crispron_xiang_luo": xiang,
        "comparison": {
            "full_30mer_gc_mean_delta_xiang_minus_deep": xiang["full_30mer_gc"]["mean"]
            - deep["full_30mer_gc"]["mean"],
            "spacer_gc_mean_delta_xiang_minus_deep": xiang["spacer_gc"]["mean"]
            - deep["spacer_gc"]["mean"],
            "PAM_JSD_bits": jensen_shannon_divergence(
                deep["PAM_distribution"]["frequencies"],
                xiang["PAM_distribution"]["frequencies"],
            ),
            "positional_nucleotide_JSD": position_jsd,
            "mean_positional_JSD_bits": float(np.mean(jsd_values)),
            "max_positional_JSD_bits": float(np.max(jsd_values)),
            "kmer": kmer_comparison,
            "interpretation": "Sequence distributions differ descriptively; these differences are not treated as biological incompatibility or a PASS/FAIL gate.",
        },
    }


def label_semantics() -> dict:
    return {
        "deepspcas9": {
            "original_publication": "Kim et al. 2019, Science Advances 5:eaax9249",
            "doi": "10.1126/sciadv.aax9249",
            "primary_source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6834390/",
            "local_source_trace": "Benchmark derivative modFreq renamed unchanged to activity by scripts/prepare_data.py",
            "experimental_assay": "Pooled lentiviral guide-plus-integrated-surrogate-target library followed by targeted deep sequencing",
            "cell_types": ["HEK293T"],
            "target_system": "Integrated 30-nt surrogate target paired with its sgRNA; not an endogenous-locus training measurement",
            "cas_nuclease": "SpCas9",
            "activity_definition": "Fraction of reads with an insertion or deletion in the 8-nt window centered on the expected cleavage site",
            "label_name": "activity (source derivative: modFreq)",
            "raw_biological_measurement": "Day-2.9 indel reads divided by total reads",
            "normalization": "Local values equal the source table's uncorrected indel percentage divided by 100; the publication separately describes background-corrected indel frequency",
            "numerical_unit": "fraction",
            "higher_means_greater_editing_efficiency": True,
            "direct_primary_measurement": True,
            "original_author_transformations": "Publication background-correction formula exists, but the project-local activity column traces to the uncorrected indel-frequency field rather than that corrected field",
            "measurement_timing": "2.9 days after SpCas9 lentiviral transduction",
            "evidence_chain": [
                "Kim et al. 2019 primary publication and Table S1 (HT_Cas9_Train)",
                "third-party Benchmarking-CRISPR-on-tools DeepSpCas9 (Library) derivative",
                "scripts/prepare_data.py renames modFreq to activity without label transformation",
                "data/raw/DeepSpCas9.csv activity",
            ],
            "evidence_caveat": "The exact response column used by the original authors' final model is not fully established from the published training code; this audit describes the repository-local label actually used by this thesis pipeline.",
        },
        "crispron_xiang_luo": {
            "original_publication": "Xiang et al. 2021, Nature Communications 12:3238",
            "doi": "10.1038/s41467-021-23576-0",
            "primary_source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8163799/",
            "official_resource_url": "https://rth.dk/resources/crispr/crispron/download",
            "experimental_assay": "Pooled lentiviral 37-bp surrogate-target library followed by targeted amplicon deep sequencing",
            "cell_types": [
                "HEK293T-SpCas9 stable line",
                "wild-type HEK293T background control",
            ],
            "target_system": "Randomly integrated 37-bp surrogate target containing 10-nt upstream, spacer, PAM, and 4-nt downstream context",
            "cas_nuclease": "Human-codon-optimized SpCas9",
            "activity_definition": "Mean Dox-free total indel efficiency at days 8 and 10 after guide/surrogate transduction",
            "label_name": "HEK293T_indel_freq_avg_d8_d10",
            "raw_biological_measurement": "Percentage of amplicon reads classified as CRISPR-induced indels after removing background sequence variants, averaged across days 8 and 10",
            "normalization": "No Phase 18A normalization; workbook-native raw day-8/day-10 average is audited. The publication later linearly rescales Xiang/Luo values for its own Kim-integrated training dataset.",
            "numerical_unit": "percent",
            "higher_means_greater_editing_efficiency": True,
            "direct_primary_measurement": True,
            "original_author_transformations": "Arithmetic day-8/day-10 averaging for this field; low-read filtering and background-variant removal. A separate integrated-label field applies linear cross-study rescaling and is not used here.",
            "measurement_timing": "8 and 10 days after guide/surrogate transduction, Dox-free, with puromycin selection from day 2",
            "evidence_chain": [
                "Xiang et al. 2021 primary publication and Supplementary Data 1",
                "official CRISPRon/RTH Luo2020_Kim2019.xlsx download",
                "Dataset values LuoSpCas92020_min200 plus Overlap_Luo_Kim2019",
                "30mer_gRNA",
                "HEK293T_indel_freq_avg_d8_d10",
            ],
            "evidence_caveat": "Official training documentation references Quant_norm_efficiency, which is absent from the recovered workbook; no identity or transformation for that field is assumed.",
        },
        "underlying_concept_assessment": {
            "same_underlying_biological_concept": True,
            "concept": "Sequence-linked SpCas9 on-target indel efficiency in HEK293T surrogate-target assays",
            "operationally_equivalent_endpoints": False,
            "reason": "Units, timing, selection, Cas9 expression context, and background handling differ materially.",
        },
    }


def biological_domain_comparison() -> dict:
    differences = [
        {
            "factor": "cell_type",
            "deepspcas9": "HEK293T",
            "crispron_xiang_luo": "Stable HEK293T-SpCas9 line plus wild-type HEK293T background control",
            "classification": "MODERATE",
        },
        {
            "factor": "assay_protocol",
            "deepspcas9": "Paired guide/30-nt target lentiviral library; Cas9 delivered after cell-library generation",
            "crispron_xiang_luo": "Paired guide/37-bp surrogate lentiviral library transduced into stable Cas9 cells",
            "classification": "MAJOR",
        },
        {
            "factor": "Cas9_expression_context",
            "deepspcas9": "SpCas9 lentivirus, MOI 5, blasticidin selection",
            "crispron_xiang_luo": "PiggyBac-integrated TRE-SpCas9 stable line; basal Dox-free expression",
            "classification": "MAJOR",
        },
        {
            "factor": "experimental_platform",
            "deepspcas9": "Lentiviral integrated surrogate and targeted deep sequencing",
            "crispron_xiang_luo": "Lentiviral integrated surrogate and targeted amplicon sequencing",
            "classification": "MINOR",
        },
        {
            "factor": "gene_target_composition",
            "deepspcas9": "GeCKOv1, selected human/mouse genes, drug-resistance genes, synthetic extreme-GC targets, and laboratory targets",
            "crispron_xiang_luo": "Early coding exons from approximately 3,832-3,834 human protein-coding genes after off-target/library filters",
            "classification": "MAJOR",
        },
        {
            "factor": "measurement_timing",
            "deepspcas9": "Day 2.9 after Cas9 delivery",
            "crispron_xiang_luo": "Mean of Dox-free days 8 and 10 after guide/surrogate delivery",
            "classification": "MAJOR",
        },
        {
            "factor": "sequence_library_construction",
            "deepspcas9": "15,656 designed 30-nt targets in paired guide-target oligos",
            "crispron_xiang_luo": "12K oligo pool with 37-bp surrogate targets and third-generation lentiviral vector",
            "classification": "MODERATE",
        },
        {
            "factor": "target_locus_composition",
            "deepspcas9": "Human-derived, mouse-derived, and synthetic target sequences measured at integrated surrogates",
            "crispron_xiang_luo": "Human coding-exon-derived sequences measured at randomly integrated surrogates",
            "classification": "MAJOR",
        },
        {
            "factor": "endogenous_vs_synthetic_context",
            "deepspcas9": "Training labels from integrated surrogate targets",
            "crispron_xiang_luo": "Training labels from randomly integrated surrogate targets",
            "classification": "MINOR",
        },
        {
            "factor": "background_handling",
            "deepspcas9": "Project-local label traces to uncorrected indel fraction; publication also reports a background-corrected endpoint",
            "crispron_xiang_luo": "Wild-type controls used to remove background variants before total-indel estimation",
            "classification": "MAJOR",
        },
    ]
    return {
        "differences": differences,
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in differences).items())
        ),
        "major_domain_differences": [
            item["factor"] for item in differences if item["classification"] == "MAJOR"
        ],
        "unknown_metadata": [
            "Exact DeepSpCas9 Cas9-vector promoter and expression level in the assayed cells",
            "Exact equivalence between CRISPRon Quant_norm_efficiency and any recovered workbook field",
        ],
        "interpretation": "Both assays target the same broad molecular concept, but major operational domain differences can shift observed label scale and ranking.",
    }


def evaluate_integration_strategies(
    same_concept: bool,
    directly_comparable_scale: bool,
    major_domain_difference_n: int,
) -> dict:
    direct_status = (
        "POSSIBLE_WITH_CAUTION"
        if same_concept and directly_comparable_scale and major_domain_difference_n == 0
        else "NOT_RECOMMENDED"
    )
    return {
        "direct_pooling": {
            "classification": direct_status,
            "scientific_defensibility": "Low under current evidence because equal numeric targets would imply endpoint and unit equivalence not established by this audit.",
            "label_assumptions": "Requires a common calibrated response scale and exchangeable assay domains.",
            "risk_of_information_loss": "LOW",
            "risk_of_artificial_harmonization": "HIGH",
            "leakage_risk": "MODERATE; 48 exact overlaps require group-aware allocation before any future split.",
            "interpretability": "HIGH only if label equivalence is real; otherwise misleading.",
            "reproducibility": "Technically high, scientifically weak without a pre-registered scale protocol.",
            "canonical_pipeline_compatibility": "Low; current pipeline assumes one 0-1 canonical endpoint.",
        },
        "normalization_then_pooling": {
            "classification": "POSSIBLE_WITH_CAUTION",
            "scientific_defensibility": "Can remove dataset-level location/scale differences but cannot remove biological timing or assay effects.",
            "label_assumptions": "Assumes within-dataset relative distances have comparable meaning after normalization.",
            "risk_of_information_loss": "MODERATE",
            "risk_of_artificial_harmonization": "HIGH",
            "leakage_risk": "HIGH unless normalization parameters are estimated within each training fold only.",
            "interpretability": "Reduced; outputs cease to be direct indel fractions/percentages.",
            "reproducibility": "Possible with an explicit fold-local transformation protocol.",
            "canonical_pipeline_compatibility": "Requires a new experimental target definition; not a silent extension.",
        },
        "rank_based": {
            "classification": "POSSIBLE_WITH_CAUTION",
            "scientific_defensibility": "Defensible for guide prioritization if relative ordering is the declared target.",
            "label_assumptions": "Assumes within-dataset percentile ranks are comparable across library compositions.",
            "risk_of_information_loss": "HIGH; absolute efficiency and interval information are discarded.",
            "risk_of_artificial_harmonization": "MODERATE",
            "leakage_risk": "HIGH unless ranks/ECDFs are derived within training folds.",
            "interpretability": "Good for relative prioritization, poor for absolute activity.",
            "reproducibility": "High with a pre-registered fold-local ranking rule.",
            "canonical_pipeline_compatibility": "Changes the target and therefore requires a separate experimental branch.",
        },
        "multi_domain": {
            "classification": "RECOMMENDED",
            "scientific_defensibility": "Best preserves both observed endpoints while explicitly representing assay identity.",
            "label_assumptions": "Assumes shared sequence signal exists but permits domain-specific offsets/scales or losses.",
            "risk_of_information_loss": "LOW",
            "risk_of_artificial_harmonization": "LOW",
            "leakage_risk": "MODERATE; exact overlaps must remain in one split group across domains.",
            "interpretability": "High if domain-specific outputs/effects are reported separately.",
            "reproducibility": "High with fixed domain labels, grouped splits, and pre-registered objectives.",
            "canonical_pipeline_compatibility": "Requires a new experimental architecture but leaves the canonical pipeline untouched.",
        },
        "transfer_learning": {
            "classification": "POSSIBLE_WITH_CAUTION",
            "scientific_defensibility": "Plausible because both domains assay SpCas9 indels, but adaptation direction and validation policy must be pre-registered.",
            "label_assumptions": "Requires transferable sequence representations, not identical raw label scales.",
            "risk_of_information_loss": "LOW_TO_MODERATE",
            "risk_of_artificial_harmonization": "LOW",
            "leakage_risk": "MODERATE; overlap and adaptation validation must be grouped and isolated.",
            "interpretability": "Moderate; performance depends on direction and frozen/adapted components.",
            "reproducibility": "High if initialization, frozen layers, splits, and adaptation criteria are fixed.",
            "canonical_pipeline_compatibility": "Separate experimental branch; must not overwrite canonical artifacts.",
        },
        "external_development_domain": {
            "classification": "RECOMMENDED",
            "scientific_defensibility": "Strong conservative comparator that tests cross-domain development without asserting label exchangeability.",
            "label_assumptions": "Only assumes higher values reflect greater editing within each domain.",
            "risk_of_information_loss": "LOW",
            "risk_of_artificial_harmonization": "LOW",
            "leakage_risk": "LOW if the 48 shared sequences are excluded or explicitly grouped for any future evaluation.",
            "interpretability": "High for domain-specific generalization, but it does not exploit all data jointly.",
            "reproducibility": "High.",
            "canonical_pipeline_compatibility": "High as a separate development domain; the locked external test remains untouched.",
        },
    }


def decide_integration_strategy(
    same_concept: bool,
    directly_comparable_scale: bool,
    controlled_experiments_supported: bool,
) -> dict:
    if same_concept and directly_comparable_scale:
        recommendation = "DIRECT_POOLING_READY"
    elif same_concept and controlled_experiments_supported:
        recommendation = "MULTI_DOMAIN_EXPERIMENT_RECOMMENDED"
    elif same_concept:
        recommendation = "KEEP_DATASETS_SEPARATE"
    else:
        recommendation = "INSUFFICIENT_EVIDENCE"
    return {
        "recommended_strategy": recommendation,
        "immediate_pooling_justified": recommendation == "DIRECT_POOLING_READY",
        "controlled_integration_experiments_justified": bool(
            same_concept and controlled_experiments_supported
        ),
    }


def build_phase18a_audit(timestamp: str) -> dict:
    before = integrity_snapshot(EXPECTED_HASHES)
    require_expected_integrity(before)
    deep_raw, deep_canonical, xiang_luo = load_phase18a_inputs()
    semantics = label_semantics()
    biology = biological_domain_comparison()
    bridge = extract_shared_sequence_bridge(deep_raw, xiang_luo)
    canonical_bridge = extract_shared_sequence_bridge(deep_canonical, xiang_luo)
    bridge["deepspcas9_population"] = {
        "name": "raw provenance population",
        "n": int(len(deep_raw)),
        "reason": (
            "The primary diagnostic bridge covers all exact overlaps established "
            "by Phase 17G-R1 before canonical modeling eligibility filtering."
        ),
    }
    bridge["canonical_modeling_population_sensitivity"] = {
        "deepspcas9_n": int(len(deep_canonical)),
        "shared_30mer_n": canonical_bridge["n"],
        "finite_pair_n": canonical_bridge["finite_pair_n"],
        "pearson": canonical_bridge["pearson"],
        "spearman": canonical_bridge["spearman"],
        "kendall": canonical_bridge["kendall"],
        "monotonic_relationship": canonical_bridge["monotonic_relationship"],
        "interpretation": (
            "Sensitivity analysis only; it does not replace the all-overlap "
            "provenance bridge or authorize label harmonization."
        ),
    }
    sequence = sequence_domain_comparison(
        deep_canonical["sequence_30mer"].tolist(),
        xiang_luo["30mer_gRNA"].tolist(),
    )
    concept = semantics["underlying_concept_assessment"]
    scale_directly_comparable = bool(
        concept["operationally_equivalent_endpoints"]
        and semantics["deepspcas9"]["numerical_unit"]
        == semantics["crispron_xiang_luo"]["numerical_unit"]
    )
    controlled_experiments_supported = bool(
        concept["same_underlying_biological_concept"]
        and len(xiang_luo) >= 2000
        and xiang_luo["30mer_gRNA"].nunique() == len(xiang_luo)
    )
    decision = decide_integration_strategy(
        same_concept=concept["same_underlying_biological_concept"],
        directly_comparable_scale=scale_directly_comparable,
        controlled_experiments_supported=controlled_experiments_supported,
    )
    strategies = evaluate_integration_strategies(
        same_concept=concept["same_underlying_biological_concept"],
        directly_comparable_scale=scale_directly_comparable,
        major_domain_difference_n=len(biology["major_domain_differences"]),
    )
    after = integrity_snapshot(EXPECTED_HASHES)
    require_expected_integrity(after)
    if before != after:
        raise RuntimeError("Protected Phase 18A inputs changed during the audit")
    reasons = [
        "Phase 17G-R1 GO establishes sufficient accepted new data, not direct label exchangeability.",
        "Both labels measure sequence-linked SpCas9 indel efficiency in HEK293T integrated-surrogate assays.",
        "DeepSpCas9 activity is a day-2.9 fraction while Xiang/Luo is a Dox-free day-8/day-10 mean percentage.",
        "Cas9 expression, selection duration, background handling, and library/target composition differ materially.",
        "The raw-population bridge contains all 48 exact shared 30-mers; the canonical-population sensitivity contains 41. Neither authorizes a harmonization transform in Phase 18A.",
        "Multi-domain experiments can preserve original labels and dataset identity while testing shared versus domain-specific signal.",
        "A separate Xiang/Luo development-domain arm should be retained as a conservative comparator.",
    ]
    return {
        "phase": "18A",
        "protocol_version": "phase18a-cross-dataset-label-compatibility-v1",
        "timestamp": timestamp,
        "scope": {
            "audit_only": True,
            "no_model_training": True,
            "no_model_retraining": True,
            "no_model_evaluation": True,
            "no_moreno_access": True,
            "no_label_normalization": True,
            "no_label_transformation": True,
            "no_harmonization_fit": True,
            "no_training_dataset_created": True,
        },
        "datasets": {
            "deepspcas9": {
                "path": _portable(DEEPSPCAS9_PATH),
                "label_column": "activity",
                "sequence_column": "sequence_30mer",
                "raw_n": int(len(deep_raw)),
                "canonical_modeling_n": int(len(deep_canonical)),
                "geometry": "[0:4] upstream; [4:24] spacer20; [24:27] PAM; [27:30] downstream",
            },
            "crispron_xiang_luo": {
                "path": _portable(CRISPRON_PATH),
                "sheet": CRISPRON_SHEET,
                "source_components": ["LuoSpCas92020_min200", "Overlap_Luo_Kim2019"],
                "label_column": "HEK293T_indel_freq_avg_d8_d10",
                "sequence_column": "30mer_gRNA",
                "raw_n": int(len(xiang_luo)),
                "unique_30mer_n": int(xiang_luo["30mer_gRNA"].nunique()),
                "new_nonoverlapping_n": 10544,
            },
        },
        "label_semantics": semantics,
        "label_distribution_comparison": {
            "no_values_transformed": True,
            "numerical_scales_directly_comparable": scale_directly_comparable,
            "deepspcas9_canonical_modeling_population": label_statistics(
                deep_canonical["activity"]
            ),
            "crispron_xiang_luo": label_statistics(
                xiang_luo["HEK293T_indel_freq_avg_d8_d10"]
            ),
            "interpretation": "Different numeric ranges do not imply biological incompatibility, but the unit and endpoint differences prevent treating values as exchangeable regression targets without a controlled study.",
        },
        "shared_sequence_bridge": bridge,
        "biological_domain_comparison": biology,
        "sequence_domain_comparison": sequence,
        "integration_strategies": strategies,
        "immediate_pooling_justified": decision["immediate_pooling_justified"],
        "controlled_integration_experiments_justified": decision[
            "controlled_integration_experiments_justified"
        ],
        "recommended_strategy": decision["recommended_strategy"],
        "decision_reasons": reasons,
        "canonical_integrity": {
            "before": before,
            "after": after,
            "unchanged": before == after,
            "all_expected_hashes_match": all(
                item["matches_expected"] for item in after.values()
            ),
            "canonical_data_modified": False,
            "canonical_model_code_modified": False,
            "canonical_artifacts_modified": False,
            "locked_external_data_accessed": False,
        },
    }
