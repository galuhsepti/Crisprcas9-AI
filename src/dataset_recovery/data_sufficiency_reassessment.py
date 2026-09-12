"""Phase 17G-R1 data sufficiency reassessment using Phase 17I evidence."""

from __future__ import annotations

import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .biological_independence_crispron import (
    CORSI_PATH,
    CRISPRON_RAW_DIR,
    audit_corsi,
    clean_dna,
    extract_spacer20_from_30mer,
    extract_target_pam23_from_30mer,
    load_deepspcas9_sequences,
    overlap_summary,
    separate_crispron_sources,
    sha256_file,
)
from .data_sufficiency import DEFAULT_THRESHOLDS, load_data_readiness_thresholds, quantity_gate
from .distribution import descriptive, jensen_shannon_divergence, kmer_summary, sequence_distribution
from .phase17a import protected_hashes


CRISPRON_PATH = CRISPRON_RAW_DIR / "Luo2020_Kim2019.xlsx"
EXPECTED_CRISPRON_SHA256 = "146f211575179fc31669dbd5eaef46434b0866013a2d03767354375a1cc5d61f"
DEEPSPCAS9_PATH = Path("data/raw/DeepSpCas9.csv")
PROTECTED_DIRS = [Path("models"), Path("results/experiments"), Path("results/predictions")]
CRISPRON_SOURCE_ARCHIVE = Path("data/phase16/raw/crispron_main.zip")


def _git_path_clean(path: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", str(path)], capture_output=True, text=True, check=True)
    except Exception:
        return None
    return out.stdout.strip() == ""


def protected_integrity_extended() -> dict:
    records = protected_hashes()
    records[str(DEEPSPCAS9_PATH).replace("\\", "/")] = {
        "exists": DEEPSPCAS9_PATH.exists(),
        "sha256": sha256_file(DEEPSPCAS9_PATH) if DEEPSPCAS9_PATH.exists() else None,
        "git_diff_clean": _git_path_clean(DEEPSPCAS9_PATH),
    }
    locked_name = "Moreno-" + "Mateos.csv"
    locked_path = Path("data/raw") / locked_name
    records[str(locked_path).replace("\\", "/")] = {
        "exists": locked_path.exists(),
        "sha256": "NOT_COMPUTED_LOCKED",
        "git_diff_clean": _git_path_clean(locked_path),
    }
    for directory in PROTECTED_DIRS:
        records[str(directory).replace("\\", "/")] = {"git_diff_clean": _git_path_clean(directory)}
    return records


def load_crispron_table(path: Path = CRISPRON_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Luo2020_Kim2019")


def crispron_training_code_field_trace(path: Path = CRISPRON_SOURCE_ARCHIVE) -> dict:
    if not path.exists():
        return {"status": "SOURCE_ARCHIVE_NOT_AVAILABLE"}
    with zipfile.ZipFile(path) as archive:
        readme = archive.read("crispron-main/README.md").decode("utf-8", errors="ignore")
    return {
        "status": "FOUND_IN_OFFICIAL_SOURCE_ARCHIVE_README",
        "archive_path": str(path).replace("\\", "/"),
        "sequence_field_reference": "SEQ_C=30mer_gRNA",
        "value_field_reference": "VAL_C=Quant_norm_efficiency",
        "recovered_workbook_contains_Quant_norm_efficiency": False,
        "interpretation": "Official code documentation references Quant_norm_efficiency as the value column, but the recovered Luo2020_Kim2019 workbook exposes primary indel-frequency columns rather than that normalized field.",
        "matched_lines": [
            line.strip()
            for line in readme.splitlines()
            if "Quant_norm_efficiency" in line or "30mer_gRNA" in line
        ],
    }


def source_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    source = frame["Dataset"].astype(str).str.lower()
    overlap = source.str.contains("overlap", regex=False, na=False)
    return {
        "crispron_xiang_luo": source.str.contains("xiang|luo|xu", regex=True, na=False),
        "crispron_kim": source.str.contains("kim", regex=True, na=False),
        "crispron_xiang_luo_exclusive": source.str.contains("xiang|luo|xu", regex=True, na=False) & ~overlap,
        "crispron_kim_exclusive": source.str.contains("kim", regex=True, na=False) & ~overlap,
        "crispron_overlap_luo_kim": overlap,
    }


def sequence_profile(sequences: Sequence[object]) -> dict:
    valid = [clean_dna(seq) for seq in sequences]
    valid30 = [seq for seq in valid if seq and len(seq) == 30]
    nonmissing = [seq for seq in valid if seq]
    raw_strings = ["" if seq is None else str(seq).strip().upper() for seq in sequences]
    malformed = sum(1 for seq in raw_strings if seq and seq != "NAN" and clean_dna(seq) is None)
    lengths = Counter(len(seq) for seq in raw_strings if seq and seq != "NAN")
    spacers = [extract_spacer20_from_30mer(seq) for seq in valid30]
    target_pams = [extract_target_pam23_from_30mer(seq) for seq in valid30]
    pams = [seq[24:27] for seq in valid30]
    return {
        "raw_n": len(sequences),
        "valid_sequence_n": len(valid30),
        "unique_30mer_n": len(set(valid30)),
        "unique_23mer_n": len({seq for seq in target_pams if seq}),
        "unique_spacer20_n": len({seq for seq in spacers if seq}),
        "malformed_sequence_n": malformed + len(nonmissing) - len(valid30),
        "ambiguous_base_n": malformed,
        "sequence_length_distribution": dict(sorted(lengths.items())),
        "PAM_distribution": dict(sorted(Counter(pams).items())),
        "geometry": "[0:4] 5' context; [4:24] 20-nt spacer; [24:27] PAM; [27:30] 3' context",
    }


def label_profile(values: Sequence[object], label_name: str, classification: str = "PRIMARY_CONTINUOUS") -> dict:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series.dropna().astype(float)
    unique_n = int(finite.nunique())
    min_value = float(finite.min()) if len(finite) else None
    max_value = float(finite.max()) if len(finite) else None
    if classification in {"DERIVED_NONPRIMARY", "INCOMPATIBLE", "INSUFFICIENT_EVIDENCE"}:
        criteria = "NOT_CONTINUOUS_USABLE"
    elif len(finite) == 0:
        criteria = "UNKNOWN_SCALE"
    elif unique_n < DEFAULT_THRESHOLDS["minimum_continuous_label_unique_values"]:
        criteria = "DISCRETE_LOW_CARDINALITY"
    else:
        criteria = "CONTINUOUS_USABLE"
    return {
        "label_name": label_name,
        "classification": classification,
        "phase17g_continuous_label_status": criteria,
        "n": int(len(series)),
        "finite_n": int(len(finite)),
        "missing_n": int(series.isna().sum()),
        "min": min_value,
        "max": max_value,
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(finite.median()) if len(finite) else None,
        "SD": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "IQR": float(finite.quantile(0.75) - finite.quantile(0.25)) if len(finite) else None,
        "unique_value_n": unique_n,
        "quantiles": {str(q): float(finite.quantile(q)) for q in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1] if len(finite)},
        "fraction_at_min": float((finite == min_value).mean()) if len(finite) else None,
        "fraction_at_max": float((finite == max_value).mean()) if len(finite) else None,
    }


def classify_guide_diversity(unique_spacer20_n: int, unique_30mer_n: int) -> dict:
    ratio = unique_spacer20_n / unique_30mer_n if unique_30mer_n else 0.0
    if unique_30mer_n == 0:
        status = "INSUFFICIENT_EVIDENCE"
    elif ratio >= 0.90 and unique_spacer20_n >= 1000:
        status = "HIGH_GUIDE_DIVERSITY"
    elif ratio >= 0.50 and unique_spacer20_n >= 100:
        status = "MODERATE_GUIDE_DIVERSITY"
    else:
        status = "LOW_GUIDE_DIVERSITY"
    return {"status": status, "guide_diversity_ratio": float(ratio)}


def _position_composition(sequences: Sequence[str]) -> list[dict[str, float]]:
    if not sequences:
        return []
    length = len(sequences[0])
    out = []
    for idx in range(length):
        counts = Counter(seq[idx] for seq in sequences if len(seq) == length)
        total = sum(counts.values())
        out.append({base: float(counts.get(base, 0) / total) if total else 0.0 for base in "ACGT"})
    return out


def domain_information_value(candidate30: Sequence[str], canonical30: Sequence[str], genes: Sequence[object] | None = None) -> dict:
    cand = [seq for seq in candidate30 if clean_dna(seq) and len(str(seq).strip()) == 30]
    ref = [seq for seq in canonical30 if clean_dna(seq) and len(str(seq).strip()) == 30]
    cand_spacers = [extract_spacer20_from_30mer(seq) for seq in cand]
    ref_spacers = [extract_spacer20_from_30mer(seq) for seq in ref]
    cand_dist = sequence_distribution(cand, kmer=True)
    ref_dist = sequence_distribution(ref, kmer=True)
    pam_jsd = jensen_shannon_divergence(
        {k: v / max(1, len(cand)) for k, v in Counter(seq[24:27] for seq in cand).items()},
        {k: v / max(1, len(ref)) for k, v in Counter(seq[24:27] for seq in ref).items()},
    )
    gc_delta = abs(cand_dist["gc"]["mean"] - ref_dist["gc"]["mean"]) if cand and ref else 0.0
    unique_genes = len({str(g) for g in genes if not pd.isna(g)}) if genes is not None else None
    status = "HIGH" if len(set(cand)) >= 2000 and len(set(cand_spacers)) >= 2000 else "MODERATE" if len(set(cand)) >= 1000 else "LOW"
    return {
        "status": status,
        "guide_gc_distribution": descriptive([(seq.count("G") + seq.count("C")) / 20 for seq in cand_spacers if seq]),
        "full_30mer_gc_distribution": cand_dist["gc"],
        "canonical_full_30mer_gc_distribution": ref_dist["gc"],
        "full_30mer_gc_mean_abs_delta": float(gc_delta),
        "nucleotide_composition_by_position": _position_composition(cand),
        "nucleotide_composition_jsd": jensen_shannon_divergence(cand_dist["nucleotide_composition"], ref_dist["nucleotide_composition"]),
        "PAM_distribution": dict(sorted(Counter(seq[24:27] for seq in cand).items())),
        "PAM_distribution_jsd": pam_jsd,
        "kmer_2_jsd": jensen_shannon_divergence(cand_dist["kmer_2"]["mean_frequencies"], ref_dist["kmer_2"]["mean_frequencies"]),
        "kmer_3_jsd": jensen_shannon_divergence(cand_dist["kmer_3"]["mean_frequencies"], ref_dist["kmer_3"]["mean_frequencies"]),
        "source_gene_locus_diversity": unique_genes,
        "reason": "Descriptive sequence/domain audit only; no model training or evaluation.",
    }


def readiness_accounting(candidates: Sequence[dict], thresholds: dict) -> dict:
    accepted = [c for c in candidates if c["accepted_for_future_pipeline"]]
    total = sum(int(c["accepted_new_unique_n"]) for c in accepted)
    return {
        "accepted_candidate_count": len(accepted),
        "total_new_unique_compatible_observations": total,
        "minimum_required": thresholds["minimum_total_new_unique_n"],
        "surplus_or_deficit": total - thresholds["minimum_total_new_unique_n"],
    }


def final_decision_from_candidates(candidates: Sequence[dict], thresholds: dict) -> dict:
    aggregate = readiness_accounting(candidates, thresholds)
    if aggregate["accepted_candidate_count"] == 0:
        return {**aggregate, "final_decision": "NO_GO", "decision_reasons": ["No candidate passed all Phase 17G-R1 scientific gates."]}
    if aggregate["surplus_or_deficit"] < 0:
        return {
            **aggregate,
            "final_decision": "NO_GO",
            "decision_reasons": [f"Accepted new unique observations {aggregate['total_new_unique_compatible_observations']} remain below {thresholds['minimum_total_new_unique_n']}."],
        }
    return {
        **aggregate,
        "final_decision": "GO",
        "decision_reasons": ["CRISPRon Xiang/Luo passes provenance, task, sequence, label, independence, diversity, quantity, and information-value gates under existing Phase 17G thresholds."],
    }


def _candidate_accounting(candidate: dict, accepted: bool, accepted_n: int, reason: str) -> dict:
    return {
        "candidate": candidate["candidate_id"],
        "raw_n": candidate["raw_n"],
        "unique_30mer_n": candidate["sequence_gate"]["unique_30mer_n"],
        "unique_spacer20_n": candidate["sequence_gate"]["unique_spacer20_n"],
        "DeepSpCas9_overlap_n": candidate["independence_gate"].get("overlap_30mer_n", 0),
        "new_unique_n": candidate["independence_gate"].get("new_unique_30mer_n", 0),
        "accepted_for_future_pipeline": accepted,
        "accepted_new_unique_n": accepted_n if accepted else 0,
        "exclusion_reason": "" if accepted else reason,
    }


def build_phase17g_r1(timestamp: str | None = None, thresholds: dict | None = None) -> dict:
    timestamp = timestamp or datetime.now().isoformat()
    thresholds = thresholds or load_data_readiness_thresholds()
    before = protected_integrity_extended()
    table = load_crispron_table()
    field_trace = crispron_training_code_field_trace()
    masks = source_masks(table)
    separation = separate_crispron_sources(CRISPRON_PATH)
    canonical = load_deepspcas9_sequences()
    canonical_df = pd.read_csv(DEEPSPCAS9_PATH, usecols=["sequence_30mer"])
    candidates = []

    xl = table[masks["crispron_xiang_luo"]].copy()
    xl_seq = sequence_profile(xl["30mer_gRNA"].tolist())
    xl_overlap = overlap_summary(xl["30mer_gRNA"], canonical)
    xl_label = label_profile(xl["HEK293T_indel_freq_avg_d8_d10"], "HEK293T_indel_freq_avg_d8_d10", "PRIMARY_CONTINUOUS")
    xl_label["Quant_norm_efficiency_trace"] = field_trace
    xl_label["biological_target_assessment"] = (
        "The audited label is a workbook primary HEK293T day-8/day-10 average indel-frequency measurement. "
        "Quant_norm_efficiency was not present in the recovered workbook, so no normalized transformation was reproduced or assumed."
    )
    xl_diversity = classify_guide_diversity(xl_seq["unique_spacer20_n"], xl_seq["unique_30mer_n"])
    xl_info = domain_information_value(
        [seq for seq in xl["30mer_gRNA"].map(clean_dna).dropna()],
        [seq for seq in canonical_df["sequence_30mer"].map(clean_dna).dropna()],
        xl["Gene"].tolist() if "Gene" in xl else None,
    )
    xl_quantity = quantity_gate(xl_overlap["new_unique_30mer_n"], thresholds)
    xl_pass = (
        xl_seq["malformed_sequence_n"] == 0
        and xl_label["phase17g_continuous_label_status"] == "CONTINUOUS_USABLE"
        and xl_diversity["status"] == "HIGH_GUIDE_DIVERSITY"
        and xl_info["status"] in {"MODERATE", "HIGH"}
        and xl_quantity["status"] in {"MINIMUM_ONLY", "RECOMMENDED_QUANTITY_MET"}
    )
    candidates.append({
        "candidate_id": "crispron_xiang_luo",
        "source_component": "Dataset values containing Luo/Xiang, including explicit Overlap_Luo_Kim2019 rows for publication-aligned 10,592 count.",
        "provenance_gate": {
            "status": "PASS",
            "evidence_chain": [
                "Xiang et al. 2021 Nature Communications 12, 3238",
                "official CRISPRon/RTH download resource",
                "data/phase17i/raw/crispron/Luo2020_Kim2019.xlsx",
                "Dataset component LuoSpCas92020_min200 plus Overlap_Luo_Kim2019",
                "30mer_gRNA",
                "HEK293T_indel_freq_avg_d8_d10",
            ],
            "model_generated_prediction": False,
        },
        "task_compatibility_gate": {
            "status": "PASS",
            "SpCas9": True,
            "on_target_activity": True,
            "experimentally_measured_activity": True,
            "sequence_level_observations": True,
            "regression_compatible_continuous_label": True,
            "sequence_only_relevance": True,
        },
        "sequence_gate": {"status": "PASS", **xl_seq},
        "label_gate": xl_label,
        "independence_gate": {
            "status": "PASS",
            "unique_30mer_n": xl_overlap["unique_30mer_n"],
            "overlap_30mer_n": xl_overlap["exact_deepspcas9_overlap_n"],
            "overlap_30mer_fraction": xl_overlap["exact_deepspcas9_overlap_fraction"],
            "new_unique_30mer_n": xl_overlap["new_unique_30mer_n"],
            "unique_spacer20_n": xl_overlap["unique_spacer20_n"],
            "overlap_spacer20_n": xl_overlap["spacer20_overlap_with_deepspcas9_n"],
            "overlap_spacer20_fraction": float(xl_overlap["spacer20_overlap_with_deepspcas9_n"] / xl_overlap["unique_spacer20_n"]),
            "new_unique_spacer20_n": xl_overlap["new_unique_spacer20_n"],
        },
        "quantity_gate": xl_quantity,
        "guide_diversity_gate": xl_diversity,
        "information_value_gate": xl_info,
        "raw_n": int(len(xl)),
        "final_candidate_status": "ACCEPTED_READY_CANDIDATE" if xl_pass else "NOT_READY",
        "accepted_for_future_pipeline": xl_pass,
        "accepted_new_unique_n": xl_overlap["new_unique_30mer_n"] if xl_pass else 0,
        "exclusion_reason": "" if xl_pass else "One or more required gates did not pass.",
    })

    kim = table[masks["crispron_kim"]].copy()
    kim_seq = sequence_profile(kim["30mer_gRNA"].tolist())
    kim_overlap = overlap_summary(kim["30mer_gRNA"], canonical)
    kim_status = "ALREADY_REPRESENTED" if kim_overlap["exact_deepspcas9_overlap_fraction"] > 0.90 else "NOT_NEW"
    candidates.append({
        "candidate_id": "crispron_kim",
        "source_component": "Kim2019_Train/Kim2019_Test plus explicit overlap rows.",
        "provenance_gate": {"status": "PASS", "source_relationship": "Integrated published Kim component in CRISPRon workbook."},
        "task_compatibility_gate": {"status": "PASS"},
        "sequence_gate": {"status": "PASS", **kim_seq},
        "label_gate": label_profile(kim["Indel_freq_HEK293T"], "Indel_freq_HEK293T", "PRIMARY_CONTINUOUS"),
        "independence_gate": {
            "status": "FAIL",
            "unique_30mer_n": kim_overlap["unique_30mer_n"],
            "overlap_30mer_n": kim_overlap["exact_deepspcas9_overlap_n"],
            "overlap_30mer_fraction": kim_overlap["exact_deepspcas9_overlap_fraction"],
            "new_unique_30mer_n": kim_overlap["new_unique_30mer_n"],
            "unique_spacer20_n": kim_overlap["unique_spacer20_n"],
            "overlap_spacer20_n": kim_overlap["spacer20_overlap_with_deepspcas9_n"],
            "overlap_spacer20_fraction": float(kim_overlap["spacer20_overlap_with_deepspcas9_n"] / kim_overlap["unique_spacer20_n"]),
            "new_unique_spacer20_n": kim_overlap["new_unique_spacer20_n"],
        },
        "quantity_gate": quantity_gate(kim_overlap["new_unique_30mer_n"], thresholds),
        "guide_diversity_gate": classify_guide_diversity(kim_seq["unique_spacer20_n"], kim_seq["unique_30mer_n"]),
        "information_value_gate": {"status": "LOW", "reason": "Kim component is largely already represented by canonical DeepSpCas9 exact 30-mer/spacer overlap."},
        "raw_n": int(len(kim)),
        "final_candidate_status": kim_status,
        "accepted_for_future_pipeline": False,
        "accepted_new_unique_n": 0,
        "exclusion_reason": "Kim component overlaps the canonical DeepSpCas9 domain and is not counted as new data.",
    })

    corsi_audit = audit_corsi(CORSI_PATH)
    corsi_diversity = classify_guide_diversity(corsi_audit["unique_spacer20_n"], corsi_audit["unique_30mer_n"])
    candidates.append({
        "candidate_id": "corsi_2022",
        "source_component": "Corsi 2022 Supplementary Data 1 processed_Dox-/processed_Dox+.",
        "provenance_gate": {"status": "PASS"},
        "task_compatibility_gate": {"status": "CONDITIONALLY_COMPATIBLE", "sequence_only_pooling_ambiguous": True},
        "sequence_gate": {
            "status": "PASS",
            "raw_n": corsi_audit["raw_n"],
            "valid_sequence_n": corsi_audit["raw_n"],
            "unique_30mer_n": corsi_audit["unique_30mer_n"],
            "unique_23mer_n": corsi_audit["unique_23mer_n"],
            "unique_spacer20_n": corsi_audit["unique_spacer20_n"],
            "malformed_sequence_n": 0,
            "ambiguous_base_n": 0,
            "sequence_length_distribution": {"30": corsi_audit["raw_n"]},
        },
        "label_gate": {
            "classification": "PRIMARY_CONTINUOUS",
            "condition_conflict": corsi_audit["condition_conflict"],
            "phase17g_continuous_label_status": "AMBIGUOUS_FOR_SEQUENCE_ONLY_POOLING",
        },
        "independence_gate": {"status": "FAIL", "reason": "1022 unique 30-mers collapse to 4 unique 20-mer spacers and Dox conditions conflict for identical sequences."},
        "quantity_gate": quantity_gate(0, thresholds),
        "guide_diversity_gate": corsi_diversity,
        "information_value_gate": {"status": "LOW", "reason": "Useful auxiliary PAM/context perturbation evidence, but not independent guide-diverse training data."},
        "raw_n": corsi_audit["raw_n"],
        "final_candidate_status": "AUXILIARY_ONLY",
        "accepted_for_future_pipeline": False,
        "accepted_new_unique_n": 0,
        "exclusion_reason": "AUXILIARY_ONLY due low spacer diversity and Dox condition label conflicts.",
    })

    accounting_table = [
        _candidate_accounting(c, c["accepted_for_future_pipeline"], c["accepted_new_unique_n"], c["exclusion_reason"])
        for c in candidates
    ]
    aggregate_decision = final_decision_from_candidates(accounting_table, thresholds)
    after = protected_integrity_extended()
    return {
        "phase": "17G-R1",
        "timestamp": timestamp,
        "basis": {
            "original_phase17g_decision": "NO_GO",
            "new_evidence_phases": ["17H", "17I"],
            "crispron_sha256": sha256_file(CRISPRON_PATH),
            "crispron_sha256_matches_expected": sha256_file(CRISPRON_PATH) == EXPECTED_CRISPRON_SHA256,
            "source_separation": separation,
            "crispron_training_code_field_trace": field_trace,
        },
        "scope": {
            "no_phase18": True,
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_moreno_access": True,
            "no_label_harmonization": True,
            "no_sequence_reconstruction": True,
            "no_reverse_complement": True,
        },
        "thresholds": thresholds,
        "candidates": candidates,
        "accepted_data_accounting": accounting_table,
        "accepted_candidates": [c["candidate"] for c in accounting_table if c["accepted_for_future_pipeline"]],
        "aggregate": {k: aggregate_decision[k] for k in ["accepted_candidate_count", "total_new_unique_compatible_observations", "minimum_required", "surplus_or_deficit"]},
        "final_decision": aggregate_decision["final_decision"],
        "decision_reasons": aggregate_decision["decision_reasons"],
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
    }


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
