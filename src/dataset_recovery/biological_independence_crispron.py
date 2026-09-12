"""Phase 17I biological-independence and CRISPRon recovery audit."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .phase17a import protected_hashes


CORSI_PATH = Path("data/phase17h/raw/corsi_2022_supplementary_data_1.xlsx")
CRISPRON_URL = "https://rth.dk/resources/crispr/crispron/downloads//Luo2020_Kim2019.xlsx"
CRISPRON_FILENAME = "Luo2020_Kim2019.xlsx"
CRISPRON_RAW_DIR = Path("data/phase17i/raw/crispron")
MANIFEST_DIR = Path("data/phase17i/manifests")
DEEPSPCAS9_PATH = Path("data/raw/DeepSpCas9.csv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_dna(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    seq = str(value).strip().upper()
    if not seq or seq == "NAN" or set(seq) - set("ACGT"):
        return None
    return seq


def extract_spacer20_from_30mer(seq: object) -> str | None:
    seq = clean_dna(seq)
    if seq is None or len(seq) != 30:
        return None
    return seq[4:24]


def extract_target_pam23_from_30mer(seq: object) -> str | None:
    seq = clean_dna(seq)
    if seq is None or len(seq) != 30:
        return None
    return seq[4:27]


def unique_count(values: Iterable[object]) -> int:
    return len({v for v in values if v is not None})


def effective_guide_diversity_ratio(unique_spacer20_n: int, unique_30mer_n: int) -> float:
    return float(unique_spacer20_n / unique_30mer_n) if unique_30mer_n else 0.0


def detect_condition_conflicts(frame: pd.DataFrame, sequence_col: str, condition_col: str, label_col: str) -> dict:
    grouped = defaultdict(list)
    for _, row in frame.iterrows():
        seq = clean_dna(row.get(sequence_col))
        if seq is None:
            continue
        label = row.get(label_col)
        if pd.isna(label):
            continue
        grouped[seq].append((str(row.get(condition_col)), float(label)))
    diffs = []
    for records in grouped.values():
        conditions = {c for c, _ in records}
        labels = [v for _, v in records]
        if len(conditions) > 1 and max(labels) != min(labels):
            diffs.append(max(labels) - min(labels))
    abs_diffs = np.asarray(diffs, dtype=float)
    return {
        "shared_sequence_n": sum(1 for records in grouped.values() if len({c for c, _ in records}) > 1),
        "shared_sequence_with_label_difference_n": int(abs_diffs.size),
        "median_absolute_label_difference": float(np.median(abs_diffs)) if abs_diffs.size else 0.0,
        "mean_absolute_label_difference": float(np.mean(abs_diffs)) if abs_diffs.size else 0.0,
        "pooling_creates_target_ambiguity": bool(abs_diffs.size),
    }


def _frame_with_geometry(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = df.copy()
    out["condition"] = condition
    out["sequence_30mer_clean"] = out["30mer_gRNA"].map(clean_dna)
    out["spacer20_clean"] = out["gRNA"].map(clean_dna)
    out["spacer20_from_30mer"] = out["30mer_gRNA"].map(extract_spacer20_from_30mer)
    out["target_pam23_from_30mer"] = out["30mer_gRNA"].map(extract_target_pam23_from_30mer)
    return out


def _observation_frequency(counter: Counter) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(counter.values()).items())}


def _corsi_condition_summary(df: pd.DataFrame) -> dict:
    valid30 = df["sequence_30mer_clean"].dropna()
    valid23 = df["target_pam23_from_30mer"].dropna()
    spacer = df["spacer20_clean"].dropna()
    per_spacer = Counter(spacer)
    context = df.groupby("spacer20_clean", dropna=True).agg(
        unique_30mer_n=("sequence_30mer_clean", "nunique"),
        unique_pam_n=("PAM", "nunique"),
        unique_pam4_n=("PAM(4bp)", "nunique"),
        unique_pam_context_n=("PAM context", "nunique"),
    )
    return {
        "raw_row_count": int(len(df)),
        "valid_30mer_count": int(valid30.size),
        "unique_30mer_count": int(valid30.nunique()),
        "unique_23mer_target_pam_count": int(valid23.nunique()),
        "unique_20mer_spacer_count": int(spacer.nunique()),
        "observations_per_20mer_frequency_distribution": _observation_frequency(per_spacer),
        "pam_context_diversity_per_spacer": {
            "median_unique_30mer_per_spacer": float(context["unique_30mer_n"].median()) if len(context) else 0.0,
            "max_unique_30mer_per_spacer": int(context["unique_30mer_n"].max()) if len(context) else 0,
            "median_unique_pam_context_per_spacer": float(context["unique_pam_context_n"].median()) if len(context) else 0.0,
            "max_unique_pam_context_per_spacer": int(context["unique_pam_context_n"].max()) if len(context) else 0,
        },
        "gene_or_locus_column": "gRNA_ID",
        "unique_genes_loci_if_explicitly_available": int(df["gRNA_ID"].nunique()) if "gRNA_ID" in df else None,
    }


def audit_corsi(path: Path = CORSI_PATH) -> dict:
    dox_minus = _frame_with_geometry(pd.read_excel(path, sheet_name="processed_Dox-"), "Dox-")
    dox_plus = _frame_with_geometry(pd.read_excel(path, sheet_name="processed_Dox+"), "Dox+")
    combined = pd.concat([dox_minus, dox_plus], ignore_index=True)
    shared_spacers = set(dox_minus["spacer20_clean"].dropna()) & set(dox_plus["spacer20_clean"].dropna())
    shared_30 = set(dox_minus["sequence_30mer_clean"].dropna()) & set(dox_plus["sequence_30mer_clean"].dropna())
    conflict = detect_condition_conflicts(combined, "sequence_30mer_clean", "condition", "Indel frequency (% avg D6-D10)")
    unique30 = int(combined["sequence_30mer_clean"].nunique())
    unique20 = int(combined["spacer20_clean"].nunique())
    condition_pairs = combined[["spacer20_clean", "condition"]].dropna().drop_duplicates()
    suitability = "AUXILIARY_ONLY" if conflict["pooling_creates_target_ambiguity"] else "CONDITIONALLY_SUITABLE"
    if effective_guide_diversity_ratio(unique20, unique30) < 0.25:
        suitability = "AUXILIARY_ONLY"
    return {
        "raw_n": int(len(combined)),
        "corsi_raw_observation_n": int(len(combined)),
        "corsi_unique_30mer_n": unique30,
        "corsi_unique_23mer_n": int(combined["target_pam23_from_30mer"].nunique()),
        "corsi_unique_spacer20_n": unique20,
        "corsi_unique_spacer_condition_pairs_n": int(len(condition_pairs)),
        "unique_30mer_n": unique30,
        "unique_23mer_n": int(combined["target_pam23_from_30mer"].nunique()),
        "unique_spacer20_n": unique20,
        "effective_guide_diversity_ratio": effective_guide_diversity_ratio(unique20, unique30),
        "processed_Dox_minus": _corsi_condition_summary(dox_minus),
        "processed_Dox_plus": _corsi_condition_summary(dox_plus),
        "joint": _corsi_condition_summary(combined),
        "shared_20mer_spacers_between_conditions": len(shared_spacers),
        "identical_30mers_shared_between_conditions": len(shared_30),
        "condition_conflict": conflict,
        "pam_context_variant_conclusion": "Unique 30-mers are primarily PAM/context variants over a small spacer set.",
        "training_suitability": suitability,
        "training_suitability_reason": "Sequence-only pooling would assign different labels to identical 30-mers across Dox conditions, and the 30-mer count substantially overstates independent spacer diversity.",
    }


def retrieve_crispron(dest_dir: Path = CRISPRON_RAW_DIR, source_url: str = CRISPRON_URL, timestamp: str | None = None) -> dict:
    timestamp = timestamp or datetime.now().isoformat()
    dest_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / CRISPRON_FILENAME
    request = urllib.request.Request(source_url, headers={"User-Agent": "Phase17I-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            ctype = response.headers.get("Content-Type")
            with dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        manifest = {
            "retrieval_status": "FAILED_MANUAL_RECOVERY_REQUIRED",
            "source_url": source_url,
            "expected_filename": CRISPRON_FILENAME,
            "retrieval_timestamp": timestamp,
            "error": str(exc),
        }
        (MANIFEST_DIR / "crispron_manual_recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {**manifest, "local_path": None, "file_sha256": None, "file_size_bytes": None, "http_content_type": None}
    payload = {
        "retrieval_status": "RECOVERED",
        "source_url": source_url,
        "retrieval_timestamp": timestamp,
        "original_filename": CRISPRON_FILENAME,
        "local_path": str(dest).replace("\\", "/"),
        "file_size_bytes": dest.stat().st_size,
        "file_sha256": sha256_file(dest),
        "http_content_type": ctype,
        "publication_relationship": "Official CRISPRon/RTH training-data download associated with Xiang et al. 2021 Nature Communications 12, 3238.",
    }
    (MANIFEST_DIR / "crispron_recovery_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _column_matches(columns: Iterable[object], needles: Iterable[str]) -> list[str]:
    found = []
    for col in columns:
        lower = str(col).lower()
        if any(n in lower for n in needles):
            found.append(str(col))
    return found


def inventory_crispron_workbook(path: Path) -> dict:
    xls = pd.ExcelFile(path)
    sheets = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        sheets.append({
            "sheet_name": sheet,
            "row_count": int(len(df)),
            "column_names": [str(c) for c in df.columns],
            "sequence_columns": _column_matches(df.columns, ["30mer", "grna", "sgrna", "sequence", "spacer"]),
            "label_columns": _column_matches(df.columns, ["efficiency", "indel", "activity", "label", "score"]),
            "dataset_source_columns": _column_matches(df.columns, ["dataset", "source", "study", "author", "publication", "origin"]),
            "partition_columns": _column_matches(df.columns, ["train", "valid", "test", "fold", "partition"]),
            "has_30mer_gRNA": "30mer_gRNA" in df.columns,
            "has_Quant_norm_efficiency": "Quant_norm_efficiency" in df.columns,
        })
    return {"sheet_names": xls.sheet_names, "sheets": sheets}


def separate_crispron_sources(path: Path) -> dict:
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df["_sheet"] = sheet
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    source_cols = _column_matches(combined.columns, ["dataset", "source", "study", "author", "publication", "origin"])
    seq_col = "30mer_gRNA" if "30mer_gRNA" in combined.columns else None
    if not source_cols or not seq_col:
        return {
            "source_identity_recoverable": False,
            "source_columns": source_cols,
            "sequence_column": seq_col,
            "xiang_luo": {"raw_n": None, "unique_30mer_n": None, "unique_spacer20_n": None},
            "kim": {"raw_n": None, "unique_30mer_n": None, "unique_spacer20_n": None},
            "overlap_luo_kim": {"raw_n": None, "unique_30mer_n": None, "unique_spacer20_n": None},
            "other": {"raw_n": None, "unique_30mer_n": None, "unique_spacer20_n": None},
        }
    source_col = source_cols[0]
    tmp = combined.copy()
    tmp["_source_text"] = tmp[source_col].astype(str).str.lower()
    tmp["_seq30"] = tmp[seq_col].map(clean_dna)
    tmp["_spacer20"] = tmp[seq_col].map(extract_spacer20_from_30mer)
    overlap_mask = tmp["_source_text"].str.contains("overlap", regex=False, na=False)
    masks = {
        "xiang_luo": tmp["_source_text"].str.contains("xiang|luo|xu", regex=True, na=False),
        "kim": tmp["_source_text"].str.contains("kim", regex=True, na=False),
        "xiang_luo_exclusive": tmp["_source_text"].str.contains("xiang|luo|xu", regex=True, na=False) & ~overlap_mask,
        "kim_exclusive": tmp["_source_text"].str.contains("kim", regex=True, na=False) & ~overlap_mask,
        "overlap_luo_kim": overlap_mask,
    }
    masks["other"] = ~(masks["xiang_luo"] | masks["kim"])

    def summarize(mask: pd.Series) -> dict:
        part = tmp[mask]
        return {
            "raw_n": int(len(part)),
            "unique_30mer_n": int(part["_seq30"].dropna().nunique()),
            "unique_spacer20_n": int(part["_spacer20"].dropna().nunique()),
        }

    return {
        "source_identity_recoverable": True,
        "source_column_used": source_col,
        "sequence_column": seq_col,
        "xiang_luo": summarize(masks["xiang_luo"]),
        "kim": summarize(masks["kim"]),
        "xiang_luo_exclusive": summarize(masks["xiang_luo_exclusive"]),
        "kim_exclusive": summarize(masks["kim_exclusive"]),
        "overlap_luo_kim": summarize(masks["overlap_luo_kim"]),
        "other": summarize(masks["other"]),
    }


def load_deepspcas9_sequences(path: Path = DEEPSPCAS9_PATH) -> dict:
    df = pd.read_csv(path)
    return {
        "unique_30mer": set(df["sequence_30mer"].map(clean_dna).dropna()),
        "unique_spacer20": set(df["guide_sequence"].map(clean_dna).dropna()),
    }


def overlap_summary(candidate30: Iterable[object], canonical: dict) -> dict:
    c30 = {clean_dna(v) for v in candidate30}
    c30.discard(None)
    c20 = {extract_spacer20_from_30mer(v) for v in c30}
    c20.discard(None)
    o30 = c30 & canonical["unique_30mer"]
    o20 = c20 & canonical["unique_spacer20"]
    return {
        "raw_n": len(list(candidate30)) if not isinstance(candidate30, pd.Series) else int(candidate30.size),
        "unique_30mer_n": len(c30),
        "exact_deepspcas9_overlap_n": len(o30),
        "exact_deepspcas9_overlap_fraction": float(len(o30) / len(c30)) if c30 else 0.0,
        "new_unique_30mer_n": len(c30 - canonical["unique_30mer"]),
        "unique_spacer20_n": len(c20),
        "spacer20_overlap_with_deepspcas9_n": len(o20),
        "new_unique_spacer20_n": len(c20 - canonical["unique_spacer20"]),
    }


def audit_crispron(path: Path | None, retrieval: dict) -> dict:
    base = {
        "retrieval_status": retrieval["retrieval_status"],
        "file_sha256": retrieval.get("file_sha256"),
        "source_url": retrieval.get("source_url"),
        "file_size_bytes": retrieval.get("file_size_bytes"),
        "http_content_type": retrieval.get("http_content_type"),
        "publication_relationship": retrieval.get("publication_relationship"),
        "sources": [],
    }
    if not path or not path.exists() or retrieval["retrieval_status"] != "RECOVERED":
        return {**base, "xiang_luo": {}, "kim": {}, "overlap_with_deepspcas9": {}, "label_audit": {}, "preliminary_status": "BLOCKED"}
    inventory = inventory_crispron_workbook(path)
    separation = separate_crispron_sources(path)
    canonical = load_deepspcas9_sequences()
    overlap = {}
    label_audit = {"status": "INSUFFICIENT_EVIDENCE"}
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df["_sheet"] = sheet
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "30mer_gRNA" in combined.columns:
        overlap["all_recoverable"] = overlap_summary(combined["30mer_gRNA"], canonical)
        if separation["source_identity_recoverable"]:
            source_col = separation["source_column_used"]
            source_text = combined[source_col].astype(str).str.lower()
            overlap_mask = source_text.str.contains("overlap", regex=False, na=False)
            masks = {
                "xiang_luo": source_text.str.contains("xiang|luo|xu", regex=True, na=False),
                "kim": source_text.str.contains("kim", regex=True, na=False),
                "xiang_luo_exclusive": source_text.str.contains("xiang|luo|xu", regex=True, na=False) & ~overlap_mask,
                "kim_exclusive": source_text.str.contains("kim", regex=True, na=False) & ~overlap_mask,
                "overlap_luo_kim": overlap_mask,
            }
            masks["other"] = ~(masks["xiang_luo"] | masks["kim"])
            for name, mask in masks.items():
                overlap[name] = overlap_summary(combined.loc[mask, "30mer_gRNA"], canonical)
    label_col = "Quant_norm_efficiency" if "Quant_norm_efficiency" in combined.columns else (
        "HEK293T_indel_freq_avg_d8_d10" if "HEK293T_indel_freq_avg_d8_d10" in combined.columns else (
            "Indel_freq_HEK293T" if "Indel_freq_HEK293T" in combined.columns else None
        )
    )
    if label_col:
        values = pd.to_numeric(combined[label_col], errors="coerce").dropna()
        classification = "NORMALIZED_PRIMARY_CONTINUOUS" if label_col == "Quant_norm_efficiency" else "PRIMARY_CONTINUOUS"
        label_audit = {
            "label_name": label_col,
            "label_definition": "Workbook-native SpCas9 on-target indel-frequency/efficiency label; inspected without rescaling.",
            "experimental_measurement": "Experimentally measured HEK293T indel frequency for the Luo/Xiang component; Kim rows are integrated published data and remain source-separated.",
            "scale": "percent-like numeric scale in the workbook",
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "SD": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "IQR": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else None,
            "quantiles": {str(q): float(values.quantile(q)) for q in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1] if len(values)},
            "number_of_unique_values": int(values.nunique()),
            "missing_values": int(combined[label_col].isna().sum()),
            "classification": classification if separation["source_identity_recoverable"] else "INSUFFICIENT_EVIDENCE",
            "Quant_norm_efficiency_present": "Quant_norm_efficiency" in combined.columns,
            "derivation_assessment": "Quant_norm_efficiency is not present in the recovered workbook. The available average day-8/day-10 indel-frequency field is treated as a primary continuous measurement for audit purposes only.",
        }
    prelim = "PROMISING_FOR_17G" if (
        separation["source_identity_recoverable"]
        and separation["xiang_luo"].get("unique_30mer_n", 0)
        and overlap.get("all_recoverable", {}).get("new_unique_30mer_n", 0) >= 1000
        and label_audit.get("classification") in {"NORMALIZED_PRIMARY_CONTINUOUS", "PRIMARY_CONTINUOUS"}
    ) else "INSUFFICIENT_EVIDENCE"
    return {
        **base,
        "workbook_inventory": inventory,
        "sources": separation,
        "xiang_luo": separation["xiang_luo"],
        "kim": separation["kim"],
        "overlap_luo_kim": separation["overlap_luo_kim"],
        "other": separation["other"],
        "overlap_with_deepspcas9": overlap,
        "label_audit": label_audit,
        "preliminary_status": prelim,
    }


def build_phase17i(timestamp: str | None = None, retrieve: bool = True) -> dict:
    timestamp = timestamp or datetime.now().isoformat()
    before = protected_hashes()
    corsi = audit_corsi()
    retrieval = retrieve_crispron(timestamp=timestamp) if retrieve else {"retrieval_status": "NOT_ATTEMPTED", "file_sha256": None}
    crispron_path = Path(retrieval["local_path"]) if retrieval.get("local_path") else None
    crispron = audit_crispron(crispron_path, retrieval)
    after = protected_hashes()
    return {
        "phase": "17I",
        "timestamp": timestamp,
        "scope": {
            "no_phase18": True,
            "no_model_training": True,
            "no_moreno_access": True,
            "no_phase17g_rerun": True,
            "no_label_harmonization": True,
        },
        "corsi": corsi,
        "crispron": crispron,
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
    }


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
