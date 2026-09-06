"""
Dataset inventory, loaders and provenance registry for Phase 10.

AUDIT RESULT (run before design was frozen):
- Only DeepHF is a *new* training candidate that passes all inclusion
  criteria. Every other locally-available dataset is either redundant with
  the canonical DeepSpCas9 training domain, a locked benchmark *test* set with
  incomparable label semantics, or carries verifiably synthetic flanking
  context. None of them overlaps Moreno-Mateos, but they are excluded for the
  documented scientific reasons below.

INCLUSION CRITERIA (all must hold):
A. experimental activity labels
B. sequence sufficient to construct the canonical 30-mer input
C. label semantics understood
D. geometry mappable to canonical 30-mer, or a documented, justifiable
   transformation (see config.GEOMETRY_TRANSFORMATION)
E. no known contamination with Moreno-Mateos test sequences
F. provenance documented
G. label scale interpretable
H. experimental context recorded
"""

from pathlib import Path
from typing import Dict, List, Optional

import hashlib
import pandas as pd

from .config import (
    FLANK3_PADDING,
    FLANK5_PADDING,
    DEEP_SPCAS9_PATH,
    MORENO_MATEOS_PATH,
    DEEP_HF_PATH,
)
from ..data.validation import validate_sequence

# Column names produced by every loader for internal consumption.
NORMALIZED_SEQ_COL = "normalized_sequence"
ACTIVITY_COL = "activity_label"
SOURCE_COL = "source_dataset"


# ---------------------------------------------------------------------------
# Dataset inventory (audit of everything locally available)
# ---------------------------------------------------------------------------
DATASET_INVENTORY = [
    {
        "dataset": "DeepSpCas9",
        "source": "Kim et al. (2019), Science Advances 5, eaax9249",
        "local_path": DEEP_SPCAS9_PATH,
        "n_raw": 12832,
        "n_eligible": 10117,
        "label": "modFreq (modification frequency, 0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer (guide [4:24], PAM [24:27])",
        "pam": "NGG (100%)",
        "strand": "target strand 5'->3'",
        "assay": "SpCas9 pooled lentiviral screen, NGS modified-read fraction",
        "species": "human",
        "label_scale": "[0, 1] continuous",
        "comparable": True,
        "replicates": "none in file",
        "duplicates": 0,
        "used_in_project": True,
        "status": "PRIMARY TRAINING DOMAIN",
        "geometry_note": "genuine 30-mer with flanking context",
    },
    {
        "dataset": "Moreno-Mateos",
        "source": "Moreno-Mateos et al. (2015), Nature Methods 12, 982",
        "local_path": MORENO_MATEOS_PATH,
        "n_raw": 1020,
        "n_eligible": 810,
        "label": "germline editing efficiency (0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG (100%)",
        "strand": "target strand 5'->3'",
        "assay": "zebrafish embryo mutagenesis, tRNA/CRISPRscan activity",
        "species": "zebrafish (some human/Xenopus)",
        "label_scale": "[0, 1] continuous",
        "comparable": True,
        "replicates": "none in file",
        "duplicates": 0,
        "used_in_project": True,
        "status": "LOCKED EXTERNAL TEST (never train/tune/select)",
        "geometry_note": "genuine 30-mer with flanking context",
    },
    {
        "dataset": "DeepHF",
        "source": ("Wang et al. (2019), Nat. Biotechnol. 37, 224; "
                   "Xiang et al. (2021), Nat. Commun. 12, 3238"),
        "local_path": DEEP_HF_PATH,
        "n_raw": 59852,
        "n_eligible": 48295,
        "label": "Wt_Efficiency (0-1)",
        "seq_length": 20 + 3,  # guide + PAM only in the official schema
        "seq_type": "DNA 21-mer (guide + PAM); no genomic flank in source",
        "pam": "NGG (AGG/GGG/TGG/CGG), 1 row malformed ('GG')",
        "strand": "target strand 5'->3'",
        "assay": ("large pooled SpCas9 screens (HEK293T/HCT116/T cells), "
                  "deep-sequenced edited-read fraction"),
        "species": "human",
        "label_scale": "[0, 1] continuous",
        "comparable": True,
        "replicates": "none in file",
        "duplicates": 0,
        "used_in_project": False,
        "status": "INCLUDED (only new training dataset)",
        "geometry_note": "21-mer source; canonical 30-mer via documented "
                         "constant-flank transformation; 3,670 mapped rows "
                         "with NaN Wt_Efficiency labels excluded at load "
                         "(targetless rows carry no supervision)",
    },
    {
        "dataset": "DeepSpCas9 (Library).csv",
        "source": "benchmark Training/ folder - duplicate of canonical",
        "local_path": "data/external/Benchmarking-CRISPR-on-tools/Training/"
                      "DeepSpCas9 (Library).csv",
        "n_raw": 12832,
        "n_eligible": 10117,
        "label": "modFreq (0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "target strand 5'->3'",
        "assay": "as DeepSpCas9",
        "species": "human",
        "label_scale": "[0, 1]",
        "comparable": True,
        "replicates": "n/a",
        "duplicates": "identical copy",
        "used_in_project": True,
        "status": "EXCLUDED - byte-identical duplicate of the canonical "
                  "DeepSpCas9 training set, adds no diversity",
    },
    {
        "dataset": "Doench (A375)",
        "source": "Doench et al. (2016), Nat. Biotechnol. 34, 184",
        "local_path": "Supplementary Data/Supplementary Data2.xlsx",
        "n_raw": 2333,
        "n_eligible": 1934,
        "label": "normalized log2-enrichment activity (0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "target strand",
        "assay": "A375 pooled lentiviral screen, log2 selection ratio",
        "species": "human",
        "label_scale": "[0, 1] continuous",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - 81% of its valid 30-mers are EXACT duplicates of "
                  "canonical DeepSpCas9 training sequences (1568/1934); "
                  "repurposing it adds almost no new domain",
        "geometry_note": "5'/3' context saturation pattern matches canonical "
                         "data; not independently verifiable as genomic",
    },
    {
        "dataset": "Chen (HEK293T)",
        "source": "Chen et al. (2019), Nucleic Acids Res. 47, 7989",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 3066,
        "n_eligible": 3060,
        "label": "edited-read fraction (0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "Amplicon NGS after HDR/NHEJ in HEK293T",
        "species": "human",
        "label_scale": "[0, 1]",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - verifiable synthetic flanking context: ALL 3066 "
                  "rows share a single 5' 4-mer and a single 3' 3-mer "
                  "(unique5=1, unique3=1), i.e. seq30 flanks are not genomic",
        "geometry_note": "PROVEN non-genuine 30-mer geometry",
    },
    {
        "dataset": "Hart (HCT116)",
        "source": "Hart et al. (2015), Cell 163, 1515",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 4239,
        "n_eligible": 4239,
        "label": "log2 screen score (approx -3 .. +9)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "HCT116 fitness screen",
        "species": "human",
        "label_scale": "log2 ratio - UNBOUNDED log scale",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - label measures positive-selection fitness "
                  "(log2 ratio), a different biological quantity than a "
                  "0-1 editing-efficiency fraction; benchmark test set",
    },
    {
        "dataset": "Labuhn (HEL)",
        "source": "Labuhn et al. (2018), Nucleic Acids Res. 46, 1375",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 424,
        "n_eligible": 362,
        "label": "fluorescent-reporter 0-1 fraction",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "haematopoietic cell fluorescent reporter",
        "species": "human",
        "label_scale": "[0, 1]",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - small (362-424), single reporter cell line, "
                  "benchmark-role test set; 1 valid 30-mer duplicates a "
                  "DeepSpCas9 training sequence; label is not the same "
                  "modality as the primary corpus",
    },
    {
        "dataset": "Koike-Yusa (mESC)",
        "source": "Koike-Yusa et al. (2014), Nat. Biotechnol. 32, 267",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 1064,
        "n_eligible": 906,
        "label": "log2 screen score (approx -1.5 .. +6.5)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "mouse embryonic stem-cell fitness screen",
        "species": "mouse",
        "label_scale": "log2 ratio - UNBOUNDED log scale",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - log2 fitness-ratio label (different quantity), "
                  "mouse species, benchmark test set",
    },
    {
        "dataset": "Gagnon (zebrafish)",
        "source": "Gagnon et al. (2014), PLoS One 9, e98186",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 111,
        "n_eligible": 93,
        "label": "germline mutation rate (%)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "zebrafish embryo microinjection",
        "species": "zebrafish",
        "label_scale": "0-95 % - PERCENT scale, not fraction",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - percent-scale label, different species/assay, "
                  "tiny (93 valid), benchmark test set",
    },
    {
        "dataset": "Varshney (zebrafish)",
        "source": "Varshney et al. (2015), Genome Res. 25, 1030",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 102,
        "n_eligible": 88,
        "label": "indel rate (%)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "zebrafish embryo microinjection, indel radio",
        "species": "zebrafish",
        "label_scale": "0-100 % - PERCENT scale",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - percent scale, different species/assay, tiny, "
                  "benchmark test set",
    },
    {
        "dataset": "Teboul (mouse i.v.)",
        "source": "Teboul et al. in Haeussler et al. (2016), Genome Biol. 17, 148",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 30,
        "n_eligible": 25,
        "label": "embryos with induced indel (0-1)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer",
        "pam": "NGG",
        "strand": "-",
        "assay": "mouse in-vivo oocyte microinjection",
        "species": "mouse in vivo",
        "label_scale": "[0, 1]",
        "comparable": False,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - n=25 valid, in-vivo microinjection context not "
                  "comparable to pooled-screen efficiency, benchmark test set",
    },
    {
        "dataset": "Xiang (DeepHF) in Data2",
        "source": "benchmark Supplementary Data2 'test set' facsimile of DeepHF",
        "local_path": "Supplementary Data2.xlsx",
        "n_raw": 10592,
        "n_eligible": 8573,
        "label": "DeepHF activity (0-1 unit)",
        "seq_length": 30,
        "seq_type": "DNA 30-mer with flanks",
        "pam": "NGG",
        "strand": "-",
        "assay": "subset of the DeepHF screens (as in its benchmark role)",
        "species": "human",
        "label_scale": "[0, 1]",
        "comparable": True,
        "replicates": "n/a",
        "duplicates": "n/a",
        "used_in_project": False,
        "status": "EXCLUDED - the benchmark paper collected it as an "
                  "independent TEST set, and because the underlying DeepHF "
                  "data contains NO genomic flanks, its seq30 flanks must be "
                  "synthetic (unverifiable). The primary DeepHF training file "
                  "is used with a documented transformation instead.",
        "geometry_note": "flanks cannot be genuine (source has no flanks)",
    },
]


def dataset_inventory_table() -> pd.DataFrame:
    """Tabular rendering of DATASET_INVENTORY (for reporting)."""
    rows = []
    for e in DATASET_INVENTORY:
        rows.append({
            "Dataset": e["dataset"],
            "Included": "YES" if e["status"].startswith("INCLUDED") else "NO",
            "Reason": e["status"],
            "N sequences": e["n_eligible"],
            "Label range": e["label_scale"],
            "Assay/context": e["assay"],
            "Known limitations": e.get("geometry_note", "-"),
        })
    return pd.DataFrame(rows)


def allowed_additional_datasets() -> List[str]:
    """Datasets that passed all inclusion criteria."""
    return [e["dataset"] for e in DATASET_INVENTORY
            if e["status"].startswith("INCLUDED")]


# ---------------------------------------------------------------------------
# Geometry / canonical validation
# ---------------------------------------------------------------------------
def construct_deephf_30mer(guide: str, pam: str) -> Optional[str]:
    """
    Map a DeepHF guide+PAM pair onto the canonical 30-mer geometry using the
    pre-registered constant-flank transformation.

    Returns None when the source geometry cannot be mapped (wrong length or a
    malformed PAM such as the single 2 bp 'GG' row).
    """
    guide = str(guide).strip().upper()
    pam = str(pam).strip().upper()
    if len(guide) != 20 or len(pam) != 3:
        return None
    return FLANK5_PADDING + guide + pam + FLANK3_PADDING


def canonical_validate(df: pd.DataFrame, sequence_col: str,
                       check_pam: bool = False) -> pd.DataFrame:
    """
    Apply the SAME validation used by the canonical train scripts
    (validate_sequence with check_pam=False): length 30, ACGT alphabet, and
    the guide homopolymer filter.
    """
    mask = []
    for seq in df[sequence_col]:
        is_valid, _ = validate_sequence(seq, check_pam=check_pam)
        mask.append(is_valid)
    return df[mask].copy().reset_index(drop=True)


def load_deepspcas9(path: str = DEEP_SPCAS9_PATH) -> pd.DataFrame:
    """Load and canonically validate DeepSpCas9 (train/validation source)."""
    df = pd.read_csv(path)
    df = canonical_validate(df, "sequence_30mer")
    df = df.rename(columns={"sequence_30mer": NORMALIZED_SEQ_COL,
                            "activity": ACTIVITY_COL})
    df[SOURCE_COL] = "deepspcas9"
    return df.reset_index(drop=True)


def load_moreno_mateos(path: str = MORENO_MATEOS_PATH) -> pd.DataFrame:
    """Load and canonically validate the LOCKED external test (never trained)."""
    df = pd.read_csv(path)
    df = canonical_validate(df, "sequence_30mer")
    df = df.rename(columns={"sequence_30mer": NORMALIZED_SEQ_COL,
                            "activity": ACTIVITY_COL})
    df[SOURCE_COL] = "moreno_mateos"
    return df.reset_index(drop=True)


def load_deephf(path: str = DEEP_HF_PATH) -> pd.DataFrame:
    """
    Load DeepHF training data and map it onto the canonical 30-mer geometry.

    Columns produced: normalized_sequence (constructed 30-mer), activity_label
    (Wt_Efficiency), guide_sequence, pam, and provenance metadata.

    Data-quality exclusions (documented, NOT a design decision):
    - rows whose PAM cannot be mapped to the canonical geometry (e.g. the
      single malformed 'GG' PAM row) are dropped;
    - rows whose Wt_Efficiency label is NaN (3,670 of the mapped rows) are
      dropped: a targetless row carries no supervision signal.
    """
    df = pd.read_excel(path)
    df["constructed_30mer"] = [
        construct_deephf_30mer(g, p) for g, p
        in zip(df["gRNA_Seq"], df["PAM"])
    ]
    df = df[df["constructed_30mer"].notna()].copy()
    df = canonical_validate(df, "constructed_30mer")
    n_pre = len(df)
    df = df[df["Wt_Efficiency"].notna()].copy()
    n_labeled = len(df)
    n_unlabeled = n_pre - n_labeled
    df = df.rename(columns={"constructed_30mer": NORMALIZED_SEQ_COL,
                            "Wt_Efficiency": ACTIVITY_COL,
                            "gRNA_Seq": "guide_sequence", "PAM": "pam"})
    df[SOURCE_COL] = "deephf"
    df["original_sequence"] = df["guide_sequence"] + df["pam"]
    df["experimental_context"] = (
        "DeepHF pooled SpCas9 screens (HEK293T/HCT116/T cells), edited-read "
        "fraction")
    df.attrs["n_unlabeled_dropped"] = n_unlabeled
    df.attrs["n_mapped_valid"] = n_pre
    return df.reset_index(drop=True)


def sha256_file(path: str) -> str:
    """SHA-256 of a file (chunked, for the reproducibility audit)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()