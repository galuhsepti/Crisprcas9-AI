"""
Sequence-region ablation module (Phase 7).

Defines the sequence regions of the 30-mer CRISPR window and provides
extraction of region-specific one-hot encodings for the CNN ablation study.

Region layout (fixed geometry, see config.yaml):
    positions       0    4              24    27    30
                    5' context |  guide   | PAM | 3' context

Regions analysed:
    - full        : [0:30]            (canonical input, 30 bp)
    - guide       : [4:24]            (20 bp guide sequence)
    - guide_pam   : [4:27]            (guide + upstream NGG PAM, 23 bp)
    - context_pam : [0:4] + [24:30]   (flanking context + PAM, 10 bp;
                                       everything except the guide)

These regions are used to train CNN variants with an IDENTICAL training
recipe (same kernels, filters, split, seed) to isolate which parts of the
sequence carry the predictive signal. The canonical model and its artifacts
are never modified.
"""

from typing import Dict, List
import numpy as np

from ..bioinformatics.positional_features import extract_one_hot_encoding

# Canonical 30-mer geometry
CONTEXT_LENGTH = 30
GUIDE_START = 4
GUIDE_END = 24
PAM_END = 27

REGION_POSITIONS: Dict[str, List[int]] = {
    'full': list(range(0, 30)),
    'guide': list(range(GUIDE_START, GUIDE_END)),
    'guide_pam': list(range(GUIDE_START, PAM_END)),
    'context_pam': list(range(0, GUIDE_START)) + list(range(GUIDE_END, CONTEXT_LENGTH)),
}

REGION_SEQUENCE_LENGTHS: Dict[str, int] = {
    name: len(pos) for name, pos in REGION_POSITIONS.items()
}

SUPPORTED_REGIONS = list(REGION_POSITIONS.keys())


def extract_region_one_hot(
    sequences,
    region: str,
    context_length: int = CONTEXT_LENGTH
) -> np.ndarray:
    """
    Extract one-hot encoding restricted to a sequence region.

    Args:
        sequences: Iterable of DNA sequence strings of length ``context_length``
        region: One of 'full', 'guide', 'guide_pam', 'context_pam'
        context_length: Full sequence length (30)

    Returns:
        NumPy array of shape (n_sequences, region_length, 4)

    Raises:
        ValueError: If the region is unknown or a sequence has the wrong length
    """
    if region not in REGION_POSITIONS:
        raise ValueError(
            f"Unknown region '{region}'. Supported: {SUPPORTED_REGIONS}"
        )

    positions = REGION_POSITIONS[region]
    region_length = len(positions)

    rows = []
    for seq in sequences:
        if len(seq) != context_length:
            raise ValueError(
                f"Expected sequence of length {context_length}, got {len(seq)}"
            )
        region_seq = ''.join(seq[i] for i in positions)
        rows.append(extract_one_hot_encoding(region_seq, region_length))

    return np.asarray(rows, dtype=np.float32)


def region_sequence_length(region: str) -> int:
    """Return the encoded length (in bp) of a region."""
    if region not in REGION_POSITIONS:
        raise ValueError(
            f"Unknown region '{region}'. Supported: {SUPPORTED_REGIONS}"
        )
    return REGION_SEQUENCE_LENGTHS[region]