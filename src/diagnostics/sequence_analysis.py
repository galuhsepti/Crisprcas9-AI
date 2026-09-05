"""
Sequence-distribution diagnostics for Phase 9A.

Compares the sequence composition of two datasets under the canonical 30-mer
geometry:

    5' context [0:4] | guide [4:24] | PAM [24:27] | 3' context [27:30]

Naming follows the repository geometry; the region definitions are NOT
changed here. Analyses are descriptive; observed differences are reported as
"sequence distribution differences" and are not biological claims.
"""

from typing import Dict, Iterable, List, Sequence
import numpy as np

from ..bioinformatics.gc_content import calculate_gc_content
from ..bioinformatics.kmer import (
    extract_kmer_matrix,
    extract_kmer_frequencies,
    generate_kmers
)
from ..bioinformatics.nucleotide_composition import (
    count_nucleotides,
    calculate_frequencies
)
from ..bioinformatics.positional_features import extract_position_frequency_matrix

VALID_NUCLEOTIDES = set('ACGT')


def gc_content_array(sequences: Sequence[str]) -> np.ndarray:
    """Per-sequence GC content as a numpy array."""
    return np.asarray([calculate_gc_content(s) for s in sequences], dtype=float)


def nucleotide_frequency_array(sequences: Sequence[str]) -> np.ndarray:
    """Per-sequence A/C/G/T frequency array of shape (n, 4)."""
    out = np.zeros((len(sequences), 4), dtype=float)
    order = ['A', 'C', 'G', 'T']
    for i, seq in enumerate(sequences):
        freqs = calculate_frequencies(seq)
        out[i] = [freqs.get(nuc, 0.0) for nuc in order]
    return out


def mean_positional_frequency_matrix(sequences: Sequence[str],
                                     length: int = 30) -> np.ndarray:
    """Mean position x nucleotide frequency matrix (4 x length)."""
    return extract_position_frequency_matrix(list(sequences), length)


def positional_entropy(pfm: np.ndarray) -> np.ndarray:
    """
    Shannon entropy per position, normalized by log2(4)=2 so it ranges in
    [0, 1]. 1.0 = uniformly distributed nucleotides.
    """
    pfm = np.asarray(pfm, dtype=float)
    if pfm.ndim != 2 or pfm.shape[0] != 4:
        raise ValueError("pfm must be a 4 x L matrix")
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(pfm > 0, pfm * np.log2(pfm), 0.0)
    entropy = -logp.sum(axis=0) / 2.0
    return entropy


def mean_kmer_frequencies(sequences: Sequence[str], k: int) -> np.ndarray:
    """Mean frequency vector over all 4^k k-mers (ordered by generate_kmers)."""
    matrix = extract_kmer_matrix(list(sequences), k)
    return matrix.mean(axis=0)


def sequence_length_summary(sequences: Sequence[str]) -> Dict[str, float]:
    """Length statistics (expected 30 for the canonical window)."""
    lengths = np.asarray([len(s) for s in sequences])
    return {
        'n': int(lengths.size),
        'min': float(lengths.min()),
        'max': float(lengths.max()),
        'mean': float(lengths.mean()),
        'uniform': bool(np.all(lengths == lengths[0]))
    }


def ambiguous_character_summary(sequences: Sequence[str]) -> Dict[str, float]:
    """Count of non-ACGT characters and sequences containing them."""
    n_seqs = 0
    n_chars = 0
    char_counts: Dict[str, int] = {}
    for seq in sequences:
        bad = [c.upper() for c in seq if c.upper() not in VALID_NUCLEOTIDES]
        if bad:
            n_seqs += 1
        n_chars += len(bad)
        for c in bad:
            char_counts[c] = char_counts.get(c, 0) + 1
    return {
        'sequences_with_ambiguous': int(n_seqs),
        'fraction_sequences': float(n_seqs / len(sequences)) if sequences else 0.0,
        'total_ambiguous_characters': int(n_chars),
        'characters': {k: int(v) for k, v in sorted(char_counts.items())}
    }


def duplicate_summary(sequences: Sequence[str]) -> Dict[str, float]:
    """Within-dataset duplicate statistics (exact sequence duplicates)."""
    from collections import Counter
    counts = Counter(sequences)
    unique = len(counts)
    duplicated = {s: c for s, c in counts.items() if c > 1}
    return {
        'total': int(len(sequences)),
        'unique': int(unique),
        'duplicated_sequences': int(len(duplicated)),
        'fraction_unique': float(unique / len(sequences)) if sequences else 0.0,
        'duplicate_rate': float(1.0 - unique / len(sequences)) if sequences else 0.0,
        'max_occurrences': int(max(counts.values())) if counts else 0
    }


def exact_sequence_overlap(a: Sequence[str], b: Sequence[str]) -> Dict[str, float]:
    """Exact-sequence overlap between two datasets (diagnosis only)."""
    set_a = set(a)
    set_b = set(b)
    shared = set_a & set_b
    denom = len(set_a | set_b)
    union = denom if denom > 0 else 1
    return {
        'shared_unique': int(len(shared)),
        'in_a_only': int(len(set_a - set_b)),
        'in_b_only': int(len(set_b - set_a)),
        'jaccard': float(len(shared) / union),
        'fraction_of_a_shared': float(len(shared) / len(set_a)) if set_a else 0.0,
        'fraction_of_b_shared': float(len(shared) / len(set_b)) if set_b else 0.0,
    }


def top_differing_kmers(
    seqs_a: Sequence[str],
    seqs_b: Sequence[str],
    k: int = 3,
    top_k: int = 10
) -> List[Dict[str, float]]:
    """
    K-mers ranked by largest mean-frequency difference between two datasets.
    Descriptive only (multiple comparisons; large n makes small differences
    significant, so rankings are reported without formal tests).
    """
    fa = mean_kmer_frequencies(seqs_a, k)
    fb = mean_kmer_frequencies(seqs_b, k)
    kmers = generate_kmers(k)
    diffs = fa - fb
    order = np.argsort(-np.abs(diffs))
    out = []
    for idx in order[:top_k]:
        out.append({
            'kmer': kmers[idx],
            'mean_freq_a': float(fa[idx]),
            'mean_freq_b': float(fb[idx]),
            'difference': float(diffs[idx])
        })
    return out


def kmer_profile_correlation(
    seqs_a: Sequence[str],
    seqs_b: Sequence[str],
    k: int
) -> Dict[str, float]:
    """
    Pearson correlation between the two datasets' mean k-mer frequency
    vectors (across all 4^k k-mers), plus mean absolute difference.
    """
    from scipy.stats import pearsonr
    fa = mean_kmer_frequencies(seqs_a, k)
    fb = mean_kmer_frequencies(seqs_b, k)
    r, p = pearsonr(fa, fb)
    return {
        'pearson_across_kmers': float(r),
        'p_value': float(p),
        'mean_abs_frequency_difference': float(np.mean(np.abs(fa - fb)))
    }