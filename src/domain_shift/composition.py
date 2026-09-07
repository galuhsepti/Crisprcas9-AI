"""
Phase 14 sequence-composition shift analysis.

Aggregate descriptors of sequence composition for the internal (train/val) and
external (locked held-out test) corpora. No per-k-mer hypothesis tests are
performed: k-mer shifts are summarised by Jensen-Shannon divergence against an
internal-derived null band (bootstrap halves of the internal TRAIN corpus),
avoiding dozens of individual tests without correction.

Cohen's d formula (frozen): d = (mean_a - mean_b) / s_pooled with
    s_pooled = sqrt(((na-1) sa^2 + (nb-1) sb^2) / (na + nb - 2))
"""

from typing import Dict, List, Sequence

import numpy as np

from .statistics import reproducible_bootstrap_seed


def gc_array(sequences: Sequence[str]) -> np.ndarray:
    """Per-sequence GC fraction over the full 30-mer."""
    out = np.empty(len(sequences), dtype=float)
    for i, s in enumerate(sequences):
        s = s.upper()
        out[i] = (s.count("G") + s.count("C")) / len(s)
    return out


def gc_array_guide(
    sequences: Sequence[str], guide_start: int = 4, guide_length: int = 20
) -> np.ndarray:
    """Per-sequence GC fraction over the guide [4:24] window."""
    out = np.empty(len(sequences), dtype=float)
    for i, s in enumerate(sequences):
        g = s[guide_start:guide_start + guide_length].upper()
        out[i] = (g.count("G") + g.count("C")) / len(g)
    return out


def nucleotide_frequencies(sequences: Sequence[str]) -> Dict[str, float]:
    """Mean nucleotide frequencies over all positions of the corpus."""
    nucs = ["A", "C", "G", "T"]
    seqs = [s.upper() for s in sequences]
    total = sum(len(s) for s in seqs)
    return {n: sum(s.count(n) for s in seqs) / total for n in nucs}


def kmer_frequency_vector(
    sequences: Sequence[str], k: int = 2, vocabulary: Sequence[str] = ("A", "C", "G", "T")
) -> np.ndarray:
    """
    Corpus-level mean k-mer frequency vector (normalized to sum ~1).

    Order is fixed by the deterministic generation over the vocabulary, so the
    vector is comparable across corpora of the same k.
    """
    kmers = ["".join(p) for p in __import__("itertools").product(vocabulary, repeat=k)]
    seqs = [s.upper() for s in sequences]
    counts = np.zeros(len(kmers), dtype=float)
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    n = len(seqs)
    if n == 0:
        raise ValueError("Empty sequence list")
    for s in seqs:
        window = len(s) - k + 1
        if window <= 0:
            raise ValueError(f"Sequence length {len(s)} < k={k}")
        for i in range(window):
            kmer = s[i:i + k]
            if kmer in kmer_to_idx:
                counts[kmer_to_idx[kmer]] += 1.0
    counts /= counts.sum()
    return counts


def _js(p: np.ndarray, q: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> Dict[str, float]:
    """Jensen-Shannon divergence between two frequency vectors."""
    return {"js_divergence": _js(p, q)}


def internal_null_band(
    train_sequences: Sequence[str],
    k: int = 2,
    n_boot: int = 500,
    seed: int = 20260907,
    vocabulary: Sequence[str] = ("A", "C", "G", "T"),
) -> Dict[str, float]:
    """
    Internal null band for k-mer JS divergence.

    Repeatedly compares two disjoint halves of the INTERNAL TRAIN corpus only.
    The 2.5/50/97.5 percentiles of the within-domain JS measure the sampling
    noise of the divergence estimate itself. Reported descriptive band; not a
    significance test.
    """
    seqs = list(train_sequences)
    n = len(seqs)
    if n < 40:
        raise ValueError("Train corpus too small for null band")
    rng = reproducible_bootstrap_seed(seed)
    js_vals = np.empty(n_boot)
    for i in range(n_boot):
        perm = rng.permutation(n)
        half = n // 2
        a = [seqs[j] for j in perm[:half]]
        b = [seqs[j] for j in perm[half:]]
        va = kmer_frequency_vector(a, k=k, vocabulary=vocabulary)
        vb = kmer_frequency_vector(b, k=k, vocabulary=vocabulary)
        js_vals[i] = _js(va, vb)
    q2, q50, q97 = np.percentile(js_vals, [2.5, 50.0, 97.5])
    return {
        "k": int(k),
        "n_boot": int(n_boot),
        "observed_within_train_median": float(q50),
        "null_band_2.5pct": float(q2),
        "null_band_97.5pct": float(q97),
        "description": "internal-train half-split null band (descriptive)",
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """
    Standardized mean difference d = (mean_a - mean_b)/s_pooled.

    Frozen formula: pooled SD uses (na-1) and (nb-1) weights (sample SDs).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return {"cohens_d": None, "n_a": int(na), "n_b": int(nb)}
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    s_pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if s_pooled == 0:
        return {"cohens_d": 0.0, "n_a": int(na), "n_b": int(nb)}
    return {
        "cohens_d": float((ma - mb) / s_pooled),
        "mean_a": float(ma),
        "mean_b": float(mb),
        "pooled_sd": float(s_pooled),
        "n_a": int(na),
        "n_b": int(nb),
    }


def position_frequencies(
    sequences: Sequence[str], sequence_length: int = 30
) -> Dict[str, List[float]]:
    """Per-position nucleotide frequencies (4 x L matrix)."""
    nucs = ["A", "C", "G", "T"]
    mat = {n: np.zeros(sequence_length, dtype=float) for n in nucs}
    for s in sequences:
        s = s.upper()
        if len(s) != sequence_length:
            continue
        for i, ch in enumerate(s):
            if ch in mat:
                mat[ch][i] += 1.0
    n = len(sequences)
    for nuc in nucs:
        mat[nuc] /= n if n > 0 else 1.0
    return {n: mat[n].tolist() for n in nucs}


def mad_between(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> float:
    """
    Mean absolute deviation between two position-frequency dictionaries over
    the union of their nucleotide keys.
    """
    keys = list(set(a.keys()) | set(b.keys()))
    acc = 0.0
    n = 0
    for k in keys:
        va = np.asarray(a.get(k, []), dtype=float)
        vb = np.asarray(b.get(k, []), dtype=float)
        if va.shape != vb.shape or va.size == 0:
            continue
        acc += float(np.mean(np.abs(va - vb)))
        n += 1
    return acc / n if n else 0.0