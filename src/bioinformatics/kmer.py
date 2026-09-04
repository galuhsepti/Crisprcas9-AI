"""
k-mer feature extraction module for CRISPR-Cas9 sgRNA analysis.

This module provides functions to extract k-mer features from DNA sequences,
which can be used as input features for machine learning models.
"""

from typing import Dict, List, Optional, Tuple
from itertools import product
import numpy as np
from collections import Counter


# Valid nucleotides
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def generate_kmers(k: int) -> List[str]:
    """
    Generate all possible k-mers for DNA.
    
    Args:
        k: k-mer size
        
    Returns:
        List of all possible k-mers
    """
    return [''.join(combo) for combo in product(NUCLEOTIDES, repeat=k)]


def extract_kmer_counts(
    sequence: str,
    k: int = 3
) -> Dict[str, int]:
    """
    Extract k-mer counts from a sequence.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        
    Returns:
        Dictionary with k-mer counts
    """
    if len(sequence) < k:
        raise ValueError(f"Sequence length {len(sequence)} < k={k}")
    
    sequence = sequence.upper()
    kmers = generate_kmers(k)
    
    # Count k-mers
    counts = {kmer: 0 for kmer in kmers}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in counts:
            counts[kmer] += 1
    
    return counts


def extract_kmer_frequencies(
    sequence: str,
    k: int = 3
) -> Dict[str, float]:
    """
    Extract k-mer frequencies from a sequence.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        
    Returns:
        Dictionary with k-mer frequencies (0.0 to 1.0)
    """
    counts = extract_kmer_counts(sequence, k)
    total = sum(counts.values())
    
    if total == 0:
        return {kmer: 0.0 for kmer in counts}
    
    return {kmer: count / total for kmer, count in counts.items()}


def extract_kmer_features(
    sequence: str,
    k_values: List[int] = [2, 3, 4]
) -> Dict[str, float]:
    """
    Extract k-mer features for multiple k values.
    
    Args:
        sequence: DNA sequence string
        k_values: List of k values to extract
        
    Returns:
        Dictionary with k-mer features
    """
    features = {}
    
    for k in k_values:
        freq = extract_kmer_frequencies(sequence, k)
        for kmer, f in freq.items():
            features[f'k{k}_{kmer}'] = f
    
    return features


def extract_kmer_matrix(
    sequences: List[str],
    k: int = 3
) -> np.ndarray:
    """
    Extract k-mer frequency matrix for multiple sequences.
    
    Args:
        sequences: List of DNA sequence strings
        k: k-mer size
        
    Returns:
        NumPy array of shape (n_sequences, 4^k)
    """
    kmers = generate_kmers(k)
    matrix = []
    
    for seq in sequences:
        freq = extract_kmer_frequencies(seq, k)
        matrix.append([freq[kmer] for kmer in kmers])
    
    return np.array(matrix)


def extract_position_specific_kmers(
    sequence: str,
    k: int = 3,
    positions: Optional[List[int]] = None
) -> Dict[str, int]:
    """
    Extract k-mers at specific positions.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        positions: List of starting positions (if None, extract all)
        
    Returns:
        Dictionary with position-specific k-mer counts
    """
    sequence = sequence.upper()
    
    if positions is None:
        positions = range(len(sequence) - k + 1)
    
    kmers = {}
    for pos in positions:
        if 0 <= pos <= len(sequence) - k:
            kmer = sequence[pos:pos+k]
            kmers[f'pos{pos}_{kmer}'] = 1
    
    return kmers


def calculate_kmer_entropy(
    sequence: str,
    k: int = 3
) -> float:
    """
    Calculate Shannon entropy of k-mer distribution.
    
    Higher entropy indicates more uniform k-mer distribution.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        
    Returns:
        Entropy value
    """
    freq = extract_kmer_frequencies(sequence, k)
    
    entropy = 0.0
    for f in freq.values():
        if f > 0:
            entropy -= f * np.log2(f)
    
    return entropy


def calculate_kmer_complexity(
    sequence: str,
    k: int = 3
) -> float:
    """
    Calculate k-mer complexity (number of unique k-mers / total possible).
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        
    Returns:
        Complexity value (0.0 to 1.0)
    """
    counts = extract_kmer_counts(sequence, k)
    unique_kmers = sum(1 for count in counts.values() if count > 0)
    total_possible = len(counts)
    
    return unique_kmers / total_possible


def extract_guide_kmer_features(
    sequence: str,
    context_length: int = 30,
    guide_length: int = 20,
    k_values: List[int] = [2, 3]
) -> Dict[str, float]:
    """
    Extract k-mer features specifically from guide sequence.
    
    For CRISPR 30-mer: guide is at positions 4-23.
    
    Args:
        sequence: 30-mer DNA sequence
        context_length: Full sequence length
        guide_length: Guide sequence length
        k_values: List of k values to extract
        
    Returns:
        Dictionary with guide k-mer features
    """
    if len(sequence) != context_length:
        raise ValueError(f"Expected length {context_length}, got {len(sequence)}")
    
    # Extract guide sequence
    start = (context_length - guide_length) // 2
    guide = sequence[start:start + guide_length]
    
    features = {}
    
    # k-mer frequencies
    for k in k_values:
        freq = extract_kmer_frequencies(guide, k)
        for kmer, f in freq.items():
            features[f'guide_k{k}_{kmer}'] = f
    
    # k-mer entropy
    for k in k_values:
        features[f'guide_k{k}_entropy'] = calculate_kmer_entropy(guide, k)
    
    # k-mer complexity
    for k in k_values:
        features[f'guide_k{k}_complexity'] = calculate_kmer_complexity(guide, k)
    
    return features
