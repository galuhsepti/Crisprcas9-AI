"""
Nucleotide composition analysis module.

This module provides functions to analyze nucleotide frequencies
and composition patterns in DNA sequences.
"""

from typing import Dict, List
import numpy as np


# Valid nucleotides
NUCLEOTIDES = ['A', 'C', 'G', 'T']
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def count_nucleotides(sequence: str) -> Dict[str, int]:
    """
    Count occurrences of each nucleotide.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with nucleotide counts
    """
    sequence = sequence.upper()
    return {nuc: sequence.count(nuc) for nuc in NUCLEOTIDES}


def calculate_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate frequency of each nucleotide.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with nucleotide frequencies (0.0 to 1.0)
    """
    if not sequence:
        raise ValueError("Empty sequence")
    
    counts = count_nucleotides(sequence)
    length = len(sequence)
    
    return {nuc: count / length for nuc, count in counts.items()}


def calculate_frequencies_array(sequences: List[str]) -> np.ndarray:
    """
    Calculate nucleotide frequencies for multiple sequences.
    
    Args:
        sequences: List of DNA sequence strings
        
    Returns:
        NumPy array of shape (n_sequences, 4) with A, C, G, T frequencies
    """
    freqs = []
    for seq in sequences:
        f = calculate_frequencies(seq)
        freqs.append([f['A'], f['C'], f['G'], f['T']])
    return np.array(freqs)


def calculate_position_frequencies(
    sequences: List[str],
    sequence_length: int = 20
) -> Dict[str, np.ndarray]:
    """
    Calculate nucleotide frequencies at each position.
    
    Args:
        sequences: List of DNA sequence strings
        sequence_length: Expected sequence length
        
    Returns:
        Dictionary with position-specific frequency arrays
    """
    n_sequences = len(sequences)
    
    # Initialize frequency matrix
    freq_matrix = {nuc: np.zeros(sequence_length) for nuc in NUCLEOTIDES}
    
    for seq in sequences:
        seq = seq.upper()
        if len(seq) != sequence_length:
            continue
        
        for i, nuc in enumerate(seq):
            if nuc in NUCLEOTIDES:
                freq_matrix[nuc][i] += 1
    
    # Normalize by number of sequences
    for nuc in NUCLEOTIDES:
        freq_matrix[nuc] /= n_sequences
    
    return freq_matrix


def calculate_dinucleotide_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate dinucleotide frequencies.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with dinucleotide frequencies
    """
    if len(sequence) < 2:
        raise ValueError("Sequence too short for dinucleotide analysis")
    
    sequence = sequence.upper()
    
    # Generate all possible dinucleotides
    dinucleotides = [n1 + n2 for n1 in NUCLEOTIDES for n2 in NUCLEOTIDES]
    
    # Count dinucleotides
    counts = {}
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        if dinuc in counts:
            counts[dinuc] += 1
        else:
            counts[dinuc] = 1
    
    # Calculate frequencies
    total = len(sequence) - 1
    return {dinuc: counts.get(dinuc, 0) / total for dinuc in dinucleotides}


def calculate_kmer_frequencies(
    sequence: str,
    k: int = 3
) -> Dict[str, float]:
    """
    Calculate k-mer frequencies.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
        
    Returns:
        Dictionary with k-mer frequencies
    """
    if len(sequence) < k:
        raise ValueError(f"Sequence length {len(sequence)} < k={k}")
    
    sequence = sequence.upper()
    
    # Generate all possible k-mers
    from itertools import product
    kmers = [''.join(combo) for combo in product(NUCLEOTIDES, repeat=k)]
    
    # Count k-mers
    counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in counts:
            counts[kmer] += 1
        else:
            counts[kmer] = 1
    
    # Calculate frequencies
    total = len(sequence) - k + 1
    return {kmer: counts.get(kmer, 0) / total for kmer in kmers}


def calculate_composition_features(sequence: str) -> Dict[str, float]:
    """
    Calculate comprehensive nucleotide composition features.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with various composition features
    """
    freq = calculate_frequencies(sequence)
    
    features = {
        'freq_A': freq['A'],
        'freq_C': freq['C'],
        'freq_G': freq['G'],
        'freq_T': freq['T'],
        'gc_content': freq['G'] + freq['C'],
        'at_content': freq['A'] + freq['T'],
        'purine_content': freq['A'] + freq['G'],
        'pyrimidine_content': freq['C'] + freq['T'],
    }
    
    # Add dinucleotide frequencies
    dinuc_freq = calculate_dinucleotide_frequencies(sequence)
    for dinuc, freq in dinuc_freq.items():
        features[f'dinuc_{dinuc}'] = freq
    
    return features


def get_reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Reverse complement sequence
    """
    return ''.join(COMPLEMENT.get(nuc, nuc) for nuc in reversed(sequence.upper()))


def calculate_heterogeneity(sequence: str) -> float:
    """
    Calculate sequence heterogeneity (Shannon entropy).
    
    Higher entropy means more uniform nucleotide distribution.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Entropy value (0.0 to 2.0 for DNA)
    """
    if not sequence:
        raise ValueError("Empty sequence")
    
    freq = calculate_frequencies(sequence)
    entropy = 0.0
    
    for f in freq.values():
        if f > 0:
            entropy -= f * np.log2(f)
    
    return entropy
