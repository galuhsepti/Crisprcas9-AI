"""
Positional nucleotide features module.

This module provides functions to extract position-specific
nucleotide features from DNA sequences.
"""

from typing import Dict, List, Optional
import numpy as np


# Valid nucleotides
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def extract_position_features(
    sequence: str,
    sequence_length: Optional[int] = None
) -> Dict[str, int]:
    """
    Extract binary position features for each nucleotide.
    
    Creates one-hot encoding-like features for each position.
    
    Args:
        sequence: DNA sequence string
        sequence_length: Expected sequence length (if None, use actual length)
        
    Returns:
        Dictionary with position-specific binary features
    """
    sequence = sequence.upper()
    
    if sequence_length is None:
        sequence_length = len(sequence)
    
    features = {}
    
    for i in range(min(len(sequence), sequence_length)):
        nuc = sequence[i]
        if nuc in NUCLEOTIDES:
            features[f'pos_{i}_{nuc}'] = 1
    
    return features


def extract_one_hot_encoding(
    sequence: str,
    sequence_length: Optional[int] = None
) -> np.ndarray:
    """
    Convert sequence to one-hot encoding matrix.
    
    Args:
        sequence: DNA sequence string
        sequence_length: Expected sequence length
        
    Returns:
        NumPy array of shape (sequence_length, 4)
    """
    sequence = sequence.upper()
    
    if sequence_length is None:
        sequence_length = len(sequence)
    
    # Nucleotide to index mapping
    nuc_to_idx = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}
    
    # Initialize matrix
    encoding = np.zeros((sequence_length, 4), dtype=np.float32)
    
    # Fill in encoding
    for i, nuc in enumerate(sequence[:sequence_length]):
        if nuc in nuc_to_idx:
            encoding[i, nuc_to_idx[nuc]] = 1.0
    
    return encoding


def extract_one_hot_batch(
    sequences: List[str],
    sequence_length: int = 30
) -> np.ndarray:
    """
    Convert multiple sequences to one-hot encoding batch.
    
    Args:
        sequences: List of DNA sequence strings
        sequence_length: Expected sequence length
        
    Returns:
        NumPy array of shape (n_sequences, sequence_length, 4)
    """
    return np.array([
        extract_one_hot_encoding(seq, sequence_length)
        for seq in sequences
    ])


def extract_nucleotide_at_position(
    sequences: List[str],
    position: int
) -> Dict[str, int]:
    """
    Count nucleotides at a specific position across sequences.
    
    Args:
        sequences: List of DNA sequence strings
        position: Position index (0-based)
        
    Returns:
        Dictionary with nucleotide counts
    """
    counts = {nuc: 0 for nuc in NUCLEOTIDES}
    
    for seq in sequences:
        if position < len(seq):
            nuc = seq[position].upper()
            if nuc in counts:
                counts[nuc] += 1
    
    return counts


def extract_position_frequency_matrix(
    sequences: List[str],
    sequence_length: int = 20
) -> np.ndarray:
    """
    Build position frequency matrix (PFM).
    
    Args:
        sequences: List of DNA sequence strings
        sequence_length: Expected sequence length
        
    Returns:
        NumPy array of shape (4, sequence_length) with nucleotide frequencies
    """
    # Initialize count matrix
    count_matrix = np.zeros((4, sequence_length), dtype=np.float32)
    
    nuc_to_idx = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}
    
    valid_count = 0
    for seq in sequences:
        if len(seq) >= sequence_length:
            for i, nuc in enumerate(seq[:sequence_length]):
                if nuc in nuc_to_idx:
                    count_matrix[nuc_to_idx[nuc], i] += 1
            valid_count += 1
    
    # Normalize
    if valid_count > 0:
        count_matrix /= valid_count
    
    return count_matrix


def extract_conservation_score(
    sequences: List[str],
    sequence_length: int = 20
) -> np.ndarray:
    """
    Calculate conservation score at each position.
    
    Conservation = 1 - entropy/entropy_max
    Higher score means more conserved position.
    
    Args:
        sequences: List of DNA sequence strings
        sequence_length: Expected sequence length
        
    Returns:
        NumPy array of conservation scores
    """
    pfm = extract_position_frequency_matrix(sequences, sequence_length)
    
    conservation = np.zeros(sequence_length, dtype=np.float32)
    
    for i in range(sequence_length):
        # Get frequencies at this position
        freqs = pfm[:, i]
        
        # Calculate entropy
        freqs_nonzero = freqs[freqs > 0]
        if len(freqs_nonzero) > 0:
            entropy = -np.sum(freqs_nonzero * np.log2(freqs_nonzero))
            # Max entropy for 4 nucleotides is 2
            conservation[i] = 1 - entropy / 2.0
        else:
            conservation[i] = 0.0
    
    return conservation


def extract_position_specific_features(
    sequence: str,
    context_length: int = 30,
    guide_start: int = 4,
    guide_length: int = 20
) -> Dict[str, float]:
    """
    Extract position-specific features for guide sequence.
    
    For CRISPR 30-mer: guide is at positions 4-23.
    
    Args:
        sequence: 30-mer DNA sequence
        context_length: Full sequence length
        guide_start: Start position of guide
        guide_length: Length of guide
        
    Returns:
        Dictionary with position-specific features
    """
    if len(sequence) != context_length:
        raise ValueError(f"Expected length {context_length}, got {len(sequence)}")
    
    guide = sequence[guide_start:guide_start + guide_length]
    features = {}
    
    # One-hot encoding features for guide (always include all nucleotides for consistency)
    for i in range(len(guide)):
        for nuc in NUCLEOTIDES:
            features[f'guide_pos_{i}_{nuc}'] = 1.0 if (guide[i] == nuc) else 0.0
    
    return features


def calculate_positional_gc_content(
    sequence: str,
    window_size: int = 5
) -> np.ndarray:
    """
    Calculate GC content at each position using sliding window.
    
    Args:
        sequence: DNA sequence string
        window_size: Size of sliding window
        
    Returns:
        NumPy array of GC content values
    """
    sequence = sequence.upper()
    n = len(sequence)
    
    if n < window_size:
        return np.array([calculate_gc_content(sequence)])
    
    gc_values = np.zeros(n - window_size + 1, dtype=np.float32)
    
    for i in range(n - window_size + 1):
        window = sequence[i:i + window_size]
        gc_count = window.count('G') + window.count('C')
        gc_values[i] = gc_count / window_size
    
    return gc_values


def extract_dinucleotide_at_positions(
    sequence: str,
    positions: List[int]
) -> Dict[str, int]:
    """
    Extract dinucleotides at specific positions.
    
    Args:
        sequence: DNA sequence string
        positions: List of starting positions
        
    Returns:
        Dictionary with dinucleotide counts
    """
    sequence = sequence.upper()
    dinucs = {}
    
    for pos in positions:
        if 0 <= pos <= len(sequence) - 2:
            dinuc = sequence[pos:pos+2]
            dinucs[f'dinuc_{pos}_{dinuc}'] = 1
    
    return dinucs
