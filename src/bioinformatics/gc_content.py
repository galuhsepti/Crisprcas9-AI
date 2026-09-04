"""
GC content calculation module for CRISPR-Cas9 sgRNA analysis.

This module provides functions to calculate GC content in DNA sequences,
which is an important feature for predicting sgRNA activity.
"""

from typing import Union, List, Optional
import numpy as np


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    GC content = (count(G) + count(C)) / length
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as a fraction (0.0 to 1.0)
        
    Raises:
        ValueError: If sequence is empty
    """
    if not sequence:
        raise ValueError("Empty sequence")
    
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def calculate_gc_content_array(sequences: List[str]) -> np.ndarray:
    """
    Calculate GC content for multiple sequences.
    
    Args:
        sequences: List of DNA sequence strings
        
    Returns:
        NumPy array of GC content values
    """
    return np.array([calculate_gc_content(seq) for seq in sequences])


def calculate_gc_content_rolling(
    sequence: str,
    window_size: int = 5
) -> List[float]:
    """
    Calculate rolling GC content along a sequence.
    
    Args:
        sequence: DNA sequence string
        window_size: Size of the sliding window
        
    Returns:
        List of GC content values for each window position
    """
    if len(sequence) < window_size:
        raise ValueError(f"Sequence length {len(sequence)} < window_size {window_size}")
    
    sequence = sequence.upper()
    gc_values = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        gc_count = window.count('G') + window.count('C')
        gc_values.append(gc_count / window_size)
    
    return gc_values


def calculate_gc_skew(sequence: str) -> float:
    """
    Calculate GC skew: (G - C) / (G + C)
    
    Useful for identifying strand bias.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC skew value (-1.0 to 1.0)
    """
    if not sequence:
        raise ValueError("Empty sequence")
    
    sequence = sequence.upper()
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    
    if g_count + c_count == 0:
        return 0.0
    
    return (g_count - c_count) / (g_count + c_count)


def calculate_at_skew(sequence: str) -> float:
    """
    Calculate AT skew: (A - T) / (A + T)
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        AT skew value (-1.0 to 1.0)
    """
    if not sequence:
        raise ValueError("Empty sequence")
    
    sequence = sequence.upper()
    a_count = sequence.count('A')
    t_count = sequence.count('T')
    
    if a_count + t_count == 0:
        return 0.0
    
    return (a_count - t_count) / (a_count + t_count)


def calculate_gc_content_by_region(
    sequence: str,
    context_length: int = 30,
    guide_length: int = 20,
    guide_start: int = 4,
    pam_length: int = 3
) -> dict:
    """
    Calculate GC content for different regions of a 30-mer.
    
    For CRISPR 30-mer format:
    - 5' context: positions 0-3 (4 bp)
    - Guide: positions 4-23 (20 bp)
    - PAM: positions 24-26 (3 bp)
    - 3' context: positions 27-29 (3 bp)
    
    Args:
        sequence: 30-mer DNA sequence
        context_length: Expected full length
        guide_length: Expected guide length
        guide_start: Start position of guide (fixed at 4)
        pam_length: Length of PAM sequence (3 for NGG)
        
    Returns:
        Dictionary with GC content for each region
    """
    if len(sequence) != context_length:
        raise ValueError(f"Expected sequence length {context_length}, got {len(sequence)}")
    
    guide_end = guide_start + guide_length
    pam_start = guide_end
    pam_end = pam_start + pam_length
    
    return {
        'gc_5prime_context': calculate_gc_content(sequence[:guide_start]),
        'gc_guide': calculate_gc_content(sequence[guide_start:guide_end]),
        'gc_pam': calculate_gc_content(sequence[pam_start:pam_end]),
        'gc_3prime_context': calculate_gc_content(sequence[pam_end:]),
        'gc_full': calculate_gc_content(sequence),
        'gc_guide_skew': calculate_gc_skew(sequence[guide_start:guide_end]),
    }


def is_optimal_gc(gc_content: float, min_gc: float = 0.4, max_gc: float = 0.6) -> bool:
    """
    Check if GC content is within optimal range for CRISPR.
    
    Optimal GC content for sgRNAs is typically 40-60%.
    
    Args:
        gc_content: GC content as a fraction
        min_gc: Minimum optimal GC content
        max_gc: Maximum optimal GC content
        
    Returns:
        True if GC content is optimal
    """
    return min_gc <= gc_content <= max_gc
