"""
Sequence validation module for CRISPR-Cas9 sgRNA prediction pipeline.

This module provides functions to validate DNA sequences for CRISPR-Cas9
sgRNA activity prediction.
"""

from typing import Optional, Tuple, List
import re


# Valid DNA nucleotides
VALID_NUCLEOTIDES = {'A', 'C', 'G', 'T'}

# Standard SpCas9 PAM sequence (NGG, where N is any nucleotide)
PAM_PATTERN = re.compile(r'^[ACGTN]GG$')


def validate_nucleotides(sequence: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a sequence contains only valid DNA nucleotides.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, "Empty sequence"
    
    sequence_upper = sequence.upper()
    invalid_chars = set(sequence_upper) - VALID_NUCLEOTIDES
    
    if invalid_chars:
        return False, f"Invalid nucleotides found: {invalid_chars}"
    
    return True, None


def validate_sequence_length(
    sequence: str,
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate sequence length.
    
    Args:
        sequence: DNA sequence string
        expected_length: Exact expected length (optional)
        min_length: Minimum acceptable length (defaults to expected_length if set)
        max_length: Maximum acceptable length (defaults to expected_length if set)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    seq_len = len(sequence)
    
    if expected_length is not None:
        if seq_len != expected_length:
            return False, f"Expected length {expected_length}, got {seq_len}"
        return True, None
    
    if min_length is not None and seq_len < min_length:
        return False, f"Sequence too short: {seq_len} < {min_length}"
    
    if max_length is not None and seq_len > max_length:
        return False, f"Sequence too long: {seq_len} > {max_length}"
    
    return True, None


def validate_pam_sequence(
    sequence: str,
    pam_start: int = 20,
    pam_length: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Validate PAM sequence in context sequence.
    
    For SpCas9, PAM is NGG located immediately 3' of the 20bp guide.
    In a 30-mer context (4bp + 20bp guide + 3bp PAM + 3bp),
    PAM is at positions 24-26 (0-indexed: 23-25).
    
    Args:
        sequence: Full context sequence (30-mer)
        pam_start: Start position of PAM (0-indexed)
        pam_length: Length of PAM to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(sequence) < pam_start + pam_length:
        return False, f"Sequence too short to contain PAM at position {pam_start}"
    
    pam_seq = sequence[pam_start:pam_start + pam_length].upper()
    
    # Check if PAM matches NGG pattern
    if not PAM_PATTERN.match(pam_seq):
        return False, f"Invalid PAM sequence: {pam_seq} (expected NGG)"
    
    return True, None


def validate_guide_sequence(
    sequence: str,
    guide_start: int = 4,
    guide_length: int = 20
) -> Tuple[bool, Optional[str]]:
    """
    Validate guide RNA sequence within context.
    
    Args:
        sequence: Full context sequence (30-mer)
        guide_start: Start position of guide (0-indexed)
        guide_length: Length of guide sequence
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(sequence) < guide_start + guide_length:
        return False, f"Sequence too short to contain guide at position {guide_start}"
    
    guide_seq = sequence[guide_start:guide_start + guide_length].upper()
    
    # Check for homopolymer runs (4+ consecutive same nucleotides)
    for nuc in 'ACGT':
        if nuc * 4 in guide_seq:
            return False, f"Guide contains homopolymer run of {nuc}"
    
    return True, None


def validate_sequence(
    sequence: str,
    context_length: int = 30,
    guide_length: int = 20,
    check_pam: bool = True
) -> Tuple[bool, List[str]]:
    """
    Comprehensive sequence validation.
    
    Args:
        sequence: DNA sequence string
        context_length: Expected full context length
        guide_length: Expected guide length
        check_pam: Whether to validate PAM sequence
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Check nucleotides
    is_valid, error = validate_nucleotides(sequence)
    if not is_valid:
        errors.append(error)
    
    # Check length
    is_valid, error = validate_sequence_length(
        sequence,
        expected_length=context_length
    )
    if not is_valid:
        errors.append(error)
    
    # Check guide sequence
    guide_start = (context_length - guide_length) // 2
    is_valid, error = validate_guide_sequence(
        sequence,
        guide_start=guide_start,
        guide_length=guide_length
    )
    if not is_valid:
        errors.append(error)
    
    # Check PAM if requested
    if check_pam and len(sequence) >= context_length:
        pam_start = guide_start + guide_length
        is_valid, error = validate_pam_sequence(
            sequence,
            pam_start=pam_start,
            pam_length=3
        )
        if not is_valid:
            errors.append(error)
    
    return len(errors) == 0, errors


def is_valid_dna(sequence: str) -> bool:
    """
    Quick check if sequence contains only valid DNA nucleotides.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        True if valid, False otherwise
    """
    is_valid, _ = validate_nucleotides(sequence)
    return is_valid


def normalize_sequence(sequence: str) -> str:
    """
    Normalize sequence to uppercase and strip whitespace.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Normalized sequence
    """
    return sequence.upper().strip()


def convert_rna_to_dna(sequence: str) -> str:
    """
    Convert RNA sequence to DNA by replacing U with T.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        DNA sequence string
    """
    return sequence.upper().replace('U', 'T')
