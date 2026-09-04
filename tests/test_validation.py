"""
Unit tests for sequence validation module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validation import (
    validate_nucleotides,
    validate_sequence_length,
    validate_pam_sequence,
    validate_guide_sequence,
    validate_sequence,
    is_valid_dna,
    normalize_sequence,
    convert_rna_to_dna
)


class TestNucleotideValidation:
    """Test nucleotide validation functions."""
    
    def test_valid_sequence(self):
        """Test valid DNA sequence."""
        is_valid, error = validate_nucleotides("ACGT")
        assert is_valid is True
        assert error is None
    
    def test_valid_sequence_lowercase(self):
        """Test valid sequence in lowercase."""
        is_valid, error = validate_nucleotides("acgt")
        assert is_valid is True
        assert error is None
    
    def test_invalid_sequence(self):
        """Test sequence with invalid nucleotides."""
        is_valid, error = validate_nucleotides("ACGU")
        assert is_valid is False
        assert "U" in error
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        is_valid, error = validate_nucleotides("")
        assert is_valid is False
        assert "Empty" in error
    
    def test_sequence_with_numbers(self):
        """Test sequence with numbers."""
        is_valid, error = validate_nucleotides("AC12GT")
        assert is_valid is False
    
    def test_sequence_with_spaces(self):
        """Test sequence with spaces."""
        is_valid, error = validate_nucleotides("AC GT")
        assert is_valid is False


class TestSequenceLength:
    """Test sequence length validation."""
    
    def test_correct_length(self):
        """Test sequence with correct length."""
        is_valid, error = validate_sequence_length("ACGTACGT", expected_length=8)
        assert is_valid is True
        assert error is None
    
    def test_incorrect_length(self):
        """Test sequence with incorrect length."""
        is_valid, error = validate_sequence_length("ACGT", expected_length=8)
        assert is_valid is False
        assert "Expected length 8" in error
    
    def test_too_short(self):
        """Test sequence that is too short."""
        is_valid, error = validate_sequence_length("AC", min_length=5)
        assert is_valid is False
        assert "too short" in error
    
    def test_too_long(self):
        """Test sequence that is too long."""
        is_valid, error = validate_sequence_length("ACGTACGTACGT", max_length=8)
        assert is_valid is False
        assert "too long" in error
    
    def test_within_range(self):
        """Test sequence within length range."""
        is_valid, error = validate_sequence_length("ACGT", min_length=3, max_length=5)
        assert is_valid is True
        assert error is None


class TestPAMValidation:
    """Test PAM sequence validation."""
    
    def test_valid_pam(self):
        """Test valid NGG PAM sequence."""
        # 30-mer with PAM at position 24-26
        sequence = "ACAGCTGATCTCCAGATATGACCATGGGTT"  # PAM = GGG
        is_valid, error = validate_pam_sequence(sequence, pam_start=24, pam_length=3)
        assert is_valid is True
        assert error is None
    
    def test_invalid_pam(self):
        """Test invalid PAM sequence."""
        # 30-mer with invalid PAM
        sequence = "ACAGCTGATCTCCAGATATGACCATGAACT"  # PAM = AAC
        is_valid, error = validate_pam_sequence(sequence, pam_start=24, pam_length=3)
        assert is_valid is False
        assert "Invalid PAM" in error
    
    def test_pam_too_short(self):
        """Test sequence too short for PAM."""
        sequence = "ACGT"
        is_valid, error = validate_pam_sequence(sequence, pam_start=24, pam_length=3)
        assert is_valid is False
        assert "too short" in error
    
    def test_pam_with_n(self):
        """Test PAM with N (any nucleotide) at first position."""
        # 30-mer with PAM = NGG at positions 24-26 (0-indexed)
        # 24 chars context/guide + NGG + 3 chars = 30
        sequence = "ACAGCTGATCTCCAGATATGTCTANGGTCA"  # PAM = NGG
        is_valid, error = validate_pam_sequence(sequence, pam_start=24, pam_length=3)
        assert is_valid is True
        assert error is None


class TestGuideValidation:
    """Test guide sequence validation."""
    
    def test_valid_guide(self):
        """Test valid guide sequence."""
        sequence = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        is_valid, error = validate_guide_sequence(sequence, guide_start=4, guide_length=20)
        assert is_valid is True
        assert error is None
    
    def test_guide_with_homopolymer(self):
        """Test guide with homopolymer run."""
        sequence = "ACAGCTGATCTCCAGATTTTTACCATGGGTT"  # TTTTT at positions 18-22
        is_valid, error = validate_guide_sequence(sequence, guide_start=4, guide_length=20)
        assert is_valid is False
        assert "homopolymer" in error
    
    def test_guide_too_short(self):
        """Test sequence too short for guide."""
        sequence = "ACGT"
        is_valid, error = validate_guide_sequence(sequence, guide_start=4, guide_length=20)
        assert is_valid is False
        assert "too short" in error


class TestComprehensiveValidation:
    """Test comprehensive sequence validation."""
    
    def test_valid_30mer(self):
        """Test valid 30-mer sequence."""
        sequence = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        is_valid, errors = validate_sequence(sequence, context_length=30)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_nucleotides(self):
        """Test sequence with invalid nucleotides."""
        sequence = "ACAGCTGATCTCCAGATATGACCATGGUTT"
        is_valid, errors = validate_sequence(sequence, context_length=30)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_wrong_length(self):
        """Test sequence with wrong length."""
        sequence = "ACGTACGT"
        is_valid, errors = validate_sequence(sequence, context_length=30)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_no_pam_check(self):
        """Test sequence without PAM validation."""
        sequence = "ACAGCTGATCTCCAGATATGACCATGAACT"  # Invalid PAM
        is_valid, errors = validate_sequence(
            sequence,
            context_length=30,
            check_pam=False
        )
        # Should pass without PAM check
        assert is_valid is True or len(errors) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_is_valid_dna(self):
        """Test is_valid_dna function."""
        assert is_valid_dna("ACGT") is True
        assert is_valid_dna("ACGU") is False
        assert is_valid_dna("") is False
    
    def test_normalize_sequence(self):
        """Test normalize_sequence function."""
        assert normalize_sequence("acgt") == "ACGT"
        assert normalize_sequence("  ACGT  ") == "ACGT"
        assert normalize_sequence("AcGt") == "ACGT"
    
    def test_convert_rna_to_dna(self):
        """Test convert_rna_to_dna function."""
        assert convert_rna_to_dna("ACGU") == "ACGT"
        assert convert_rna_to_dna("acgu") == "ACGT"
        assert convert_rna_to_dna("ACGT") == "ACGT"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_nucleotide(self):
        """Test single nucleotide."""
        is_valid, error = validate_nucleotides("A")
        assert is_valid is True
    
    def test_long_sequence(self):
        """Test very long sequence."""
        long_seq = "ACGT" * 1000
        is_valid, error = validate_nucleotides(long_seq)
        assert is_valid is True
    
    def test_all_same_nucleotides(self):
        """Test sequence with all same nucleotides."""
        is_valid, error = validate_nucleotides("AAAA")
        assert is_valid is True
    
    def test_mixed_case(self):
        """Test mixed case sequence."""
        is_valid, error = validate_nucleotides("aCcGgT")
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
