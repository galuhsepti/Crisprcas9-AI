"""
Unit tests for bioinformatics feature extraction modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bioinformatics.gc_content import (
    calculate_gc_content,
    calculate_gc_content_array,
    calculate_gc_content_by_region,
    calculate_gc_skew,
    calculate_at_skew,
    is_optimal_gc
)
from src.bioinformatics.nucleotide_composition import (
    count_nucleotides,
    calculate_frequencies,
    calculate_frequencies_array,
    calculate_dinucleotide_frequencies,
    calculate_composition_features,
    calculate_heterogeneity,
    get_reverse_complement
)
from src.bioinformatics.kmer import (
    generate_kmers,
    extract_kmer_counts,
    extract_kmer_frequencies,
    extract_kmer_features,
    calculate_kmer_entropy,
    calculate_kmer_complexity,
    extract_guide_kmer_features
)
from src.bioinformatics.positional_features import (
    extract_one_hot_encoding,
    extract_one_hot_batch,
    extract_position_frequency_matrix,
    extract_conservation_score
)
from src.bioinformatics.sequence_features import SequenceFeatureExtractor


# Test sequences
VALID_30MER = "ACAGCTGATCTCCAGATATGACCATGGGTT"
VALID_20MER = "CTGATCTCCAGATATGACC"
SEQUENCES = [
    "ACAGCTGATCTCCAGATATGACCATGGGTT",
    "CAGCTGATCTCCAGATATGACCATGGGTTT",
    "CCAGAAGTTTGAGCCACAAACCCATGGTCA"
]


class TestGCContent:
    """Test GC content functions."""
    
    def test_calculate_gc_content(self):
        """Test basic GC content calculation."""
        # Sequence with known GC content
        seq = "ACGT"  # 50% GC
        assert calculate_gc_content(seq) == 0.5
    
    def test_gc_content_all_gc(self):
        """Test GC content with all GC."""
        seq = "GCGC"
        assert calculate_gc_content(seq) == 1.0
    
    def test_gc_content_no_gc(self):
        """Test GC content with no GC."""
        seq = "ATAT"
        assert calculate_gc_content(seq) == 0.0
    
    def test_gc_content_lowercase(self):
        """Test GC content with lowercase."""
        seq = "acgt"
        assert calculate_gc_content(seq) == 0.5
    
    def test_gc_content_empty(self):
        """Test GC content with empty sequence."""
        with pytest.raises(ValueError):
            calculate_gc_content("")
    
    def test_gc_content_by_region(self):
        """Test regional GC content."""
        seq = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        result = calculate_gc_content_by_region(seq)
        
        assert 'gc_guide' in result
        assert 'gc_pam' in result
        assert 'gc_full' in result
        assert 0 <= result['gc_guide'] <= 1
    
    def test_gc_skew(self):
        """Test GC skew calculation."""
        seq = "GCGC"  # Equal G and C
        assert calculate_gc_skew(seq) == 0.0
        
        seq = "GGGG"  # All G
        assert calculate_gc_skew(seq) == 1.0
    
    def test_at_skew(self):
        """Test AT skew calculation."""
        seq = "ATAT"  # Equal A and T
        assert calculate_at_skew(seq) == 0.0
    
    def test_is_optimal_gc(self):
        """Test optimal GC check."""
        assert is_optimal_gc(0.5) is True
        assert is_optimal_gc(0.3) is False
        assert is_optimal_gc(0.7) is False


class TestNucleotideComposition:
    """Test nucleotide composition functions."""
    
    def test_count_nucleotides(self):
        """Test nucleotide counting."""
        seq = "ACGT"
        counts = count_nucleotides(seq)
        assert counts == {'A': 1, 'C': 1, 'G': 1, 'T': 1}
    
    def test_calculate_frequencies(self):
        """Test frequency calculation."""
        seq = "ACGT"
        freq = calculate_frequencies(seq)
        assert freq == {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
    
    def test_calculate_frequencies_array(self):
        """Test frequency array calculation."""
        freqs = calculate_frequencies_array(["ACGT", "GGGG"])
        assert freqs.shape == (2, 4)
        assert freqs[1, 2] == 1.0  # G frequency for GGGG
    
    def test_dinucleotide_frequencies(self):
        """Test dinucleotide frequencies."""
        seq = "ACGTACGT"
        freq = calculate_dinucleotide_frequencies(seq)
        assert 'AC' in freq
        assert 'CG' in freq
        assert 'GT' in freq
    
    def test_composition_features(self):
        """Test composition features extraction."""
        seq = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        features = calculate_composition_features(seq)
        assert 'freq_A' in features
        assert 'gc_content' in features
        assert 'dinuc_AC' in features
    
    def test_heterogeneity(self):
        """Test heterogeneity calculation."""
        seq = "AAAA"  # Low heterogeneity
        h = calculate_heterogeneity(seq)
        assert h == 0.0
        
        seq = "ACGT"  # High heterogeneity
        h = calculate_heterogeneity(seq)
        assert h > 0
    
    def test_reverse_complement(self):
        """Test reverse complement."""
        assert get_reverse_complement("ATCG") == "CGAT"
        assert get_reverse_complement("AAAA") == "TTTT"
        assert get_reverse_complement("ACGT") == "ACGT"


class TestKmerFeatures:
    """Test k-mer feature functions."""
    
    def test_generate_kmers(self):
        """Test k-mer generation."""
        kmers = generate_kmers(2)
        assert len(kmers) == 16  # 4^2
        assert 'AA' in kmers
        assert 'TT' in kmers
    
    def test_extract_kmer_counts(self):
        """Test k-mer counting."""
        seq = "ACGTACGT"
        counts = extract_kmer_counts(seq, k=2)
        assert 'AC' in counts
        assert counts['AC'] == 2
    
    def test_extract_kmer_frequencies(self):
        """Test k-mer frequencies."""
        seq = "ACGTACGT"
        freq = extract_kmer_frequencies(seq, k=2)
        assert abs(sum(freq.values()) - 1.0) < 1e-6
    
    def test_extract_kmer_features(self):
        """Test k-mer features extraction."""
        seq = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        features = extract_kmer_features(seq, k_values=[2, 3])
        assert 'k2_AA' in features
        assert 'k3_AAA' in features
    
    def test_kmer_entropy(self):
        """Test k-mer entropy."""
        seq = "AAAA"  # Low entropy
        e = calculate_kmer_entropy(seq, k=1)
        assert e == 0.0
    
    def test_kmer_complexity(self):
        """Test k-mer complexity."""
        seq = "ACGTACGT"
        c = calculate_kmer_complexity(seq, k=2)
        assert 0 <= c <= 1
    
    def test_guide_kmer_features(self):
        """Test guide k-mer features."""
        seq = "ACAGCTGATCTCCAGATATGACCATGGGTT"
        features = extract_guide_kmer_features(seq)
        assert 'guide_k2_AA' in features or 'guide_k3_AAA' in features


class TestPositionalFeatures:
    """Test positional feature functions."""
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        seq = "ACGT"
        encoding = extract_one_hot_encoding(seq, sequence_length=4)
        assert encoding.shape == (4, 4)
        assert encoding[0, 0] == 1  # A at position 0
        assert encoding[1, 1] == 1  # C at position 1
    
    def test_one_hot_batch(self):
        """Test one-hot batch encoding."""
        batch = extract_one_hot_batch(["ACGT", "GGGG"], sequence_length=4)
        assert batch.shape == (2, 4, 4)
    
    def test_position_frequency_matrix(self):
        """Test position frequency matrix."""
        pfm = extract_position_frequency_matrix(
            ["ACGT", "ACGT", "ACGT"],
            sequence_length=4
        )
        assert pfm.shape == (4, 4)
        # First position should be all A
        assert pfm[0, 0] == 1.0
    
    def test_conservation_score(self):
        """Test conservation score."""
        # Identical sequences should have high conservation
        cons = extract_conservation_score(
            ["ACGT", "ACGT", "ACGT"],
            sequence_length=4
        )
        assert all(c > 0.9 for c in cons)


class TestSequenceFeatureExtractor:
    """Test integrated feature extractor."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = SequenceFeatureExtractor()
        assert extractor.context_length == 30
        assert extractor.guide_length == 20
    
    def test_extract_all_features(self):
        """Test full feature extraction."""
        extractor = SequenceFeatureExtractor()
        features = extractor.extract_all_features(VALID_30MER)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'gc_content' in features
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        extractor = SequenceFeatureExtractor()
        df = extractor.extract_features_batch(SEQUENCES)
        
        assert df.shape[0] == 3
        assert df.shape[1] > 0
    
    def test_feature_names(self):
        """Test feature names retrieval."""
        extractor = SequenceFeatureExtractor()
        names = extractor.get_feature_names()
        assert len(names) > 0
    
    def test_feature_count(self):
        """Test feature count."""
        extractor = SequenceFeatureExtractor()
        count = extractor.get_feature_count()
        assert count > 0


class TestGuideGeometry:
    """Regression tests for correct 30-mer geometry (guide [4:24], PAM [24:27])."""

    def test_guide_start_is_4(self):
        """guide_start must be 4, not derived as (30-20)//2 = 5."""
        extractor = SequenceFeatureExtractor()
        assert extractor.guide_start == 4
        assert extractor.guide_end == 24

    def test_pam_bounds(self):
        """PAM must span [24:27]."""
        extractor = SequenceFeatureExtractor()
        assert extractor.pam_start == 24
        assert extractor.pam_end == 27

    def test_gc_region_slices(self):
        """Regional GC slices must match guide [4:24] and PAM [24:27]."""
        from src.bioinformatics.gc_content import calculate_gc_content_by_region
        # Build a sequence where each region has distinct GC content:
        #  5' context (0-3): ACGT -> 0.5
        #  guide (4-23): C*20 -> 1.0
        #  PAM (24-26): GGG -> 1.0
        #  3' context (27-29): AAA -> 0.0
        seq = "ACGT" + "C" * 20 + "GGG" + "AAA"
        result = calculate_gc_content_by_region(seq)
        assert result['gc_5prime_context'] == 0.5
        assert result['gc_guide'] == 1.0
        assert result['gc_pam'] == 1.0
        assert result['gc_3prime_context'] == 0.0

    def test_positional_features_use_guide_4_24(self):
        """guide_pos_0 must correspond to position 4 of the 30-mer."""
        extractor = SequenceFeatureExtractor(include_gc=False, include_composition=False,
                                             include_kmer=False, include_positional=True)
        # Distinct at each position: position 4 of 30-mer is the first guide base
        seq = "TTTT" + "C" + "A" * 19 + "N" * 6  # guide = C + 19 A
        features = extractor.extract_all_features(seq)
        assert features['guide_pos_0_C'] == 1.0
        assert features['guide_pos_0_A'] == 0.0
        assert features['guide_pos_0_G'] == 0.0
        assert features['guide_pos_0_T'] == 0.0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_short_sequence(self):
        """Test with short sequence."""
        seq = "AC"
        freq = calculate_frequencies(seq)
        assert freq['A'] == 0.5
    
    def test_long_sequence(self):
        """Test with long sequence."""
        seq = "ACGT" * 1000
        gc = calculate_gc_content(seq)
        assert gc == 0.5
    
    def test_all_same_nucleotide(self):
        """Test with all same nucleotide."""
        seq = "AAAA"
        freq = calculate_frequencies(seq)
        assert freq['A'] == 1.0
        assert freq['C'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
