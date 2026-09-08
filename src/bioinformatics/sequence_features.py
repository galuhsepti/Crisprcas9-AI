"""
Sequence features module for CRISPR-Cas9 sgRNA analysis.

This module integrates all bioinformatics feature extraction functions
into a unified interface.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from .gc_content import (
    calculate_gc_content,
    calculate_gc_content_by_region,
    calculate_gc_skew,
    calculate_at_skew,
    is_optimal_gc
)
from .nucleotide_composition import (
    calculate_frequencies,
    calculate_dinucleotide_frequencies,
    calculate_heterogeneity,
)
from .kmer import (
    extract_kmer_features,
    calculate_kmer_entropy,
    calculate_kmer_complexity
)
from .positional_features import (
    extract_one_hot_batch,
    extract_position_specific_features
)


class SequenceFeatureExtractor:
    """
    Comprehensive feature extractor for CRISPR-Cas9 sgRNA sequences.
    """
    
    def __init__(
        self,
        context_length: int = 30,
        guide_length: int = 20,
        guide_start: int = 4,
        pam_length: int = 3,
        k_values: List[int] = [2, 3],
        include_one_hot: bool = True,
        include_gc: bool = True,
        include_composition: bool = True,
        include_kmer: bool = True,
        include_positional: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            context_length: Full context sequence length
            guide_length: Guide sequence length
            guide_start: Start position of guide (0-indexed), fixed at 4 for 30-mer:
                         4bp 5' context (0-3) + 20bp guide (4-23) + 3bp PAM (24-26)
                         + 3bp 3' context (27-29)
            pam_length: Length of PAM sequence (3 for NGG)
            k_values: List of k values for k-mer features
            include_one_hot: Include one-hot encoding features
            include_gc: Include GC content features
            include_composition: Include nucleotide composition features
            include_kmer: Include k-mer features
            include_positional: Include positional features
        """
        self.context_length = context_length
        self.guide_length = guide_length
        self.guide_start = guide_start
        self.pam_length = pam_length
        self.guide_end = guide_start + guide_length
        self.pam_start = guide_start + guide_length
        self.pam_end = guide_start + guide_length + pam_length
        self.k_values = k_values
        self.include_one_hot = include_one_hot
        self.include_gc = include_gc
        self.include_composition = include_composition
        self.include_kmer = include_kmer
        self.include_positional = include_positional
    
    def extract_gc_features(self, sequence: str) -> Dict[str, float]:
        """Extract GC content features."""
        features = {}
        
        # Overall GC
        features['gc_content'] = calculate_gc_content(sequence)
        
        # Regional GC
        gc_regions = calculate_gc_content_by_region(
            sequence,
            self.context_length,
            self.guide_length
        )
        features.update(gc_regions)
        
        # Skew features
        features['gc_skew'] = calculate_gc_skew(sequence)
        features['at_skew'] = calculate_at_skew(sequence)
        
        # Optimal GC flag
        features['is_optimal_gc'] = float(is_optimal_gc(features['gc_content']))
        
        return features
    
    def extract_composition_features(self, sequence: str) -> Dict[str, float]:
        """Extract nucleotide composition features."""
        features = {}
        
        # Basic frequencies
        freq = calculate_frequencies(sequence)
        for nuc, f in freq.items():
            features[f'freq_{nuc}'] = f
        
        # Additional composition features
        features['purine_content'] = freq['A'] + freq['G']
        features['pyrimidine_content'] = freq['C'] + freq['T']
        
        # Dinucleotide frequencies
        dinuc_freq = calculate_dinucleotide_frequencies(sequence)
        for dinuc, f in dinuc_freq.items():
            features[f'dinuc_{dinuc}'] = f
        
        # Heterogeneity
        features['heterogeneity'] = calculate_heterogeneity(sequence)
        
        return features
    
    def extract_kmer_features(self, sequence: str) -> Dict[str, float]:
        """Extract k-mer features."""
        features = {}
        
        for k in self.k_values:
            kmer_features = extract_kmer_features(sequence, [k])
            features.update(kmer_features)
            
            # Entropy and complexity
            features[f'k{k}_entropy'] = calculate_kmer_entropy(sequence, k)
            features[f'k{k}_complexity'] = calculate_kmer_complexity(sequence, k)
        
        return features
    
    def extract_positional_features(self, sequence: str) -> Dict[str, float]:
        """Extract positional features."""
        features = {}
        
        # Position-specific features for guide
        pos_features = extract_position_specific_features(
            sequence,
            self.context_length,
            self.guide_start,
            self.guide_length
        )
        features.update(pos_features)
        
        return features
    
    def extract_all_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract all features from a single sequence.
        
        Args:
            sequence: 30-mer DNA sequence
            
        Returns:
            Dictionary with all extracted features
        """
        sequence = sequence.upper()
        features = {}
        
        if self.include_gc:
            features.update(self.extract_gc_features(sequence))
        
        if self.include_composition:
            features.update(self.extract_composition_features(sequence))
        
        if self.include_kmer:
            features.update(self.extract_kmer_features(sequence))
        
        if self.include_positional:
            features.update(self.extract_positional_features(sequence))
        
        return features
    
    def extract_features_batch(
        self,
        sequences: List[str],
        include_one_hot: bool = False
    ) -> pd.DataFrame:
        """
        Extract features for multiple sequences.
        
        Args:
            sequences: List of 30-mer DNA sequences
            include_one_hot: Whether to include one-hot encoding
            
        Returns:
            DataFrame with extracted features
        """
        feature_list = []
        
        for seq in sequences:
            features = self.extract_all_features(seq)
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # Create a dummy sequence to extract feature names
        dummy_seq = 'A' * self.context_length
        features = self.extract_all_features(dummy_seq)
        return list(features.keys())
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_feature_names())


def extract_one_hot_for_cnn(
    sequences: List[str],
    context_length: int = 30
) -> np.ndarray:
    """
    Extract one-hot encoding for CNN input.
    
    Args:
        sequences: List of 30-mer DNA sequences
        context_length: Sequence length
        
    Returns:
        NumPy array of shape (n_sequences, context_length, 4)
    """
    return extract_one_hot_batch(sequences, context_length)
