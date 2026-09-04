"""
Bioinformatics module for CRISPR-Cas9 sgRNA prediction pipeline.

This module provides functions for extracting biological features
from DNA sequences for machine learning models.
"""

from .gc_content import (
    calculate_gc_content,
    calculate_gc_content_array,
    calculate_gc_content_by_region,
    calculate_gc_skew,
    calculate_at_skew,
    is_optimal_gc
)

from .nucleotide_composition import (
    count_nucleotides,
    calculate_frequencies,
    calculate_frequencies_array,
    calculate_position_frequencies,
    calculate_dinucleotide_frequencies,
    calculate_kmer_frequencies,
    calculate_composition_features,
    get_reverse_complement,
    calculate_heterogeneity
)

from .kmer import (
    generate_kmers,
    extract_kmer_counts,
    extract_kmer_frequencies,
    extract_kmer_features,
    extract_kmer_matrix,
    calculate_kmer_entropy,
    calculate_kmer_complexity,
    extract_guide_kmer_features
)

from .positional_features import (
    extract_position_features,
    extract_one_hot_encoding,
    extract_one_hot_batch,
    extract_nucleotide_at_position,
    extract_position_frequency_matrix,
    extract_conservation_score,
    extract_position_specific_features
)

from .sequence_features import (
    SequenceFeatureExtractor,
    extract_one_hot_for_cnn
)

__all__ = [
    # GC content
    'calculate_gc_content',
    'calculate_gc_content_array',
    'calculate_gc_content_by_region',
    'calculate_gc_skew',
    'calculate_at_skew',
    'is_optimal_gc',
    
    # Nucleotide composition
    'count_nucleotides',
    'calculate_frequencies',
    'calculate_frequencies_array',
    'calculate_position_frequencies',
    'calculate_dinucleotide_frequencies',
    'calculate_kmer_frequencies',
    'calculate_composition_features',
    'get_reverse_complement',
    'calculate_heterogeneity',
    
    # k-mer
    'generate_kmers',
    'extract_kmer_counts',
    'extract_kmer_frequencies',
    'extract_kmer_features',
    'extract_kmer_matrix',
    'calculate_kmer_entropy',
    'calculate_kmer_complexity',
    'extract_guide_kmer_features',
    
    # Positional features
    'extract_position_features',
    'extract_one_hot_encoding',
    'extract_one_hot_batch',
    'extract_nucleotide_at_position',
    'extract_position_frequency_matrix',
    'extract_conservation_score',
    'extract_position_specific_features',
    
    # Sequence features
    'SequenceFeatureExtractor',
    'extract_one_hot_for_cnn'
]
