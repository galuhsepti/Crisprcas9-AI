"""
Diagnostics module for Phase 9A (model diagnostics & domain-shift analysis).
"""

from .stats import (
    descriptive_stats,
    proportion_near_zero,
    proportion_above,
    ks_test,
    welch_ttest,
    cohens_d,
    histogram_data,
    ecdf_data
)
from .sequence_analysis import (
    gc_content_array,
    nucleotide_frequency_array,
    mean_positional_frequency_matrix,
    positional_entropy,
    mean_kmer_frequencies,
    sequence_length_summary,
    ambiguous_character_summary,
    duplicate_summary,
    exact_sequence_overlap,
    top_differing_kmers,
    kmer_profile_correlation
)
from .model_diagnostics import (
    prediction_summary,
    error_by_bins,
    model_agreement,
    prediction_vs_true_regression
)

__all__ = [
    'descriptive_stats',
    'proportion_near_zero',
    'proportion_above',
    'ks_test',
    'welch_ttest',
    'cohens_d',
    'histogram_data',
    'ecdf_data',
    'gc_content_array',
    'nucleotide_frequency_array',
    'mean_positional_frequency_matrix',
    'positional_entropy',
    'mean_kmer_frequencies',
    'sequence_length_summary',
    'ambiguous_character_summary',
    'duplicate_summary',
    'exact_sequence_overlap',
    'top_differing_kmers',
    'kmer_profile_correlation',
    'prediction_summary',
    'error_by_bins',
    'model_agreement',
    'prediction_vs_true_regression'
]