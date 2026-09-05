"""
Domain-adaptation module for the Phase 9C experiment.

Provides internal-only importance reweighting utilities used to retrain the
canonical model families under a domain-weighted training recipe, then lock
the selected recipe for a single external (Moreno-Mateos) evaluation.
"""

from .weighting import (
    inverse_density_weights,
    uniform_weights,
    combine_weights,
    label_reweighting_weights,
    gc_reweighting_weights,
    combined_reweighting_weights,
    weight_summary,
)

__all__ = [
    'inverse_density_weights',
    'uniform_weights',
    'combine_weights',
    'label_reweighting_weights',
    'gc_reweighting_weights',
    'combined_reweighting_weights',
    'weight_summary',
]