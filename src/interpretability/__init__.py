"""
Interpretability module for the CRISPR-Cas9 sgRNA prediction pipeline.
"""

from .cnn_attribution import (
    position_saliency,
    integrated_gradients,
    attribution_by_region
)
from .tabular_importance import (
    classify_feature,
    importance_frame,
    aggregate_importance_by_group,
    top_features
)

__all__ = [
    'position_saliency',
    'integrated_gradients',
    'attribution_by_region',
    'classify_feature',
    'importance_frame',
    'aggregate_importance_by_group',
    'top_features'
]