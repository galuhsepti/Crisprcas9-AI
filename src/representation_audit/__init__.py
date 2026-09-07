"""
Phase 13: Controlled Representation Audit.

Tests whether replacing the canonical 30-mer one-hot sequence representation
with a richer (learned character-embedding) representation changes predictive
resolution / external generalization, under the SAME training data, SAME
canonical split, SAME locked external test, and a controlled model-development
procedure.

Reported phases in this module reuse the canonical data, splits, checkpoint,
predictions, and evaluation conventions from Phases 3-12.
"""

from .config import REPRESENTATION_AUDIT_CONFIG, Phase13ExperimentConfig
from .representation import NucleotideEmbeddingRepresentation, KMerEmbeddingRepresentation
from .model import EmbeddingCNNModel, EmbeddingCNN
from .evaluation import (
    run_internal_evaluation,
    run_final_external_evaluation,
    prediction_compression_analysis,
)
from .stats import (
    percentile_bootstrap_ci,
    bootstrap_delta_mae_ci,
    reproducible_bootstrap_seed,
)

__all__ = [
    'REPRESENTATION_AUDIT_CONFIG',
    'Phase13ExperimentConfig',
    'NucleotideEmbeddingRepresentation',
    'KMerEmbeddingRepresentation',
    'EmbeddingCNNModel',
    'EmbeddingCNN',
    'run_internal_evaluation',
    'run_final_external_evaluation',
    'prediction_compression_analysis',
    'percentile_bootstrap_ci',
    'bootstrap_delta_mae_ci',
    'reproducible_bootstrap_seed',
]
