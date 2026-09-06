"""
Phase 10 - Multi-Dataset Training-Domain Diversification experiment.

Modules
-------
config : pre-registered protocol, constants and verdict rule
datasets : dataset inventory, loaders and provenance registry
audit : contamination + duplicate audits
pool : training-pool construction with provenance columns
stats : paired external statistics (bootstrap CI, Cohen's dz, tests)

The locked Moreno-Mateos external test is never used for method selection.
"""

from . import config, datasets, audit, pool, stats
from .config import decide_verdict, VERDICT_RULE

__all__ = ["config", "datasets", "audit", "pool", "stats", "decide_verdict"]