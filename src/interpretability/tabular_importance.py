"""
Feature-importance aggregation for the tabular baselines (RF / XGBoost).

The 197 tabular features belong to a small number of conceptual groups:
GC/composition statistics, dinucleotide counts (k=2), trinucleotide counts
(k=3), and per-position one-hot guide features. Group-level aggregation lets us
compare, at the feature level, what the baselines rely on.

Interpretation notes
--------------------
- Random Forest importances are mean impurity decrease (Gini-based, computed
  at fit time); XGBoost 'gain' importances reflect average split-gain over
  trees. These are vendor metrics, not statistical tests, and groups/ordering
  should be read descriptively.
- Aggregated group shares are computed on the vendor values; they report the
  relative total weight of each group under that metric.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Ordered prefix rules; the first matching rule assigns the group.
_GROUP_RULES: List[str] = [
    'guide_pos_',
    'k3_',
    'k2_',
    'dinuc_',
    'gc_content',
    'gc_',
    'freq_',
    'gc_skew',
    'gc_guide_skew',
    'at_skew',
    'is_optimal_gc',
    'purine_content',
    'pyrimidine_content',
    'heterogeneity',
]

# Mapping rule-prefix groups -> display group (also used by classify_feature).
_GROUP_BY_RULE: List[str] = [
    'positional_guide_onehot',
    'kmer_3',
    'kmer_2',
    'kmer_2',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
    'gc_composition',
]


def classify_feature(name: str) -> str:
    """
    Assign a feature name to a conceptual group.

    Groups: 'positional_guide_onehot', 'kmer_3', 'kmer_2', 'gc_composition'
    or 'other' for anything unmatched.
    """
    for rule, group in zip(_GROUP_RULES, _GROUP_BY_RULE):
        if name.startswith(rule):
            return group
    return 'other'


def importance_frame(
    model,
    feature_names: Optional[List[str]] = None,
    importance_type: str = 'gain'
) -> pd.DataFrame:
    """
    Extract the feature-importance ranking of a fitted tabular model.

    Args:
        model: Fitted RandomForestModel or XGBoostModel wrapper (anything with
            ``get_feature_importance``).
        feature_names: Explicit feature list (used by models already storing
            their names; provided here as a fallback check).
        importance_type: For XGBoost: 'gain', 'weight' or 'cover'. Ignored by
            RandomForestModel.

    Returns:
        DataFrame with columns 'feature', 'importance', sorted descending.
    """
    try:
        frame = model.get_feature_importance(top_k=None,
                                             importance_type=importance_type)
    except TypeError:
        # RandomForestModel.get_feature_importance has no importance_type arg.
        frame = model.get_feature_importance(top_k=None)
    frame = frame[['feature', 'importance']].copy()
    if feature_names is not None:
        known = set(frame['feature'])
        missing = [f for f in feature_names if f not in known]
        if missing and model.feature_names is None:
            frame = pd.DataFrame({'feature': feature_names,
                                  'importance': np.zeros(len(feature_names))})
    frame = frame.sort_values('importance', ascending=False).reset_index(drop=True)
    return frame


def aggregate_importance_by_group(
    importance_df: pd.DataFrame,
    normalize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Sum feature importances within each conceptual group.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        normalize: Divide by the total importance so each group yields a
            fraction summing to 1.

    Returns:
        Mapping group name -> {'importance': sum, 'n_features': count,
        'share': fraction (if normalize)}.
    """
    if 'feature' not in importance_df.columns or \
       'importance' not in importance_df.columns:
        raise ValueError("importance_df must have 'feature' and 'importance' columns")

    frame = importance_df.copy()
    frame['group'] = frame['feature'].map(classify_feature)

    groups = {}
    for group, sub in frame.groupby('group'):
        total = float(sub['importance'].sum())
        share = total / float(frame['importance'].sum()) if normalize else None
        groups[group] = {
            'importance': total,
            'n_features': int(len(sub)),
            'share': share
        }
    return groups


def top_features(importance_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Return the top-k features by importance."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return importance_df.head(k).reset_index(drop=True)