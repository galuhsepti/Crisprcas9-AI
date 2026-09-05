"""
Unit tests for tabular feature-importance utilities (Phase 8).
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interpretability.tabular_importance import (
    classify_feature,
    importance_frame,
    aggregate_importance_by_group,
    top_features
)


class TestClassifyFeature:
    """Tests for classify_feature."""

    @pytest.mark.parametrize("name,expected", [
        ('gc_content', 'gc_composition'),
        ('gc_guide', 'gc_composition'),
        ('gc_skew', 'gc_composition'),
        ('gc_guide_skew', 'gc_composition'),
        ('at_skew', 'gc_composition'),
        ('is_optimal_gc', 'gc_composition'),
        ('purine_content', 'gc_composition'),
        ('pyrimidine_content', 'gc_composition'),
        ('heterogeneity', 'gc_composition'),
        ('freq_A', 'gc_composition'),
        ('dinuc_GG', 'kmer_2'),
        ('k2_TA', 'kmer_2'),
        ('k2_entropy', 'kmer_2'),
        ('k3_ACG', 'kmer_3'),
        ('k3_entropy', 'kmer_3'),
        ('guide_pos_0_A', 'positional_guide_onehot'),
        ('guide_pos_19_T', 'positional_guide_onehot'),
        ('bogus_feature', 'other'),
    ])
    def test_mapping(self, name, expected):
        assert classify_feature(name) == expected


class TestAggregateImportance:
    """Tests for aggregate_importance_by_group."""

    def test_single_group_sum_and_shares(self):
        df = pd.DataFrame({
            'feature': ['guide_pos_0_A', 'guide_pos_1_C', 'gc_content'],
            'importance': [2.0, 3.0, 5.0]
        })
        res = aggregate_importance_by_group(df)
        assert res['positional_guide_onehot']['n_features'] == 2
        assert res['positional_guide_onehot']['importance'] == pytest.approx(5.0)
        assert res['positional_guide_onehot']['share'] == pytest.approx(0.5)
        assert res['gc_composition']['share'] == pytest.approx(0.5)
        total = sum(v['share'] for v in res.values())
        assert total == pytest.approx(1.0)

    def test_unmatched_features_land_in_other(self):
        df = pd.DataFrame({
            'feature': ['whatever_x', 'dinuc_AA'],
            'importance': [1.0, 9.0]
        })
        res = aggregate_importance_by_group(df)
        assert res['kmer_2']['share'] == pytest.approx(0.9)
        assert res['other']['share'] == pytest.approx(0.1)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            aggregate_importance_by_group(pd.DataFrame({'feature': ['a']}))


class TestTopFeatures:
    """Tests for top_features."""

    def test_top_k(self):
        df = pd.DataFrame({'feature': [f'f{i}' for i in range(5)],
                           'importance': np.arange(5, 0, -1)})
        top = top_features(df, k=2)
        assert list(top['feature']) == ['f0', 'f1']
        assert top['importance'].tolist() == [5.0, 4.0]

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            top_features(pd.DataFrame({'feature': ['a'], 'importance': [1.0]}), k=0)


class TestImportanceFrameOnFittedModels:
    """integration checks on the model wrappers."""

    @staticmethod
    def _synthetic():
        rng = np.random.default_rng(7)
        n = 300
        # Signal lives in the first feature; last two are noise.
        X = np.column_stack([
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        ])
        y = 4.0 * X[:, 0] + 0.1 * rng.normal(size=n)
        names = ['guide_pos_3_A', 'guide_pos_5_C', 'kmer_3_ACG', 'noise_thing']
        return X, y, names

    def test_rf_frame_and_groups(self):
        from src.models import RandomForestModel
        X, y, names = self._synthetic()
        model = RandomForestModel(n_estimators=50, max_depth=6,
                                  random_state=42)
        model.fit(X, y, feature_names=names)
        frame = importance_frame(model)
        assert set(frame.columns) == {'feature', 'importance'}
        assert len(frame) == 4
        assert frame.iloc[0]['feature'] == 'guide_pos_3_A'
        groups = aggregate_importance_by_group(frame)
        assert groups['positional_guide_onehot']['share'] > 0.8

    def test_xgb_frame_and_groups(self):
        from src.models import XGBoostModel
        X, y, names = self._synthetic()
        model = XGBoostModel(n_estimators=50, max_depth=3,
                             learning_rate=0.1, random_state=42)
        model.fit(X, y, feature_names=names)
        frame = importance_frame(model, importance_type='gain')
        assert set(frame.columns) == {'feature', 'importance'}
        assert len(frame) <= 4
        groups = aggregate_importance_by_group(frame)
        # The signal feature should dominate the positional group.
        assert groups['positional_guide_onehot']['share'] > 0.4