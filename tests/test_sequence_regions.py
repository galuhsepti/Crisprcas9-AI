"""
Unit tests for the sequence-region ablation module (Phase 7).
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ablation import extract_region_one_hot, region_sequence_length
from src.ablation.sequence_regions import (
    REGION_POSITIONS,
    SUPPORTED_REGIONS,
    GUIDE_START,
    GUIDE_END,
    PAM_END
)

# Build a canonical 30-mer manually with distinct nucleotides per region.
FIVE_FLANK = "ACGT"
GUIDE = "TACCGGTAATTTACCGGATG"   # 20 bp
PAM = "GGG"
THREE_FLANK = "TCA"
SEQ30 = FIVE_FLANK + GUIDE + PAM + THREE_FLANK

assert len(SEQ30) == 30


def require_one_hot_poi(x):
    """Assert x looks like a valid one-hot (n, L, 4)."""
    assert x.ndim == 3
    assert x.shape[2] == 4
    assert np.all(x.sum(axis=2) == 1.0)


class TestRegionExtraction:
    """Tests for extract_region_one_hot."""

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError):
            extract_region_one_hot([SEQ30], "pam_only")
        with pytest.raises(ValueError):
            region_sequence_length("nope")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            extract_region_one_hot(["ACGT"], "guide")

    def test_region_lengths(self):
        assert region_sequence_length('full') == 30
        assert region_sequence_length('guide') == 20
        assert region_sequence_length('guide_pam') == 23
        assert region_sequence_length('context_pam') == 10
        for r in SUPPORTED_REGIONS:
            assert region_sequence_length(r) == len(REGION_POSITIONS[r])

    def test_supported_regions_expected_set(self):
        assert set(SUPPORTED_REGIONS) == {'full', 'guide', 'guide_pam', 'context_pam'}

    def test_full_matches_reference_encoding(self):
        from src.bioinformatics.positional_features import extract_one_hot_encoding
        ref = extract_one_hot_encoding(SEQ30, 30)
        got = extract_region_one_hot([SEQ30], 'full')[0]
        assert np.array_equal(got, ref)

    def test_guide_extracts_correct_positions(self):
        from src.bioinformatics.positional_features import extract_one_hot_encoding
        got = extract_region_one_hot([SEQ30], 'guide')[0]
        require_one_hot_poi(got[np.newaxis, ...])
        ref = np.array([
            extract_one_hot_encoding(SEQ30[i], 1)[0]
            for i in range(GUIDE_START, GUIDE_END)
        ])
        assert np.array_equal(got, ref)

    def test_guide_is_subset_of_full(self):
        full = extract_region_one_hot([SEQ30], 'full')[0]
        guide = extract_region_one_hot([SEQ30], 'guide')[0]
        assert np.array_equal(guide, full[GUIDE_START:GUIDE_END, :])

    def test_guide_pam_is_subset_of_full(self):
        full = extract_region_one_hot([SEQ30], 'full')[0]
        gp = extract_region_one_hot([SEQ30], 'guide_pam')[0]
        assert np.array_equal(gp, full[GUIDE_START:PAM_END, :])

    def test_context_pam_concat_is_subset_of_full(self):
        full = extract_region_one_hot([SEQ30], 'full')[0]
        cp = extract_region_one_hot([SEQ30], 'context_pam')[0]
        expected = np.concatenate([
            full[0:GUIDE_START, :], full[GUIDE_END:30, :]
        ], axis=0)
        assert np.array_equal(cp, expected)

    def test_batch_shape_and_content(self):
        seqs = [SEQ30, "TTTT" + GUIDE + "AAA" + "CAC"]
        got = extract_region_one_hot(seqs, 'guide_pam')
        assert got.shape == (2, 23, 4)
        require_one_hot_poi(got)
        # The two sequences differ at PAM positions; encodings must differ.
        assert not np.array_equal(got[0], got[1])