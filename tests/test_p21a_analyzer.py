from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from scripts.analyze_ssi_mag_scd_p21a import _assert_covariance_identity, _corr


def test_spearman_uses_average_ranks_for_ties() -> None:
    left = np.array([1.0, 1.0, 2.0, 4.0, 4.0])
    right = np.array([2.0, 3.0, 3.0, 5.0, 5.0])
    expected = float(spearmanr(left, right).statistic)
    assert _corr(left, right, spearman=True) == pytest.approx(expected)


def test_covariance_identity_is_checked_for_nonconstant_eta() -> None:
    content = np.array([0.0, 1.0, 2.0, 3.0])
    reference = np.array([0.0, -0.5, 0.5, 1.0])
    relation = np.array([1.0, 0.0, -1.0, 2.0])
    eta = 2.0 + content + reference + relation
    values = _assert_covariance_identity(content, reference, relation, eta, label="synthetic")
    assert values[-1] == pytest.approx(1.0)


def test_covariance_identity_rejects_inconsistent_components() -> None:
    with pytest.raises(AssertionError, match="covariance identity failed"):
        _assert_covariance_identity(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 4.0]),
            label="inconsistent",
        )
