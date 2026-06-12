import numpy as np
import pytest

from lci_reduce.flow_priority import build_greedy_ladder, prefix_length_for_tau
from lci_reduce.reducer import signed_tau_cover


def test_signed_cancellation():
    matrix = np.array([[100, -90, 5]], dtype=float)
    result = signed_tau_cover(matrix, 0.95, exchange_keys=["a", "b", "c"])
    assert result["coverage_pos_by_category"][0] >= 0.95
    assert result["coverage_neg_by_category"][0] >= 0.95
    assert result["selected_neg"][1]


def test_negative_cf_effect():
    matrix = np.array([[-10, 20]], dtype=float)
    result = signed_tau_cover(matrix, 0.9, exchange_keys=["a", "b"])
    assert result["coverage_pos_by_category"][0] >= 0.9
    assert result["coverage_neg_by_category"][0] >= 0.9


def test_signed_tau_cover_uses_ratio_space_tolerance_for_tiny_active_rows():
    matrix = np.array([[1.0e-12, 0.9e-12]], dtype=float)
    result = signed_tau_cover(matrix, 0.95, exchange_keys=["a", "b"])
    assert result["selected"].tolist() == [True, True]
    assert result["coverage_pos_by_category"][0] == pytest.approx(1.0)


def test_priority_prefix_length_uses_ratio_space_threshold_for_tiny_active_rows():
    matrix = np.array([[1.0e-12, 0.9e-12]], dtype=float)
    ladder = build_greedy_ladder(matrix, exchange_keys=["a", "b"], tol=1e-12)
    assert prefix_length_for_tau(ladder, 0.95, tol=1e-12) == 2
