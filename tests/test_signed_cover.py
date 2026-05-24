import numpy as np

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
