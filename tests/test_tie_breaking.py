import numpy as np

from lci_reduce.reducer import greedy_tau_cover


def test_deterministic_tie_breaking():
    matrix = np.array([[1, 1]], dtype=float)
    selected = greedy_tau_cover(matrix, 0.5, exchange_keys=["b", "a"])
    assert selected.tolist() == [False, True]


def test_equal_exchange_keys_fall_back_to_original_order():
    matrix = np.array([[1, 1]], dtype=float)
    selected = greedy_tau_cover(matrix, 0.5, exchange_keys=["same", "same"])
    assert selected.tolist() == [True, False]
