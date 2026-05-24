import numpy as np

from lci_reduce.reducer import greedy_tau_cover


def test_zero_categories():
    matrix = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
    selected = greedy_tau_cover(matrix, 0.9, exchange_keys=["a", "b", "c"])
    retained = matrix[:, selected].sum(axis=1)
    full = matrix.sum(axis=1)
    assert full[0] == 0
    assert retained[1] >= 0.9 * full[1] - 1e-12
