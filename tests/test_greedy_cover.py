import numpy as np
import pytest

from lci_reduce.reducer import greedy_tau_cover


def test_basic_multi_category_cover():
    matrix = np.array(
        [
            [55, 20, 0, 25, 0],
            [20, 55, 0, 0, 25],
            [35, 0, 40, 0, 50],
            [0, 0, 25, 0, 0],
        ],
        dtype=float,
    )
    tau = 0.7
    selected = greedy_tau_cover(matrix, tau, exchange_keys=["0", "1", "2", "3", "4"])
    assert selected.shape == (5,)
    retained = matrix[:, selected].sum(axis=1)
    full = matrix.sum(axis=1)
    active = full > 1e-12
    assert np.all(retained[active] >= tau * full[active] - 1e-12)


def test_invalid_tau():
    matrix = np.array([[1.0]])
    with pytest.raises(ValueError):
        greedy_tau_cover(matrix, 0)
    with pytest.raises(ValueError):
        greedy_tau_cover(matrix, 1.1)


def test_non_finite_input():
    with pytest.raises(ValueError):
        greedy_tau_cover(np.array([[np.nan]]), 0.9)
    with pytest.raises(ValueError):
        greedy_tau_cover(np.array([[np.inf]]), 0.9)


def test_negative_input_rejected():
    with pytest.raises(ValueError):
        greedy_tau_cover(np.array([[-1.0, 2.0]]), 0.9)


def test_small_positive_contributions_near_tolerance_still_cover():
    matrix = np.array([[6e-13, 6e-13]], dtype=float)
    selected = greedy_tau_cover(matrix, 0.95, exchange_keys=["a", "b"], tol=1e-12)
    retained = matrix[:, selected].sum(axis=1)
    full = matrix.sum(axis=1)
    active = full > 1e-12
    assert active[0]
    assert retained[0] >= 0.95 * full[0] - 1e-12


def test_greedy_cover_is_invariant_to_category_rescaling():
    matrix = np.array(
        [
            [90.0, 0.0, 10.0],
            [0.0, 5.0, 1.0],
        ],
        dtype=float,
    )
    scaled = matrix.copy()
    scaled[0] *= 1000.0

    base_selected = greedy_tau_cover(matrix, 0.8, exchange_keys=["a", "b", "c"])
    scaled_selected = greedy_tau_cover(scaled, 0.8, exchange_keys=["a", "b", "c"])

    assert base_selected.tolist() == [True, True, False]
    assert scaled_selected.tolist() == [True, True, False]
