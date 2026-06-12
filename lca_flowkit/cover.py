"""Deterministic signed tau-cover utilities.

This module contains the central reduction rule used by the package.
Everything else in :mod:`lca_flowkit` eventually builds a non-negative matrix
and calls these functions.

The key mathematical idea is:

``matrix[row, column]``
    characterised contribution of one exchange occurrence (column) to one
    coverage row (row).

For reduction we do not work directly on the signed matrix.  We split it into:

``A_pos = max(A, 0)``
    positive characterised contributions.

``A_neg = max(-A, 0)``
    magnitudes of negative characterised contributions.

Greedy tau-cover is then applied to ``A_pos`` and ``A_neg`` separately, and
their selected columns are unioned later.  This mirrors the scientific rule in
 the project instructions and makes the coverage guarantee explicit:

``retained_sign[row] >= tau * full_sign[row]``

for every active row of each sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import CoverageError


def _coverage_ratios(retained: np.ndarray, full: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Compute per-row coverage with inactive rows treated as already covered.

    ``active`` marks rows whose full magnitude is meaningfully positive.  A row
    with ``full == 0`` has no demand to satisfy, so by convention we treat its
    coverage as ``1`` rather than dividing by zero.
    """
    coverage = np.ones_like(full, dtype=float)
    if active.any():
        coverage[active] = retained[active] / full[active]
    return coverage


def _gap_exceeds_tolerance(demand: np.ndarray, full: np.ndarray, active: np.ndarray, tol: float) -> bool:
    """Test the remaining demand in ratio space, not raw magnitude space.

    The reducer's certificate is stated in *fractional* coverage terms.  A row
    with a very large raw magnitude should not dominate the stopping criterion
    merely because its numbers are larger.  Dividing by ``full`` keeps the test
    aligned with the scientific statement ``retained >= tau * full``.
    """
    if not active.any():
        return False
    ratios = np.zeros_like(full, dtype=float)
    ratios[active] = demand[active] / full[active]
    return bool(np.any(ratios[active] > tol))


def _tie_rank(keys: Sequence[str]) -> np.ndarray:
    """Precompute deterministic tie-breaking rank from keys then order.

    ``np.lexsort`` returns the order that would sort the keys.  We then invert
    that permutation so later code can ask "which of these tied columns has the
    best deterministic rank?" without rebuilding the sort each iteration.
    """
    order = np.lexsort((np.arange(len(keys)), np.asarray(keys, dtype=object)))
    rank = np.empty(len(keys), dtype=int)
    rank[order] = np.arange(len(keys))
    return rank


def greedy_tau_cover(matrix: np.ndarray, tau: float, exchange_keys: Sequence[str], tol: float = 1e-12) -> np.ndarray:
    """Run the weighted greedy tau-cover on a non-negative contribution matrix.

    The input ``matrix`` must already be non-negative.  In practice this means
    callers pass either the positive or the negative sign-split contribution
    matrix.

    The greedy score for each currently unselected column ``e`` is:

    ``score[e] = sum_r min(matrix[r, e], demand[r]) / full[r]``

    where:

    ``full[r] = sum_e matrix[r, e]``
        total magnitude available in row ``r``.

    ``demand[r]``
        the remaining amount needed to reach ``tau * full[r]``.

    The ``min`` term means a column only receives credit for filling the part
    of a row that is still missing.  The division by ``full[r]`` makes the
    rule invariant to simple rescaling of one LCIA row.

    The algorithm is deterministic, but it is not globally optimal.  It is a
    reproducible heuristic that satisfies the tau certificate when a solution
    exists in the provided matrix.
    """
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("matrix must be a finite 2D array")
    if (array < 0).any():
        raise ValueError("greedy_tau_cover expects non-negative contributions")
    n_rows, n_cols = array.shape
    if len(exchange_keys) != n_cols:
        raise ValueError("exchange_keys length must match matrix columns")
    selected = np.zeros(n_cols, dtype=bool)
    if n_rows == 0 or n_cols == 0:
        return selected
    full = array.sum(axis=1)
    active = full > tol
    if not active.any():
        return selected
    weights = np.zeros(n_rows, dtype=float)
    weights[active] = 1.0 / full[active]
    demand = np.where(active, tau * full, 0.0)
    rank = _tie_rank(exchange_keys)
    while _gap_exceeds_tolerance(demand, full, active, tol):
        # ``demand[:, None]`` turns the 1D demand vector into a column-shaped
        # array so NumPy broadcasts it against every exchange column.
        scores = (np.minimum(array, demand[:, None]) * weights[:, None]).sum(axis=0)
        scores[selected] = -np.inf
        best_score = float(scores.max(initial=-np.inf))
        if not np.isfinite(best_score) or best_score <= 0.0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        # Columns within tolerance of the best score are tied.  We break those
        # ties using the precomputed stable rank derived from exchange ID and
        # original order.
        contenders = np.flatnonzero(~selected & (best_score - scores <= tol))
        if contenders.size == 0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        chosen = int(contenders[np.argmin(rank[contenders])])
        selected[chosen] = True
        demand = np.maximum(demand - array[:, chosen], 0.0)
    return selected


def verify_signed_coverage(matrix: np.ndarray, selected: np.ndarray, tau: float, tol: float = 1e-12) -> dict[str, object]:
    """Verify positive and negative sign-split coverage for a fixed selection.

    This is the formal certificate check used after selection.  Given a signed
    matrix ``A`` and a chosen column mask ``selected``, we compute:

    ``A_pos = max(A, 0)``
    ``A_neg = max(-A, 0)``

    and verify for every active row:

    ``sum_selected A_pos[row, :] >= tau * sum_all A_pos[row, :]``
    ``sum_selected A_neg[row, :] >= tau * sum_all A_neg[row, :]``

    The returned dictionary intentionally contains both booleans and raw arrays
    so callers can write auditable manifests rather than only a pass/fail flag.
    """
    array = np.asarray(matrix, dtype=float)
    chosen = np.asarray(selected, dtype=bool)
    positive = np.maximum(array, 0.0)
    negative = np.maximum(-array, 0.0)
    full_pos = positive.sum(axis=1)
    full_neg = negative.sum(axis=1)
    retained_pos = positive[:, chosen].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    retained_neg = negative[:, chosen].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    active_pos = full_pos > tol
    active_neg = full_neg > tol
    cover_pos = _coverage_ratios(retained_pos, full_pos, active_pos)
    cover_neg = _coverage_ratios(retained_neg, full_neg, active_neg)
    return {
        "positive_cover_ok": bool(np.all(cover_pos[active_pos] + tol >= tau)) if active_pos.any() else True,
        "negative_cover_ok": bool(np.all(cover_neg[active_neg] + tol >= tau)) if active_neg.any() else True,
        "min_positive_coverage": float(cover_pos[active_pos].min()) if active_pos.any() else 1.0,
        "min_negative_coverage": float(cover_neg[active_neg].min()) if active_neg.any() else 1.0,
        "n_active_positive_rows": int(active_pos.sum()),
        "n_active_negative_rows": int(active_neg.sum()),
        "retained_pos": retained_pos,
        "retained_neg": retained_neg,
        "full_pos": full_pos,
        "full_neg": full_neg,
        "cover_pos": cover_pos,
        "cover_neg": cover_neg,
    }


def signed_tau_cover(matrix: np.ndarray, tau: float, exchange_keys: Sequence[str], tol: float = 1e-12) -> dict[str, np.ndarray]:
    """Apply greedy tau-cover separately to positive and negative contributions.

    This function is a thin orchestration layer around :func:`greedy_tau_cover`.
    Its job is to enforce the project rule that positive and negative
    characterised contributions are covered independently before their selected
    exchange sets are unioned.
    """
    array = np.asarray(matrix, dtype=float)
    positive = np.maximum(array, 0.0)
    negative = np.maximum(-array, 0.0)
    positive_selected = greedy_tau_cover(positive, tau, exchange_keys, tol=tol)
    negative_selected = greedy_tau_cover(negative, tau, exchange_keys, tol=tol)
    selected = positive_selected | negative_selected
    verification = verify_signed_coverage(array, selected, tau, tol=tol)
    if not verification["positive_cover_ok"]:
        raise CoverageError("Positive signed coverage verification failed")
    if not verification["negative_cover_ok"]:
        raise CoverageError("Negative signed coverage verification failed")
    return {
        "selected": selected,
        "selected_pos": positive_selected,
        "selected_neg": negative_selected,
        "full_pos_by_row": verification["full_pos"],
        "full_neg_by_row": verification["full_neg"],
        "retained_pos_by_row": verification["retained_pos"],
        "retained_neg_by_row": verification["retained_neg"],
        "coverage_pos_by_row": verification["cover_pos"],
        "coverage_neg_by_row": verification["cover_neg"],
    }


@dataclass(frozen=True)
class GreedyLadder:
    """Greedy selection order plus the tau entry thresholds of each column.

    ``order``
        the deterministic column order chosen by the greedy routine.

    ``entry_thresholds[column]``
        the minimum row coverage *before* that column entered the ladder.
        Intuitively, this is the largest tau at which the column is still not
        required yet.

    ``lambda_after[k]``
        the minimum row coverage after selecting the first ``k + 1`` columns of
        ``order``.  This lets the priority workflow ask how much of the ladder
        would be retained at a chosen audit tau.
    """
    order: tuple[int, ...]
    entry_thresholds: np.ndarray
    lambda_after: np.ndarray
    full: np.ndarray
    active: np.ndarray


def build_greedy_ladder(matrix: np.ndarray, exchange_keys: Sequence[str], tol: float = 1e-12) -> GreedyLadder:
    """Record the greedy selection ladder for later tau-audit calculations.

    The ladder is not a second algorithm.  It replays the same greedy rule used
    in reduction, but instead of stopping at one ``tau`` it records the whole
    selection sequence.  The priority workflow later reuses this sequence to
    ask:

    - when does each flow first enter the retained prefix?
    - what is the exact single-flow loss if one selected flow is omitted?

    Keeping the ladder derived from the same rule prevents drift between the
    reducer and the priority-file generator.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("matrix must be a finite 2D array")
    if (array < 0).any():
        raise ValueError("Greedy ladders require non-negative contributions")
    n_rows, n_cols = array.shape
    if len(exchange_keys) != n_cols:
        raise ValueError("exchange_keys length must match matrix columns")
    full = array.sum(axis=1)
    active = full > tol
    entry_thresholds = np.full(n_cols, np.nan, dtype=float)
    if n_rows == 0 or n_cols == 0 or not active.any():
        return GreedyLadder(order=tuple(), entry_thresholds=entry_thresholds, lambda_after=np.zeros(0), full=full, active=active)
    weights = np.zeros(n_rows, dtype=float)
    weights[active] = 1.0 / full[active]
    rank = _tie_rank(exchange_keys)
    selected = np.zeros(n_cols, dtype=bool)
    retained = np.zeros(n_rows, dtype=float)
    order: list[int] = []
    lambda_after: list[float] = []
    while True:
        remaining = np.maximum(full - retained, 0.0)
        scores = (np.minimum(array, remaining[:, None]) * weights[:, None]).sum(axis=0)
        scores[selected] = -np.inf
        best_score = float(scores.max(initial=-np.inf))
        if not np.isfinite(best_score) or best_score <= 0.0:
            break
        contenders = np.flatnonzero(~selected & np.isclose(scores, best_score, rtol=tol, atol=0.0))
        chosen = int(contenders[np.argmin(rank[contenders])]) if contenders.size else int(np.argmax(scores))
        coverage_before = _coverage_ratios(retained, full, active)
        entry_thresholds[chosen] = float(coverage_before[active].min()) if active.any() else 1.0
        selected[chosen] = True
        retained += array[:, chosen]
        coverage_after = _coverage_ratios(retained, full, active)
        order.append(chosen)
        lambda_after.append(float(coverage_after[active].min()) if active.any() else 1.0)
    return GreedyLadder(order=tuple(order), entry_thresholds=entry_thresholds, lambda_after=np.asarray(lambda_after, dtype=float), full=full, active=active)


def prefix_length_for_tau(ladder: GreedyLadder, tau: float, tol: float = 1e-12) -> int:
    """Return how much of a greedy ladder is retained at a given tau.

    ``lambda_after`` stores the minimum achieved coverage after each greedy
    step.  The first index where ``lambda_after >= tau`` is therefore the
    shortest retained prefix that satisfies the requested threshold.
    """
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    for index, threshold in enumerate(ladder.lambda_after, start=1):
        if float(threshold) + tol >= tau:
            return index
    return len(ladder.order)


def single_flow_shortfall(tau: float, coverage_before_loss: float, loss: float) -> float:
    """Exact single-flow shortfall for one witness context.

    For one selected flow in one already-fixed witness context:

    ``eta = max(tau - (coverage_before_loss - loss), 0)``

    where ``loss`` is the normalised contribution of that flow to the witness
    row.  This formula is exact for a *single-flow omission*.  It is not an
    exact grouped-flow formula.
    """
    return max(float(tau) - (float(coverage_before_loss) - float(loss)), 0.0)
