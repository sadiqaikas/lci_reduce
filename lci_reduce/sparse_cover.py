"""Exact sparse cover structures and greedy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .contribution import _CategoryRowSet
from .errors import CoverageError
from .models import CoverageRowMetadata


def _coverage_ratios(retained: np.ndarray, full: np.ndarray, active: np.ndarray) -> np.ndarray:
    coverage = np.ones_like(full, dtype=float)
    if active.any():
        coverage[active] = retained[active] / full[active]
    return coverage


def _coverage_requirement_satisfied(
    retained: np.ndarray,
    full: np.ndarray,
    active: np.ndarray,
    tau: float,
    tol: float,
) -> bool:
    if not active.any():
        return True
    coverage = _coverage_ratios(retained, full, active)
    return bool(np.all(coverage[active] + tol >= tau))


def _coverage_gap_exceeds_tolerance(
    demand: np.ndarray,
    full: np.ndarray,
    active: np.ndarray,
    tol: float,
) -> bool:
    if not active.any():
        return False
    coverage_gap = np.zeros_like(full, dtype=float)
    coverage_gap[active] = demand[active] / full[active]
    return bool(np.any(coverage_gap[active] > tol))


@dataclass(frozen=True)
class SparseExactBlock:
    row_start: int
    row_stop: int
    columns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SparseCoverMatrix:
    n_rows: int
    n_cols: int
    full: np.ndarray
    active: np.ndarray
    weights: np.ndarray
    row_metadata: Tuple[CoverageRowMetadata, ...]
    exact_blocks: Tuple[SparseExactBlock, ...]
    exact_ranges_by_col: Tuple[Tuple[Tuple[int, int, float], ...], ...]
    scenario_row_indices: np.ndarray
    scenario_col_indices: np.ndarray
    scenario_values: np.ndarray
    scenario_rows_by_col: Tuple[np.ndarray, ...]
    scenario_values_by_col: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SparseGreedyLadder:
    order: Tuple[int, ...]
    entry_thresholds: np.ndarray
    lambda_after: np.ndarray
    full: np.ndarray
    active: np.ndarray


def build_sparse_cover_matrix(
    rowsets: Sequence[_CategoryRowSet],
    n_cols: int,
    *,
    positive: bool,
    tol: float,
) -> SparseCoverMatrix:
    row_metadata: list[CoverageRowMetadata] = []
    full_values: list[float] = []
    exact_blocks: list[SparseExactBlock] = []
    exact_ranges_by_col: list[list[tuple[int, int, float]]] = [[] for _ in range(n_cols)]
    scenario_rows_by_col: list[list[int]] = [[] for _ in range(n_cols)]
    scenario_values_by_col: list[list[float]] = [[] for _ in range(n_cols)]
    flat_row_indices: list[int] = []
    flat_col_indices: list[int] = []
    flat_values: list[float] = []

    row_cursor = 0
    for rowset in rowsets:
        row_count = len(rowset.scenario_rows)
        block_end = row_cursor + row_count
        exact_columns: list[int] = []
        exact_values: list[float] = []
        exact_row_sum = 0.0
        for local_index, value in rowset.exact_items:
            signed_value = float(value) if positive else -float(value)
            if signed_value <= tol:
                continue
            exact_columns.append(int(local_index))
            exact_values.append(signed_value)
            exact_row_sum += signed_value
            exact_ranges_by_col[int(local_index)].append((row_cursor, block_end, signed_value))
        if exact_columns:
            exact_blocks.append(
                SparseExactBlock(
                    row_start=row_cursor,
                    row_stop=block_end,
                    columns=np.asarray(exact_columns, dtype=np.int64),
                    values=np.asarray(exact_values, dtype=float),
                )
            )
        for local_row_index, scenario_sparse in enumerate(rowset.scenario_rows):
            global_row_index = row_cursor + local_row_index
            row_total = exact_row_sum
            for local_index, value in scenario_sparse:
                signed_value = float(value) if positive else -float(value)
                if signed_value <= tol:
                    continue
                local_index = int(local_index)
                row_total += signed_value
                flat_row_indices.append(global_row_index)
                flat_col_indices.append(local_index)
                flat_values.append(signed_value)
                scenario_rows_by_col[local_index].append(global_row_index)
                scenario_values_by_col[local_index].append(signed_value)
            full_values.append(row_total)
            if local_row_index < len(rowset.metadata_rows):
                row_metadata.append(rowset.metadata_rows[local_row_index])
            else:
                row_metadata.append(
                    CoverageRowMetadata(
                        row_id=f"row-{global_row_index}",
                        category_id="",
                        category_name="",
                        scenario_id=str(global_row_index),
                        scenario_label="",
                        scenario_type="",
                    )
                )
        row_cursor = block_end

    full = np.asarray(full_values, dtype=float)
    active = full > tol
    weights = np.zeros(full.size, dtype=float)
    weights[active] = 1.0 / full[active]
    return SparseCoverMatrix(
        n_rows=full.size,
        n_cols=n_cols,
        full=full,
        active=active,
        weights=weights,
        row_metadata=tuple(row_metadata),
        exact_blocks=tuple(exact_blocks),
        exact_ranges_by_col=tuple(tuple(item) for item in exact_ranges_by_col),
        scenario_row_indices=np.asarray(flat_row_indices, dtype=np.int64),
        scenario_col_indices=np.asarray(flat_col_indices, dtype=np.int64),
        scenario_values=np.asarray(flat_values, dtype=float),
        scenario_rows_by_col=tuple(np.asarray(rows, dtype=np.int64) for rows in scenario_rows_by_col),
        scenario_values_by_col=tuple(np.asarray(values, dtype=float) for values in scenario_values_by_col),
    )


def _deterministic_tie_rank(n_cols: int, exchange_keys: Sequence[str] | None) -> np.ndarray:
    keys = list(exchange_keys) if exchange_keys is not None else [f"{index:012d}" for index in range(n_cols)]
    if len(keys) != n_cols:
        raise ValueError("exchange_keys length must match number of exchanges")
    tie_order = np.lexsort((np.arange(n_cols), np.asarray(keys, dtype=object)))
    tie_rank = np.empty(n_cols, dtype=int)
    tie_rank[tie_order] = np.arange(n_cols)
    return tie_rank


def _block_exact_scores(
    block: SparseExactBlock,
    demand: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if block.columns.size == 0 or block.row_start >= block.row_stop:
        return np.zeros(block.columns.size, dtype=float)
    block_demand = demand[block.row_start:block.row_stop]
    block_weights = weights[block.row_start:block.row_stop]
    active = block_weights > 0.0
    if not np.any(active):
        return np.zeros(block.columns.size, dtype=float)
    sorted_indices = np.argsort(block_demand[active], kind="mergesort")
    demand_sorted = block_demand[active][sorted_indices]
    weights_sorted = block_weights[active][sorted_indices]
    prefix_weighted_demand = np.cumsum(demand_sorted * weights_sorted)
    suffix_weights = np.cumsum(weights_sorted[::-1])[::-1]
    insertion_points = np.searchsorted(demand_sorted, block.values, side="left")
    scores = np.zeros(block.values.size, dtype=float)
    has_prefix = insertion_points > 0
    scores[has_prefix] += prefix_weighted_demand[insertion_points[has_prefix] - 1]
    has_suffix = insertion_points < demand_sorted.size
    scores[has_suffix] += block.values[has_suffix] * suffix_weights[insertion_points[has_suffix]]
    return scores


def _compute_scores(
    model: SparseCoverMatrix,
    demand: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(model.n_cols, dtype=float)
    if model.scenario_row_indices.size:
        scenario_scores = np.minimum(model.scenario_values, demand[model.scenario_row_indices])
        scenario_scores *= model.weights[model.scenario_row_indices]
        scores += np.bincount(
            model.scenario_col_indices,
            weights=scenario_scores,
            minlength=model.n_cols,
        )
    for block in model.exact_blocks:
        block_scores = _block_exact_scores(block, demand, model.weights)
        if block_scores.size:
            np.add.at(scores, block.columns, block_scores)
    scores[selected] = -np.inf
    return scores


def add_column_contribution(model: SparseCoverMatrix, target: np.ndarray, column_index: int) -> None:
    for row_start, row_stop, value in model.exact_ranges_by_col[column_index]:
        target[row_start:row_stop] += float(value)
    row_indices = model.scenario_rows_by_col[column_index]
    if row_indices.size:
        target[row_indices] += model.scenario_values_by_col[column_index]


def subtract_column_from_demand(model: SparseCoverMatrix, demand: np.ndarray, column_index: int) -> None:
    for row_start, row_stop, value in model.exact_ranges_by_col[column_index]:
        demand[row_start:row_stop] = np.maximum(demand[row_start:row_stop] - float(value), 0.0)
    row_indices = model.scenario_rows_by_col[column_index]
    if row_indices.size:
        demand[row_indices] = np.maximum(demand[row_indices] - model.scenario_values_by_col[column_index], 0.0)


def retained_by_row(model: SparseCoverMatrix, selected: np.ndarray) -> np.ndarray:
    retained = np.zeros(model.n_rows, dtype=float)
    for block in model.exact_blocks:
        if block.columns.size == 0:
            continue
        block_sum = float(block.values[selected[block.columns]].sum())
        if block_sum > 0.0:
            retained[block.row_start:block.row_stop] += block_sum
    if model.scenario_row_indices.size:
        active_entries = selected[model.scenario_col_indices]
        if np.any(active_entries):
            np.add.at(
                retained,
                model.scenario_row_indices[active_entries],
                model.scenario_values[active_entries],
            )
    return retained


def column_nonzero_mask(model: SparseCoverMatrix) -> np.ndarray:
    return np.array(
        [
            bool(model.exact_ranges_by_col[index]) or bool(model.scenario_rows_by_col[index].size)
            for index in range(model.n_cols)
        ],
        dtype=bool,
    )


def build_greedy_ladder_sparse(
    model: SparseCoverMatrix,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> SparseGreedyLadder:
    entry_thresholds = np.full(model.n_cols, np.nan, dtype=float)
    if model.n_rows == 0 or model.n_cols == 0 or not model.active.any():
        return SparseGreedyLadder(
            order=tuple(),
            entry_thresholds=entry_thresholds,
            lambda_after=np.zeros(0, dtype=float),
            full=model.full,
            active=model.active,
        )
    tie_rank = _deterministic_tie_rank(model.n_cols, exchange_keys)
    selected = np.zeros(model.n_cols, dtype=bool)
    retained = np.zeros(model.n_rows, dtype=float)
    order: list[int] = []
    lambda_after: list[float] = []
    while True:
        remaining = np.maximum(model.full - retained, 0.0)
        scores = _compute_scores(model, remaining, selected)
        best_score = float(scores.max(initial=-np.inf))
        if not np.isfinite(best_score) or best_score <= 0.0:
            break
        contenders = np.flatnonzero(~selected & (best_score - scores <= tol))
        if contenders.size == 0:
            best_index = int(np.argmax(scores))
        else:
            best_index = int(contenders[np.argmin(tie_rank[contenders])])
        coverage_before = np.ones(model.n_rows, dtype=float)
        coverage_before[model.active] = np.clip(retained[model.active] / model.full[model.active], 0.0, 1.0)
        entry_thresholds[best_index] = float(coverage_before[model.active].min()) if model.active.any() else 1.0
        selected[best_index] = True
        order.append(best_index)
        add_column_contribution(model, retained, best_index)
        coverage_after = np.ones(model.n_rows, dtype=float)
        coverage_after[model.active] = np.clip(retained[model.active] / model.full[model.active], 0.0, 1.0)
        lambda_after.append(float(coverage_after[model.active].min()) if model.active.any() else 1.0)
    return SparseGreedyLadder(
        order=tuple(order),
        entry_thresholds=entry_thresholds,
        lambda_after=np.asarray(lambda_after, dtype=float),
        full=model.full,
        active=model.active,
    )


def greedy_tau_cover_sparse(
    model: SparseCoverMatrix,
    tau: float,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    selected = np.zeros(model.n_cols, dtype=bool)
    if model.n_rows == 0 or model.n_cols == 0 or not model.active.any():
        return selected
    tie_rank = _deterministic_tie_rank(model.n_cols, exchange_keys)
    demand = np.where(model.active, tau * model.full, 0.0)
    while _coverage_gap_exceeds_tolerance(demand, model.full, model.active, tol):
        scores = _compute_scores(model, demand, selected)
        best_score = float(scores.max(initial=-np.inf))
        if not np.isfinite(best_score) or best_score <= 0.0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        contenders = np.flatnonzero(~selected & (best_score - scores <= tol))
        if contenders.size == 0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        best_index = int(contenders[np.argmin(tie_rank[contenders])])
        selected[best_index] = True
        subtract_column_from_demand(model, demand, best_index)
    return selected


def signed_tau_cover_sparse(
    positive_model: SparseCoverMatrix,
    negative_model: SparseCoverMatrix,
    tau: float,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> dict[str, np.ndarray]:
    selected_pos = greedy_tau_cover_sparse(positive_model, tau, exchange_keys=exchange_keys, tol=tol)
    selected_neg = greedy_tau_cover_sparse(negative_model, tau, exchange_keys=exchange_keys, tol=tol)
    selected = selected_pos | selected_neg
    retained_pos = retained_by_row(positive_model, selected)
    retained_neg = retained_by_row(negative_model, selected)
    if not _coverage_requirement_satisfied(retained_pos, positive_model.full, positive_model.active, tau, tol):
        raise CoverageError("Positive tau coverage verification failed")
    if not _coverage_requirement_satisfied(retained_neg, negative_model.full, negative_model.active, tau, tol):
        raise CoverageError("Negative tau coverage verification failed")
    coverage_pos = _coverage_ratios(retained_pos, positive_model.full, positive_model.active)
    coverage_neg = _coverage_ratios(retained_neg, negative_model.full, negative_model.active)
    return {
        "selected": selected,
        "selected_pos": selected_pos,
        "selected_neg": selected_neg,
        "full_pos_by_category": positive_model.full,
        "full_neg_by_category": negative_model.full,
        "retained_pos_by_category": retained_pos,
        "retained_neg_by_category": retained_neg,
        "coverage_pos_by_category": coverage_pos,
        "coverage_neg_by_category": coverage_neg,
        "active_pos_by_category": positive_model.active,
        "active_neg_by_category": negative_model.active,
    }
