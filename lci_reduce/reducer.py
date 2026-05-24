"""Reduction algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .cf_resolution import CFResolutionManager
from .contribution import (
    build_contribution_matrix,
    elementary_manifest_base,
    exchange_flow_id,
    has_provider,
    is_quantitative_reference,
)
from .errors import CoverageError, MissingFlowError, UncharacterisedExchangeError
from .models import CFAmbiguityRecord, FlowInfo, ImpactCategory, ProcessReductionResult, UnitInfo, WarningRecord


def greedy_tau_cover(
    M: np.ndarray,
    tau: float,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    array = np.asarray(M, dtype=float)
    if array.ndim != 2:
        raise ValueError("M must be 2D")
    if not np.isfinite(array).all():
        raise ValueError("M must be finite")
    if (array < 0).any():
        raise ValueError("greedy_tau_cover only accepts non-negative matrices")
    n_categories, n_exchanges = array.shape
    selected = np.zeros(n_exchanges, dtype=bool)
    if n_categories == 0 or n_exchanges == 0:
        return selected
    full = array.sum(axis=1)
    active = full > tol
    if not active.any():
        return selected
    weights = np.zeros(n_categories, dtype=float)
    weights[active] = 1.0 / full[active]
    demand = np.where(active, tau * full, 0.0)
    keys = list(exchange_keys) if exchange_keys is not None else [f"{index:012d}" for index in range(n_exchanges)]
    if len(keys) != n_exchanges:
        raise ValueError("exchange_keys length must match number of exchanges")
    # Precompute the deterministic tie order once so repeated greedy rounds stay cheap.
    tie_order = np.lexsort((np.arange(n_exchanges), np.asarray(keys, dtype=object)))
    tie_rank = np.empty(n_exchanges, dtype=int)
    tie_rank[tie_order] = np.arange(n_exchanges)
    while np.any(demand[active] > tol):
        scores = (np.minimum(array, demand[:, None]) * weights[:, None]).sum(axis=0)
        scores[selected] = -np.inf
        best_score = float(scores.max(initial=-np.inf))
        # A small positive marginal fill can still be required to satisfy demand when
        # the characterised contributions are close to the numerical tolerance.
        if not np.isfinite(best_score) or best_score <= 0.0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        contenders = np.flatnonzero(~selected & (best_score - scores <= tol))
        if contenders.size == 0:
            raise CoverageError("Tau coverage cannot be satisfied by remaining exchanges")
        best_index = int(contenders[np.argmin(tie_rank[contenders])])
        selected[best_index] = True
        demand = np.maximum(demand - array[:, best_index], 0.0)
    return selected


def signed_tau_cover(
    A: np.ndarray,
    tau: float,
    exchange_keys: Sequence[str] | None = None,
    tol: float = 1e-12,
) -> Dict[str, np.ndarray]:
    array = np.asarray(A, dtype=float)
    if array.ndim != 2:
        raise ValueError("A must be 2D")
    if not np.isfinite(array).all():
        raise ValueError("A must be finite")
    A_pos = np.maximum(array, 0.0)
    A_neg = np.maximum(-array, 0.0)
    selected_pos = greedy_tau_cover(A_pos, tau, exchange_keys=exchange_keys, tol=tol)
    selected_neg = greedy_tau_cover(A_neg, tau, exchange_keys=exchange_keys, tol=tol)
    selected = selected_pos | selected_neg
    full_pos = A_pos.sum(axis=1)
    full_neg = A_neg.sum(axis=1)
    retained_pos = A_pos[:, selected].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    retained_neg = A_neg[:, selected].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    active_pos = full_pos > tol
    active_neg = full_neg > tol
    if active_pos.any() and np.any(retained_pos[active_pos] < tau * full_pos[active_pos] - tol):
        raise CoverageError("Positive tau coverage verification failed")
    if active_neg.any() and np.any(retained_neg[active_neg] < tau * full_neg[active_neg] - tol):
        raise CoverageError("Negative tau coverage verification failed")
    coverage_pos = np.ones_like(full_pos)
    coverage_neg = np.ones_like(full_neg)
    coverage_pos[active_pos] = retained_pos[active_pos] / full_pos[active_pos]
    coverage_neg[active_neg] = retained_neg[active_neg] / full_neg[active_neg]
    return {
        "selected": selected,
        "selected_pos": selected_pos,
        "selected_neg": selected_neg,
        "full_pos_by_category": full_pos,
        "full_neg_by_category": full_neg,
        "retained_pos_by_category": retained_pos,
        "retained_neg_by_category": retained_neg,
        "coverage_pos_by_category": coverage_pos,
        "coverage_neg_by_category": coverage_neg,
        "active_pos_by_category": active_pos,
        "active_neg_by_category": active_neg,
    }


def _protected_reason(
    exchange: Dict[str, Any],
    characterised: bool,
    uncharacterised_policy: str,
) -> str | None:
    if has_provider(exchange):
        return "provider_link"
    if is_quantitative_reference(exchange):
        return "quantitative_reference"
    if not characterised and uncharacterised_policy == "keep":
        return "uncharacterised_keep"
    return None


def _signed_coverage_stats(
    matrix: np.ndarray,
    selected_mask: np.ndarray,
    tau: float,
    tol: float,
) -> dict[str, Any]:
    array = np.asarray(matrix, dtype=float)
    selected = np.asarray(selected_mask, dtype=bool)
    A_pos = np.maximum(array, 0.0)
    A_neg = np.maximum(-array, 0.0)
    full_pos = A_pos.sum(axis=1)
    full_neg = A_neg.sum(axis=1)
    retained_pos = A_pos[:, selected].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    retained_neg = A_neg[:, selected].sum(axis=1) if array.shape[1] else np.zeros(array.shape[0], dtype=float)
    active_pos = full_pos > tol
    active_neg = full_neg > tol
    coverage_pos = np.ones_like(full_pos)
    coverage_neg = np.ones_like(full_neg)
    coverage_pos[active_pos] = retained_pos[active_pos] / full_pos[active_pos]
    coverage_neg[active_neg] = retained_neg[active_neg] / full_neg[active_neg]
    positive_cover_ok = not active_pos.any() or bool(np.all(retained_pos[active_pos] >= tau * full_pos[active_pos] - tol))
    negative_cover_ok = not active_neg.any() or bool(np.all(retained_neg[active_neg] >= tau * full_neg[active_neg] - tol))
    min_positive_coverage = float(coverage_pos[active_pos].min()) if active_pos.any() else 1.0
    min_negative_coverage = float(coverage_neg[active_neg].min()) if active_neg.any() else 1.0
    return {
        "full_pos": full_pos,
        "full_neg": full_neg,
        "retained_pos": retained_pos,
        "retained_neg": retained_neg,
        "active_pos": active_pos,
        "active_neg": active_neg,
        "positive_cover_ok": positive_cover_ok,
        "negative_cover_ok": negative_cover_ok,
        "min_positive_coverage": min_positive_coverage,
        "min_negative_coverage": min_negative_coverage,
        "n_active_positive_categories": int(active_pos.sum()),
        "n_active_negative_categories": int(active_neg.sum()),
    }


def reduce_process(
    process_data: Dict[str, Any],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    tau: float,
    uncharacterised_policy: str,
    strict_units: bool,
    tol: float,
    database_name: str,
    unit_registry: Optional[Dict[str, UnitInfo]] = None,
    *,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
) -> ProcessReductionResult:
    unit_registry = unit_registry or {}
    exchanges = list(process_data.get("exchanges") or [])
    process_id = str(process_data.get("@id") or process_data.get("id") or "")
    process_name = str(process_data.get("name") or "")

    matrix, candidate_indices, exchange_keys, characterised_flags = build_contribution_matrix(
        exchanges=exchanges,
        flow_lookup=flow_lookup,
        categories=categories,
        unit_registry=unit_registry,
        strict_units=strict_units,
        tol=tol,
        process_data=process_data,
        warning_records=warning_records,
        ambiguity_records=ambiguity_records,
        diagnostic_file=diagnostic_file,
        resolution_manager=resolution_manager,
    )
    cover = signed_tau_cover(matrix, tau=tau, exchange_keys=exchange_keys, tol=tol)

    selected_mask = cover["selected"].copy()
    selected_pos = cover["selected_pos"]
    selected_neg = cover["selected_neg"]

    rows: List[Dict[str, Any]] = []
    kept_indices: List[int] = []
    removed_indices: List[int] = []
    protected_count = 0

    candidate_lookup = {candidate_index: local_index for local_index, candidate_index in enumerate(candidate_indices)}
    new_exchanges: List[Dict[str, Any]] = []

    for exchange_index, exchange in enumerate(exchanges):
        flow_id = exchange_flow_id(exchange)
        if not flow_id or flow_id not in flow_lookup:
            raise MissingFlowError(f"Cannot resolve flow for exchange at index {exchange_index}")
        flow = flow_lookup[flow_id]
        if not flow.is_elementary:
            new_exchanges.append(exchange)
            continue

        local_index = candidate_lookup[exchange_index]
        characterised = characterised_flags[local_index]
        if not characterised and uncharacterised_policy == "fail":
            raise UncharacterisedExchangeError(
                f"Uncharacterised elementary exchange found in process {process_name} ({process_id})"
            )

        protected_reason = _protected_reason(exchange, characterised, uncharacterised_policy)
        protected = protected_reason is not None
        if protected:
            protected_count += 1
            selected_mask[local_index] = True

        selected = bool(selected_mask[local_index])
        removed = not selected
        if removed and not characterised and uncharacterised_policy == "drop":
            removal_reason = "uncharacterised_drop"
        elif removed:
            removal_reason = "tau_cover"
        else:
            removal_reason = protected_reason or "retained"

        if selected:
            new_exchanges.append(exchange)
            kept_indices.append(exchange_index)
        else:
            removed_indices.append(exchange_index)

        row = elementary_manifest_base(
            process_id=process_id,
            process_name=process_name,
            exchange=exchange,
            exchange_index=exchange_index,
            flow=flow,
        )
        row.update(
            {
                "database_name": database_name,
                "characterised": characterised,
                "selected": selected,
                "removed": removed,
                "protected": protected,
                "selected_by_positive_cover": bool(selected_pos[local_index]),
                "selected_by_negative_cover": bool(selected_neg[local_index]),
                "uncharacterised": not characterised,
                "removal_reason": removal_reason,
                "tau": tau,
                "method_selection": "",
            }
        )
        rows.append(row)

    reduced_process = dict(process_data)
    reduced_process["exchanges"] = new_exchanges
    n_total_before = len(exchanges)
    n_total_after = len(new_exchanges)
    n_elementary_before = len(candidate_indices)
    n_elementary_after = len(kept_indices)
    n_elementary_uncharacterised = int(sum(not flag for flag in characterised_flags))
    n_elementary_characterised = int(sum(1 for flag in characterised_flags if flag))
    n_uncharacterised_kept = sum(1 for row in rows if row["uncharacterised"] and row["selected"])
    n_uncharacterised_removed = sum(1 for row in rows if row["uncharacterised"] and row["removed"])
    coverage_stats = _signed_coverage_stats(matrix, selected_mask, tau=tau, tol=tol)
    if not coverage_stats["positive_cover_ok"]:
        raise CoverageError("Final positive tau coverage verification failed")
    if not coverage_stats["negative_cover_ok"]:
        raise CoverageError("Final negative tau coverage verification failed")
    coverage_failure = False

    process_row = {
        "process_id": process_id,
        "process_name": process_name,
        "n_total_exchanges_before": n_total_before,
        "n_total_exchanges_after": n_total_after,
        "n_elementary_before": n_elementary_before,
        "n_elementary_after": n_elementary_after,
        "n_elementary_characterised": n_elementary_characterised,
        "n_elementary_uncharacterised": n_elementary_uncharacterised,
        "n_uncharacterised_kept": n_uncharacterised_kept,
        "n_uncharacterised_removed": n_uncharacterised_removed,
        "n_elementary_removed": len(removed_indices),
        "n_protected_exchanges": protected_count,
        "positive_cover_ok": coverage_stats["positive_cover_ok"],
        "negative_cover_ok": coverage_stats["negative_cover_ok"],
        "min_positive_coverage": coverage_stats["min_positive_coverage"],
        "min_negative_coverage": coverage_stats["min_negative_coverage"],
        "n_active_positive_categories": coverage_stats["n_active_positive_categories"],
        "n_active_negative_categories": coverage_stats["n_active_negative_categories"],
        "coverage_failure": coverage_failure,
        "status": "modified" if removed_indices else "unchanged",
        "warnings": "",
    }
    return ProcessReductionResult(
        process_id=process_id,
        process_name=process_name,
        process_path="",
        original_process=process_data,
        reduced_process=reduced_process,
        selected_mask=selected_mask,
        selected_pos_mask=selected_pos,
        selected_neg_mask=selected_neg,
        candidate_indices=candidate_indices,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        elementary_rows=rows,
        process_row=process_row,
        full_pos=coverage_stats["full_pos"],
        full_neg=coverage_stats["full_neg"],
        retained_pos=coverage_stats["retained_pos"],
        retained_neg=coverage_stats["retained_neg"],
        active_pos=coverage_stats["active_pos"],
        active_neg=coverage_stats["active_neg"],
        n_uncharacterised_kept=n_uncharacterised_kept,
        n_uncharacterised_removed=n_uncharacterised_removed,
        warnings=[],
    )
