"""Reduction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .cf_resolution import CFResolutionManager
from .contribution import (
    build_contribution_details,
    build_sparse_contribution_details,
    elementary_manifest_base,
    exchange_flow_id,
    has_provider,
    is_quantitative_reference,
)
from .errors import CoverageError, MissingFlowError, UncharacterisedExchangeError
from .models import CFAmbiguityRecord, FlowInfo, ImpactCategory, ProcessReductionResult, UnitInfo, WarningRecord
from .sparse_cover import build_sparse_cover_matrix, retained_by_row, signed_tau_cover_sparse


@dataclass(frozen=True)
class PreparedSparseReductionContext:
    exchanges: List[Dict[str, Any]]
    sparse_details: Any
    candidate_indices: List[int]
    exchange_keys: List[str]
    characterised_flags: List[bool]
    positive_model: Any
    negative_model: Any
    protected_mask: np.ndarray


def _coverage_ratios(
    retained: np.ndarray,
    full: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
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
    while _coverage_gap_exceeds_tolerance(demand, full, active, tol):
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
    if not _coverage_requirement_satisfied(retained_pos, full_pos, active_pos, tau, tol):
        raise CoverageError("Positive tau coverage verification failed")
    if not _coverage_requirement_satisfied(retained_neg, full_neg, active_neg, tau, tol):
        raise CoverageError("Negative tau coverage verification failed")
    coverage_pos = _coverage_ratios(retained_pos, full_pos, active_pos)
    coverage_neg = _coverage_ratios(retained_neg, full_neg, active_neg)
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
    coverage_pos = _coverage_ratios(retained_pos, full_pos, active_pos)
    coverage_neg = _coverage_ratios(retained_neg, full_neg, active_neg)
    positive_cover_ok = _coverage_requirement_satisfied(retained_pos, full_pos, active_pos, tau, tol)
    negative_cover_ok = _coverage_requirement_satisfied(retained_neg, full_neg, active_neg, tau, tol)
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


def _signed_coverage_stats_sparse(
    positive_model: Any,
    negative_model: Any,
    selected_mask: np.ndarray,
    tau: float,
    tol: float,
) -> dict[str, Any]:
    full_pos = positive_model.full
    full_neg = negative_model.full
    retained_pos = retained_by_row(positive_model, selected_mask)
    retained_neg = retained_by_row(negative_model, selected_mask)
    active_pos = positive_model.active
    active_neg = negative_model.active
    coverage_pos = _coverage_ratios(retained_pos, full_pos, active_pos)
    coverage_neg = _coverage_ratios(retained_neg, full_neg, active_neg)
    positive_cover_ok = _coverage_requirement_satisfied(retained_pos, full_pos, active_pos, tau, tol)
    negative_cover_ok = _coverage_requirement_satisfied(retained_neg, full_neg, active_neg, tau, tol)
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


def prepare_sparse_reduction_context(
    process_data: Dict[str, Any],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    *,
    uncharacterised_policy: str,
    strict_units: bool,
    tol: float,
    allow_water_mass_volume_override: bool = False,
    unit_registry: Optional[Dict[str, UnitInfo]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 1000,
    include_row_metadata: bool = False,
) -> PreparedSparseReductionContext:
    unit_registry = unit_registry or {}
    exchanges = list(process_data.get("exchanges") or [])
    sparse_details = build_sparse_contribution_details(
        exchanges=exchanges,
        flow_lookup=flow_lookup,
        categories=categories,
        unit_registry=unit_registry,
        strict_units=strict_units,
        tol=tol,
        allow_water_mass_volume_override=allow_water_mass_volume_override,
        process_data=process_data,
        warning_records=warning_records,
        ambiguity_records=ambiguity_records,
        diagnostic_file=diagnostic_file,
        resolution_manager=resolution_manager,
        max_scenario_rows_per_process=max_scenario_rows_per_process,
        max_candidate_set_size=max_candidate_set_size,
        include_resolved_mask=False,
        include_row_metadata=include_row_metadata,
    )
    candidate_indices = list(sparse_details.candidate_indices)
    exchange_keys = list(sparse_details.exchange_keys)
    characterised_flags = list(sparse_details.characterised_flags)
    positive_model = build_sparse_cover_matrix(
        sparse_details.rowsets,
        len(candidate_indices),
        positive=True,
        tol=tol,
    )
    negative_model = build_sparse_cover_matrix(
        sparse_details.rowsets,
        len(candidate_indices),
        positive=False,
        tol=tol,
    )
    protected_mask = np.zeros(len(candidate_indices), dtype=bool)
    candidate_lookup = {candidate_index: local_index for local_index, candidate_index in enumerate(candidate_indices)}
    for exchange_index, exchange in enumerate(exchanges):
        flow_id = exchange_flow_id(exchange)
        if not flow_id or flow_id not in flow_lookup:
            raise MissingFlowError(f"Cannot resolve flow for exchange at index {exchange_index}")
        flow = flow_lookup[flow_id]
        if not flow.is_elementary:
            continue
        local_index = candidate_lookup[exchange_index]
        characterised = characterised_flags[local_index]
        protected_mask[local_index] = _protected_reason(exchange, characterised, uncharacterised_policy) is not None
    return PreparedSparseReductionContext(
        exchanges=exchanges,
        sparse_details=sparse_details,
        candidate_indices=candidate_indices,
        exchange_keys=exchange_keys,
        characterised_flags=characterised_flags,
        positive_model=positive_model,
        negative_model=negative_model,
        protected_mask=protected_mask,
    )


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
    allow_water_mass_volume_override: bool = False,
    *,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 1000,
    return_payload: str = "full",
) -> ProcessReductionResult:
    if return_payload not in {"full", "minimal"}:
        raise ValueError("return_payload must be 'full' or 'minimal'")
    process_id = str(process_data.get("@id") or process_data.get("id") or "")
    process_name = str(process_data.get("name") or "")
    context = prepare_sparse_reduction_context(
        process_data,
        flow_lookup,
        categories,
        uncharacterised_policy=uncharacterised_policy,
        strict_units=strict_units,
        tol=tol,
        allow_water_mass_volume_override=allow_water_mass_volume_override,
        unit_registry=unit_registry,
        warning_records=warning_records,
        ambiguity_records=ambiguity_records,
        diagnostic_file=diagnostic_file,
        resolution_manager=resolution_manager,
        max_scenario_rows_per_process=max_scenario_rows_per_process,
        max_candidate_set_size=max_candidate_set_size,
        include_row_metadata=False,
    )
    exchanges = context.exchanges
    sparse_details = context.sparse_details
    candidate_indices = context.candidate_indices
    exchange_keys = context.exchange_keys
    characterised_flags = context.characterised_flags
    cf_ambiguity_stats = sparse_details.cf_ambiguity_stats
    positive_model = context.positive_model
    negative_model = context.negative_model
    cover = signed_tau_cover_sparse(
        positive_model,
        negative_model,
        tau=tau,
        exchange_keys=exchange_keys,
        tol=tol,
    )

    selected_mask = cover["selected"].copy()
    selected_pos = cover["selected_pos"]
    selected_neg = cover["selected_neg"]

    rows: Optional[List[Dict[str, Any]]] = [] if return_payload == "full" else None
    kept_indices: Optional[List[int]] = [] if return_payload == "full" else None
    removed_indices: List[int] = []
    protected_count = 0
    n_elementary_after = 0
    n_uncharacterised_kept = 0
    n_uncharacterised_removed = 0

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

        protected = bool(context.protected_mask[local_index])
        protected_reason = _protected_reason(exchange, characterised, uncharacterised_policy) if protected else None
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
            n_elementary_after += 1
            if kept_indices is not None:
                kept_indices.append(exchange_index)
            if not characterised:
                n_uncharacterised_kept += 1
        else:
            removed_indices.append(exchange_index)
            if not characterised:
                n_uncharacterised_removed += 1

        if rows is not None:
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
    n_elementary_uncharacterised = int(sum(not flag for flag in characterised_flags))
    n_elementary_characterised = int(sum(1 for flag in characterised_flags if flag))
    n_elementary_removed = n_elementary_before - n_elementary_after
    coverage_stats = _signed_coverage_stats_sparse(
        positive_model,
        negative_model,
        selected_mask,
        tau=tau,
        tol=tol,
    )
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
        "n_elementary_removed": n_elementary_removed,
        "n_protected_exchanges": protected_count,
        "positive_cover_ok": coverage_stats["positive_cover_ok"],
        "negative_cover_ok": coverage_stats["negative_cover_ok"],
        "min_positive_coverage": coverage_stats["min_positive_coverage"],
        "min_negative_coverage": coverage_stats["min_negative_coverage"],
        "n_active_positive_categories": coverage_stats["n_active_positive_categories"],
        "n_active_negative_categories": coverage_stats["n_active_negative_categories"],
        "n_active_positive_rows": coverage_stats["n_active_positive_categories"],
        "n_active_negative_rows": coverage_stats["n_active_negative_categories"],
        "coverage_failure": coverage_failure,
        "status": "modified" if n_elementary_removed else "unchanged",
        "warnings": "",
    }
    return ProcessReductionResult(
        process_id=process_id,
        process_name=process_name,
        process_path="",
        original_process=process_data if return_payload == "full" else None,
        reduced_process=reduced_process,
        selected_mask=selected_mask if return_payload == "full" else None,
        selected_pos_mask=selected_pos if return_payload == "full" else None,
        selected_neg_mask=selected_neg if return_payload == "full" else None,
        candidate_indices=candidate_indices if return_payload == "full" else [],
        kept_indices=kept_indices or [],
        removed_indices=removed_indices,
        elementary_rows=rows or [],
        process_row=process_row,
        full_pos=coverage_stats["full_pos"] if return_payload == "full" else None,
        full_neg=coverage_stats["full_neg"] if return_payload == "full" else None,
        retained_pos=coverage_stats["retained_pos"] if return_payload == "full" else None,
        retained_neg=coverage_stats["retained_neg"] if return_payload == "full" else None,
        active_pos=coverage_stats["active_pos"] if return_payload == "full" else None,
        active_neg=coverage_stats["active_neg"] if return_payload == "full" else None,
        n_uncharacterised_kept=n_uncharacterised_kept,
        n_uncharacterised_removed=n_uncharacterised_removed,
        cf_ambiguity_stats=cf_ambiguity_stats,
        warnings=[],
    )
