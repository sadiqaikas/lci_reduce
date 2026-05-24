"""Contribution matrix construction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cf_resolution import CFResolutionManager
from .errors import MissingFlowError
from .lcia import candidates_for_flow, resolve_cf_for_exchange
from .models import (
    CFAmbiguityRecord,
    FlowInfo,
    ImpactCategory,
    ResolvedCharacterisationFactor,
    UnitInfo,
    WarningRecord,
)
from .schema_detect import extract_name, reference_id


def exchange_unit_name(exchange: Dict[str, Any]) -> Optional[str]:
    candidates = [
        exchange.get("unitName"),
        (exchange.get("unit") or {}).get("name") if isinstance(exchange.get("unit"), dict) else None,
        (exchange.get("referenceUnit") or {}).get("name") if isinstance(exchange.get("referenceUnit"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def exchange_amount(exchange: Dict[str, Any]) -> float:
    amount = exchange.get("amount")
    if amount is None:
        raise ValueError("Exchange is missing amount")
    return float(amount)


def exchange_flow_id(exchange: Dict[str, Any]) -> Optional[str]:
    return reference_id(exchange.get("flow"))


def _exchange_location_key(exchange: Dict[str, Any]) -> Tuple[str, str, str]:
    location = exchange.get("location")
    if not isinstance(location, dict):
        return "", "", ""
    location_id = str(reference_id(location) or "")
    location_name = str(extract_name(location) or "")
    region = str(location.get("code") or location.get("region") or location_name or "")
    return location_id, location_name, region


def _process_location_key(process_data: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    if not process_data:
        return "", "", ""
    location = process_data.get("location")
    if not isinstance(location, dict):
        return "", "", ""
    location_id = str(reference_id(location) or "")
    location_name = str(extract_name(location) or "")
    region = str(location.get("code") or location.get("region") or location_name or "")
    return location_id, location_name, region


def _serialise_text(value: Optional[str]) -> str:
    return value or ""


def _resolution_cache_key(
    *,
    category: ImpactCategory,
    exchange: Dict[str, Any],
    exchange_index: int,
    flow: FlowInfo,
    process_data: Optional[Dict[str, Any]],
    strict_units: bool,
) -> Tuple[str, ...]:
    exchange_unit = exchange.get("unit") or exchange.get("referenceUnit") or {}
    if not isinstance(exchange_unit, dict):
        exchange_unit = {}
    exchange_flow_property = exchange.get("flowProperty")
    if not isinstance(exchange_flow_property, dict):
        exchange_flow_property = {}
    exchange_location_id, exchange_location_name, exchange_region = _exchange_location_key(exchange)
    process_location_id, process_location_name, process_region = _process_location_key(process_data)
    return (
        category.category_id,
        flow.flow_id,
        str(exchange_index),
        "1" if strict_units else "0",
        _serialise_text(str(reference_id(exchange_unit) or "")),
        _serialise_text(exchange_unit_name(exchange)),
        _serialise_text(str(reference_id(exchange_flow_property) or "")),
        _serialise_text(extract_name(exchange_flow_property) or ""),
        _serialise_text(exchange_location_id),
        _serialise_text(exchange_location_name),
        _serialise_text(exchange_region),
        _serialise_text(flow.category_path),
        _serialise_text(flow.location_id),
        _serialise_text(flow.location_name),
        _serialise_text(flow.location_region),
        _serialise_text(flow.reference_flow_property_id),
        _serialise_text(flow.reference_flow_property_name),
        _serialise_text(process_location_id),
        _serialise_text(process_location_name),
        _serialise_text(process_region),
    )


def flow_compartment(flow: FlowInfo) -> Tuple[str, str]:
    parts = [part for part in flow.category_path.split("/") if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "/".join(parts[1:])


def is_quantitative_reference(exchange: Dict[str, Any]) -> bool:
    return bool(
        exchange.get("quantitativeReference")
        or exchange.get("isQuantitativeReference")
        or exchange.get("referenceFlow")
    )


def has_provider(exchange: Dict[str, Any]) -> bool:
    return reference_id(exchange.get("provider")) is not None


def is_candidate_elementary_exchange(exchange: Dict[str, Any], flow: FlowInfo) -> bool:
    return flow.is_elementary


def build_contribution_matrix(
    exchanges: Sequence[Dict[str, Any]],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool,
    tol: float,
    *,
    process_data: Optional[Dict[str, Any]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
) -> Tuple[np.ndarray, List[int], List[str], List[bool]]:
    matrix, candidate_indices, exchange_keys, characterised_flags, _resolved_mask = build_contribution_details(
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
    return matrix, candidate_indices, exchange_keys, characterised_flags


def build_contribution_details(
    exchanges: Sequence[Dict[str, Any]],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool,
    tol: float,
    *,
    process_data: Optional[Dict[str, Any]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
) -> Tuple[np.ndarray, List[int], List[str], List[bool], np.ndarray]:
    del tol
    candidate_indices: List[int] = []
    exchange_keys: List[str] = []
    characterised_flags: List[bool] = []
    rows: List[List[float]] = [[] for _ in categories]
    resolved_rows: List[List[bool]] = [[] for _ in categories]
    resolution_cache: Dict[Tuple[str, ...], Optional[ResolvedCharacterisationFactor]] = {}
    candidate_cache = {
        category.category_id: {
            flow_id: candidates_for_flow(category, flow_id)
            for flow_id in {exchange_flow_id(exchange) for exchange in exchanges if exchange_flow_id(exchange)}
        }
        for category in categories
    }

    for index, exchange in enumerate(exchanges):
        flow_id = exchange_flow_id(exchange)
        if not flow_id or flow_id not in flow_lookup:
            raise MissingFlowError(f"Cannot resolve flow for exchange at index {index}")
        flow = flow_lookup[flow_id]
        if not is_candidate_elementary_exchange(exchange, flow):
            continue
        candidate_indices.append(index)
        exchange_key = (
            str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or f"{flow_id}:{index}")
        )
        exchange_keys.append(exchange_key)
        amount = exchange_amount(exchange)
        characterised = False
        for row_index, category in enumerate(categories):
            cache_key = _resolution_cache_key(
                category=category,
                exchange=exchange,
                exchange_index=index,
                flow=flow,
                process_data=process_data,
                strict_units=strict_units,
            )
            if cache_key in resolution_cache:
                resolved = resolution_cache[cache_key]
            else:
                resolved = resolve_cf_for_exchange(
                    category=category,
                    exchange=exchange,
                    exchange_index=index,
                    flow=flow,
                    candidates=candidate_cache.get(category.category_id, {}).get(flow_id, []),
                    unit_registry=unit_registry,
                    strict_units=strict_units,
                    strict=True,
                    process_data=process_data,
                    warning_records=warning_records,
                    ambiguity_records=ambiguity_records,
                    diagnostic_file=diagnostic_file,
                    resolution_manager=resolution_manager,
                )
                resolution_cache[cache_key] = resolved
            if resolved is None:
                rows[row_index].append(0.0)
                resolved_rows[row_index].append(False)
                continue
            characterised = True
            rows[row_index].append(amount * float(resolved.conversion_factor) * float(resolved.candidate.cf_value))
            resolved_rows[row_index].append(True)
        characterised_flags.append(characterised)

    if not categories:
        matrix = np.zeros((0, len(candidate_indices)), dtype=float)
        resolved_mask = np.zeros((0, len(candidate_indices)), dtype=bool)
    elif candidate_indices:
        matrix = np.array(rows, dtype=float)
        resolved_mask = np.array(resolved_rows, dtype=bool)
    else:
        matrix = np.zeros((len(categories), 0), dtype=float)
        resolved_mask = np.zeros((len(categories), 0), dtype=bool)
    if matrix.size and np.isnan(matrix).any():
        raise ValueError("Contribution matrix contains NaN")
    if matrix.size and np.isinf(matrix).any():
        raise ValueError("Contribution matrix contains Inf")
    return matrix, candidate_indices, exchange_keys, characterised_flags, resolved_mask


def elementary_manifest_base(
    process_id: str,
    process_name: str,
    exchange: Dict[str, Any],
    exchange_index: int,
    flow: FlowInfo,
) -> Dict[str, Any]:
    compartment, subcompartment = flow_compartment(flow)
    return {
        "process_id": process_id,
        "process_name": process_name,
        "exchange_id": str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or exchange_index),
        "exchange_index": exchange_index,
        "flow_id": flow.flow_id,
        "flow_name": flow.name,
        "flow_type": flow.flow_type,
        "compartment": compartment,
        "subcompartment": subcompartment,
        "amount": exchange_amount(exchange),
        "unit": exchange_unit_name(exchange) or "",
    }
