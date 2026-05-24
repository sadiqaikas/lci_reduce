"""LCIA selection and characterisation factor resolution."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from .cf_resolution import (
    CFAmbiguityContext,
    CFResolutionManager,
    _normalised_candidate_identity,
    ambiguity_key as cf_ambiguity_key,
    candidate_metadata,
)
from .errors import (
    AmbiguousCharacterisationFactorError,
    DataFormatError,
    DuplicateMethodConflictError,
    RunCancelledError,
    UnitCompatibilityError,
)
from .models import (
    CFAmbiguityRecord,
    CharacterisationFactorCandidate,
    CharacterizationFactor,
    FlowInfo,
    ImpactCategory,
    ImpactCategoryReport,
    JsonLdArchive,
    ResolvedCharacterisationFactor,
    UnitCompatibilityResult,
    UnitInfo,
    WarningRecord,
)
from .schema_detect import category_path_text, extract_name, reference_id, reference_name


DECISION_RELEVANT_FIELDS = (
    "cf_value",
    "cf_unit",
    "cf_unit_id",
    "cf_flow_property_id",
    "cf_location_id",
    "cf_region",
    "cf_compartment",
    "cf_subcompartment",
)


def normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def normalise_unit(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = " ".join(value.lower().split())
    return token or None


def _normalise_optional(value: Optional[str]) -> str:
    if value is None:
        return ""
    return normalise_text(value)


def _normalise_float(value: float) -> str:
    return format(float(value), ".15g")


def _normalise_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def parse_method_selection(selection: str) -> Tuple[str, str]:
    if selection == "all":
        return "all", ""
    if ":" not in selection:
        raise ValueError("method-selection must be all, family:<text>, method:<text>, or category:<text>")
    mode, query = selection.split(":", 1)
    mode = mode.strip().lower()
    query = query.strip()
    if mode not in {"family", "method", "category"} or not query:
        raise ValueError("invalid method-selection")
    return mode, query


def category_search_text(category: ImpactCategory) -> str:
    parts = [
        category.name,
        category.method_name or "",
        category.method_path or "",
        category.path,
        category.metadata_text,
        category.category_id,
        category.method_id or "",
    ]
    return normalise_text(" ".join(parts))


def _extract_unit_metadata(data: dict) -> Tuple[Optional[str], Optional[str]]:
    for key in ("unit", "referenceUnit"):
        value = data.get(key)
        unit_id = reference_id(value)
        unit_name = extract_name(value)
        if unit_id or unit_name:
            return unit_id, unit_name or None
    unit_name = data.get("unitName") or data.get("referenceUnitName") or data.get("refUnitName") or data.get("refUnit")
    if isinstance(unit_name, str) and unit_name.strip():
        return None, unit_name.strip()
    return None, None


def _extract_flow_property_metadata(data: dict) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        data.get("flowProperty"),
        data.get("flowPropertyRef"),
        data.get("referenceFlowProperty"),
    ]
    flow_property_factor = data.get("flowPropertyFactor")
    if isinstance(flow_property_factor, dict):
        candidates.append(flow_property_factor.get("flowProperty"))
    for candidate in candidates:
        property_id = reference_id(candidate)
        property_name = extract_name(candidate)
        if property_id or property_name:
            return property_id, property_name or None
    return None, None


def _extract_location_metadata(data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    location = data.get("location")
    location_id = reference_id(location)
    location_name = extract_name(location)
    region = None
    if isinstance(location, dict):
        for key in ("code", "region", "regionCode", "name"):
            value = location.get(key)
            if isinstance(value, str) and value.strip():
                region = value.strip()
                break
    for key in ("region", "regionCode", "locationName"):
        if region:
            break
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            region = value.strip()
    return location_id, location_name or None, region


def _extract_compartments(data: dict) -> Tuple[Optional[str], Optional[str]]:
    path = ""
    if isinstance(data, dict):
        path = data.get("categoryPath", "") or category_path_text(data)
        flow_ref = data.get("flow")
        if not path and isinstance(flow_ref, dict):
            path = flow_ref.get("categoryPath", "") or category_path_text(flow_ref)
    parts = [part.strip() for part in path.split("/") if part.strip()]
    if parts:
        return parts[0], "/".join(parts[1:]) if len(parts) > 1 else None
    compartment = data.get("compartment")
    subcompartment = data.get("subCompartment") or data.get("subcompartment")
    return (
        compartment.strip() if isinstance(compartment, str) and compartment.strip() else None,
        subcompartment.strip() if isinstance(subcompartment, str) and subcompartment.strip() else None,
    )


def _method_path(data: dict) -> str:
    return str(data.get("categoryPath") or category_path_text(data) or "")


def _method_metadata(method_id: Optional[str], method_lookup: Dict[str, dict]) -> dict:
    method_data = method_lookup.get(method_id or "")
    return {
        "method_id": method_id or "",
        "method_name": str(method_data.get("name") or "") if method_data else "",
        "method_path": _method_path(method_data) if method_data else "",
        "method_source_file": str(method_data.get("__source_file") or "") if method_data else "",
    }


def _extract_method_category_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, list):
        for item in value:
            refs.update(_extract_method_category_refs(item))
        return refs
    if isinstance(value, dict):
        direct_ref = reference_id(value)
        if direct_ref:
            refs.add(direct_ref)
        for key in ("impactCategory", "impactCategories"):
            nested = value.get(key)
            if nested is not None:
                refs.update(_extract_method_category_refs(nested))
    return refs


def build_method_category_lookup(method_lookup: Dict[str, dict]) -> Dict[str, List[dict]]:
    relevant_keys = {
        "impactcategories",
        "impactcategoryrefs",
        "impactcategory",
    }
    category_lookup: Dict[str, List[dict]] = {}
    for method_id, method_data in method_lookup.items():
        method_meta = _method_metadata(method_id, method_lookup)
        for key, value in method_data.items():
            if _normalise_key(key) not in relevant_keys:
                continue
            for category_id in _extract_method_category_refs(value):
                category_lookup.setdefault(category_id, []).append(method_meta)
    return category_lookup


def resolve_category_method_info(
    *,
    category_data: dict,
    method_lookup: Dict[str, dict],
    method_category_lookup: Dict[str, List[dict]],
) -> dict:
    method_ref = category_data.get("impactMethod") or category_data.get("method")
    direct_method_id = reference_id(method_ref)
    direct_method_name = extract_name(method_ref) or ""
    if direct_method_id and direct_method_id in method_lookup:
        metadata = _method_metadata(direct_method_id, method_lookup)
        if direct_method_name and not metadata["method_name"]:
            metadata["method_name"] = direct_method_name
        return metadata

    category_id = str(category_data.get("@id") or category_data.get("id") or category_data.get("uuid") or "")
    parent_methods = method_category_lookup.get(category_id, [])
    if direct_method_id:
        matching = [item for item in parent_methods if item.get("method_id") == direct_method_id]
        if len(matching) == 1:
            return matching[0]
    if direct_method_name:
        token = _normalise_optional(direct_method_name)
        matching = [item for item in parent_methods if _normalise_optional(item.get("method_name")) == token]
        if len(matching) == 1:
            return matching[0]
    if len(parent_methods) == 1:
        return parent_methods[0]
    return {
        "method_id": direct_method_id or "",
        "method_name": direct_method_name,
        "method_path": "",
        "method_source_file": "",
    }


def build_cf_candidate_index(
    category_data: dict,
    method_lookup: Dict[str, dict],
    path: str,
    *,
    resolved_method_id: Optional[str] = None,
    resolved_method_name: Optional[str] = None,
) -> Dict[str, List[CharacterisationFactorCandidate]]:
    method_ref = category_data.get("impactMethod") or category_data.get("method")
    method_id = resolved_method_id if resolved_method_id is not None else reference_id(method_ref)
    method_name = (
        resolved_method_name
        if resolved_method_name is not None
        else (
            method_lookup.get(method_id or "", {}).get("name", "")
            if method_id and method_id in method_lookup
            else (method_ref.get("name", "") if isinstance(method_ref, dict) else "")
        )
    )
    category_id = category_data.get("@id") or category_data.get("id") or category_data.get("uuid")
    if not category_id:
        raise DataFormatError(f"Impact category at {path} is missing an id")
    category_name = str(category_data.get("name") or "")
    factors_data = None
    for key in ("impactFactors", "factors", "characterizationFactors", "characterisationFactors"):
        value = category_data.get(key)
        if isinstance(value, list):
            factors_data = value
            break
    if not factors_data:
        return {}
    factor_candidates: Dict[str, List[CharacterisationFactorCandidate]] = {}
    for factor in factors_data:
        if not isinstance(factor, dict):
            continue
        flow_ref = factor.get("flow") or factor.get("flowRef")
        flow_id = reference_id(flow_ref)
        if not flow_id:
            continue
        value = factor.get("value")
        if value is None:
            value = factor.get("amount")
        if value is None:
            continue
        unit_id, unit_name = _extract_unit_metadata(factor)
        if unit_name is None:
            category_unit_id, category_unit_name = _extract_unit_metadata(category_data)
            unit_id = unit_id or category_unit_id
            unit_name = unit_name or category_unit_name
        flow_property_id, flow_property_name = _extract_flow_property_metadata(factor)
        location_id, location_name, region = _extract_location_metadata(factor)
        compartment, subcompartment = _extract_compartments(factor)
        candidate = CharacterisationFactorCandidate(
            category_id=str(category_id),
            category_name=category_name,
            method_id=method_id,
            method_name=method_name or None,
            flow_id=flow_id,
            flow_name=reference_name(flow_ref) or str(factor.get("flowName") or ""),
            cf_value=float(value),
            cf_unit=unit_name,
            cf_unit_id=unit_id,
            cf_flow_property_id=flow_property_id,
            cf_flow_property_name=flow_property_name,
            cf_location_id=location_id,
            cf_location_name=location_name,
            cf_region=region,
            cf_compartment=compartment,
            cf_subcompartment=subcompartment,
            source_file=path,
            raw_factor_object=factor,
        )
        factor_candidates.setdefault(flow_id, []).append(candidate)
    return factor_candidates


def _candidate_decision_tuple(candidate: CharacterisationFactorCandidate) -> Tuple[str, ...]:
    return (
        _normalise_float(candidate.cf_value),
        _normalise_optional(candidate.cf_unit),
        _normalise_optional(candidate.cf_unit_id),
        _normalise_optional(candidate.cf_flow_property_id),
        _normalise_optional(candidate.cf_location_id),
        _normalise_optional(candidate.cf_region),
        _normalise_optional(candidate.cf_compartment),
        _normalise_optional(candidate.cf_subcompartment),
    )


def _candidate_signature(candidate: CharacterisationFactorCandidate) -> Tuple[str, ...]:
    return (
        candidate.category_id,
        candidate.method_id or "",
        candidate.flow_id,
        *_candidate_decision_tuple(candidate),
    )


def _differing_fields(candidates: Sequence[CharacterisationFactorCandidate]) -> List[str]:
    differing: List[str] = []
    if not candidates:
        return differing
    for field_name in DECISION_RELEVANT_FIELDS:
        values = {getattr(candidate, field_name) for candidate in candidates}
        if field_name == "cf_value":
            values = {_normalise_float(value) for value in values}  # type: ignore[arg-type]
        else:
            values = {_normalise_optional(value) for value in values}  # type: ignore[arg-type]
        if len(values) > 1:
            differing.append(field_name)
    return differing


def _format_cf_value(candidate: CharacterisationFactorCandidate) -> str:
    return _normalise_float(candidate.cf_value)


def _append_cf_records(
    candidates: Sequence[CharacterisationFactorCandidate],
    *,
    severity: str,
    issue_type: str,
    group_key: str,
    message: str,
    differing_fields: Sequence[str],
    ambiguity_records: Optional[List[CFAmbiguityRecord]],
    unit_compatibility: Optional[UnitCompatibilityResult] = None,
    flow: Optional[FlowInfo] = None,
    process_id: str = "",
    process_name: str = "",
    exchange_id: str = "",
    exchange_index: int | None = None,
    ambiguity_key: str = "",
    resolution_status: str = "",
    choice_origin: str = "",
    occurrence_timestamp: str = "",
    chosen_candidate: Optional[CharacterisationFactorCandidate] = None,
) -> None:
    if ambiguity_records is None:
        return
    candidate_count = len(candidates)
    differing = ",".join(differing_fields)
    chosen_identity = _normalised_candidate_identity(chosen_candidate) if chosen_candidate is not None else None
    all_candidate_values = json.dumps(
        [_format_cf_value(candidate) for candidate in candidates],
        ensure_ascii=True,
        sort_keys=True,
    )
    all_candidate_metadata = json.dumps(
        [
            {
                **candidate_metadata(candidate),
                "cf_value": _format_cf_value(candidate),
            }
            for candidate in candidates
        ],
        ensure_ascii=True,
        sort_keys=True,
    )
    rejected_values = json.dumps(
        [
            _format_cf_value(candidate)
            for candidate in candidates
            if chosen_identity is None or _normalised_candidate_identity(candidate) != chosen_identity
        ],
        ensure_ascii=True,
        sort_keys=True,
    )
    for index, candidate in enumerate(candidates):
        candidate_identity = _normalised_candidate_identity(candidate)
        candidate_selected = chosen_identity is not None and candidate_identity == chosen_identity
        ambiguity_records.append(
            CFAmbiguityRecord(
                severity=severity,
                method_id=candidate.method_id or "",
                method_name=candidate.method_name or "",
                category_id=candidate.category_id,
                category_name=candidate.category_name,
                flow_id=candidate.flow_id,
                flow_name=candidate.flow_name,
                candidate_count=candidate_count,
                candidate_index=index,
                cf_value=_format_cf_value(candidate),
                cf_unit=candidate.cf_unit or "",
                cf_unit_id=candidate.cf_unit_id or "",
                cf_flow_property_id=candidate.cf_flow_property_id or "",
                cf_flow_property_name=candidate.cf_flow_property_name or "",
                cf_compartment=candidate.cf_compartment or "",
                cf_subcompartment=candidate.cf_subcompartment or "",
                cf_location_id=candidate.cf_location_id or "",
                cf_location_name=candidate.cf_location_name or "",
                cf_region=candidate.cf_region or "",
                exchange_unit=(unit_compatibility.exchange_unit_name if unit_compatibility else "") or "",
                exchange_unit_id=(unit_compatibility.exchange_unit_id if unit_compatibility else "") or "",
                exchange_flow_property_id=(unit_compatibility.exchange_flow_property_id if unit_compatibility else "") or "",
                exchange_flow_property_name=(unit_compatibility.exchange_flow_property_name if unit_compatibility else "") or "",
                flow_reference_flow_property_id=(
                    unit_compatibility.flow_reference_flow_property_id
                    if unit_compatibility
                    else (flow.reference_flow_property_id if flow else "")
                )
                or "",
                flow_reference_flow_property_name=(
                    unit_compatibility.flow_reference_flow_property_name
                    if unit_compatibility
                    else (flow.reference_flow_property_name if flow else "")
                )
                or "",
                source_file=candidate.source_file,
                differing_fields=differing,
                message=message,
                issue_type=issue_type,
                group_key=group_key,
                process_id=process_id,
                process_name=process_name,
                exchange_id=exchange_id,
                exchange_index="" if exchange_index is None else str(exchange_index),
                ambiguity_key=ambiguity_key,
                resolution_status=resolution_status,
                choice_origin=choice_origin,
                occurrence_timestamp=occurrence_timestamp,
                all_candidate_cf_values=all_candidate_values,
                all_candidate_metadata=all_candidate_metadata,
                chosen_cf_value=_format_cf_value(chosen_candidate) if chosen_candidate is not None else "",
                rejected_cf_values=rejected_values,
                candidate_selected="true" if candidate_selected else "false",
            )
        )


def deduplicate_exact_cf_candidates(
    candidates: Sequence[CharacterisationFactorCandidate],
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
) -> List[CharacterisationFactorCandidate]:
    deduplicated: List[CharacterisationFactorCandidate] = []
    grouped: Dict[Tuple[str, ...], List[CharacterisationFactorCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_candidate_decision_tuple(candidate), []).append(candidate)
    for group_index, group in enumerate(grouped.values()):
        if len(group) > 1:
            message = (
                f"Exact duplicate characterisation factor candidates were deduplicated for category "
                f"{group[0].category_name} and flow {group[0].flow_name or group[0].flow_id}"
            )
            _append_cf_records(
                group,
                severity="warning",
                issue_type="duplicate_deduplicated",
                group_key=f"duplicate:{group[0].category_id}:{group[0].flow_id}:{group_index}",
                message=message,
                differing_fields=[],
                ambiguity_records=ambiguity_records,
            )
        deduplicated.append(group[0])
    return deduplicated


def _legacy_candidates_for_flow(category: ImpactCategory, flow_id: str) -> List[CharacterisationFactorCandidate]:
    factor = category.factors.get(flow_id)
    if factor is None:
        return []
    return [
        CharacterisationFactorCandidate(
            category_id=category.category_id,
            category_name=category.name,
            method_id=category.method_id,
            method_name=category.method_name,
            flow_id=flow_id,
            flow_name="",
            cf_value=float(factor.value),
            cf_unit=factor.unit_name,
            cf_unit_id=None,
            cf_flow_property_id=None,
            cf_flow_property_name=None,
            cf_location_id=None,
            cf_location_name=None,
            cf_region=None,
            cf_compartment=None,
            cf_subcompartment=None,
            source_file=category.source_file,
            raw_factor_object=factor.raw,
        )
    ]


def candidates_for_flow(category: ImpactCategory, flow_id: str) -> List[CharacterisationFactorCandidate]:
    if category.factor_candidates:
        return list(category.factor_candidates.get(flow_id, []))
    return _legacy_candidates_for_flow(category, flow_id)


def _candidate_location_key(candidate: CharacterisationFactorCandidate) -> Tuple[str, str, str]:
    return (
        _normalise_optional(candidate.cf_location_id),
        _normalise_optional(candidate.cf_location_name),
        _normalise_optional(candidate.cf_region),
    )


def _context_location_keys(location_id: Optional[str], location_name: Optional[str], region: Optional[str]) -> List[str]:
    keys = []
    if location_id:
        keys.append(_normalise_optional(location_id))
    if location_name:
        keys.append(_normalise_optional(location_name))
    if region:
        keys.append(_normalise_optional(region))
    return [key for key in keys if key]


def _category_is_regionalised(category: ImpactCategory) -> bool:
    text = category_search_text(category)
    return any(token in text for token in ("regional", "region", "country", "territor", "state", "province"))


def _flow_compartments(flow: FlowInfo) -> Tuple[Optional[str], Optional[str]]:
    parts = [part.strip() for part in flow.category_path.split("/") if part.strip()]
    if not parts:
        return None, None
    return parts[0], "/".join(parts[1:]) if len(parts) > 1 else None


def _extract_exchange_unit_id(exchange: dict) -> Optional[str]:
    for key in ("unit", "referenceUnit"):
        unit_id = reference_id(exchange.get(key))
        if unit_id:
            return unit_id
    return None


def _extract_exchange_unit_name(exchange: dict) -> Optional[str]:
    value = exchange.get("unitName")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for key in ("unit", "referenceUnit"):
        name = extract_name(exchange.get(key))
        if name:
            return name
    return None


def _exchange_id(exchange: dict) -> str:
    return str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or "")


def _extract_exchange_flow_property(exchange: dict) -> Tuple[Optional[str], Optional[str]]:
    flow_property = exchange.get("flowProperty")
    property_id = reference_id(flow_property)
    property_name = extract_name(flow_property)
    if property_id or property_name:
        return property_id, property_name or None
    flow_property_factor = exchange.get("flowPropertyFactor")
    if isinstance(flow_property_factor, dict):
        property_ref = flow_property_factor.get("flowProperty")
        property_id = reference_id(property_ref)
        property_name = extract_name(property_ref)
        if property_id or property_name:
            return property_id, property_name or None
    return None, None


def _compare_flow_property_identity(
    left_id: Optional[str],
    left_name: Optional[str],
    right_id: Optional[str],
    right_name: Optional[str],
) -> bool:
    if left_id and right_id:
        return left_id == right_id
    if left_name and right_name:
        return _normalise_optional(left_name) == _normalise_optional(right_name)
    return False


def _unit_conversion_factor(
    exchange_unit_id: Optional[str],
    cf_unit_id: Optional[str],
    unit_registry: Dict[str, UnitInfo],
) -> Optional[float]:
    if not exchange_unit_id or not cf_unit_id:
        return None
    exchange_unit_info = unit_registry.get(exchange_unit_id)
    cf_unit_info = unit_registry.get(cf_unit_id)
    if exchange_unit_info is None or cf_unit_info is None:
        return None
    if not exchange_unit_info.group_id or exchange_unit_info.group_id != cf_unit_info.group_id:
        return None
    if exchange_unit_info.conversion_factor is None or cf_unit_info.conversion_factor is None:
        return None
    return float(exchange_unit_info.conversion_factor) / float(cf_unit_info.conversion_factor)


def check_unit_compatibility(
    exchange_unit: Optional[dict],
    exchange_flow_property: Optional[dict],
    flow: FlowInfo,
    cf_candidate: CharacterisationFactorCandidate,
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool = True,
) -> UnitCompatibilityResult:
    del strict_units
    exchange_unit_id = reference_id(exchange_unit)
    exchange_unit_name = extract_name(exchange_unit)
    if not exchange_unit_name and exchange_unit and isinstance(exchange_unit.get("name"), str):
        exchange_unit_name = exchange_unit.get("name")

    exchange_flow_property_id = reference_id(exchange_flow_property)
    exchange_flow_property_name = extract_name(exchange_flow_property)
    flow_reference_flow_property_id = flow.reference_flow_property_id
    flow_reference_flow_property_name = flow.reference_flow_property_name

    exchange_unit_info = unit_registry.get(exchange_unit_id) if exchange_unit_id else None
    cf_unit_info = unit_registry.get(cf_candidate.cf_unit_id) if cf_candidate.cf_unit_id else None

    effective_exchange_flow_property_id = exchange_flow_property_id
    effective_exchange_flow_property_name = exchange_flow_property_name
    if effective_exchange_flow_property_id is None and effective_exchange_flow_property_name is None:
        effective_exchange_flow_property_id = flow_reference_flow_property_id
        effective_exchange_flow_property_name = flow_reference_flow_property_name
    if effective_exchange_flow_property_id is None and effective_exchange_flow_property_name is None and exchange_unit_info:
        effective_exchange_flow_property_id = exchange_unit_info.flow_property_id
        effective_exchange_flow_property_name = exchange_unit_info.flow_property_name

    effective_cf_flow_property_id = cf_candidate.cf_flow_property_id
    effective_cf_flow_property_name = cf_candidate.cf_flow_property_name
    if effective_cf_flow_property_id is None and effective_cf_flow_property_name is None and cf_unit_info:
        effective_cf_flow_property_id = cf_unit_info.flow_property_id
        effective_cf_flow_property_name = cf_unit_info.flow_property_name

    if effective_exchange_flow_property_id or effective_exchange_flow_property_name:
        if effective_cf_flow_property_id or effective_cf_flow_property_name:
            if not _compare_flow_property_identity(
                effective_exchange_flow_property_id,
                effective_exchange_flow_property_name,
                effective_cf_flow_property_id,
                effective_cf_flow_property_name,
            ):
                return UnitCompatibilityResult(
                    compatible=False,
                    conversion_factor=0.0,
                    reason="flow_property_mismatch",
                    exchange_unit_id=exchange_unit_id,
                    exchange_unit_name=exchange_unit_name or None,
                    exchange_flow_property_id=effective_exchange_flow_property_id,
                    exchange_flow_property_name=effective_exchange_flow_property_name,
                    flow_reference_flow_property_id=flow_reference_flow_property_id,
                    flow_reference_flow_property_name=flow_reference_flow_property_name,
                    cf_unit_id=cf_candidate.cf_unit_id,
                    cf_unit_name=cf_candidate.cf_unit,
                    cf_flow_property_id=effective_cf_flow_property_id,
                    cf_flow_property_name=effective_cf_flow_property_name,
                    flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
                    flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
                )

    if exchange_unit_id and cf_candidate.cf_unit_id and exchange_unit_id == cf_candidate.cf_unit_id:
        return UnitCompatibilityResult(
            compatible=True,
            conversion_factor=1.0,
            reason="direct_unit_id_match",
            exchange_unit_id=exchange_unit_id,
            exchange_unit_name=exchange_unit_name or None,
            exchange_flow_property_id=effective_exchange_flow_property_id,
            exchange_flow_property_name=effective_exchange_flow_property_name,
            flow_reference_flow_property_id=flow_reference_flow_property_id,
            flow_reference_flow_property_name=flow_reference_flow_property_name,
            cf_unit_id=cf_candidate.cf_unit_id,
            cf_unit_name=cf_candidate.cf_unit,
            cf_flow_property_id=effective_cf_flow_property_id,
            cf_flow_property_name=effective_cf_flow_property_name,
            flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
            flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
        )

    if exchange_unit_name and cf_candidate.cf_unit and normalise_unit(exchange_unit_name) == normalise_unit(cf_candidate.cf_unit):
        return UnitCompatibilityResult(
            compatible=True,
            conversion_factor=1.0,
            reason="direct_unit_name_match",
            exchange_unit_id=exchange_unit_id,
            exchange_unit_name=exchange_unit_name or None,
            exchange_flow_property_id=effective_exchange_flow_property_id,
            exchange_flow_property_name=effective_exchange_flow_property_name,
            flow_reference_flow_property_id=flow_reference_flow_property_id,
            flow_reference_flow_property_name=flow_reference_flow_property_name,
            cf_unit_id=cf_candidate.cf_unit_id,
            cf_unit_name=cf_candidate.cf_unit,
            cf_flow_property_id=effective_cf_flow_property_id,
            cf_flow_property_name=effective_cf_flow_property_name,
            flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
            flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
        )

    if exchange_unit_id is None and cf_candidate.cf_unit_id is None and exchange_unit_name is None and cf_candidate.cf_unit is None:
        return UnitCompatibilityResult(
            compatible=True,
            conversion_factor=1.0,
            reason="no_units_present",
            exchange_unit_id=exchange_unit_id,
            exchange_unit_name=exchange_unit_name or None,
            exchange_flow_property_id=effective_exchange_flow_property_id,
            exchange_flow_property_name=effective_exchange_flow_property_name,
            flow_reference_flow_property_id=flow_reference_flow_property_id,
            flow_reference_flow_property_name=flow_reference_flow_property_name,
            cf_unit_id=cf_candidate.cf_unit_id,
            cf_unit_name=cf_candidate.cf_unit,
            cf_flow_property_id=effective_cf_flow_property_id,
            cf_flow_property_name=effective_cf_flow_property_name,
            flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
            flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
        )

    conversion_factor = _unit_conversion_factor(exchange_unit_id, cf_candidate.cf_unit_id, unit_registry)
    if conversion_factor is not None:
        return UnitCompatibilityResult(
            compatible=True,
            conversion_factor=conversion_factor,
            reason="unit_group_conversion",
            exchange_unit_id=exchange_unit_id,
            exchange_unit_name=exchange_unit_name or None,
            exchange_flow_property_id=effective_exchange_flow_property_id,
            exchange_flow_property_name=effective_exchange_flow_property_name,
            flow_reference_flow_property_id=flow_reference_flow_property_id,
            flow_reference_flow_property_name=flow_reference_flow_property_name,
            cf_unit_id=cf_candidate.cf_unit_id,
            cf_unit_name=cf_candidate.cf_unit,
            cf_flow_property_id=effective_cf_flow_property_id,
            cf_flow_property_name=effective_cf_flow_property_name,
            flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
            flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
        )

    return UnitCompatibilityResult(
        compatible=False,
        conversion_factor=0.0,
        reason="missing_explicit_unit_conversion",
        exchange_unit_id=exchange_unit_id,
        exchange_unit_name=exchange_unit_name or None,
        exchange_flow_property_id=effective_exchange_flow_property_id,
        exchange_flow_property_name=effective_exchange_flow_property_name,
        flow_reference_flow_property_id=flow_reference_flow_property_id,
        flow_reference_flow_property_name=flow_reference_flow_property_name,
        cf_unit_id=cf_candidate.cf_unit_id,
        cf_unit_name=cf_candidate.cf_unit,
        cf_flow_property_id=effective_cf_flow_property_id,
        cf_flow_property_name=effective_cf_flow_property_name,
        flow_property_id=effective_cf_flow_property_id or effective_exchange_flow_property_id,
        flow_property_name=effective_cf_flow_property_name or effective_exchange_flow_property_name,
    )


def _warning_record(
    *,
    severity: str,
    category: ImpactCategory,
    flow: FlowInfo,
    process_id: str,
    process_name: str,
    message: str,
) -> WarningRecord:
    return WarningRecord(
        severity=severity,
        object_type="characterisation_factor",
        object_id=category.category_id,
        object_name=category.name,
        process_id=process_id,
        process_name=process_name,
        flow_id=flow.flow_id,
        flow_name=flow.name,
        category_id=category.category_id,
        category_name=category.name,
        message=message,
        method_id=category.method_id or "",
        method_name=category.method_name or "",
        source_file=category.source_file,
    )


def _error_message(
    *,
    category: ImpactCategory,
    flow: FlowInfo,
    process_id: str,
    process_name: str,
    candidates: Sequence[CharacterisationFactorCandidate],
    differing_fields: Sequence[str],
    diagnostic_file: str,
) -> str:
    return (
        f"category_id={category.category_id}; category_name={category.name}; "
        f"flow_id={flow.flow_id}; flow_name={flow.name}; "
        f"process_id={process_id}; process_name={process_name}; "
        f"candidate_count={len(candidates)}; differing_fields={','.join(differing_fields) or 'none'}; "
        f"diagnostic_file={diagnostic_file or 'cf_ambiguities.csv'}"
    )


def _unit_failure_detail(result: UnitCompatibilityResult) -> str:
    return (
        f"exchange_unit={result.exchange_unit_name or ''}; "
        f"exchange_unit_id={result.exchange_unit_id or ''}; "
        f"exchange_flow_property_id={result.exchange_flow_property_id or ''}; "
        f"flow_reference_flow_property_id={result.flow_reference_flow_property_id or ''}; "
        f"cf_unit={result.cf_unit_name or ''}; "
        f"cf_unit_id={result.cf_unit_id or ''}; "
        f"cf_flow_property_id={result.cf_flow_property_id or ''}; "
        f"reason={result.reason}"
    )


def _append_resolution_warning(
    *,
    category: ImpactCategory,
    flow: FlowInfo,
    process_id: str,
    process_name: str,
    warning_records: Optional[List[WarningRecord]],
    message: str,
) -> None:
    if warning_records is None:
        return
    warning_records.append(
        _warning_record(
            severity="info",
            category=category,
            flow=flow,
            process_id=process_id,
            process_name=process_name,
            message=message,
        )
    )


def _location_match_score(
    candidate: CharacterisationFactorCandidate,
    context_keys: Sequence[str],
) -> int:
    score = 0
    if candidate.cf_location_id and _normalise_optional(candidate.cf_location_id) in context_keys:
        score = max(score, 3)
    if candidate.cf_location_name and _normalise_optional(candidate.cf_location_name) in context_keys:
        score = max(score, 2)
    if candidate.cf_region and _normalise_optional(candidate.cf_region) in context_keys:
        score = max(score, 1)
    return score


def _prefer_compatible_candidate(
    candidates: Sequence[CharacterisationFactorCandidate],
    compatibility_results: Sequence[UnitCompatibilityResult],
) -> Tuple[Optional[CharacterisationFactorCandidate], Optional[UnitCompatibilityResult], str]:
    unit_reason_rank = {
        "direct_unit_id_match": 4,
        "direct_unit_name_match": 3,
        "unit_group_conversion": 2,
        "no_units_present": 1,
    }

    scored: List[Tuple[Tuple[int, int], CharacterisationFactorCandidate, UnitCompatibilityResult]] = []
    for candidate, result in zip(candidates, compatibility_results):
        explicit_flow_property_match = 0
        if _compare_flow_property_identity(
            result.exchange_flow_property_id,
            result.exchange_flow_property_name,
            result.cf_flow_property_id,
            result.cf_flow_property_name,
        ):
            explicit_flow_property_match = 1
        score = (
            explicit_flow_property_match,
            unit_reason_rank.get(result.reason, 0),
        )
        scored.append((score, candidate, result))

    if not scored:
        return None, None, ""

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    if best_score <= (0, 0):
        return None, None, ""
    winners = [item for item in scored if item[0] == best_score]
    if len(winners) != 1:
        return None, None, ""

    reason = "Automatically selected the only CF candidate with the strongest unit/flow-property match."
    return winners[0][1], winners[0][2], reason


def resolve_cf_for_exchange(
    category: ImpactCategory,
    exchange: dict,
    flow: FlowInfo,
    candidates: Sequence[CharacterisationFactorCandidate],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool = True,
    strict: bool = True,
    *,
    exchange_index: int = -1,
    process_data: Optional[dict] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
) -> Optional[ResolvedCharacterisationFactor]:
    if not candidates:
        return None

    process_id = str((process_data or {}).get("@id") or (process_data or {}).get("id") or "")
    process_name = str((process_data or {}).get("name") or "")
    exchange_id = _exchange_id(exchange)
    exchange_unit_payload = exchange.get("unit") or exchange.get("referenceUnit")
    if exchange_unit_payload is None and _extract_exchange_unit_name(exchange):
        exchange_unit_payload = {"name": _extract_exchange_unit_name(exchange)}
    exchange_flow_property_id, exchange_flow_property_name = _extract_exchange_flow_property(exchange)
    exchange_flow_property_payload = None
    if exchange_flow_property_id or exchange_flow_property_name:
        exchange_flow_property_payload = {
            "@id": exchange_flow_property_id,
            "name": exchange_flow_property_name,
        }
    ambiguity_seen = len(candidates) > 1
    deduplicated = deduplicate_exact_cf_candidates(candidates, ambiguity_records=ambiguity_records)
    differing_fields = _differing_fields(deduplicated)
    ambiguity_context: Optional[CFAmbiguityContext] = None
    ambiguity_key = ""
    if ambiguity_seen and resolution_manager is not None:
        ambiguity_context = CFAmbiguityContext(
            category_id=category.category_id,
            category_name=category.name,
            method_id=category.method_id or "",
            method_name=category.method_name or "",
            flow_id=flow.flow_id,
            flow_name=flow.name,
            process_id=process_id,
            process_name=process_name,
            exchange_id=exchange_id,
            exchange_index=exchange_index,
            diagnostic_file=diagnostic_file or "cf_ambiguities.csv",
            differing_fields=list(differing_fields),
        )
        ambiguity_key = cf_ambiguity_key(ambiguity_context)
        resolution_manager.note_found(ambiguity_key)

    remaining = list(deduplicated)
    resolution_source = ""
    resolution_reason = ""
    resolution_candidates: Sequence[CharacterisationFactorCandidate] = []
    decision = None
    flow_compartment, flow_subcompartment = _flow_compartments(flow)
    if ambiguity_seen and len(deduplicated) == 1:
        resolution_source = "automatic"
        resolution_reason = "Collapsed exact duplicate CF candidates with identical values and metadata."
        resolution_candidates = list(candidates)
    if any(field in differing_fields for field in ("cf_compartment", "cf_subcompartment")):
        if not flow_compartment and strict:
            message = (
                "Unable to disambiguate compartment-specific characterisation factors because the flow lacks "
                "reliable compartment metadata; "
                + _error_message(
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    candidates=remaining,
                    differing_fields=differing_fields,
                    diagnostic_file=diagnostic_file,
                )
            )
            _append_cf_records(
                remaining,
                severity="error",
                issue_type="ambiguity_failure",
                group_key=f"ambiguity:{category.category_id}:{flow.flow_id}:{process_id}:compartment",
                message=message,
                differing_fields=differing_fields,
                ambiguity_records=ambiguity_records,
                process_id=process_id,
                process_name=process_name,
                exchange_id=exchange_id,
                exchange_index=exchange_index,
                ambiguity_key=ambiguity_key,
            )
            if warning_records is not None:
                warning_records.append(
                    _warning_record(
                        severity="error",
                        category=category,
                        flow=flow,
                        process_id=process_id,
                        process_name=process_name,
                        message=message,
                    )
                )
            raise AmbiguousCharacterisationFactorError(message)
        prior_remaining = list(remaining)
        compartment_matches = []
        for candidate in remaining:
            if _normalise_optional(candidate.cf_compartment) != _normalise_optional(flow_compartment):
                continue
            if candidate.cf_subcompartment and _normalise_optional(candidate.cf_subcompartment) != _normalise_optional(flow_subcompartment):
                continue
            if candidate.cf_compartment:
                compartment_matches.append(candidate)
        if flow_subcompartment:
            exact_subcompartment_matches = [
                candidate
                for candidate in compartment_matches
                if candidate.cf_subcompartment
                and _normalise_optional(candidate.cf_subcompartment) == _normalise_optional(flow_subcompartment)
            ]
            if exact_subcompartment_matches:
                compartment_matches = exact_subcompartment_matches
        if len(compartment_matches) == 1:
            remaining = compartment_matches
            differing_fields = _differing_fields(remaining)
            if ambiguity_seen and len(prior_remaining) > 1:
                resolution_source = "automatic"
                resolution_reason = (
                    "Automatically selected the only CF candidate whose compartment/subcompartment matched the flow."
                )
                resolution_candidates = prior_remaining
        elif len(compartment_matches) > 1:
            remaining = compartment_matches
            differing_fields = _differing_fields(remaining)
        elif strict:
            message = (
                "No compartment-specific characterisation factor matched the referenced flow metadata; "
                + _error_message(
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    candidates=remaining,
                    differing_fields=differing_fields,
                    diagnostic_file=diagnostic_file,
                )
            )
            _append_cf_records(
                remaining,
                severity="error",
                issue_type="ambiguity_failure",
                group_key=f"ambiguity:{category.category_id}:{flow.flow_id}:{process_id}:compartment_miss",
                message=message,
                differing_fields=differing_fields,
                ambiguity_records=ambiguity_records,
                process_id=process_id,
                process_name=process_name,
                exchange_id=exchange_id,
                exchange_index=exchange_index,
                ambiguity_key=ambiguity_key,
            )
            if warning_records is not None:
                warning_records.append(
                    _warning_record(
                        severity="error",
                        category=category,
                        flow=flow,
                        process_id=process_id,
                        process_name=process_name,
                        message=message,
                    )
                )
            raise AmbiguousCharacterisationFactorError(message)

    compatibility_results = [
        check_unit_compatibility(
            exchange_unit_payload,
            exchange_flow_property_payload,
            flow,
            candidate,
            unit_registry,
            strict_units=strict_units,
        )
        for candidate in remaining
    ]
    compatibility_by_identity = {
        id(candidate): result for candidate, result in zip(remaining, compatibility_results)
    }
    compatible_candidates = [
        candidate
        for candidate, result in zip(remaining, compatibility_results)
        if result.compatible
    ]
    compatible_results = [
        result
        for result in compatibility_results
        if result.compatible
    ]
    selected_unit_result: Optional[UnitCompatibilityResult] = None
    prior_remaining = list(remaining)
    if len(compatible_candidates) == 1:
        remaining = compatible_candidates
        selected_unit_result = compatible_results[0]
        differing_fields = _differing_fields(remaining)
        if ambiguity_seen and len(prior_remaining) > 1:
            resolution_source = "automatic"
            resolution_reason = (
                "Automatically selected the only unit-compatible CF candidate after strict unit and flow-property checks."
            )
            resolution_candidates = prior_remaining
    elif not compatible_candidates:
        representative_result = compatibility_results[0]
        message = (
            "No directly compatible characterisation factor candidate remained after unit/flow-property checks; "
            + _error_message(
                category=category,
                flow=flow,
                process_id=process_id,
                process_name=process_name,
                candidates=remaining,
                differing_fields=["cf_unit", "cf_flow_property_id"],
                diagnostic_file=diagnostic_file,
            )
            + "; "
            + _unit_failure_detail(representative_result)
        )
        _append_cf_records(
            remaining,
            severity="error",
            issue_type="unit_conflict",
            group_key=f"unit:{category.category_id}:{flow.flow_id}:{process_id}",
            message=message,
            differing_fields=["cf_unit", "cf_flow_property_id"],
            ambiguity_records=ambiguity_records,
            unit_compatibility=representative_result,
            flow=flow,
            process_id=process_id,
            process_name=process_name,
            exchange_id=exchange_id,
            exchange_index=exchange_index,
            ambiguity_key=ambiguity_key,
        )
        if warning_records is not None:
            warning_records.append(
                _warning_record(
                    severity="error",
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    message=message,
                )
            )
        raise UnitCompatibilityError(message)
    else:
        remaining = compatible_candidates
        preferred_candidate, preferred_result, preferred_reason = _prefer_compatible_candidate(
            compatible_candidates,
            compatible_results,
        )
        if preferred_candidate is not None and preferred_result is not None:
            remaining = [preferred_candidate]
            selected_unit_result = preferred_result
            differing_fields = _differing_fields(remaining)
            if ambiguity_seen and len(prior_remaining) > 1:
                resolution_source = "automatic"
                resolution_reason = preferred_reason
                resolution_candidates = prior_remaining
        else:
            selected_unit_result = compatible_results[0] if len(compatible_results) == 1 else None
            differing_fields = _differing_fields(remaining)

    regionalised = any(
        _candidate_location_key(candidate) != _candidate_location_key(remaining[0])
        for candidate in remaining[1:]
    )
    if regionalised:
        prior_remaining = list(remaining)
        contexts = [
            _extract_location_metadata(exchange),
            (flow.location_id, flow.location_name, flow.location_region),
        ]
        if process_data is not None and _category_is_regionalised(category):
            contexts.append(_extract_location_metadata(process_data))
        for context_id, context_name, context_region in contexts:
            context_keys = _context_location_keys(context_id, context_name, context_region)
            if not context_keys:
                continue
            scored_matches = [
                (candidate, _location_match_score(candidate, context_keys))
                for candidate in remaining
            ]
            scored_matches = [
                (candidate, score)
                for candidate, score in scored_matches
                if score > 0
            ]
            if not scored_matches:
                continue
            best_score = max(score for _, score in scored_matches)
            current_matches = [
                candidate
                for candidate, score in scored_matches
                if score == best_score
            ]
            if len(current_matches) == 1:
                remaining = current_matches
                differing_fields = _differing_fields(remaining)
                if ambiguity_seen and len(prior_remaining) > 1:
                    resolution_source = "automatic"
                    resolution_reason = (
                        "Automatically selected the only CF candidate with the strongest location match."
                    )
                    resolution_candidates = prior_remaining
                break
            if len(current_matches) > 1:
                remaining = current_matches
                differing_fields = _differing_fields(remaining)
        if len(remaining) > 1 and strict:
            message = (
                "Regionalised characterisation factor candidates remained ambiguous after exact location checks; "
                + _error_message(
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    candidates=remaining,
                    differing_fields=differing_fields or ["cf_location_id", "cf_region"],
                    diagnostic_file=diagnostic_file,
                )
            )
            _append_cf_records(
                remaining,
                severity="error",
                issue_type="regional_conflict",
                group_key=f"regional:{category.category_id}:{flow.flow_id}:{process_id}",
                message=message,
                differing_fields=differing_fields or ["cf_location_id", "cf_region"],
                ambiguity_records=ambiguity_records,
                process_id=process_id,
                process_name=process_name,
                exchange_id=exchange_id,
                exchange_index=exchange_index,
                ambiguity_key=ambiguity_key,
            )
            if warning_records is not None:
                warning_records.append(
                    _warning_record(
                        severity="error",
                        category=category,
                        flow=flow,
                        process_id=process_id,
                        process_name=process_name,
                        message=message,
                    )
                )
            raise AmbiguousCharacterisationFactorError(message)

    if len(remaining) > 1 and ambiguity_context is not None and resolution_manager is not None:
        ambiguity_context.differing_fields = list(differing_fields)
        decision = resolution_manager.resolve_ambiguity(ambiguity_context, remaining)
        if decision.status in {"user_choice", "reused_choice"} and decision.candidate is not None:
            resolution_source = decision.status
            resolution_reason = decision.reason
            resolution_candidates = list(remaining)
            remaining = [decision.candidate]
            selected_unit_result = compatibility_by_identity.get(id(decision.candidate), selected_unit_result)
            differing_fields = _differing_fields(remaining)
        elif decision.status == "cancel_run":
            message = (
                "Run cancelled during CF ambiguity resolution; "
                + decision.reason
                + "; "
                + _error_message(
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    candidates=remaining,
                    differing_fields=differing_fields,
                    diagnostic_file=diagnostic_file,
                )
            )
            raise RunCancelledError(message)

    if len(remaining) != 1:
        message = (
            "Multiple conflicting characterisation factor candidates remained after disambiguation; "
            + _error_message(
                category=category,
                flow=flow,
                process_id=process_id,
                process_name=process_name,
                candidates=remaining,
                differing_fields=differing_fields,
                diagnostic_file=diagnostic_file,
            )
        )
        _append_cf_records(
            remaining,
            severity="error",
            issue_type="ambiguity_failure",
            group_key=f"ambiguity:{category.category_id}:{flow.flow_id}:{process_id}:final",
            message=message,
            differing_fields=differing_fields,
            ambiguity_records=ambiguity_records,
            process_id=process_id,
            process_name=process_name,
            exchange_id=exchange_id,
            exchange_index=exchange_index,
            ambiguity_key=ambiguity_key,
        )
        if warning_records is not None:
            warning_records.append(
                _warning_record(
                    severity="error",
                    category=category,
                    flow=flow,
                    process_id=process_id,
                    process_name=process_name,
                    message=message,
                )
            )
        raise AmbiguousCharacterisationFactorError(message)

    if selected_unit_result is None:
        selected_unit_result = compatibility_by_identity.get(id(remaining[0]))

    if ambiguity_seen and resolution_source:
        if resolution_source == "automatic" and resolution_manager is not None:
            resolution_manager.record_automatic_resolution()
        chosen_candidate = remaining[0]
        occurrence_timestamp = (
            decision.audit_record.timestamp
            if decision is not None and getattr(decision, "audit_record", None) is not None
            else ""
        )
        choice_origin = (
            decision.audit_record.choice_origin
            if decision is not None and getattr(decision, "audit_record", None) is not None
            else ("automatic" if resolution_source == "automatic" else "")
        )
        message = (
            f"CF ambiguity resolved {resolution_source}: {resolution_reason}; "
            f"chosen_cf_value={_format_cf_value(chosen_candidate)}; "
            f"category_id={category.category_id}; flow_id={flow.flow_id}; "
            f"process_id={process_id}; exchange_id={exchange_id}; exchange_index={exchange_index}; "
            f"source_file={chosen_candidate.source_file}; "
            f"candidate_count={len(resolution_candidates) or len(candidates)}"
        )
        _append_cf_records(
            resolution_candidates or candidates,
            severity="info",
            issue_type="ambiguity_resolution",
            group_key=f"resolution:{category.category_id}:{flow.flow_id}:{process_id}:{exchange_id}:{exchange_index}",
            message=message,
            differing_fields=differing_fields,
            ambiguity_records=ambiguity_records,
            process_id=process_id,
            process_name=process_name,
            exchange_id=exchange_id,
            exchange_index=exchange_index,
            ambiguity_key=ambiguity_key,
            resolution_status=resolution_source,
            choice_origin=choice_origin,
            occurrence_timestamp=occurrence_timestamp,
            chosen_candidate=chosen_candidate,
        )
        _append_resolution_warning(
            category=category,
            flow=flow,
            process_id=process_id,
            process_name=process_name,
            warning_records=warning_records,
            message=message,
        )

    return ResolvedCharacterisationFactor(
        candidate=remaining[0],
        candidate_count=len(deduplicated),
        differing_fields=differing_fields,
        conversion_factor=selected_unit_result.conversion_factor if selected_unit_result else 1.0,
        unit_compatibility=selected_unit_result,
    )


def _legacy_factor_map(
    candidate_index: Dict[str, List[CharacterisationFactorCandidate]],
) -> Dict[str, CharacterizationFactor]:
    factor_map: Dict[str, CharacterizationFactor] = {}
    for flow_id, candidates in candidate_index.items():
        deduplicated = deduplicate_exact_cf_candidates(candidates)
        if len(deduplicated) != 1:
            continue
        candidate = deduplicated[0]
        factor_map[flow_id] = CharacterizationFactor(
            flow_id=flow_id,
            value=candidate.cf_value,
            unit_name=candidate.cf_unit,
            raw=candidate.raw_factor_object,
        )
    return factor_map


def build_impact_category(
    raw: dict,
    method_lookup: Dict[str, dict],
    method_category_lookup: Dict[str, List[dict]],
    path: str,
) -> ImpactCategory:
    category_id = raw.get("@id") or raw.get("id") or raw.get("uuid")
    if not category_id:
        raise DataFormatError(f"Impact category at {path} is missing an id")
    method_info = resolve_category_method_info(
        category_data=raw,
        method_lookup=method_lookup,
        method_category_lookup=method_category_lookup,
    )
    metadata_parts = [
        raw.get("name", ""),
        raw.get("description", ""),
        raw.get("version", ""),
        raw.get("categoryPath", ""),
        method_info.get("method_name", ""),
        method_info.get("method_path", ""),
    ]
    factor_candidates = build_cf_candidate_index(
        raw,
        method_lookup,
        path,
        resolved_method_id=method_info.get("method_id") or None,
        resolved_method_name=method_info.get("method_name") or None,
    )
    return ImpactCategory(
        category_id=category_id,
        name=raw.get("name", ""),
        method_id=method_info.get("method_id") or None,
        method_name=method_info.get("method_name") or None,
        path=raw.get("categoryPath", "") or category_path_text(raw),
        metadata_text=" ".join(str(part) for part in metadata_parts if part),
        reference_unit=_extract_unit_metadata(raw)[1],
        factors=_legacy_factor_map(factor_candidates),
        raw=raw,
        factor_candidates=factor_candidates,
        source_file=path,
        method_path=method_info.get("method_path", ""),
        method_source_file=method_info.get("method_source_file", ""),
    )


def _category_dedup_signature(category: ImpactCategory) -> str:
    rows = []
    for flow_id in sorted(category.factor_candidates):
        signatures = sorted(_candidate_signature(candidate) for candidate in deduplicate_exact_cf_candidates(category.factor_candidates[flow_id]))
        rows.append((flow_id, signatures))
    return json.dumps(
        {
            "category_id": category.category_id,
            "method_id": category.method_id,
            "rows": rows,
        },
        sort_keys=True,
    )


def impact_category_report(category: ImpactCategory) -> ImpactCategoryReport:
    return ImpactCategoryReport(
        category_id=category.category_id,
        category_name=category.name,
        method_id=category.method_id or "",
        method_name=category.method_name or "",
        method_path=category.method_path or "",
        source_file=category.source_file,
    )


ArchiveT = TypeVar("ArchiveT")


def resolve_lcia_archives(
    database_archive: ArchiveT,
    methods_archive: Optional[ArchiveT],
) -> Tuple[List[ArchiveT], str, bool]:
    if methods_archive is not None:
        return [methods_archive], "external", bool(getattr(database_archive, "impact_categories", {}))
    return [database_archive], "database", False


def collect_categories(
    archives: Iterable[JsonLdArchive],
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
) -> List[ImpactCategory]:
    categories: Dict[str, ImpactCategory] = {}
    signatures: Dict[str, str] = {}
    for archive in archives:
        for category_id, category in archive.impact_categories.items():
            signature = _category_dedup_signature(category)
            if category_id not in categories:
                categories[category_id] = category
                signatures[category_id] = signature
                continue
            if category.method_id != categories[category_id].method_id:
                message = (
                    f"Conflicting duplicate LCIA category UUIDs detected for category_id={category_id}; "
                    f"diagnostic_file={diagnostic_file or 'cf_ambiguities.csv'}"
                )
                _append_cf_records(
                    [candidate for flow_candidates in category.factor_candidates.values() for candidate in flow_candidates],
                    severity="error",
                    issue_type="duplicate_method_conflict",
                    group_key=f"duplicate-category:{category_id}",
                    message=message,
                    differing_fields=["method_id"],
                    ambiguity_records=ambiguity_records,
                )
                raise DuplicateMethodConflictError(message)
            if signatures[category_id] != signature:
                message = (
                    f"Duplicate LCIA category UUID has conflicting factors for category_id={category_id}; "
                    f"diagnostic_file={diagnostic_file or 'cf_ambiguities.csv'}"
                )
                _append_cf_records(
                    [candidate for flow_candidates in category.factor_candidates.values() for candidate in flow_candidates],
                    severity="error",
                    issue_type="duplicate_method_conflict",
                    group_key=f"duplicate-category:{category_id}",
                    message=message,
                    differing_fields=["cf_value", "cf_unit", "cf_location_id", "cf_region"],
                    ambiguity_records=ambiguity_records,
                )
                raise DuplicateMethodConflictError(message)
            _append_cf_records(
                [candidate for flow_candidates in category.factor_candidates.values() for candidate in flow_candidates],
                severity="warning",
                issue_type="duplicate_method_deduplicated",
                group_key=f"duplicate-category:{category_id}",
                message=f"Duplicate LCIA category UUID was deduplicated safely for category_id={category_id}",
                differing_fields=[],
                ambiguity_records=ambiguity_records,
            )
    return list(categories.values())


def ensure_category_factor_quality(
    categories: Sequence[ImpactCategory],
    warnings: Optional[List[WarningRecord]] = None,
) -> List[ImpactCategory]:
    empty_categories: List[ImpactCategory] = []
    for category in categories:
        if category.factor_candidates or category.factors:
            continue
        empty_categories.append(category)
        if warnings is not None:
            warnings.append(
                WarningRecord(
                    severity="warning",
                    object_type="impact_category",
                    object_id=category.category_id,
                    object_name=category.name,
                    process_id="",
                    process_name="",
                    flow_id="",
                    flow_name="",
                    category_id=category.category_id,
                    category_name=category.name,
                    message=(
                        "Selected LCIA category contains no usable characterisation factors; "
                        f"method_id={category.method_id or ''}; "
                        f"method_name={category.method_name or ''}; "
                        f"source_file={category.source_file}"
                    ),
                    method_id=category.method_id or "",
                    method_name=category.method_name or "",
                    source_file=category.source_file,
                )
            )
    return empty_categories


def select_lcia_categories(
    categories: Sequence[ImpactCategory],
    selection: str,
) -> List[ImpactCategory]:
    mode, query = parse_method_selection(selection)
    if mode == "all":
        selected = list(categories)
    elif mode == "family":
        token = normalise_text(query)
        selected = [category for category in categories if token in category_search_text(category)]
    elif mode == "method":
        token = normalise_text(query)
        exact_ids = {category.method_id for category in categories if _normalise_optional(category.method_id) == token}
        if exact_ids:
            selected = [category for category in categories if category.method_id in exact_ids]
        else:
            matching_method_ids = {
                category.method_id
                for category in categories
                if category.method_id and token in normalise_text(category.method_name or "")
            }
            if len(matching_method_ids) > 1:
                raise DataFormatError(
                    f"Selection '{selection}' matched multiple methods; use method:<uuid> to disambiguate"
                )
            if not matching_method_ids:
                raise DataFormatError(f"No LCIA categories matched selection '{selection}'")
            selected = [category for category in categories if category.method_id in matching_method_ids]
    else:
        token = normalise_text(query)
        exact_ids = {category.category_id for category in categories if _normalise_optional(category.category_id) == token}
        if exact_ids:
            selected = [category for category in categories if category.category_id in exact_ids]
        else:
            matching_categories = [
                category for category in categories if token in normalise_text(category.name)
            ]
            matched_ids = {category.category_id for category in matching_categories}
            if len(matched_ids) > 1:
                raise DataFormatError(
                    f"Selection '{selection}' matched multiple categories; use category:<uuid> to disambiguate"
                )
            if not matching_categories:
                raise DataFormatError(f"No LCIA categories matched selection '{selection}'")
            selected = matching_categories
    return sorted(selected, key=lambda category: (category.method_name or "", category.name, category.category_id))


select_impact_categories = select_lcia_categories
