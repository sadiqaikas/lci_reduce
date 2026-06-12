"""Contribution matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cf_resolution import CFResolutionManager
from .errors import MissingFlowError, ScenarioExpansionError
from .lcia import candidates_for_flow, resolve_admissible_cf_set_for_exchange
from .models import (
    CFAmbiguityStats,
    CFAmbiguityRecord,
    CoverageRowMetadata,
    FlowInfo,
    ImpactCategory,
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


def _normalise_optional(value: Optional[str]) -> str:
    return value or ""


def _resolution_cache_key(
    *,
    category: ImpactCategory,
    exchange: Dict[str, Any],
    exchange_index: int,
    flow: FlowInfo,
    process_data: Optional[Dict[str, Any]],
    strict_units: bool,
    allow_water_mass_volume_override: bool,
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
        "1" if allow_water_mass_volume_override else "0",
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
    parts = [part.strip() for part in str(flow.category_path or "").split("/") if part.strip()]

    if not parts:
        return "", ""

    # openLCA-style elementary-flow paths often start with a root container.
    # This is not the environmental compartment.
    root_labels = {
        "Elementary flows",
        "Elementary flow",
        "Elementary Flows",
        "elementary flows",
        "elementary flow",
    }

    if parts[0] in root_labels:
        parts = parts[1:]

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


def _contribution_sign(value: float, tol: float) -> str:
    if value > tol:
        return "pos"
    if value < -tol:
        return "neg"
    return "zero"


def _candidate_location_key(location_id: Optional[str], location_name: Optional[str], region: Optional[str]) -> Tuple[str, str, str]:
    return (
        _normalise_optional(location_id),
        _normalise_optional(location_name),
        _normalise_optional(region),
    )


def _scenario_location_source(
    exchange: Dict[str, Any],
    process_data: Optional[Dict[str, Any]],
) -> Tuple[str, Tuple[str, str, str]]:
    exchange_key = _exchange_location_key(exchange)
    if any(exchange_key):
        return "exchange", exchange_key
    process_key = _process_location_key(process_data)
    if any(process_key):
        return "process", process_key
    return "", ("", "", "")


@dataclass(frozen=True)
class _PreparedOption:
    candidate: Any
    conversion_factor: float
    location_key: Tuple[str, str, str]
    contribution: float
    equivalent_count: int = 1


@dataclass(frozen=True)
class _ScenarioVariant:
    scenario_id: str
    scenario_label: str
    contributions: Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class _LocationAxisMember:
    local_index: int
    exchange: Dict[str, Any]
    exchange_index: int
    options: Tuple[_PreparedOption, ...]
    # Regional CF ambiguity is represented as real-location rows with local
    # within-location variants. Generic/unlocated options are fallback inside
    # those rows only; true non-location ambiguity stays in finite groups.
    location_options: Tuple[Tuple[Tuple[str, str, str], Tuple[_PreparedOption, ...]], ...]
    fallback_option: _PreparedOption | None

    @property
    def location_keys(self) -> Tuple[Tuple[str, str, str], ...]:
        return tuple(location_key for location_key, _option in self.location_options)

    def options_for_location(self, location_key: Tuple[str, str, str]) -> Tuple[_PreparedOption, ...]:
        for option_location_key, options in self.location_options:
            if option_location_key == location_key:
                return options
        return tuple()


@dataclass(frozen=True)
class _ScenarioGroup:
    group_id: str
    group_label: str
    scenario_type: str
    classification: str
    diagnostic_reason: str
    variants: Tuple[_ScenarioVariant, ...]
    exchange_indices: Tuple[int, ...]
    member_summaries: Tuple[str, ...]
    member_count: int
    location_count: int
    has_unlocated_fallback: bool
    location_preview: Tuple[str, ...]
    amount_min: float
    amount_max: float
    contribution_min: float
    contribution_max: float


@dataclass(frozen=True)
class _CategoryRowSet:
    exact_items: Tuple[Tuple[int, float], ...]
    scenario_rows: Tuple[Tuple[Tuple[int, float], ...], ...]
    metadata_rows: Tuple[CoverageRowMetadata, ...]


@dataclass(frozen=True)
class SparseContributionDetails:
    rowsets: Tuple[_CategoryRowSet, ...]
    candidate_indices: Tuple[int, ...]
    exchange_keys: Tuple[str, ...]
    characterised_flags: Tuple[bool, ...]
    resolved_mask: np.ndarray
    row_metadata: Tuple[CoverageRowMetadata, ...]
    cf_ambiguity_stats: CFAmbiguityStats


def _sparse_signature(
    sparse_items: Sequence[Tuple[int, float]],
    tol: float,
) -> Tuple[Tuple[int, str], ...]:
    return tuple(
        (int(index), format(float(value), ".15g"))
        for index, value in sorted(sparse_items)
        if abs(float(value)) > tol
    )


def _merge_sparse_tuples(
    left: Sequence[Tuple[int, float]],
    right: Sequence[Tuple[int, float]],
    tol: float,
) -> Tuple[Tuple[int, float], ...]:
    merged: Dict[int, float] = {}
    for local_index, value in left:
        _add_sparse_contribution(merged, int(local_index), float(value), tol)
    for local_index, value in right:
        _add_sparse_contribution(merged, int(local_index), float(value), tol)
    return tuple(sorted(merged.items()))


def _effect_location_key(
    location_key: Tuple[str, str, str],
    contribution: float,
    tol: float,
) -> Tuple[str, str, str]:
    if abs(float(contribution)) <= tol:
        return "", "", ""
    return location_key


def _prepared_option_effect_key(
    *,
    location_key: Tuple[str, str, str],
    contribution: float,
    tol: float,
) -> Tuple[str, str, str, str]:
    effect_location_key = _effect_location_key(location_key, contribution, tol)
    return (
        effect_location_key[0],
        effect_location_key[1],
        effect_location_key[2],
        format(float(contribution), ".15g"),
    )


def _location_display(location_key: Tuple[str, str, str]) -> str:
    for value in (location_key[1], location_key[0], location_key[2]):
        if value:
            return value
    return "unlocated"


def _unlocated_fallback_label() -> str:
    return "generic/unlocated"


def _location_preview(
    location_keys: Sequence[Tuple[str, str, str]],
    *,
    has_unlocated_fallback: bool,
    limit: int = 5,
) -> Tuple[str, ...]:
    labels = [_location_display(location_key) for location_key in location_keys[:limit]]
    if len(location_keys) > limit:
        labels.append("...")
    if has_unlocated_fallback:
        labels.append(_unlocated_fallback_label())
    return tuple(labels)


def _candidate_label(candidate: Any, conversion_factor: float, location_key: Tuple[str, str, str]) -> str:
    location_label = _location_display(location_key)
    cf_label = format(float(getattr(candidate, "cf_value", 0.0)), ".15g")
    conversion_label = format(float(conversion_factor), ".15g")
    return f"cf={cf_label}; conv={conversion_label}; location={location_label}"


def _prepared_option_label(option: _PreparedOption) -> str:
    label = _candidate_label(option.candidate, option.conversion_factor, option.location_key)
    if option.equivalent_count > 1:
        return f"{label}; equivalent_candidates={option.equivalent_count}"
    return label


def _case_location_tokens(location_key: Tuple[str, str, str]) -> Tuple[str, ...]:
    tokens = []
    for value in location_key:
        normalised = _normalise_optional(value)
        if normalised and normalised not in tokens:
            tokens.append(normalised)
    return tuple(tokens)


def _coverage_entry_tokens(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        normalised = _normalise_optional(value)
        return (normalised,) if normalised else tuple()
    if isinstance(value, dict):
        tokens = []
        for key in ("@id", "id", "code", "region", "name", "location", "locationId", "locationName"):
            item = value.get(key)
            if isinstance(item, str):
                normalised = _normalise_optional(item)
                if normalised and normalised not in tokens:
                    tokens.append(normalised)
        return tuple(tokens)
    return tuple()


def _authorised_parent_coverage_tokens(option: _PreparedOption) -> Tuple[str, ...]:
    raw = getattr(option.candidate, "raw_factor_object", {}) or {}
    coverage: List[str] = []
    for key in ("locationCoverage", "location_coverage", "coveredLocations", "covered_locations"):
        value = raw.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            for token in _coverage_entry_tokens(item):
                if token not in coverage:
                    coverage.append(token)
    return tuple(coverage)


def _resolve_options_for_case(
    member: _LocationAxisMember,
    scenario_location_key: Tuple[str, str, str],
) -> Tuple[Tuple[_PreparedOption, ...], str]:
    exact_options = member.options_for_location(scenario_location_key)
    if exact_options:
        return exact_options, "exact_match"

    case_tokens = set(_case_location_tokens(scenario_location_key))
    parent_fallbacks: List[Tuple[_PreparedOption, int]] = []
    for option in member.options:
        if not any(option.location_key):
            continue
        coverage_tokens = set(_authorised_parent_coverage_tokens(option))
        if not coverage_tokens or not case_tokens.intersection(coverage_tokens):
            continue
        parent_fallbacks.append((option, len(coverage_tokens)))
    if parent_fallbacks:
        best_specificity = min(size for _option, size in parent_fallbacks)
        return (
            tuple(option for option, size in parent_fallbacks if size == best_specificity),
            "authorised_parent_region_fallback",
        )

    if member.fallback_option is not None:
        return (member.fallback_option,), "generic_default_fallback"
    return tuple(), "no_applicable_cf"


def _prepare_admissible_options(
    *,
    exchange: Dict[str, Any],
    admissible: Any,
    tol: float,
) -> Tuple[_PreparedOption, ...]:
    prepared: Dict[Tuple[str, str, str, str], _PreparedOption] = {}
    for option in admissible.options:
        location_key = _candidate_location_key(
            option.candidate.cf_location_id,
            option.candidate.cf_location_name,
            option.candidate.cf_region,
        )
        contribution = float(option.conversion_factor) * float(option.candidate.cf_value) * exchange_amount(exchange)
        effect_key = _prepared_option_effect_key(
            location_key=location_key,
            contribution=contribution,
            tol=tol,
        )
        existing = prepared.get(effect_key)
        if existing is None:
            prepared[effect_key] = _PreparedOption(
                candidate=option.candidate,
                conversion_factor=float(option.conversion_factor),
                location_key=location_key,
                contribution=contribution,
                equivalent_count=1,
            )
            continue
        prepared[effect_key] = _PreparedOption(
            candidate=existing.candidate,
            conversion_factor=existing.conversion_factor,
            location_key=existing.location_key,
            contribution=existing.contribution,
            equivalent_count=existing.equivalent_count + 1,
        )
    return tuple(
        item
        for item in sorted(
            prepared.values(),
            key=lambda option: (
                _effect_location_key(option.location_key, option.contribution, tol),
                float(option.contribution),
                float(option.conversion_factor),
                _normalise_optional(getattr(option.candidate, "cf_value", None)),
            ),
        )
    )


def _scenario_group_summary(group: _ScenarioGroup) -> str:
    preview = ", ".join(group.location_preview) if group.location_preview else "-"
    member_preview = " || ".join(group.member_summaries[:3]) if group.member_summaries else "-"
    if len(group.member_summaries) > 3:
        member_preview += " || ..."
    return (
        f"{group.group_label}"
        f"(type={group.scenario_type}"
        f", classification={group.classification}"
        f", reason={group.diagnostic_reason}"
        f", exchange_indices={list(group.exchange_indices)}"
        f", members={group.member_count}"
        f", locations={group.location_count}"
        f", has_unlocated_fallback={'yes' if group.has_unlocated_fallback else 'no'}"
        f", location_preview=[{preview}]"
        f", member_preview=[{member_preview}]"
        f", variants={len(group.variants)}"
        f", amount_range=[{format(group.amount_min, '.6g')}, {format(group.amount_max, '.6g')}]"
        f", contribution_range=[{format(group.contribution_min, '.6g')}, {format(group.contribution_max, '.6g')}])"
    )


def _dense_vector_from_sparse(
    n_columns: int,
    sparse_items: Sequence[Tuple[int, float]],
) -> np.ndarray:
    vector = np.zeros(n_columns, dtype=float)
    for local_index, value in sparse_items:
        vector[local_index] += float(value)
    return vector


def _add_sparse_contribution(
    target: Dict[int, float],
    local_index: int,
    contribution: float,
    tol: float,
) -> None:
    if abs(float(contribution)) <= tol:
        return
    target[local_index] = target.get(local_index, 0.0) + float(contribution)


def _scenario_limit_error(
    *,
    process_id: str,
    process_name: str,
    category: ImpactCategory,
    exchange: Optional[Dict[str, Any]],
    exchange_index: Optional[int],
    message: str,
) -> ScenarioExpansionError:
    exchange_id = ""
    if exchange is not None:
        exchange_id = str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or "")
    return ScenarioExpansionError(
        f"{message}; "
        f"process_id={process_id}; process_name={process_name}; "
        f"category_id={category.category_id}; category_name={category.name}; "
        f"exchange_id={exchange_id}; "
        f"exchange_index={'' if exchange_index is None else exchange_index}"
    )


def _exchange_label(exchange: Dict[str, Any], exchange_index: int) -> str:
    return str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or exchange_index)


def _location_option_groups(
    options: Sequence[_PreparedOption],
) -> Tuple[Dict[Tuple[str, str, str], List[_PreparedOption]], List[_PreparedOption]]:
    by_location: Dict[Tuple[str, str, str], List[_PreparedOption]] = {}
    fallback: List[_PreparedOption] = []
    for option in options:
        if any(option.location_key):
            by_location.setdefault(option.location_key, []).append(option)
        else:
            fallback.append(option)
    return by_location, fallback


def _effective_option_count_summary(
    by_location: Dict[Tuple[str, str, str], List[_PreparedOption]],
    fallback: Sequence[_PreparedOption],
) -> str:
    parts = [
        f"{_location_display(location_key)}:{len(location_options)}"
        for location_key, location_options in sorted(by_location.items())
    ]
    if fallback:
        parts.append(f"{_unlocated_fallback_label()}:{len(fallback)}")
    return ", ".join(parts) if parts else "-"


def _location_axis_member_summary(
    exchange_index: int,
    by_location: Dict[Tuple[str, str, str], List[_PreparedOption]],
    fallback: Sequence[_PreparedOption],
) -> str:
    locations = _location_preview(
        sorted(by_location),
        has_unlocated_fallback=bool(fallback),
    )
    return (
        f"exchange_index={exchange_index}"
        f"; locations=[{', '.join(locations) if locations else '-'}]"
        f"; effective_options_per_location=[{_effective_option_count_summary(by_location, fallback)}]"
    )


def _classify_location_axis_member(
    *,
    local_index: int,
    exchange: Dict[str, Any],
    exchange_index: int,
    options: Sequence[_PreparedOption],
) -> Tuple[_LocationAxisMember | None, str]:
    by_location, fallback = _location_option_groups(options)
    if len(by_location) < 2:
        return None, (
            "finite_nonlocation:insufficient_nonempty_locations"
            f"; {_location_axis_member_summary(exchange_index, by_location, fallback)}"
        )
    if len(fallback) > 1:
        return None, (
            "finite_nonlocation:multiple_generic_fallback_options"
            f"; {_location_axis_member_summary(exchange_index, by_location, fallback)}"
        )
    return (
        _LocationAxisMember(
            local_index=local_index,
            exchange=exchange,
            exchange_index=exchange_index,
            options=tuple(options),
            location_options=tuple(
                (location_key, tuple(location_options))
                for location_key, location_options in sorted(by_location.items())
            ),
            fallback_option=fallback[0] if fallback else None,
        ),
        (
            "location_axis:accepted"
            f"; {_location_axis_member_summary(exchange_index, by_location, fallback)}"
        ),
    )


def _location_axis_group_label(
    scenario_locations: Sequence[Tuple[str, str, str]],
) -> str:
    preview = ",".join(_location_display(location_key) for location_key in scenario_locations[:3])
    if len(scenario_locations) > 3:
        preview += ",..."
    return f"location_axis:unified:{preview or 'unlocated'}"


def _compress_independent_options(
    options: Sequence[_PreparedOption],
) -> Tuple[_PreparedOption, ...]:
    if len(options) <= 2:
        return tuple(options)
    min_option = min(
        options,
        key=lambda option: (
            float(option.contribution),
            option.location_key,
            float(option.conversion_factor),
        ),
    )
    max_option = max(
        options,
        key=lambda option: (
            float(option.contribution),
            option.location_key,
            float(option.conversion_factor),
        ),
    )
    if min_option == max_option:
        return (min_option,)
    compressed = [min_option]
    if max_option != min_option:
        compressed.append(max_option)
    return tuple(
        sorted(
            compressed,
            key=lambda option: (
                _effect_location_key(option.location_key, option.contribution, 0.0),
                float(option.contribution),
                float(option.conversion_factor),
            ),
        )
    )


def _build_location_axis_groups(
    *,
    axis_members: Sequence[_LocationAxisMember],
    tol: float,
    category: ImpactCategory,
    process_id: str,
    process_name: str,
    max_scenario_rows_per_process: int,
) -> List[_ScenarioGroup]:
    if not axis_members:
        return []

    # Design note: per process-category, regional CF ambiguity is one sparse
    # spatial coverage table over the union of real CF locations. Each real
    # location is a coverage case; exchanges contribute to that case only when
    # they have a location-specific CF or one generic/unlocated fallback.
    # Exchanges that do not apply to a case are absent from that row rather than
    # forcing a fully specified joint scenario. Same-location multiplicity stays
    # local to that coverage case, and non-location ambiguity remains separate.
    scenario_locations = tuple(
        sorted({location_key for member in axis_members for location_key in member.location_keys})
    )
    group_label = _location_axis_group_label(scenario_locations)
    return [
        _build_regional_group(
            group_label=group_label,
            members=tuple(sorted(axis_members, key=lambda item: (item.exchange_index, item.local_index))),
            scenario_locations=scenario_locations,
            tol=tol,
            category=category,
            process_id=process_id,
            process_name=process_name,
            max_scenario_rows_per_process=max_scenario_rows_per_process,
        )
    ]


def _build_independent_group(
    *,
    exchange: Dict[str, Any],
    exchange_index: int,
    local_index: int,
    options: Sequence[_PreparedOption],
    tol: float,
    diagnostic_reason: str,
) -> _ScenarioGroup:
    compressed_options = _compress_independent_options(options)
    exchange_label = _exchange_label(exchange, exchange_index)
    group_label = f"exchange:{exchange_label}"
    specific_locations = sorted({option.location_key for option in options if any(option.location_key)})
    has_unlocated_fallback = any(not any(option.location_key) for option in options)
    variants: List[_ScenarioVariant] = []
    for option in compressed_options:
        sparse: Dict[int, float] = {}
        _add_sparse_contribution(sparse, local_index, option.contribution, tol)
        effect_key = _prepared_option_effect_key(
            location_key=option.location_key,
            contribution=option.contribution,
            tol=tol,
        )
        scenario_id = "effect:" + "|".join(effect_key)
        variants.append(
            _ScenarioVariant(
                scenario_id=f"{group_label}|{scenario_id}",
                scenario_label=f"{group_label} | {_prepared_option_label(option)}",
                contributions=tuple(sorted(sparse.items())),
            )
        )
    contributions = [option.contribution for option in options] or [0.0]
    amount = exchange_amount(exchange)
    return _ScenarioGroup(
        group_id=group_label,
        group_label=group_label,
        scenario_type="finite_cf",
        classification="finite_nonlocation",
        diagnostic_reason=diagnostic_reason,
        variants=tuple(variants),
        exchange_indices=(exchange_index,),
        member_summaries=(
            _location_axis_member_summary(
                exchange_index,
                * _location_option_groups(options),
            ),
        ),
        member_count=1,
        location_count=len(specific_locations),
        has_unlocated_fallback=has_unlocated_fallback,
        location_preview=_location_preview(
            specific_locations,
            has_unlocated_fallback=has_unlocated_fallback,
        ),
        amount_min=amount,
        amount_max=amount,
        contribution_min=min(contributions),
        contribution_max=max(contributions),
    )


def _build_regional_group(
    *,
    group_label: str,
    members: Sequence[_LocationAxisMember],
    scenario_locations: Sequence[Tuple[str, str, str]],
    tol: float,
    category: ImpactCategory,
    process_id: str,
    process_name: str,
    max_scenario_rows_per_process: int,
) -> _ScenarioGroup:
    specific_locations = tuple(sorted(scenario_locations))
    has_unlocated_fallback = any(member.fallback_option is not None for member in members)
    variants_by_signature: Dict[Tuple[Tuple[int, str], ...], _ScenarioVariant] = {}
    for scenario_location_key in specific_locations:
        location_label = _location_display(scenario_location_key)
        per_member_choices: List[List[Tuple[str, str, Tuple[Tuple[int, float], ...]]]] = []
        member_variant_counts: List[str] = []
        missing_members: List[str] = []
        for member in members:
            matching, resolution_kind = _resolve_options_for_case(member, scenario_location_key)
            if not matching:
                missing_members.append(f"exchange_index={member.exchange_index}:not_applicable")
                continue
            used_fallback = resolution_kind != "exact_match"
            member_variant_counts.append(
                f"exchange_index={member.exchange_index}:{len(matching)}"
                + (f"({resolution_kind})" if used_fallback else "")
            )
            choices_by_signature: Dict[Tuple[Tuple[int, str], ...], Tuple[str, str, Tuple[Tuple[int, float], ...]]] = {}
            for option in matching:
                sparse: Dict[int, float] = {}
                _add_sparse_contribution(sparse, member.local_index, option.contribution, tol)
                choice_signature = _sparse_signature(tuple(sorted(sparse.items())), tol)
                choice_id = "effect:" + "|".join(
                    _prepared_option_effect_key(
                        location_key=option.location_key,
                        contribution=option.contribution,
                        tol=tol,
                    )
                )
                exchange_label = _exchange_label(member.exchange, member.exchange_index)
                choices_by_signature.setdefault(
                    choice_signature,
                    (
                        choice_id,
                        (
                            f"{exchange_label}:{_prepared_option_label(option)}"
                            + (
                                "; fallback_for_shared_region=yes"
                                if resolution_kind == "generic_default_fallback"
                                else "; fallback_for_case=authorised_parent_region"
                                if resolution_kind == "authorised_parent_region_fallback"
                                else ""
                            )
                        ),
                        tuple(sorted(sparse.items())),
                    ),
                )
            choices = [choices_by_signature[key] for key in sorted(choices_by_signature)]
            per_member_choices.append(choices)

        combinations: Dict[
            Tuple[Tuple[int, str], ...],
            Tuple[List[str], List[str], Tuple[Tuple[int, float], ...]],
        ] = {tuple(): ([], [], tuple())}
        for member_choices in per_member_choices:
            next_combinations: Dict[
                Tuple[Tuple[int, str], ...],
                Tuple[List[str], List[str], Tuple[Tuple[int, float], ...]],
            ] = {}
            for current_ids, current_labels, current_sparse in combinations.values():
                for choice_id, choice_label, choice_sparse in member_choices:
                    merged_sparse = _merge_sparse_tuples(current_sparse, choice_sparse, tol)
                    merged_signature = _sparse_signature(merged_sparse, tol)
                    next_combinations.setdefault(
                        merged_signature,
                        (
                            [*current_ids, choice_id],
                            [*current_labels, choice_label] if choice_label else list(current_labels),
                            merged_sparse,
                        )
                    )
            combinations = next_combinations
            if len(combinations) > max_scenario_rows_per_process:
                raise _scenario_limit_error(
                    process_id=process_id,
                    process_name=process_name,
                    category=category,
                    exchange=None,
                    exchange_index=None,
                    message=(
                        "Regional CF coverage-row expansion exceeded max_scenario_rows_per_process"
                        " due to excessive within-case ambiguity"
                        f"; max_scenario_rows_per_process={max_scenario_rows_per_process}"
                        f"; current_group={group_label}"
                        f"; scenario_location={location_label}"
                        f"; projected_location_variants={len(combinations)}"
                        f"; member_variant_counts=[{', '.join(member_variant_counts)}]"
                    ),
                )

        if not per_member_choices:
            continue

        for choice_ids, choice_labels, merged_sparse in combinations.values():
            if not merged_sparse:
                continue
            scenario_id_prefix = f"location:{scenario_location_key[0]}|{scenario_location_key[1]}|{scenario_location_key[2]}"
            suffix = ""
            nonempty_labels = [label for label in choice_labels if label]
            if nonempty_labels:
                suffix = " | " + ", ".join(nonempty_labels)
            if missing_members:
                suffix += (" | " if suffix else " | ") + ", ".join(missing_members)
            variant = _ScenarioVariant(
                scenario_id=f"{group_label}|{scenario_id_prefix}|{'|'.join(choice_ids)}",
                scenario_label=f"{group_label}={location_label}{suffix}",
                contributions=merged_sparse,
            )
            variants_by_signature.setdefault(
                _sparse_signature(merged_sparse, tol),
                variant,
            )
        if len(variants_by_signature) > max_scenario_rows_per_process:
            raise _scenario_limit_error(
                process_id=process_id,
                process_name=process_name,
                category=category,
                exchange=None,
                exchange_index=None,
                message=(
                    "Regional CF coverage-row table exceeded max_scenario_rows_per_process"
                    f"; max_scenario_rows_per_process={max_scenario_rows_per_process}"
                    f"; current_group={group_label}"
                    f"; scenario_location={location_label}"
                    f"; projected_scenario_rows={len(variants_by_signature)}"
                    f"; member_variant_counts=[{', '.join(member_variant_counts)}]"
                ),
            )
    amounts = [exchange_amount(member.exchange) for member in members] or [0.0]
    contributions = [option.contribution for member in members for option in member.options] or [0.0]
    return _ScenarioGroup(
        group_id=group_label,
        group_label=group_label,
        scenario_type="regional_cf",
        classification="location_axis",
        diagnostic_reason="accepted:sparse_location_axis_coverage_rows",
        variants=tuple(variants_by_signature[key] for key in sorted(variants_by_signature)),
        exchange_indices=tuple(sorted(member.exchange_index for member in members)),
        member_summaries=tuple(
            _location_axis_member_summary(
                member.exchange_index,
                {location_key: list(options) for location_key, options in member.location_options},
                [member.fallback_option] if member.fallback_option is not None else [],
            )
            for member in sorted(members, key=lambda item: item.exchange_index)
        ),
        member_count=len(members),
        location_count=len(specific_locations),
        has_unlocated_fallback=has_unlocated_fallback,
        location_preview=_location_preview(
            specific_locations,
            has_unlocated_fallback=has_unlocated_fallback,
        ),
        amount_min=min(amounts),
        amount_max=max(amounts),
        contribution_min=min(contributions),
        contribution_max=max(contributions),
    )


def _build_category_rowset(
    *,
    category: ImpactCategory,
    admissible_by_exchange: Sequence[Tuple[int, Dict[str, Any], int, Sequence[_PreparedOption]]],
    tol: float,
    process_id: str,
    process_name: str,
    process_data: Optional[Dict[str, Any]],
    stats: CFAmbiguityStats,
    max_scenario_rows_per_process: int,
    include_row_metadata: bool = True,
) -> _CategoryRowSet:
    exact_sparse: Dict[int, float] = {}
    location_axis_members: List[_LocationAxisMember] = []
    independent_groups: List[_ScenarioGroup] = []
    has_ambiguity = False

    for local_index, exchange, exchange_index, options in admissible_by_exchange:
        if len(options) == 1:
            _add_sparse_contribution(exact_sparse, local_index, options[0].contribution, tol)
            continue

        has_ambiguity = True
        axis_member, classification_reason = _classify_location_axis_member(
            local_index=local_index,
            exchange=exchange,
            exchange_index=exchange_index,
            options=options,
        )
        if axis_member is not None:
            location_axis_members.append(axis_member)
            continue
        independent_groups.append(
            _build_independent_group(
                exchange=exchange,
                exchange_index=exchange_index,
                local_index=local_index,
                options=options,
                tol=tol,
                diagnostic_reason=classification_reason,
            )
        )

    groups: List[_ScenarioGroup] = []
    location_axis_groups = _build_location_axis_groups(
        axis_members=location_axis_members,
        tol=tol,
        category=category,
        process_id=process_id,
        process_name=process_name,
        max_scenario_rows_per_process=max_scenario_rows_per_process,
    )
    groups.extend(location_axis_groups)
    stats.n_regional_scenario_groups += len(location_axis_groups)
    groups.extend(independent_groups)
    stats.n_independent_candidate_groups += len(independent_groups)

    if not has_ambiguity:
        metadata_rows: Tuple[CoverageRowMetadata, ...]
        if include_row_metadata:
            metadata_rows = (
                CoverageRowMetadata(
                    row_id=f"{category.category_id}::exact",
                    category_id=category.category_id,
                    category_name=category.name,
                    scenario_id="exact",
                    scenario_label="exact",
                    scenario_type="exact",
                ),
            )
        else:
            metadata_rows = tuple()
        return _CategoryRowSet(
            exact_items=tuple(sorted(exact_sparse.items())),
            scenario_rows=(tuple(),),
            metadata_rows=metadata_rows,
        )

    groups = sorted(groups, key=lambda group: group.group_id)
    combinations: Dict[
        Tuple[Tuple[int, str], ...],
        Tuple[List[str], List[str], Tuple[Tuple[int, float], ...], bool],
    ] = {tuple(): ([], [], tuple(), False)}
    for group in groups:
        next_combinations: Dict[
            Tuple[Tuple[int, str], ...],
            Tuple[List[str], List[str], Tuple[Tuple[int, float], ...], bool],
        ] = {}
        combinations_before = len(combinations)
        for current_ids, current_labels, current_sparse, current_has_regional in combinations.values():
            for variant in group.variants:
                merged_sparse = _merge_sparse_tuples(current_sparse, variant.contributions, tol)
                merged_signature = _sparse_signature(merged_sparse, tol)
                next_combinations.setdefault(
                    merged_signature,
                    (
                        [*current_ids, variant.scenario_id],
                        [*current_labels, variant.scenario_label],
                        merged_sparse,
                        current_has_regional or group.scenario_type == "regional_cf",
                    )
                )
        combinations = next_combinations
        if stats.n_scenario_rows_added + len(combinations) > max_scenario_rows_per_process:
            largest_groups = " || ".join(
                _scenario_group_summary(item)
                for item in sorted(groups, key=lambda value: (-len(value.variants), value.group_id))[:5]
            )
            raise _scenario_limit_error(
                process_id=process_id,
                process_name=process_name,
                category=category,
                exchange=None,
                exchange_index=None,
                message=(
                    "Finite CF scenario-row expansion exceeded max_scenario_rows_per_process"
                    f"; max_scenario_rows_per_process={max_scenario_rows_per_process}"
                    f"; combinations_before_group={combinations_before}"
                    f"; current_group={group.group_label}"
                    f"; current_group_type={group.scenario_type}"
                    f"; current_group_variants={len(group.variants)}"
                    f"; projected_scenario_rows={stats.n_scenario_rows_added + len(combinations)}"
                    f"; current_group_summary={_scenario_group_summary(group)}"
                    f"; largest_groups={largest_groups}"
                ),
            )

    metadata_rows: List[CoverageRowMetadata] = []
    scenario_rows: List[Tuple[Tuple[int, float], ...]] = []
    stats.n_scenario_rows_added += len(combinations)
    for current_ids, current_labels, current_sparse, current_has_regional in combinations.values():
        scenario_id = "__".join(current_ids) if current_ids else "exact"
        scenario_label = " | ".join(current_labels) if current_labels else "exact"
        scenario_type = "regional_cf" if current_has_regional else "finite_cf"
        scenario_rows.append(current_sparse)
        if include_row_metadata:
            metadata_rows.append(
                CoverageRowMetadata(
                    row_id=f"{category.category_id}::{scenario_id}",
                    category_id=category.category_id,
                    category_name=category.name,
                    scenario_id=scenario_id,
                    scenario_label=scenario_label,
                    scenario_type=scenario_type,
                )
            )
    return _CategoryRowSet(
        exact_items=tuple(sorted(exact_sparse.items())),
        scenario_rows=tuple(scenario_rows),
        metadata_rows=tuple(metadata_rows),
    )


def build_contribution_matrix(
    exchanges: Sequence[Dict[str, Any]],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool,
    tol: float,
    allow_water_mass_volume_override: bool = False,
    *,
    process_data: Optional[Dict[str, Any]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 1000,
) -> Tuple[np.ndarray, List[int], List[str], List[bool]]:
    (
        matrix,
        candidate_indices,
        exchange_keys,
        characterised_flags,
        _resolved_mask,
        _row_metadata,
        _cf_ambiguity_stats,
    ) = build_contribution_details(
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
    )
    return matrix, candidate_indices, exchange_keys, characterised_flags


def build_contribution_details(
    exchanges: Sequence[Dict[str, Any]],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool,
    tol: float,
    allow_water_mass_volume_override: bool = False,
    *,
    process_data: Optional[Dict[str, Any]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 1000,
    include_resolved_mask: bool = True,
    include_row_metadata: bool = True,
) -> Tuple[np.ndarray, List[int], List[str], List[bool], np.ndarray, List[CoverageRowMetadata], CFAmbiguityStats]:
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
        include_resolved_mask=include_resolved_mask,
        include_row_metadata=include_row_metadata,
    )

    matrix_rows: List[np.ndarray] = []
    n_columns = len(sparse_details.candidate_indices)
    for rowset in sparse_details.rowsets:
        for scenario_sparse in rowset.scenario_rows:
            matrix_rows.append(
                _dense_vector_from_sparse(
                    n_columns,
                    _merge_sparse_tuples(rowset.exact_items, scenario_sparse, tol),
                )
            )
    if not matrix_rows:
        matrix = np.zeros((0, n_columns), dtype=float)
    else:
        matrix = np.vstack(matrix_rows)
    if matrix.size and np.isnan(matrix).any():
        raise ValueError("Contribution matrix contains NaN")
    if matrix.size and np.isinf(matrix).any():
        raise ValueError("Contribution matrix contains Inf")
    return (
        matrix,
        list(sparse_details.candidate_indices),
        list(sparse_details.exchange_keys),
        list(sparse_details.characterised_flags),
        sparse_details.resolved_mask,
        list(sparse_details.row_metadata),
        sparse_details.cf_ambiguity_stats,
    )


def build_sparse_contribution_details(
    exchanges: Sequence[Dict[str, Any]],
    flow_lookup: Dict[str, FlowInfo],
    categories: Sequence[ImpactCategory],
    unit_registry: Dict[str, UnitInfo],
    strict_units: bool,
    tol: float,
    allow_water_mass_volume_override: bool = False,
    *,
    process_data: Optional[Dict[str, Any]] = None,
    warning_records: Optional[List[WarningRecord]] = None,
    ambiguity_records: Optional[List[CFAmbiguityRecord]] = None,
    diagnostic_file: str = "",
    resolution_manager: Optional[CFResolutionManager] = None,
    max_scenario_rows_per_process: int = 300000,
    max_candidate_set_size: int = 1000,
    include_resolved_mask: bool = True,
    include_row_metadata: bool = True,
    explore_only: bool = False,
) -> SparseContributionDetails:
    process_id = str((process_data or {}).get("@id") or (process_data or {}).get("id") or "")
    process_name = str((process_data or {}).get("name") or "")
    candidate_indices: List[int] = []
    exchange_keys: List[str] = []
    candidate_exchanges: List[Dict[str, Any]] = []
    candidate_flows: List[FlowInfo] = []
    for index, exchange in enumerate(exchanges):
        flow_id = exchange_flow_id(exchange)
        if not flow_id or flow_id not in flow_lookup:
            raise MissingFlowError(f"Cannot resolve flow for exchange at index {index}")
        flow = flow_lookup[flow_id]
        if not is_candidate_elementary_exchange(exchange, flow):
            continue
        candidate_indices.append(index)
        candidate_exchanges.append(exchange)
        candidate_flows.append(flow)
        exchange_keys.append(
            str(exchange.get("@id") or exchange.get("id") or exchange.get("internalId") or f"{flow_id}:{index}")
        )

    characterised_flags: List[bool] = [False] * len(candidate_indices)
    resolved_rows: Optional[List[List[bool]]]
    if include_resolved_mask:
        resolved_rows = [[False] * len(candidate_indices) for _ in categories]
    else:
        resolved_rows = None
    stats = CFAmbiguityStats()
    resolution_cache: Dict[Tuple[str, ...], Any] = {}
    candidate_cache: Dict[tuple[str, str], Any] = {}
    admissible_by_category: List[List[Tuple[int, Dict[str, Any], int, Tuple[_PreparedOption, ...]]]] = [[] for _ in categories]

    for local_index, (index, exchange, flow) in enumerate(zip(candidate_indices, candidate_exchanges, candidate_flows)):
        for row_index, category in enumerate(categories):
            cache_key = _resolution_cache_key(
                category=category,
                exchange=exchange,
                exchange_index=index,
                flow=flow,
                process_data=process_data,
                strict_units=strict_units,
                allow_water_mass_volume_override=allow_water_mass_volume_override,
            )
            if cache_key in resolution_cache:
                admissible = resolution_cache[cache_key]
            else:
                admissible = resolve_admissible_cf_set_for_exchange(
                    category=category,
                    exchange=exchange,
                    exchange_index=index,
                    flow=flow,
                    candidates=candidate_cache.setdefault(
                        (category.category_id, flow.flow_id),
                        candidates_for_flow(category, flow.flow_id),
                    ),
                    unit_registry=unit_registry,
                    strict_units=strict_units,
                    allow_water_mass_volume_override=allow_water_mass_volume_override,
                    strict=True,
                    process_data=process_data,
                    warning_records=warning_records,
                    ambiguity_records=ambiguity_records,
                    diagnostic_file=diagnostic_file,
                    resolution_manager=resolution_manager,
                    record_only=explore_only,
                )
                resolution_cache[cache_key] = admissible
            if admissible is None:
                continue
            characterised_flags[local_index] = True
            if resolved_rows is not None:
                resolved_rows[row_index][local_index] = True
            prepared_options = _prepare_admissible_options(
                exchange=exchange,
                admissible=admissible,
                tol=tol,
            )
            option_count = len(prepared_options)
            if option_count == 1:
                stats.n_exact_cf_resolutions += 1
            else:
                stats.n_finite_cf_candidate_sets += 1
                stats.max_candidate_set_size = max(stats.max_candidate_set_size, option_count)
                if option_count > max_candidate_set_size:
                    stats.n_unresolved_cf_ambiguities += 1
                    if explore_only:
                        continue
                    raise _scenario_limit_error(
                        process_id=process_id,
                        process_name=process_name,
                        category=category,
                        exchange=exchange,
                        exchange_index=index,
                        message=(
                            "Finite CF candidate-set expansion exceeded max_candidate_set_size"
                            f"; candidate_count={option_count}; max_candidate_set_size={max_candidate_set_size}"
                        ),
                    )
            admissible_by_category[row_index].append((local_index, exchange, index, prepared_options))

    if include_resolved_mask:
        resolved_mask = (
            np.array(resolved_rows, dtype=bool)
            if categories and candidate_indices and resolved_rows is not None
            else np.zeros((len(categories), len(candidate_indices)), dtype=bool)
        )
    else:
        resolved_mask = np.zeros((0, 0), dtype=bool)
    if explore_only:
        return SparseContributionDetails(
            rowsets=(),
            candidate_indices=tuple(candidate_indices),
            exchange_keys=tuple(exchange_keys),
            characterised_flags=tuple(characterised_flags),
            resolved_mask=resolved_mask,
            row_metadata=(),
            cf_ambiguity_stats=stats,
        )
    rowsets: List[_CategoryRowSet] = []
    row_metadata: List[CoverageRowMetadata] = []
    for category, category_records in zip(categories, admissible_by_category):
        if not category_records:
            continue
        category_rowset = _build_category_rowset(
            category=category,
            admissible_by_exchange=category_records,
            tol=tol,
            process_id=process_id,
            process_name=process_name,
            process_data=process_data,
            stats=stats,
            max_scenario_rows_per_process=max_scenario_rows_per_process,
            include_row_metadata=include_row_metadata,
        )
        rowsets.append(category_rowset)
        if include_row_metadata:
            row_metadata.extend(category_rowset.metadata_rows)

    return SparseContributionDetails(
        rowsets=tuple(rowsets),
        candidate_indices=tuple(candidate_indices),
        exchange_keys=tuple(exchange_keys),
        characterised_flags=tuple(characterised_flags),
        resolved_mask=resolved_mask,
        row_metadata=tuple(row_metadata),
        cf_ambiguity_stats=stats,
    )


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
