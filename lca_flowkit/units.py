"""Strict unit compatibility checks for exchange-to-factor matching.

Unit handling is deliberately separated from CF ambiguity handling.

That distinction matters scientifically:

- multiple *compatible* CFs are a modelling ambiguity and may require scenario
  expansion;
- incompatible units are a physical resolution problem and must not be treated
  as ordinary ambiguity.

This module therefore answers one narrow question:

``Can this exchange amount be expressed in the unit basis expected by this CF?``
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import UnitResolutionError
from .models import Exchange, FlowDefinition, ImpactFactorCandidate, UnitDefinition
from .utils import fold_text


@dataclass(frozen=True)
class UnitResolution:
    """Outcome of one exchange/factor unit compatibility check.

    ``conversion_factor`` is the scalar that converts the exchange amount into
    the factor's unit basis:

    ``amount_in_factor_units = exchange.amount * conversion_factor``
    """
    compatible: bool
    conversion_factor: float
    reason: str


def _lookup(unit_id: str, registry: dict[str, UnitDefinition]) -> UnitDefinition | None:
    """Resolve a unit definition when the caller supplied a concrete ID.

    The helper exists mainly to make later control flow read more clearly.
    Returning ``None`` for empty IDs keeps the caller logic explicit rather than
    pretending that an empty unit reference is a real registry lookup.
    """
    return registry.get(unit_id) if unit_id else None


def resolve_unit_conversion(
    exchange: Exchange,
    flow: FlowDefinition,
    candidate: ImpactFactorCandidate,
    unit_registry: dict[str, UnitDefinition],
    *,
    strict_units: bool,
) -> UnitResolution:
    """Resolve the scalar needed to express an exchange in factor units.

    Unit-based exclusions are kept separate from CF ambiguity handling so audit
    output can distinguish physical incompatibility from modelling ambiguity.

    Resolution order:

    1. if both units are missing, accept the pair as unspecified rather than
       inventing a conversion;
    2. if concrete unit IDs match, accept exactly;
    3. if normalised unit names match, accept exactly;
    4. if both units resolve through the unit registry and share group and flow
       property, compute the registry-based conversion factor;
    5. if flow properties are provably incompatible, fail immediately;
    6. otherwise either fail strictly or accept a non-strict fallback.

    The strict path is the default because "unknown unit compatibility" should
    not silently become "compatible enough" in a scientific reduction.
    """
    exchange_unit = fold_text(exchange.unit_name)
    candidate_unit = fold_text(candidate.unit_name)
    if not exchange_unit and not candidate_unit:
        return UnitResolution(True, 1.0, "both_units_unspecified")
    if exchange.unit_id and candidate.unit_id and exchange.unit_id == candidate.unit_id:
        return UnitResolution(True, 1.0, "same_unit_id")
    if exchange_unit and candidate_unit and exchange_unit == candidate_unit:
        return UnitResolution(True, 1.0, "same_unit_name")

    exchange_def = _lookup(exchange.unit_id, unit_registry)
    candidate_def = _lookup(candidate.unit_id, unit_registry)
    if exchange_def and candidate_def:
        same_group = exchange_def.group_id and candidate_def.group_id and exchange_def.group_id == candidate_def.group_id
        same_property = (
            exchange_def.flow_property_id
            and candidate_def.flow_property_id
            and exchange_def.flow_property_id == candidate_def.flow_property_id
        )
        if same_group and same_property and exchange_def.conversion_factor and candidate_def.conversion_factor:
            # Both conversion factors are expressed relative to the same
            # reference unit of the unit group, so their ratio converts between
            # the exchange basis and the factor basis.
            factor = float(exchange_def.conversion_factor) / float(candidate_def.conversion_factor)
            return UnitResolution(True, factor, "registry_group_conversion")

    exchange_property = exchange.flow_property_id or flow.reference_flow_property_id
    candidate_property = candidate.flow_property_id
    if exchange_property and candidate_property and exchange_property != candidate_property:
        raise UnitResolutionError(
            f"Exchange flow property {exchange_property} is not compatible with factor flow property {candidate_property}"
        )

    if strict_units:
        raise UnitResolutionError(
            f"Could not prove unit compatibility for exchange unit {exchange.unit_name or '-'} and factor unit {candidate.unit_name or '-'}"
        )
    return UnitResolution(True, 1.0, "non_strict_fallback")
