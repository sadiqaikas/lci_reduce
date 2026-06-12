"""Characterisation factor ambiguity resolution helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .models import CharacterisationFactorCandidate


def _normalise_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def _normalise_float(value: float) -> str:
    return format(float(value), ".15g")


@dataclass
class CFAmbiguityContext:
    category_id: str
    category_name: str
    method_id: str
    method_name: str
    flow_id: str
    flow_name: str
    process_id: str
    process_name: str
    exchange_id: str
    exchange_index: int
    diagnostic_file: str
    differing_fields: list[str]


@dataclass
class CFResolutionSummary:
    n_cf_ambiguities_found: int = 0
    n_cf_ambiguity_keys_unique: int = 0
    n_cf_ambiguities_resolved_automatically: int = 0
    n_cf_ambiguities_unresolved: int = 0


@dataclass
class CFResolutionDecision:
    status: str
    candidate: Optional[CharacterisationFactorCandidate]
    reason: str


def ambiguity_key(context: CFAmbiguityContext) -> str:
    return "|".join(
        [
            context.method_id or "",
            context.category_id,
            context.flow_id,
        ]
    )


def candidate_metadata(candidate: CharacterisationFactorCandidate) -> Dict[str, object]:
    return {
        "method_id": candidate.method_id or "",
        "method_name": candidate.method_name or "",
        "category_id": candidate.category_id,
        "category_name": candidate.category_name,
        "flow_id": candidate.flow_id,
        "flow_name": candidate.flow_name,
        "cf_value": _normalise_float(candidate.cf_value),
        "cf_unit": candidate.cf_unit or "",
        "cf_unit_id": candidate.cf_unit_id or "",
        "cf_flow_property_id": candidate.cf_flow_property_id or "",
        "cf_flow_property_name": candidate.cf_flow_property_name or "",
        "cf_compartment": candidate.cf_compartment or "",
        "cf_subcompartment": candidate.cf_subcompartment or "",
        "cf_location_id": candidate.cf_location_id or "",
        "cf_location_name": candidate.cf_location_name or "",
        "cf_region": candidate.cf_region or "",
        "source_file": candidate.source_file,
        "raw_factor_object": candidate.raw_factor_object,
    }


def _normalised_candidate_identity(candidate: CharacterisationFactorCandidate) -> Tuple[str, str]:
    metadata = {
        key: _normalise_text(str(value) if value is not None else "")
        for key, value in candidate_metadata(candidate).items()
        if key != "raw_factor_object"
    }
    return _normalise_float(candidate.cf_value), json.dumps(metadata, ensure_ascii=True, sort_keys=True)


class CFResolutionManager:
    def __init__(
        self,
        *,
        mode: str,
    ) -> None:
        self.mode = mode
        self.summary = CFResolutionSummary()
        self._ambiguity_keys_seen: set[str] = set()

    def note_found(self, key: str) -> None:
        self.summary.n_cf_ambiguities_found += 1
        if key not in self._ambiguity_keys_seen:
            self._ambiguity_keys_seen.add(key)
            self.summary.n_cf_ambiguity_keys_unique = len(self._ambiguity_keys_seen)

    def record_automatic_resolution(self) -> None:
        self.summary.n_cf_ambiguities_resolved_automatically += 1
